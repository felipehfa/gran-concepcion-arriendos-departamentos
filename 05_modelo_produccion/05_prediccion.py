"""
Cálculo de precio predicho, confianza y etiqueta — pipeline de producción.

Toma las features calculadas por la Etapa 6 (`04_ingenieria_variables_produccion.py`),
predice con el ensamble de `modelo_produccion.pkl` (Etapa 1), y aplica la
calibración de oportunidad/confianza guardada en `parametros_produccion.json`
(bordes de deciles de precio + mediana/MAD de error por decil + terciles de
CV) para etiquetar cada aviso, sin necesitar recalcular esos bordes con una
sola fila nueva (imposible: qcut necesita una distribución).

El modelo vigente puede ser XGBoost o LightGBM (lo decide
`seleccionar_algoritmo.py`, ver Tarea 1/3) — `cargar_modelo_y_calibracion`
lee el campo "algoritmo" de `parametros_produccion.json` y carga solo el
script de investigación correspondiente, para usar SU `predict_ensemble_matrix`
(mismo nombre en ambos módulos, pero cada uno sabe predecir con su propia
librería). El resto de esta etapa (z_robusto, calibración de
oportunidad/confianza, conversión UF→CLP, UPSERT) es agnóstico al algoritmo.

Convierte el precio real del aviso a CLP antes de compararlo contra la
predicción — reutiliza `convertir_precios_uf_a_clp` de
01_ingenieria_variables.py (consulta mindicador.cl con caché, ahora en la
propia base de datos de producción en vez de la original).

Inserta en `predicciones` vía UPSERT sobre (id_aviso, version_modelo): re-
ejecutar esta etapa no duplica ni corrompe nada.
"""

import importlib.util
import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

import db

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
ENTRENAMIENTO_DIR = SCRIPT_DIR / "entrenamiento"

MODULOS_INVESTIGACION = {
    "xgboost": REPO_ROOT / "04_modelamiento" / "01_xgboost.py",
    "lightgbm": REPO_ROOT / "04_modelamiento" / "02_lightgbm.py",
}
INGENIERIA_VARIABLES_PATH = REPO_ROOT / "03_ingenieria_variables" / "01_ingenieria_variables.py"
INGENIERIA_VARIABLES_PRODUCCION_PATH = SCRIPT_DIR / "04_ingenieria_variables_produccion.py"


def _cargar_modulo(nombre: str, ruta: Path):
    spec = importlib.util.spec_from_file_location(nombre, ruta)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


def _cargar_modulo_investigacion(algoritmo: str):
    if algoritmo not in MODULOS_INVESTIGACION:
        raise ValueError(f"Algoritmo desconocido en parametros_produccion.json: {algoritmo!r}")
    return _cargar_modulo(f"modelo_investigacion_{algoritmo}", MODULOS_INVESTIGACION[algoritmo])


iv = _cargar_modulo("ingenieria_variables_original", INGENIERIA_VARIABLES_PATH)
ivp = _cargar_modulo("ingenieria_variables_produccion", INGENIERIA_VARIABLES_PRODUCCION_PATH)


# ------------------------------------------------------------------
# Carga del modelo vigente + su calibración
# ------------------------------------------------------------------
def cargar_modelo_y_calibracion() -> dict:
    """
    Carga el modelo VIGENTE desde su carpeta versionada
    (entrenamiento/versiones/{version_actual}/) — cada versión queda
    archivada por separado (no se sobrescribe), así que el modelo exacto
    usado en cualquier predicción pasada (predicciones.version_modelo)
    siempre se puede recuperar, no solo su identificador.

    El algoritmo con el que predecir se determina leyendo el campo
    "algoritmo" de parametros_produccion.json (Tarea 3) — no se asume
    XGBoost. modelo_produccion.pkl guarda {"algoritmo": ..., "modelos": [...]},
    así que de paso se valida que ambas fuentes coincidan.
    """
    control = json.loads((ENTRENAMIENTO_DIR / "version_modelo.json").read_text(encoding="utf-8"))
    version_actual = control["version_actual"]
    version_dir = ENTRENAMIENTO_DIR / "versiones" / version_actual

    params = json.loads((version_dir / "parametros_produccion.json").read_text(encoding="utf-8"))
    algoritmo = params["algoritmo"]

    with open(version_dir / "modelo_produccion.pkl", "rb") as f:
        modelo_guardado = pickle.load(f)

    if modelo_guardado["algoritmo"] != algoritmo:
        raise ValueError(
            f"Inconsistencia en {version_dir}: parametros_produccion.json dice "
            f"algoritmo={algoritmo!r} pero modelo_produccion.pkl fue guardado con "
            f"algoritmo={modelo_guardado['algoritmo']!r}."
        )

    mx = _cargar_modulo_investigacion(algoritmo)

    return {
        "models": modelo_guardado["modelos"],
        "algoritmo": algoritmo,
        "predict_ensemble_matrix": mx.predict_ensemble_matrix,
        "version_modelo": params["version_modelo"],
        "features": params["features"],
        "calibracion": params["calibracion_oportunidad"],
    }


# ------------------------------------------------------------------
# Precio real en CLP (con conversión UF -> CLP si corresponde)
# ------------------------------------------------------------------
def calcular_precio_clp(con_produccion, id_avisos: list) -> pd.DataFrame:
    if not id_avisos:
        return pd.DataFrame(columns=["id_aviso", "precio_clp"])

    placeholders = ",".join("?" for _ in id_avisos)
    datos = pd.read_sql_query(f"""
        SELECT a.id_aviso, a.precio, a.moneda, d.fecha_publicacion_aprox
        FROM avisos a
        JOIN avisos_detalle d ON a.id_aviso = d.id_aviso
        WHERE a.id_aviso IN ({placeholders})
    """, con_produccion, params=id_avisos)

    if (datos["moneda"] == "UF").any():
        try:
            datos = iv.convertir_precios_uf_a_clp(datos, ruta_bd=str(db.RUTA_BD_PRODUCCION))
        except Exception as e:
            log.warning(f"No se pudo convertir precios UF->CLP ({e}). "
                        f"Los avisos en UF se excluyen de esta corrida.")
            datos["precio_clp"] = np.where(datos["moneda"] != "UF", datos["precio"].astype(float), np.nan)
    else:
        datos["precio_clp"] = datos["precio"].astype(float)

    return datos[["id_aviso", "precio_clp"]]


# ------------------------------------------------------------------
# Persistencia
# ------------------------------------------------------------------
def guardar_prediccion(con, id_aviso: str, version_modelo: str, precio_predicho: float,
                        z_robusto: float, decil_precio: int, etiqueta: str,
                        nivel_confianza: str, cv_ensamble: float) -> None:
    con.execute("""
        INSERT INTO predicciones (
            id_aviso, version_modelo, fecha_prediccion, precio_predicho,
            z_robusto, decil_precio, etiqueta, nivel_confianza, cv_ensamble
        ) VALUES (?, ?, datetime('now'), ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id_aviso, version_modelo) DO UPDATE SET
            fecha_prediccion = excluded.fecha_prediccion,
            precio_predicho  = excluded.precio_predicho,
            z_robusto        = excluded.z_robusto,
            decil_precio     = excluded.decil_precio,
            etiqueta         = excluded.etiqueta,
            nivel_confianza  = excluded.nivel_confianza,
            cv_ensamble      = excluded.cv_ensamble
    """, (id_aviso, version_modelo, precio_predicho, z_robusto, decil_precio,
          etiqueta, nivel_confianza, cv_ensamble))
    con.commit()


# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------
def generar_predicciones(con_produccion, con_original, features_df: pd.DataFrame = None) -> dict:
    """
    Si `features_df` ya viene calculado (ej. por el orquestador, que corre
    la etapa de variables por separado para loggearla como su propia etapa),
    se usa directo y no se recalcula. Si se llama de forma standalone
    (`python 05_prediccion.py`), lo calcula acá mismo.
    """
    modelo_info = cargar_modelo_y_calibracion()
    log.info(f"Modelo vigente: {modelo_info['version_modelo']}")

    if features_df is None:
        features_df = ivp.construir_features_produccion(con_produccion, con_original)
    if features_df.empty:
        log.info("No hay avisos pendientes de predicción.")
        return {"predicciones_generadas": 0, "avisos_saltados_sin_precio": 0,
                "version_modelo": modelo_info["version_modelo"]}

    precios = calcular_precio_clp(con_produccion, features_df["id_aviso"].tolist())
    # Avisos con precio_clp fuera de rango (ventas mal clasificadas, datos
    # corruptos) quedan fuera de precios -> el merge left les deja precio_clp
    # en NaN -> el loop de abajo ya sabe saltarlos como "sin precio_clp válido".
    precios = iv.filtrar_precio_maximo(precios)
    features_df = features_df.merge(precios, on="id_aviso", how="left")

    X = features_df.reindex(columns=modelo_info["features"], fill_value=0).fillna(0)
    pred_matrix = modelo_info["predict_ensemble_matrix"](modelo_info["models"], X)
    y_pred = pred_matrix.mean(axis=0)
    pred_std = pred_matrix.std(axis=0)
    cv_ensamble = pred_std / np.where(y_pred == 0, 1e-6, y_pred)

    calibracion = modelo_info["calibracion"]
    bordes_deciles = calibracion["bordes_deciles_precio"]
    stats_por_decil = calibracion["stats_por_decil"]
    bordes_cv = calibracion["bordes_cv_confianza"]
    etiquetas_confianza = calibracion["etiquetas_confianza"]
    mad_scale = calibracion["mad_scale_const"]
    umbral_oportunidad = calibracion["umbral_oportunidad"]
    umbral_caro = calibracion["umbral_caro"]

    generadas = 0
    saltados = 0

    for i, fila in features_df.reset_index(drop=True).iterrows():
        precio_real = fila["precio_clp"]
        if pd.isna(precio_real):
            log.warning(f"{fila['id_aviso']}: sin precio_clp válido. Se salta esta corrida.")
            saltados += 1
            continue

        precio_predicho = float(y_pred[i])
        error = precio_real - precio_predicho

        decil = int(pd.cut([precio_real], bins=bordes_deciles, labels=False, include_lowest=True)[0])
        stats_decil = stats_por_decil.get(str(decil))
        mediana_decil = stats_decil["mediana_error"]
        mad_decil = stats_decil["mad_error"]
        mad_ajustado = mad_decil * mad_scale
        z_robusto = (error - mediana_decil) / (mad_ajustado if mad_ajustado else 1e-6)

        if z_robusto < -umbral_oportunidad:
            etiqueta = "oportunidad"
        elif z_robusto > umbral_caro:
            etiqueta = "caro"
        else:
            etiqueta = "precio_de_mercado"

        idx_confianza = int(pd.cut([cv_ensamble[i]], bins=bordes_cv, labels=False, include_lowest=True)[0])
        nivel_confianza = etiquetas_confianza[idx_confianza]

        guardar_prediccion(
            con_produccion, fila["id_aviso"], modelo_info["version_modelo"],
            precio_predicho, float(z_robusto), decil, etiqueta, nivel_confianza, float(cv_ensamble[i]),
        )
        log.info(f"{fila['id_aviso']}: precio_predicho={precio_predicho:,.0f}  "
                  f"etiqueta={etiqueta}  confianza={nivel_confianza}")
        generadas += 1

    return {
        "predicciones_generadas": generadas,
        "avisos_saltados_sin_precio": saltados,
        "version_modelo": modelo_info["version_modelo"],
    }


if __name__ == "__main__":
    con_produccion = db.conectar_produccion()
    con_original = db.conectar_original()

    resumen = generar_predicciones(con_produccion, con_original)

    con_produccion.close()
    con_original.close()

    log.info(
        f"Corrida completa. Predicciones generadas: {resumen['predicciones_generadas']} | "
        f"Avisos saltados por precio inválido: {resumen['avisos_saltados_sin_precio']}"
    )
