"""
Entrenamiento del modelo de PRODUCCIÓN.

No duplica la lógica de modelamiento: carga como módulo (vía importlib, ya
que su nombre empieza con dígito) el script de investigación del algoritmo
GANADOR — `02_modelos/01_xgboost.py` o `02_modelos/02_lightgbm.py`,
según lo que haya decidido `seleccionar_algoritmo.py` y quedó registrado en
`algoritmo_seleccionado.json` — y reutiliza sus funciones
(optimización de hiperparámetros, bagging, evaluación, SHAP). Solo se carga
y entrena el algoritmo elegido; el otro no se toca, para no duplicar el
costo de Optuna innecesariamente. Los dos scripts de investigación exponen
la misma API (mismos nombres de función/constante), así que el resto de este
script es agnóstico a cuál de los dos se cargó.

Diferencias respecto al script de investigación:
  - Split 85/15 (train/test) en vez de 70/15/15. El ensamble de bagging
    necesita igualmente un set para early stopping, así que se separa
    internamente un 10% del 85% de train solo para eso (~76.5% train_fit /
    ~8.5% early-stopping / 15% test) - es un detalle interno de
    entrenamiento, no una partición pública reportada.
  - Genera y persiste un identificador de VERSIÓN del modelo
    (ver `generar_version_modelo`), para poder distinguir después en la
    base de datos de producción qué predicciones se hicieron con qué
    versión cuando el modelo se reentrene.
  - Guarda el ensamble y sus parámetros en `versiones/{version}/` (no en
    `02_modelos/save/model/`, que es el modelo de investigación).
    Cada versión queda archivada en su propia carpeta (no se sobrescribe),
    para poder recuperar el modelo EXACTO usado en cualquier predicción
    pasada (`predicciones.version_modelo` -> `versiones/{esa versión}/`).

Dataset de entrada (`--origen-datos`, por defecto `supabase`):

  - `supabase`: regenera el dataset en el momento corriendo la ingeniería de
    variables contra la base de PRODUCCIÓN (`ejecutar_pipeline(con=...)`), con
    los avisos que el pipeline lleva acumulados. Es el modo normal: entrena con
    los datos actualizados en vez de un CSV congelado.
  - `csv`: usa el CSV curado tal como esté en disco, sin regenerarlo. Sirve para
    reproducir un entrenamiento anterior sobre exactamente los mismos datos.

Qué avisos entran al dataset se controla con `--estados` y
`--meses-max-finalizados` (ver ESTADOS_ENTRENAMIENTO_DEFAULT en
01_ingenieria_variables.py). No se entrena solo con los 'activo' a propósito:
esos son, por definición, los que el mercado todavía no absorbió, así que
sobre-representan unidades caras para su segmento.

ARTEFACTOS DE FEATURES POR VERSIÓN:
Regenerar el dataset también regenera `niveles_barrio.json` y
`modelos_superficie/*.pkl`, que NO son solo del entrenamiento: los usa
`04_ingenieria_variables_produccion.py` en cada inferencia. Como esos archivos
viven en una ruta global y se sobrescriben, se archiva una copia en
`versiones/{version}/artefactos_features/` junto al modelo, para poder
reconstruir después con qué artefactos exactos se entrenó cada versión.
"""

import argparse
import hashlib
import importlib.util
import json
import pickle
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ------------------------------------------------------------------
# Carga del script de investigación del algoritmo GANADOR como módulo
# (nombre empieza con dígito, no se puede hacer un `import` normal).
# ------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent
MODELAMIENTO_ROOT = REPO_ROOT / "modelamiento"

ALGORITMO_SELECCIONADO_PATH = SCRIPT_DIR / "algoritmo_seleccionado.json"

MODULOS_MODELAMIENTO = {
    "xgboost": MODELAMIENTO_ROOT / "02_modelos" / "01_xgboost.py",
    "lightgbm": MODELAMIENTO_ROOT / "02_modelos" / "02_lightgbm.py",
}
# Atributo bajo el cual cada módulo expone su propia librería de boosting
# (para poder imprimir su versión sin hardcodear un solo algoritmo).
LIB_ATTR_POR_ALGORITMO = {"xgboost": "xgb", "lightgbm": "lgb"}


def _cargar_algoritmo_seleccionado() -> str:
    if not ALGORITMO_SELECCIONADO_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró {ALGORITMO_SELECCIONADO_PATH}. Corre primero "
            f"seleccionar_algoritmo.py (Tarea 1) para decidir qué algoritmo entrenar."
        )
    seleccion = json.loads(ALGORITMO_SELECCIONADO_PATH.read_text(encoding="utf-8"))
    algoritmo = seleccion["algoritmo"]
    if algoritmo not in MODULOS_MODELAMIENTO:
        raise ValueError(f"Algoritmo desconocido en {ALGORITMO_SELECCIONADO_PATH}: {algoritmo!r}")
    return algoritmo


def _cargar_modulo_modelamiento(algoritmo: str):
    ruta = MODULOS_MODELAMIENTO[algoritmo]
    spec = importlib.util.spec_from_file_location(f"modelo_modelamiento_{algoritmo}", ruta)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


ALGORITMO = _cargar_algoritmo_seleccionado()
mx = _cargar_modulo_modelamiento(ALGORITMO)

# `db` vive un nivel más arriba (produccion/01_modelo_produccion/) y no es un
# paquete instalable, así que se agrega esa carpeta al path para poder
# importarlo. Se hace acá y no arriba porque solo el modo `--origen-datos
# supabase` lo necesita.
MODELO_PRODUCCION_ROOT = SCRIPT_DIR.parent
if str(MODELO_PRODUCCION_ROOT) not in sys.path:
    sys.path.insert(0, str(MODELO_PRODUCCION_ROOT))

INGENIERIA_VARIABLES_PATH = MODELAMIENTO_ROOT / "01_ingenieria_variables" / "01_ingenieria_variables.py"


def _cargar_ingenieria_variables():
    """Carga 01_ingenieria_variables.py como módulo (su nombre empieza con
    dígito, así que no admite un `import` normal). Es el MISMO módulo que usa
    el pipeline de inferencia, para que entrenamiento e inferencia no puedan
    divergir en la lógica de features."""
    spec = importlib.util.spec_from_file_location(
        "ingenieria_variables_entrenamiento", INGENIERIA_VARIABLES_PATH
    )
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
SEED = 42

TEST_SIZE_PRODUCCION = 0.15
EARLY_STOP_FRACTION_OF_TRAIN = 0.10  # sobre el 85% de train, solo para early stopping del bagging

N_TRIALS_OPTUNA_PRODUCCION = mx.N_TRIALS_OPTUNA
CV_SPLITS_OPTUNA_PRODUCCION = mx.CV_SPLITS_OPTUNA
N_SEEDS_BAGGING_PRODUCCION = mx.N_SEEDS_BAGGING

SAVE_DIR = SCRIPT_DIR
CONTROL_VERSION_PATH = SAVE_DIR / "version_modelo.json"
# Cada versión se archiva en su propia carpeta (en vez de sobrescribir un
# único modelo_produccion.pkl/parametros_produccion.json) para poder
# recuperar el modelo EXACTO que se usó en cualquier predicción pasada
# (predicciones.version_modelo -> versiones/{esa versión}/).
VERSIONES_DIR = SAVE_DIR / "versiones"


# ------------------------------------------------------------------
# Regeneración del dataset desde la base de PRODUCCIÓN
# ------------------------------------------------------------------
def regenerar_dataset_desde_supabase(estados, meses_max_finalizados) -> dict:
    """
    Corre la ingeniería de variables contra la base de producción y sobrescribe
    el CSV curado que después lee `mx.load_dataset`.

    Devuelve un dict con la procedencia del dataset (filas, filtros aplicados,
    fecha), que se guarda en `parametros_produccion.json` para que quede
    registrado con qué población se entrenó cada versión - dato que el CSV solo
    no permite reconstruir, porque se sobrescribe en cada corrida.
    """
    import db

    iv = _cargar_ingenieria_variables()

    print(f"Regenerando dataset desde la base de PRODUCCIÓN (Supabase)...")
    print(f"  estados={tuple(estados)}  meses_max_finalizados={meses_max_finalizados}")

    con = db.conectar_produccion()
    try:
        df = iv.ejecutar_pipeline(
            con=con,
            estados=tuple(estados),
            meses_max_antiguedad_finalizados=meses_max_finalizados,
        )
    finally:
        con.close()

    print(f"  -> dataset regenerado: {len(df)} filas, {len(df.columns)} columnas")
    print(f"  -> guardado en {mx.DATASET_PATH}")

    return {
        "origen": "supabase",
        "fecha_regeneracion": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "filas": int(len(df)),
        "estados_incluidos": list(estados),
        "meses_max_antiguedad_finalizados": meses_max_finalizados,
        "columna_antiguedad": iv.COLUMNA_ANTIGUEDAD_AVISO,
    }


def archivar_artefactos_features(version_dir: Path) -> list:
    """
    Copia a `version_dir/artefactos_features/` los archivos que la ingeniería de
    variables genera y que la INFERENCIA también usa: `selected_features.csv`,
    `niveles_barrio.json` y `modelos_superficie/`.

    Por qué: esos tres viven en rutas globales dentro de modelamiento/ y se
    sobrescriben cada vez que se regenera el dataset, mientras que los modelos
    quedan archivados por versión. Sin esta copia no hay forma de saber después
    con qué niveles de barrio o con qué modelos de imputación de superficie se
    entrenó una versión dada.
    """
    destino = version_dir / "artefactos_features"
    destino.mkdir(parents=True, exist_ok=True)

    origenes = [
        Path(mx.FEATURES_PATH),
        Path(mx.DATASET_PATH).parent / "niveles_barrio.json",
        Path(mx.DATASET_PATH).parent / "modelos_superficie",
    ]

    archivados = []
    for origen in origenes:
        if not origen.exists():
            print(f"  Advertencia: no existe {origen}, no se archiva.")
            continue
        if origen.is_dir():
            shutil.copytree(origen, destino / origen.name, dirs_exist_ok=True)
        else:
            shutil.copy2(origen, destino / origen.name)
        archivados.append(origen.name)

    print(f"  Artefactos de features archivados en {destino}: {archivados}")
    return archivados


# ------------------------------------------------------------------
# Versionado del modelo
# ------------------------------------------------------------------
def _hash_hiperparametros(algoritmo: str, best_params: dict) -> str:
    """sha256 (primeros 8 hex) de (algoritmo + hiperparámetros ganadores), en
    JSON canónico (claves ordenadas) para que la misma combinación siempre dé
    el mismo hash. Se incluye `algoritmo` para que dos algoritmos que por
    coincidencia comparten nombres de hiperparámetros con igual valor no
    colisionen en el mismo hash."""
    canon = json.dumps({"algoritmo": algoritmo, "params": best_params}, sort_keys=True, default=str)
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()[:8]


def generar_version_modelo(algoritmo: str, best_params: dict, control_path: Path = CONTROL_VERSION_PATH) -> tuple:
    """
    Genera el identificador de versión:
    v{contador:04d}_{YYYYMMDDHHMMSS}_{algoritmo}_{hash8}

    El algoritmo queda explícito en el propio identificador de versión (no
    solo en parametros_produccion.json) para poder distinguir de un vistazo,
    en `predicciones.version_modelo` o en el nombre de la carpeta de
    `versiones/`, qué algoritmo generó cada predicción histórica.

    El contador es incremental y persiste en `control_path` junto con un
    historial de versiones anteriores. Se genera una versión nueva en cada
    corrida de este script, sin importar si los hiperparámetros cambiaron.
    """
    if control_path.exists():
        control = json.loads(control_path.read_text(encoding="utf-8"))
    else:
        control = {"contador": 0, "version_actual": None, "historial": []}

    contador = control["contador"] + 1
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    hash8 = _hash_hiperparametros(algoritmo, best_params)
    version = f"v{contador:04d}_{timestamp}_{algoritmo}_{hash8}"

    control["contador"] = contador
    control["version_actual"] = version
    control["historial"].append({
        "version": version,
        "algoritmo": algoritmo,
        "fecha_entrenamiento": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "hash_hiperparametros": hash8,
    })
    return version, control


def guardar_control_version(control: dict, control_path: Path = CONTROL_VERSION_PATH) -> None:
    control_path.write_text(json.dumps(control, indent=2, ensure_ascii=False), encoding="utf-8")


# ------------------------------------------------------------------
# Calibración de oportunidad/confianza — para poder etiquetar avisos NUEVOS
# en producción (05_prediccion.py) de forma consistente con el test set, en vez de
# recalcular deciles/terciles de una sola fila (imposible: qcut necesita una
# distribución). A diferencia de `etiquetar_oportunidades` en
# 02_modelos/01_xgboost.py (que aplica los bordes calculados a ESE
# mismo test set y los descarta), acá se GUARDAN los bordes para reaplicarlos
# después con `pd.cut` sobre avisos que ni siquiera existían al entrenar.
# ------------------------------------------------------------------
def calcular_calibracion_oportunidad(
    models: list, X_test: pd.DataFrame, y_test: pd.Series,
    n_deciles: int = None, n_grupos_confianza: int = None,
) -> dict:
    n_deciles = n_deciles or mx.N_DECILES_OPORTUNIDAD
    n_grupos_confianza = n_grupos_confianza or mx.N_GRUPOS_CONFIANZA

    y_true = np.asarray(y_test, dtype=float)
    pred_matrix = mx.predict_ensemble_matrix(models, X_test)
    y_pred = pred_matrix.mean(axis=0)
    error = y_true - y_pred

    # Bordes de deciles de precio_clp (test), extendidos a ±inf en los
    # extremos para que un aviso nuevo con precio fuera del rango visto en
    # test igual caiga en el decil extremo correspondiente, en vez de NaN.
    _, bordes_deciles = pd.qcut(y_true, q=n_deciles, retbins=True, duplicates="drop")
    bordes_deciles[0], bordes_deciles[-1] = -np.inf, np.inf
    deciles = pd.cut(y_true, bins=bordes_deciles, labels=False, include_lowest=True)

    stats_por_decil = {}
    for d in sorted(pd.unique(deciles)):
        mask = deciles == d
        mediana_d, mad_d = mx.mediana_y_mad(error[mask])
        stats_por_decil[int(d)] = {
            "n": int(mask.sum()), "mediana_error": mediana_d, "mad_error": mad_d,
        }

    # Terciles del coeficiente de variación del ensamble (test), mismos
    # extremos ±inf por la misma razón.
    pred_std = pred_matrix.std(axis=0)
    cv_ensamble = pred_std / np.where(y_pred == 0, 1e-6, y_pred)
    _, bordes_cv = pd.qcut(cv_ensamble, q=n_grupos_confianza, retbins=True, duplicates="drop")
    bordes_cv[0], bordes_cv[-1] = -np.inf, np.inf

    return {
        "bordes_deciles_precio": bordes_deciles.tolist(),
        "stats_por_decil": stats_por_decil,
        "bordes_cv_confianza": bordes_cv.tolist(),
        "etiquetas_confianza": mx.ETIQUETAS_CONFIANZA,
        "mad_scale_const": mx.MAD_SCALE_CONST,
        "umbral_oportunidad": mx.UMBRAL_OPORTUNIDAD,
        "umbral_caro": mx.UMBRAL_CARO,
    }


# ------------------------------------------------------------------
# Split 85/15 + carve interno para early stopping
# ------------------------------------------------------------------
def split_produccion(df, features, target_col=None, seed: int = SEED):
    target_col = target_col or mx.TARGET_COL

    estratos = mx.construir_estratos_precio(df[target_col], mx.N_ESTRATOS_PRECIO)
    train_85, test_15 = train_test_split(
        df, test_size=TEST_SIZE_PRODUCCION, random_state=seed, shuffle=True, stratify=estratos,
    )

    estratos_train = mx.construir_estratos_precio(train_85[target_col], mx.N_ESTRATOS_PRECIO)
    train_fit, train_earlystop = train_test_split(
        train_85, test_size=EARLY_STOP_FRACTION_OF_TRAIN, random_state=seed, shuffle=True,
        stratify=estratos_train,
    )

    print("\nSplit de producción (85/15, con carve interno de early-stopping)")
    print(f"  Train (fit):          {len(train_fit)} filas ({len(train_fit)/len(df)*100:.1f}%)")
    print(f"  Train (early-stop):   {len(train_earlystop)} filas ({len(train_earlystop)/len(df)*100:.1f}%)")
    print(f"  Test (holdout final): {len(test_15)} filas ({len(test_15)/len(df)*100:.1f}%)")

    return train_fit, train_earlystop, test_15


# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------
def parsear_argumentos(argv=None):
    iv_defaults = _cargar_ingenieria_variables()
    parser = argparse.ArgumentParser(
        description="Entrena el modelo de producción. Por defecto regenera el "
                    "dataset desde la base de producción (Supabase)."
    )
    parser.add_argument(
        "--origen-datos", choices=("supabase", "csv"), default="supabase",
        help="supabase: regenera el dataset desde la base de producción (default). "
             "csv: usa el CSV curado tal como está en disco.",
    )
    parser.add_argument(
        "--estados", nargs="+", default=list(iv_defaults.ESTADOS_ENTRENAMIENTO_DEFAULT),
        metavar="ESTADO",
        help="estado_publicacion de los avisos a incluir. "
             f"Default: {' '.join(iv_defaults.ESTADOS_ENTRENAMIENTO_DEFAULT)}",
    )
    parser.add_argument(
        "--meses-max-finalizados", type=int,
        default=iv_defaults.MESES_MAX_ANTIGUEDAD_FINALIZADOS_DEFAULT,
        help="Antigüedad máxima (en meses, sobre fecha_publicacion_aprox) de los avisos "
             "finalizados. Usar 0 para no aplicar límite. "
             f"Default: {iv_defaults.MESES_MAX_ANTIGUEDAD_FINALIZADOS_DEFAULT}",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parsear_argumentos(argv)
    np.random.seed(SEED)

    lib_boosting = getattr(mx, LIB_ATTR_POR_ALGORITMO[ALGORITMO])
    print(f"Algoritmo seleccionado (Tarea 1, {ALGORITMO_SELECCIONADO_PATH.name}): {ALGORITMO}")
    print(f"Versiones — {ALGORITMO}={lib_boosting.__version__}  optuna={mx.optuna.__version__}  "
          f"numpy={np.__version__}")

    if args.origen_datos == "supabase":
        # 0 se interpreta como "sin límite" para poder pedirlo desde la CLI,
        # donde no hay forma natural de pasar None.
        meses = args.meses_max_finalizados or None
        procedencia_dataset = regenerar_dataset_desde_supabase(args.estados, meses)
    else:
        print(f"Usando el CSV curado sin regenerar: {mx.DATASET_PATH}")
        procedencia_dataset = {"origen": "csv", "regenerado": False}

    features = mx.load_selected_features(mx.FEATURES_PATH)
    df = mx.load_dataset(mx.DATASET_PATH, features)
    print(f"Dataset: {len(df)} filas")

    train_fit, train_earlystop, test_15 = split_produccion(df, features)

    resultado = mx.entrenar_y_evaluar_modelo(
        train_fit, train_earlystop, test_15, features,
        seed=SEED,
        n_trials=N_TRIALS_OPTUNA_PRODUCCION,
        cv_splits=CV_SPLITS_OPTUNA_PRODUCCION,
        n_seeds_bagging=N_SEEDS_BAGGING_PRODUCCION,
        persistir_modelo=False,
    )

    models = resultado["models"]
    best_params = resultado["best_params"]

    print(f"\n{'='*60}")
    print(f"SHAP nativo {ALGORITMO} — Feature Importance (Test)")
    print("="*60)
    shap_importance = mx.compute_shap_native(models, resultado["X_test"])
    print(mx.pd.Series(shap_importance).head(15).to_string())

    print(f"\n{'='*60}")
    print("CALIBRACIÓN DE OPORTUNIDAD/CONFIANZA (Test)")
    print("="*60)
    calibracion = calcular_calibracion_oportunidad(models, resultado["X_test"], resultado["y_test"])
    print(f"  Bordes de deciles de precio: {[round(b, 0) for b in calibracion['bordes_deciles_precio']]}")
    print(f"  Bordes de terciles CV:       {[round(b, 4) for b in calibracion['bordes_cv_confianza']]}")

    version, control = generar_version_modelo(ALGORITMO, best_params)
    print(f"\nVersión de modelo generada: {version}")

    version_dir = VERSIONES_DIR / version
    version_dir.mkdir(parents=True, exist_ok=True)

    artefactos_archivados = archivar_artefactos_features(version_dir)

    # Se guarda un dict (no la lista de modelos "pelada") para que
    # 05_prediccion.py sepa con qué librería fue entrenado el ensamble y
    # pueda elegir la función de predicción correcta (XGBoost y LightGBM
    # tienen APIs de predict distintas) sin tener que inferirlo de otra
    # parte ni asumir un único algoritmo posible.
    modelo_path = version_dir / "modelo_produccion.pkl"
    with open(modelo_path, "wb") as f:
        pickle.dump({"algoritmo": ALGORITMO, "modelos": models}, f)

    params_data = {
        "version_modelo": version,
        "algoritmo": ALGORITMO,
        "fecha_entrenamiento": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "features": features,
        "target_col": mx.TARGET_COL,
        "hiperparametros": best_params,
        "optimizacion": resultado["optim_info"],
        "n_seeds_bagging": N_SEEDS_BAGGING_PRODUCCION,
        "tiempo_bagging_seg": resultado["tiempo_bagging_seg"],
        "split": {
            "test_size": TEST_SIZE_PRODUCCION,
            "early_stop_fraction_of_train": EARLY_STOP_FRACTION_OF_TRAIN,
            "seed": SEED,
            "n_train_fit": len(train_fit),
            "n_train_earlystop": len(train_earlystop),
            "n_test": len(test_15),
        },
        "evaluacion_subset_early_stopping": resultado["eval_val"],
        "evaluacion_test": resultado["eval_test"],
        "shap_importance": {k: float(v) for k, v in shap_importance.items()},
        "calibracion_oportunidad": calibracion,
        "dataset_origen": str(mx.DATASET_PATH),
        "features_origen": str(mx.FEATURES_PATH),
        # Con qué población se entrenó esta versión. El CSV se sobrescribe en
        # cada regeneración, así que sin esto no queda registro de los filtros.
        "procedencia_dataset": procedencia_dataset,
        "artefactos_features_archivados": artefactos_archivados,
    }
    params_path = version_dir / "parametros_produccion.json"
    with open(params_path, "w", encoding="utf-8") as f:
        json.dump(params_data, f, indent=4, ensure_ascii=False, cls=mx.NumpyEncoder)

    guardar_control_version(control)

    print(f"\n{'='*60}")
    print("RESUMEN FINAL")
    print("="*60)
    eval_test = resultado["eval_test"]
    print(f"  Algoritmo: {ALGORITMO}")
    print(f"  Versión:  {version}")
    print(f"  TEST → MAE={eval_test['metricas_globales']['mae']:,.0f}  "
          f"RMSE={eval_test['metricas_globales']['rmse']:,.0f}  "
          f"R²={eval_test['metricas_globales']['r2']:.4f}  "
          f"MAPE={eval_test['metricas_globales']['mape']:.2f}%")
    print(f"\n  Modelo guardado:      {modelo_path}")
    print(f"  Parámetros guardados: {params_path}")
    print(f"  Control de versión:   {CONTROL_VERSION_PATH}")


if __name__ == "__main__":
    main()
