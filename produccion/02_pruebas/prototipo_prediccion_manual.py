"""
PROTOTIPO DE PRUEBA — NO ES PARTE DEL PIPELINE PRODUCTIVO.

Permite ingresar a mano lat/lon + características de un departamento y
obtener (a) el precio estimado por el ensamble de producción vigente y
(b) la banda de precio que el sistema consideraría "precio de mercado" (ni
oportunidad ni caro), reutilizando el modelo y la calibración ya entrenados
en `produccion/01_modelo_produccion/` (sin duplicar esa lógica, vía importlib).

No toca ningún archivo fuera de `05.1_pruebas/`. Vive aislado a propósito:
es solo para validar la idea antes de integrarla a la app de Streamlit.

CÓMO CORRERLO:
    python 05.1_pruebas/prototipo_prediccion_manual.py
Edita el bloque de constantes más abajo para probar una propiedad distinta.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from shapely.geometry import Point

SCRIPT_DIR = Path(__file__).resolve().parent
PRODUCCION_ROOT = SCRIPT_DIR.parent
MODELO_PRODUCCION_DIR = PRODUCCION_ROOT / "01_modelo_produccion"

# `05_prediccion.py` y sus dependencias (`db.py`, `04_ingenieria_variables_
# produccion.py`) usan `import db` normal (no importlib) - eso solo resuelve
# si el directorio está en sys.path, así que se agrega ANTES de cargarlos.
sys.path.insert(0, str(MODELO_PRODUCCION_DIR))


def _cargar_modulo(nombre: str, ruta: Path):
    spec = importlib.util.spec_from_file_location(nombre, ruta)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


# pred: 05_prediccion.py -> cargar_modelo_y_calibracion(), y de paso deja
# disponibles pred.ivp (04_ingenieria_variables_produccion.py ya cargado) y
# pred.db (db.py ya importado), sin volver a cargarlos por separado.
pred = _cargar_modulo("prediccion_produccion", MODELO_PRODUCCION_DIR / "05_prediccion.py")
vuln = _cargar_modulo("vulnerabilidad_produccion", MODELO_PRODUCCION_DIR / "03_vulnerabilidad_produccion.py")

ivp = pred.ivp
db = pred.db


# ══════════════════════════════════════════════════════════════════════════
# BLOQUE DE CONSTANTES EDITABLES — cambia estos valores para probar una
# propiedad distinta. Cada una indica su formato/opciones válidas.
# ══════════════════════════════════════════════════════════════════════════

# --- Ubicación (para distancias, comparables cercanos y vulnerabilidad) ---
LATITUD = -36.8265            # float, grados decimales (ej. -36.8265)
LONGITUD = -73.0524           # float, grados decimales (ej. -73.0524)
COMUNA = "concepcion-biobio"   # una de: concepcion-biobio, talcahuano-biobio,
                               # hualpen-biobio, san-pedro-de-la-paz-biobio,
                               # chiguayante-biobio, penco-biobio, tome-biobio,
                               # coronel-biobio, hualqui-biobio, lota-biobio
BARRIO = ""                    # nombre de barrio tal como en niveles_barrio.json
                               # (03_ingenieria_variables/save/ingeniaria_variables/
                               # niveles_barrio.json); "" o cualquier valor no
                               # reconocido cae al nivel por defecto

# --- Características básicas ---
DORMITORIOS = 2                # int, 0-6 (solo para filtros de sanidad e
                                # imputación de superficie; no es feature del modelo)
BANOS = 2                      # int, 0-5
ESTACIONAMIENTOS = 2           # int, 0-15
SUPERFICIE_UTIL_M2 = 56.0      # float, m2
SUPERFICIE_TOTAL_M2 = 60.0     # float, m2
PISO_UNIDAD = 2                # int (1 = primer piso; >30 se trata como dato
                                # erróneo y se reemplaza por el promedio histórico)
ANTIGUEDAD_ANOS = 3         # int, o None para estimarla por comparables
                                # cercanos/comuna (igual que producción cuando
                                # el aviso no trae antigüedad)
GASTOS_COMUNES = 95000         # float, CLP/mes (0 si no tiene)

# --- Atributos sí/no (0 = no, 1 = sí) ---
AMOBLADO = 0
PISCINA = 0
ASCENSOR = 1
CONSERJERIA = 1
CONDOMINIO_CERRADO = 0
ESTACIONAMIENTO_VISITAS = 0

# --- Bodegas (cantidad, no booleano) ---
BODEGAS = 0                    # int, cantidad de bodegas (0 si no tiene)

# --- Entorno / POIs cercanos (cantidad de cada uno) ---
# En producción estos conteos vienen de una búsqueda tipo Google Places
# durante el scraping, no de una fórmula geográfica simple: no se replican
# desde lat/lon en este prototipo (fuera de alcance), así que quedan como
# input manual aunque conceptualmente dependan de la ubicación. Si no se
# sabe el valor real, una aproximación razonable es mirar un aviso cercano
# ya existente en la base y copiar sus conteos.
CANTIDAD_COLEGIOS = 2               # int, >= 0
CANTIDAD_SUPERMERCADOS = 0          # int, >= 0
CANTIDAD_JARDINES_INFANTILES = 0    # int, >= 0
CANTIDAD_PARADEROS = 5              # int, >= 0
CANTIDAD_CENTROS_COMERCIALES = 0    # int, >= 0
CANTIDAD_PLAZAS = 2                 # int, >= 0
CANTIDAD_FARMACIAS = 0              # int, >= 0
CANTIDAD_CLINICAS = 0               # int, >= 0


# ══════════════════════════════════════════════════════════════════════════
# Distancias al centro (duplica COMUNA_CENTROS/haversine de
# 01_obtener_datos/02_scraper_detalle.py::calcular_distancias_centro - no se
# importa ese archivo completo solo por esta constante geográfica estable,
# para no arrastrar sus dependencias de scraping (requests/bs4/lxml).)
# ══════════════════════════════════════════════════════════════════════════
COMUNA_CENTROS = {
    "concepcion-biobio":          (-36.8265, -73.0524),
    "talcahuano-biobio":          (-36.7249, -73.1149),
    "hualpen-biobio":             (-36.7690, -73.1000),
    "san-pedro-de-la-paz-biobio": (-36.8380, -73.0970),
    "chiguayante-biobio":         (-36.9280, -73.0230),
    "penco-biobio":               (-36.7420, -72.9970),
    "tome-biobio":                (-36.6180, -72.9570),
    "coronel-biobio":             (-37.0270, -73.1370),
    "hualqui-biobio":             (-36.9670, -72.9420),
    "lota-biobio":                (-37.0920, -73.1600),
}
CENTRO_CONCEPCION = COMUNA_CENTROS["concepcion-biobio"]


def haversine_metros(lat1, lon1, lat2, lon2) -> float:
    radio_tierra_m = 6371000
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    a = np.sin(delta_phi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2) ** 2
    return 2 * radio_tierra_m * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


# ══════════════════════════════════════════════════════════════════════════
# Vulnerabilidad socioterritorial (rank_nac/pob_rsh_uv/p_urbano/c_ig_com):
# cruce punto-en-polígono real (IGVUST) reutilizando la tabla ya precalculada
# `poligonos_vulnerabilidad_uv`, igual que 03_vulnerabilidad_produccion.py.
# Si el punto no cae en ningún polígono, cae al mismo fallback por comuna/
# global que ya usa `Referencia.vulnerabilidad`.
# ══════════════════════════════════════════════════════════════════════════
def resolver_vulnerabilidad_puntual(con_produccion, referencia, lat: float, lon: float, comuna: str) -> dict:
    poligonos = vuln.cargar_poligonos_gran_concepcion(con_produccion)
    punto = Point(lon, lat)
    encontrado = next((p for p in poligonos if p["geometria"].contains(punto)), None)

    columnas = ["rank_nac", "pob_rsh_uv", "p_urbano", "c_ig_com"]
    if encontrado is not None:
        return {c: float(encontrado[c]) for c in columnas}
    return {c: referencia.vulnerabilidad(comuna, c, np.nan) for c in columnas}


# ══════════════════════════════════════════════════════════════════════════
# Construcción de features para UNA consulta puntual, reutilizando
# `Referencia` (BallTree de comparables/antigüedad + fallbacks) de
# 04_ingenieria_variables_produccion.py tal cual - ya soporta consultas
# punto por punto, solo hace falta construirla una vez sobre la población de
# referencia y llamarla directo, sin pasar por `construir_features_produccion`
# (que además hace I/O de BD de avisos pendientes que no aplica acá).
# ══════════════════════════════════════════════════════════════════════════
def construir_fila_features(con_produccion, con_original, features_esperadas: list) -> pd.DataFrame:
    referencia_df = ivp.construir_poblacion_referencia(con_original)
    referencia = ivp.Referencia(referencia_df)

    precio_m2_sector, tiene_comparables = referencia.precio_m2_sector(LATITUD, LONGITUD)
    antiguedad = referencia.antiguedad(LATITUD, LONGITUD, COMUNA, ANTIGUEDAD_ANOS)
    piso_unidad = referencia.piso_unidad(PISO_UNIDAD)
    vulnerabilidad = resolver_vulnerabilidad_puntual(con_produccion, referencia, LATITUD, LONGITUD, COMUNA)

    centro_comuna = COMUNA_CENTROS.get(COMUNA)
    distancia_centro_comuna_m = (
        haversine_metros(LATITUD, LONGITUD, *centro_comuna) if centro_comuna else np.nan
    )
    distancia_centro_concepcion_m = haversine_metros(LATITUD, LONGITUD, *CENTRO_CONCEPCION)

    nivel_barrio = ivp.MAPA_BARRIO_A_NIVEL.get(BARRIO, ivp.NIVEL_BARRIO_DEFAULT)

    fila = {
        "amoblado": AMOBLADO,
        "estacionamientos": ESTACIONAMIENTOS,
        "piso_unidad": piso_unidad,
        "precio_m2_sector_departamento": precio_m2_sector,
        "banos": BANOS,
        "gastos_comunes": GASTOS_COMUNES,
        "rank_nac": vulnerabilidad["rank_nac"],
        "antiguedad_anos": antiguedad,
        "distancia_centro_concepcion_m": distancia_centro_concepcion_m,
        "bodegas": BODEGAS,
        "piscina": PISCINA,
        "superficie_util_m2": SUPERFICIE_UTIL_M2,
        "distancia_centro_comuna_m": distancia_centro_comuna_m,
        "superficie_total_m2": SUPERFICIE_TOTAL_M2,
        "ascensor": ASCENSOR,
        "ratio_total_util": SUPERFICIE_TOTAL_M2 / SUPERFICIE_UTIL_M2,
        "p_urbano": vulnerabilidad["p_urbano"],
        "pob_rsh_uv": vulnerabilidad["pob_rsh_uv"],
        "nivel_barrio": nivel_barrio,
        "cantidad_colegios": CANTIDAD_COLEGIOS,
        "condominio_cerrado": CONDOMINIO_CERRADO,
        "cantidad_supermercados": CANTIDAD_SUPERMERCADOS,
        "tiene_comparables_cercanos": int(tiene_comparables),
        "cantidad_jardines_infantiles": CANTIDAD_JARDINES_INFANTILES,
        "cantidad_paraderos": CANTIDAD_PARADEROS,
        "estacionamiento_visitas": ESTACIONAMIENTO_VISITAS,
        "conserjeria": CONSERJERIA,
        "cantidad_centros_comerciales": CANTIDAD_CENTROS_COMERCIALES,
        "cantidad_plazas": CANTIDAD_PLAZAS,
        "cantidad_farmacias": CANTIDAD_FARMACIAS,
        "cantidad_clinicas": CANTIDAD_CLINICAS,
        "c_ig_com": vulnerabilidad["c_ig_com"],
    }

    return pd.DataFrame([fila])[features_esperadas]


# ══════════════════════════════════════════════════════════════════════════
# Banda de precio "de mercado": como no hay un precio real que comparar (a
# diferencia de un aviso ya publicado), se ubica el decil de calibración con
# el propio precio_predicho en vez del precio_real - es la única ancla
# disponible, y el error mediano por decil es chico frente al ancho de cada
# banda de precio, así que casi siempre cae en el mismo decil que caería el
# precio real. Fórmula derivada despejando precio_real de z_robusto en el
# punto donde cruza cada umbral (mismo signo que 05_prediccion.py: z_robusto
# > umbral_caro -> "caro", z_robusto < -umbral_oportunidad -> "oportunidad").
# ══════════════════════════════════════════════════════════════════════════
def calcular_banda_mercado(precio_predicho: float, calibracion: dict) -> dict:
    bordes_deciles = calibracion["bordes_deciles_precio"]
    stats_por_decil = calibracion["stats_por_decil"]
    mad_scale = calibracion["mad_scale_const"]
    umbral_oportunidad = calibracion["umbral_oportunidad"]
    umbral_caro = calibracion["umbral_caro"]

    decil = int(pd.cut([precio_predicho], bins=bordes_deciles, labels=False, include_lowest=True)[0])
    stats_decil = stats_por_decil[str(decil)]
    mediana_decil = stats_decil["mediana_error"]
    mad_ajustado = stats_decil["mad_error"] * mad_scale

    limite_inferior = precio_predicho + mediana_decil - umbral_oportunidad * mad_ajustado
    limite_superior = precio_predicho + mediana_decil + umbral_caro * mad_ajustado

    return {
        "decil": decil,
        "limite_inferior": limite_inferior,
        "limite_superior": limite_superior,
    }


def nivel_confianza_de(cv_ensamble: float, calibracion: dict) -> str:
    bordes_cv = calibracion["bordes_cv_confianza"]
    etiquetas_confianza = calibracion["etiquetas_confianza"]
    idx = int(pd.cut([cv_ensamble], bins=bordes_cv, labels=False, include_lowest=True)[0])
    return etiquetas_confianza[idx]


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════
def predecir_propiedad_manual() -> dict:
    modelo_info = pred.cargar_modelo_y_calibracion()
    con_produccion = db.conectar_produccion()
    con_original = db.conectar_original()

    try:
        X = construir_fila_features(con_produccion, con_original, modelo_info["features"])
    finally:
        con_produccion.close()
        con_original.close()

    pred_matrix = modelo_info["predict_ensemble_matrix"](modelo_info["models"], X)
    precio_predicho = float(pred_matrix.mean(axis=0)[0])
    pred_std = float(pred_matrix.std(axis=0)[0])
    cv_ensamble = pred_std / precio_predicho if precio_predicho else float("inf")

    banda = calcular_banda_mercado(precio_predicho, modelo_info["calibracion"])
    confianza = nivel_confianza_de(cv_ensamble, modelo_info["calibracion"])

    resultado = {
        "version_modelo": modelo_info["version_modelo"],
        "precio_predicho": precio_predicho,
        "cv_ensamble": cv_ensamble,
        "nivel_confianza": confianza,
        **banda,
    }

    print(f"Modelo vigente: {resultado['version_modelo']}")
    print(f"Precio estimado: {resultado['precio_predicho']:,.0f} CLP")
    print(f"Rango de precio 'de mercado': {resultado['limite_inferior']:,.0f} - {resultado['limite_superior']:,.0f} CLP "
          f"(decil {resultado['decil']})")
    print(f"Confianza de la estimación: {resultado['nivel_confianza']} (cv_ensamble={resultado['cv_ensamble']:.4f})")

    return resultado


if __name__ == "__main__":
    predecir_propiedad_manual()
