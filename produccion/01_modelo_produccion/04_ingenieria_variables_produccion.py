"""
Ingeniería de variables para AVISOS NUEVOS — pipeline de producción.

A diferencia de `01_ingenieria_variables.py` (que recalcula todo en modo
batch/auto-referencial sobre el dataset completo cada vez que corre), este
script PUNTÚA avisos nuevos contra una POBLACIÓN DE REFERENCIA fija:
  - `datos_ingenieria_variables.csv` (el dataset histórico ya limpio e
    imputado) + un SELECT de solo lectura contra la base ORIGINAL para
    recuperar latitud/longitud/comuna (se descartan del CSV final, pero acá
    hacen falta para las búsquedas geográficas).
  - Reutiliza los modelos de imputación de superficie YA ENTRENADOS
    (`aplicar_modelo_guardado`, de 01_ingenieria_variables.py) — no
    reentrena nada.

Solo procesa avisos `tipo_propiedad='departamento'` (igual que el pipeline
de investigación) con detalle scrapeado y que todavía no tengan fila en
`predicciones`. La codificación one-hot de `comuna` no se calcula: no forma
parte de las features seleccionadas.

`FEATURES_ESPERADAS` se lee dinámicamente desde `selected_features.csv` (no
hay un número fijo de features hardcodeado): si la selección de variables de
investigación cambia de tamaño en el futuro, este script no necesita tocarse,
siempre que las features nuevas ya estén cubiertas por `construir_features_produccion`.
"""

import importlib.util
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

import db

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
INVESTIGACION_ROOT = REPO_ROOT / "investigacion"
INGENIERIA_VARIABLES_PATH = INVESTIGACION_ROOT / "03_ingenieria_variables" / "01_ingenieria_variables.py"
FEATURES_PATH = INVESTIGACION_ROOT / "03_ingenieria_variables" / "save" / "seleccion_variables" / "selected_features.csv"


def _cargar_modulo_ingenieria_variables():
    spec = importlib.util.spec_from_file_location("ingenieria_variables_original", INGENIERIA_VARIABLES_PATH)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


iv = _cargar_modulo_ingenieria_variables()
FEATURES_ESPERADAS = pd.read_csv(FEATURES_PATH)["feature"].tolist()

with open(iv.RUTA_SALIDA_NIVELES_BARRIO_DEFAULT, encoding="utf-8") as f:
    _NIVELES_BARRIO = json.load(f)
MAPA_BARRIO_A_NIVEL = _NIVELES_BARRIO["mapa_barrio_a_nivel"]
NIVEL_BARRIO_DEFAULT = _NIVELES_BARRIO["nivel_default"]

ID_COL = "id_aviso"
COLUMNAS_VULNERABILIDAD = ["rank_nac", "pob_rsh_uv", "p_urbano", "c_ig_com"]

MAX_DORMITORIOS = 6
MAX_BANOS = 5
MAX_ESTACIONAMIENTOS = 15


# ------------------------------------------------------------------
# Pendientes
# ------------------------------------------------------------------
def obtener_avisos_pendientes(con) -> pd.DataFrame:
    """Avisos departamento con detalle scrapeado y sin predicción todavía."""
    return pd.read_sql_query("""
        SELECT
            a.id_aviso, a.titulo, a.comuna, a.banos, a.superficie_m2,
            d.dormitorios, d.estacionamientos, d.gastos_comunes,
            d.piscina, d.ascensor, d.cantidad_paraderos, d.cantidad_colegios,
            d.distancia_centro_comuna_m, d.distancia_centro_concepcion_m,
            d.piso_unidad, d.superficie_util_m2, d.superficie_total_m2,
            d.antiguedad_anos, d.rank_nac, d.pob_rsh_uv, d.p_urbano, d.c_ig_com,
            d.latitud, d.longitud,
            d.barrio, d.bodegas, d.conserjeria, d.estacionamiento_visitas,
            d.condominio_cerrado, d.cantidad_jardines_infantiles,
            d.cantidad_supermercados, d.cantidad_plazas, d.cantidad_farmacias,
            d.cantidad_universidades, d.cantidad_centros_comerciales, d.cantidad_clinicas
        FROM avisos a
        JOIN avisos_detalle d ON a.id_aviso = d.id_aviso
        LEFT JOIN predicciones p ON a.id_aviso = p.id_aviso
        WHERE a.tipo_propiedad = 'departamento' AND p.id_aviso IS NULL
    """, con)


def aplicar_filtros_sanidad(df: pd.DataFrame) -> tuple:
    """
    Descarta (con motivo) filas con valores extremos de dormitorios/baños/
    estacionamientos (mismo criterio que `eliminar_outliers_habitaciones`) o
    sin NINGÚN dato de superficie que permita imputar (más laxo que
    `eliminar_registros_incompletos`, que exige puntualmente superficie_m2:
    acá basta con que exista superficie_total_m2, superficie_util_m2 o
    superficie_m2 - cualquiera sirve como punto de partida para imputar).
    Devuelve (df_validos, lista_de_(id_aviso, motivo)_descartados).
    """
    validas, descartadas = [], []

    for _, fila in df.iterrows():
        motivo = None
        if pd.isna(fila["dormitorios"]) or pd.isna(fila["banos"]):
            motivo = "sin dormitorios/baños"
        elif fila["dormitorios"] > MAX_DORMITORIOS:
            motivo = f"dormitorios extremos ({fila['dormitorios']})"
        elif fila["banos"] > MAX_BANOS:
            motivo = f"baños extremos ({fila['banos']})"
        elif pd.notna(fila["estacionamientos"]) and fila["estacionamientos"] > MAX_ESTACIONAMIENTOS:
            motivo = f"estacionamientos extremos ({fila['estacionamientos']})"
        elif pd.isna(fila["superficie_util_m2"]) and pd.isna(fila["superficie_total_m2"]) and pd.isna(fila["superficie_m2"]):
            motivo = "sin ningún dato de superficie"

        (descartadas if motivo else validas).append((fila, motivo))

    df_validas = pd.DataFrame([f for f, _ in validas]) if validas else df.iloc[0:0]
    lista_descartadas = [(f["id_aviso"], m) for f, m in descartadas]
    return df_validas, lista_descartadas


# ------------------------------------------------------------------
# Población de referencia
# ------------------------------------------------------------------
def construir_poblacion_referencia(con_original) -> pd.DataFrame:
    """
    Dataset histórico ya limpio/imputado (datos_ingenieria_variables.csv) +
    latitud/longitud/comuna recuperadas con un SELECT de solo lectura contra
    la base original (esas columnas se descartan del CSV final, pero acá
    hacen falta para las búsquedas geográficas).
    """
    df_csv = pd.read_csv(iv.RUTA_SALIDA_CSV_DEFAULT)

    coords_comuna = pd.read_sql_query("""
        SELECT a.id_aviso, a.comuna, d.latitud, d.longitud
        FROM avisos a
        JOIN avisos_detalle d ON a.id_aviso = d.id_aviso
        WHERE a.tipo_propiedad = 'departamento'
    """, con_original)

    referencia = df_csv.merge(coords_comuna, on="id_aviso", how="inner")
    referencia["latitud"] = pd.to_numeric(referencia["latitud"], errors="coerce")
    referencia["longitud"] = pd.to_numeric(referencia["longitud"], errors="coerce")
    referencia = referencia.dropna(subset=["latitud", "longitud"]).reset_index(drop=True)
    referencia["precio_m2"] = referencia["precio_clp"] / referencia["superficie_util_m2"].replace(0, np.nan)

    log.info(f"Población de referencia: {len(referencia)} departamentos históricos con coordenadas.")
    return referencia


class Referencia:
    """Encapsula los BallTree y fallbacks precalculados UNA vez por corrida
    a partir de la población de referencia, para no reconstruirlos por cada
    aviso nuevo."""

    def __init__(self, referencia: pd.DataFrame):
        self.referencia = referencia

        # --- Árbol para antigüedad (todas las filas: ya viene imputada en el CSV) ---
        self.radio_rad_antiguedad = iv.RADIO_METROS_ANTIGUEDAD / iv.RADIO_TIERRA_M
        self.coords_todas = np.radians(referencia[["latitud", "longitud"]].values)
        self.arbol_todas = BallTree(self.coords_todas, metric="haversine")
        self.antiguedad_valores = referencia["antiguedad_anos"].values
        self.mediana_antiguedad_por_comuna = referencia.groupby("comuna")["antiguedad_anos"].median()
        self.mediana_antiguedad_global = referencia["antiguedad_anos"].median()

        # --- Árbol para precio_m2_sector (solo filas con precio/m2 válido y dentro del IQR) ---
        self.radio_rad_precio = iv.RADIO_METROS_COMPARADOR_SECTOR / iv.RADIO_TIERRA_M
        precio_m2_valido = referencia["precio_m2"].notna()
        lim_inf, lim_sup = iv.limites_iqr(referencia.loc[precio_m2_valido, "precio_m2"], iv.MULTIPLICADOR_IQR)
        es_razonable = referencia["precio_m2"].between(lim_inf, lim_sup)
        idx_validos = referencia.index[precio_m2_valido & es_razonable]

        self.precios_m2_validos = referencia.loc[idx_validos, "precio_m2"].values
        self.mediana_precio_m2_fallback = (
            float(np.median(self.precios_m2_validos)) if len(self.precios_m2_validos) else np.nan
        )
        self.arbol_precio_m2 = (
            BallTree(np.radians(referencia.loc[idx_validos, ["latitud", "longitud"]].values), metric="haversine")
            if len(idx_validos) else None
        )

        # --- Fallbacks de vulnerabilidad por comuna ---
        self.medias_vuln_por_comuna = {
            c: referencia.groupby("comuna")[c].mean() for c in COLUMNAS_VULNERABILIDAD
        }
        self.medias_vuln_global = {c: referencia[c].mean() for c in COLUMNAS_VULNERABILIDAD}

        # --- Fallback de piso_unidad (promedio histórico, ya sin outliers en el CSV) ---
        self.piso_promedio = referencia["piso_unidad"].mean()

    def antiguedad(self, lat, lon, comuna, valor_actual) -> float:
        if pd.notna(valor_actual):
            return float(valor_actual)
        if pd.notna(lat) and pd.notna(lon):
            punto = np.radians([[lat, lon]])
            vecinos = self.arbol_todas.query_radius(punto, r=self.radio_rad_antiguedad)[0]
            if len(vecinos) > 0:
                return float(np.median(self.antiguedad_valores[vecinos]))
        if comuna in self.mediana_antiguedad_por_comuna.index and pd.notna(self.mediana_antiguedad_por_comuna[comuna]):
            return float(self.mediana_antiguedad_por_comuna[comuna])
        return float(self.mediana_antiguedad_global)

    def precio_m2_sector(self, lat, lon) -> tuple:
        if self.arbol_precio_m2 is None or pd.isna(lat) or pd.isna(lon):
            return self.mediana_precio_m2_fallback, False
        punto = np.radians([[lat, lon]])
        vecinos = self.arbol_precio_m2.query_radius(punto, r=self.radio_rad_precio)[0]
        if len(vecinos) == 0:
            return self.mediana_precio_m2_fallback, False
        return float(np.median(self.precios_m2_validos[vecinos])), True

    def vulnerabilidad(self, comuna, columna, valor_actual) -> float:
        if pd.notna(valor_actual):
            return float(valor_actual)
        serie_comuna = self.medias_vuln_por_comuna[columna]
        if comuna in serie_comuna.index and pd.notna(serie_comuna[comuna]):
            return float(serie_comuna[comuna])
        return float(self.medias_vuln_global[columna])

    def piso_unidad(self, valor_actual) -> float:
        if pd.isna(valor_actual):
            return 1.0
        if valor_actual > 30:
            return float(self.piso_promedio)
        return float(valor_actual)


# ------------------------------------------------------------------
# Imputación de superficie (modelos ya entrenados, sin reentrenar)
# ------------------------------------------------------------------
def imputar_superficies(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["superficie_util_m2"] = pd.to_numeric(df["superficie_util_m2"], errors="coerce")
    df["superficie_total_m2"] = pd.to_numeric(df["superficie_total_m2"], errors="coerce")

    necesita_util = df["superficie_util_m2"].isna() | (df["superficie_util_m2"] < iv.UMBRAL_MINIMO_M2)
    if necesita_util.any():
        try:
            estimaciones = iv.aplicar_modelo_guardado(df.loc[necesita_util], "superficie_util_m2", "departamento")
            df.loc[necesita_util, "superficie_util_m2"] = np.round(estimaciones, 1)
        except FileNotFoundError:
            log.warning("No se encontró el modelo de imputación de superficie_util_m2; se deja sin corregir.")

    necesita_total = df["superficie_total_m2"].isna() | (df["superficie_total_m2"] < iv.UMBRAL_MINIMO_M2)
    if necesita_total.any():
        try:
            estimaciones = iv.aplicar_modelo_guardado(df.loc[necesita_total], "superficie_total_m2", "departamento")
            df.loc[necesita_total, "superficie_total_m2"] = np.round(estimaciones, 1)
        except FileNotFoundError:
            log.warning("No se encontró el modelo de imputación de superficie_total_m2; se deja sin corregir.")

    # Último respaldo: superficie_m2 cruda de la grilla (igual que
    # `completar_superficies_faltantes` en el pipeline de investigación).
    df["superficie_util_m2"] = df["superficie_util_m2"].fillna(df["superficie_m2"])
    df["superficie_total_m2"] = df["superficie_total_m2"].fillna(df["superficie_m2"])

    return df


# ------------------------------------------------------------------
# Normalización de las columnas nuevas (features que se sumaron a las 20
# originales) — mismo patrón que `preprocesar_variables_amenities` y
# `rellenar_cantidad_pois_cercanos` en 01_ingenieria_variables.py: ausencia
# de dato equivale a "no tiene"/"no hay ninguno cerca", así que se rellena
# con 0 en vez de imputar contra la población de referencia.
#
# A diferencia de la base de investigación (donde estos campos llegan como
# texto crudo "Sí"/"No"), `02_scraper_detalle_incremental.py` YA los
# convierte a 0/1/NULL antes de insertarlos en avisos_detalle de producción
# (ver `_a_booleano`/`_a_entero` ahí) — acá solo falta rellenar los NULL.
# ------------------------------------------------------------------
def normalizar_columnas_nuevas(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    columnas_a_cero = [
        "bodegas", "conserjeria", "estacionamiento_visitas",
        "cantidad_jardines_infantiles", "cantidad_supermercados",
        "cantidad_plazas", "cantidad_farmacias", "cantidad_universidades",
        "cantidad_centros_comerciales", "cantidad_clinicas",
    ]
    df[columnas_a_cero] = df[columnas_a_cero].apply(pd.to_numeric, errors="coerce")
    df[columnas_a_cero] = df[columnas_a_cero].fillna(0)

    # condominio_cerrado: sin dato -> False/0 (mismo fallback final que usa
    # 01_ingenieria_variables.py cuando ni la moda por edificio resuelve el
    # nulo). No se intenta imputar contra la población de referencia.
    df["condominio_cerrado"] = pd.to_numeric(df["condominio_cerrado"], errors="coerce").fillna(0).astype(int)

    # nivel_barrio: diccionario barrio->nivel ya calculado en investigación
    # (03_ingenieria_variables/save/ingeniaria_variables/niveles_barrio.json).
    # Un barrio ausente o no visto en el diccionario cae a NIVEL_BARRIO_DEFAULT.
    df["nivel_barrio"] = df["barrio"].map(MAPA_BARRIO_A_NIVEL).fillna(NIVEL_BARRIO_DEFAULT).astype(int)

    return df


# ------------------------------------------------------------------
# Construcción de las features seleccionadas (30 actualmente, leídas
# dinámicamente desde selected_features.csv)
# ------------------------------------------------------------------
def construir_features_produccion(con_produccion, con_original) -> pd.DataFrame:
    columnas_salida = [ID_COL] + FEATURES_ESPERADAS

    pendientes = obtener_avisos_pendientes(con_produccion)
    log.info(f"{len(pendientes)} avisos departamento pendientes de calcular features.")
    if pendientes.empty:
        return pd.DataFrame(columns=columnas_salida)

    pendientes, descartados = aplicar_filtros_sanidad(pendientes)
    for id_aviso, motivo in descartados:
        log.warning(f"{id_aviso}: se salta esta corrida ({motivo}).")

    if pendientes.empty:
        return pd.DataFrame(columns=columnas_salida)

    referencia_df = construir_poblacion_referencia(con_original)
    referencia = Referencia(referencia_df)

    pendientes = imputar_superficies(pendientes)
    pendientes["ratio_total_util"] = (
        pendientes["superficie_total_m2"] / pendientes["superficie_util_m2"].replace(0, np.nan)
    )
    pendientes = normalizar_columnas_nuevas(pendientes)

    filas = []
    for _, fila in pendientes.iterrows():
        precio_m2_sector, tiene_comparables = referencia.precio_m2_sector(fila["latitud"], fila["longitud"])
        titulo = fila["titulo"] or ""

        filas.append({
            "id_aviso": fila["id_aviso"],
            "amoblado": int(bool(iv.PATRON_AMOBLADO.search(titulo))),
            "estacionamientos": fila["estacionamientos"] if pd.notna(fila["estacionamientos"]) else 0,
            "gastos_comunes": fila["gastos_comunes"] if pd.notna(fila["gastos_comunes"]) else 0,
            "banos": fila["banos"],
            "antiguedad_anos": referencia.antiguedad(fila["latitud"], fila["longitud"], fila["comuna"], fila["antiguedad_anos"]),
            "precio_m2_sector_departamento": precio_m2_sector,
            "rank_nac": referencia.vulnerabilidad(fila["comuna"], "rank_nac", fila["rank_nac"]),
            "piso_unidad": referencia.piso_unidad(fila["piso_unidad"]),
            "piscina": fila["piscina"] if pd.notna(fila["piscina"]) else 0,
            "distancia_centro_concepcion_m": fila["distancia_centro_concepcion_m"],
            "superficie_util_m2": fila["superficie_util_m2"],
            "superficie_total_m2": fila["superficie_total_m2"],
            "ratio_total_util": fila["ratio_total_util"],
            "pob_rsh_uv": referencia.vulnerabilidad(fila["comuna"], "pob_rsh_uv", fila["pob_rsh_uv"]),
            "cantidad_paraderos": fila["cantidad_paraderos"] if pd.notna(fila["cantidad_paraderos"]) else 0,
            "distancia_centro_comuna_m": fila["distancia_centro_comuna_m"],
            "cantidad_colegios": fila["cantidad_colegios"] if pd.notna(fila["cantidad_colegios"]) else 0,
            "p_urbano": referencia.vulnerabilidad(fila["comuna"], "p_urbano", fila["p_urbano"]),
            "ascensor": fila["ascensor"] if pd.notna(fila["ascensor"]) else 0,
            "tiene_comparables_cercanos": int(tiene_comparables),
            "bodegas": fila["bodegas"],
            "conserjeria": fila["conserjeria"],
            "estacionamiento_visitas": fila["estacionamiento_visitas"],
            "condominio_cerrado": int(fila["condominio_cerrado"]),
            "cantidad_jardines_infantiles": fila["cantidad_jardines_infantiles"],
            "cantidad_supermercados": fila["cantidad_supermercados"],
            "cantidad_plazas": fila["cantidad_plazas"],
            "cantidad_farmacias": fila["cantidad_farmacias"],
            "cantidad_universidades": fila["cantidad_universidades"],
            "cantidad_centros_comerciales": fila["cantidad_centros_comerciales"],
            "cantidad_clinicas": fila["cantidad_clinicas"],
            "c_ig_com": referencia.vulnerabilidad(fila["comuna"], "c_ig_com", fila["c_ig_com"]),
            "nivel_barrio": fila["nivel_barrio"],
        })

    resultado = pd.DataFrame(filas)
    return resultado[columnas_salida]


if __name__ == "__main__":
    con_produccion = db.conectar_produccion()
    con_original = db.conectar_original()

    features_df = construir_features_produccion(con_produccion, con_original)

    con_produccion.close()
    con_original.close()

    log.info(f"Features calculadas para {len(features_df)} avisos.")
    if not features_df.empty:
        print(features_df.to_string())
