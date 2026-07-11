"""
Vulnerabilidad socioterritorial (IGVUST) — pipeline de producción.

A diferencia de las demás etapas, NO reutiliza
`01_obtener_datos/03_vulnerabilidad_socioterritorial.py` vía importlib: ese
script es código de nivel de módulo (sin funciones, sin `if __name__ ==
"__main__"`) que escribe directamente en la base de datos ORIGINAL apenas se
importa — cargarlo dispararía esa escritura contra una base que para
nosotros es de solo lectura. En su lugar, la lógica de cruce espacial
(carga de shapefile, filtro de comunas, sjoin punto-en-polígono) se
reimplementa acá; solo se duplican dos piezas chicas de configuración
(`normalizar_nombre`, `COMUNAS_ANALIZADAS`).

A diferencia de la base original (que guarda `vulnerabilidad_uv` y
`avisos_igvust` como tablas de referencia separadas), acá el resultado del
cruce se resuelve DIRECTO a columnas de `avisos_detalle`
(uv_rsh, rank_nac, pob_rsh_uv, p_urbano) — ver esquema de la Etapa 2.

Incremental: solo procesa avisos con coordenadas y uv_rsh todavía NULL. Una
vez resuelto, no se vuelve a tocar (el cruce no cambia salvo que el
shapefile mismo se actualice).
"""

import logging
import unicodedata
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

import db

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
RUTA_SHAPEFILE = REPO_ROOT / "01_obtener_datos" / "datos_vulnerabilidad" / "202505_IGVUST_UV_cuartil.shp"

# Mismo mapeo que `01_obtener_datos/03_vulnerabilidad_socioterritorial.py`:
# slug de comuna (como en la columna `comuna` de producción) -> nombre tal
# como aparece en el shapefile (columna "Comuna", mayúsculas sin tildes).
COMUNAS_ANALIZADAS = {
    "concepcion-biobio": "CONCEPCION",
    "talcahuano-biobio": "TALCAHUANO",
    "hualpen-biobio": "HUALPEN",
    "san-pedro-de-la-paz-biobio": "SAN PEDRO DE LA PAZ",
    "chiguayante-biobio": "CHIGUAYANTE",
    "penco-biobio": "PENCO",
    "tome-biobio": "TOME",
    "coronel-biobio": "CORONEL",
    "hualqui-biobio": "HUALQUI",
    "lota-biobio": "LOTA",
}


def normalizar_nombre(texto: str) -> str:
    """Mayúsculas y sin tildes, para comparar nombres de comuna sin depender del formato exacto."""
    texto = str(texto).upper().strip()
    texto = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode("ascii")
    return texto


# ------------------------------------------------------------------
# Shapefile
# ------------------------------------------------------------------
def cargar_poligonos_gran_concepcion(ruta_shapefile: Path = RUTA_SHAPEFILE) -> gpd.GeoDataFrame:
    if not ruta_shapefile.exists():
        raise FileNotFoundError(
            f"No se encontró el shapefile de vulnerabilidad en {ruta_shapefile}. "
            f"Es un dato externo no versionado en el repo (ver README, sección de vulnerabilidad)."
        )

    poligonos = gpd.read_file(ruta_shapefile)
    poligonos["comuna_normalizada"] = poligonos["Comuna"].apply(normalizar_nombre)

    nombres_buscados = set(COMUNAS_ANALIZADAS.values())
    filtrado = poligonos[poligonos["comuna_normalizada"].isin(nombres_buscados)].copy()

    encontradas = set(filtrado["comuna_normalizada"].unique())
    no_encontradas = nombres_buscados - encontradas
    if no_encontradas:
        log.warning(f"No se encontraron Unidades Vecinales para: {no_encontradas}")

    log.info(f"Unidades Vecinales del Gran Concepción cargadas: {len(filtrado)}")
    return filtrado


# ------------------------------------------------------------------
# Pendientes
# ------------------------------------------------------------------
def obtener_avisos_pendientes(con) -> pd.DataFrame:
    """Avisos con coordenadas válidas y uv_rsh todavía sin resolver."""
    pendientes = pd.read_sql_query("""
        SELECT id_aviso, latitud, longitud
        FROM avisos_detalle
        WHERE latitud IS NOT NULL AND longitud IS NOT NULL AND uv_rsh IS NULL
    """, con)
    return pendientes


# ------------------------------------------------------------------
# Cruce espacial + guardado
# ------------------------------------------------------------------
def resolver_vulnerabilidad(con, poligonos: gpd.GeoDataFrame, pendientes: pd.DataFrame) -> dict:
    if pendientes.empty:
        return {"avisos_procesados": 0, "avisos_sin_uv": 0}

    geometria = [Point(lon, lat) for lat, lon in zip(pendientes["latitud"], pendientes["longitud"])]
    avisos_geo = gpd.GeoDataFrame(pendientes, geometry=geometria, crs="EPSG:4326")

    if poligonos.crs != avisos_geo.crs:
        poligonos = poligonos.to_crs(avisos_geo.crs)

    resultado = gpd.sjoin(avisos_geo, poligonos, how="left", predicate="within")
    # sjoin puede duplicar filas si un punto cae en más de un polígono (no
    # debería pasar con Unidades Vecinales, pero por seguridad nos quedamos
    # con la primera coincidencia por id_aviso).
    resultado = resultado.drop_duplicates(subset="id_aviso", keep="first")

    columnas_a_guardar = ["uv_rsh", "rank_nac", "pob_rsh_uv", "p_urbano"]
    columnas_disponibles = [c for c in columnas_a_guardar if c in resultado.columns]

    avisos_sin_uv = 0
    for _, fila in resultado.iterrows():
        valores = {c: fila.get(c) for c in columnas_disponibles}
        if pd.isna(valores.get("uv_rsh")):
            avisos_sin_uv += 1
            continue

        con.execute("""
            UPDATE avisos_detalle
            SET uv_rsh = ?, rank_nac = ?, pob_rsh_uv = ?, p_urbano = ?
            WHERE id_aviso = ?
        """, (
            valores.get("uv_rsh"),
            _a_real(valores.get("rank_nac")),
            _a_entero(valores.get("pob_rsh_uv")),
            _a_real(valores.get("p_urbano")),
            fila["id_aviso"],
        ))

    con.commit()

    return {
        "avisos_procesados": len(resultado),
        "avisos_sin_uv": avisos_sin_uv,
    }


def _a_entero(valor):
    if valor is None or pd.isna(valor):
        return None
    try:
        return int(valor)
    except (TypeError, ValueError):
        return None


def _a_real(valor):
    if valor is None or pd.isna(valor):
        return None
    try:
        return float(valor)
    except (TypeError, ValueError):
        return None


# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------
def procesar_vulnerabilidad(con) -> dict:
    pendientes = obtener_avisos_pendientes(con)
    log.info(f"{len(pendientes)} avisos con coordenadas pendientes de resolver vulnerabilidad.")

    if pendientes.empty:
        return {"avisos_procesados": 0, "avisos_sin_uv": 0}

    poligonos = cargar_poligonos_gran_concepcion()
    resumen = resolver_vulnerabilidad(con, poligonos, pendientes)

    if resumen["avisos_sin_uv"]:
        log.warning(f"{resumen['avisos_sin_uv']} avisos sin Unidad Vecinal asignada "
                    f"(coordenadas fuera de las comunas analizadas o sin polígono coincidente).")

    return resumen


if __name__ == "__main__":
    con = db.conectar_produccion()
    resumen = procesar_vulnerabilidad(con)
    con.close()

    log.info(
        f"Corrida completa. Avisos procesados: {resumen['avisos_procesados']} | "
        f"Sin Unidad Vecinal asignada: {resumen['avisos_sin_uv']}"
    )
