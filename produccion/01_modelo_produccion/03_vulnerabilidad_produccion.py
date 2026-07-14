"""
Vulnerabilidad socioterritorial (IGVUST) — pipeline de producción.

Los polígonos de Unidad Vecinal de las 10 comunas del Gran Concepción ya NO
se leen desde el shapefile IGVUST en cada corrida: viven precalculados en la
tabla `poligonos_vulnerabilidad_uv` de esta misma base de datos de
producción. Esa tabla se llena UNA VEZ (o cada vez que el shapefile se
actualiza) corriendo `migrar_poligonos_vulnerabilidad.py` a mano y
localmente — no es parte de esta etapa ni del orquestador. Esto evita que el
pipeline de producción dependa de un archivo externo no versionado en el
repo (el shapefile está en .gitignore, así que en GitHub Actions no existe)
y, de paso, saca a `geopandas`/GDAL de las dependencias de producción: el
cruce punto-en-polígono se resuelve acá con `shapely` puro sobre la
geometría ya guardada como WKT (en EPSG:4326).

A diferencia de la base original (que guarda `vulnerabilidad_uv` y
`avisos_igvust` como tablas de referencia separadas), acá el resultado del
cruce se resuelve DIRECTO a columnas de `avisos_detalle`
(uv_rsh, rank_nac, pob_rsh_uv, p_urbano, c_ig_com) — ver esquema en `db.py`.

Incremental: solo procesa avisos con coordenadas y uv_rsh todavía NULL. Una
vez resuelto, no se vuelve a tocar (el cruce no cambia salvo que el
shapefile mismo se actualice y se re-corra la migración).
"""

import logging
import unicodedata

import pandas as pd
from shapely import wkt
from shapely.geometry import Point

import db

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# Mismo mapeo que `01_obtener_datos/03_vulnerabilidad_socioterritorial.py`:
# slug de comuna (como en la columna `comuna` de producción) -> nombre tal
# como aparece en el shapefile (columna "Comuna", mayúsculas sin tildes). Se
# mantiene acá — aunque esta etapa ya no lee el shapefile — porque
# `migrar_poligonos_vulnerabilidad.py` la reutiliza vía importlib para no
# triplicar esta configuración.
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
# Polígonos (desde la tabla precalculada, no desde el shapefile)
# ------------------------------------------------------------------
def cargar_poligonos_gran_concepcion(con) -> list:
    filas = con.execute("""
        SELECT uv_rsh, comuna, rank_nac, pob_rsh_uv, p_urbano, c_ig_com, geometria_wkt
        FROM poligonos_vulnerabilidad_uv
    """).fetchall()

    if not filas:
        raise RuntimeError(
            "La tabla poligonos_vulnerabilidad_uv está vacía. Corre "
            "migrar_poligonos_vulnerabilidad.py una vez, localmente (con el "
            "shapefile IGVUST disponible), antes de correr esta etapa."
        )

    poligonos = [
        {
            "uv_rsh": f[0], "comuna": f[1], "rank_nac": f[2], "pob_rsh_uv": f[3],
            "p_urbano": f[4], "c_ig_com": f[5], "geometria": wkt.loads(f[6]),
        }
        for f in filas
    ]
    log.info(f"{len(poligonos)} polígonos de Unidad Vecinal cargados desde poligonos_vulnerabilidad_uv.")
    return poligonos


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
# Cruce espacial (shapely puro) + guardado
# ------------------------------------------------------------------
def resolver_vulnerabilidad(con, poligonos: list, pendientes: pd.DataFrame) -> dict:
    if pendientes.empty:
        return {"avisos_procesados": 0, "avisos_sin_uv": 0}

    avisos_sin_uv = 0
    for _, fila in pendientes.iterrows():
        punto = Point(fila["longitud"], fila["latitud"])
        # Primera Unidad Vecinal cuyo polígono contiene el punto (no
        # deberían solaparse entre sí, pero por seguridad nos quedamos con
        # la primera coincidencia).
        encontrado = next((p for p in poligonos if p["geometria"].contains(punto)), None)

        if encontrado is None:
            avisos_sin_uv += 1
            continue

        con.execute("""
            UPDATE avisos_detalle
            SET uv_rsh = ?, rank_nac = ?, pob_rsh_uv = ?, p_urbano = ?, c_ig_com = ?
            WHERE id_aviso = ?
        """, (
            encontrado["uv_rsh"], encontrado["rank_nac"], encontrado["pob_rsh_uv"],
            encontrado["p_urbano"], encontrado["c_ig_com"], fila["id_aviso"],
        ))

    con.commit()

    return {
        "avisos_procesados": len(pendientes),
        "avisos_sin_uv": avisos_sin_uv,
    }


# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------
def procesar_vulnerabilidad(con) -> dict:
    pendientes = obtener_avisos_pendientes(con)
    log.info(f"{len(pendientes)} avisos con coordenadas pendientes de resolver vulnerabilidad.")

    if pendientes.empty:
        return {"avisos_procesados": 0, "avisos_sin_uv": 0}

    poligonos = cargar_poligonos_gran_concepcion(con)
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
