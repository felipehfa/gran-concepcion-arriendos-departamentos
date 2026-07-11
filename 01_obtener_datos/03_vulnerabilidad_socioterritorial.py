"""
03_vulnerabilidad_socioterritorial.py

Cruza las coordenadas de cada aviso (tabla avisos_detalle) con los polígonos
de Unidad Vecinal del IGVUST, filtrando SOLO las comunas del Gran Concepción
que estamos analizando, y guarda dos tablas nuevas en la misma base de datos:

  - `vulnerabilidad_uv`: los datos de vulnerabilidad por Unidad Vecinal
    (uv_rsh como llave), solo para las comunas analizadas.
  - `avisos_igvust`: el cruce id_aviso -> uv_rsh (la llave para unir cada
    aviso con su Unidad Vecinal correspondiente en la tabla de arriba).

Requiere que 202505_IGVUST_UV_cuartil.shp (+ .dbf/.shx/.prj) estén en la
carpeta `datos_vulnerabilidad`, dentro de la misma carpeta que este script.
"""

import sqlite3
import unicodedata
import geopandas as gpd
import pandas as pd
from pathlib import Path
from shapely.geometry import Point

CARPETA_SCRIPT = Path(__file__).resolve().parent
RUTA_SHAPEFILE = CARPETA_SCRIPT / "datos_vulnerabilidad" / "202505_IGVUST_UV_cuartil.shp"
RUTA_BD = CARPETA_SCRIPT / "avisos_gran_concepcion.db"

# Comunas del Gran Concepción que estamos analizando (mismo slug que usa
# 01_scraper_grilla.py), mapeadas a como aparece el nombre en el shapefile
# (columna "Comuna", en mayúsculas y sin tildes).
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
# Paso 1: Cargar el shapefile de polígonos de Unidad Vecinal
# ------------------------------------------------------------------
uv_poligonos = gpd.read_file(RUTA_SHAPEFILE)
print(f"CRS del shapefile: {uv_poligonos.crs}")
print(f"Unidades Vecinales totales (todo Chile): {len(uv_poligonos)}")

# ------------------------------------------------------------------
# Paso 2: Filtrar SOLO las comunas que estamos analizando
# ------------------------------------------------------------------
uv_poligonos["comuna_normalizada"] = uv_poligonos["Comuna"].apply(normalizar_nombre)
nombres_buscados = set(COMUNAS_ANALIZADAS.values())

uv_filtrado = uv_poligonos[uv_poligonos["comuna_normalizada"].isin(nombres_buscados)].copy()

# Chequeo de sanidad: ¿todas las comunas de nuestra lista aparecieron en el shapefile?
encontradas = set(uv_filtrado["comuna_normalizada"].unique())
no_encontradas = nombres_buscados - encontradas
if no_encontradas:
    print(f"ADVERTENCIA: no se encontraron Unidades Vecinales para: {no_encontradas} "
          f"- revisa si el nombre en COMUNAS_ANALIZADAS coincide con el shapefile.")

print(f"Unidades Vecinales tras filtrar por las {len(COMUNAS_ANALIZADAS)} comunas analizadas: {len(uv_filtrado)}")

# Mapear de vuelta al slug del proyecto (concepcion-biobio, etc.)
mapa_nombre_a_slug = {v: k for k, v in COMUNAS_ANALIZADAS.items()}
uv_filtrado["comuna_slug"] = uv_filtrado["comuna_normalizada"].map(mapa_nombre_a_slug)

# ------------------------------------------------------------------
# Paso 3: Cargar coordenadas de los avisos desde la base de datos
# ------------------------------------------------------------------
con = sqlite3.connect(RUTA_BD)
avisos = pd.read_sql_query(
    "SELECT id_aviso, latitud, longitud FROM avisos_detalle "
    "WHERE latitud IS NOT NULL AND longitud IS NOT NULL",
    con,
)

avisos["latitud"] = pd.to_numeric(avisos["latitud"], errors="coerce")
avisos["longitud"] = pd.to_numeric(avisos["longitud"], errors="coerce")
avisos = avisos.dropna(subset=["latitud", "longitud"])

print(f"\nAvisos con coordenadas válidas en la BD: {len(avisos)}")

# ------------------------------------------------------------------
# Paso 4: Convertir a GeoDataFrame de puntos e igualar CRS
# ------------------------------------------------------------------
geometria_puntos = [Point(lon, lat) for lat, lon in zip(avisos["latitud"], avisos["longitud"])]
avisos_geo = gpd.GeoDataFrame(avisos, geometry=geometria_puntos, crs="EPSG:4326")

if uv_filtrado.crs != avisos_geo.crs:
    print(f"Convirtiendo el shapefile de {uv_filtrado.crs} a {avisos_geo.crs}...")
    uv_filtrado = uv_filtrado.to_crs(avisos_geo.crs)

# ------------------------------------------------------------------
# Paso 5: Cruce espacial (point-in-polygon), solo contra las UV filtradas
# ------------------------------------------------------------------
resultado = gpd.sjoin(avisos_geo, uv_filtrado, how="left", predicate="within")

n_asignados = resultado["index_right"].notna().sum()
n_sin_asignar = resultado["index_right"].isna().sum()
print(f"\nAsignados a una Unidad Vecinal (de las comunas analizadas): {n_asignados}")
print(f"Sin Unidad Vecinal encontrada: {n_sin_asignar}")

# ------------------------------------------------------------------
# Paso 6a: Guardar la tabla de referencia `vulnerabilidad_uv` - SOLO datos
# de las comunas analizadas, con uv_rsh como llave primaria
# ------------------------------------------------------------------
columnas_vulnerabilidad = [
    "uv_rsh", "cod_com", "Comuna", "comuna_slug", "rank_nac",
    "c_ig_com", "c_ig_reg", "c_ig_nac", "pob_rsh_uv", "hog_uv", "p_urbano",
]
columnas_disponibles_vuln = [c for c in columnas_vulnerabilidad if c in uv_filtrado.columns]

tabla_vulnerabilidad = uv_filtrado[columnas_disponibles_vuln].copy()
tabla_vulnerabilidad.to_sql("vulnerabilidad_uv", con, if_exists="replace", index=False)
con.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_vulnerabilidad_uv_rsh ON vulnerabilidad_uv(uv_rsh)")

# ------------------------------------------------------------------
# Paso 6b: Guardar la llave de cruce `avisos_igvust` (id_aviso -> uv_rsh)
# ------------------------------------------------------------------
tabla_cruce = resultado[["id_aviso", "uv_rsh"]].copy()
tabla_cruce.to_sql("avisos_igvust", con, if_exists="replace", index=False)
con.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_avisos_igvust_id ON avisos_igvust(id_aviso)")

con.commit()
con.close()

print(f"\nGuardado 'vulnerabilidad_uv' ({len(tabla_vulnerabilidad)} filas) "
      f"y 'avisos_igvust' ({len(tabla_cruce)} filas) en {RUTA_BD.name}")
print("\nEjemplo de vulnerabilidad_uv:")
print(tabla_vulnerabilidad.head())
print("\nEjemplo de avisos_igvust:")
print(tabla_cruce.head())