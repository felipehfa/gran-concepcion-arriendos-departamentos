"""
Pipeline de ingeniería de variables para los avisos de arriendo del Gran Concepción.

Lee las tablas crudas de avisos, limpia valores atípicos y faltantes, deriva
variables (nivel de precio del barrio, precio/m2 del sector, antigüedad
imputada por cercanía geográfica, variables dummy, etc.) y deja el resultado
listo para modelar en un CSV.

Lee de la base de PRODUCCIÓN (Postgres/Supabase). Hasta 2026-07-29 leía de una
BD SQLite local de investigación, que se retiró porque los datos actualizados ya
viven en producción; por eso toda función que toque la BD recibe una conexión
`con` ya abierta y no la cierra (el caller es dueño de su ciclo de vida).

Qué avisos entran al dataset se controla con `estados` /
`meses_max_antiguedad_finalizados` (ver ESTADOS_ENTRENAMIENTO_DEFAULT).

Pensado para ser importado desde un script orquestador externo:

    from ingenieria_variables import ejecutar_pipeline
    df_final = ejecutar_pipeline(con=mi_conexion_postgres)
"""

import re
import time
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import numpy as np
import requests
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.neighbors import BallTree


# ------------------------------------------------------------------
# Constantes de configuración
# ------------------------------------------------------------------
# Todas las rutas por defecto se anclan al directorio de este archivo (no al
# directorio de trabajo actual), para que el script funcione igual sin
# importar desde dónde se invoque o desde dónde lo importe un orquestador.
DIRECTORIO_SCRIPT = Path(__file__).resolve().parent
DIRECTORIO_SALIDA = DIRECTORIO_SCRIPT / "save" / "ingeniaria_variables"

RUTA_SALIDA_CSV_DEFAULT = str(DIRECTORIO_SALIDA / "datos_ingenieria_variables.csv")
RUTA_SALIDA_NIVELES_BARRIO_DEFAULT = str(DIRECTORIO_SALIDA / "niveles_barrio.json")

API_UF_URL = "https://mindicador.cl/api/uf/{fecha}"
UF_API_MAX_WORKERS = 8  # fechas faltantes en paralelo al arrancar con la BD fría
TASA_USD_CLP = 930  # tasa fija: solo hay un puñado de avisos en US$, no justifica ir a buscar el valor por fecha a una API
PRECIO_MAXIMO_ARRIENDO_CLP = 8_000_000  # sobre este monto ya no es un arriendo real (venta mal clasificada o dato corrupto)

RADIO_TIERRA_M = 6_371_000

# Caja de sanidad geográfica del Gran Concepción, generosa a propósito (la
# región real cabe holgadamente: lat observada -37.01..-36.52, lon -73.19..-73.16).
# Un aviso fuera de acá no es un outlier de magnitud sino una coordenada
# equivocada por quien publicó, y sus variables de ubicación (distancias al
# centro, POIs cercanos, unidad vecinal) quedan sin sentido - así que la fila
# se descarta completa en vez de imputarse.
# Caso real que motivó el filtro: MLC-4044413466 traía lat 40.4953/lon -3.8785
# (Madrid, España), lo que producía distancia_centro_concepcion_m = 11.114.141 m
# (~11.114 km, tres órdenes de magnitud sobre el resto) y entraba tal cual como
# feature del modelo. Estaba también en la BD histórica de investigación, así
# que venía contaminando el entrenamiento desde antes.
CAJA_LATITUD_GRAN_CONCEPCION = (-37.5, -36.0)
CAJA_LONGITUD_GRAN_CONCEPCION = (-73.7, -72.5)

N_CATEGORIAS = 5          # muy barato, barato, medio, alto, muy alto
K_SUAVIZADO = 20          # "avisos virtuales" del promedio general que se mezclan
                          # con el promedio de cada barrio (barrios chicos se
                          # acercan más al promedio general; barrios grandes casi no se ajustan)
NOMBRES_NIVELES = ["muy_barato", "barato", "medio", "alto", "muy_alto"]

UMBRAL_MINIMO_M2 = 30
COLUMNAS_PREDICTORAS_SUPERFICIE = ["dormitorios", "banos", "comuna"]
MINIMO_FILAS_PARA_ENTRENAR = 20
RUTA_MODELOS = DIRECTORIO_SALIDA / "modelos_superficie"

# El pipeline solo trabaja con departamentos: la base de datos ya no incluye
# avisos de casa, por lo que no se generan variables derivadas de ese tipo.
TIPO_PROPIEDAD_OBJETIVO = "departamento"

# ------------------------------------------------------------------
# Población de entrenamiento (solo aplica a la ruta Postgres/producción)
#
# Qué avisos entran al dataset. Se exponen como parámetros de `cargar_datos` /
# `ejecutar_pipeline` -y no hardcodeados en la consulta- para poder comparar
# configuraciones sin editar código.
#
# Por qué NO se entrena solo con los 'activo': un aviso que sigue activo es,
# por definición, uno que el mercado todavía no absorbió, así que entrenar solo
# con esos sobre-representa unidades caras para su segmento (sesgo de
# supervivencia, en la dirección equivocada). Los 'finalizado' son
# observaciones válidas -y son justamente las que sí se arrendaron-, por eso
# entran. Medido sobre la BD de producción: solo activos son 1414 filas (menos
# que las 1627 del CSV histórico), mientras que esta configuración da ~2453.
ESTADOS_ENTRENAMIENTO_DEFAULT = ("activo", "pausado", "finalizado")

# El límite de antigüedad se aplica solo a los estados terminales: un aviso
# activo sigue vigente hoy sin importar cuándo se publicó, mientras que uno
# finalizado hace un año refleja un precio que ya no es de mercado.
ESTADOS_CON_LIMITE_ANTIGUEDAD = ("finalizado",)
MESES_MAX_ANTIGUEDAD_FINALIZADOS_DEFAULT = 3

# Se filtra por fecha_publicacion_aprox y NO por fecha_ultimo_chequeo_estado:
# esa última marca cuándo el pipeline DETECTÓ el cierre, no cuándo ocurrió, así
# que en la práctica no discrimina nada (los 1118 finalizados de la BD caen
# dentro de cualquier ventana, porque el re-chequeo empezó recién). Tampoco
# first_seen, que solo tiene 3 semanas de historia. fecha_publicacion_aprox es
# la edad real del precio observado y tiene un año de rango.
COLUMNA_ANTIGUEDAD_AVISO = "fecha_publicacion_aprox"
DIAS_POR_MES = 30

RADIO_METROS_COMPARADOR_SECTOR = 300
TIPOS_PROPIEDAD_COLUMNAS = {
    "departamento": "precio_m2_sector_departamento",
}
MULTIPLICADOR_IQR = 3

RADIO_METROS_ANTIGUEDAD = 200

COMUNAS_A_AGRUPAR = [
    "talcahuano-biobio",
    "chiguayante-biobio",
    "hualpen-biobio",
    "coronel-biobio",
    "tome-biobio",
    "penco-biobio",
    "hualqui-biobio",
]

PATRON_AMOBLADO = re.compile(r"amoblad[oa]|amueblad[oa]", re.IGNORECASE)


# ------------------------------------------------------------------
# Carga y diagnóstico inicial
# ------------------------------------------------------------------
def cargar_datos(
    con,
    tipo_propiedad: str = TIPO_PROPIEDAD_OBJETIVO,
    estados=ESTADOS_ENTRENAMIENTO_DEFAULT,
    meses_max_antiguedad_finalizados=MESES_MAX_ANTIGUEDAD_FINALIZADOS_DEFAULT,
):
    """Lee de la base de PRODUCCIÓN (Postgres/Supabase) las tablas crudas
    necesarias para construir el dataset: `avisos` + `avisos_detalle`.

    `con` es una conexión ya abierta y no se cierra acá (el caller es dueño de
    su ciclo de vida), igual que en `convertir_precios_uf_a_clp`.

    Devuelve 4 elementos por compatibilidad con `combinar_tablas`, pero los dos
    últimos son siempre None: la vulnerabilidad socioterritorial ya viene
    DESNORMALIZADA dentro de `avisos_detalle` (la escribe
    03_vulnerabilidad_produccion.py), así que no hay un join por uv_rsh contra
    tablas aparte que hacer. Respecto del viejo esquema SQLite faltan `c_ig_reg`
    y `c_ig_nac`, pero ninguna de las dos sobrevive a selected_features.csv, y
    las funciones que las tocan ya omiten columnas ausentes.

    El INNER JOIN con `avisos_detalle` cumple dos funciones: permite filtrar por
    la fecha de publicación (que vive en esa tabla) y descarta de entrada los
    avisos que todavía no tienen detalle scrapeado, que no servirían para
    entrenar.
    """
    condiciones = ["a.tipo_propiedad = %s", "a.estado_publicacion = ANY(%s)"]
    parametros = [tipo_propiedad, list(estados)]

    if meses_max_antiguedad_finalizados is not None:
        fecha_corte = (
            pd.Timestamp.today().normalize()
            - pd.Timedelta(days=DIAS_POR_MES * meses_max_antiguedad_finalizados)
        ).date().isoformat()
        # Los estados sin límite de antigüedad pasan siempre; los terminales
        # solo si su fecha de publicación es posterior al corte. La columna es
        # TEXT en formato ISO (YYYY-MM-DD), así que la comparación
        # lexicográfica coincide con la cronológica. Una fecha NULL no cumple
        # la condición y queda fuera, que es el comportamiento conservador
        # deseado (no sabemos qué tan viejo es ese precio).
        condiciones.append(
            f"(a.estado_publicacion <> ALL(%s) OR d.{COLUMNA_ANTIGUEDAD_AVISO} >= %s)"
        )
        parametros.extend([list(ESTADOS_CON_LIMITE_ANTIGUEDAD), fecha_corte])

    df_avisos = pd.read_sql_query(
        f"""
        SELECT a.*
        FROM avisos a
        INNER JOIN avisos_detalle d ON a.id_aviso = d.id_aviso
        WHERE {' AND '.join(condiciones)}
        """,
        con,
        params=parametros,
    )
    # Se lee `avisos_detalle` completa y no restringida a los ids de arriba: el
    # merge de `combinar_tablas` es left sobre df_avisos, así que las filas de
    # detalle que sobren se descartan solas, y evitar un IN con miles de ids
    # mantiene la consulta simple.
    df_detalle = pd.read_sql_query("SELECT * FROM avisos_detalle", con)

    return df_avisos, df_detalle, None, None


def seleccionar_columnas_relevantes(df_avisos: pd.DataFrame, df_detalle: pd.DataFrame):
    """Se queda solo con las columnas de `avisos` y `avisos_detalle` que
    efectivamente se usan más adelante en el pipeline."""
    df_avisos = df_avisos[
        ["id_aviso", "titulo", "comuna", "tipo_propiedad", "precio", "moneda", "banos", "superficie_m2"]
    ]

    columnas_detalle = [
        "id_aviso", "estacionamientos", "dormitorios", "antiguedad_anos", "condominio_cerrado",
        "distancia_centro_comuna_m", "distancia_centro_concepcion_m", "cantidad_paraderos",
        "cantidad_jardines_infantiles", "cantidad_colegios", "cantidad_universidades",
        "cantidad_plazas", "cantidad_supermercados", "cantidad_farmacias",
        "cantidad_centros_comerciales", "cantidad_hospitales", "cantidad_clinicas",
        "cantidad_estaciones_metro", "latitud", "longitud", "fecha_publicacion_aprox",
        "superficie_total_m2", "superficie_util_m2",
        "bodegas", "gastos_comunes", "estacionamiento_visitas",
        "piscina", "quincho", "conserjeria", "ascensor", "barrio",
        "piso_unidad",
    ]

    # En la ruta Postgres las columnas de vulnerabilidad ya vienen dentro de
    # `avisos_detalle` y hay que conservarlas acá o se perderían antes del
    # merge; en la ruta SQLite no están en esta tabla (llegan después, del join
    # con vulnerabilidad_uv que hace `combinar_tablas`), así que la lista queda
    # igual que antes. De ahí el filtro por columnas presentes.
    columnas_detalle += [c for c in COLUMNAS_VULNERABILIDAD if c in df_detalle.columns]
    if "uv_rsh" in df_detalle.columns:
        columnas_detalle.append("uv_rsh")

    df_detalle = df_detalle[columnas_detalle]

    return df_avisos, df_detalle


def combinar_tablas(
    df_avisos: pd.DataFrame,
    df_detalle: pd.DataFrame,
    df_llave_vulnerabilidad: pd.DataFrame,
    df_vulnerabilidad: pd.DataFrame,
) -> pd.DataFrame:
    """Une avisos + detalle + vulnerabilidad socioterritorial en un único
    DataFrame de trabajo, descartando las columnas llave/auxiliares que ya
    cumplieron su propósito de unión.

    En la ruta Postgres/producción los dos DataFrames de vulnerabilidad llegan
    en None: ahí esas columnas ya vienen desnormalizadas dentro de
    `avisos_detalle`, así que no hay nada que unir por uv_rsh - y volver a
    mergearlas duplicaría cada indicador (rank_nac_x / rank_nac_y).
    """
    df = pd.merge(df_avisos, df_detalle, on="id_aviso", how="left")
    if df_llave_vulnerabilidad is not None:
        df = pd.merge(df, df_llave_vulnerabilidad, on="id_aviso", how="left")
    if df_vulnerabilidad is not None:
        df = pd.merge(df, df_vulnerabilidad, on="uv_rsh", how="left")
    # errors="ignore": cod_com/Comuna/comuna_slug son columnas auxiliares del
    # esquema SQLite de investigación que el de producción no trae.
    df = df.drop(columns=["uv_rsh", "cod_com", "Comuna", "comuna_slug"], errors="ignore")
    return df


# ------------------------------------------------------------------
# Limpieza preliminar
# ------------------------------------------------------------------
def eliminar_outliers_habitaciones(df: pd.DataFrame, max_dormitorios: int = 6,
                                    max_banos: int = 5, max_estacionamientos: int = 15) -> pd.DataFrame:
    """Descarta avisos con valores extremos de dormitorios, baños o
    estacionamientos (probablemente errores de digitación o propiedades que
    no son departamentos/casas individuales)."""
    columnas = ["dormitorios", "banos", "estacionamientos"]
    df[columnas] = df[columnas].apply(pd.to_numeric, errors="coerce")
    df = df[
        (df["dormitorios"] <= max_dormitorios)
        & (df["banos"] <= max_banos)
        & (df["estacionamientos"] <= max_estacionamientos)
    ].reset_index(drop=True)
    return df


def eliminar_registros_incompletos(df: pd.DataFrame) -> pd.DataFrame:
    """Descarta avisos sin datos básicos de dormitorios, baños o superficie."""
    return df.dropna(subset=["dormitorios", "banos", "superficie_m2"])


def descartar_coordenadas_fuera_de_region(
    df: pd.DataFrame,
    caja_latitud: tuple = CAJA_LATITUD_GRAN_CONCEPCION,
    caja_longitud: tuple = CAJA_LONGITUD_GRAN_CONCEPCION,
) -> pd.DataFrame:
    """Descarta avisos cuyas coordenadas caen fuera del Gran Concepción (ver
    CAJA_LATITUD_GRAN_CONCEPCION). Las filas SIN coordenadas se conservan: son
    un faltante legítimo que el pipeline imputa más adelante, no un dato
    erróneo.

    Las coordenadas se castean a numérico solo para evaluar el filtro (sin
    tocar el dtype del df): desde SQLite llegan como TEXT y la comparación
    directa contra los límites tiraría TypeError, mientras que desde Postgres
    ya vienen como DOUBLE PRECISION. Un valor no parseable queda NaN y por lo
    tanto se trata como "sin coordenadas", que es el criterio conservador."""
    lat = pd.to_numeric(df["latitud"], errors="coerce")
    lon = pd.to_numeric(df["longitud"], errors="coerce")
    sin_coordenadas = lat.isna() | lon.isna()
    dentro = lat.between(*caja_latitud) & lon.between(*caja_longitud)

    fuera = (~sin_coordenadas) & (~dentro)
    if fuera.any():
        for _, fila in df.loc[fuera, ["id_aviso", "latitud", "longitud"]].iterrows():
            print(f"Advertencia: {fila['id_aviso']} tiene coordenadas fuera del Gran Concepción "
                  f"(lat={fila['latitud']}, lon={fila['longitud']}) - se descarta el aviso.")

    return df[sin_coordenadas | dentro].reset_index(drop=True)


def convertir_superficie_total_a_numerico(df: pd.DataFrame) -> pd.DataFrame:
    """Castea 'superficie_total_m2' a numérico (llega como texto desde SQLite)."""
    df["superficie_total_m2"] = pd.to_numeric(df["superficie_total_m2"], errors="coerce")
    return df



COLUMNAS_VULNERABILIDAD = [
    "rank_nac", "c_ig_com", "c_ig_reg", "c_ig_nac", "pob_rsh_uv", "hog_uv", "p_urbano",
]
 

# ------------------------------------------------------------------
# Imputar datos de vulnerabilidad
# ------------------------------------------------------------------
 
def imputar_vulnerabilidad_por_comuna(df: pd.DataFrame,
                                       columnas: list = COLUMNAS_VULNERABILIDAD,
                                       columna_comuna: str = "comuna") -> pd.DataFrame:
    """
    Para cada columna de vulnerabilidad, rellena los NaN con el promedio de
    esa columna dentro de la misma comuna (calculado solo con los valores
    reales disponibles, sin contar los que ya son NaN).
 
    Si una comuna entera no tiene NINGÚN dato de vulnerabilidad (las
    coordenadas del aviso no cayeron en ninguna Unidad Vecinal mapeada),
    cae de respaldo al promedio general de todo el dataset para esa columna.
    """
    df = df.copy()
 
    for columna in columnas:
        if columna not in df.columns:
            print(f"Advertencia: la columna '{columna}' no existe en el df, se omite.")
            continue
 
        nulos_antes = df[columna].isna().sum()
 
        promedio_por_comuna = df.groupby(columna_comuna)[columna].transform("mean")
        df[columna] = df[columna].fillna(promedio_por_comuna)
 
        # Respaldo: si toda la comuna estaba sin datos, promedio_por_comuna
        # también sale NaN para esas filas - ahí cae al promedio global.
        promedio_global = df[columna].mean()
        df[columna] = df[columna].fillna(promedio_global)
 
        nulos_despues = df[columna].isna().sum()
        print(f"{columna}: {nulos_antes} nulos -> {nulos_despues} nulos "
              f"({nulos_antes - nulos_despues} imputados con el promedio de su comuna)")
 
    return df
 

# ------------------------------------------------------------------
# Imputación de barrio por cercanía geográfica
# ------------------------------------------------------------------
def fill_barrio_nan(df: pd.DataFrame, lat_col: str = "latitud", lon_col: str = "longitud",
                     barrio_col: str = "barrio") -> pd.DataFrame:
    """
    Rellena los valores NaN de `barrio_col` usando el barrio de la vivienda
    más cercana (según lat/long) que sí tenga barrio conocido.
    """
    df = df.copy()

    for col in [lat_col, lon_col]:
        if df[col].dtype == object:
            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
                .str.replace(',', '.', regex=False)  # por si viene con coma decimal
            )
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Si algún lat/long quedó NaN tras la conversión, no podemos usarlo ni como
    # referencia ni como destino de la búsqueda
    coords_validas = df[lat_col].notna() & df[lon_col].notna()

    mask_conocido = df[barrio_col].notna() & coords_validas
    mask_nan = df[barrio_col].isna() & coords_validas

    if mask_nan.sum() == 0:
        return df
    if mask_conocido.sum() == 0:
        raise ValueError("No hay registros con barrio conocido y coordenadas válidas para usar como referencia.")

    # BallTree con haversine necesita las coordenadas en radianes
    coords_conocidas = np.radians(df.loc[mask_conocido, [lat_col, lon_col]].values.astype(float))
    coords_nan = np.radians(df.loc[mask_nan, [lat_col, lon_col]].values.astype(float))

    tree = BallTree(coords_conocidas, metric='haversine')
    dist, idx = tree.query(coords_nan, k=1)  # vecino más cercano

    barrios_conocidos = df.loc[mask_conocido, barrio_col].values
    barrios_asignados = barrios_conocidos[idx.flatten()]

    df.loc[mask_nan, barrio_col] = barrios_asignados

    # Distancia en metros al vecino usado, útil para revisar la calidad del match
    df.loc[mask_nan, 'barrio_distancia_m'] = dist.flatten() * RADIO_TIERRA_M
    df.drop('barrio_distancia_m', axis=1, inplace=True)

    return df


# ------------------------------------------------------------------
# Variables internas de la página (amenities, piso, etc.)
# ------------------------------------------------------------------
def preprocesar_variables_amenities(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza las variables de amenities: convierte los "Sí"/NaN de las
    columnas booleanas a 1/0, castea a numérico, rellena con 0 los amenities
    sin dato (ausencia = no tiene) y asume 1er piso cuando falta piso_unidad."""
    columnas_bool = ["estacionamiento_visitas", "piscina", "quincho", "conserjeria", "ascensor"]
    df[columnas_bool] = df[columnas_bool].fillna(0)
    df[columnas_bool] = df[columnas_bool].replace({"Sí": 1})

    columnas_numericas = [
        "bodegas", "gastos_comunes", "estacionamiento_visitas",
        "piscina", "quincho", "conserjeria", "ascensor", "piso_unidad",
    ]
    df[columnas_numericas] = df[columnas_numericas].apply(pd.to_numeric, errors="coerce")

    amenities = ["piscina", "quincho", "conserjeria", "ascensor", "bodegas", "gastos_comunes"]
    df[amenities] = df[amenities].fillna(0)

    df["piso_unidad"] = df["piso_unidad"].fillna(1)
    return df


def corregir_piso_unidad_outliers(df: pd.DataFrame, umbral: int = 30) -> pd.DataFrame:
    """Reemplaza pisos > `umbral` (probablemente errores de digitación, p.ej.
    el año de construcción cargado en el campo de piso) por el promedio de
    los pisos válidos."""
    promedio_piso = df.loc[df["piso_unidad"] <= umbral, "piso_unidad"].mean()
    df.loc[df["piso_unidad"] > umbral, "piso_unidad"] = promedio_piso
    return df


def rellenar_cantidad_pois_cercanos(df: pd.DataFrame) -> pd.DataFrame:
    """Castea a numérico y rellena con 0 las columnas de cantidad de puntos
    de interés cercanos (paraderos, colegios, plazas, etc.): la ausencia de
    dato equivale a "no hay ninguno cerca"."""
    columnas_cantidad = [
        "cantidad_paraderos", "cantidad_jardines_infantiles", "cantidad_colegios",
        "cantidad_universidades", "cantidad_plazas", "cantidad_supermercados",
        "cantidad_farmacias", "cantidad_centros_comerciales", "cantidad_hospitales",
        "cantidad_clinicas", "cantidad_estaciones_metro", "estacionamientos",
    ]
    df[columnas_cantidad] = df[columnas_cantidad].apply(pd.to_numeric, errors="coerce")
    df[columnas_cantidad] = df[columnas_cantidad].fillna(0)
    return df


# ------------------------------------------------------------------
# Conversión de precios en UF a CLP (con caché en la propia BD)
#
# Estas funciones reciben la conexión (`con`) desde afuera: las usan tanto el
# reentrenamiento como el pipeline de producción y el dashboard
# (05_prediccion.py y data.py del visualizador), cada uno dueño del ciclo de
# vida de su propia conexión. `_es_postgres` se conserva porque el dashboard
# corre con un rol de privilegios reducidos y hay que distinguir ese caso al
# crear la tabla de caché (ver inicializar_tabla_uf).
# ------------------------------------------------------------------
def _es_postgres(con) -> bool:
    return con.__class__.__module__.startswith("psycopg2")


def inicializar_tabla_uf(con) -> None:
    """Crea (si no existe) la tabla que cachea el valor de la UF por fecha.

    En Postgres/producción, `db.py` ya crea esta tabla con el rol `postgres`
    (dueño del schema) antes de que el dashboard llegue a este punto; el rol
    `streamlit_app` del dashboard solo tiene SELECT/INSERT/UPDATE sobre ella
    (ver SETUP.md), sin privilegio CREATE sobre el schema. Si por algún motivo
    la tabla no existiera todavía (orquestador que aún no corrió una primera
    vez), el CREATE TABLE de acá fallaría con InsufficientPrivilege para ese
    rol - se ignora ese error puntual porque no es responsabilidad del
    dashboard crear tablas, solo usarlas.
    """
    es_postgres = _es_postgres(con)
    tipo_valor = "DOUBLE PRECISION" if es_postgres else "REAL"
    try:
        con.cursor().execute(f"""
            CREATE TABLE IF NOT EXISTS valores_uf (
                fecha TEXT PRIMARY KEY,   -- formato YYYY-MM-DD
                valor {tipo_valor}
            )
        """)
        con.commit()
    except Exception as exc:
        # Import local (no hay psycopg2 en modelamiento/requirements.txt -
        # este módulo también corre con solo SQLite): si `con` es de Postgres,
        # ya se importó al abrir esa conexión, así que este import es gratis.
        if es_postgres:
            import psycopg2.errors
            if isinstance(exc, psycopg2.errors.InsufficientPrivilege):
                con.rollback()
                return
        raise


def obtener_valores_uf_desde_bd(con, fechas: list) -> dict:
    """Devuelve {Timestamp: valor} solo para las fechas que YA están cacheadas en la BD."""
    if not fechas:
        return {}
    marcador = "%s" if _es_postgres(con) else "?"
    fechas_str = [f.strftime("%Y-%m-%d") for f in fechas]
    placeholders = ",".join(marcador for _ in fechas_str)
    cur = con.cursor()
    cur.execute(f"SELECT fecha, valor FROM valores_uf WHERE fecha IN ({placeholders})", fechas_str)
    return {pd.Timestamp(fecha): valor for fecha, valor in cur.fetchall()}


def guardar_valor_uf_en_bd(con, fecha: pd.Timestamp, valor: float) -> None:
    """Guarda (o reemplaza) el valor de la UF de una fecha, con commit inmediato
    para no perder la consulta si la corrida se interrumpe a mitad de camino."""
    if _es_postgres(con):
        sql = ("INSERT INTO valores_uf (fecha, valor) VALUES (%s, %s) "
               "ON CONFLICT (fecha) DO UPDATE SET valor = excluded.valor")
    else:
        sql = "INSERT OR REPLACE INTO valores_uf (fecha, valor) VALUES (?, ?)"
    con.cursor().execute(sql, (fecha.strftime("%Y-%m-%d"), valor))
    con.commit()


def obtener_valor_uf_api(fecha: pd.Timestamp, reintentos: int = 3, espera: float = 1.0):
    """Consulta mindicador.cl. Devuelve None si no logra obtenerlo tras los reintentos."""
    fecha_str = fecha.strftime("%d-%m-%Y")
    url = API_UF_URL.format(fecha=fecha_str)

    for _ in range(reintentos):
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                datos = resp.json()
                serie = datos.get("serie", [])
                if serie:
                    return serie[0]["valor"]
            return None
        except requests.RequestException:
            time.sleep(espera)

    return None


def convertir_precios_uf_a_clp(df: pd.DataFrame, con=None) -> pd.DataFrame:
    """
    Crea la columna 'precio_clp': para avisos publicados en CLP la copia tal
    cual, para los publicados en UF la convierte usando el valor de la UF de
    su fecha de publicación (consultando mindicador.cl, con caché en la
    propia base de datos para no repetir consultas entre corridas; si a un
    aviso en UF le falta la fecha de publicación, usa como respaldo el
    promedio de fechas de todo el dataset), y para los publicados en US$ usa
    la tasa fija TASA_USD_CLP.

    `con` es una conexión ya abierta (Postgres) y no se cierra acá: el caller
    es dueño de su ciclo de vida. La pasan `ejecutar_pipeline`, 05_prediccion.py
    y data.py del visualizador.
    """
    # Paso 0: asegurar que fecha_publicacion_aprox sea datetime
    df["fecha_publicacion_aprox"] = pd.to_datetime(df["fecha_publicacion_aprox"], errors="coerce")

    # Paso 1: fecha de respaldo = promedio de todas las fecha_publicacion_aprox
    fecha_promedio = df["fecha_publicacion_aprox"].mean().normalize()

    # Paso 2: para cada fila en UF, determinar qué fecha usar
    mask_uf = df["moneda"] == "UF"
    fechas_a_usar = df.loc[mask_uf, "fecha_publicacion_aprox"].dt.normalize().fillna(fecha_promedio)

    fechas_unicas = [pd.Timestamp(f) for f in fechas_a_usar.unique()]
    fechas_a_verificar = list(set(fechas_unicas) | {fecha_promedio})  # incluye la de respaldo por si se necesita

    # Paso 3: revisar primero la caché en la BD; solo consultar la API las fechas que falten ahí
    if con is None:
        raise ValueError(
            "convertir_precios_uf_a_clp necesita una conexión `con` (Postgres). "
            "La BD SQLite de investigación ya no existe."
        )
    inicializar_tabla_uf(con)

    valor_uf_por_fecha = obtener_valores_uf_desde_bd(con, fechas_a_verificar)

    fechas_faltantes = [f for f in fechas_a_verificar if f not in valor_uf_por_fecha]

    # Fetch en paralelo (I/O-bound): en la primera corrida con la BD fría
    # esto puede ser una docena de fechas distintas, y consultarlas de a una
    # con sleep(0.3) entre cada una hacía que la carga inicial se sintiera
    # colgada. Los requests corren en threads; la escritura a sqlite se hace
    # en el hilo principal a medida que van llegando (la conexión no es
    # thread-safe para usarse desde varios threads a la vez).
    if fechas_faltantes:
        with ThreadPoolExecutor(max_workers=UF_API_MAX_WORKERS) as executor:
            futuros = {executor.submit(obtener_valor_uf_api, fecha): fecha for fecha in fechas_faltantes}
            for futuro in as_completed(futuros):
                fecha = futuros[futuro]
                valor = futuro.result()
                if valor is not None:
                    guardar_valor_uf_en_bd(con, fecha, valor)  # se guarda de inmediato para la próxima corrida
                    valor_uf_por_fecha[fecha] = valor

    # Paso 4: convertir a CLP (usa el valor de la fecha propia; si esa fecha
    # específica no se pudo obtener, cae al valor ya resuelto de fecha_promedio)
    df["precio_clp"] = df["precio"].astype(float)

    valores_uf_resueltos = fechas_a_usar.map(valor_uf_por_fecha)
    valor_uf_fallback = valor_uf_por_fecha.get(fecha_promedio)
    if valor_uf_fallback is not None:
        valores_uf_resueltos = valores_uf_resueltos.fillna(valor_uf_fallback)

    df.loc[mask_uf, "precio_clp"] = df.loc[mask_uf, "precio"] * valores_uf_resueltos

    # Paso 5: US$ -> CLP con tasa fija (no fluctúa por fecha como la UF, no
    # amerita el mismo mecanismo de caché/API)
    mask_usd = df["moneda"] == "US$"
    df.loc[mask_usd, "precio_clp"] = df.loc[mask_usd, "precio"] * TASA_USD_CLP

    return df


def filtrar_precio_maximo(df: pd.DataFrame, precio_maximo: float = PRECIO_MAXIMO_ARRIENDO_CLP) -> pd.DataFrame:
    """Descarta avisos con precio_clp por sobre `precio_maximo`: a ese nivel
    ya no son arriendos sino ventas mal clasificadas. Requiere que
    'precio_clp' ya haya sido calculada (ver `convertir_precios_uf_a_clp`).
    El filtro se aplica sobre el arriendo nominal (precio_clp), no sobre el
    costo total: lo que detecta es una venta mal clasificada como arriendo,
    algo que depende del precio pactado, no de los gastos comunes."""
    df = df[df["precio_clp"] <= precio_maximo]
    return df


def agregar_costo_total(df: pd.DataFrame) -> pd.DataFrame:
    """Crea 'costo_total_clp' = precio_clp + gastos_comunes: el costo mensual
    real de vivir en la propiedad (arriendo + gastos comunes), que es la
    variable objetivo del modelo. Requiere que 'precio_clp' (convertir_precios_uf_a_clp)
    y 'gastos_comunes' (ya limpia/numérica vía preprocesar_variables_amenities)
    existan de antemano. 'precio_clp' se conserva en el dataset como columna
    informativa, no como feature ni target."""
    df["costo_total_clp"] = df["precio_clp"] + df["gastos_comunes"]
    return df


# ------------------------------------------------------------------
# Nivel de precio del barrio (precio/m2 suavizado, agrupado en niveles)
# ------------------------------------------------------------------
def calcular_niveles_barrio(datos: pd.DataFrame, columna_precio_m2: str,
                             columna_barrio: str, n_categorias: int = N_CATEGORIAS,
                             k: float = K_SUAVIZADO):
    """
    Calcula el precio/m2 promedio SUAVIZADO de cada barrio (usando TODOS los
    datos disponibles) y los agrupa en `n_categorias` niveles ordinales,
    usando cuantiles PONDERADOS POR CANTIDAD DE AVISOS (no por cantidad de
    barrios) - así "alto" representa realmente ~20% de las propiedades más
    caras, no ~20% de los barrios sin importar cuántos avisos tenga cada uno.

    Devuelve:
      - mapa_barrio_a_nivel: dict {barrio: nivel_ordinal (0..n_categorias-1)}
      - nivel_default: nivel a usar para barrios nuevos que no existían al
        momento de construir este diccionario (útil para avisos futuros)
      - cortes_valor: los puntos de corte usados (para inspección/auditoría)
      - stats: tabla completa con el detalle de cada barrio (promedio crudo,
        cantidad de avisos, promedio ajustado, y nivel asignado)
    """
    promedio_general = datos[columna_precio_m2].mean()

    stats = datos.groupby(columna_barrio)[columna_precio_m2].agg(["mean", "count"])

    stats["promedio_ajustado"] = (
        (stats["count"] * stats["mean"] + k * promedio_general) / (stats["count"] + k)
    )

    stats_ordenado = stats.sort_values("promedio_ajustado")
    peso_acumulado = stats_ordenado["count"].cumsum() / stats_ordenado["count"].sum()

    limites_percentiles = np.linspace(0, 1, n_categorias + 1)[1:-1]
    cortes_valor = list(np.interp(
        limites_percentiles,
        peso_acumulado.values,
        stats_ordenado["promedio_ajustado"].values,
    ))

    def _clasificar(valor_ajustado):
        nivel = 0
        for corte in cortes_valor:
            if valor_ajustado > corte:
                nivel += 1
        return nivel

    stats["nivel"] = stats["promedio_ajustado"].apply(_clasificar)
    stats["nivel_nombre"] = stats["nivel"].map(lambda n: NOMBRES_NIVELES[n])

    mapa_barrio_a_nivel = stats["nivel"].to_dict()
    nivel_default = _clasificar(promedio_general)

    return mapa_barrio_a_nivel, nivel_default, cortes_valor, stats


def aplicar_niveles_barrio(datos: pd.DataFrame, columna_barrio: str,
                            mapa_barrio_a_nivel: dict, nivel_default: int) -> pd.Series:
    """Aplica el diccionario ya calculado a cualquier dataframe (actual o futuro),
    asignando `nivel_default` a cualquier barrio que no esté en el diccionario."""
    return (
        datos[columna_barrio]
        .map(mapa_barrio_a_nivel)
        .fillna(nivel_default)
        .astype(int)
    )


def agregar_nivel_barrio(df: pd.DataFrame, ruta_salida_json: str = RUTA_SALIDA_NIVELES_BARRIO_DEFAULT) -> pd.DataFrame:
    """
    Construye el diccionario barrio -> nivel de precio usando TODOS los datos
    disponibles, lo guarda en `ruta_salida_json` (para reutilizarlo con avisos
    futuros sin tener que recalcularlo) y agrega la columna 'nivel_barrio' al
    DataFrame, reemplazando a la columna 'barrio' original.

    Se basa en costo_total_clp (arriendo + gastos comunes), no en precio_clp:
    "qué tan caro es el barrio" debe reflejar el costo mensual real, no solo
    el arriendo nominal.
    """
    columnas_numericas = ["costo_total_clp", "superficie_util_m2"]
    df[columnas_numericas] = df[columnas_numericas].apply(pd.to_numeric, errors="coerce")
    df["_precio_m2"] = df["costo_total_clp"] / df["superficie_util_m2"].replace(0, np.nan)

    mapa_barrio_a_nivel, nivel_default, cortes_valor, stats_barrios = calcular_niveles_barrio(
        df, columna_precio_m2="_precio_m2", columna_barrio="barrio"
    )

    df = df.drop(columns=["_precio_m2"])

    Path(ruta_salida_json).resolve().parent.mkdir(parents=True, exist_ok=True)
    with open(ruta_salida_json, "w", encoding="utf-8") as f:
        json.dump({
            "mapa_barrio_a_nivel": mapa_barrio_a_nivel,
            "nivel_default": nivel_default,
            "cortes_valor": cortes_valor,
            "nombres_niveles": NOMBRES_NIVELES,
            "k_suavizado": K_SUAVIZADO,
        }, f, ensure_ascii=False, indent=2)

    df["nivel_barrio"] = aplicar_niveles_barrio(df, "barrio", mapa_barrio_a_nivel, nivel_default)
    df.drop("barrio", axis=1, inplace=True)
    return df


# ------------------------------------------------------------------
# Estimación de superficies bajo el umbral mínimo (modelo por tipo de propiedad)
# ------------------------------------------------------------------
def estimar_superficie(df: pd.DataFrame, columna_objetivo: str,
                        columnas_predictoras: list = COLUMNAS_PREDICTORAS_SUPERFICIE,
                        umbral_minimo: float = UMBRAL_MINIMO_M2,
                        columna_agrupacion: str = "tipo_propiedad",
                        ruta_modelos: Path = None) -> pd.Series:
    """
    Entrena UN MODELO SEPARADO por cada valor de `columna_agrupacion` (por
    defecto 'tipo_propiedad'; como el pipeline solo procesa departamentos,
    en la práctica esto entrena un único modelo, para 'departamento'), usando
    SOLO las filas con `columna_objetivo` >= umbral_minimo como datos
    confiables, y estima el valor para las filas con valor < umbral_minimo o
    NaN.

    El modelo FINAL de cada grupo se entrena con el 100% de los datos
    confiables de ese grupo (no solo el 80% usado para la validación), y se
    guarda en un .pkl (junto con la lista de columnas que espera como input)
    para poder reutilizarlo en producción sin tener que reentrenar.

    `ruta_modelos` permite redirigir dónde se guardan esos .pkl. Es importante
    poder hacerlo: los de la ruta por defecto (RUTA_MODELOS) NO son solo un
    subproducto del entrenamiento, los lee `aplicar_modelo_guardado` en cada
    inferencia de producción. Sin este parámetro, cualquier corrida
    exploratoria del pipeline (ej. comparar configuraciones de `estados`) los
    sobrescribe en silencio y le cambia el comportamiento al modelo ya
    desplegado.

    Devuelve la columna ya corregida (float).
    """
    ruta_modelos = RUTA_MODELOS if ruta_modelos is None else Path(ruta_modelos)
    ruta_modelos.mkdir(parents=True, exist_ok=True)
    resultado = df[columna_objetivo].astype(float).copy()

    es_confiable = resultado.notna() & (resultado >= umbral_minimo)
    necesita_estimacion = ~es_confiable

    for valor_grupo, indices_grupo in df.groupby(columna_agrupacion).groups.items():
        indices_grupo = pd.Index(indices_grupo)

        conf_grupo = es_confiable.loc[indices_grupo]
        est_grupo = necesita_estimacion.loc[indices_grupo]

        if conf_grupo.sum() < MINIMO_FILAS_PARA_ENTRENAR:
            continue

        X_grupo = df.loc[indices_grupo, columnas_predictoras]
        columnas_categoricas = X_grupo.select_dtypes(include=["object", "category", "str"]).columns.tolist()
        X_grupo = pd.get_dummies(X_grupo, columns=columnas_categoricas, drop_first=True)

        X_train_full = X_grupo[conf_grupo]
        y_train_full = resultado.loc[indices_grupo][conf_grupo]

        # --- Validación rápida (solo para reportar qué tan confiable es) ---
        X_tr, X_val, y_tr, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)

        modelo_val = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
        modelo_val.fit(X_tr, y_tr)
        pred_val = modelo_val.predict(X_val)

        mae_val = mean_absolute_error(y_val, pred_val)
        r2_val = r2_score(y_val, pred_val)

        # --- Modelo FINAL: entrenado con el 100% de los datos confiables del grupo ---
        modelo_final = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
        modelo_final.fit(X_train_full, y_train_full)

        # --- Guardar modelo + columnas esperadas, para reutilizar en producción ---
        nombre_archivo = ruta_modelos / f"modelo_{columna_objetivo}_{valor_grupo}.pkl"
        joblib.dump({
            "modelo": modelo_final,
            "columnas_entrenamiento": list(X_train_full.columns),
            "columnas_predictoras_originales": columnas_predictoras,
            "columna_objetivo": columna_objetivo,
            "grupo": valor_grupo,
            "mae_validacion": mae_val,
            "r2_validacion": r2_val,
        }, nombre_archivo)

        # --- Aplicar la estimación a las filas que la necesitan (si hay) ---
        if est_grupo.sum() == 0:
            continue

        X_a_estimar = X_grupo[est_grupo]
        estimaciones = np.round(modelo_final.predict(X_a_estimar), 1)
        resultado.loc[X_a_estimar.index] = estimaciones

    return resultado


def aplicar_modelo_guardado(df_nuevo: pd.DataFrame, columna_objetivo: str, grupo: str,
                             columna_agrupacion: str = "tipo_propiedad") -> np.ndarray:
    """
    Carga un modelo ya entrenado y guardado (.pkl) y lo aplica a datos nuevos
    en producción. `df_nuevo` debe tener las mismas columnas originales
    (dormitorios, banos, comuna, etc.) que se usaron al entrenar.
    """
    nombre_archivo = RUTA_MODELOS / f"modelo_{columna_objetivo}_{grupo}.pkl"
    paquete = joblib.load(nombre_archivo)

    X_nuevo = df_nuevo[paquete["columnas_predictoras_originales"]]
    columnas_categoricas = X_nuevo.select_dtypes(include=["object", "category", "str"]).columns.tolist()
    X_nuevo = pd.get_dummies(X_nuevo, columns=columnas_categoricas, drop_first=True)

    # Alinear columnas exactamente como en entrenamiento (agrega con 0 las que
    # falten - ej. una comuna que no apareció en este lote nuevo - y descarta
    # cualquier columna nueva que el modelo no conozca)
    X_nuevo = X_nuevo.reindex(columns=paquete["columnas_entrenamiento"], fill_value=0)

    return paquete["modelo"].predict(X_nuevo)


def corregir_superficies_bajo_umbral(df: pd.DataFrame, ruta_modelos: Path = None) -> pd.DataFrame:
    """
    Aplica `estimar_superficie` por separado a superficie_util_m2 y
    superficie_total_m2. Para superficie_total_m2 se agrega
    superficie_util_m2 como predictor adicional (ya viene corregida del paso
    anterior).

    `ruta_modelos` se propaga a `estimar_superficie` (ver ahí por qué importa
    poder redirigirla).
    """
    columnas_numericas = ["superficie_util_m2", "superficie_total_m2"]
    df[columnas_numericas] = df[columnas_numericas].apply(pd.to_numeric, errors="coerce")

    df["superficie_util_m2"] = estimar_superficie(
        df, "superficie_util_m2", ruta_modelos=ruta_modelos,
    )
    df["superficie_total_m2"] = estimar_superficie(
        df, "superficie_total_m2",
        columnas_predictoras=COLUMNAS_PREDICTORAS_SUPERFICIE + ["superficie_util_m2"],
        ruta_modelos=ruta_modelos,
    )
    return df


# ------------------------------------------------------------------
# Comparador de precios de viviendas cercanas (precio/m2 de sector)
# ------------------------------------------------------------------
def limites_iqr(serie: pd.Series, multiplicador: float = MULTIPLICADOR_IQR):
    """Calcula el rango [Q1 - m*IQR, Q3 + m*IQR] usado para descartar outliers."""
    q1, q3 = serie.quantile([0.25, 0.75])
    iqr = q3 - q1
    return q1 - multiplicador * iqr, q3 + multiplicador * iqr


def agregar_precio_m2_sector(df: pd.DataFrame, radio_metros: float = RADIO_METROS_COMPARADOR_SECTOR,
                              tipos_propiedad_columnas: dict = TIPOS_PROPIEDAD_COLUMNAS) -> pd.DataFrame:
    """
    Para cada aviso, calcula el precio/m2 mediano de las viviendas del mismo
    tipo de propiedad dentro de `radio_metros` (excluyéndose a sí mismo), y lo
    guarda en 'precio_m2_sector_departamento' (único tipo que maneja el
    pipeline). precio_m2 se usa aquí SOLO como variable auxiliar interna
    para promediar el sector - no se guarda en df ni se usa como feature
    directa del modelo (usarla tal cual sería fuga de datos, porque viene del
    costo de la propia fila). Se basa en costo_total_clp (arriendo + gastos
    comunes), no en precio_clp, para que el comparador de mercado quede en
    la misma escala que el target. Si un aviso no tiene vecinos cercanos
    válidos, cae a la mediana general del grupo, y queda marcado en
    'tiene_comparables_cercanos' = False para poder distinguir ese caso de un
    vecino real con valor 0.
    """
    precio_m2_auxiliar = df["costo_total_clp"] / df["superficie_util_m2"].replace(0, np.nan)

    for columna in tipos_propiedad_columnas.values():
        df[columna] = np.nan  # se rellena con el fallback más abajo si no hay vecinos

    df["tiene_comparables_cercanos"] = False

    radio_rad = radio_metros / RADIO_TIERRA_M

    for tipo, indices_grupo in df.groupby("tipo_propiedad").groups.items():
        columna_destino = tipos_propiedad_columnas.get(tipo)
        if columna_destino is None:
            continue

        indices_grupo = pd.Index(indices_grupo)
        precio_m2_grupo = precio_m2_auxiliar.loc[indices_grupo]

        tiene_coords = df.loc[indices_grupo, ["latitud", "longitud"]].notna().all(axis=1)
        tiene_precio_m2 = precio_m2_grupo.notna()

        lim_inf, lim_sup = limites_iqr(precio_m2_grupo[tiene_precio_m2])
        es_razonable = precio_m2_grupo.between(lim_inf, lim_sup)

        idx_validos = indices_grupo[tiene_coords & tiene_precio_m2 & es_razonable]
        mediana_general_grupo = precio_m2_grupo[idx_validos].median()  # fallback si no hay vecinos

        if len(idx_validos) < 2:
            df.loc[indices_grupo, columna_destino] = mediana_general_grupo
            continue

        coords = np.radians(df.loc[idx_validos, ["latitud", "longitud"]].astype(float).values)
        precios_m2 = precio_m2_grupo.loc[idx_validos].values

        arbol = BallTree(coords, metric="haversine")
        vecinos_por_fila = arbol.query_radius(coords, r=radio_rad)

        for pos, idx_fila in enumerate(idx_validos):
            vecinos_idx = vecinos_por_fila[pos]
            vecinos_sin_si_mismo = vecinos_idx[vecinos_idx != pos]

            if len(vecinos_sin_si_mismo) == 0:
                df.loc[idx_fila, columna_destino] = mediana_general_grupo
            else:
                df.loc[idx_fila, columna_destino] = np.median(precios_m2[vecinos_sin_si_mismo])
                df.loc[idx_fila, "tiene_comparables_cercanos"] = True

        # Filas sin coords / sin precio_m2 válido / outlier: también caen al fallback general
        idx_resto = indices_grupo.difference(idx_validos)
        df.loc[idx_resto, columna_destino] = mediana_general_grupo

    return df


def eliminar_columnas_no_necesarias(df: pd.DataFrame) -> pd.DataFrame:
    """Descarta 'precio' y 'moneda': ya se usaron para calcular 'precio_clp'."""
    df.drop(["precio", "moneda"], axis=1, inplace=True)
    return df


# ------------------------------------------------------------------
# Imputación de antigüedad y condominio_cerrado
# ------------------------------------------------------------------
def imputar_antiguedad_y_condominio_por_coordenadas(df: pd.DataFrame) -> pd.DataFrame:
    """Para 'antiguedad_anos' y 'condominio_cerrado', rellena los NaN con la
    moda de otros avisos que comparten exactamente la misma latitud/longitud
    (edificios/condominios ya vistos antes).

    Lo que siga sin dato en 'condominio_cerrado' lo resuelve después
    `convertir_condominio_a_booleano` (asume False): el faltante se rellena
    allá y no acá porque el valor a inyectar depende de la representación de la
    columna, que cambia según el origen de datos (texto en SQLite, 1/0 en
    Postgres) - un `fillna("No")` acá corrompía la columna numérica de
    Postgres."""
    df["antiguedad_anos"] = df.groupby(["latitud", "longitud"])["antiguedad_anos"].transform(
        lambda x: x.fillna(x.mode()[0] if not x.mode().empty else x)
    )
    df["condominio_cerrado"] = df.groupby(["latitud", "longitud"])["condominio_cerrado"].transform(
        lambda x: x.fillna(x.mode()[0] if not x.mode().empty else x)
    )

    return df


def imputar_por_cercania_200m(df: pd.DataFrame, columna: str = "antiguedad_anos",
                               radio_metros: float = RADIO_METROS_ANTIGUEDAD) -> pd.Series:
    """
    Para cada fila con NaN en `columna`, busca otras filas del mismo
    tipo_propiedad dentro de `radio_metros` (usando latitud/longitud) y
    devuelve la mediana de esos vecinos. Si no hay vecinos, deja el NaN
    (se resuelve en los fallbacks siguientes).
    """
    valores_originales = df[columna].copy()
    resultado = valores_originales.copy()
    radio_rad = radio_metros / RADIO_TIERRA_M

    for _, grupo in df.groupby("tipo_propiedad"):
        tiene_coords = grupo[["latitud", "longitud"]].notna().all(axis=1)
        tiene_valor = valores_originales.loc[grupo.index].notna()

        idx_con_dato = grupo.index[tiene_valor & tiene_coords]
        idx_sin_dato = grupo.index[~tiene_valor & tiene_coords]  # sin coords no se puede buscar vecinos

        if len(idx_con_dato) == 0 or len(idx_sin_dato) == 0:
            continue

        coords_con_dato = np.radians(
            grupo.loc[idx_con_dato, ["latitud", "longitud"]].astype(float).values
        )
        coords_sin_dato = np.radians(
            grupo.loc[idx_sin_dato, ["latitud", "longitud"]].astype(float).values
        )

        arbol = BallTree(coords_con_dato, metric="haversine")
        vecinos_por_fila = arbol.query_radius(coords_sin_dato, r=radio_rad)

        for idx_fila, vecinos in zip(idx_sin_dato, vecinos_por_fila):
            if len(vecinos) > 0:
                valores_vecinos = valores_originales.loc[idx_con_dato].iloc[vecinos]
                resultado.loc[idx_fila] = valores_vecinos.median()

    return resultado


def imputar_antiguedad_anos_por_vecinos(df: pd.DataFrame, radio_metros: float = RADIO_METROS_ANTIGUEDAD) -> pd.DataFrame:
    """
    Termina de completar 'antiguedad_anos' en cascada de fallbacks, todos
    calculados sobre los valores ORIGINALES (nunca sobre estimaciones de un
    paso anterior, para no contaminar las medianas con datos ya imputados):
      1) mediana de vecinos dentro de `radio_metros` (mismo tipo_propiedad)
      2) mediana por tipo_propiedad
      3) mediana por comuna
      4) mediana global
    Al final castea a entero, dado que ya no debería quedar ningún NaN.
    """
    df["antiguedad_anos"] = pd.to_numeric(df["antiguedad_anos"], errors="coerce")
    df["latitud"] = pd.to_numeric(df["latitud"], errors="coerce")
    df["longitud"] = pd.to_numeric(df["longitud"], errors="coerce")

    valores_originales = df["antiguedad_anos"].copy()

    # Paso 1: vecinos dentro del radio, mismo tipo_propiedad
    df["antiguedad_anos"] = imputar_por_cercania_200m(df, "antiguedad_anos", radio_metros)

    # Paso 2: mediana por tipo_propiedad (calculada desde los datos ORIGINALES)
    mediana_por_tipo = df.groupby("tipo_propiedad")["tipo_propiedad"].transform(
        lambda serie: valores_originales.loc[serie.index].median()
    )
    df["antiguedad_anos"] = df["antiguedad_anos"].fillna(mediana_por_tipo)

    # Paso 3: mediana por comuna (calculada desde los datos ORIGINALES)
    mediana_por_comuna = df.groupby("comuna")["comuna"].transform(
        lambda serie: valores_originales.loc[serie.index].median()
    )
    df["antiguedad_anos"] = df["antiguedad_anos"].fillna(mediana_por_comuna)

    # Paso 4: mediana global (también desde los datos ORIGINALES)
    df["antiguedad_anos"] = df["antiguedad_anos"].fillna(valores_originales.median())

    df["antiguedad_anos"] = df["antiguedad_anos"].round().astype("Int64")

    return df


def eliminar_columnas_geograficas(df: pd.DataFrame) -> pd.DataFrame:
    """Descarta latitud/longitud/fecha_publicacion_aprox: ya cumplieron su
    propósito (imputaciones por cercanía y conversión UF->CLP)."""
    df.drop(columns=["latitud", "longitud", "fecha_publicacion_aprox"], inplace=True)
    return df


# ------------------------------------------------------------------
# Codificación de variables categóricas y variables derivadas finales
# ------------------------------------------------------------------
def eliminar_columna_tipo_propiedad(df: pd.DataFrame) -> pd.DataFrame:
    """Descarta 'tipo_propiedad': el pipeline solo procesa departamentos (las
    casas se filtran en el origen, ver `cargar_datos`), por lo que la columna
    queda constante y no aporta información como feature."""
    return df.drop(columns=["tipo_propiedad"])


def convertir_columnas_a_numerico_final(df: pd.DataFrame) -> pd.DataFrame:
    """Castea a numérico dormitorios, baños, superficie_m2 y estacionamientos."""
    columnas = ["dormitorios", "banos", "superficie_m2", "estacionamientos"]
    df[columnas] = df[columnas].apply(pd.to_numeric, errors="coerce")
    return df


def codificar_comuna(df: pd.DataFrame, comunas_a_agrupar: list = COMUNAS_A_AGRUPAR) -> pd.DataFrame:
    """Agrupa en 'otras comunas' las comunas con pocos avisos (para no generar
    dummies con muy pocas observaciones) y luego aplica one-hot encoding."""
    df.loc[df["comuna"].isin(comunas_a_agrupar), "comuna"] = "otras comunas"
    return pd.get_dummies(df, columns=["comuna"], prefix="comuna", drop_first=True)


def crear_variable_amoblado(df: pd.DataFrame, patron: re.Pattern = PATRON_AMOBLADO) -> pd.DataFrame:
    """Deriva 'amoblado' (1/0) buscando "amoblado/a" o "amueblado/a" en el
    título del aviso, y descarta el título ya que no se usa como feature."""
    df["amoblado"] = df["titulo"].str.contains(patron, na=False).astype(int)
    df.drop(columns=["titulo"], inplace=True)
    return df


def completar_superficies_faltantes(df: pd.DataFrame) -> pd.DataFrame:
    """Si 'superficie_total_m2' o 'superficie_util_m2' quedaron en NaN, las
    rellena con 'superficie_m2' (la superficie general del aviso). Descarta
    los avisos que ni así logran completarse, y elimina 'superficie_m2' ya
    que su información quedó absorbida en las otras dos columnas."""
    df["superficie_total_m2"] = df["superficie_total_m2"].fillna(df["superficie_m2"])
    df["superficie_util_m2"] = df["superficie_util_m2"].fillna(df["superficie_m2"])
    df[["superficie_total_m2", "superficie_util_m2"]] = df[["superficie_total_m2", "superficie_util_m2"]].apply(
        pd.to_numeric, errors="coerce"
    )

    df.dropna(subset=["superficie_total_m2", "superficie_util_m2"], inplace=True)
    df.drop(columns=["superficie_m2"], inplace=True)
    return df


def convertir_condominio_a_booleano(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza 'condominio_cerrado' a booleano, aceptando las DOS
    representaciones que puede traer la columna según el origen de datos:

      - texto "Sí"/"No" — esquema SQLite de investigación, tal como lo deja el
        scraper original.
      - 1/0 — esquema Postgres de producción, donde
        02_scraper_detalle_incremental.py ya lo castea con `_a_booleano`.

    Antes esta función asumía solo texto (`.str.lower().map(...)`). Contra
    Postgres eso convertía en NaN justamente las filas que SÍ tenían dato -el
    accesor .str devuelve NaN sobre valores numéricos-, dejando la columna con
    78% de nulos y conservando solo las que NO tenían información. El
    tratamiento del faltante ("sin dato" -> False) se hace acá y no en
    `imputar_antiguedad_y_condominio_por_coordenadas`, para no tener que
    inyectar un valor de un tipo que dependa del origen.
    """
    serie = df["condominio_cerrado"]

    if serie.dtype == object:
        texto = serie.astype("string").str.strip().str.lower()
        # "1"/"0" y "1.0"/"0.0" cubren el caso de una columna numérica que
        # quedó como object por una imputación previa de tipo mixto.
        mapeado = texto.map({
            "sí": True, "si": True, "no": False,
            "true": True, "false": False,
            "1": True, "0": False, "1.0": True, "0.0": False,
        })
    else:
        mapeado = pd.to_numeric(serie, errors="coerce").map({1: True, 0: False})

    df["condominio_cerrado"] = mapeado.fillna(False).astype(bool)
    return df


def crear_ratio_superficie(df: pd.DataFrame) -> pd.DataFrame:
    """Crea 'ratio_total_util' = superficie_total_m2 / superficie_util_m2,
    como proxy de cuánta superficie común/no habitable tiene la propiedad."""
    df["ratio_total_util"] = (
        df["superficie_total_m2"] / df["superficie_util_m2"].replace(0, float("nan"))
    )
    return df



def guardar_resultado(df: pd.DataFrame, ruta_salida: str = RUTA_SALIDA_CSV_DEFAULT) -> pd.DataFrame:
    """Guarda el dataset final en CSV, listo para la etapa de modelamiento."""
    Path(ruta_salida).resolve().parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(ruta_salida, index=False)
    return df


def imprimir_resumen_final(df: pd.DataFrame) -> None:
    """Imprime un resumen del dataset final: dimensiones y cantidad de
    valores nulos por variable (si los hay)."""
    print("=" * 70)
    print("RESUMEN FINAL DEL PIPELINE")
    print("=" * 70)
    print(f"Filas: {df.shape[0]} | Columnas: {df.shape[1]}")

    nulos = df.isna().sum()
    nulos = nulos[nulos > 0].sort_values(ascending=False)

    if nulos.empty:
        print("No quedaron valores nulos en ninguna variable.")
    else:
        print(f"Variables con valores nulos ({len(nulos)}):")
        for columna, cantidad in nulos.items():
            print(f"  {columna}: {cantidad} nulos ({cantidad / df.shape[0] * 100:.2f}%)")
    print("=" * 70)

# ------------------------------------------------------------------
# Variables derivadas adicionales (ratios, densidades e interacciones)
# ------------------------------------------------------------------
def crear_variables_derivadas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea features derivadas a partir de columnas ya existentes en el
    pipeline, pensadas para darle al modelo señales que hoy están implícitas
    pero no explícitas (ratios de tamaño, densidad de amenities/POIs,
    interacciones ubicación-superficie). Ninguna usa 'precio_clp' de la
    propia fila, por lo que no hay riesgo de fuga de datos.

    Debe llamarse después de que 'nivel_barrio', 'precio_m2_sector_departamento'
    y las columnas de amenities/POIs ya existan en el DataFrame.
    """
    df = df.copy()

    dormitorios_safe = df["dormitorios"].replace(0, np.nan)
    superficie_util_safe = df["superficie_util_m2"].replace(0, np.nan)
    superficie_total_safe = df["superficie_total_m2"].replace(0, np.nan)
    distancia_comuna_safe = df["distancia_centro_comuna_m"].replace(0, np.nan)

    # --- Tamaño relativo por habitación ---
    # Cuánto espacio "le toca" a cada dormitorio: dos deptos de igual
    # superficie total pueden sentirse muy distintos según cuántos dormitorios
    # reparten ese espacio.
    df["superficie_util_por_dormitorio"] = (
        df["superficie_util_m2"] / dormitorios_safe
    ).fillna(df["superficie_util_m2"])

    df["banos_por_dormitorio"] = (df["banos"] / dormitorios_safe).fillna(df["banos"])

    df["estacionamientos_por_dormitorio"] = (
        df["estacionamientos"] / dormitorios_safe
    ).fillna(0)

    # --- Costo de mantención relativo al tamaño ---
    # gastos_comunes ya está en el modelo en términos absolutos; en términos
    # de CLP/m2 es más comparable entre deptos grandes y chicos.
    df["gastos_comunes_por_m2"] = (df["gastos_comunes"] / superficie_util_safe).fillna(0)

    # --- Índice de amenities ---
    # Conteo simple de comodidades presentes, en vez de que el árbol tenga
    # que combinar 5 columnas binarias por separado en cada split.
    columnas_amenities = ["piscina", "quincho", "conserjeria", "ascensor", "estacionamiento_visitas"]
    df["indice_amenities"] = df[columnas_amenities].sum(axis=1)

    # --- Interacción piso alto sin ascensor ---
    # Un piso 8 con ascensor no es lo mismo que un piso 8 sin ascensor;
    # esta interacción lo deja explícito para el árbol.
    df["piso_sin_ascensor"] = df["piso_unidad"] * (1 - df["ascensor"])

    # --- Densidad total de puntos de interés cercanos ---
    # Accesibilidad general del sector en un solo número, en vez de 11
    # columnas de conteo por separado.
    columnas_pois = [
        "cantidad_paraderos", "cantidad_jardines_infantiles", "cantidad_colegios",
        "cantidad_universidades", "cantidad_plazas", "cantidad_supermercados",
        "cantidad_farmacias", "cantidad_centros_comerciales", "cantidad_hospitales",
        "cantidad_clinicas", "cantidad_estaciones_metro",
    ]
    df["densidad_pois_cercanos"] = df[columnas_pois].sum(axis=1)

    # --- Ubicación relativa dentro de la comuna ---
    # Qué tan cerca está del centro de Concepción en relación a qué tan cerca
    # está del centro de su propia comuna (proxy de si es "periferia
    # bien conectada" vs. "periferia aislada").
    df["ratio_distancia_concepcion_comuna"] = (
        df["distancia_centro_concepcion_m"] / distancia_comuna_safe
    ).fillna(df["distancia_centro_concepcion_m"])

    # --- Valor de mercado implícito del sector, aplicado a la superficie propia ---
    # precio_m2_sector_departamento viene del entorno (vecinos), no de la
    # propia fila, así que no hay fuga. Multiplicarlo por la superficie propia
    # da una estimación de "cuánto valdría según el mercado local", que puede
    # ayudar especialmente en el segmento de precios altos donde el modelo
    # tiene menos ejemplos.
    df["valor_mercado_estimado_sector"] = df["precio_m2_sector_departamento"] * df["superficie_util_m2"]

    # --- Interacción nivel de barrio x superficie ---
    # Un m2 adicional no vale lo mismo en un barrio "muy_alto" que en uno
    # "muy_barato"; esta interacción lo deja explícito.
    df["nivel_barrio_x_superficie"] = df["nivel_barrio"] * df["superficie_util_m2"]

    # --- Eficiencia de espacio ---
    # Qué proporción de la superficie total es realmente útil/habitable
    # (inverso conceptual de ratio_total_util, calculado aquí de forma
    # independiente por si esta función se usa antes de esa otra).
    df["eficiencia_espacio"] = (superficie_util_safe / superficie_total_safe).fillna(1.0)

    return df

# ------------------------------------------------------------------
# Orquestación del pipeline completo
# ------------------------------------------------------------------
def ejecutar_pipeline(
    con,
    ruta_salida_csv: str = RUTA_SALIDA_CSV_DEFAULT,
    ruta_salida_niveles_barrio: str = RUTA_SALIDA_NIVELES_BARRIO_DEFAULT,
    estados=ESTADOS_ENTRENAMIENTO_DEFAULT,
    meses_max_antiguedad_finalizados=MESES_MAX_ANTIGUEDAD_FINALIZADOS_DEFAULT,
    ruta_modelos_superficie=None,
) -> pd.DataFrame:
    """
    Ejecuta de punta a punta la ingeniería de variables y devuelve el
    DataFrame final (además de guardarlo en `ruta_salida_csv`), para que un
    script orquestador externo pueda encadenarlo con otros pasos.

    `con` es una conexión Postgres/Supabase ya abierta (el caller es dueño de
    su ciclo de vida). Los filtros de población de entrenamiento `estados` /
    `meses_max_antiguedad_finalizados` se documentan en `cargar_datos`.

    OJO con los tres artefactos que esta función ESCRIBE, porque no son solo
    salidas del entrenamiento - `04_ingenieria_variables_produccion.py` los lee
    en cada inferencia: el CSV (`ruta_salida_csv`), los niveles de barrio
    (`ruta_salida_niveles_barrio`) y los modelos de imputación de superficie
    (`ruta_modelos_superficie`). Los tres son redirigibles justamente para que
    una corrida exploratoria -comparar configuraciones de `estados`, por
    ejemplo- pueda escribir en otro lado en vez de pisarle los artefactos al
    modelo que está desplegado.
    """
    df_avisos, df_detalle, df_llave_vulnerabilidad, df_vulnerabilidad = cargar_datos(
        con,
        estados=estados,
        meses_max_antiguedad_finalizados=meses_max_antiguedad_finalizados,
    )

    df_avisos, df_detalle = seleccionar_columnas_relevantes(df_avisos, df_detalle)
    df = combinar_tablas(df_avisos, df_detalle, df_llave_vulnerabilidad, df_vulnerabilidad)

    df = eliminar_outliers_habitaciones(df)
    df = eliminar_registros_incompletos(df)
    # Antes de cualquier paso que use coordenadas (imputación por cercanía,
    # precio/m2 del sector, nivel de barrio): una coordenada errónea no solo
    # ensucia su propia fila, también contaminaría a sus "vecinos".
    df = descartar_coordenadas_fuera_de_region(df)
    df = imputar_vulnerabilidad_por_comuna(df)
    df = convertir_superficie_total_a_numerico(df)

    df = fill_barrio_nan(df)

    df = preprocesar_variables_amenities(df)
    df = corregir_piso_unidad_outliers(df)
    df = rellenar_cantidad_pois_cercanos(df)

    df = convertir_precios_uf_a_clp(df, con=con)
    # El filtro por precio_clp necesita esta columna ya calculada, por eso se
    # aplica aquí y no inmediatamente después de eliminar_registros_incompletos.
    df = filtrar_precio_maximo(df)
    # costo_total_clp (target del modelo) se calcula después del filtro de
    # arriendo máximo, para que ese filtro siga evaluando solo el arriendo
    # nominal, y antes de nivel_barrio/precio_m2_sector, que ya lo usan.
    df = agregar_costo_total(df)

    df = agregar_nivel_barrio(df, ruta_salida_niveles_barrio)

    df = corregir_superficies_bajo_umbral(df, ruta_modelos=ruta_modelos_superficie)
    df = agregar_precio_m2_sector(df)

    df = eliminar_columnas_no_necesarias(df)

    df = imputar_antiguedad_y_condominio_por_coordenadas(df)
    df = imputar_antiguedad_anos_por_vecinos(df)
    df = eliminar_columnas_geograficas(df)

    df = eliminar_columna_tipo_propiedad(df)
    df = convertir_columnas_a_numerico_final(df)
    df = codificar_comuna(df)
    df = crear_variable_amoblado(df)
    df = completar_superficies_faltantes(df)
    df = convertir_condominio_a_booleano(df)
    df = crear_ratio_superficie(df)
    #df = crear_variables_derivadas(df) 

    df = guardar_resultado(df, ruta_salida_csv)

    imprimir_resumen_final(df)

    return df


if __name__ == "__main__":
    # Conexión propia a producción, armada acá y no vía produccion/db.py a
    # propósito: este módulo no debe depender de produccion/ (lo importan
    # también el dashboard y el pipeline, y la dirección de la dependencia es
    # la inversa). El import de psycopg2 es local por lo mismo: el módulo no lo
    # necesita para nada más.
    import os

    import psycopg2

    # Lectura del .env duplicada respecto de `cargar_env_si_existe` en
    # produccion/01_modelo_produccion/db.py, por esa misma dirección de
    # dependencia (mismo criterio que SUBCATEGORIAS_POI_COLUMNAS allá). Sin
    # esto, correr este script como lo documenta SETUP.md falla con
    # KeyError('BD_STRING'): nada más lee ese archivo.
    ruta_env = DIRECTORIO_SCRIPT.parent.parent / ".env"
    if ruta_env.exists():
        for linea in ruta_env.read_text(encoding="utf-8").splitlines():
            linea = linea.strip()
            if not linea or linea.startswith("#") or "=" not in linea:
                continue
            clave, valor = linea.split("=", 1)
            clave = clave.strip()
            if clave and clave not in os.environ:
                os.environ[clave] = valor.strip().strip('"').strip("'")

    if "BD_STRING" not in os.environ:
        raise SystemExit(
            f"Falta la variable de entorno BD_STRING. Definila en {ruta_env} "
            f"(ver SETUP.md, sección 3)."
        )

    con_produccion = psycopg2.connect(os.environ["BD_STRING"], sslmode="require")
    try:
        ejecutar_pipeline(con_produccion)
    finally:
        con_produccion.close()
