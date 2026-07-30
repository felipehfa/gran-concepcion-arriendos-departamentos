"""
Esquema y conexión de la base de datos de PRODUCCIÓN (Supabase / Postgres).

Este módulo NO scrapea ni calcula nada — solo define las tablas y ofrece
helpers de conexión, para que el resto de los scripts de produccion/01_modelo_produccion/
lo importen (`from db import ...`).

La conexión se arma desde la variable de entorno BD_STRING (connection string
del connection pooler de Supabase, modo transaction - ver .env).

Tablas (ver inicializar_bd_produccion más abajo para el detalle de columnas):
avisos, avisos_detalle, poligonos_vulnerabilidad_uv, predicciones, corridas,
logs_ejecucion, historial_diario_avisos, control, valores_uf.
"""

import os
from pathlib import Path

import psycopg2

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RUTA_ENV = REPO_ROOT / ".env"


def cargar_env_si_existe(ruta_env: Path = RUTA_ENV) -> None:
    """Carga al entorno las variables de un archivo `.env` en la raíz del repo.

    Hace falta porque nada más lo leía: SETUP.md indica crear ese archivo con
    `BD_STRING` para correr localmente, pero cualquier script fallaba con
    KeyError('BD_STRING') porque `os.environ` no se entera solo del archivo.

    "Si existe" es la parte importante: en GitHub Actions y en Streamlit Cloud no
    hay `.env` (las variables llegan como secrets ya en el entorno), así que ahí
    esto es un no-op. Las variables YA definidas en el entorno tienen prioridad
    sobre el archivo, para que un `export BD_STRING=...` puntual gane.

    No se usa python-dotenv para no agregar una dependencia por diez líneas -
    mismo criterio que el resto del repo, que no tiene requirements en la raíz.
    """
    if not ruta_env.exists():
        return

    for linea in ruta_env.read_text(encoding="utf-8").splitlines():
        linea = linea.strip()
        if not linea or linea.startswith("#") or "=" not in linea:
            continue
        clave, valor = linea.split("=", 1)
        clave = clave.strip()
        # Se aceptan comillas alrededor del valor: el formato TOML de los
        # secrets de Streamlit Cloud las usa, y es fácil copiar/pegar de ahí.
        valor = valor.strip().strip('"').strip("'")
        if clave and clave not in os.environ:
            os.environ[clave] = valor


# Mismas subcategorías de punto de interés que `SUBCATEGORIAS_POI.values()`
# en `scrapers_base/02_scraper_detalle.py`. Se duplica esta lista (en vez
# de importar ese script) para que `db.py` no arrastre la dependencia de
# Playwright — este módulo lo importan también etapas puramente de
# datos/modelo (ingeniería de variables, predicción) que no deberían
# necesitar un navegador instalado. Si cambia la lista de subcategorías en
# el scraper de detalle, hay que reflejarlo acá también.
SUBCATEGORIAS_POI_COLUMNAS = [
    "paraderos",
    "estaciones_metro",
    "jardines_infantiles",
    "colegios",
    "universidades",
    "plazas",
    "supermercados",
    "farmacias",
    "centros_comerciales",
    "hospitales",
    "clinicas",
]


# ------------------------------------------------------------------
# Conexión
# ------------------------------------------------------------------
def conectar_produccion() -> psycopg2.extensions.connection:
    """Abre la conexión a la base de datos de producción (Supabase) y se
    asegura de que las tablas existan."""
    cargar_env_si_existe()
    if "BD_STRING" not in os.environ:
        raise RuntimeError(
            f"Falta la variable de entorno BD_STRING. Localmente se define en {RUTA_ENV} "
            f"(ver SETUP.md, sección 3); en GitHub Actions y Streamlit Cloud, como secret."
        )
    # sslmode='require' explícito: sin esto psycopg2 usa 'prefer' por
    # default, que se degrada en silencio a texto plano si la negociación
    # TLS llega a fallar por algún motivo, en vez de cortar la conexión.
    con = psycopg2.connect(os.environ["BD_STRING"], sslmode="require")
    inicializar_bd_produccion(con)
    return con


# ------------------------------------------------------------------
# Esquema
# ------------------------------------------------------------------
def inicializar_bd_produccion(con: psycopg2.extensions.connection) -> None:
    cur = con.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS avisos (
            id_aviso                    TEXT PRIMARY KEY,
            comuna                      TEXT NOT NULL,
            tipo_propiedad              TEXT NOT NULL,
            operacion                   TEXT NOT NULL,
            titulo                      TEXT,
            precio                      DOUBLE PRECISION,
            moneda                      TEXT,
            ubicacion                   TEXT,
            dormitorios                 INTEGER,
            banos                       INTEGER,
            superficie_m2               DOUBLE PRECISION,
            url                         TEXT,
            first_seen                  TEXT,
            estado_publicacion          TEXT NOT NULL DEFAULT 'activo'
                                         CHECK (estado_publicacion IN
                                               ('activo', 'pausado', 'finalizado', 'no_disponible')),
            fecha_ultimo_chequeo_estado TEXT,
            intentos_fallidos_detalle   INTEGER DEFAULT 0
        )
    """)

    columnas_poi = ",\n            ".join(
        f"cantidad_{clave} INTEGER,\n            distancia_min_m_{clave} DOUBLE PRECISION"
        for clave in SUBCATEGORIAS_POI_COLUMNAS
    )

    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS avisos_detalle (
            id_aviso                       TEXT PRIMARY KEY REFERENCES avisos(id_aviso),
            descripcion                    TEXT,
            fecha_publicacion_texto        TEXT,
            fecha_publicacion_aprox        TEXT,
            fecha_publicacion_precision    TEXT,
            superficie_total_m2            DOUBLE PRECISION,
            superficie_util_m2             DOUBLE PRECISION,
            dormitorios                    INTEGER,
            banos                          INTEGER,
            estacionamientos               INTEGER,
            antiguedad_anos                INTEGER,
            amoblado                       INTEGER,
            admite_mascotas                INTEGER,
            condominio_cerrado             INTEGER,
            bodegas                        INTEGER,
            gastos_comunes                 DOUBLE PRECISION,
            estacionamiento_visitas        INTEGER,
            solo_familias                  INTEGER,
            max_habitantes                 INTEGER,
            piscina                        INTEGER,
            quincho                        INTEGER,
            conserjeria                    INTEGER,
            ascensor                       INTEGER,
            piso_unidad                    INTEGER,
            deptos_por_piso                INTEGER,
            barrio                         TEXT,
            latitud                        DOUBLE PRECISION,
            longitud                       DOUBLE PRECISION,
            distancia_centro_comuna_m      DOUBLE PRECISION,
            distancia_centro_concepcion_m  DOUBLE PRECISION,
            {columnas_poi},
            uv_rsh                         TEXT,
            rank_nac                       DOUBLE PRECISION,
            pob_rsh_uv                     INTEGER,
            p_urbano                       DOUBLE PRECISION,
            c_ig_com                       DOUBLE PRECISION,
            hog_uv                         DOUBLE PRECISION,
            fecha_scrapeo                  TEXT
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS poligonos_vulnerabilidad_uv (
            uv_rsh         TEXT PRIMARY KEY,
            comuna         TEXT NOT NULL,
            rank_nac       DOUBLE PRECISION,
            pob_rsh_uv     INTEGER,
            p_urbano       DOUBLE PRECISION,
            c_ig_com       DOUBLE PRECISION,
            hog_uv         DOUBLE PRECISION,
            geometria_wkt  TEXT NOT NULL
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS predicciones (
            id                     BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            id_aviso               TEXT NOT NULL REFERENCES avisos(id_aviso),
            version_modelo         TEXT NOT NULL,
            fecha_prediccion       TEXT NOT NULL,
            costo_total_predicho   DOUBLE PRECISION NOT NULL,
            z_robusto              DOUBLE PRECISION,
            decil_precio           INTEGER,
            etiqueta               TEXT CHECK (etiqueta IN ('oportunidad', 'caro', 'precio_de_mercado')),
            nivel_confianza        TEXT CHECK (nivel_confianza IN ('alta confianza', 'confianza media', 'baja confianza')),
            cv_ensamble            DOUBLE PRECISION,
            UNIQUE (id_aviso, version_modelo)
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS corridas (
            id_corrida                 BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            fecha_inicio                TEXT NOT NULL,
            fecha_fin                   TEXT,
            resultado                   TEXT CHECK (resultado IN ('ok', 'error', 'parcial')),
            version_modelo_usada        TEXT,
            avisos_nuevos_grilla        INTEGER DEFAULT 0,
            avisos_nuevos_detalle       INTEGER DEFAULT 0,
            avisos_rechequeados         INTEGER DEFAULT 0,
            avisos_cambio_estado        INTEGER DEFAULT 0,
            paginas_recorridas_grilla   INTEGER DEFAULT 0,
            motivo_corte_grilla         TEXT CHECK (motivo_corte_grilla IN
                                         ('paginas_vacias_consecutivas', 'limite_paginas', 'limite_tiempo')),
            etapa_fallida                TEXT,
            mensaje_error                TEXT
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS logs_ejecucion (
            id          BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            id_corrida  BIGINT REFERENCES corridas(id_corrida),
            timestamp   TEXT NOT NULL,
            etapa       TEXT CHECK (etapa IN
                        ('scraper_grilla', 'scraper_detalle', 'rechequeo_estado',
                         'vulnerabilidad', 'variables', 'prediccion', 'historial_diario',
                         'insercion_bd', 'orquestador')),
            nivel       TEXT CHECK (nivel IN ('info', 'warning', 'error')),
            mensaje     TEXT NOT NULL,
            detalle     TEXT
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS historial_diario_avisos (
            fecha               TEXT NOT NULL,
            id_aviso            TEXT NOT NULL REFERENCES avisos(id_aviso),
            estado_publicacion  TEXT NOT NULL,
            PRIMARY KEY (fecha, id_aviso)
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS control (
            clave  TEXT PRIMARY KEY,
            valor  TEXT
        )
    """)

    # valores_uf: caché del valor de la UF por fecha, usada por
    # convertir_precios_uf_a_clp (modelamiento/01_ingenieria_variables/01_ingenieria_variables.py).
    # Se crea acá (con el rol `postgres`, dueño del schema) y no solo desde esa
    # función, porque el rol `streamlit_app` del dashboard (ver SETUP.md) no
    # tiene privilegio CREATE sobre el schema - si esta tabla no existiera ya
    # cuando el dashboard hace su primera consulta, el CREATE TABLE IF NOT
    # EXISTS de esa función fallaría con InsufficientPrivilege.
    cur.execute("""
        CREATE TABLE IF NOT EXISTS valores_uf (
            fecha TEXT PRIMARY KEY,
            valor DOUBLE PRECISION
        )
    """)

    con.commit()


# ------------------------------------------------------------------
# Helpers de `avisos.intentos_fallidos_detalle`
# (fallos persistentes de scraping ENTRE corridas - ver
# 02_scraper_detalle_incremental.py para el umbral y la lógica de cuándo
# marcar un aviso como 'no_disponible')
# ------------------------------------------------------------------
def incrementar_intentos_fallidos_detalle(con: psycopg2.extensions.connection, id_aviso: str) -> int:
    """Suma 1 al contador de fallos consecutivos de este aviso y devuelve el nuevo valor."""
    cur = con.cursor()
    cur.execute(
        "UPDATE avisos SET intentos_fallidos_detalle = COALESCE(intentos_fallidos_detalle, 0) + 1 "
        "WHERE id_aviso = %s",
        (id_aviso,),
    )
    con.commit()
    cur.execute("SELECT intentos_fallidos_detalle FROM avisos WHERE id_aviso = %s", (id_aviso,))
    fila = cur.fetchone()
    return fila[0] if fila else 0


def resetear_intentos_fallidos_detalle(con: psycopg2.extensions.connection, id_aviso: str) -> None:
    """Vuelve el contador a 0 - se llama cuando un aviso finalmente se scrapea con éxito."""
    con.cursor().execute("UPDATE avisos SET intentos_fallidos_detalle = 0 WHERE id_aviso = %s", (id_aviso,))
    con.commit()


# ------------------------------------------------------------------
# Helpers de la tabla `control` (clave/valor genérico)
# ------------------------------------------------------------------
def leer_control(con: psycopg2.extensions.connection, clave: str) -> str:
    cur = con.cursor()
    cur.execute("SELECT valor FROM control WHERE clave = %s", (clave,))
    fila = cur.fetchone()
    return fila[0] if fila else None


def escribir_control(con: psycopg2.extensions.connection, clave: str, valor: str) -> None:
    con.cursor().execute(
        "INSERT INTO control (clave, valor) VALUES (%s, %s) "
        "ON CONFLICT (clave) DO UPDATE SET valor = excluded.valor",
        (clave, valor),
    )
    con.commit()


if __name__ == "__main__":
    con = conectar_produccion()
    cur = con.cursor()
    cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' ORDER BY table_name")
    tablas = cur.fetchall()
    print("Base de datos de producción inicializada en Supabase.")
    print(f"Tablas: {[t[0] for t in tablas]}")
    con.close()
