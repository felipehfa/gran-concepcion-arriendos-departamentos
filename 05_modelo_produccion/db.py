"""
Esquema y conexión de la base de datos de PRODUCCIÓN
(05_modelo_produccion/produccion_gran_concepcion.db).

Este módulo NO scrapea ni calcula nada — solo define las tablas y ofrece
helpers de conexión, para que el resto de los scripts de 05_modelo_produccion/
lo importen (`from db import ...`).

Tablas:
  - avisos           : nivel grilla (igual que la tabla `avisos` original,
                        + estado_publicacion / fecha_ultimo_chequeo_estado)
  - avisos_detalle   : nivel detalle, 1:1 con `avisos` (igual que la tabla
                        `avisos_detalle` original, + columnas de
                        vulnerabilidad IGVUST resueltas directo, sin las
                        tablas `vulnerabilidad_uv`/`avisos_igvust` separadas
                        que usa la base de datos original)
  - predicciones     : una fila por (id_aviso, version_modelo)
  - corridas         : metadatos de cada corrida del orquestador
  - logs_ejecucion   : log persistente de cada etapa, por corrida
  - control          : clave/valor genérico para estado interno de los
                        scripts (ej. cooldown tras CAPTCHA del scraper de
                        detalle) — equivalente a la tabla `estado` de la
                        base de datos original, pero propia de producción
"""

import sqlite3
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

RUTA_BD_PRODUCCION = SCRIPT_DIR / "produccion_gran_concepcion.db"
RUTA_BD_ORIGINAL = REPO_ROOT / "01_obtener_datos" / "avisos_gran_concepcion.db"

# Mismas subcategorías de punto de interés que `SUBCATEGORIAS_POI.values()`
# en `01_obtener_datos/02_scraper_detalle.py`. Se duplica esta lista (en vez
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
def conectar_produccion(ruta_bd: Path = RUTA_BD_PRODUCCION) -> sqlite3.Connection:
    """Abre (o crea) la base de datos de producción, con foreign_keys ON,
    y se asegura de que las tablas existan."""
    con = sqlite3.connect(ruta_bd)
    con.execute("PRAGMA foreign_keys = ON")
    _migrar_esquema_avisos(con)
    inicializar_bd_produccion(con)
    return con


# ------------------------------------------------------------------
# Migración de `avisos` sobre una base de datos YA existente (con datos)
# ------------------------------------------------------------------
def _migrar_esquema_avisos(con: sqlite3.Connection) -> None:
    """
    Migración idempotente, pensada para correr en cada conexión sin que haga
    falta un paso manual aparte:
      1. Si la tabla `avisos` todavía no existe, no hace nada - la
         CREATE TABLE IF NOT EXISTS de inicializar_bd_produccion() más abajo
         ya trae el esquema final completo (columna + estado incluidos).
      2. Si existe pero le falta la columna `intentos_fallidos_detalle`
         (contador de fallos de scraping consecutivos, ver
         02_scraper_detalle_incremental.py), la agrega con
         ALTER TABLE ... ADD COLUMN - operación segura e inmediata en
         SQLite, no reescribe filas y aplica el DEFAULT a las ya existentes.
      3. Si el CHECK de `estado_publicacion` todavía no incluye
         'no_disponible' (estado nuevo para avisos que superaron el umbral
         de fallos persistentes, distinto de 'finalizado' = arriendo
         terminado con normalidad), reconstruye la tabla completa: SQLite no
         permite modificar un CHECK con ALTER TABLE, así que se crea
         `avisos_nuevo` con el esquema final, se copian todas las filas tal
         cual, y se reemplaza la tabla vieja - todo dentro de una
         transacción con foreign_keys desactivadas (procedimiento
         recomendado por SQLite para reconstrucciones de tabla), con
         rollback si algo falla a mitad de camino.
    """
    columnas = {fila[1] for fila in con.execute("PRAGMA table_info(avisos)").fetchall()}
    if not columnas:
        return

    if "intentos_fallidos_detalle" not in columnas:
        con.execute("ALTER TABLE avisos ADD COLUMN intentos_fallidos_detalle INTEGER DEFAULT 0")
        con.commit()

    definicion_actual = con.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='avisos'"
    ).fetchone()[0]
    if "no_disponible" in definicion_actual:
        return

    filas_antes = con.execute("SELECT COUNT(*) FROM avisos").fetchone()[0]

    con.execute("PRAGMA foreign_keys = OFF")
    try:
        con.execute("BEGIN")
        con.execute("""
            CREATE TABLE avisos_nuevo (
                id_aviso                    TEXT PRIMARY KEY,
                comuna                      TEXT NOT NULL,
                tipo_propiedad              TEXT NOT NULL,
                operacion                   TEXT NOT NULL,
                titulo                      TEXT,
                precio                      REAL,
                moneda                      TEXT,
                ubicacion                   TEXT,
                dormitorios                 INTEGER,
                banos                       INTEGER,
                superficie_m2               REAL,
                url                         TEXT,
                first_seen                  TEXT,
                estado_publicacion          TEXT NOT NULL DEFAULT 'activo'
                                             CHECK(estado_publicacion IN
                                                   ('activo', 'pausado', 'finalizado', 'no_disponible')),
                fecha_ultimo_chequeo_estado TEXT,
                intentos_fallidos_detalle   INTEGER DEFAULT 0
            )
        """)
        con.execute("INSERT INTO avisos_nuevo SELECT * FROM avisos")
        con.execute("DROP TABLE avisos")
        con.execute("ALTER TABLE avisos_nuevo RENAME TO avisos")
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise
    finally:
        con.execute("PRAGMA foreign_keys = ON")

    filas_despues = con.execute("SELECT COUNT(*) FROM avisos").fetchone()[0]
    if filas_despues != filas_antes:
        raise RuntimeError(
            f"Migración de `avisos` perdió filas: antes={filas_antes}, después={filas_despues}. "
            f"Revisa manualmente antes de seguir."
        )


def conectar_original(ruta_bd: Path = RUTA_BD_ORIGINAL) -> sqlite3.Connection:
    """Abre la base de datos ORIGINAL en modo solo-lectura (URI mode).
    Ningún script de 05_modelo_produccion/ debe escribir en esta base."""
    uri = f"file:{ruta_bd.as_posix()}?mode=ro"
    return sqlite3.connect(uri, uri=True)


# ------------------------------------------------------------------
# Esquema
# ------------------------------------------------------------------
def inicializar_bd_produccion(con: sqlite3.Connection) -> None:
    con.execute("""
        CREATE TABLE IF NOT EXISTS avisos (
            id_aviso                    TEXT PRIMARY KEY,
            comuna                      TEXT NOT NULL,
            tipo_propiedad              TEXT NOT NULL,
            operacion                   TEXT NOT NULL,
            titulo                      TEXT,
            precio                      REAL,
            moneda                      TEXT,
            ubicacion                   TEXT,
            dormitorios                 INTEGER,
            banos                       INTEGER,
            superficie_m2               REAL,
            url                         TEXT,
            first_seen                  TEXT,
            estado_publicacion          TEXT NOT NULL DEFAULT 'activo'
                                         CHECK(estado_publicacion IN
                                               ('activo', 'pausado', 'finalizado', 'no_disponible')),
            fecha_ultimo_chequeo_estado TEXT,
            intentos_fallidos_detalle   INTEGER DEFAULT 0
        )
    """)

    columnas_poi = ",\n            ".join(
        f"cantidad_{clave} INTEGER,\n            distancia_min_m_{clave} REAL"
        for clave in SUBCATEGORIAS_POI_COLUMNAS
    )

    con.execute(f"""
        CREATE TABLE IF NOT EXISTS avisos_detalle (
            id_aviso                       TEXT PRIMARY KEY REFERENCES avisos(id_aviso),
            descripcion                    TEXT,
            fecha_publicacion_texto        TEXT,
            fecha_publicacion_aprox        TEXT,
            superficie_total_m2            REAL,
            superficie_util_m2             REAL,
            dormitorios                    INTEGER,
            banos                          INTEGER,
            estacionamientos               INTEGER,
            antiguedad_anos                INTEGER,
            amoblado                       INTEGER,
            admite_mascotas                INTEGER,
            condominio_cerrado             INTEGER,
            bodegas                        INTEGER,
            gastos_comunes                 REAL,
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
            latitud                        REAL,
            longitud                       REAL,
            distancia_centro_comuna_m      REAL,
            distancia_centro_concepcion_m  REAL,
            {columnas_poi},
            uv_rsh                         TEXT,
            rank_nac                       REAL,
            pob_rsh_uv                     INTEGER,
            p_urbano                       REAL,
            fecha_scrapeo                  TEXT
        )
    """)

    con.execute("""
        CREATE TABLE IF NOT EXISTS predicciones (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            id_aviso          TEXT NOT NULL REFERENCES avisos(id_aviso),
            version_modelo    TEXT NOT NULL,
            fecha_prediccion  TEXT NOT NULL,
            precio_predicho   REAL NOT NULL,
            z_robusto         REAL,
            decil_precio      INTEGER,
            etiqueta          TEXT CHECK(etiqueta IN ('oportunidad', 'caro', 'precio_de_mercado')),
            nivel_confianza   TEXT CHECK(nivel_confianza IN ('alta confianza', 'confianza media', 'baja confianza')),
            cv_ensamble       REAL,
            UNIQUE(id_aviso, version_modelo)
        )
    """)

    con.execute("""
        CREATE TABLE IF NOT EXISTS corridas (
            id_corrida                 INTEGER PRIMARY KEY AUTOINCREMENT,
            fecha_inicio                TEXT NOT NULL,
            fecha_fin                   TEXT,
            resultado                   TEXT CHECK(resultado IN ('ok', 'error', 'parcial')),
            version_modelo_usada        TEXT,
            avisos_nuevos_grilla        INTEGER DEFAULT 0,
            avisos_nuevos_detalle       INTEGER DEFAULT 0,
            avisos_rechequeados         INTEGER DEFAULT 0,
            avisos_cambio_estado        INTEGER DEFAULT 0,
            paginas_recorridas_grilla   INTEGER DEFAULT 0,
            motivo_corte_grilla         TEXT CHECK(motivo_corte_grilla IN
                                         ('paginas_vacias_consecutivas', 'limite_paginas', 'limite_tiempo')),
            etapa_fallida                TEXT,
            mensaje_error                TEXT
        )
    """)

    con.execute("""
        CREATE TABLE IF NOT EXISTS logs_ejecucion (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            id_corrida  INTEGER REFERENCES corridas(id_corrida),
            timestamp   TEXT NOT NULL,
            etapa       TEXT CHECK(etapa IN
                        ('scraper_grilla', 'scraper_detalle', 'rechequeo_estado',
                         'vulnerabilidad', 'variables', 'prediccion', 'insercion_bd', 'orquestador')),
            nivel       TEXT CHECK(nivel IN ('info', 'warning', 'error')),
            mensaje     TEXT NOT NULL,
            detalle     TEXT
        )
    """)

    con.execute("""
        CREATE TABLE IF NOT EXISTS control (
            clave  TEXT PRIMARY KEY,
            valor  TEXT
        )
    """)

    con.commit()


# ------------------------------------------------------------------
# Helpers de `avisos.intentos_fallidos_detalle`
# (fallos persistentes de scraping ENTRE corridas - ver
# 02_scraper_detalle_incremental.py para el umbral y la lógica de cuándo
# marcar un aviso como 'no_disponible')
# ------------------------------------------------------------------
def incrementar_intentos_fallidos_detalle(con: sqlite3.Connection, id_aviso: str) -> int:
    """Suma 1 al contador de fallos consecutivos de este aviso y devuelve el nuevo valor."""
    con.execute(
        "UPDATE avisos SET intentos_fallidos_detalle = COALESCE(intentos_fallidos_detalle, 0) + 1 "
        "WHERE id_aviso = ?",
        (id_aviso,),
    )
    con.commit()
    fila = con.execute("SELECT intentos_fallidos_detalle FROM avisos WHERE id_aviso = ?", (id_aviso,)).fetchone()
    return fila[0] if fila else 0


def resetear_intentos_fallidos_detalle(con: sqlite3.Connection, id_aviso: str) -> None:
    """Vuelve el contador a 0 - se llama cuando un aviso finalmente se scrapea con éxito."""
    con.execute("UPDATE avisos SET intentos_fallidos_detalle = 0 WHERE id_aviso = ?", (id_aviso,))
    con.commit()


# ------------------------------------------------------------------
# Helpers de la tabla `control` (clave/valor genérico)
# ------------------------------------------------------------------
def leer_control(con: sqlite3.Connection, clave: str) -> str:
    fila = con.execute("SELECT valor FROM control WHERE clave = ?", (clave,)).fetchone()
    return fila[0] if fila else None


def escribir_control(con: sqlite3.Connection, clave: str, valor: str) -> None:
    con.execute(
        "INSERT INTO control (clave, valor) VALUES (?, ?) "
        "ON CONFLICT(clave) DO UPDATE SET valor = excluded.valor",
        (clave, valor),
    )
    con.commit()


if __name__ == "__main__":
    con = conectar_produccion()
    tablas = con.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").fetchall()
    print(f"Base de datos de producción inicializada en: {RUTA_BD_PRODUCCION}")
    print(f"Tablas: {[t[0] for t in tablas]}")
    con.close()
