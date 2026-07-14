"""Carga y join de datos desde la base de producción."""

import importlib.util
import sqlite3
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

SCRIPT_DIR = Path(__file__).resolve().parent
PRODUCCION_ROOT = SCRIPT_DIR.parent
REPO_ROOT = PRODUCCION_ROOT.parent
INVESTIGACION_ROOT = REPO_ROOT / "investigacion"
DB_PATH = PRODUCCION_ROOT / "01_modelo_produccion" / "produccion_gran_concepcion.db"
INGENIERIA_VARIABLES_PATH = INVESTIGACION_ROOT / "03_ingenieria_variables" / "01_ingenieria_variables.py"

AMENITY_COLUMNS = {
    "amoblado": "Amoblado",
    "piscina": "Piscina",
    "ascensor": "Ascensor",
    "estacionamientos": "Estacionamiento",
    "conserjeria": "Conserjería",
}

# Solo se excluyen definitivamente los avisos 'no_disponible' (dados de baja) y
# 'finalizado' (publicación cerrada, no es posible arrendarlo). 'activo' y 'pausado'
# se cargan ambos y se filtran en la UI, para no invalidar el cache al tocar ese filtro.
_QUERY = """
WITH ultima_prediccion AS (
    SELECT
        id_aviso,
        precio_predicho,
        z_robusto,
        etiqueta,
        nivel_confianza,
        ROW_NUMBER() OVER (
            PARTITION BY id_aviso ORDER BY fecha_prediccion DESC, id DESC
        ) AS rn
    FROM predicciones
)
SELECT
    a.id_aviso,
    a.titulo,
    a.precio,
    a.moneda,
    a.comuna,
    a.dormitorios,
    a.banos,
    a.url,
    a.estado_publicacion,
    a.fecha_ultimo_chequeo_estado,
    d.barrio,
    d.superficie_util_m2,
    d.gastos_comunes,
    d.antiguedad_anos,
    d.amoblado,
    d.piscina,
    d.ascensor,
    d.estacionamientos,
    d.conserjeria,
    d.latitud,
    d.longitud,
    d.fecha_publicacion_aprox,
    p.precio_predicho,
    p.z_robusto,
    p.etiqueta,
    p.nivel_confianza
FROM avisos a
JOIN avisos_detalle d ON a.id_aviso = d.id_aviso
JOIN ultima_prediccion p ON a.id_aviso = p.id_aviso AND p.rn = 1
WHERE a.tipo_propiedad = 'departamento'
  AND a.operacion = 'arriendo'
  AND a.estado_publicacion IN ('activo', 'pausado')
  AND d.latitud IS NOT NULL
  AND d.longitud IS NOT NULL
"""


def _cargar_ingenieria_variables():
    """Importa el módulo del pipeline con truco de import dinámico (mismo que
    usa 05_prediccion.py, porque el directorio empieza con dígitos y no es
    importable como paquete normal). De ahí se reutilizan convertir_precios_uf_a_clp
    y filtrar_precio_maximo: son las mismas funciones que estandarizan y acotan
    el precio real contra el que se compara la predicción, así que la vista no
    puede tener su propia lógica: tiene que ser literalmente esta."""
    nombre_modulo = "ingenieria_variables_original"
    # Cacheado en sys.modules: el archivo importa sklearn (RandomForestRegressor,
    # BallTree, etc.) en su top-level, y load_data() se re-ejecuta cada vez que
    # expira el cache de 10 min. Sin este cacheo, cada expiración repetía esos
    # imports y la ejecución completa del módulo en vez de reutilizar el ya cargado.
    if nombre_modulo in sys.modules:
        return sys.modules[nombre_modulo]
    spec = importlib.util.spec_from_file_location(nombre_modulo, INGENIERIA_VARIABLES_PATH)
    modulo = importlib.util.module_from_spec(spec)
    sys.modules[nombre_modulo] = modulo
    spec.loader.exec_module(modulo)
    return modulo


@st.cache_data(ttl=600, show_spinner="Cargando avisos...")
def load_data() -> pd.DataFrame:
    """Lee avisos de departamentos en arriendo con su predicción más reciente.

    Cacheada 10 min: la base la actualiza el orquestador en segundo plano,
    no en cada request del usuario.
    """
    # timeout alto: el orquestador puede estar escribiendo en la misma BD en
    # paralelo (una fila cada ~0.5s durante la etapa de predicción).
    conn = sqlite3.connect(DB_PATH, timeout=30)
    try:
        df = pd.read_sql_query(_QUERY, conn)
    finally:
        conn.close()

    if not df.empty:
        iv = _cargar_ingenieria_variables()
        df = iv.convertir_precios_uf_a_clp(df, ruta_bd=str(DB_PATH))
        df["precio"] = df["precio_clp"]
        # Avisos en UF cuya fecha no se pudo resolver ni con la API ni con el
        # respaldo (precio_clp queda en None): sin precio estandarizado no se
        # pueden mostrar ni filtrar de forma consistente con el resto.
        df = df.dropna(subset=["precio"])
        # Mismo tope que en entrenamiento y producción: sobre este monto ya no
        # son arriendos reales (ventas mal clasificadas, datos corruptos como
        # una moneda mal detectada), así que tampoco deberían aparecer acá.
        df = df[df["precio"] <= iv.PRECIO_MAXIMO_ARRIENDO_CLP]
        df = df.drop(columns=["precio_clp", "moneda"])

    for bool_col in ["amoblado", "piscina", "ascensor", "conserjeria"]:
        df[bool_col] = df[bool_col].fillna(0).astype(int).astype(bool)
    df["estacionamiento"] = df["estacionamientos"].fillna(0) > 0
    # Mismo criterio que usa el pipeline al armar el feature para el modelo
    # (ver 01_ingenieria_variables.py y 04_ingenieria_variables_produccion.py):
    # aviso sin gastos comunes informados -> 0, no NaN. Así el número que se
    # muestra acá es el mismo que vio el modelo al predecir ese aviso.
    df["gastos_comunes"] = df["gastos_comunes"].fillna(0)

    df["dormitorios"] = df["dormitorios"].fillna(0).astype(int)
    df["banos"] = df["banos"].fillna(0).astype(int)

    return df
