"""
Scraper de DETALLE incremental — pipeline de producción.

No duplica el parsing HTML/JSON: carga `01_obtener_datos/02_scraper_detalle.py`
como módulo (vía importlib) y reutiliza `extraer_detalle`,
`extraer_json_estado_pagina`, `calcular_distancias_centro`, `hay_captcha`,
`esperar_resolucion_manual`, `simular_comportamiento_humano`,
`construir_referer`, y sus constantes.

Dos responsabilidades, en la misma sesión de Playwright:
  1. Avisos NUEVOS de producción sin detalle todavía (LEFT JOIN, igual que
     el script original pero contra las tablas de producción).
  2. RE-CHEQUEO de avisos con estado_publicacion='activo' cuyo último
     chequeo tiene más de DIAS_MIN_ENTRE_RECHEQUEOS días, en batches de
     MAX_AVISOS_RECHEQUEO_POR_CORRIDA (los más antiguos primero).

Extrae además `estado_publicacion` (activo/pausado/finalizado) del mismo
JSON embebido que ya se usa para los puntos de interés, buscando el
componente item_status_message / item_status_short_description_message
dentro de components.head o components.short_description.

El guardado usa UPSERT (ON CONFLICT DO UPDATE) en vez de INSERT OR REPLACE:
así una re-visita nunca borra las columnas de vulnerabilidad
(uv_rsh/rank_nac/pob_rsh_uv/p_urbano) que llena la Etapa 5 aparte.
"""

import importlib.util
import logging
import random
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

import db

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
LIMITE_NUEVOS_POR_CORRIDA = 2000

DIAS_MIN_ENTRE_RECHEQUEOS = 7
MAX_AVISOS_RECHEQUEO_POR_CORRIDA = 50

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SCRAPER_DETALLE_ORIGINAL_PATH = REPO_ROOT / "01_obtener_datos" / "02_scraper_detalle.py"


def _cargar_modulo_scraper_detalle():
    spec = importlib.util.spec_from_file_location("scraper_detalle_original", SCRAPER_DETALLE_ORIGINAL_PATH)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


sd = _cargar_modulo_scraper_detalle()

MODO_MANUAL_CAPTCHA = sd.MODO_MANUAL_CAPTCHA
COOLDOWN_TRAS_CAPTCHA_MINUTOS = sd.COOLDOWN_TRAS_CAPTCHA_MINUTOS


# ------------------------------------------------------------------
# Extracción de estado_publicacion
# ------------------------------------------------------------------
CLAVES_ESTADO_PUBLICACION = ("item_status_message", "item_status_short_description_message")


def _buscar_nodo_por_clave(nodo, claves) -> dict:
    """Búsqueda recursiva del primer dict que contenga alguna de `claves`
    como llave propia, devolviendo su valor (el sub-dict del componente)."""
    if isinstance(nodo, dict):
        for clave in claves:
            if clave in nodo:
                return nodo[clave]
        for valor in nodo.values():
            resultado = _buscar_nodo_por_clave(valor, claves)
            if resultado is not None:
                return resultado
    elif isinstance(nodo, list):
        for item in nodo:
            resultado = _buscar_nodo_por_clave(item, claves)
            if resultado is not None:
                return resultado
    return None


def extraer_estado_publicacion(estado: dict) -> str:
    """
    Busca, dentro de components.head y components.short_description, un
    componente item_status_message / item_status_short_description_message.
    Si no aparece -> 'activo'. Si aparece, clasifica su body.text:
      - contiene "pausad"  -> 'pausado'
      - contiene "finaliz" -> 'finalizado'
      - texto no reconocido -> 'pausado' (por precaución: la sola presencia
        del componente ya señala que el aviso no está activo con normalidad),
        con un WARNING en el log para revisión manual.
    """
    if not estado or not isinstance(estado, dict):
        return "activo"

    components = estado.get("components") if isinstance(estado.get("components"), dict) else {}
    subarboles = [components.get("head"), components.get("short_description")]

    for subarbol in subarboles:
        if not subarbol:
            continue
        componente = _buscar_nodo_por_clave(subarbol, CLAVES_ESTADO_PUBLICACION)
        if not componente:
            continue

        texto = ""
        if isinstance(componente, dict):
            body = componente.get("body")
            if isinstance(body, dict):
                texto = (body.get("text") or "")

        texto_normalizado = texto.lower()
        if "pausad" in texto_normalizado:
            return "pausado"
        if "finaliz" in texto_normalizado:
            return "finalizado"

        log.warning(f"Componente de estado de publicación encontrado con texto no reconocido: "
                    f"'{texto}'. Se marca como 'pausado' por precaución.")
        return "pausado"

    return "activo"


# ------------------------------------------------------------------
# Conversión de tipos — misma semántica que
# 03_ingenieria_variables/01_ingenieria_variables.py sobre estos campos
# (pd.to_numeric directo, sin separador de miles chileno; "Sí"/"No" -> 1/0,
# ausente -> NULL, no 0 — la imputación queda para la etapa de variables).
# ------------------------------------------------------------------
CAMPOS_BOOLEANOS = [
    "amoblado", "admite_mascotas", "condominio_cerrado", "estacionamiento_visitas",
    "solo_familias", "piscina", "quincho", "conserjeria", "ascensor",
]
CAMPOS_ENTEROS = [
    "dormitorios", "banos", "estacionamientos", "antiguedad_anos",
    "bodegas", "max_habitantes", "piso_unidad", "deptos_por_piso",
]
CAMPOS_REALES = ["superficie_total_m2", "superficie_util_m2", "gastos_comunes", "latitud", "longitud"]


def _a_entero(valor):
    if valor is None:
        return None
    try:
        return int(float(valor))
    except (TypeError, ValueError):
        return None


def _a_real(valor):
    if valor is None:
        return None
    try:
        return float(valor)
    except (TypeError, ValueError):
        return None


def _a_booleano(valor):
    if valor is None:
        return None
    texto = str(valor).strip().lower()
    if texto in ("sí", "si"):
        return 1
    if texto == "no":
        return 0
    return None


def _convertir_datos(datos: dict) -> dict:
    convertido = dict(datos)
    for campo in CAMPOS_BOOLEANOS:
        convertido[campo] = _a_booleano(datos.get(campo))
    for campo in CAMPOS_ENTEROS:
        convertido[campo] = _a_entero(datos.get(campo))
    for campo in CAMPOS_REALES:
        convertido[campo] = _a_real(datos.get(campo))
    return convertido


# ------------------------------------------------------------------
# Consultas de pendientes
# ------------------------------------------------------------------
def obtener_pendientes_nuevos(con, limite: int = LIMITE_NUEVOS_POR_CORRIDA) -> pd.DataFrame:
    pendientes = pd.read_sql_query("""
        SELECT a.id_aviso, a.url, a.comuna, a.tipo_propiedad
        FROM avisos a
        LEFT JOIN avisos_detalle d ON a.id_aviso = d.id_aviso
        WHERE d.id_aviso IS NULL
    """, con)
    pendientes = pendientes.dropna(subset=["url"])
    return pendientes.head(limite)


def obtener_pendientes_rechequeo(
    con, dias_min: int = DIAS_MIN_ENTRE_RECHEQUEOS, batch: int = MAX_AVISOS_RECHEQUEO_POR_CORRIDA,
) -> pd.DataFrame:
    fecha_limite = (date.today() - timedelta(days=dias_min)).isoformat()
    pendientes = pd.read_sql_query("""
        SELECT id_aviso, url, comuna, tipo_propiedad, fecha_ultimo_chequeo_estado
        FROM avisos
        WHERE estado_publicacion = 'activo'
          AND fecha_ultimo_chequeo_estado IS NOT NULL
          AND fecha_ultimo_chequeo_estado <= ?
        ORDER BY fecha_ultimo_chequeo_estado ASC
        LIMIT ?
    """, con, params=(fecha_limite, batch))
    return pendientes.dropna(subset=["url"])


# ------------------------------------------------------------------
# Persistencia
# ------------------------------------------------------------------
def guardar_detalle_produccion(con, id_aviso: str, datos: dict) -> None:
    """
    UPSERT sobre avisos_detalle: solo toca las columnas que este scraper
    efectivamente escribe (nunca uv_rsh/rank_nac/pob_rsh_uv/p_urbano, que
    llena la etapa de vulnerabilidad aparte), para que un re-chequeo no
    borre esos datos ya resueltos.
    """
    columnas_base = [
        "descripcion", "fecha_publicacion_texto", "fecha_publicacion_aprox",
        "superficie_total_m2", "superficie_util_m2", "dormitorios", "banos", "estacionamientos",
        "antiguedad_anos", "amoblado", "admite_mascotas", "condominio_cerrado",
        "bodegas", "gastos_comunes", "estacionamiento_visitas", "solo_familias",
        "max_habitantes", "piscina", "quincho", "conserjeria", "ascensor",
        "piso_unidad", "deptos_por_piso", "barrio",
        "latitud", "longitud", "distancia_centro_comuna_m", "distancia_centro_concepcion_m",
    ]
    columnas_poi = []
    for clave in db.SUBCATEGORIAS_POI_COLUMNAS:
        columnas_poi.append(f"cantidad_{clave}")
        columnas_poi.append(f"distancia_min_m_{clave}")

    columnas_editables = columnas_base + columnas_poi + ["fecha_scrapeo"]
    datos_convertidos = _convertir_datos(datos)
    valores_editables = [datos_convertidos.get(c) for c in columnas_base] \
        + [datos_convertidos.get(c) for c in columnas_poi] \
        + [date.today().isoformat()]

    todas_las_columnas = ["id_aviso"] + columnas_editables
    todos_los_valores = [id_aviso] + valores_editables

    placeholders = ", ".join("?" for _ in todas_las_columnas)
    nombres_columnas = ", ".join(todas_las_columnas)
    actualizaciones = ", ".join(f"{c} = excluded.{c}" for c in columnas_editables)

    con.execute(f"""
        INSERT INTO avisos_detalle ({nombres_columnas})
        VALUES ({placeholders})
        ON CONFLICT(id_aviso) DO UPDATE SET {actualizaciones}
    """, todos_los_valores)
    con.commit()


def actualizar_estado_publicacion(con, id_aviso: str, estado_publicacion: str) -> None:
    con.execute("""
        UPDATE avisos
        SET estado_publicacion = ?, fecha_ultimo_chequeo_estado = ?
        WHERE id_aviso = ?
    """, (estado_publicacion, date.today().isoformat(), id_aviso))
    con.commit()


# ------------------------------------------------------------------
# Cooldown tras CAPTCHA (tabla `control`)
# ------------------------------------------------------------------
def registrar_captcha(con) -> None:
    db.escribir_control(con, "ultimo_captcha_detalle", datetime.now().isoformat())


def tiempo_restante_cooldown(con) -> timedelta:
    valor = db.leer_control(con, "ultimo_captcha_detalle")
    if not valor:
        return None
    ultimo_captcha = datetime.fromisoformat(valor)
    transcurrido = datetime.now() - ultimo_captcha
    cooldown_total = timedelta(minutes=COOLDOWN_TRAS_CAPTCHA_MINUTOS)
    return (cooldown_total - transcurrido) if transcurrido < cooldown_total else None


# ------------------------------------------------------------------
# Visita de un aviso (nuevo o re-chequeo)
# ------------------------------------------------------------------
def visitar_aviso(page, con, fila, es_rechequeo: bool) -> str:
    """Devuelve: 'ok', 'cambio_estado' (solo relevante en re-chequeo),
    'captcha' o 'error'."""
    id_aviso = fila["id_aviso"]
    url = fila["url"]
    referer = sd.construir_referer(fila["comuna"], fila["tipo_propiedad"])

    log.info(f"{'Re-chequeando' if es_rechequeo else 'Visitando'} {id_aviso}: {url}")

    try:
        page.goto(url, wait_until="domcontentloaded", timeout=30000, referer=referer)
        sd.simular_comportamiento_humano(page)
        page.wait_for_timeout(1000)
    except sd.PlaywrightTimeoutError:
        log.warning(f"Timeout cargando {url}. Se salta (queda pendiente para la próxima corrida).")
        return "error"
    except Exception as e:
        log.warning(f"Error navegando a {url}: {e}. Se salta.")
        return "error"

    if sd.hay_captcha(page):
        if MODO_MANUAL_CAPTCHA:
            resuelto = sd.esperar_resolucion_manual(page, url)
            if not resuelto:
                log.error("Sigue detectándose CAPTCHA después del intento manual. Deteniendo la corrida.")
                registrar_captcha(con)
                return "captcha"
        else:
            log.error(f"CAPTCHA detectado en {url}. Deteniendo la corrida ahora mismo. "
                      f"Cooldown de {COOLDOWN_TRAS_CAPTCHA_MINUTOS} minutos antes de la próxima.")
            registrar_captcha(con)
            return "captcha"

    datos = sd.extraer_detalle(page)
    estado_json = sd.extraer_json_estado_pagina(page)
    estado_publicacion = extraer_estado_publicacion(estado_json)
    distancias = sd.calcular_distancias_centro(datos.get("latitud"), datos.get("longitud"), fila["comuna"])
    datos.update(distancias)

    guardar_detalle_produccion(con, id_aviso, datos)
    actualizar_estado_publicacion(con, id_aviso, estado_publicacion)

    log.info(f"  -> Guardado. estado_publicacion={estado_publicacion}")

    time.sleep(random.uniform(sd.DELAY_MIN, sd.DELAY_MAX))

    if es_rechequeo and estado_publicacion != "activo":
        return "cambio_estado"
    return "ok"


# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------
def scrapear_detalle_incremental(con) -> dict:
    if not MODO_MANUAL_CAPTCHA:
        restante = tiempo_restante_cooldown(con)
        if restante:
            minutos = int(restante.total_seconds() // 60) + 1
            log.warning(f"En cooldown tras un CAPTCHA reciente. Faltan ~{minutos} minutos. "
                        f"No se hace nada en esta corrida.")
            return {
                "nuevos_procesados": 0, "rechequeos_procesados": 0,
                "cambios_estado": 0, "detenido_por_captcha": False,
            }
    else:
        log.info("MODO_MANUAL_CAPTCHA activado: navegador visible, cooldown desactivado.")

    if not sd.STEALTH_DISPONIBLE:
        log.warning("playwright-stealth no está instalado. Riesgo de bloqueo más alto.")

    pendientes_nuevos = obtener_pendientes_nuevos(con)
    pendientes_rechequeo = obtener_pendientes_rechequeo(con)

    log.info(f"{len(pendientes_nuevos)} avisos nuevos pendientes de detalle. "
              f"{len(pendientes_rechequeo)} avisos activos pendientes de re-chequeo.")

    if pendientes_nuevos.empty and pendientes_rechequeo.empty:
        log.info("Nada que hacer en esta corrida.")
        return {
            "nuevos_procesados": 0, "rechequeos_procesados": 0,
            "cambios_estado": 0, "detenido_por_captcha": False,
        }

    nuevos_procesados = 0
    rechequeos_procesados = 0
    cambios_estado = 0
    detenido_por_captcha = False

    with sd.sync_playwright() as p:
        browser = p.chromium.launch(headless=not MODO_MANUAL_CAPTCHA)
        context = browser.new_context(user_agent=sd.USER_AGENT, locale="es-CL")
        if sd.STEALTH_DISPONIBLE:
            sd.Stealth().apply_stealth_sync(context)
        page = context.new_page()

        for _, fila in pendientes_nuevos.iterrows():
            resultado = visitar_aviso(page, con, fila, es_rechequeo=False)
            if resultado == "captcha":
                detenido_por_captcha = True
                break
            if resultado == "ok":
                nuevos_procesados += 1

        if not detenido_por_captcha:
            for _, fila in pendientes_rechequeo.iterrows():
                resultado = visitar_aviso(page, con, fila, es_rechequeo=True)
                if resultado == "captcha":
                    detenido_por_captcha = True
                    break
                if resultado in ("ok", "cambio_estado"):
                    rechequeos_procesados += 1
                if resultado == "cambio_estado":
                    cambios_estado += 1

        browser.close()

    return {
        "nuevos_procesados": nuevos_procesados,
        "rechequeos_procesados": rechequeos_procesados,
        "cambios_estado": cambios_estado,
        "detenido_por_captcha": detenido_por_captcha,
    }


if __name__ == "__main__":
    con = db.conectar_produccion()
    resumen = scrapear_detalle_incremental(con)
    con.close()

    log.info(
        f"Corrida completa. Nuevos procesados: {resumen['nuevos_procesados']} | "
        f"Re-chequeados: {resumen['rechequeos_procesados']} | "
        f"Cambios de estado detectados: {resumen['cambios_estado']} | "
        f"Detenido por CAPTCHA: {resumen['detenido_por_captcha']}"
    )
