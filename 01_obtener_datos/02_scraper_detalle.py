import re
import json
import time
import random
import logging
import sqlite3
from pathlib import Path
from datetime import date, datetime, timedelta
from math import radians, sin, cos, sqrt, atan2
from typing import Optional

import pandas as pd
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError

try:
    from playwright_stealth import Stealth
    STEALTH_DISPONIBLE = True
except ImportError:
    STEALTH_DISPONIBLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ------------------------------------------------------------------
# CONFIGURACIÓN
# ------------------------------------------------------------------

# Las rutas se anclan a la carpeta donde vive ESTE archivo .py, sin importar
# desde qué directorio lo ejecutes (terminal en la raíz del proyecto, VS Code
# "Run", etc.). Esto evita que se creen bases de datos nuevas y vacías en
# lugares inesperados.
CARPETA_SCRIPT = Path(__file__).resolve().parent

# Una sola base de datos, compartida con el scraper de grilla. Este script
# solo LEE la tabla `avisos` y solo ESCRIBE en su propia tabla `avisos_detalle`.
BD_PRINCIPAL = CARPETA_SCRIPT / "avisos_gran_concepcion.db"

DELAY_MIN = 10.0   # segundos entre cada visita a un aviso individual
DELAY_MAX = 25.0

LIMITE_POR_CORRIDA = 1500   # avisos a procesar como máximo en UNA ejecución del script
                          # (usa cron para correrlo varias veces al día en vez de subir esto mucho)

COOLDOWN_TRAS_CAPTCHA_MINUTOS = 60   # tiempo mínimo de espera antes de reintentar tras un bloqueo

# MODO MANUAL: si el bloqueo automático persiste incluso con stealth/delays/referer,
# esta es la alternativa de respaldo. Ponla en True para correr el script en TU
# computador (con pantalla) en vez del servidor. El navegador se abre visible
# (headless=False) y, si aparece un CAPTCHA, el script se PAUSA y te espera a
# que lo resuelvas tú mismo en esa ventana antes de continuar.
MODO_MANUAL_CAPTCHA = False

BASE_URL = "https://www.portalinmobiliario.com"
OPERACION = "arriendo"

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)

# ------------------------------------------------------------------
# PATRONES DE EXTRACCIÓN
# Se aplican sobre el TEXTO COMPLETO de la página (no sobre selectores CSS
# específicos), buscando las etiquetas en español tal como las muestra el
# sitio. Esto es más resistente a cambios de clases/estructura HTML.
# ------------------------------------------------------------------
RE_FECHA_PUBLICACION = re.compile(r"Publicado hace ([^\n\|]+)", re.IGNORECASE)
RE_SUPERFICIE_TOTAL = re.compile(r"Superficie total\s*([\d.,]+)\s*m", re.IGNORECASE)
RE_SUPERFICIE_UTIL = re.compile(r"Superficie útil\s*([\d.,]+)\s*m", re.IGNORECASE)
RE_DORMITORIOS = re.compile(r"Dormitorios\s*(\d+)", re.IGNORECASE)
RE_BANOS = re.compile(r"Baños\s*(\d+)", re.IGNORECASE)
RE_ESTACIONAMIENTOS = re.compile(r"Estacionamientos:?\s*(\d+)", re.IGNORECASE)
RE_ANTIGUEDAD = re.compile(r"Antigüedad\s*(\d+)\s*años?", re.IGNORECASE)
RE_AMOBLADO = re.compile(r"Amoblado:?\s*(Sí|No)", re.IGNORECASE)
RE_ADMITE_MASCOTAS = re.compile(r"Admite mascotas:?\s*(Sí|No)", re.IGNORECASE)
RE_CONDOMINIO_CERRADO = re.compile(r"En condominio cerrado:?\s*(Sí|No)", re.IGNORECASE)

# Coordenadas: vienen embebidas en un bloque de JavaScript de configuración de
# la página (no son visibles como texto), por eso se buscan sobre el HTML
# completo (page.content()) en vez del texto visible (inner_text()). También
# aparecen repetidas en la URL de la imagen del mapa estático de Google, así
# que agregamos esa como respaldo por si la primera cambia de formato.
RE_LATITUD = re.compile(r'"latitude":"(-?[\d.]+)"')
RE_LONGITUD = re.compile(r'"longitude":"(-?[\d.]+)"')
RE_LATLON_MAPA = re.compile(r"center=(-?[\d.]+)%2C(-?[\d.]+)")

# Selectores candidatos para la descripción completa (probar en orden)
SELECTORES_DESCRIPCION = [
    "[data-testid='core-description'] p",
    "p.ui-pdp-description__content",
    "div.ui-pdp-description",
]

# ------------------------------------------------------------------
# PUNTOS DE INTERÉS (colegios, paraderos, áreas verdes, comercios, salud)
# ------------------------------------------------------------------

# Mapeo del nombre de subcategoría tal como lo muestra el sitio -> clave de
# columna (snake_case, sin tildes). Solo estas subcategorías se guardan; si el
# sitio agrega una nueva que no está en esta lista, se ignora silenciosamente
# (se loguea a nivel INFO) en vez de romper el script.
SUBCATEGORIAS_POI = {
    "paraderos": "paraderos",
    "estaciones de metro": "estaciones_metro",
    "jardines infantiles": "jardines_infantiles",
    "colegios": "colegios",
    "universidades": "universidades",
    "plazas": "plazas",
    "supermercados": "supermercados",
    "farmacias": "farmacias",
    "centros comerciales": "centros_comerciales",
    "hospitales": "hospitales",
    "clinicas": "clinicas",
}

# Coordenadas aproximadas del centro (plaza principal / casco histórico) de
# cada comuna del Gran Concepción. Son APROXIMADAS - ajústalas si tienes
# coordenadas más precisas. Las claves coinciden con la columna `comuna` de
# la tabla `avisos` (mismo slug que usa 01_scraper_grilla.py).
COMUNA_CENTROS = {
    "concepcion-biobio":          (-36.8265, -73.0524),  # Plaza de la Independencia
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

# Referencia fija para "distancia al centro de Concepción" (independiente de
# en qué comuna esté el aviso) - mismo punto que el centro de la comuna
# "concepcion-biobio".
CENTRO_CONCEPCION = COMUNA_CENTROS["concepcion-biobio"]


# ------------------------------------------------------------------
# BASE DE DATOS
# ------------------------------------------------------------------

def inicializar_bd(ruta_bd: Path = BD_PRINCIPAL) -> sqlite3.Connection:
    """
    Abre la base de datos principal (la misma que usa el scraper de grilla)
    y crea la tabla `avisos_detalle` si no existe. No toca la tabla `avisos`.
    """
    con = sqlite3.connect(ruta_bd)

    columnas_poi = ",\n            ".join(
        f"cantidad_{clave} INTEGER,\n            distancia_min_m_{clave} REAL"
        for clave in SUBCATEGORIAS_POI.values()
    )

    con.execute(f"""
        CREATE TABLE IF NOT EXISTS avisos_detalle (
            id_aviso                    TEXT PRIMARY KEY,
            url                         TEXT,
            descripcion                 TEXT,
            fecha_publicacion_texto     TEXT,
            fecha_publicacion_aprox     TEXT,
            superficie_total_m2         TEXT,
            superficie_util_m2          TEXT,
            dormitorios                 TEXT,
            banos                       TEXT,
            estacionamientos            TEXT,
            antiguedad_anos             TEXT,
            amoblado                    TEXT,
            admite_mascotas             TEXT,
            condominio_cerrado          TEXT,
            latitud                     TEXT,
            longitud                    TEXT,
            distancia_centro_comuna_m       REAL,
            distancia_centro_concepcion_m   REAL,
            {columnas_poi},
            fecha_scrapeo               TEXT
        )
    """)
    con.commit()
    return con


def obtener_pendientes(con: sqlite3.Connection, limite: int = LIMITE_POR_CORRIDA) -> pd.DataFrame:
    """
    Devuelve los avisos de la tabla `avisos` que todavía no tienen fila en
    `avisos_detalle` (misma base de datos, un solo JOIN), hasta un máximo de
    `limite` filas por corrida.
    """
    pendientes = pd.read_sql_query("""
        SELECT a.id_aviso, a.url, a.comuna, a.tipo_propiedad
        FROM avisos a
        LEFT JOIN avisos_detalle d ON a.id_aviso = d.id_aviso
        WHERE d.id_aviso IS NULL
    """, con)

    pendientes = pendientes.dropna(subset=["url"])
    return pendientes.head(limite)


def guardar_detalle(con: sqlite3.Connection, id_aviso: str, url: str, datos: dict) -> None:
    """Inserta un aviso de detalle y hace commit inmediatamente (guardado incremental)."""
    columnas_base = [
        "id_aviso", "url", "descripcion", "fecha_publicacion_texto", "fecha_publicacion_aprox",
        "superficie_total_m2", "superficie_util_m2", "dormitorios", "banos", "estacionamientos",
        "antiguedad_anos", "amoblado", "admite_mascotas", "condominio_cerrado",
        "latitud", "longitud", "distancia_centro_comuna_m", "distancia_centro_concepcion_m",
    ]
    columnas_poi = []
    for clave in SUBCATEGORIAS_POI.values():
        columnas_poi.append(f"cantidad_{clave}")
        columnas_poi.append(f"distancia_min_m_{clave}")

    todas_las_columnas = columnas_base + columnas_poi + ["fecha_scrapeo"]
    valores = [id_aviso, url] + [datos.get(c) for c in columnas_base[2:]] \
              + [datos.get(c) for c in columnas_poi] + [date.today().isoformat()]

    placeholders = ", ".join("?" for _ in todas_las_columnas)
    nombres_columnas = ", ".join(todas_las_columnas)

    con.execute(f"""
        INSERT OR REPLACE INTO avisos_detalle ({nombres_columnas})
        VALUES ({placeholders})
    """, valores)
    con.commit()   # <- commit por cada aviso, no al final. Así no se pierde nada si se corta.


def registrar_captcha(con: sqlite3.Connection) -> None:
    """Anota la hora del bloqueo, para activar el cooldown."""
    con.execute("CREATE TABLE IF NOT EXISTS estado (clave TEXT PRIMARY KEY, valor TEXT)")
    con.execute(
        "INSERT OR REPLACE INTO estado (clave, valor) VALUES ('ultimo_captcha', ?)",
        (datetime.now().isoformat(),),
    )
    con.commit()


def tiempo_restante_cooldown(con: sqlite3.Connection) -> Optional[timedelta]:
    """
    Si hubo un CAPTCHA reciente, devuelve cuánto falta para que se acabe el
    cooldown. Devuelve None si no hay cooldown activo (se puede correr).
    """
    con.execute("CREATE TABLE IF NOT EXISTS estado (clave TEXT PRIMARY KEY, valor TEXT)")
    cur = con.execute("SELECT valor FROM estado WHERE clave = 'ultimo_captcha'")
    fila = cur.fetchone()

    if not fila:
        return None

    ultimo_captcha = datetime.fromisoformat(fila[0])
    transcurrido = datetime.now() - ultimo_captcha
    cooldown_total = timedelta(minutes=COOLDOWN_TRAS_CAPTCHA_MINUTOS)

    if transcurrido < cooldown_total:
        return cooldown_total - transcurrido
    return None


# ------------------------------------------------------------------
# EXTRACCIÓN
# ------------------------------------------------------------------

def parsear_fecha_relativa(texto: Optional[str]) -> Optional[str]:
    """
    Convierte 'hace 3 meses' -> fecha aproximada (YYYY-MM-DD).
    Es una aproximación (asume 30 días por mes, 365 por año) ya que el sitio
    no muestra la fecha exacta en la página de detalle.
    """
    if not texto:
        return None

    m = re.search(r"(\d+)\s*(día|dias|semana|mes|meses|año|años)", texto, re.IGNORECASE)
    if not m:
        return None

    cantidad = int(m.group(1))
    unidad = m.group(2).lower()

    if "día" in unidad or "dia" in unidad:
        delta = timedelta(days=cantidad)
    elif "semana" in unidad:
        delta = timedelta(weeks=cantidad)
    elif "mes" in unidad:
        delta = timedelta(days=30 * cantidad)
    elif "año" in unidad:
        delta = timedelta(days=365 * cantidad)
    else:
        return None

    return (date.today() - delta).isoformat()


def extraer_descripcion(page) -> Optional[str]:
    for selector in SELECTORES_DESCRIPCION:
        loc = page.locator(selector)
        if loc.count() > 0:
            try:
                return loc.first.inner_text().strip()
            except Exception:
                continue
    return None


def extraer_coordenadas(page) -> dict:
    """
    Las coordenadas no son texto visible: vienen embebidas en un bloque de
    JavaScript de configuración de la página. Por eso se buscan sobre el
    HTML completo (page.content()), no sobre el texto visible.
    Nota: suelen ser una ubicación aproximada del sector, no la dirección
    exacta de la propiedad (por privacidad/seguridad del arrendador).
    """
    html_completo = page.content()

    m_lat = RE_LATITUD.search(html_completo)
    m_lon = RE_LONGITUD.search(html_completo)

    if m_lat and m_lon:
        return {"latitud": m_lat.group(1), "longitud": m_lon.group(1)}

    # Respaldo: si el bloque JSON cambia de formato, intentar sacarlas de la
    # URL del mapa estático de Google (center=lat%2Clon)
    m_mapa = RE_LATLON_MAPA.search(html_completo)
    if m_mapa:
        return {"latitud": m_mapa.group(1), "longitud": m_mapa.group(2)}

    return {"latitud": None, "longitud": None}


def normalizar_texto(texto: str) -> str:
    """Minúsculas y sin tildes, para comparar nombres de subcategoría sin depender de mayúsculas/acentos."""
    reemplazos = {"á": "a", "é": "e", "í": "i", "ó": "o", "ú": "u", "ñ": "n"}
    texto = texto.lower().strip()
    for con_tilde, sin_tilde in reemplazos.items():
        texto = texto.replace(con_tilde, sin_tilde)
    return texto


def parsear_distancia_metros(texto: str) -> Optional[float]:
    """
    Convierte '0 mins - 15 metros' -> 15.0, o '12 mins - 1.2 km' -> 1200.0.
    Devuelve None si no logra encontrar un número reconocible.
    """
    m = re.search(r"([\d.,]+)\s*(metros|km)", texto, re.IGNORECASE)
    if not m:
        return None

    valor = float(m.group(1).replace(".", "").replace(",", "."))  # maneja separador de miles chileno
    unidad = m.group(2).lower()

    return valor * 1000 if unidad == "km" else valor


def extraer_json_estado_pagina(page) -> Optional[dict]:
    """
    Extrae y parsea el bloque de configuración embebido en `window._n.ctx.r`.
    Este JSON contiene TODOS los datos de la página, incluidas las categorías
    de puntos de interés de las pestañas que no están visualmente activas -
    el HTML renderizado solo trae el contenido de la pestaña seleccionada por
    defecto (Transporte), pero este JSON las trae todas completas.
    """
    html_completo = page.content()
    m = re.search(r"_n\.ctx\.r=(\{.*?\});_n\.ctx\.r\.assets\.manifest=", html_completo, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        return None


def _buscar_categorias_poi(nodo) -> Optional[list]:
    """
    Búsqueda recursiva dentro del JSON de estado de la lista de categorías de
    puntos de interés (Transporte, Educación, Áreas verdes, Comercios, Salud).
    Se busca por forma (una lista de dicts con clave 'subcategories') en vez
    de por una ruta fija, para no depender de la ubicación exacta dentro del
    JSON si el sitio reorganiza su estructura interna.
    """
    if isinstance(nodo, dict):
        categorias = nodo.get("categories")
        if isinstance(categorias, list) and categorias and isinstance(categorias[0], dict) \
                and "subcategories" in categorias[0]:
            return categorias
        for valor in nodo.values():
            resultado = _buscar_categorias_poi(valor)
            if resultado:
                return resultado
    elif isinstance(nodo, list):
        for item in nodo:
            resultado = _buscar_categorias_poi(item)
            if resultado:
                return resultado
    return None


def extraer_puntos_interes(page) -> dict:
    """
    Extrae, para cada subcategoría conocida (colegios, paraderos, plazas,
    etc.), cuántos hay y la distancia del más cercano. Lee el JSON embebido
    en la página (ver extraer_json_estado_pagina) en vez del HTML visible,
    porque el DOM solo renderiza la pestaña activa por defecto.
    """
    resultado = {}
    for clave in SUBCATEGORIAS_POI.values():
        resultado[f"cantidad_{clave}"] = None
        resultado[f"distancia_min_m_{clave}"] = None

    estado = extraer_json_estado_pagina(page)
    if not estado:
        return resultado

    categorias = _buscar_categorias_poi(estado)
    if not categorias:
        return resultado

    for categoria in categorias:
        for subcategoria in categoria.get("subcategories", []):
            nombre = normalizar_texto(subcategoria.get("title", {}).get("text", ""))
            clave = SUBCATEGORIAS_POI.get(nombre)

            if not clave:
                log.info(f"Subcategoría de POI no reconocida (se ignora): '{nombre}'")
                continue

            items = subcategoria.get("items", [])
            cantidad = len(items)

            distancia_min = None
            if items:
                texto_subtitulo = items[0].get("subtitle", {}).get("label", {}).get("text", "")
                distancia_min = parsear_distancia_metros(texto_subtitulo)

            resultado[f"cantidad_{clave}"] = cantidad
            resultado[f"distancia_min_m_{clave}"] = distancia_min

    return resultado


def haversine_metros(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distancia en línea recta (metros) entre dos puntos geográficos."""
    radio_tierra_m = 6371000
    phi1, phi2 = radians(lat1), radians(lat2)
    delta_phi = radians(lat2 - lat1)
    delta_lambda = radians(lon2 - lon1)

    a = sin(delta_phi / 2) ** 2 + cos(phi1) * cos(phi2) * sin(delta_lambda / 2) ** 2
    return 2 * radio_tierra_m * atan2(sqrt(a), sqrt(1 - a))


def calcular_distancias_centro(latitud: Optional[str], longitud: Optional[str], comuna: str) -> dict:
    """
    Distancia en línea recta desde el aviso hasta el centro de su comuna, y
    hasta el centro de Concepción. None si no hay coordenadas del aviso o si
    la comuna no está en COMUNA_CENTROS.
    """
    resultado = {"distancia_centro_comuna_m": None, "distancia_centro_concepcion_m": None}

    if not latitud or not longitud:
        return resultado

    try:
        lat_aviso, lon_aviso = float(latitud), float(longitud)
    except (TypeError, ValueError):
        return resultado

    centro_comuna = COMUNA_CENTROS.get(comuna)
    if centro_comuna:
        resultado["distancia_centro_comuna_m"] = round(
            haversine_metros(lat_aviso, lon_aviso, *centro_comuna), 1
        )

    resultado["distancia_centro_concepcion_m"] = round(
        haversine_metros(lat_aviso, lon_aviso, *CENTRO_CONCEPCION), 1
    )

    return resultado


def extraer_detalle(page) -> dict:
    texto_completo = page.locator("body").inner_text()

    def buscar(patron):
        m = patron.search(texto_completo)
        return m.group(1).strip() if m else None

    fecha_texto = buscar(RE_FECHA_PUBLICACION)
    coordenadas = extraer_coordenadas(page)
    puntos_interes = extraer_puntos_interes(page)

    return {
        "descripcion": extraer_descripcion(page),
        "fecha_publicacion_texto": fecha_texto,
        "fecha_publicacion_aprox": parsear_fecha_relativa(fecha_texto),
        "superficie_total_m2": buscar(RE_SUPERFICIE_TOTAL),
        "superficie_util_m2": buscar(RE_SUPERFICIE_UTIL),
        "dormitorios": buscar(RE_DORMITORIOS),
        "banos": buscar(RE_BANOS),
        "estacionamientos": buscar(RE_ESTACIONAMIENTOS),
        "antiguedad_anos": buscar(RE_ANTIGUEDAD),
        "amoblado": buscar(RE_AMOBLADO),
        "admite_mascotas": buscar(RE_ADMITE_MASCOTAS),
        "condominio_cerrado": buscar(RE_CONDOMINIO_CERRADO),
        "latitud": coordenadas["latitud"],
        "longitud": coordenadas["longitud"],
        **puntos_interes,
    }


def hay_captcha(page) -> bool:
    """
    Detecta un bloqueo real, no solo la mención de la palabra "captcha".
    Muchos sitios (incluido este) incluyen el script de Google reCAPTCHA de
    forma permanente e invisible en casi todas sus páginas como medida
    preventiva estándar, sin que eso signifique que te están desafiando
    activamente. Por eso exigimos DOS condiciones:
      1) La palabra "captcha" aparece en el HTML, Y
      2) El contenido normal del aviso (dormitorios/superficie) NO logró
         cargar - señal de que la página real fue efectivamente bloqueada.
    Si el contenido normal sí está ahí, asumimos que es solo el script de
    fondo y NO es un bloqueo real.
    """
    contenido = page.content().lower()
    if "captcha" not in contenido[:8000]:
        return False

    texto = page.locator("body").inner_text()
    parece_contenido_real = bool(
        RE_SUPERFICIE_TOTAL.search(texto)
        or RE_SUPERFICIE_UTIL.search(texto)
        or RE_DORMITORIOS.search(texto)
    )

    return not parece_contenido_real


def esperar_resolucion_manual(page, url: str) -> bool:
    """
    Pausa la ejecución y espera a que la persona resuelva el CAPTCHA a mano
    en la ventana visible del navegador. Devuelve True si al continuar ya no
    se detecta CAPTCHA, False si sigue bloqueado.
    """
    print("\n" + "=" * 70)
    print(f"CAPTCHA detectado en: {url}")
    print("Resuélvelo en la ventana del navegador que se abrió.")
    input("Cuando hayas terminado, vuelve aquí y presiona Enter para continuar...")
    print("=" * 70 + "\n")

    try:
        page.reload(wait_until="domcontentloaded", timeout=30000)
        simular_comportamiento_humano(page)
    except Exception as e:
        log.warning(f"Error al recargar tras resolver el CAPTCHA: {e}")

    return not hay_captcha(page)


def simular_comportamiento_humano(page) -> None:
    """
    Pequeño scroll aleatorio antes de extraer datos. Ayuda a que el contenido
    dinámico termine de cargar (lazy-load) y se parece más a la navegación
    real de una persona que a un script que lee el HTML apenas carga.
    """
    try:
        for _ in range(random.randint(2, 4)):
            distancia = random.randint(200, 800)
            page.mouse.wheel(0, distancia)
            page.wait_for_timeout(random.randint(300, 900))
    except Exception:
        pass  # si falla el scroll simulado, no es crítico, seguimos igual


def construir_referer(comuna: str, tipo_propiedad: str) -> str:
    """
    URL de la página de búsqueda correspondiente, usada como 'referer' al
    visitar el detalle - simula que se llegó navegando desde ahí, como
    haría una persona real, en vez de entrar directo a la URL del aviso.
    """
    return f"{BASE_URL}/{OPERACION}/{tipo_propiedad}/{comuna}"


# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------

def main():
    con = inicializar_bd()

    if not MODO_MANUAL_CAPTCHA:
        restante = tiempo_restante_cooldown(con)
        if restante:
            minutos = int(restante.total_seconds() // 60) + 1
            log.warning(f"En cooldown tras un CAPTCHA reciente. Faltan ~{minutos} minutos. "
                        f"No se hace nada en esta corrida - vuelve a intentarlo después.")
            con.close()
            return
    else:
        log.info("MODO_MANUAL_CAPTCHA activado: navegador visible, cooldown desactivado "
                 "(se asume que estás supervisando la corrida).")

    if not STEALTH_DISPONIBLE:
        log.warning("playwright-stealth no está instalado. El scraper funcionará igual, pero con más "
                    "riesgo de bloqueo. Considera instalarlo: pip install playwright-stealth")

    pendientes = obtener_pendientes(con)

    log.info(f"{len(pendientes)} avisos a procesar en esta corrida "
              f"(límite configurado: {LIMITE_POR_CORRIDA}).")

    if pendientes.empty:
        log.info("No hay avisos pendientes. Nada que hacer.")
        con.close()
        return

    procesados = 0
    detenido_por_captcha = False

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=not MODO_MANUAL_CAPTCHA)
        context = browser.new_context(user_agent=USER_AGENT, locale="es-CL")

        if STEALTH_DISPONIBLE:
            Stealth().apply_stealth_sync(context)

        page = context.new_page()

        for _, fila in pendientes.iterrows():
            id_aviso = fila["id_aviso"]
            url = fila["url"]
            referer = construir_referer(fila["comuna"], fila["tipo_propiedad"])

            log.info(f"Visitando {id_aviso}: {url}")

            try:
                page.goto(url, wait_until="domcontentloaded", timeout=30000, referer=referer)
                simular_comportamiento_humano(page)
                page.wait_for_timeout(1000)
            except PlaywrightTimeoutError:
                log.warning(f"Timeout cargando {url}. Se salta este aviso (queda pendiente para la próxima corrida).")
                continue
            except Exception as e:
                log.warning(f"Error navegando a {url}: {e}. Se salta este aviso.")
                continue

            if hay_captcha(page):
                if MODO_MANUAL_CAPTCHA:
                    resuelto = esperar_resolucion_manual(page, url)
                    if not resuelto:
                        log.error("Sigue detectándose CAPTCHA después de tu intento. "
                                  "Deteniendo la corrida por esta vez.")
                        registrar_captcha(con)
                        detenido_por_captcha = True
                        break
                    # Ya resuelto - seguimos y extraemos este mismo aviso normalmente
                else:
                    log.error(f"CAPTCHA detectado en {url}. Deteniendo la corrida ahora mismo. "
                              f"Lo ya procesado ({procesados} avisos) queda guardado. "
                              f"Se activa un cooldown de {COOLDOWN_TRAS_CAPTCHA_MINUTOS} minutos "
                              f"antes de permitir la siguiente corrida.")
                    registrar_captcha(con)
                    detenido_por_captcha = True
                    break

            datos = extraer_detalle(page)
            distancias = calcular_distancias_centro(
                datos.get("latitud"), datos.get("longitud"), fila["comuna"]
            )
            datos.update(distancias)

            guardar_detalle(con, id_aviso, url, datos)
            procesados += 1

            log.info(f"  -> Guardado ({procesados}/{len(pendientes)})")

            time.sleep(random.uniform(DELAY_MIN, DELAY_MAX))

        browser.close()

    con.close()

    if detenido_por_captcha:
        log.info(f"Corrida detenida por CAPTCHA. Avisos procesados en esta corrida: {procesados}.")
    else:
        log.info(f"Corrida completa. Avisos procesados en esta corrida: {procesados}.")


if __name__ == "__main__":
    main()