"""
Scraper de DETALLE - Portal Inmobiliario (Gran Concepción)

Complementa al scraper de grilla (01_scraper_grilla.py). Este script:
  1. Lee los avisos ya descubiertos en avisos_gran_concepcion.db (tabla `avisos`).
  2. Visita cada URL individual con requests (ruta principal, sin navegador)
     y extrae: descripción completa, fecha de publicación, y características
     principales del inmueble.
  3. Guarda cada aviso INMEDIATAMENTE (commit por aviso) en la MISMA base de
     datos, en una tabla nueva: `avisos_detalle` (dentro de avisos_gran_concepcion.db).
  4. Es reanudable: si se corta a mitad de camino (bloqueo, corte de luz,
     Ctrl+C, etc.), la próxima vez que lo corras retoma solo los avisos que
     todavía no tienen detalle guardado - no vuelve a visitar los que ya
     procesó ni pierde lo avanzado.

RUTA PRINCIPAL: requests (sin navegador)
    pip install requests beautifulsoup4 lxml pandas
    Se confirmó con pruebas (6 URLs comparadas 1:1 contra Playwright, y una
    corrida de volumen de 150 URLs seguidas) que requests devuelve el mismo
    JSON embebido y el mismo texto de características que Playwright, sin
    señales de bloqueo ni degradación con volumen sostenido (ver
    01_obtener_datos/pruebas/). Por eso ya no se necesita un navegador para
    la ruta normal de este script, lo que evita la dependencia pesada de
    Playwright en entornos como GitHub Actions.

RUTA DE RESPALDO: Playwright (NO es la ruta principal)
    Si en algún momento el sitio empezara a bloquear las requests simples de
    forma persistente (no observado en las pruebas), o si necesitas resolver
    un CAPTCHA a mano, existe `main_fallback_playwright()` como red de
    seguridad. Es un camino aparte, no se ejecuta automáticamente. Para
    usarlo:
        pip install playwright playwright-stealth
        playwright install chromium
        python 02_scraper_detalle.py --fallback-playwright
    El import de Playwright es perezoso (ocurre DENTRO de esa función), así
    que el resto del script - incluida la ruta principal - funciona sin
    problema en un entorno donde Playwright no está instalado.

CÓMO CORRERLO EN UN SERVIDOR:
- Este script está pensado para correr en tandas pequeñas vía cron (ver
  LIMITE_POR_CORRIDA más abajo), NO para intentar procesar miles de avisos
  de una sola sentada. Ejemplo de cron para correr cada 2 horas:
      0 */2 * * * cd /ruta/al/proyecto/01_obtener_datos && /ruta/al/python 02_scraper_detalle.py
- Si el sitio te bloquea con CAPTCHA, el script queda en "cooldown" (ver
  COOLDOWN_TRAS_CAPTCHA_MINUTOS) y las siguientes corridas se saltan solas
  hasta que pase ese tiempo - no hace falta que tú lo controles manualmente.

NOTAS IMPORTANTES:
- Este scraper es más "sensible" que el de grilla: entra a UNA página por
  cada aviso, en vez de una página que lista 48 de una vez. Eso significa
  muchas más visitas en total, por lo que el riesgo de bloqueo es mayor.
- Si el sitio muestra CAPTCHA, el script se DETIENE de inmediato dejando
  guardado todo lo que alcanzó a procesar, y activa un cooldown antes de
  permitir la siguiente corrida.
- Un 404 (u otro error) aislado puede ser transitorio: cada aviso se
  reintenta automáticamente (ver REINTENTOS_TRAS_ERROR) antes de darlo por
  fallido EN ESTA corrida. El manejo de fallos persistentes ENTRE corridas
  (para no reintentar para siempre un aviso realmente eliminado) vive en
  05_modelo_produccion/02_scraper_detalle_incremental.py, vía el contador
  `intentos_fallidos_detalle` - este script original (de exploración inicial,
  sin ese contador) simplemente deja el aviso pendiente para la próxima vez.
- Igual que con el otro scraper: revisa el robots.txt / Términos de Uso antes
  de correrlo a gran escala, y no reproduzcas ni redistribuyas contenido con
  derechos de terceros (fotos, descripciones completas de otros) sin permiso.
- Los selectores CSS y el texto en español pueden cambiar con el tiempo. Si
  el script deja de traer datos, revisa la sección SELECTORES y los patrones
  RE_* más abajo.
- Este script solo LEE de la tabla `avisos` (nunca la modifica). Solo escribe
  en sus propias tablas `avisos_detalle` y `estado`.
"""

import re
import sys
import json
import time
import random
import logging
import sqlite3
from pathlib import Path
from datetime import date, datetime, timedelta
from math import radians, sin, cos, sqrt, atan2
from typing import Optional

import requests
import pandas as pd
from bs4 import BeautifulSoup

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

DELAY_MIN = 2.0   # segundos entre cada visita a un aviso individual
DELAY_MAX = 4.0

LIMITE_POR_CORRIDA = 2000   # avisos a procesar como máximo en UNA ejecución del script
                          # (usa cron para correrlo varias veces al día en vez de subir esto mucho)

COOLDOWN_TRAS_CAPTCHA_MINUTOS = 60   # tiempo mínimo de espera antes de reintentar tras un bloqueo

# Reintentos ante un fallo de red/HTTP (ej. 404) al visitar UN aviso, dentro
# de la MISMA corrida - un fallo aislado puede ser transitorio y no
# reproducirse al reintentar segundos después. No confundir con el
# manejo de fallos PERSISTENTES entre corridas (eso vive en
# 05_modelo_produccion/02_scraper_detalle_incremental.py).
REINTENTOS_TRAS_ERROR = 2          # reintentos adicionales tras el primer intento (total = 1 + este valor)
BACKOFF_REINTENTO_MIN = 3.0        # segundos de espera antes de cada reintento
BACKOFF_REINTENTO_MAX = 6.0

TIMEOUT_REQUEST_SEG = 15

# MODO MANUAL: si el bloqueo automático persiste incluso con delays/referer,
# esta es la alternativa de respaldo. Ponla en True para usar
# main_fallback_playwright() con un navegador visible en TU computador (con
# pantalla) en vez del servidor. Si aparece un CAPTCHA, el script se PAUSA y
# te espera a que lo resuelvas tú mismo en esa ventana antes de continuar.
# La ruta principal (requests) no puede mostrarte una ventana de navegador -
# este modo solo tiene efecto dentro de main_fallback_playwright().
MODO_MANUAL_CAPTCHA = False

BASE_URL = "https://www.portalinmobiliario.com"
OPERACION = "arriendo"

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)

# Headers para la ruta principal (requests) - mismo criterio que
# 01_scraper_grilla.py (HEADERS), que ya está validado en producción para
# este sitio. El Referer se arma aparte por request (ver construir_referer).
HEADERS_BASE = {
    "User-Agent": USER_AGENT,
    "Accept-Language": "es-CL,es;q=0.9,en;q=0.8",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
}


def headers_requests(referer: str) -> dict:
    return {**HEADERS_BASE, "Referer": referer}

# ------------------------------------------------------------------
# PATRONES DE EXTRACCIÓN
# Se aplican sobre el TEXTO COMPLETO de la página (no sobre selectores CSS
# específicos), buscando las etiquetas en español tal como las muestra el
# sitio. Esto es más resistente a cambios de clases/estructura HTML.
# ------------------------------------------------------------------
RE_FECHA_PUBLICACION = re.compile(r"Publicado hace ([^\n\|]+)", re.IGNORECASE)
RE_SUPERFICIE_TOTAL = re.compile(r"Superficie total\s*([\d.,]+)\s*m", re.IGNORECASE)
RE_SUPERFICIE_UTIL = re.compile(r"Superficie útil\s*([\d.,]+)\s*m", re.IGNORECASE)

# Sin IGNORECASE a propósito: el carrusel de "recomendados" de la página trae
# tarjetas de OTROS avisos con "N dormitorios" en minúscula (ej. "3
# dormitorios\n2 baños\n84 m² útiles"), mientras que la sección de
# características del aviso actual trae "Dormitorios" con mayúscula DESPUÉS
# del número (ej. "Dormitorios\n3"). Matchear solo "Dormitorios" con D
# mayúscula evita que re.search capture, desde una tarjeta del carrusel, un
# número que no corresponde al aviso actual.
RE_DORMITORIOS = re.compile(r"Dormitorios\s*(\d+)")

# Sin IGNORECASE a propósito: la insignia superior de la página trae "N
# baños" en minúscula ANTES del número (ej. "2 baños\n75 m² totales"),
# mientras que la sección de características trae "Baños" con mayúscula
# DESPUÉS del número (ej. "Baños\n2"). Matchear solo "Baños" con B mayúscula
# evita que re.search capture, desde la insignia, la superficie total en
# vez de la cantidad real de baños.
RE_BANOS = re.compile(r"Baños\s*(\d+)")
RE_ESTACIONAMIENTOS = re.compile(r"Estacionamientos:?\s*(\d+)", re.IGNORECASE)
RE_ANTIGUEDAD = re.compile(r"Antigüedad\s*(\d+)\s*años?", re.IGNORECASE)
RE_AMOBLADO = re.compile(r"Amoblado:?\s*(Sí|No)", re.IGNORECASE)
RE_ADMITE_MASCOTAS = re.compile(r"Admite mascotas:?\s*(Sí|No)", re.IGNORECASE)
RE_CONDOMINIO_CERRADO = re.compile(r"En condominio cerrado:?\s*(Sí|No)", re.IGNORECASE)

# --- Campos nuevos: comunes a casa y departamento ---
RE_BODEGAS = re.compile(r"Bodegas\s*(\d+)", re.IGNORECASE)

# Igual que con RE_BANOS/RE_DORMITORIOS más arriba: "Gastos comunes" aparece
# DOS VECES en la página. Primero en la insignia superior de resumen, con el
# formato "Gastos comunes desde $ X" (la palabra "desde" antes del monto hace
# que este patrón NUNCA matchee ahí) y, más abajo, en la sección de
# características del inmueble, con el formato "Gastos comunes" / "X CLP" en
# líneas separadas - que es el que este patrón captura normalmente.
# RE_GASTOS_COMUNES_RESUMEN es un FALLBACK explícito (no se usa por defecto):
# hay avisos donde la sección de características simplemente no incluye este
# campo, pero la insignia superior sí trae el monto - sin este fallback esos
# casos quedan NULL aunque el dato exista en la página.
RE_GASTOS_COMUNES = re.compile(r"Gastos comunes:?\s*\$?\s*([\d.,]+)", re.IGNORECASE)
RE_GASTOS_COMUNES_RESUMEN = re.compile(r"Gastos comunes\s+desde\s*\$?\s*([\d.,]+)", re.IGNORECASE)
RE_ESTACIONAMIENTO_VISITAS = re.compile(r"Estacionamiento de visitas:?\s*(Sí|No)", re.IGNORECASE)
RE_SOLO_FAMILIAS = re.compile(r"Solo familias:?\s*(Sí|No)", re.IGNORECASE)
RE_MAX_HABITANTES = re.compile(r"Cantidad máxima de habitantes\s*(\d+)", re.IGNORECASE)
RE_PISCINA = re.compile(r"Piscina:?\s*(Sí|No)", re.IGNORECASE)
RE_QUINCHO = re.compile(r"Quincho\D{0,15}?:?\s*(Sí|No)", re.IGNORECASE)
RE_CONSERJERIA = re.compile(r"Conserjería:?\s*(Sí|No)", re.IGNORECASE)

# --- Exclusivos de departamento (quedan None en casas, es esperado) ---
RE_ASCENSOR = re.compile(r"Ascensor:?\s*(Sí|No)", re.IGNORECASE)
RE_PISO_UNIDAD = re.compile(r"Número de piso de la unidad\s*(\d+)", re.IGNORECASE)
RE_DEPTOS_POR_PISO = re.compile(r"Departamentos por piso\s*(\d+)", re.IGNORECASE)

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

# Radio máximo (en metros) para considerar un punto de interés como "cercano".
# Los que el sitio muestra más lejos que esto se ignoran por completo (no
# cuentan ni se usan como "el más cercano"). El sitio a veces trae POIs hasta
# 2km, que es demasiada distancia para considerarla relevante a pie.
RADIO_MAXIMO_POI_M = 500

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
            bodegas                     TEXT,
            gastos_comunes              TEXT,
            estacionamiento_visitas     TEXT,
            solo_familias               TEXT,
            max_habitantes              TEXT,
            piscina                     TEXT,
            quincho                     TEXT,
            conserjeria                 TEXT,
            ascensor                    TEXT,
            piso_unidad                 TEXT,
            deptos_por_piso             TEXT,
            barrio                      TEXT,
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
        "bodegas", "gastos_comunes", "estacionamiento_visitas", "solo_familias",
        "max_habitantes", "piscina", "quincho", "conserjeria", "ascensor",
        "piso_unidad", "deptos_por_piso", "barrio",
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
# ADAPTADOR HTML -> interfaz mínima de Playwright Page
# ------------------------------------------------------------------
# Todas las funciones de extracción de más abajo (extraer_detalle,
# extraer_coordenadas, extraer_descripcion, extraer_json_estado_pagina,
# hay_captcha) fueron escritas originalmente para recibir un `page` de
# Playwright y llamar `page.content()` / `page.locator(sel).inner_text()`.
# En vez de reescribir esa lógica de negocio para requests, este adaptador
# imita el subconjunto mínimo de esa interfaz sobre HTML estático - así las
# funciones de extracción quedan intactas y funcionan igual con la ruta
# principal (requests) y con la de respaldo (Playwright real).
class _LocatorHTMLEstatico:
    def __init__(self, soup: BeautifulSoup, selector: str):
        self._soup = soup
        self._selector = selector

    @property
    def first(self):
        return self

    def count(self) -> int:
        if self._selector == "body":
            return 1
        return len(self._soup.select(self._selector))

    def inner_text(self) -> str:
        if self._selector == "body":
            nodo = self._soup.body or self._soup
        else:
            elementos = self._soup.select(self._selector)
            if not elementos:
                return ""
            nodo = elementos[0]
        # separador "\n" para aproximar cómo Playwright inner_text() separa
        # bloques visuales - varios patrones RE_* (ej. RE_FECHA_PUBLICACION)
        # dependen de que haya un salto de línea real entre secciones.
        return nodo.get_text("\n", strip=True)


class PaginaHTMLEstatico:
    """Envuelve HTML ya descargado (vía requests) para que se comporte, para
    los fines de extraer_detalle() y compañía, como un `page` de Playwright."""

    def __init__(self, html: str):
        self._html = html or ""
        self._soup = BeautifulSoup(self._html, "lxml")

    def content(self) -> str:
        return self._html

    def locator(self, selector: str) -> _LocatorHTMLEstatico:
        return _LocatorHTMLEstatico(self._soup, selector)


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


def _buscar_valor_por_clave(nodo, clave: str):
    """
    Busca recursivamente todas las apariciones de `clave` dentro del JSON de
    estado y devuelve el primer valor no vacío que encuentre. Se recorre todo
    el árbol (en vez de detenerse en la primera coincidencia) porque el sitio
    a veces repite el mismo bloque de datos en más de un lugar del JSON, y
    alguna de esas copias puede venir vacía/None mientras otra sí trae el dato.
    """
    encontrados = []

    def _recorrer(n):
        if isinstance(n, dict):
            if clave in n:
                encontrados.append(n[clave])
            for v in n.values():
                _recorrer(v)
        elif isinstance(n, list):
            for item in n:
                _recorrer(item)

    _recorrer(nodo)

    for valor in encontrados:
        if valor:
            return valor
    return None


def extraer_puntos_interes(estado: Optional[dict]) -> dict:
    """
    Extrae, para cada subcategoría conocida (colegios, paraderos, plazas,
    etc.), cuántos hay DENTRO DE RADIO_MAXIMO_POI_M y la distancia del más
    cercano de esos. Los puntos más lejanos que el radio se ignoran por
    completo (ni cuentan ni se consideran como "el más cercano") - si no hay
    ninguno dentro del radio, queda cantidad=0 y distancia=None.

    Recibe el JSON de estado ya parseado (ver extraer_json_estado_pagina) en
    vez de `page` directamente, para no tener que parsear el JSON dos veces
    si ya se hizo en otro punto de extraer_detalle.
    """
    resultado = {}
    for clave in SUBCATEGORIAS_POI.values():
        resultado[f"cantidad_{clave}"] = None
        resultado[f"distancia_min_m_{clave}"] = None

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

            distancias_dentro_del_radio = []
            for item in items:
                texto_subtitulo = item.get("subtitle", {}).get("label", {}).get("text", "")
                distancia = parsear_distancia_metros(texto_subtitulo)
                if distancia is not None and distancia <= RADIO_MAXIMO_POI_M:
                    distancias_dentro_del_radio.append(distancia)

            resultado[f"cantidad_{clave}"] = len(distancias_dentro_del_radio)
            resultado[f"distancia_min_m_{clave}"] = (
                min(distancias_dentro_del_radio) if distancias_dentro_del_radio else None
            )

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

    gastos_comunes = buscar(RE_GASTOS_COMUNES)
    if gastos_comunes is None:
        gastos_comunes = buscar(RE_GASTOS_COMUNES_RESUMEN)

    fecha_texto = buscar(RE_FECHA_PUBLICACION)
    coordenadas = extraer_coordenadas(page)

    estado = extraer_json_estado_pagina(page)
    puntos_interes = extraer_puntos_interes(estado)
    barrio = _buscar_valor_por_clave(estado, "neighborhood") if estado else None

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
        "bodegas": buscar(RE_BODEGAS),
        "gastos_comunes": gastos_comunes,
        "estacionamiento_visitas": buscar(RE_ESTACIONAMIENTO_VISITAS),
        "solo_familias": buscar(RE_SOLO_FAMILIAS),
        "max_habitantes": buscar(RE_MAX_HABITANTES),
        "piscina": buscar(RE_PISCINA),
        "quincho": buscar(RE_QUINCHO),
        "conserjeria": buscar(RE_CONSERJERIA),
        "ascensor": buscar(RE_ASCENSOR),
        "piso_unidad": buscar(RE_PISO_UNIDAD),
        "deptos_por_piso": buscar(RE_DEPTOS_POR_PISO),
        "latitud": coordenadas["latitud"],
        "longitud": coordenadas["longitud"],
        "barrio": barrio,
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
# FETCH - ruta principal (requests)
# ------------------------------------------------------------------

def obtener_detalle_aviso(url: str, comuna: str, tipo_propiedad: str) -> dict:
    """
    Punto de entrada de alto nivel de la ruta principal: descarga el HTML del
    aviso vía requests, detecta CAPTCHA y extrae todos los campos - con
    reintento automático ante fallos de red/HTTP (ej. un 404 transitorio)
    antes de darlo por fallido EN ESTA corrida.

    La reutilizan tanto main() (este script) como
    05_modelo_produccion/02_scraper_detalle_incremental.py (vía `sd.` sobre
    este módulo), para no duplicar la lógica de fetch+reintento.

    Devuelve un dict con:
      "resultado": "ok" | "captcha" | "error"
      "datos": dict de extraer_detalle() + distancias (solo si resultado == "ok")
      "estado_json": JSON de extraer_json_estado_pagina(), ya parseado (solo si
          resultado == "ok") - se expone para que un llamador (ej. el scraper
          incremental de producción, que también necesita estado_publicacion)
          no tenga que volver a descargar la página para leerlo.
      "status_http": último código HTTP observado (o None si fue un error de red)
      "motivo": descripción corta del fallo (solo si resultado != "ok")
    """
    referer = construir_referer(comuna, tipo_propiedad)
    intentos_totales = 1 + REINTENTOS_TRAS_ERROR
    ultimo_status = None
    ultimo_motivo = None

    for intento in range(1, intentos_totales + 1):
        try:
            resp = requests.get(url, headers=headers_requests(referer), timeout=TIMEOUT_REQUEST_SEG)
            ultimo_status = resp.status_code

            if resp.status_code != 200:
                ultimo_motivo = f"status HTTP {resp.status_code}"
                raise ValueError(ultimo_motivo)

            pagina = PaginaHTMLEstatico(resp.text)

            if hay_captcha(pagina):
                return {"resultado": "captcha", "status_http": ultimo_status, "motivo": "captcha"}

            datos = extraer_detalle(pagina)
            distancias = calcular_distancias_centro(datos.get("latitud"), datos.get("longitud"), comuna)
            datos.update(distancias)
            estado_json = extraer_json_estado_pagina(pagina)
            return {
                "resultado": "ok", "datos": datos, "estado_json": estado_json,
                "status_http": ultimo_status, "motivo": None,
            }

        except (requests.RequestException, ValueError) as e:
            ultimo_motivo = ultimo_motivo or str(e)
            if intento < intentos_totales:
                espera = random.uniform(BACKOFF_REINTENTO_MIN, BACKOFF_REINTENTO_MAX)
                log.warning(f"Intento {intento}/{intentos_totales} falló para {url} ({ultimo_motivo}). "
                            f"Reintentando en {espera:.1f}s...")
                time.sleep(espera)
            else:
                log.warning(f"Agotados los {intentos_totales} intentos para {url} ({ultimo_motivo}).")

    return {"resultado": "error", "status_http": ultimo_status, "motivo": ultimo_motivo}


# ------------------------------------------------------------------
# MAIN - ruta principal (requests, sin navegador)
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
        log.info("MODO_MANUAL_CAPTCHA activado, pero la ruta principal (requests) no puede mostrarte "
                 "una ventana de navegador para resolver un CAPTCHA a mano. Si necesitas eso, usa "
                 "'python 02_scraper_detalle.py --fallback-playwright' en su lugar.")

    pendientes = obtener_pendientes(con)

    log.info(f"{len(pendientes)} avisos a procesar en esta corrida "
              f"(límite configurado: {LIMITE_POR_CORRIDA}).")

    if pendientes.empty:
        log.info("No hay avisos pendientes. Nada que hacer.")
        con.close()
        return

    procesados = 0
    detenido_por_captcha = False

    for _, fila in pendientes.iterrows():
        id_aviso = fila["id_aviso"]
        url = fila["url"]

        log.info(f"Visitando {id_aviso}: {url}")

        resultado = obtener_detalle_aviso(url, fila["comuna"], fila["tipo_propiedad"])

        if resultado["resultado"] == "captcha":
            log.error(f"CAPTCHA detectado en {url}. Deteniendo la corrida ahora mismo. "
                      f"Lo ya procesado ({procesados} avisos) queda guardado. "
                      f"Se activa un cooldown de {COOLDOWN_TRAS_CAPTCHA_MINUTOS} minutos "
                      f"antes de permitir la siguiente corrida.")
            registrar_captcha(con)
            detenido_por_captcha = True
            break

        if resultado["resultado"] == "error":
            log.warning(f"No se pudo obtener {id_aviso} tras reintentos ({resultado['motivo']}). "
                        f"Se salta este aviso (queda pendiente para la próxima corrida).")
            time.sleep(random.uniform(DELAY_MIN, DELAY_MAX))
            continue

        guardar_detalle(con, id_aviso, url, resultado["datos"])
        procesados += 1

        log.info(f"  -> Guardado ({procesados}/{len(pendientes)})")

        time.sleep(random.uniform(DELAY_MIN, DELAY_MAX))

    con.close()

    if detenido_por_captcha:
        log.info(f"Corrida detenida por CAPTCHA. Avisos procesados en esta corrida: {procesados}.")
    else:
        log.info(f"Corrida completa. Avisos procesados en esta corrida: {procesados}.")


# ------------------------------------------------------------------
# MAIN - ruta de RESPALDO (Playwright + navegador real)
# ------------------------------------------------------------------
# NO es la ruta principal. Solo se ejecuta si se invoca explícitamente (ver
# `if __name__ == "__main__":` más abajo) - nunca automáticamente desde main()
# ni al importar este módulo. El import de Playwright es perezoso (ocurre
# recién al llamar esta función) para que el resto del script funcione sin
# problema en entornos donde Playwright no está instalado (ej. GitHub Actions).
def main_fallback_playwright():
    try:
        from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError
    except ImportError:
        raise RuntimeError(
            "Playwright no está instalado. Instálalo con: "
            "pip install playwright && playwright install chromium"
        )

    try:
        from playwright_stealth import Stealth
        stealth_disponible = True
    except ImportError:
        stealth_disponible = False

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

    if not stealth_disponible:
        log.warning("playwright-stealth no está instalado. El scraper funcionará igual, pero con más "
                    "riesgo de bloqueo. Considera instalarlo: pip install playwright-stealth")

    pendientes = obtener_pendientes(con)

    log.info(f"[FALLBACK PLAYWRIGHT] {len(pendientes)} avisos a procesar en esta corrida "
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

        if stealth_disponible:
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
    if "--fallback-playwright" in sys.argv:
        main_fallback_playwright()
    else:
        main()