"""
Scraper de Portal Inmobiliario - Arriendos (casas y departamentos)
Comunas del Gran Concepción

REQUISITOS:
    pip install requests beautifulsoup4 pandas lxml

NOTAS IMPORTANTES:
- Portal Inmobiliario (grupo MercadoLibre) cambia frecuentemente los nombres de
  clases CSS de sus tarjetas de resultado. Si el scraper deja de traer datos,
  lo primero que hay que revisar es la sección `SELECTORES` más abajo:
  entra a una página de resultados, abre el inspector (F12) y confirma los
  nombres de clase actuales.
- El sitio puede mostrar CAPTCHA si detecta tráfico automatizado muy agresivo.
  Este script usa delays aleatorios y un User-Agent de navegador real para
  reducir el riesgo, pero no lo elimina. Si te empieza a bloquear:
    * Aumenta DELAY_MIN / DELAY_MAX
    * Reduce la cantidad de comunas/páginas por corrida
    * Considera usar un proxy residencial o Playwright con perfil real
- Este script es para uso personal/investigación. Revisa los Términos de Uso
  y robots.txt del sitio (portalinmobiliario.com/robots.txt) antes de scrapear
  a gran escala, y no reproduzcas ni redistribuyas contenido con derechos de
  terceros (ej. fotos) sin permiso. Nota: el robots.txt de este sitio permite
  explícitamente el patrón de paginación `_Desde_` que usa este script
  (Allow: /*_Desde_), pero NO usamos parámetros de orden (_OrderId_) porque
  ese sí está bloqueado (Disallow: *_OrderId_).
- CORRIDAS REPETIDAS (ej. todos los días): cada vez que corres el script,
  vuelve a buscar en todas las comunas, pero:
    1) Se detiene apenas una página completa resulte ser 100% avisos que ya
       existían en la base de datos (asume que lo nuevo tiende a aparecer
       antes que lo viejo en el orden por defecto del sitio).
    2) Guarda cada página de inmediato en la BBDD (no espera a terminar todo),
       así no se pierde nada si el script se corta a mitad de camino.
"""

import re
import time
import random
import logging
import sqlite3
from pathlib import Path
from datetime import date
from dataclasses import dataclass, asdict
from typing import Optional

import requests
from bs4 import BeautifulSoup
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ------------------------------------------------------------------
# CONFIGURACIÓN
# ------------------------------------------------------------------

# Comunas del Gran Concepción (slug tal como aparece en las URLs del sitio,
# formato "comuna-region"). Ajusta/quita según lo que necesites.
COMUNAS_GRAN_CONCEPCION = [
    "concepcion-biobio",
    "talcahuano-biobio",
    "hualpen-biobio",
    "san-pedro-de-la-paz-biobio",
    "chiguayante-biobio",
    "penco-biobio",
    "tome-biobio",
    "coronel-biobio",
    "hualqui-biobio",
    "lota-biobio",
]

TIPOS_PROPIEDAD = ["casa", "departamento"]
OPERACION = "arriendo"

MAX_PAGINAS_POR_BUSQUEDA = 1000   # tope de seguridad; el corte real es por duplicados o por 404
RESULTADOS_POR_PAGINA = 48        # tamaño real de página del sitio - NO tocar salvo que el sitio cambie esto
DELAY_MIN = 3.0                   # segundos, entre requests
DELAY_MAX = 7.0

BASE_URL = "https://www.portalinmobiliario.com"

# La ruta se ancla a la carpeta donde vive este script, sin importar desde
# dónde lo ejecutes (terminal en la raíz del proyecto, VS Code "Run", etc.)
CARPETA_SCRIPT = Path(__file__).resolve().parent
RUTA_BD = CARPETA_SCRIPT / "avisos_gran_concepcion.db"

# Extrae el ID único del aviso desde su URL (ej. ".../MLC-1234567890-titulo..." -> "MLC-1234567890")
RE_ID_AVISO = re.compile(r"(MLC-\d+)")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "es-CL,es;q=0.9,en;q=0.8",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
}

# ------------------------------------------------------------------
# SELECTORES (lo más probable que necesites tocar esto con el tiempo)
# ------------------------------------------------------------------
SELECTORES = {
    # contenedor de cada aviso en la grilla de resultados
    "tarjeta": ["div.ui-search-result__wrapper", "div.andes-card", "li.ui-search-layout__item"],
    "titulo": ["h2.ui-search-item__title", "h3.poly-component__title", "a.poly-component__title"],
    "link": ["a.ui-search-link", "a.poly-component__title"],
    "precio": ["span.andes-money-amount__fraction"],
    "moneda": ["span.andes-money-amount__currency-symbol"],
    "ubicacion": ["span.ui-search-item__location", "span.poly-component__location"],
}

# Regex para dormitorios / baños / m2. Se aplican sobre el TEXTO COMPLETO de la
# tarjeta en vez de depender de una clase CSS específica para la lista de
# atributos (esa clase cambia seguido; el texto "N dormitorios | N baños | N m²"
# se ha mantenido más estable con el tiempo).
RE_DORMITORIOS = re.compile(r"(\d+)\s*dormitorios?", re.IGNORECASE)
RE_BANOS = re.compile(r"(\d+)\s*ba[ñn]os?", re.IGNORECASE)
RE_M2 = re.compile(r"([\d.,]+)\s*m[²2]\b", re.IGNORECASE)


@dataclass
class Aviso:
    comuna: str
    tipo_propiedad: str
    operacion: str
    id_aviso: Optional[str] = None
    titulo: Optional[str] = None
    precio: Optional[str] = None
    moneda: Optional[str] = None
    ubicacion: Optional[str] = None
    dormitorios: Optional[str] = None
    banos: Optional[str] = None
    superficie_m2: Optional[str] = None
    url: Optional[str] = None


def extraer_id_aviso(url: Optional[str]) -> Optional[str]:
    """Extrae el ID único del aviso (ej. 'MLC-1234567890') desde su URL."""
    if not url:
        return None
    m = RE_ID_AVISO.search(url)
    return m.group(1) if m else None


def _first_match(soup_or_tag, selector_list):
    """Prueba una lista de selectores CSS y devuelve el primer resultado (o lista vacía)."""
    for sel in selector_list:
        found = soup_or_tag.select(sel)
        if found:
            return found
    return []


def construir_url(tipo_propiedad: str, comuna: str, pagina: int) -> str:
    """
    Construye la URL de búsqueda. Portal Inmobiliario pagina mediante el
    parámetro _Desde_N (N = offset del primer resultado, no el número de página).
    """
    offset = 1 + (pagina - 1) * RESULTADOS_POR_PAGINA
    if pagina == 1:
        return f"{BASE_URL}/{OPERACION}/{tipo_propiedad}/{comuna}"
    return f"{BASE_URL}/{OPERACION}/{tipo_propiedad}/{comuna}/_Desde_{offset}_NoIndex_True"


def obtener_html(url: str) -> Optional[str]:
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
    except requests.RequestException as e:
        log.warning(f"Error de red en {url}: {e}")
        return None

    if resp.status_code != 200:
        log.warning(f"Status {resp.status_code} en {url}")
        return None

    if "captcha" in resp.text.lower()[:5000]:
        log.warning(f"Posible CAPTCHA detectado en {url}. Deteniendo esta búsqueda.")
        return None

    return resp.text


def extraer_atributo_texto(tag, selector_list) -> Optional[str]:
    encontrados = _first_match(tag, selector_list)
    if encontrados:
        return encontrados[0].get_text(strip=True)
    return None


def parsear_atributos_regex(tarjeta) -> dict:
    """
    Extrae dormitorios / baños / m2 buscando el patrón de texto directamente
    (ej. "3 dormitorios | 4 baños | 120 m² útiles"), sin depender de que la
    tarjeta use una clase CSS específica para cada atributo.
    """
    texto_completo = tarjeta.get_text(" ", strip=True)

    resultado = {"dormitorios": None, "banos": None, "superficie_m2": None}

    m = RE_DORMITORIOS.search(texto_completo)
    if m:
        resultado["dormitorios"] = m.group(1)

    m = RE_BANOS.search(texto_completo)
    if m:
        resultado["banos"] = m.group(1)

    m = RE_M2.search(texto_completo)
    if m:
        resultado["superficie_m2"] = m.group(1)

    return resultado


def parsear_pagina(html: str, comuna: str, tipo_propiedad: str) -> list:
    soup = BeautifulSoup(html, "lxml")
    tarjetas = _first_match(soup, SELECTORES["tarjeta"])

    if not tarjetas:
        log.info(f"Sin tarjetas encontradas ({comuna}, {tipo_propiedad}). "
                  f"Puede que ya no haya más resultados o cambiaron los selectores.")
        return []

    avisos = []
    for tarjeta in tarjetas:
        titulo = extraer_atributo_texto(tarjeta, SELECTORES["titulo"])
        precio = extraer_atributo_texto(tarjeta, SELECTORES["precio"])
        moneda = extraer_atributo_texto(tarjeta, SELECTORES["moneda"])
        ubicacion = extraer_atributo_texto(tarjeta, SELECTORES["ubicacion"])

        link_tag = _first_match(tarjeta, SELECTORES["link"])
        url = link_tag[0].get("href") if link_tag else None

        atributos = parsear_atributos_regex(tarjeta)

        avisos.append(Aviso(
            comuna=comuna,
            tipo_propiedad=tipo_propiedad,
            operacion=OPERACION,
            id_aviso=extraer_id_aviso(url),
            titulo=titulo,
            precio=precio,
            moneda=moneda,
            ubicacion=ubicacion,
            dormitorios=atributos["dormitorios"],
            banos=atributos["banos"],
            superficie_m2=atributos["superficie_m2"],
            url=url,
        ))
    return avisos


def limpiar_precio(valor: Optional[str]) -> Optional[float]:
    """Convierte '450.000' -> 450000.0. Devuelve None si no se puede parsear."""
    if not valor:
        return None
    solo_numeros = re.sub(r"[^\d]", "", valor)
    return float(solo_numeros) if solo_numeros else None


# ------------------------------------------------------------------
# PERSISTENCIA (SQLite)
# ------------------------------------------------------------------

def inicializar_bd(ruta_bd: Path = RUTA_BD) -> sqlite3.Connection:
    con = sqlite3.connect(ruta_bd)
    con.execute("""
        CREATE TABLE IF NOT EXISTS avisos (
            id_aviso        TEXT PRIMARY KEY,
            comuna          TEXT,
            tipo_propiedad  TEXT,
            operacion       TEXT,
            titulo          TEXT,
            precio          REAL,
            moneda          TEXT,
            ubicacion       TEXT,
            dormitorios     TEXT,
            banos           TEXT,
            superficie_m2   TEXT,
            url             TEXT,
            first_seen      TEXT   -- fecha (YYYY-MM-DD) en que se guardó por primera vez
        )
    """)
    con.commit()
    return con


def obtener_ids_existentes(con: sqlite3.Connection) -> set:
    """Carga en memoria todos los id_aviso ya guardados, para comparar rápido sin ir a disco por cada aviso."""
    cur = con.execute("SELECT id_aviso FROM avisos")
    return {fila[0] for fila in cur.fetchall()}


def guardar_pagina_en_bd(avisos: list, con: sqlite3.Connection) -> int:
    """
    Inserta los avisos de UNA página. Si un id_aviso ya existe, lo ignora
    (INSERT OR IGNORE). Hace commit inmediatamente (guardado incremental).
    Devuelve cuántos eran realmente nuevos.
    """
    hoy = date.today().isoformat()
    cur = con.cursor()
    nuevos = 0

    for aviso in avisos:
        if not aviso.id_aviso:
            continue  # sin ID no se puede guardar de forma única, se descarta

        precio_numerico = limpiar_precio(aviso.precio)

        cur.execute("""
            INSERT OR IGNORE INTO avisos (
                id_aviso, comuna, tipo_propiedad, operacion, titulo, precio,
                moneda, ubicacion, dormitorios, banos, superficie_m2, url, first_seen
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            aviso.id_aviso, aviso.comuna, aviso.tipo_propiedad, aviso.operacion,
            aviso.titulo, precio_numerico, aviso.moneda, aviso.ubicacion,
            aviso.dormitorios, aviso.banos, aviso.superficie_m2, aviso.url, hoy,
        ))

        if cur.rowcount == 1:
            nuevos += 1

    con.commit()  # <- commit por página, no al final. Así no se pierde nada si se corta.
    return nuevos


# ------------------------------------------------------------------
# ORQUESTACIÓN PRINCIPAL
# ------------------------------------------------------------------

def scrapear_todo(con: sqlite3.Connection, comunas=None, tipos=None,
                   max_paginas=MAX_PAGINAS_POR_BUSQUEDA) -> dict:
    """
    Recorre comunas x tipos x páginas, guardando cada página de inmediato en
    la BBDD. Corta una búsqueda apenas una página completa resulte ser 100%
    avisos que ya existían (asumiendo que lo nuevo tiende a aparecer antes
    que lo viejo en el orden por defecto del sitio).
    """
    comunas = comunas or COMUNAS_GRAN_CONCEPCION
    tipos = tipos or TIPOS_PROPIEDAD

    ids_conocidos = obtener_ids_existentes(con)
    log.info(f"{len(ids_conocidos)} avisos ya existían en la BBDD antes de esta corrida.")

    total_nuevos = 0
    total_vistos = 0
    cortes_por_duplicado = 0

    for comuna in comunas:
        for tipo in tipos:
            log.info(f"--- Buscando: {OPERACION} de {tipo} en {comuna} ---")

            for pagina in range(1, max_paginas + 1):
                url = construir_url(tipo, comuna, pagina)
                log.info(f"Página {pagina}: {url}")
                html = obtener_html(url)

                if html is None:
                    break  # fin de resultados (404) o error/CAPTCHA

                avisos = parsear_pagina(html, comuna, tipo)
                if not avisos:
                    break  # página vacía = fin de resultados

                ids_de_la_pagina = [a.id_aviso for a in avisos if a.id_aviso]
                pagina_totalmente_duplicada = (
                    len(ids_de_la_pagina) > 0
                    and all(i in ids_conocidos for i in ids_de_la_pagina)
                )

                if pagina_totalmente_duplicada:
                    log.info(f"  -> Página completa ya existía en la BBDD. "
                              f"Corto esta búsqueda y paso a la siguiente.")
                    cortes_por_duplicado += 1
                    break

                nuevos_en_pagina = guardar_pagina_en_bd(avisos, con)
                ids_conocidos.update(ids_de_la_pagina)  # para que la próxima página compare bien

                total_nuevos += nuevos_en_pagina
                total_vistos += len(avisos)

                log.info(f"  -> {len(avisos)} avisos vistos, {nuevos_en_pagina} nuevos guardados "
                          f"(acumulado nuevos: {total_nuevos})")

                time.sleep(random.uniform(DELAY_MIN, DELAY_MAX))

    return {
        "total_vistos": total_vistos,
        "total_nuevos": total_nuevos,
        "cortes_por_duplicado": cortes_por_duplicado,
    }


if __name__ == "__main__":
    con = inicializar_bd()
    resumen = scrapear_todo(con)
    con.close()

    log.info(
        f"Corrida completa. Avisos vistos: {resumen['total_vistos']} | "
        f"Nuevos guardados: {resumen['total_nuevos']} | "
        f"Búsquedas cortadas por duplicado: {resumen['cortes_por_duplicado']}"
    )