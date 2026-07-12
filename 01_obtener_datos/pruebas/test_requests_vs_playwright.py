"""
PRUEBA DE CONCEPTO (aislada) - requests vs Playwright para Portal Inmobiliario.

Pregunta que responde este script: si nunca nos ha bloqueado ni ha aparecido
CAPTCHA, ¿el HTML/JSON que devuelve una llamada `requests.get()` simple
(sin navegador, sin JavaScript) es equivalente al que devuelve Playwright?

Este script NO modifica 01_scraper_grilla.py ni 02_scraper_detalle.py, y no
escribe nada en avisos_gran_concepcion.db. Solo:
  - LEE 5 URLs de muestra de la tabla `avisos` (de solo lectura).
  - IMPORTA (sin ejecutar su bloque main) las funciones/constantes de ambos
    scrapers para aplicar EXACTAMENTE la misma lógica de parseo a HTML
    obtenido por las dos vías, en vez de reimplementar el parseo aquí.

IMPORTANTE sobre el "control" de Playwright para la grilla:
  01_scraper_grilla.py YA usa requests (no Playwright) - la grilla nunca
  necesitó Playwright. Así que para la URL de grilla, el "control" Playwright
  de este script no reutiliza nada de 01_scraper_grilla.py (no hay lógica
  Playwright ahí que reutilizar): se arma con el mismo User-Agent/stealth/
  timeout que usa 02_scraper_detalle.py, solo para tener un punto de
  comparación con navegador real también en ese caso.

NOTA sobre "texto visible" para requests:
  Los regex de características (dormitorios, baños, superficie) en
  02_scraper_detalle.py se aplican sobre `page.locator("body").inner_text()`
  (texto ya renderizado por el navegador), no sobre el HTML crudo. Como
  `requests` no ejecuta JavaScript, se usa BeautifulSoup(html).get_text()
  como aproximación al texto visible. Si el sitio es SSR (probable, ya que
  el JSON de estado viene embebido en el HTML inicial), este texto debería
  coincidir; si no coincide, es en sí mismo un hallazgo relevante.

Sin dependencias nuevas: requests, bs4/lxml, playwright y playwright-stealth
ya están en uso por los scrapers existentes.
"""

import re
import time
import random
import sqlite3
import logging
import importlib.util
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ------------------------------------------------------------------
# CONFIGURACIÓN
# ------------------------------------------------------------------

CARPETA_PRUEBAS = Path(__file__).resolve().parent
CARPETA_SCRIPTS = CARPETA_PRUEBAS.parent   # 01_obtener_datos/
BD_PRINCIPAL = CARPETA_SCRIPTS / "avisos_gran_concepcion.db"

DELAY_MIN = 2.0
DELAY_MAX = 4.0

N_URLS_DETALLE = 5
URL_GRILLA_COMUNA = "concepcion-biobio"
URL_GRILLA_TIPO = "departamento"

# Patrón del bloque JSON embebido (idéntico al que usa
# extraer_json_estado_pagina en 02_scraper_detalle.py, salvo que ahí opera
# sobre `page.content()` y aquí sobre el HTML crudo de cualquiera de los dos
# métodos - es el mismo texto en ambos casos, así que el patrón debe ser
# idéntico para que la comparación sea justa).
RE_JSON_ESTADO = re.compile(r"_n\.ctx\.r=(\{.*?\});_n\.ctx\.r\.assets\.manifest=", re.DOTALL)


def _cargar_modulo(nombre_archivo: str, alias: str):
    """
    Importa un .py cuyo nombre empieza con dígitos (no se puede hacer
    `import 01_scraper_grilla`) SIN tocarlo ni ejecutar su bloque
    `if __name__ == "__main__":`. Así reutilizamos sus funciones/constantes
    reales en vez de copiarlas a mano.
    """
    ruta = CARPETA_SCRIPTS / nombre_archivo
    spec = importlib.util.spec_from_file_location(alias, ruta)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


grilla = _cargar_modulo("01_scraper_grilla.py", "scraper_grilla")
detalle = _cargar_modulo("02_scraper_detalle.py", "scraper_detalle")


def headers_requests(referer: str) -> dict:
    """
    Headers de 01_scraper_grilla.py (grilla.HEADERS) + el mismo User-Agent
    que usa Playwright en 02_scraper_detalle.py (para que ambos métodos se
    presenten como el mismo navegador) + Referer apuntando al dominio del
    sitio, igual que hace detalle.construir_referer para Playwright.
    """
    h = dict(grilla.HEADERS)
    h["User-Agent"] = detalle.USER_AGENT
    h["Referer"] = referer
    return h


def esperar_entre_requests():
    time.sleep(random.uniform(DELAY_MIN, DELAY_MAX))


# ------------------------------------------------------------------
# FETCH: PLAYWRIGHT (control) y REQUESTS (candidato)
# ------------------------------------------------------------------

def fetch_playwright(context, url: str, referer: str, capturar_texto: bool = False) -> dict:
    page = context.new_page()
    resultado = {"status": None, "html": None, "texto_visible": None, "segundos": None, "error": None}
    t0 = time.time()
    try:
        resp = page.goto(url, wait_until="domcontentloaded", timeout=30000, referer=referer)
        page.wait_for_timeout(1000)
        resultado["status"] = resp.status if resp else None
        resultado["html"] = page.content()
        if capturar_texto:
            try:
                resultado["texto_visible"] = page.locator("body").inner_text()
            except Exception:
                resultado["texto_visible"] = None
    except PlaywrightTimeoutError as e:
        resultado["error"] = f"timeout: {e}"
    except Exception as e:
        resultado["error"] = str(e)
    finally:
        resultado["segundos"] = time.time() - t0
        page.close()
    return resultado


def fetch_requests(url: str, referer: str) -> dict:
    resultado = {"status": None, "html": None, "segundos": None, "error": None}
    t0 = time.time()
    try:
        resp = requests.get(url, headers=headers_requests(referer), timeout=15)
        resultado["status"] = resp.status_code
        resultado["html"] = resp.text
    except requests.RequestException as e:
        resultado["error"] = str(e)
    finally:
        resultado["segundos"] = time.time() - t0
    return resultado


def hay_indicio_bloqueo(html: str, texto: str = None) -> bool:
    """
    Mismo criterio "de verdad" que usa hay_captcha() en 02_scraper_detalle.py:
    la palabra "captcha" aparece en el HTML Y el contenido normal del aviso
    no logró cargar. Si `texto` (visible) no se entrega, se deriva del HTML
    crudo vía BeautifulSoup como aproximación.
    """
    if not html:
        return True
    contenido = html.lower()
    if "captcha" not in contenido[:8000]:
        return False

    if texto is None:
        texto = BeautifulSoup(html, "lxml").get_text(" ", strip=True)

    parece_contenido_real = bool(
        detalle.RE_SUPERFICIE_TOTAL.search(texto)
        or detalle.RE_SUPERFICIE_UTIL.search(texto)
        or detalle.RE_DORMITORIOS.search(texto)
    )
    return not parece_contenido_real


def extraer_json_estado(html: str):
    if not html:
        return None
    m = RE_JSON_ESTADO.search(html)
    if not m:
        return None
    import json
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        return None


def extraer_coordenadas_de_html(html: str) -> dict:
    """Misma lógica que detalle.extraer_coordenadas, pero recibe HTML crudo en vez de `page`."""
    if not html:
        return {"latitud": None, "longitud": None}
    m_lat = detalle.RE_LATITUD.search(html)
    m_lon = detalle.RE_LONGITUD.search(html)
    if m_lat and m_lon:
        return {"latitud": m_lat.group(1), "longitud": m_lon.group(1)}
    m_mapa = detalle.RE_LATLON_MAPA.search(html)
    if m_mapa:
        return {"latitud": m_mapa.group(1), "longitud": m_mapa.group(2)}
    return {"latitud": None, "longitud": None}


# ------------------------------------------------------------------
# COMPARACIÓN: UNA URL DE DETALLE
# ------------------------------------------------------------------

def comparar_url_detalle(context, id_aviso: str, url: str, comuna: str, tipo_propiedad: str) -> dict:
    referer = detalle.construir_referer(comuna, tipo_propiedad)

    log.info(f"[{id_aviso}] Playwright -> {url}")
    esperar_entre_requests()
    pw = fetch_playwright(context, url, referer, capturar_texto=True)

    log.info(f"[{id_aviso}] requests   -> {url}")
    esperar_entre_requests()
    rq = fetch_requests(url, referer)

    texto_pw = pw.get("texto_visible") or ""
    texto_rq = BeautifulSoup(rq["html"], "lxml").get_text(" ", strip=True) if rq["html"] else ""

    resultado = {
        "id_aviso": id_aviso, "url": url,
        "pw_status": pw["status"], "pw_segundos": pw["segundos"], "pw_error": pw["error"],
        "rq_status": rq["status"], "rq_segundos": rq["segundos"], "rq_error": rq["error"],
        "pw_bloqueo": hay_indicio_bloqueo(pw["html"], texto_pw),
        "rq_bloqueo": hay_indicio_bloqueo(rq["html"], texto_rq),
    }

    diferencias = []
    coincide = True

    if pw["error"] or rq["error"]:
        diferencias.append(f"error de red: pw={pw['error']!r} rq={rq['error']!r}")
        coincide = False

    if resultado["pw_status"] != resultado["rq_status"]:
        diferencias.append(f"status HTTP distinto: pw={resultado['pw_status']} vs rq={resultado['rq_status']}")
        coincide = False

    if resultado["pw_bloqueo"] or resultado["rq_bloqueo"]:
        diferencias.append(f"indicio de bloqueo/captcha: pw={resultado['pw_bloqueo']} rq={resultado['rq_bloqueo']}")
        coincide = False

    estado_pw = extraer_json_estado(pw["html"])
    estado_rq = extraer_json_estado(rq["html"])
    resultado["pw_json_encontrado"] = estado_pw is not None
    resultado["rq_json_encontrado"] = estado_rq is not None

    if resultado["pw_json_encontrado"] != resultado["rq_json_encontrado"]:
        diferencias.append(
            f"bloque JSON (_n.ctx.r) encontrado: pw={resultado['pw_json_encontrado']} vs rq={resultado['rq_json_encontrado']}"
        )
        coincide = False
    elif estado_pw is None and estado_rq is None:
        diferencias.append("ninguno de los dos métodos encontró el bloque JSON _n.ctx.r")
        coincide = False

    # Coordenadas: regex directo sobre HTML crudo, mismo código en ambos métodos.
    coord_pw = extraer_coordenadas_de_html(pw["html"])
    coord_rq = extraer_coordenadas_de_html(rq["html"])
    if coord_pw != coord_rq:
        diferencias.append(f"coordenadas distintas: pw={coord_pw} vs rq={coord_rq}")
        coincide = False

    # POIs y barrio: se reutilizan las funciones REALES del scraper de detalle,
    # aplicadas al JSON ya parseado de cada método.
    if estado_pw is not None and estado_rq is not None:
        pois_pw = detalle.extraer_puntos_interes(estado_pw)
        pois_rq = detalle.extraer_puntos_interes(estado_rq)
        if pois_pw != pois_rq:
            diferencias.append(f"POIs distintos: pw={pois_pw} vs rq={pois_rq}")
            coincide = False

        barrio_pw = detalle._buscar_valor_por_clave(estado_pw, "neighborhood")
        barrio_rq = detalle._buscar_valor_por_clave(estado_rq, "neighborhood")
        if barrio_pw != barrio_rq:
            diferencias.append(f"barrio distinto: pw={barrio_pw!r} vs rq={barrio_rq!r}")
            coincide = False

        # "precio" - best-effort: 02_scraper_detalle.py no extrae precio (eso
        # lo hace la grilla), así que buscamos una clave genérica típica de
        # las páginas de producto de MercadoLibre/Portal Inmobiliario.
        precio_pw = detalle._buscar_valor_por_clave(estado_pw, "price")
        precio_rq = detalle._buscar_valor_por_clave(estado_rq, "price")
        if precio_pw != precio_rq:
            diferencias.append(f"precio (JSON, best-effort) distinto: pw={precio_pw!r} vs rq={precio_rq!r}")
            coincide = False

    # Características técnicas: mismos regex del scraper de detalle, sobre
    # el texto visible de cada método.
    for nombre, patron in [
        ("dormitorios", detalle.RE_DORMITORIOS),
        ("banos", detalle.RE_BANOS),
        ("superficie_total_m2", detalle.RE_SUPERFICIE_TOTAL),
        ("superficie_util_m2", detalle.RE_SUPERFICIE_UTIL),
    ]:
        m_pw = patron.search(texto_pw)
        m_rq = patron.search(texto_rq)
        v_pw = m_pw.group(1) if m_pw else None
        v_rq = m_rq.group(1) if m_rq else None
        if v_pw != v_rq:
            diferencias.append(f"{nombre} distinto: pw={v_pw!r} vs rq={v_rq!r}")
            coincide = False

    resultado["coincide"] = coincide
    resultado["diferencias"] = diferencias
    return resultado


# ------------------------------------------------------------------
# COMPARACIÓN: URL DE GRILLA
# ------------------------------------------------------------------

def comparar_url_grilla(context) -> dict:
    url = grilla.construir_url(URL_GRILLA_TIPO, URL_GRILLA_COMUNA, 1)
    referer = grilla.BASE_URL

    log.info(f"[GRILLA] Playwright -> {url}")
    esperar_entre_requests()
    pw = fetch_playwright(context, url, referer, capturar_texto=False)

    log.info(f"[GRILLA] requests   -> {url}")
    esperar_entre_requests()
    rq = fetch_requests(url, referer)

    resultado = {
        "id_aviso": "GRILLA", "url": url,
        "pw_status": pw["status"], "pw_segundos": pw["segundos"], "pw_error": pw["error"],
        "rq_status": rq["status"], "rq_segundos": rq["segundos"], "rq_error": rq["error"],
        "pw_bloqueo": hay_indicio_bloqueo(pw["html"]),
        "rq_bloqueo": hay_indicio_bloqueo(rq["html"]),
    }

    # Mismo parseo real que usa el scraper de grilla (grilla.parsear_pagina),
    # aplicado al HTML de cada método.
    avisos_pw = grilla.parsear_pagina(pw["html"], URL_GRILLA_COMUNA, URL_GRILLA_TIPO) if pw["html"] else []
    avisos_rq = grilla.parsear_pagina(rq["html"], URL_GRILLA_COMUNA, URL_GRILLA_TIPO) if rq["html"] else []

    ids_pw = {a.id_aviso for a in avisos_pw if a.id_aviso}
    ids_rq = {a.id_aviso for a in avisos_rq if a.id_aviso}

    diferencias = []
    coincide = True

    if pw["error"] or rq["error"]:
        diferencias.append(f"error de red: pw={pw['error']!r} rq={rq['error']!r}")
        coincide = False

    if resultado["pw_status"] != resultado["rq_status"]:
        diferencias.append(f"status HTTP distinto: pw={resultado['pw_status']} vs rq={resultado['rq_status']}")
        coincide = False

    if resultado["pw_bloqueo"] or resultado["rq_bloqueo"]:
        diferencias.append(f"indicio de bloqueo/captcha: pw={resultado['pw_bloqueo']} rq={resultado['rq_bloqueo']}")
        coincide = False

    if len(avisos_pw) != len(avisos_rq):
        diferencias.append(f"cantidad de avisos parseados distinta: pw={len(avisos_pw)} vs rq={len(avisos_rq)}")
        coincide = False

    if ids_pw != ids_rq:
        solo_pw = ids_pw - ids_rq
        solo_rq = ids_rq - ids_pw
        diferencias.append(f"IDs de avisos distintos - solo en pw: {solo_pw or '{}'} | solo en rq: {solo_rq or '{}'}")
        coincide = False

    resultado["n_avisos_pw"] = len(avisos_pw)
    resultado["n_avisos_rq"] = len(avisos_rq)
    resultado["coincide"] = coincide
    resultado["diferencias"] = diferencias
    return resultado


# ------------------------------------------------------------------
# ORQUESTACIÓN
# ------------------------------------------------------------------

def obtener_urls_detalle_muestra(n: int = N_URLS_DETALLE):
    con = sqlite3.connect(BD_PRINCIPAL)
    filas = con.execute(f"""
        SELECT id_aviso, url, comuna, tipo_propiedad
        FROM avisos
        WHERE url IS NOT NULL AND comuna IS NOT NULL AND tipo_propiedad IS NOT NULL
        ORDER BY RANDOM()
        LIMIT {int(n)}
    """).fetchall()
    con.close()
    return filas


def imprimir_resumen(resultados: list):
    print("\n" + "=" * 78)
    print("RESUMEN: requests vs Playwright (Portal Inmobiliario)")
    print("=" * 78)

    for r in resultados:
        estado_txt = "SI coincide" if r["coincide"] else "NO coincide"
        print(f"\n- {r['id_aviso']}  ({r['url']})")
        print(f"    Resultado:    {estado_txt}")
        print(f"    Status HTTP:  Playwright={r['pw_status']}   requests={r['rq_status']}")
        seg_pw = f"{r['pw_segundos']:.2f}" if r['pw_segundos'] is not None else "?"
        seg_rq = f"{r['rq_segundos']:.2f}" if r['rq_segundos'] is not None else "?"
        print(f"    Tiempo (s):   Playwright={seg_pw}   requests={seg_rq}")
        if r.get("pw_error"):
            print(f"    Error Playwright: {r['pw_error']}")
        if r.get("rq_error"):
            print(f"    Error requests:   {r['rq_error']}")
        if "n_avisos_pw" in r:
            print(f"    Avisos parseados: Playwright={r['n_avisos_pw']}   requests={r['n_avisos_rq']}")
        for d in r["diferencias"]:
            print(f"    * {d}")

    total = len(resultados)
    coinciden = sum(1 for r in resultados if r["coincide"])
    tiempos_pw = [r["pw_segundos"] for r in resultados if r["pw_segundos"] is not None]
    tiempos_rq = [r["rq_segundos"] for r in resultados if r["rq_segundos"] is not None]

    print("\n" + "-" * 78)
    print(f"Total: {coinciden}/{total} URLs coincidieron entre requests y Playwright.")
    if tiempos_pw and tiempos_rq:
        prom_pw = sum(tiempos_pw) / len(tiempos_pw)
        prom_rq = sum(tiempos_rq) / len(tiempos_rq)
        print(f"Tiempo promedio: Playwright={prom_pw:.2f}s   requests={prom_rq:.2f}s "
              f"({prom_pw / prom_rq:.1f}x más lento Playwright)" if prom_rq > 0 else "")
    print("-" * 78)


def main():
    muestra = obtener_urls_detalle_muestra()
    if len(muestra) < N_URLS_DETALLE:
        log.warning(f"Solo hay {len(muestra)} avisos con url/comuna/tipo_propiedad completos "
                    f"(se pidieron {N_URLS_DETALLE}).")

    if not detalle.STEALTH_DISPONIBLE:
        log.warning("playwright-stealth no está disponible - el control de Playwright corre sin stealth.")

    resultados = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(user_agent=detalle.USER_AGENT, locale="es-CL")

        if detalle.STEALTH_DISPONIBLE:
            detalle.Stealth().apply_stealth_sync(context)

        for id_aviso, url, comuna, tipo_propiedad in muestra:
            resultados.append(comparar_url_detalle(context, id_aviso, url, comuna, tipo_propiedad))

        resultados.append(comparar_url_grilla(context))

        browser.close()

    imprimir_resumen(resultados)


if __name__ == "__main__":
    main()
