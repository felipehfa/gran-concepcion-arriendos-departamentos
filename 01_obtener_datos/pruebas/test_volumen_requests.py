"""
PRUEBA DE CONCEPTO (aislada) - volumen con requests para Portal Inmobiliario.

La prueba anterior (test_requests_vs_playwright.py) confirmó que, para 6 URLs
aisladas, requests devuelve HTML/JSON equivalente a Playwright. Esta prueba
NO vuelve a comparar equivalencia de contenido: se enfoca en una pregunta
distinta - ¿un volumen mayor de requests SEGUIDAS (150, con delays 2-4s)
genera algún cambio de comportamiento (bloqueo, CAPTCHA, rate limiting,
respuestas distintas) respecto a las primeras requests de la corrida?

No modifica 02_scraper_detalle.py ni escribe nada en avisos_gran_concepcion.db:
solo LEE 150 URLs de muestra (estratificada por comuna/tipo_propiedad) de la
tabla `avisos`, y hace fetch de cada una solo con requests. Opcionalmente
compara una submuestra de 10 contra Playwright, solo como control adicional
de que no hay divergencia sistemática (no es el foco de esta prueba).

Sin dependencias nuevas: requests y bs4 ya están en uso por 01_scraper_grilla.py.
"""

import re
import time
import random
import sqlite3
import logging
import importlib.util
from pathlib import Path
from statistics import median

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ------------------------------------------------------------------
# CONFIGURACIÓN
# ------------------------------------------------------------------

CARPETA_PRUEBAS = Path(__file__).resolve().parent
CARPETA_SCRIPTS = CARPETA_PRUEBAS.parent
BD_PRINCIPAL = CARPETA_SCRIPTS / "avisos_gran_concepcion.db"

N_URLS = 150
N_CONTROL_PLAYWRIGHT = 10   # submuestra opcional comparada también contra Playwright

DELAY_MIN = 2.0
DELAY_MAX = 4.0

N_BASELINE = 20   # cuántas de las primeras respuestas exitosas se usan para definir el "tamaño normal"
UMBRAL_TAMANO_RATIO = 0.5   # si una respuesta mide menos de la mitad del tamaño normal, se marca como sospechosa

RE_JSON_ESTADO = re.compile(r"_n\.ctx\.r=(\{.*?\});_n\.ctx\.r\.assets\.manifest=", re.DOTALL)
PATRONES_URL_BLOQUEO = re.compile(r"captcha|challenge|verificaci[oó]n|security-check|blocked", re.IGNORECASE)


def _cargar_modulo(nombre_archivo: str, alias: str):
    """Importa un .py cuyo nombre empieza con dígitos sin tocarlo ni ejecutar su main()."""
    ruta = CARPETA_SCRIPTS / nombre_archivo
    spec = importlib.util.spec_from_file_location(alias, ruta)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


grilla = _cargar_modulo("01_scraper_grilla.py", "scraper_grilla")
detalle = _cargar_modulo("02_scraper_detalle.py", "scraper_detalle")


def headers_requests(referer: str) -> dict:
    h = dict(grilla.HEADERS)
    h["User-Agent"] = detalle.USER_AGENT
    h["Referer"] = referer
    return h


def esperar_entre_requests():
    time.sleep(random.uniform(DELAY_MIN, DELAY_MAX))


# ------------------------------------------------------------------
# MUESTRA ESTRATIFICADA (por comuna + tipo_propiedad, para variedad real)
# ------------------------------------------------------------------

def obtener_muestra_estratificada(n: int = N_URLS):
    con = sqlite3.connect(BD_PRINCIPAL)
    grupos = con.execute("""
        SELECT comuna, tipo_propiedad, COUNT(*) AS n
        FROM avisos
        WHERE url IS NOT NULL AND comuna IS NOT NULL AND tipo_propiedad IS NOT NULL
        GROUP BY comuna, tipo_propiedad
    """).fetchall()

    total = sum(g[2] for g in grupos)
    filas = []

    for comuna, tipo, cnt in grupos:
        cuota = max(1, round(n * cnt / total))
        cuota = min(cuota, cnt)
        muestra_grupo = con.execute("""
            SELECT id_aviso, url, comuna, tipo_propiedad
            FROM avisos
            WHERE comuna = ? AND tipo_propiedad = ? AND url IS NOT NULL
            ORDER BY RANDOM()
            LIMIT ?
        """, (comuna, tipo, cuota)).fetchall()
        filas.extend(muestra_grupo)

    con.close()
    random.shuffle(filas)
    return filas[:n]


# ------------------------------------------------------------------
# FETCH Y DETECCIÓN DE BLOQUEO
# ------------------------------------------------------------------

def fetch_una_url(url: str, referer: str) -> dict:
    resultado = {
        "status": None, "segundos": None, "tamano_bytes": None,
        "json_encontrado": False, "url_final": None, "error": None,
    }
    t0 = time.time()
    try:
        resp = requests.get(url, headers=headers_requests(referer), timeout=15)
        resultado["status"] = resp.status_code
        resultado["tamano_bytes"] = len(resp.content)
        resultado["url_final"] = resp.url
        resultado["json_encontrado"] = bool(RE_JSON_ESTADO.search(resp.text))
    except requests.RequestException as e:
        resultado["error"] = str(e)
    finally:
        resultado["segundos"] = time.time() - t0
    return resultado


def detectar_indicios_bloqueo(r: dict, tamano_normal: float = None) -> list:
    """
    Señales explícitas de bloqueo/comportamiento anómalo, evaluadas de forma
    independiente (pueden combinarse varias a la vez):
      - error de red
      - status HTTP distinto de 200
      - redirección a una URL que huele a CAPTCHA/verificación
      - bloque JSON _n.ctx.r ausente
      - HTML muchísimo más chico que el tamaño "normal" de la corrida (baseline)
    """
    indicios = []

    if r["error"]:
        indicios.append(f"error de red: {r['error']}")
        return indicios   # sin respuesta, no tiene sentido evaluar el resto

    if r["status"] != 200:
        indicios.append(f"status HTTP {r['status']}")

    if r["url_final"] and PATRONES_URL_BLOQUEO.search(r["url_final"]):
        indicios.append(f"redirección sospechosa: {r['url_final']}")

    if not r["json_encontrado"]:
        indicios.append("bloque JSON _n.ctx.r no encontrado")

    if tamano_normal and r["tamano_bytes"] is not None:
        ratio = r["tamano_bytes"] / tamano_normal
        if ratio < UMBRAL_TAMANO_RATIO:
            indicios.append(
                f"HTML inusualmente chico: {r['tamano_bytes']} bytes "
                f"({ratio:.0%} del tamaño normal ~{tamano_normal:.0f} bytes)"
            )

    return indicios


# ------------------------------------------------------------------
# CONTROL OPCIONAL: SUBMUESTRA CONTRA PLAYWRIGHT
# ------------------------------------------------------------------

def comparar_submuestra_playwright(muestra_control: list) -> list:
    from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError

    resultados = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(user_agent=detalle.USER_AGENT, locale="es-CL")
        if detalle.STEALTH_DISPONIBLE:
            detalle.Stealth().apply_stealth_sync(context)

        for id_aviso, url, comuna, tipo_propiedad in muestra_control:
            referer = detalle.construir_referer(comuna, tipo_propiedad)
            page = context.new_page()
            esperar_entre_requests()
            try:
                resp = page.goto(url, wait_until="domcontentloaded", timeout=30000, referer=referer)
                page.wait_for_timeout(1000)
                html = page.content()
                status = resp.status if resp else None
                json_ok = bool(RE_JSON_ESTADO.search(html))
            except PlaywrightTimeoutError:
                status, json_ok = None, False
            except Exception:
                status, json_ok = None, False
            finally:
                page.close()

            resultados.append({"id_aviso": id_aviso, "pw_status": status, "pw_json_encontrado": json_ok})

        browser.close()
    return resultados


# ------------------------------------------------------------------
# ORQUESTACIÓN
# ------------------------------------------------------------------

def main():
    muestra = obtener_muestra_estratificada(N_URLS)
    log.info(f"Muestra obtenida: {len(muestra)} URLs "
             f"({len(set((c, t) for _, _, c, t in muestra))} combinaciones comuna/tipo distintas).")

    resultados = []
    t_inicio_corrida = time.time()
    primer_problema_idx = None

    for i, (id_aviso, url, comuna, tipo_propiedad) in enumerate(muestra):
        referer = detalle.construir_referer(comuna, tipo_propiedad)

        if i > 0:
            esperar_entre_requests()

        r = fetch_una_url(url, referer)
        r["indice"] = i
        r["id_aviso"] = id_aviso
        r["url"] = url

        tamano_normal = None
        if len(resultados) >= 5:
            exitosos_previos = [
                x["tamano_bytes"] for x in resultados
                if x["status"] == 200 and x["json_encontrado"] and x["tamano_bytes"]
            ][:N_BASELINE]
            if exitosos_previos:
                tamano_normal = median(exitosos_previos)

        r["indicios"] = detectar_indicios_bloqueo(r, tamano_normal)

        if r["indicios"] and primer_problema_idx is None:
            primer_problema_idx = i
            log.warning(f"[{i+1}/{len(muestra)}] {id_aviso}: primer indicio de bloqueo -> {r['indicios']}")
        else:
            log.info(f"[{i+1}/{len(muestra)}] {id_aviso}: status={r['status']} "
                     f"tam={r['tamano_bytes']}B json={r['json_encontrado']} t={r['segundos']:.2f}s")

        resultados.append(r)

    tiempo_total = time.time() - t_inicio_corrida

    # --- Control opcional contra Playwright, sobre una submuestra ---
    muestra_control = random.sample(muestra, min(N_CONTROL_PLAYWRIGHT, len(muestra)))
    log.info(f"Comparando {len(muestra_control)} URLs de la submuestra contra Playwright (control adicional)...")
    control_pw = comparar_submuestra_playwright(muestra_control)

    imprimir_resumen(resultados, tiempo_total, primer_problema_idx, control_pw)


def imprimir_resumen(resultados: list, tiempo_total: float, primer_problema_idx, control_pw: list):
    total = len(resultados)
    exitosos = [r for r in resultados if not r["indicios"]]
    con_indicios = [r for r in resultados if r["indicios"]]

    tiempos = [r["segundos"] for r in resultados if r["segundos"] is not None]
    tamanos = [r["tamano_bytes"] for r in resultados if r["tamano_bytes"]]

    print("\n" + "=" * 78)
    print(f"RESUMEN: prueba de volumen - {total} requests con requests (solo HTTP, sin navegador)")
    print("=" * 78)

    print(f"\nExitosas (sin ningún indicio de bloqueo): {len(exitosos)}/{total}")
    print(f"Con al menos un indicio de bloqueo/anomalía: {len(con_indicios)}/{total}")

    if primer_problema_idx is not None:
        print(f"Primer indicio apareció en la request #{primer_problema_idx + 1} de {total}.")
    else:
        print("Ningún indicio de bloqueo en toda la corrida.")

    print(f"\nTiempo total de la corrida: {tiempo_total:.1f}s ({tiempo_total/60:.1f} min)")
    if tiempos:
        print(f"Tiempo promedio por request: {sum(tiempos)/len(tiempos):.2f}s "
              f"(min={min(tiempos):.2f}s, max={max(tiempos):.2f}s)")
    if tamanos:
        print(f"Tamaño de HTML: mediana={median(tamanos):.0f}B, min={min(tamanos)}B, max={max(tamanos)}B")

    if con_indicios:
        print(f"\nDetalle de las {len(con_indicios)} requests con indicios:")
        for r in con_indicios:
            print(f"  - #{r['indice']+1} {r['id_aviso']}: {r['indicios']}")

    if control_pw:
        print(f"\nControl adicional contra Playwright ({len(control_pw)} URLs de la submuestra):")
        for c in control_pw:
            print(f"  - {c['id_aviso']}: Playwright status={c['pw_status']} json={c['pw_json_encontrado']}")

    print("\n" + "-" * 78)
    print("-" * 78)


if __name__ == "__main__":
    main()
