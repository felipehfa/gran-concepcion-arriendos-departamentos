"""
Scraper de DETALLE - Portal Inmobiliario (Gran Concepción)

Complementa al scraper de grilla (01_scraper_grilla.py). Este script:
  1. Lee los avisos ya descubiertos en avisos_gran_concepcion.db (tabla `avisos`).
  2. Visita cada URL individual con Playwright y extrae: descripción completa,
     fecha de publicación, y características principales del inmueble.
  3. Guarda cada aviso INMEDIATAMENTE (commit por aviso) en la MISMA base de
     datos, en una tabla nueva: `avisos_detalle` (dentro de avisos_gran_concepcion.db).
  4. Es reanudable: si se corta a mitad de camino (bloqueo, corte de luz,
     Ctrl+C, etc.), la próxima vez que lo corras retoma solo los avisos que
     todavía no tienen detalle guardado - no vuelve a visitar los que ya
     procesó ni pierde lo avanzado.

REQUISITOS:
    pip install playwright pandas playwright-stealth
    playwright install chromium   (si no lo hiciste ya para el otro scraper)

    playwright-stealth es OPCIONAL pero muy recomendado: reduce varias señales
    que delatan un navegador automatizado (navigator.webdriver, inconsistencias
    de WebGL, etc.). Si no lo instalas, el script funciona igual pero con más
    riesgo de bloqueo.

CÓMO CORRERLO EN UN SERVIDOR SIN PANTALLA (headless obligatorio):
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
  Los delays por defecto son más largos que en el scraper de grilla.
- Si el sitio muestra CAPTCHA, el script se DETIENE de inmediato dejando
  guardado todo lo que alcanzó a procesar, y activa un cooldown antes de
  permitir la siguiente corrida.
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
import time
import random
import logging
import sqlite3
from pathlib import Path
from datetime import date, datetime, timedelta
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

LIMITE_POR_CORRIDA = 100   # avisos a procesar como máximo en UNA ejecución del script
                          # (usa cron para correrlo varias veces al día en vez de subir esto mucho)

COOLDOWN_TRAS_CAPTCHA_MINUTOS = 1   # tiempo mínimo de espera antes de reintentar tras un bloqueo

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

# Selectores candidatos para la descripción completa (probar en orden)
SELECTORES_DESCRIPCION = [
    "[data-testid='core-description'] p",
    "p.ui-pdp-description__content",
    "div.ui-pdp-description",
]


# ------------------------------------------------------------------
# BASE DE DATOS
# ------------------------------------------------------------------

def inicializar_bd(ruta_bd: Path = BD_PRINCIPAL) -> sqlite3.Connection:
    """
    Abre la base de datos principal (la misma que usa el scraper de grilla)
    y crea la tabla `avisos_detalle` si no existe. No toca la tabla `avisos`.
    """
    con = sqlite3.connect(ruta_bd)
    con.execute("""
        CREATE TABLE IF NOT EXISTS avisos_detalle (
            id_aviso                TEXT PRIMARY KEY,
            url                     TEXT,
            descripcion             TEXT,
            fecha_publicacion_texto TEXT,
            fecha_publicacion_aprox TEXT,
            superficie_total_m2     TEXT,
            superficie_util_m2      TEXT,
            dormitorios             TEXT,
            banos                   TEXT,
            estacionamientos        TEXT,
            antiguedad_anos         TEXT,
            amoblado                TEXT,
            admite_mascotas         TEXT,
            condominio_cerrado      TEXT,
            fecha_scrapeo           TEXT
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
    con.execute("""
        INSERT OR REPLACE INTO avisos_detalle (
            id_aviso, url, descripcion, fecha_publicacion_texto, fecha_publicacion_aprox,
            superficie_total_m2, superficie_util_m2, dormitorios, banos, estacionamientos,
            antiguedad_anos, amoblado, admite_mascotas, condominio_cerrado, fecha_scrapeo
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        id_aviso, url, datos.get("descripcion"),
        datos.get("fecha_publicacion_texto"), datos.get("fecha_publicacion_aprox"),
        datos.get("superficie_total_m2"), datos.get("superficie_util_m2"),
        datos.get("dormitorios"), datos.get("banos"), datos.get("estacionamientos"),
        datos.get("antiguedad_anos"), datos.get("amoblado"), datos.get("admite_mascotas"),
        datos.get("condominio_cerrado"), date.today().isoformat(),
    ))
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


def extraer_detalle(page) -> dict:
    texto_completo = page.locator("body").inner_text()

    def buscar(patron):
        m = patron.search(texto_completo)
        return m.group(1).strip() if m else None

    fecha_texto = buscar(RE_FECHA_PUBLICACION)

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