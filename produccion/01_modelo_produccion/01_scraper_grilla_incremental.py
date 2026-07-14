"""
Scraper de GRILLA incremental — pipeline de producción.

No duplica el parsing HTML: carga `01_obtener_datos/01_scraper_grilla.py`
como módulo (vía importlib, mismo patrón que las etapas anteriores) y
reutiliza `construir_url`, `obtener_html`, `parsear_pagina`,
`limpiar_precio` y sus constantes (comunas, delays, selectores).

Diferencias respecto al scraper original:
  - Solo recorre TIPOS_PROPIEDAD_PRODUCCION (departamento), no
    `sg.TIPOS_PROPIEDAD` completo (casa + departamento): el resto del
    pipeline de producción solo procesa departamentos, así que traer casas
    sería gastar presupuesto de scraping en avisos que nunca generan
    features ni predicción.
  - Guarda SOLO avisos cuyo id_aviso no exista ya en la base ORIGINAL
    (avisos_gran_concepcion.db, solo lectura) NI en la base de PRODUCCIÓN.
  - Corte por MAX_PAGINAS_VACIAS_CONSECUTIVAS páginas seguidas sin ningún
    aviso nuevo, contado por combinación comuna×tipo.
  - Techo de presupuesto por corrida (MAX_PAGINAS_POR_CORRIDA /
    MAX_MINUTOS_POR_CORRIDA), acumulado sobre TODA la corrida: si se
    alcanza antes que el criterio de páginas vacías, corta la corrida
    completa y lo deja registrado como motivo de corte distinto (no se
    confunde con haber agotado contenido nuevo real).
"""

import importlib.util
import logging
import random
import time
from datetime import date
from pathlib import Path

import db

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
MAX_PAGINAS_VACIAS_CONSECUTIVAS = 10   # por combinación comuna×tipo
MAX_PAGINAS_POR_CORRIDA = 200          # techo global, sumando todas las combinaciones
MAX_MINUTOS_POR_CORRIDA = 30           # techo global de tiempo

# El resto del pipeline de producción (04_ingenieria_variables_produccion.py)
# solo procesa tipo_propiedad='departamento' - traer 'casa' acá sería gastar
# presupuesto de scraping (tiempo, requests, páginas) en avisos que nunca
# van a generar features ni predicción. Configurable (en vez de hardcodear
# el filtro) para poder sumar tipos el día que el modelo los soporte.
TIPOS_PROPIEDAD_PRODUCCION = ["departamento"]

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
INVESTIGACION_ROOT = REPO_ROOT / "investigacion"
SCRAPER_GRILLA_ORIGINAL_PATH = INVESTIGACION_ROOT / "01_obtener_datos" / "01_scraper_grilla.py"


def _cargar_modulo_scraper_grilla():
    spec = importlib.util.spec_from_file_location("scraper_grilla_original", SCRAPER_GRILLA_ORIGINAL_PATH)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


sg = _cargar_modulo_scraper_grilla()


# ------------------------------------------------------------------
# Conversión de tipos (la tabla `avisos` de producción usa tipos
# explícitos, a diferencia del TEXT genérico de la base original)
# ------------------------------------------------------------------
def _a_entero(valor):
    try:
        return int(valor)
    except (TypeError, ValueError):
        return None


def _a_real(valor):
    if valor is None:
        return None
    try:
        return float(str(valor).replace(",", "."))
    except (TypeError, ValueError):
        return None


# ------------------------------------------------------------------
# Deduplicación contra las dos bases
# ------------------------------------------------------------------
def obtener_ids_originales(con_original) -> set:
    cur = con_original.execute("SELECT id_aviso FROM avisos")
    return {fila[0] for fila in cur.fetchall()}


def obtener_ids_produccion(con_produccion) -> set:
    cur = con_produccion.execute("SELECT id_aviso FROM avisos")
    return {fila[0] for fila in cur.fetchall()}


def guardar_pagina_en_produccion(avisos: list, con_produccion, ids_conocidos: set) -> int:
    """
    Inserta en la tabla `avisos` de producción solo los avisos cuyo
    id_aviso no esté ya en `ids_conocidos` (unión de ids de la base
    original + producción + lo visto en esta misma corrida). Hace commit
    inmediatamente (guardado incremental, igual que el scraper original).
    """
    hoy = date.today().isoformat()
    cur = con_produccion.cursor()
    nuevos = 0

    for aviso in avisos:
        if not aviso.id_aviso or aviso.id_aviso in ids_conocidos:
            continue

        cur.execute("""
            INSERT OR IGNORE INTO avisos (
                id_aviso, comuna, tipo_propiedad, operacion, titulo, precio,
                moneda, ubicacion, dormitorios, banos, superficie_m2, url,
                first_seen, estado_publicacion
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'activo')
        """, (
            aviso.id_aviso, aviso.comuna, aviso.tipo_propiedad, aviso.operacion,
            aviso.titulo, sg.limpiar_precio(aviso.precio), aviso.moneda, aviso.ubicacion,
            _a_entero(aviso.dormitorios), _a_entero(aviso.banos), _a_real(aviso.superficie_m2),
            aviso.url, hoy,
        ))

        if cur.rowcount == 1:
            nuevos += 1
        ids_conocidos.add(aviso.id_aviso)

    con_produccion.commit()
    return nuevos


# ------------------------------------------------------------------
# ORQUESTACIÓN
# ------------------------------------------------------------------
def scrapear_grilla_incremental(
    con_produccion,
    con_original,
    comunas: list = None,
    tipos: list = None,
    max_paginas_vacias: int = MAX_PAGINAS_VACIAS_CONSECUTIVAS,
    max_paginas_corrida: int = MAX_PAGINAS_POR_CORRIDA,
    max_minutos_corrida: float = MAX_MINUTOS_POR_CORRIDA,
) -> dict:
    comunas = comunas or sg.COMUNAS_GRAN_CONCEPCION
    tipos = tipos or TIPOS_PROPIEDAD_PRODUCCION

    ids_originales = obtener_ids_originales(con_original)
    ids_produccion = obtener_ids_produccion(con_produccion)
    ids_conocidos = ids_originales | ids_produccion
    log.info(f"{len(ids_originales)} avisos en la BD original, "
              f"{len(ids_produccion)} ya en producción.")

    total_vistos = 0
    total_nuevos = 0
    paginas_recorridas = 0
    motivo_corte = None
    t0 = time.time()

    for comuna in comunas:
        if motivo_corte in ("limite_paginas", "limite_tiempo"):
            break

        for tipo in tipos:
            if motivo_corte in ("limite_paginas", "limite_tiempo"):
                break

            log.info(f"--- Buscando: {sg.OPERACION} de {tipo} en {comuna} ---")
            paginas_vacias_consecutivas = 0
            pagina = 1

            while True:
                if paginas_recorridas >= max_paginas_corrida:
                    motivo_corte = "limite_paginas"
                    log.warning(f"Límite de {max_paginas_corrida} páginas por corrida alcanzado. "
                                f"Corte por presupuesto, no por agotar contenido nuevo.")
                    break

                if (time.time() - t0) / 60 >= max_minutos_corrida:
                    motivo_corte = "limite_tiempo"
                    log.warning(f"Límite de {max_minutos_corrida} minutos por corrida alcanzado. "
                                f"Corte por presupuesto, no por agotar contenido nuevo.")
                    break

                if paginas_vacias_consecutivas >= max_paginas_vacias:
                    log.info(f"  -> {max_paginas_vacias} páginas vacías consecutivas en esta "
                              f"combinación. Corto esta búsqueda y paso a la siguiente.")
                    break

                if pagina > sg.MAX_PAGINAS_POR_BUSQUEDA:
                    break

                url = sg.construir_url(tipo, comuna, pagina)
                log.info(f"Página {pagina}: {url}")
                html = sg.obtener_html(url)
                paginas_recorridas += 1

                if html is None:
                    break  # fin de resultados (404) o error/CAPTCHA

                avisos = sg.parsear_pagina(html, comuna, tipo)
                if not avisos:
                    break  # página sin tarjetas = fin de resultados real

                nuevos_en_pagina = guardar_pagina_en_produccion(avisos, con_produccion, ids_conocidos)
                total_nuevos += nuevos_en_pagina
                total_vistos += len(avisos)

                paginas_vacias_consecutivas = 0 if nuevos_en_pagina > 0 else paginas_vacias_consecutivas + 1

                log.info(f"  -> {len(avisos)} avisos vistos, {nuevos_en_pagina} nuevos guardados "
                          f"(vacías consecutivas: {paginas_vacias_consecutivas}/{max_paginas_vacias})")

                pagina += 1
                time.sleep(random.uniform(sg.DELAY_MIN, sg.DELAY_MAX))

    if motivo_corte is None:
        motivo_corte = "paginas_vacias_consecutivas"

    return {
        "total_vistos": total_vistos,
        "total_nuevos": total_nuevos,
        "paginas_recorridas": paginas_recorridas,
        "motivo_corte": motivo_corte,
        "duracion_seg": round(time.time() - t0, 1),
    }


if __name__ == "__main__":
    con_produccion = db.conectar_produccion()
    con_original = db.conectar_original()

    resumen = scrapear_grilla_incremental(con_produccion, con_original)

    con_produccion.close()
    con_original.close()

    log.info(
        f"Corrida completa. Avisos vistos: {resumen['total_vistos']} | "
        f"Nuevos guardados: {resumen['total_nuevos']} | "
        f"Páginas recorridas: {resumen['paginas_recorridas']} | "
        f"Motivo de corte: {resumen['motivo_corte']} | "
        f"Duración: {resumen['duracion_seg']:.1f}s"
    )
