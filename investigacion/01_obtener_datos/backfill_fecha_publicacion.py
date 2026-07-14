"""
Backfill ÚNICO de fecha_publicacion_texto/aprox/precision para avisos que
quedaron NULL por el bug de RE_FECHA_PUBLICACION (no reconocía "Publicado
hoy" ni "Publicado esta semana", solo "Publicado hace ..."). Ya corregido en
02_scraper_detalle.py - este script re-visita esos avisos para completar los
campos con la lógica ya arreglada.

Es un script de migración de una sola corrida (mismo patrón que
produccion/01_modelo_produccion/migracion_historico_a_produccion.py). Bórralo después
de correrlo si ya no lo necesitas.
"""

import importlib.util
import logging
import random
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("scraper_detalle", SCRIPT_DIR / "02_scraper_detalle.py")
sd = importlib.util.module_from_spec(spec)
sys.modules["scraper_detalle"] = sd
spec.loader.exec_module(sd)


def main():
    con = sd.inicializar_bd()

    cooldown = sd.tiempo_restante_cooldown(con)
    if cooldown:
        log.warning(f"En cooldown tras un CAPTCHA reciente. Faltan ~{cooldown}. Corre esto más tarde.")
        return

    pendientes = con.execute("""
        SELECT d.id_aviso, a.url, a.comuna, a.tipo_propiedad
        FROM avisos_detalle d
        JOIN avisos a ON a.id_aviso = d.id_aviso
        WHERE d.fecha_publicacion_texto IS NULL
    """).fetchall()

    log.info(f"Total pendientes de backfill: {len(pendientes)}")

    contadores = {"ok": 0, "error": 0, "captcha": 0}
    categorias = {}

    for i, (id_aviso, url, comuna, tipo_propiedad) in enumerate(pendientes):
        log.info(f"[{i+1}/{len(pendientes)}] {id_aviso}")
        resultado = sd.obtener_detalle_aviso(url, comuna, tipo_propiedad)

        if resultado["resultado"] == "captcha":
            log.error("CAPTCHA detectado. Deteniendo el backfill ahora mismo (se puede resumir después, "
                      "ya que este script vuelve a consultar los pendientes reales en cada corrida).")
            sd.registrar_captcha(con)
            contadores["captcha"] += 1
            break

        if resultado["resultado"] == "error":
            log.warning(f"  -> error ({resultado['motivo']}), se deja pendiente para otra corrida")
            contadores["error"] += 1
            time.sleep(random.uniform(sd.DELAY_MIN, sd.DELAY_MAX))
            continue

        datos = resultado["datos"]
        sd.guardar_detalle(con, id_aviso, url, datos)
        contadores["ok"] += 1
        cat = datos.get("fecha_publicacion_texto")
        categorias[cat] = categorias.get(cat, 0) + 1
        log.info(f"  -> ok. texto={cat!r} aprox={datos.get('fecha_publicacion_aprox')} "
                  f"precision={datos.get('fecha_publicacion_precision')}")

        time.sleep(random.uniform(sd.DELAY_MIN, sd.DELAY_MAX))

    con.close()
    log.info(f"FIN. contadores={contadores}")
    log.info(f"categorias recuperadas: {categorias}")


if __name__ == "__main__":
    main()
