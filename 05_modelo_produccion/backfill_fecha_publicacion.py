"""
Backfill ÚNICO de fecha_publicacion_texto/aprox/precision para avisos que
quedaron NULL por el bug de RE_FECHA_PUBLICACION (no reconocía "Publicado
hoy" ni "Publicado esta semana", solo "Publicado hace ..."). Ya corregido en
01_obtener_datos/02_scraper_detalle.py.

Reutiliza `visitar_aviso` de 02_scraper_detalle_incremental.py (mismo
guardado UPSERT, mismo manejo de CAPTCHA/reintentos/contador de fallos que
usa el pipeline real) tratando cada fila afectada como un re-chequeo.

Es un script de migración de una sola corrida (mismo patrón que
migracion_historico_a_produccion.py). Bórralo después de correrlo si ya no
lo necesitas.
"""

import importlib.util
import logging
from pathlib import Path

import db

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("scraper_detalle_incremental", SCRIPT_DIR / "02_scraper_detalle_incremental.py")
sdi = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sdi)


def main():
    con = db.conectar_produccion()

    cooldown = sdi.tiempo_restante_cooldown(con)
    if cooldown:
        log.warning(f"En cooldown tras un CAPTCHA reciente. Faltan ~{cooldown}. Corre esto más tarde.")
        return

    pendientes = con.execute("""
        SELECT a.id_aviso, a.url, a.comuna, a.tipo_propiedad
        FROM avisos_detalle d
        JOIN avisos a ON a.id_aviso = d.id_aviso
        WHERE d.fecha_publicacion_texto IS NULL
          AND a.estado_publicacion = 'activo'
    """).fetchall()

    log.info(f"Total pendientes de backfill: {len(pendientes)}")

    contadores = {"ok": 0, "cambio_estado": 0, "error": 0, "captcha": 0}

    for i, (id_aviso, url, comuna, tipo_propiedad) in enumerate(pendientes):
        log.info(f"[{i+1}/{len(pendientes)}] {id_aviso}")
        fila = {"id_aviso": id_aviso, "url": url, "comuna": comuna, "tipo_propiedad": tipo_propiedad}
        resultado = sdi.visitar_aviso(con, fila, es_rechequeo=True)
        contadores[resultado] = contadores.get(resultado, 0) + 1

        if resultado == "captcha":
            log.error("CAPTCHA detectado. Deteniendo el backfill ahora mismo (se puede resumir después, "
                      "ya que este script vuelve a consultar los pendientes reales en cada corrida).")
            break

    con.close()
    log.info(f"FIN. contadores={contadores}")


if __name__ == "__main__":
    main()
