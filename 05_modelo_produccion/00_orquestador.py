"""
Orquestador del pipeline de producción — Gran Concepción Rentals.

Ejecuta en orden: scraper grilla incremental -> scraper detalle incremental
(incluye re-chequeo de estado) -> vulnerabilidad -> ingeniería de variables
-> predicción (incluye inserción en `predicciones`).

Pensado para correr sin intervención manual vía cron / tarea programada
(diaria o varias veces al día). Cada corrida:
  - Registra una fila en la tabla `corridas` (metadatos: contadores, motivo
    de corte, versión de modelo usada, resultado final).
  - Loggea cada etapa (inicio/fin/resultado/errores) tanto a un archivo
    rotativo (`logs/orquestador.log`) como a la tabla `logs_ejecucion`.
  - Si una etapa lanza una excepción, se detiene ahí mismo (no sigue con
    las etapas siguientes usando datos parciales), loggea el error a nivel
    ERROR en ambos destinos, y marca la corrida como 'error'.
  - Si el scraper de detalle se detiene por CAPTCHA (no es una excepción,
    es un corte limpio y ya manejado por esa etapa), se sigue igual con las
    etapas siguientes sobre lo ya procesado, y la corrida queda marcada
    como 'parcial' en vez de 'ok'.
  - Al final, corre un chequeo de sanidad sobre el historial reciente de
    `corridas`: si hay N corridas consecutivas con 0 avisos nuevos en la
    grilla, deja un WARNING explícito (podría ser un problema silencioso,
    no necesariamente falta de contenido nuevo).
"""

import importlib.util
import logging
import sys
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path

import db

SCRIPT_DIR = Path(__file__).resolve().parent
LOGS_DIR = SCRIPT_DIR / "logs"
LOG_FILE = LOGS_DIR / "orquestador.log"

MAX_BYTES_LOG = 5 * 1024 * 1024   # 5 MB por archivo
BACKUP_COUNT_LOG = 5              # + 5 archivos rotados (orquestador.log.1 ... .5)

# Constante configurable: si hay esta cantidad de corridas consecutivas
# (resultado ok/parcial) con 0 avisos nuevos en la grilla, se alerta.
N_CORRIDAS_CONSECUTIVAS_SIN_NUEVOS = 5

log = logging.getLogger("orquestador")


# ------------------------------------------------------------------
# Carga perezosa de los módulos de cada etapa (deben cargarse DESPUÉS de
# `configurar_logging`, para que sus propios `logging.basicConfig(...)`
# internos sean no-op y todo termine fluyendo por los handlers de acá).
# ------------------------------------------------------------------
def _cargar_modulo(nombre: str, ruta: Path):
    spec = importlib.util.spec_from_file_location(nombre, ruta)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


# ------------------------------------------------------------------
# Etapa actual (variable global simple): la lee el handler de BD para
# etiquetar cada mensaje de log con la etapa que lo generó, sin importar
# desde qué módulo/sub-módulo anidado venga el log.
# ------------------------------------------------------------------
class _EtapaActual:
    valor = "orquestador"


etapa_actual = _EtapaActual()


class HandlerLogsEjecucion(logging.Handler):
    """Escribe cada mensaje de log también en la tabla `logs_ejecucion` de
    la base de datos de producción, etiquetado con la etapa vigente."""

    def __init__(self, con_produccion, id_corrida_holder):
        super().__init__()
        self.con = con_produccion
        self.id_corrida_holder = id_corrida_holder

    def emit(self, record):
        try:
            nivel = record.levelname.lower()
            if nivel not in ("info", "warning", "error"):
                nivel = "info"
            self.con.execute("""
                INSERT INTO logs_ejecucion (id_corrida, timestamp, etapa, nivel, mensaje)
                VALUES (?, datetime('now'), ?, ?, ?)
            """, (self.id_corrida_holder.valor, etapa_actual.valor, nivel, self.format(record)))
            self.con.commit()
        except Exception:
            pass  # un fallo de logging nunca debe tumbar la corrida


def configurar_logging(con_produccion, id_corrida_holder) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    formato_archivo = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    file_handler = RotatingFileHandler(
        LOG_FILE, maxBytes=MAX_BYTES_LOG, backupCount=BACKUP_COUNT_LOG, encoding="utf-8",
    )
    file_handler.setFormatter(formato_archivo)
    root.addHandler(file_handler)

    db_handler = HandlerLogsEjecucion(con_produccion, id_corrida_holder)
    db_handler.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(db_handler)


# ------------------------------------------------------------------
# Tabla `corridas`
# ------------------------------------------------------------------
def crear_corrida(con) -> int:
    cur = con.execute("INSERT INTO corridas (fecha_inicio, resultado) VALUES (datetime('now'), 'parcial')")
    con.commit()
    return cur.lastrowid


def actualizar_corrida(con, id_corrida: int, r: dict) -> None:
    con.execute("""
        UPDATE corridas SET
            fecha_fin = datetime('now'),
            resultado = ?,
            version_modelo_usada = ?,
            avisos_nuevos_grilla = ?,
            avisos_nuevos_detalle = ?,
            avisos_rechequeados = ?,
            avisos_cambio_estado = ?,
            paginas_recorridas_grilla = ?,
            motivo_corte_grilla = ?,
            etapa_fallida = ?,
            mensaje_error = ?
        WHERE id_corrida = ?
    """, (
        r["resultado"], r["version_modelo_usada"],
        r["avisos_nuevos_grilla"], r["avisos_nuevos_detalle"],
        r["avisos_rechequeados"], r["avisos_cambio_estado"],
        r["paginas_recorridas_grilla"], r["motivo_corte_grilla"],
        r["etapa_fallida"], r["mensaje_error"],
        id_corrida,
    ))
    con.commit()


def chequeo_sanidad_grilla(con, n_corridas: int = N_CORRIDAS_CONSECUTIVAS_SIN_NUEVOS) -> None:
    """
    Si las últimas `n_corridas` corridas exitosas/parciales tuvieron 0
    avisos nuevos en la grilla, deja un WARNING: podría ser un problema
    silencioso (cambio de estructura del sitio, bloqueo no detectado) en
    vez de simple falta de contenido nuevo. Las corridas con resultado
    'error' se excluyen (esas ya son ruidosas por su cuenta).
    """
    filas = con.execute("""
        SELECT avisos_nuevos_grilla FROM corridas
        WHERE resultado IN ('ok', 'parcial')
        ORDER BY id_corrida DESC
        LIMIT ?
    """, (n_corridas,)).fetchall()

    if len(filas) < n_corridas:
        return

    if all(f[0] == 0 for f in filas):
        log.warning(
            f"{n_corridas} corridas consecutivas con 0 avisos nuevos en el scraper de grilla. "
            f"Esto podría indicar un problema silencioso (cambio en la estructura del sitio, "
            f"bloqueo no detectado) en vez de simplemente no haber contenido nuevo."
        )


# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------
def main():
    con_produccion = db.conectar_produccion()
    con_original = db.conectar_original()

    id_corrida = crear_corrida(con_produccion)
    id_corrida_holder = _EtapaActual()
    id_corrida_holder.valor = id_corrida

    configurar_logging(con_produccion, id_corrida_holder)

    log.info(f"=== Inicio de corrida #{id_corrida} ===")

    r = {
        "resultado": "ok",
        "version_modelo_usada": None,
        "avisos_nuevos_grilla": 0,
        "avisos_nuevos_detalle": 0,
        "avisos_rechequeados": 0,
        "avisos_cambio_estado": 0,
        "paginas_recorridas_grilla": 0,
        "motivo_corte_grilla": None,
        "etapa_fallida": None,
        "mensaje_error": None,
    }
    hubo_corte_parcial = False

    try:
        etapa_actual.valor = "scraper_grilla"
        log.info("Iniciando etapa: scraper_grilla")
        t0 = time.time()
        grilla = _cargar_modulo("produccion_scraper_grilla", SCRIPT_DIR / "01_scraper_grilla_incremental.py")
        resumen = grilla.scrapear_grilla_incremental(con_produccion, con_original)
        log.info(f"Etapa scraper_grilla completada en {time.time()-t0:.1f}s: {resumen}")
        r["avisos_nuevos_grilla"] = resumen["total_nuevos"]
        r["paginas_recorridas_grilla"] = resumen["paginas_recorridas"]
        r["motivo_corte_grilla"] = resumen["motivo_corte"]

        etapa_actual.valor = "scraper_detalle"
        log.info("Iniciando etapa: scraper_detalle")
        t0 = time.time()
        detalle = _cargar_modulo("produccion_scraper_detalle", SCRIPT_DIR / "02_scraper_detalle_incremental.py")
        resumen = detalle.scrapear_detalle_incremental(con_produccion)
        log.info(f"Etapa scraper_detalle completada en {time.time()-t0:.1f}s: {resumen}")
        r["avisos_nuevos_detalle"] = resumen["nuevos_procesados"]
        r["avisos_rechequeados"] = resumen["rechequeos_procesados"]
        r["avisos_cambio_estado"] = resumen["cambios_estado"]
        if resumen["detenido_por_captcha"]:
            log.warning("scraper_detalle se detuvo por CAPTCHA. Se continúa con las etapas "
                        "siguientes sobre lo ya procesado; la corrida queda como 'parcial'.")
            hubo_corte_parcial = True

        etapa_actual.valor = "vulnerabilidad"
        log.info("Iniciando etapa: vulnerabilidad")
        t0 = time.time()
        vulnerabilidad = _cargar_modulo("produccion_vulnerabilidad", SCRIPT_DIR / "03_vulnerabilidad_produccion.py")
        resumen = vulnerabilidad.procesar_vulnerabilidad(con_produccion)
        log.info(f"Etapa vulnerabilidad completada en {time.time()-t0:.1f}s: {resumen}")

        etapa_actual.valor = "variables"
        log.info("Iniciando etapa: variables")
        t0 = time.time()
        variables = _cargar_modulo("produccion_variables", SCRIPT_DIR / "04_ingenieria_variables_produccion.py")
        features_df = variables.construir_features_produccion(con_produccion, con_original)
        log.info(f"Etapa variables completada en {time.time()-t0:.1f}s: "
                  f"{len(features_df)} avisos con features calculadas")

        etapa_actual.valor = "prediccion"
        log.info("Iniciando etapa: prediccion")
        t0 = time.time()
        prediccion = _cargar_modulo("produccion_prediccion", SCRIPT_DIR / "05_prediccion.py")
        resumen = prediccion.generar_predicciones(con_produccion, con_original, features_df=features_df)
        log.info(f"Etapa prediccion completada en {time.time()-t0:.1f}s: {resumen}")
        r["version_modelo_usada"] = resumen["version_modelo"]

        r["resultado"] = "parcial" if hubo_corte_parcial else "ok"

    except Exception as e:
        etapa_fallida = etapa_actual.valor
        log.error(f"Fallo en la etapa '{etapa_fallida}': {e}", exc_info=True)
        r["resultado"] = "error"
        r["etapa_fallida"] = etapa_fallida
        r["mensaje_error"] = str(e)

    finally:
        etapa_actual.valor = "orquestador"
        actualizar_corrida(con_produccion, id_corrida, r)

        if r["resultado"] in ("ok", "parcial"):
            chequeo_sanidad_grilla(con_produccion)

        log.info(f"=== Fin de corrida #{id_corrida}: resultado={r['resultado']} ===")

        con_produccion.close()
        con_original.close()

    return r["resultado"]


if __name__ == "__main__":
    if main() == "error":
        # Process no cero: para que GitHub Actions marque el job (y por lo
        # tanto la corrida programada) como fallido y avise, en vez de un
        # check verde silencioso mientras el detalle del fallo solo queda
        # en la tabla `corridas`/`logs_ejecucion`. El commit/push de la BD
        # sigue corriendo igual (el workflow lo marca con `if: always()`),
        # así el diagnóstico llega al repo aunque el job termine en rojo.
        sys.exit(1)
