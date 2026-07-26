"""
Snapshot diario de historial de avisos — etapa del pipeline de producción.

Guarda, en `historial_diario_avisos`, el estado_publicacion de cada aviso
departamento/arriendo tal como está HOY. Solo se guarda el estado (no
precio/comuna/etc.): la pestaña de estadísticas diarias del visualizador
cruza este historial contra los atributos ACTUALES del aviso (que rara vez
cambian día a día) para poder filtrarlo con los mismos filtros del
buscador, sin tener que re-derivar el precio en CLP de cada día histórico.

El orquestador puede correr varias veces el mismo día: UPSERT sobre
(fecha, id_aviso) hace que cada corrida sobreescriba la fila de hoy con el
estado más reciente en vez de duplicarla — interesa "cómo quedó el día",
no cada corrida individual.
"""

import logging
from datetime import date

from psycopg2.extras import execute_values

log = logging.getLogger(__name__)


def guardar_snapshot(con_produccion, fecha: str, filas: list[tuple[str, str]]) -> None:
    """`filas` es una lista de (id_aviso, estado_publicacion). Usado tanto por
    `capturar_snapshot_diario` (estado en vivo) como por
    `backfill_historial_diario.py` (estado leído de un commit histórico de la
    BD vía git).

    execute_values (no executemany): son ~2400 avisos por corrida, y
    executemany hace un round-trip de red por fila contra el pooler de
    Supabase (varios minutos en la práctica) - execute_values manda todas
    las filas en un solo INSERT multi-valor."""
    execute_values(con_produccion.cursor(), """
        INSERT INTO historial_diario_avisos (fecha, id_aviso, estado_publicacion)
        VALUES %s
        ON CONFLICT (fecha, id_aviso) DO UPDATE SET estado_publicacion = excluded.estado_publicacion
    """, [(fecha, id_aviso, estado) for id_aviso, estado in filas])
    con_produccion.commit()


def capturar_snapshot_diario(con_produccion, fecha: str | None = None) -> dict:
    fecha = fecha or date.today().isoformat()

    cur = con_produccion.cursor()
    cur.execute("""
        SELECT id_aviso, estado_publicacion
        FROM avisos
        WHERE tipo_propiedad = 'departamento' AND operacion = 'arriendo'
    """)
    filas = cur.fetchall()

    guardar_snapshot(con_produccion, fecha, filas)

    log.info(f"Snapshot diario de {fecha}: {len(filas)} avisos capturados.")
    return {"fecha": fecha, "avisos_capturados": len(filas)}
