"""
Utilitario de una sola vez: vacía por completo la tabla `avisos_detalle`
(mantiene la estructura de columnas, solo borra los datos), para que
02_scraper_detalle.py vuelva a procesar TODOS los avisos desde cero con la
extracción corregida (JSON embebido en vez de HTML renderizado).

No toca la tabla `avisos` (la lista de avisos descubiertos por el scraper
de grilla se mantiene intacta).

Corre esto UNA VEZ, luego puedes borrar este archivo.
"""

import sqlite3
from pathlib import Path

RUTA_BD = Path(__file__).resolve().parent / "avisos_gran_concepcion.db"

con = sqlite3.connect(RUTA_BD)

cur = con.execute("SELECT COUNT(*) FROM avisos_detalle")
total_actual = cur.fetchone()[0]
print(f"Registros actuales en avisos_detalle: {total_actual}")

if total_actual == 0:
    print("La tabla ya está vacía, no hay nada que borrar.")
else:
    respuesta = input(f"¿Confirmas borrar los {total_actual} registros de avisos_detalle? (escribe 'si' para continuar): ")
    if respuesta.strip().lower() == "si":
        cur = con.execute("DELETE FROM avisos_detalle")
        con.commit()
        print(f"Listo. Se borraron {cur.rowcount} registros.")
        con.execute("VACUUM")  # recupera espacio en disco tras el borrado masivo
        print("Espacio en disco recuperado (VACUUM).")
    else:
        print("Cancelado. No se borró nada.")

con.close()
print("Ahora puedes correr 02_scraper_detalle.py para reprocesar todos los avisos.")
