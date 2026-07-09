"""
Utilitario de una sola vez: agrega las columnas `cantidad_estaciones_metro` y
`distancia_min_m_estaciones_metro` a la tabla avisos_detalle, y borra los
registros ya procesados para que 02_scraper_detalle.py los vuelva a traer
esta vez incluyendo los datos de metro.

A diferencia de migrar_poi.py, este NO filtra por `cantidad_paraderos IS NULL`
(esa columna ya está llena en tus registros actuales) - filtra específicamente
por `cantidad_estaciones_metro IS NULL`, que es la columna nueva y por lo
tanto va a estar vacía en TODOS los registros existentes.

Corre esto UNA VEZ, luego puedes borrar este archivo.
"""

import sqlite3
from pathlib import Path

RUTA_BD = Path(__file__).resolve().parent / "avisos_gran_concepcion.db"

COLUMNAS_NUEVAS = ["cantidad_estaciones_metro", "distancia_min_m_estaciones_metro"]

con = sqlite3.connect(RUTA_BD)

cur = con.execute("PRAGMA table_info(avisos_detalle)")
columnas_actuales = [fila[1] for fila in cur.fetchall()]

agregadas = 0
for columna in COLUMNAS_NUEVAS:
    if columna not in columnas_actuales:
        tipo = "INTEGER" if columna.startswith("cantidad_") else "REAL"
        con.execute(f"ALTER TABLE avisos_detalle ADD COLUMN {columna} {tipo}")
        agregadas += 1
        print(f"Columna agregada: {columna} ({tipo})")

con.commit()

if agregadas == 0:
    print("Las columnas de metro ya existían. Igual se revisan registros pendientes de reproceso.")

cur = con.execute("SELECT COUNT(*) FROM avisos_detalle WHERE cantidad_estaciones_metro IS NULL")
total_a_reprocesar = cur.fetchone()[0]
print(f"Registros sin datos de metro encontrados: {total_a_reprocesar}")

if total_a_reprocesar > 0:
    cur = con.execute("DELETE FROM avisos_detalle WHERE cantidad_estaciones_metro IS NULL")
    con.commit()
    print(f"Registros borrados (quedarán pendientes para reprocesar): {cur.rowcount}")
else:
    print("No había registros pendientes.")

con.close()
print("Listo. Corre 02_scraper_detalle.py para reprocesar esos avisos, ahora con estaciones de metro incluidas.")