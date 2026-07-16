# Databricks — guía de aprendizaje (no reemplaza el pipeline actual)

Instrucciones para montar, en Databricks, una réplica de aprendizaje del pipeline de datos de
[gran-concepcion-rentals](README.md). **Esto no reemplaza el proyecto tal como está** (SQLite +
GitHub Actions + Streamlit, ver [README](README.md) y [SETUP.md](SETUP.md)) — es un ejercicio
paralelo para aprender la plataforma, empezando por lo más simple (cargar datos ya existentes) y
subiendo en complejidad (escritura incremental, transformación en SQL, y en una etapa futura,
inferencia del modelo).

---

## 0. Alcance de esta guía

| Decisión | Elegido |
|---|---|
| Edición de Databricks | **Free Edition** (compute serverless, Unity Catalog activado por defecto, sin tarjeta de crédito) |
| Scraper en Databricks | **Solo cargas manuales** — corrés el notebook a mano cuando quieras practicar, no reemplaza el cron de GitHub Actions |
| Cruce de vulnerabilidad (IGVUST) | **Tabla propia en Bronce** — se sube `poligonos_vulnerabilidad_uv` igual que `avisos`/`avisos_detalle`, y el cruce punto-en-polígono se replica en Plata (Python + shapely), en vez de copiar columnas ya resueltas |
| Predicción con el modelo | **Fuera de alcance de esta guía** — queda para una segunda etapa, una vez que bronce/plata/oro estén andando |

Como es un ejercicio de aprendizaje, esta guía prioriza que cada fase sea simple de verificar antes
de pasar a la siguiente, no fidelidad 1:1 con el pipeline de producción real.

---

## 1. Arquitectura objetivo: bronce / plata / oro

| Capa | Contenido | Cómo se llena |
|---|---|---|
| 🥉 **Bronce** | `avisos` y `avisos_detalle`, crudos, tal como los devuelve el scraper (texto sin parsear, ej. `"82.000"`, fechas relativas) | Fase 1 (carga inicial desde SQLite) + Fase 2 (notebook de scraper manual) |
| 🥈 **Plata** | Un registro limpio por aviso: números chilenos parseados, `gastos_comunes` corregido, outliers filtrados, superficie/antigüedad imputadas, columnas de vulnerabilidad copiadas | Fase 3a — mezcla SQL y Python (detalle en sección 6) |
| 🥇 **Oro** | Tabla de features lista para el modelo: distancias Haversine, `precio_m2_sector_departamento`, `nivel_barrio`, `ratio_total_util`, etc. | Fase 3b — SQL puro sobre Plata (sección 7) |

La tabla `predicciones` de una futura Etapa 2 también sería Oro (consume de la tabla Oro de
features, no de Plata directamente) — no se cubre en esta guía.

---

## 2. Preparación del workspace (Free Edition)

1. Crear cuenta en Databricks Free Edition (no requiere tarjeta).
2. Dentro del workspace, en **Catalog**, crear un catálogo y esquema propios para este ejercicio
   en vez de usar `workspace.default` — por ejemplo `gran_concepcion.bronce`,
   `gran_concepcion.plata`, `gran_concepcion.oro` (un esquema por capa, así el nombre de la tabla
   no necesita prefijo/sufijo de capa).
3. Crear un **Volume** dentro de uno de esos esquemas (ej. `gran_concepcion.bronce.landing`) — es
   el lugar donde vas a subir los archivos exportados desde SQLite antes de convertirlos en tabla
   Delta.
4. Verificar que el compute serverless tiene salida a internet: en un notebook nuevo, correr un
   `requests.get(...)` simple contra cualquier URL antes de portar el scraper completo (Fase 2) —
   si falla acá, no tiene sentido seguir con esa fase todavía.

---

## 3. Fase 1 — Bronce: cargar grilla + detalle

**Origen recomendado**: `produccion/01_modelo_produccion/produccion_gran_concepcion.db`, tablas
`avisos`, `avisos_detalle` y `poligonos_vulnerabilidad_uv` (ver esquema en
[`db.py`](produccion/01_modelo_produccion/db.py)) — preferible a la base de investigación porque
`avisos_detalle` ya trae columnas de coordenadas listas para el cruce, sin el join extra que tiene
`avisos_gran_concepcion.db` contra `vulnerabilidad_uv`/`avisos_igvust`.

1. Exportar las tres tablas a CSV o Parquet localmente (por ejemplo con `sqlite3` + `pandas`, en tu
   máquina, fuera de Databricks — no hace falta que esto corra en la nube). En
   `poligonos_vulnerabilidad_uv`, la columna `geometria_wkt` es texto plano (WKT), así que viaja
   sin problema en CSV/Parquet igual que cualquier otra columna.
2. Subir los archivos al Volume creado en el paso 2.3, vía drag-and-drop en la UI de **Catalog
   Explorer**.
3. Usar el asistente **"Create table from file"** (o `CREATE TABLE ... AS SELECT * FROM
   read_files('/Volumes/gran_concepcion/bronce/landing/avisos.csv')`) para materializar
   `gran_concepcion.bronce.avisos`, `gran_concepcion.bronce.avisos_detalle` y
   `gran_concepcion.bronce.poligonos_vulnerabilidad_uv` como tablas Delta.
4. Verificar recuento de filas contra la base SQLite de origen antes de seguir — si no coincide,
   revisar el export antes de tocar nada más.

`poligonos_vulnerabilidad_uv` queda como tabla estática en Bronce (se re-sube a mano si el
shapefile IGVUST de origen se actualiza, igual que en producción) — el cruce contra `avisos_detalle`
recién se resuelve en Plata (sección 5).

---

## 4. Fase 2 — Bronce: scraper manual (notebook)

Notebook nuevo (no toca el código del repo local) que reutiliza la lógica de
[`01_scraper_grilla.py`](investigacion/01_obtener_datos/01_scraper_grilla.py) /
[`02_scraper_detalle.py`](investigacion/01_obtener_datos/02_scraper_detalle.py), con dos
diferencias respecto al original:

- **Sin persistencia fila por fila**: SQLite hace `INSERT OR IGNORE` con commit inmediato por
  página/aviso; en Delta eso se traduce mal (muchas transacciones chicas). En vez de eso, juntá
  los avisos nuevos de **toda la corrida manual** en un DataFrame de Spark y hacé un solo `MERGE
  INTO ... ON avisos.id_aviso = nuevos.id_aviso WHEN NOT MATCHED THEN INSERT` al final.
- **Sin scheduling**: correr el notebook a mano cuando quieras cargar avisos nuevos — no hay Job
  programado (decisión de la sección 0).

Mantené la misma lógica de extracción (regex, parsing de POIs desde `window._n.ctx.r`, etc.) que
ya está validada en el proyecto original — lo que cambia acá es solo *dónde y cómo* se escribe el
resultado, no cómo se scrapea.

---

## 5. Fase 3a — Plata: limpieza

Reimplementando lo que hoy hace
[`01_ingenieria_variables.py`](investigacion/03_ingenieria_variables/01_ingenieria_variables.py),
separado por si conviene SQL o Python:

**En SQL** (transformación set-based, sin dependencias externas):
- Parseo del separador de miles chileno (`REPLACE`/`CASE WHEN` sobre el texto crudo).
- El fix de `gastos_comunes` de tres casos — con punto (miles real), dígito suelto <1.000 sin
  punto (placeholder → `0.0`), sin punto y ≥500.000 (outlier → `NULL`) — ver
  [README, sección 3.3](README.md#33-el-bug-de-gastos_comunes-impacto-medible-en-el-modelo) para
  el detalle de cada caso.
- Filtros de valores imposibles (dormitorios, baños, estacionamientos) y tope de precio
  (8.000.000 CLP).
- La conversión UF/USD→CLP en sí (el *join* contra una tabla de tasas ya cacheada).
- Imputación de antigüedad por vecino geográfico: es *nearest-neighbor* por distancia, no requiere
  el modelo entrenado — se puede hacer con un self-join + `ROW_NUMBER()` ordenado por distancia
  Haversine, con la misma cascada de fallbacks (vecino → mediana por tipo → por comuna → global)
  que ya usa el original.

**En Python** (no reduce razonablemente a SQL):
- El fetch de la tasa UF a la API de `mindicador.cl` (llamada HTTP externa) — cachear el resultado
  en una tabla chica `gran_concepcion.plata.valores_uf`, para que el resto sea un join en SQL.
- Imputación de superficie corrupta con `RandomForestRegressor` — necesita `scikit-learn`, sin
  traducción razonable a SQL. Corre como notebook Python normal (no requiere Spark), escribe el
  resultado de vuelta a la tabla Plata.
- El cruce punto-en-polígono de vulnerabilidad: mismo enfoque que
  [`03_vulnerabilidad_produccion.py`](produccion/01_modelo_produccion/03_vulnerabilidad_produccion.py)
  — leer `gran_concepcion.bronce.poligonos_vulnerabilidad_uv`, parsear `geometria_wkt` con
  `shapely.wkt.loads`, y para cada aviso con coordenadas válidas encontrar el polígono que lo
  contiene (`Point(longitud, latitud).within(...)`, o `geopandas.sjoin` si el volumen de datos lo
  justifica) para resolver `uv_rsh`, `rank_nac`, `pob_rsh_uv`, `p_urbano`, `c_ig_com`. Igual que la
  imputación de superficie, corre como notebook Python normal (no Spark, el dataset es chico) y
  escribe el resultado a la tabla Plata — no hace falta `geopandas`/GDAL si te alcanza con
  `shapely` puro sobre el WKT ya guardado.

---

## 6. Fase 3b — Oro: ingeniería de variables

Sobre la tabla Plata ya limpia, 100% SQL (agregaciones y window functions):

- Distancias Haversine al centro de la comuna y al centro de Concepción.
- `ratio_total_util` (superficie total / superficie útil).
- `nivel_barrio`: precio/m² promedio por barrio, suavizado hacia la media general, agrupado en 5
  niveles con cuantiles ponderados por cantidad de avisos (`PERCENTILE_CONT` sobre una expansión
  ponderada, o una window function equivalente).
- `precio_m2_sector_departamento`: el más interesante para practicar SQL — self-join contra otros
  avisos dentro de 300m (excluyendo la propia fila vía `id_aviso != id_aviso`), filtro IQR
  (×3) sobre el precio/m² del sector, mediana de los vecinos válidos, con fallback a la mediana
  general cuando no hay vecinos (marcado en `tiene_comparables_cercanos`) — ver
  [README, sección 3.1](README.md#31-fuga-de-datos-en-precio_m2) para la lógica completa que hay
  que replicar. Al tamaño de este dataset (~1.600-3.000 filas) un cross join filtrado por
  Haversine anda bien sin necesidad de índices espaciales.

El resultado es el equivalente Oro de `datos_ingenieria_variables.csv` — la tabla lista para
alimentar un modelo, cuando llegues a esa etapa.

---

## 7. Etapa 2 (futuro, fuera de alcance de esta guía): predicciones

Cuando bronce/plata/oro estén andando, el siguiente paso natural sería registrar el modelo
(`lightgbm_regression_precio.pkl`, ensamble de 10) en **MLflow** (nativo de Databricks) y aplicar
batch inference sobre la tabla Oro. No se detalla acá porque todavía no está definido cómo — queda
para retomar en una conversación aparte una vez completadas las fases 1-6.

---

## 8. Limitaciones y decisiones pendientes

- **Free Edition tiene cuotas de compute**: si el self-join de `precio_m2_sector_departamento`
  (sección 6) resulta pesado, vale la pena revisar el plan de ejecución antes de asumir que hace
  falta más compute.
- **Esta guía no cubre automatización**: no hay Jobs programados ni alertas — es coherente con la
  decisión de la sección 0 de que el scraper en Databricks es solo para practicar, no para
  reemplazar el cron actual.
- **No hay sincronización de vuelta** hacia `produccion_gran_concepcion.db`: los datos que entren
  a Databricks quedan ahí: este ejercicio no alimenta el dashboard Streamlit ni el pipeline real.
- Antes de escalar el volumen de scraping manual (Fase 2), revisar el `robots.txt` / Términos de
  Uso del sitio, igual que se indica en [SETUP.md](SETUP.md#2-cómo-correrlo).
