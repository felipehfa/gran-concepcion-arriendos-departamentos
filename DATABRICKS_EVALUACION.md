# Evaluación de migración a Databricks — `05_modelo_produccion/`

> Documento de dos partes: (1) diagnóstico de viabilidad de migrar el pipeline de producción completo a Databricks, y (2) guía de un POC acotado para aprender Databricks (Delta Lake + MERGE INTO, Jobs/Workflows) sin comprometerse a migrar el proyecto real.

> **Nota de actualización (2026-07-13)**: el diagnóstico original se escribió antes de que existiera `.github/workflows/orquestador.yml`. Desde entonces se confirmó un runner real en GitHub Actions (corriendo cada 4h) y se refactorizó `03_vulnerabilidad_produccion.py` para dejar de depender de `geopandas`/el shapefile en cada corrida. Las secciones 1 y 2 de la Parte 1 se corrigieron para reflejar el estado actual; el resto del diagnóstico (base de datos, orquestador, entrenamiento, recomendación final) sigue vigente.

---

# Parte 1 — Diagnóstico de viabilidad

## 1. Inventario de dependencias y tecnologías

| Script | Librerías propias | Vía importlib (ruta relativa) |
|---|---|---|
| `00_orquestador.py` | `logging`, `RotatingFileHandler`, `importlib.util`, `pathlib` | carga las 5 etapas |
| `01_scraper_grilla_incremental.py` | `requests`-based (heredado) | `01_obtener_datos/01_scraper_grilla.py` |
| `02_scraper_detalle_incremental.py` | `pandas` | `01_obtener_datos/02_scraper_detalle.py` |
| `03_vulnerabilidad_produccion.py` | `shapely`, `pandas` (**ya no `geopandas`**, ver corrección abajo) | ninguno (reimplementa a propósito, ver docstring) |
| `04_ingenieria_variables_produccion.py` | `numpy`, `pandas`, `sklearn.neighbors.BallTree` | `03_ingenieria_variables/01_ingenieria_variables.py` |
| `05_prediccion.py` | `numpy`, `pandas`, `pickle` | `04_modelamiento/{01_xgboost,02_lightgbm}.py` + `04_ingenieria_variables_produccion.py` |
| `entrenamiento/01_entrenar_modelo_produccion.py` | `numpy`, `pandas`, `sklearn.model_selection`, `hashlib` | `04_modelamiento/{01_xgboost,02_lightgbm}.py` (que a su vez traen `xgboost`/`lightgbm`, `optuna`, `scipy.stats`) |
| `entrenamiento/seleccionar_algoritmo.py` | solo `json`/`argparse` | ninguno |

**Playwright — confirmado**: aparece en 5 archivos del repo, pero dentro de `05_modelo_produccion/` **no aparece en absoluto**. El único lugar real es `01_obtener_datos/02_scraper_detalle.py`, donde:
- El import (`from playwright.sync_api import ...`) está **dentro** de `main_fallback_playwright()`, no a nivel de módulo — perezoso.
- Esa función solo se invoca con `--fallback-playwright` explícito en CLI, nunca automáticamente.
- `02_scraper_detalle_incremental.py` (producción) llama únicamente a `sd.obtener_detalle_aviso(...)`, que es la ruta `requests`. El orquestador nunca toca la ruta Playwright.

**SHAP**: `04_modelamiento/02_lightgbm.py` usa TreeSHAP **nativo** de LightGBM (`Booster.predict(..., pred_contrib=True)`), sin la librería `shap` externa.

**Dependencias de sistema/rutas**:
- Todos los scripts resuelven rutas con `Path(__file__).resolve().parent` + `REPO_ROOT` — no hay rutas absolutas de Windows hardcodeadas, pero **sí asumen que todo el repo (`01_obtener_datos/`, `03_ingenieria_variables/`, `04_modelamiento/`, `05_modelo_produccion/`) está presente junto**, con la misma estructura relativa.
- `sqlite3.connect()` sobre archivos locales: `produccion_gran_concepcion.db` (r/w) y `avisos_gran_concepcion.db` (modo URI `?mode=ro`).
- **Corregido tras un refactor posterior**: `03_vulnerabilidad_produccion.py` **ya no lee el shapefile en cada corrida** ni depende de `geopandas`. El cruce punto-en-polígono se resuelve con `shapely` puro contra la tabla `poligonos_vulnerabilidad_uv` (polígonos IGVUST precalculados una vez, guardados como WKT en la propia BD de producción). El shapefile y `geopandas` solo hacen falta para `migrar_poligonos_vulnerabilidad.py`, que se corre a mano y localmente cuando el shapefile se actualiza — nunca dentro del pipeline automatizado. Esto **elimina una fricción real** que este diagnóstico originalmente contaba contra la portabilidad del pipeline (una dependencia de sistema pesada, GDAL, y acceso a un archivo externo no versionado, ambos fuera del runner de CI).
- `RotatingFileHandler` escribe a `05_modelo_produccion/logs/orquestador.log`.
- **Supuesto corregido**: al momento de escribir el diagnóstico original no existía ningún workflow en `.github/workflows/`, y se asumía que "GitHub Actions" en el README era solo un ejemplo hipotético. Esto ya no es así: `.github/workflows/orquestador.yml` es real y es el mecanismo de ejecución vigente — corre en un runner `ubuntu-latest`, con cron (`0 */4 * * *`, cada 4h; subió de 12h a 6h a 4h en tres cambios sucesivos) más `workflow_dispatch` para disparo manual. El job instala `05_modelo_produccion/requirements.txt` (ahora explícitamente sin `geopandas`, con `lxml` agregado), corre `00_orquestador.py`, y comitea/pushea la BD actualizada con `if: always()` — el diagnóstico (`corridas`/`logs_ejecucion`) llega al repo incluso si el job termina en rojo. Esto es relevante para la Sección 4: el patrón "Job con pasos secuenciales + notificación de fallo" que se buscaría en Databricks Jobs **ya existe hoy**, corriendo en CI en vez de localmente.
- **Fricción nueva encontrada al operar el workflow**: al comitear la BD desde el runner, corridas concurrentes (o un push manual mientras corre una programada) pueden generar conflictos de git. Se agregó `git pull --rebase origin main` antes del `push` en el paso de commit para evitarlos — un problema de concurrencia distinto al incidente de Google Drive de abajo, pero de la misma familia: **versionar una base SQLite completa en git tiene costos de concurrencia crecientes** a medida que aumenta la frecuencia de corridas (ya en 4h/día). Es una señal más a favor de migrar la persistencia a algo remoto (Delta, o incluso solo una base gestionada) si la frecuencia sigue subiendo — no necesariamente a favor de Databricks completo (ver recomendación en sección 6).
- **Incidente ya documentado (README/CHANGELOG)**: el repo vive dentro de una carpeta sincronizada por Google Drive. Una corrida de migración con ~1628 commits SQLite seguidos perdió 25 filas silenciosamente — causa más probable: interferencia del sincronizador de Drive con el archivo `.db` durante la ventana de escritura larga.

## 2. La base de datos: SQLite vs. tablas Delta

**Esquema completo** (`db.py`, `inicializar_bd_produccion`):

| Tabla | PK / constraints clave |
|---|---|
| `avisos` | `id_aviso TEXT PRIMARY KEY`; NOT NULL en comuna/tipo_propiedad/operacion; `estado_publicacion` CHECK IN (activo, pausado, finalizado, no_disponible), DEFAULT 'activo'; `intentos_fallidos_detalle` DEFAULT 0 |
| `avisos_detalle` | `id_aviso TEXT PRIMARY KEY REFERENCES avisos(id_aviso)` (FK); ~40 columnas incl. 11 subcategorías POI × 2 + columnas de vulnerabilidad |
| `poligonos_vulnerabilidad_uv` *(nueva desde el refactor de geopandas)* | `uv_rsh TEXT PRIMARY KEY`; `geometria_wkt TEXT NOT NULL` (polígono IGVUST en WKT/EPSG:4326) + columnas escalares (`rank_nac`, `pob_rsh_uv`, `p_urbano`, `c_ig_com`) — poblada una única vez vía `migrar_poligonos_vulnerabilidad.py`, de solo lectura para el resto del pipeline |
| `predicciones` | `id INTEGER PRIMARY KEY AUTOINCREMENT`; FK a avisos; `etiqueta`/`nivel_confianza` CHECK; **UNIQUE(id_aviso, version_modelo)** |
| `corridas` | `id_corrida INTEGER PRIMARY KEY AUTOINCREMENT`; `resultado` CHECK IN (ok, error, parcial); `motivo_corte_grilla` CHECK |
| `logs_ejecucion` | `id INTEGER PRIMARY KEY AUTOINCREMENT`; FK a corridas; `etapa` CHECK IN (7 valores); `nivel` CHECK |
| `control` | `clave TEXT PRIMARY KEY` (key/value genérico) |

**Qué se traduce directo a Delta/Unity Catalog**:
- Tipos escalares — sin fricción.
- PRIMARY KEY / FOREIGN KEY existen en Unity Catalog como constraints **informativos** (no enforced por defecto) — SQLite sí los enforcea hoy (`PRAGMA foreign_keys=ON`), así que hay enforcement real que se perdería.
- CHECK constraints: Delta soporta `ALTER TABLE ... ADD/DROP CONSTRAINT ... CHECK (...)` — **más simple** que hoy: en SQLite cambiar un CHECK obliga a reconstruir la tabla completa (`_migrar_esquema_avisos` en `db.py`: crear tabla nueva, copiar filas, drop, rename).
- AUTOINCREMENT no existe en Delta — se reemplaza con `GENERATED ALWAYS AS IDENTITY`, pero el patrón `cur.lastrowid` (usado en `00_orquestador.py:crear_corrida()`) no tiene equivalente 1:1: hay que hacer un SELECT de vuelta tras el INSERT.
- `poligonos_vulnerabilidad_uv` es, de las siete tablas, la que mejor encajaría en Delta sin fricción: se escribe una sola vez (o rara vez) y se lee siempre — el caso de uso típico de una tabla Delta de referencia, sin el problema de UPSERT fila-a-fila que sí tienen las demás (ver más abajo).

**UPSERT → MERGE INTO** — confirmado en 3 archivos dentro de `05_modelo_produccion/`:
1. `db.py:escribir_control()` — `ON CONFLICT(clave) DO UPDATE`
2. `02_scraper_detalle_incremental.py:guardar_detalle_produccion()` — `ON CONFLICT(id_aviso) DO UPDATE`, columnas dinámicas
3. `05_prediccion.py:guardar_prediccion()` — `ON CONFLICT(id_aviso, version_modelo) DO UPDATE`

Más un `INSERT OR IGNORE` en `01_scraper_grilla_incremental.py:guardar_pagina_en_produccion()`, que usa `cur.rowcount == 1` para contar inserciones reales — sin equivalente directo en MERGE (hay que contar por anti-join antes del MERGE).

**El problema real no es la sintaxis, es el patrón transaccional**: los UPSERT se ejecutan **fila por fila dentro de un loop Python** con commit inmediato tras cada fila — razonable contra SQLite local, pésimo contra un SQL Warehouse remoto (cada `execute()+commit()` sería un round-trip de red). Migrar bien esto exige reescribir la capa de persistencia para trabajar en **batch** (un solo `MERGE INTO` por etapa, no N ejecuciones individuales), no solo cambiar el conector.

**Puntos de conexión**: solo 2 funciones centrales (`db.conectar_produccion()`, `db.conectar_original()`), pero las llamadas SQL específicas de SQLite (`datetime('now')`, `PRAGMA foreign_keys`, `PRAGMA table_info`, `lastrowid`) están dispersas en al menos 5 archivos y no tienen equivalente directo en Databricks SQL.

## 3. El scraping: compatibilidad con clusters de Databricks

Confirmado: la ruta principal de ambos scrapers incrementales es 100% `requests` — correrían sin problema en un cluster compartido, sin necesidad de navegador.

Dos fricciones reales (no relacionadas con Playwright):

1. **`importlib.util.spec_from_file_location` con rutas relativas** — el obstáculo estructural más grande. Aparece en 6 archivos para cargar scripts hermanos cuyo nombre empieza con dígito (no son `import`ables directamente). Exige que el árbol completo del repo esté presente con la misma jerarquía relativa. En un notebook de Databricks esto se rompe (`__file__` no se resuelve igual); con Databricks Repos + script `.py` como job task *podría* funcionar pero es frágil. La solución limpia es convertir el proyecto en un **paquete Python instalable** (wheel) con imports normales — implica renombrar cada script que empieza con dígito, tocando cada referencia cruzada del repo.
2. **Acceso a filesystem local**: `RotatingFileHandler`, shapefile, CSV/JSON de referencia y `.pkl` de modelo se leen de rutas locales bajo el repo. En Databricks deberían vivir en un Unity Catalog Volume (o DBFS), no en disco local efímero del cluster.

## 4. El orquestador: `00_orquestador.py` vs. Databricks Jobs/Workflows

**Lo que se traduce bien**: 5 etapas secuenciales con corte-en-error mapea de forma natural a un Job con 5 Tasks encadenadas por `depends_on`.

**Lo que no es gratis**: el corte por CAPTCHA (no es excepción, es un corte limpio, las etapas siguientes deben seguir corriendo, corrida marcada `'parcial'`) no es una falla de task en el sentido de Databricks Jobs. Reproducirlo con tasks independientes reales exige propagar el flag vía `dbutils.jobs.taskValues` — trabajo real de rediseño, no una migración directa. Alternativa barata: mantener el orquestador como una sola task (cero reescritura, se pierde observabilidad granular).

**Qué se gana**: reintentos nativos por task, alertas nativas, historial de corridas navegable en la UI, sin mantener el `RotatingFileHandler` local.

**Actualización**: desde este diagnóstico, `00_orquestador.py` ya implementa a mano una versión ligera de lo que Jobs daría nativo — `main()` retorna el resultado de la corrida y el bloque `if __name__ == "__main__"` hace `sys.exit(1)` cuando es `"error"`, para que el runner de GitHub Actions marque el job (y por lo tanto la corrida programada) como fallido. El workflow persiste el diagnóstico (commit/push de `corridas`/`logs_ejecucion`) con `if: always()` incluso cuando el job termina en rojo. Esto confirma el punto de "qué se pierde/reimplementa": ya se reimplementó, con ~10 líneas de código propio — la ganancia real de migrar a Jobs seguiría siendo la UI/alertas nativas, no la señal de fallo en sí, que ya existe.

**Qué se pierde/reimplementa**: el `RotatingFileHandler` se vuelve redundante y se puede eliminar. Pero `corridas`/`logs_ejecucion` (metadata de negocio: avisos_nuevos_grilla, motivo_corte_grilla, etc.) no las reemplaza el historial genérico de Jobs — ese logging seguiría siendo código propio, escribiendo a Delta en vez de SQLite.

## 5. El entrenamiento del modelo: XGBoost/LightGBM en Databricks

Optuna, bagging manual multi-seed y TreeSHAP nativo son trabajo Python/NumPy puro sobre ~1600 filas — sin GPU, sin nada shaped-for-Spark. Correr esto en Databricks sería funcionalmente idéntico a correrlo en cualquier máquina normal — se pagaría cómputo de cluster para reubicar algo que ya corre en minutos localmente. Sin beneficio real de Spark en este volumen.

**Versionado casero vs. MLflow Model Registry**: el sistema actual (`versiones/{version}/` + `version_modelo.json`) ya cubre lo esencial (recuperar el modelo exacto usado en cualquier predicción pasada). MLflow aportaría tracking automático, UI de comparación de runs, y transiciones de stage — pero el ensamble de N modelos (bagging) necesitaría un `pyfunc` custom (los flavors nativos de XGBoost/LightGBM asumen un modelo por run). Mejora de conveniencia/gobernanza, no una necesidad — solo se justifica sola si de todas formas se adopta Databricks por otras razones.

## 6. Esfuerzo estimado y recomendación final

| Componente | Esfuerzo | Por qué |
|---|---|---|
| Scraping | Bajo-Medio | La ruta requests corre tal cual; el costo real es el refactor a paquete instalable + mover logs/artefactos a Volumes |
| Base de datos | Alto | Reescribir el patrón fila-a-fila a operaciones batch con MERGE INTO, reemplazar construcciones SQLite-only, tocar ≥4 archivos |
| Orquestador | Medio | Buen encaje conceptual, pero el estado "parcial por CAPTCHA" exige propagar señales entre tasks |
| Entrenamiento | Bajo esfuerzo, valor bajo | CPU-bound puro sin nada Spark-shaped; migrarlo no compra capacidad |

**Orden incremental si se migrara**: (1) reestructurar el repo como paquete instalable, (2) orquestador como Jobs, (3) capa de datos a Delta con reescritura batch, (4) entrenamiento al final (lo que menos se beneficia).

**Recomendación honesta**: con ~1600-2000 avisos, un solo usuario, y sin necesidad de cómputo distribuido, **no vale la pena migrar todo esto a Databricks por necesidad del proyecto**. El costo (reescritura de persistencia, restructuración a paquete, gestión de clusters/jobs, facturación) es desproporcionado frente al beneficio en este volumen. La ejecución ya se movió a un runner remoto (GitHub Actions, sección 1) desde este diagnóstico, lo que resuelve el incidente original de Google Drive para la corrida programada — pero introdujo su propia fricción de concurrencia (conflictos de git al comitear la BD, mitigados con `rebase`). Ambos son síntomas del mismo patrón (SQLite versionada en git como mecanismo de persistencia), no una razón para un Lakehouse completo; una base gestionada remota (o Delta, si de todas formas se adopta Databricks) los resolvería de raíz, con mucho menos costo que migrar todo el pipeline.

**Componente que sí valdría la pena mover independientemente**: MLflow (open-source, sin necesitar Databricks completo) para el versionado de modelos, si el catálogo de versiones crece lo suficiente. No es urgente hoy.

> **Nota**: el usuario de este proyecto confirmó que el motivo real para explorar Databricks es **aprender la herramienta**, no una necesidad del proyecto — de ahí la Parte 2 de este documento.

---

# Parte 2 — Guía de POC para aprender Databricks

Alcance acotado, decidido explícitamente para maximizar aprendizaje sin comprometerse a migrar el proyecto real:

- **Entorno**: Databricks Community Edition.
- **Foco**: Delta Lake + `MERGE INTO`, y Jobs/Workflows.
- **Fuera de alcance por ahora**: MLflow, PySpark distribuido, scraping real, entrenamiento real de XGBoost/LightGBM, Unity Catalog (no disponible en CE).
- **Datos**: sintéticos, creados directo en el notebook (`spark.createDataFrame`) — no se exporta la base real, todo el POC vive dentro de Databricks.

Cada pieza del POC se mapea 1:1 a un patrón real ya identificado en la Parte 1:

| Patrón real en el repo | Dónde aparece | Pieza del POC que lo practica |
|---|---|---|
| `INSERT OR IGNORE` + dedup contando filas nuevas | `01_scraper_grilla_incremental.py:guardar_pagina_en_produccion` | Notebook 1: `MERGE INTO ... WHEN NOT MATCHED THEN INSERT` (sin UPDATE), contando insertados vía anti-join previo |
| `INSERT ... ON CONFLICT DO UPDATE` | `db.py:escribir_control`, `02_scraper_detalle_incremental.py:guardar_detalle_produccion`, `05_prediccion.py:guardar_prediccion` | Notebook 2: `MERGE INTO ... WHEN MATCHED THEN UPDATE ... WHEN NOT MATCHED THEN INSERT` |
| Tabla `corridas` (metadata de cada corrida) | `00_orquestador.py` | Notebook 3: escribe una fila de resumen a una tabla Delta `corridas_delta` |
| Orquestador secuencial con corte por error, y corte "parcial" no-fatal (CAPTCHA) | `00_orquestador.py` | Job/Workflow con 3 tasks encadenadas por `depends_on`, más un flag pasado entre tasks vía `dbutils.jobs.taskValues` |
| CHECK constraints reconstruyendo la tabla completa (SQLite no permite `ALTER ... MODIFY CHECK`) | `db.py:_migrar_esquema_avisos` | Equivalente Delta: `ALTER TABLE ... ADD/DROP CONSTRAINT ... CHECK (...)`, sin reconstrucción |

## 2.1. Preparación del workspace

1. Creá un cluster (Compute → Create compute), un solo nodo, el Runtime más reciente disponible (trae Delta Lake integrado).
2. Confirmá si tenés **Workflows** en el menú lateral (la "Free Edition" nueva lo incluye; la Community Edition clásica históricamente no). Si no aparece, usá el fallback con `dbutils.notebook.run` de la sección 2.6.
3. Creá 3 notebooks Python adjuntos al mismo cluster: `01_ingesta_nuevos`, `02_enriquecer_detalle`, `03_resumen_corrida`.

## 2.2. Esquema Delta

Correlo una sola vez, en cualquiera de los 3 notebooks:

```sql
CREATE TABLE IF NOT EXISTS avisos_delta (
    id_aviso            STRING NOT NULL,
    comuna              STRING NOT NULL,
    precio              DOUBLE,
    estado_publicacion  STRING NOT NULL DEFAULT 'activo',
    first_seen          STRING
) USING DELTA;

ALTER TABLE avisos_delta ADD CONSTRAINT pk_avisos PRIMARY KEY (id_aviso);

ALTER TABLE avisos_delta ADD CONSTRAINT chk_estado_publicacion
    CHECK (estado_publicacion IN ('activo', 'pausado', 'finalizado', 'no_disponible'));

CREATE TABLE IF NOT EXISTS avisos_detalle_delta (
    id_aviso     STRING NOT NULL,
    dormitorios  INT,
    banos        INT,
    uv_rsh       STRING,     -- la resuelve otra etapa, nunca esta
    rank_nac     DOUBLE      -- idem
) USING DELTA;

ALTER TABLE avisos_detalle_delta ADD CONSTRAINT fk_detalle_avisos
    FOREIGN KEY (id_aviso) REFERENCES avisos_delta (id_aviso);

CREATE TABLE IF NOT EXISTS corridas_delta (
    id_corrida       BIGINT GENERATED ALWAYS AS IDENTITY,
    fecha            TIMESTAMP,
    avisos_nuevos    INT,
    avisos_actualizados INT
) USING DELTA;
```

Comparación con SQLite — cambiar el CHECK sin reconstruir la tabla:

```sql
ALTER TABLE avisos_delta DROP CONSTRAINT chk_estado_publicacion;
ALTER TABLE avisos_delta ADD CONSTRAINT chk_estado_publicacion
    CHECK (estado_publicacion IN ('activo', 'pausado', 'finalizado', 'no_disponible', 'nuevo_estado'));
```

## 2.3. Notebook `01_ingesta_nuevos` — dedup vía MERGE

```python
from pyspark.sql import Row

nuevos_avisos = spark.createDataFrame([
    Row(id_aviso="MLC-001", comuna="concepcion-biobio", precio=350000.0, estado_publicacion="activo", first_seen="2026-07-12"),
    Row(id_aviso="MLC-002", comuna="talcahuano-biobio", precio=280000.0, estado_publicacion="activo", first_seen="2026-07-12"),
    Row(id_aviso="MLC-003", comuna="concepcion-biobio", precio=410000.0, estado_publicacion="activo", first_seen="2026-07-12"),
])
nuevos_avisos.createOrReplaceTempView("staging_avisos")

# Anti-join ANTES del merge: MERGE no da un rowcount por fila como sqlite3.
insertados = spark.sql("""
    SELECT s.* FROM staging_avisos s
    LEFT ANTI JOIN avisos_delta a ON s.id_aviso = a.id_aviso
""")
print(f"Avisos nuevos a insertar: {insertados.count()}")

spark.sql("""
    MERGE INTO avisos_delta a
    USING staging_avisos s
    ON a.id_aviso = s.id_aviso
    WHEN NOT MATCHED THEN INSERT *
""")

display(spark.sql("SELECT * FROM avisos_delta"))
```

**Verificación**: primera corrida → 3 filas, "3 nuevos". Segunda corrida con el mismo staging → 0 nuevos, sigue en 3 (no duplica).

## 2.4. Notebook `02_enriquecer_detalle` — UPSERT vía MERGE

```python
from pyspark.sql import Row

detalle_nuevo = spark.createDataFrame([
    Row(id_aviso="MLC-001", dormitorios=2, banos=1),
    Row(id_aviso="MLC-004", dormitorios=3, banos=2),   # no existe todavía en avisos_delta, a propósito
])
detalle_nuevo.createOrReplaceTempView("staging_detalle")

spark.sql("""
    MERGE INTO avisos_detalle_delta d
    USING staging_detalle s
    ON d.id_aviso = s.id_aviso
    WHEN MATCHED THEN UPDATE SET
        d.dormitorios = s.dormitorios,
        d.banos = s.banos
        -- uv_rsh/rank_nac NO están en este SET: una revisita no debe
        -- borrar lo que resolvió la etapa de vulnerabilidad, igual que
        -- guardar_detalle_produccion en el código real.
    WHEN NOT MATCHED THEN INSERT (id_aviso, dormitorios, banos)
        VALUES (s.id_aviso, s.dormitorios, s.banos)
""")

display(spark.sql("SELECT * FROM avisos_detalle_delta"))
```

**Para ver la protección de columnas**: antes de correr el MERGE, seteá `UPDATE avisos_detalle_delta SET uv_rsh = 'UV-99' WHERE id_aviso = 'MLC-001'` a mano. Después de correr el notebook, `uv_rsh` de MLC-001 debe seguir en `'UV-99'`.

## 2.5. Notebook `03_resumen_corrida`

```python
nuevos = spark.sql("SELECT COUNT(*) c FROM avisos_delta").collect()[0]["c"]
actualizados = spark.sql("SELECT COUNT(*) c FROM avisos_detalle_delta").collect()[0]["c"]

spark.sql(f"""
    INSERT INTO corridas_delta (fecha, avisos_nuevos, avisos_actualizados)
    VALUES (current_timestamp(), {nuevos}, {actualizados})
""")

display(spark.sql("SELECT * FROM corridas_delta ORDER BY id_corrida DESC"))
```

## 2.6. Encadenar como Job/Workflow

**Con Workflows disponible**:
1. Workflows → Create Job.
2. Task 1 `id_corrida_ingesta` → notebook `01_ingesta_nuevos`.
3. Task 2 `enriquecer` → notebook `02_enriquecer_detalle`, Depends on → Task 1.
4. Task 3 `resumen` → notebook `03_resumen_corrida`, Depends on → Task 2.
5. Reintentos en Task 2 (Settings → Retries → 1-2 intentos).
6. Run now → mirar la vista de Gantt (duración/estado por task, logs con un click).

**Simular el flag "detenido_por_captcha"** (corte parcial no-fatal, la parte no-trivial de portar el orquestador real):

En `01_ingesta_nuevos`, al final:
```python
dbutils.jobs.taskValues.set(key="hubo_bloqueo", value=False)  # probar con True también
```

En `02_enriquecer_detalle`, al principio:
```python
hubo_bloqueo = dbutils.jobs.taskValues.get(taskKey="id_corrida_ingesta", key="hubo_bloqueo", default=False)
if hubo_bloqueo:
    print("Corte limpio simulado (ej. CAPTCHA) — igual sigo con lo ya procesado, como el orquestador real.")
```

**Sin Workflows** (fallback en un solo notebook):
```python
resultado_1 = dbutils.notebook.run("01_ingesta_nuevos", timeout_seconds=300)
resultado_2 = dbutils.notebook.run("02_enriquecer_detalle", timeout_seconds=300)
resultado_3 = dbutils.notebook.run("03_resumen_corrida", timeout_seconds=300)
```

## 2.7. Checklist de verificación

| Paso | Resultado esperado |
|---|---|
| Notebook 1, primera corrida | `avisos_delta` con 3 filas, "3 nuevos" |
| Notebook 1, segunda corrida (mismo staging) | 0 nuevos, sigue en 3 (no duplica) |
| Notebook 2 | MLC-001 actualizado, MLC-004 insertado |
| Notebook 2, con `uv_rsh` seteado a mano antes | `uv_rsh='UV-99'` intacto después del MERGE |
| Notebook 3 | fila nueva en `corridas_delta` por corrida, `id_corrida` incrementando |
| Job con 3 tasks | Gantt con las 3 tasks en orden, logs propios por task |
| `taskValues` con `hubo_bloqueo=True` | notebook 2 imprime el aviso, Job completa sin marcar error |

## 2.8. Límites de esta guía

Nada de esto se ejecutó ni se verificó desde este entorno (sin CLI/credenciales de Databricks configuradas acá). El código sigue la sintaxis estándar de Delta/Spark SQL, pero la única verificación real es la que hagas corriéndolo en tu workspace. Si algo falla por una diferencia de versión de Runtime (`GENERATED ALWAYS AS IDENTITY` o `ADD CONSTRAINT` pueden variar entre versiones viejas de Delta), el mensaje de error exacto ayuda a ajustarlo.
