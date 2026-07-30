# Setup

Instalación, dependencias y cómo correr cada etapa del proyecto
[gran-concepcion-rentals](README.md). Para el resumen del proyecto, arquitectura y hallazgos
técnicos, ver el [README](README.md).

---

## 1. Dependencias

El repo no incluye un `requirements.txt` en la raíz (sí uno propio en `produccion/03_visualizacion/`,
ver sección 3); estas son las dependencias reales de cada etapa, inferidas de los `import` de cada
una (probado con Python 3.11):

| Etapa                          | Librerías                                                        |
|----------------------------------|-------------------------------------------------------------------|
| Scraping (grilla)                | `requests`, `beautifulsoup4`, `lxml`, `pandas`                    |
| Scraping (detalle)                | `requests`, `beautifulsoup4`, `lxml`, `pandas` — `playwright`/`playwright-stealth` opcionales, solo para la ruta de respaldo (`--fallback-playwright`) |
| Vulnerabilidad socioterritorial   | `geopandas`, `shapely`, `pandas`                                   |
| Ingeniería de variables           | `pandas`, `numpy`, `requests`, `joblib`, `scikit-learn`            |
| Selección de variables            | `pandas`, `numpy`, `xgboost`, `optuna`, `scikit-learn`             |
| Modelamiento                      | `pandas`, `numpy`, `xgboost`, `lightgbm`, `optuna`, `scikit-learn`, `scipy` |
| Visualización (`produccion/03_visualizacion/`) | `streamlit`, `folium`, `streamlit-folium`, `pandas`, `numpy`, `requests`, `joblib`, `scikit-learn` — las últimas cuatro porque `data.py` importa dinámicamente `modelamiento/01_ingenieria_variables/01_ingenieria_variables.py` (ver sección 3) |

Instalación sugerida (sin versiones pineadas, ya que no existen en el repo):

```bash
pip install requests beautifulsoup4 lxml pandas \
            geopandas shapely scikit-learn joblib xgboost lightgbm optuna scipy

# Opcional, solo para la ruta de respaldo de 02_scraper_detalle.py:
pip install playwright playwright-stealth
playwright install chromium
```

Playwright **no es necesario** para el camino normal (ambos scrapers usan `requests`, sin
navegador). Solo instálalo si vas a usar la ruta de respaldo de `02_scraper_detalle.py` (ver
[README, sección 3.2](README.md#32-scraping-arquitectura-y-decisiones)).

---

## 2. Cómo correrlo

> **Desde el 2026-07-29 no hay dos stacks de datos.** Antes existía una base SQLite propia de
> investigación (`avisos_gran_concepcion.db`) que se llenaba corriendo los scrapers a mano. Se
> retiró: **todo lee de Supabase**, así que estos pasos necesitan `BD_STRING` configurado
> (sección 3). La adquisición de datos la hace el orquestador de producción, no un paso manual.

### 2.1. Reproducir la ingeniería de variables y el modelamiento

Desde la raíz del repo, en orden:

```bash
# 1. Construye el dataset leyendo de Supabase y lo guarda en
#    modelamiento/01_ingenieria_variables/save/ingeniaria_variables/
python modelamiento/01_ingenieria_variables/01_ingenieria_variables.py

# 2. Selección de variables sobre ese dataset -> selected_features.csv
python modelamiento/01_ingenieria_variables/02_seleccion_variables.py

# 3. Entrenamiento + etiquetado oportunidad/caro (elegí uno)
python modelamiento/02_modelos/01_xgboost.py
python modelamiento/02_modelos/02_lightgbm.py
```

> **Cuidado con los artefactos compartidos**: el paso 1 sobrescribe `niveles_barrio.json` y
> `modelos_superficie/*.pkl`, y el paso 2 sobrescribe `selected_features.csv` — los tres los lee
> **el pipeline de producción en cada inferencia**. El modelo desplegado espera un set de
> features exacto (`entrenamiento/versiones/{version}/parametros_produccion.json`); si el dataset
> cambió de tamaño y corrés el paso 2 con los defaults, la selección puede dar un set distinto y
> la próxima inferencia falla hasta reentrenar (02_seleccion_variables.py avisa esto si detecta un
> modelo desplegado). Si estás explorando (probando filtros de población, por ejemplo) y no querés
> tocarle los artefactos al modelo desplegado: para el paso 1, importá `ejecutar_pipeline` y
> redirigí las salidas con `ruta_salida_csv`, `ruta_salida_niveles_barrio` y
> `ruta_modelos_superficie`; para el paso 2, pasale a `run_feature_selection` un `output_dir`
> propio.

### 2.2. Reentrenar el modelo de producción

Es el camino recomendado: regenera el dataset desde Supabase y entrena, versionando el modelo y
archivando los artefactos con los que se entrenó.

```bash
python produccion/01_modelo_produccion/entrenamiento/01_entrenar_modelo_produccion.py \
    --origen-datos supabase --estados activo pausado finalizado --meses-max-finalizados 3
```

Los filtros de población son parametrizables para poder comparar configuraciones. **No entrenes solo
con `activo`**: son por definición los avisos que el mercado todavía no absorbió, así que
sobre-representan unidades caras para su segmento, y además dan menos filas (1414) que incluir los
finalizados recientes (~2430). El límite de antigüedad se evalúa sobre `fecha_publicacion_aprox`.

### 2.3. Scraping

Lo corre el orquestador de producción cada 4h (`.github/workflows/orquestador.yml`), y también se
puede disparar a mano con `workflow_dispatch`. Localmente:

```bash
python produccion/01_modelo_produccion/00_orquestador.py
```

Los scrapers base viven en `produccion/01_modelo_produccion/scrapers_base/` (el pipeline los importa
vía `importlib` para no duplicar el parsing HTML/JSON). Sus `main()` standalone son legacy:
escribían a la base SQLite que ya no existe.

El cruce geoespacial de vulnerabilidad lo hace la etapa `03_vulnerabilidad_produccion.py` del
orquestador contra la tabla `poligonos_vulnerabilidad_uv` de Supabase, que ya está poblada — solo
necesitás el shapefile si querés regenerar ese cruce desde cero.

> **Nota sobre el shapefile de vulnerabilidad**: la carpeta
> `produccion/01_modelo_produccion/scrapers_base/datos_vulnerabilidad/` está excluida del repo vía
> `.gitignore` (dato pesado de origen externo).

> **Nota sobre el scraping**: revisa el `robots.txt` / Términos de Uso del sitio antes de correr
> los scrapers a gran escala, y no redistribuyas contenido con derechos de terceros (fotos,
> descripciones) sin permiso.

---

## 3. Base de datos de producción (Supabase/Postgres)

Desde el 2026-07-26 la base de producción vive en Supabase (Postgres), no en un archivo `.db`
versionado — ver [README, sección 9.4](README.md#94-base-de-datos-por-qué-supabase-y-no-un-db-versionado).
Tanto el pipeline de `produccion/01_modelo_produccion/` como el dashboard leen el connection
string desde la variable de entorno `BD_STRING`:

```
BD_STRING = postgresql://postgres.<project-ref>:<password>@aws-0-<región>.pooler.supabase.com:6543/postgres
```

Usá el connection string del **connection pooler, modo Transaction** (Project Settings → Database
en el dashboard de Supabase) — no el de conexión directa: los runners de GitHub Actions no tienen
salida IPv6 confiable, y la conexión directa de Supabase resuelve solo por IPv6.

**Dos roles distintos, principio de mínimo privilegio** (el orquestador escribe, el dashboard casi
no):

| Rol | Usado por | Privilegios |
|---|---|---|
| `postgres` | Orquestador (GitHub Actions) | Acceso completo (dueño del schema) |
| `streamlit_app` | Dashboard (Streamlit Cloud) | `SELECT` en todo el schema `public` + `INSERT`/`UPDATE` solo en `valores_uf` (la única escritura que hace el dashboard, vía la caché de UF) |

`streamlit_app` se creó a mano una vez contra Supabase (fuera de `db.py`, que solo gestiona
tablas, no roles):

```sql
CREATE ROLE streamlit_app LOGIN PASSWORD '<password-fuerte>';
GRANT CONNECT ON DATABASE postgres TO streamlit_app;
GRANT USAGE ON SCHEMA public TO streamlit_app;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO streamlit_app;
GRANT INSERT, UPDATE ON valores_uf TO streamlit_app;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO streamlit_app;
```

Configuración por entorno:

- **Local**: creá un archivo `.env` en la raíz del repo con `BD_STRING` (rol `postgres` — ya está
  en `.gitignore`, nunca se commitea). Para probar el dashboard localmente con el rol restringido,
  usá el connection string de `streamlit_app` en su lugar.
- **GitHub Actions** (orquestador): repository secret `BD_STRING` con el connection string del rol
  `postgres`, en Settings → Secrets and variables → Actions.
- **Streamlit Cloud** (dashboard): Settings → Secrets de la app, en formato TOML
  (`BD_STRING = "postgresql://streamlit_app.<project-ref>:..."`, con comillas) — el connection
  string del rol `streamlit_app`, **no** el de `postgres`.

El schema **no requiere ningún paso manual**: `db.py` corre `CREATE TABLE IF NOT EXISTS` para las
9 tablas en cada conexión (`conectar_produccion()`), así que apuntar `BD_STRING` a un proyecto
Supabase vacío y correr el pipeline una vez ya lo deja listo — no hay un `.sql` aparte que
mantener sincronizado a mano. Esto incluye `valores_uf`: hace falta que el orquestador (rol
`postgres`) la cree al menos una vez antes de que el dashboard (rol `streamlit_app`, sin privilegio
`CREATE` sobre el schema) la toque, porque su propio intento de `CREATE TABLE IF NOT EXISTS` en
`inicializar_tabla_uf()` fallaría con `InsufficientPrivilege` si la tabla no existiera todavía.

---

## 4. Dashboard (`produccion/03_visualizacion/`)

```bash
pip install -r produccion/03_visualizacion/requirements.txt
cd produccion/03_visualizacion
streamlit run app.py
```

`requirements.txt` incluye, además de `streamlit`/`folium`/`streamlit-folium`/`pandas`/`psycopg2-binary`,
`numpy`/`requests`/`joblib`/`scikit-learn`: son transitivas de
`modelamiento/01_ingenieria_variables/01_ingenieria_variables.py`, que `data.py` importa
dinámicamente — sin ellas el deploy falla con `ModuleNotFoundError` al cargar ese módulo
(Streamlit Cloud solo instala lo declarado en este `requirements.txt`, no las dependencias de las
demás etapas del pipeline).

Requiere `BD_STRING` configurado (sección 3) y que la base ya tenga al menos un aviso en
`predicciones` (ver [README, sección 9](README.md#9-pipeline-de-producción-produccion01_modelo_produccion))
— si la base está vacía o el orquestador todavía no corrió la etapa de predicción, el dashboard
muestra un aviso de "sin datos" en vez de una tabla vacía silenciosa.
