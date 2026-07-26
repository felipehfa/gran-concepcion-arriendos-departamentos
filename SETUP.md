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
| Visualización (`produccion/03_visualizacion/`) | `streamlit`, `folium`, `streamlit-folium`, `pandas`, `numpy`, `requests`, `joblib`, `scikit-learn` — las últimas cuatro porque `data.py` importa dinámicamente `investigacion/03_ingenieria_variables/01_ingenieria_variables.py` (ver sección 3) |

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

### 2.1. Camino rápido — usar los datos ya incluidos

Si solo quieres reproducir la ingeniería de variables y el modelamiento (sin volver a
scrapear), corre en orden desde la raíz del repo:

```bash
python investigacion/03_ingenieria_variables/01_ingenieria_variables.py
python investigacion/03_ingenieria_variables/02_seleccion_variables.py
python investigacion/04_modelamiento/01_xgboost.py   # entrenamiento + etiquetado oportunidad/caro
python investigacion/04_modelamiento/02_lightgbm.py  # entrenamiento + etiquetado oportunidad/caro
```

### 2.2. Camino completo — scraping desde cero

```bash
# 1. Grilla de búsqueda (requests + BeautifulSoup, sin navegador)
python investigacion/01_obtener_datos/01_scraper_grilla.py

# 2. Detalle de cada aviso (requests, sin navegador - más sensible a bloqueo que
#    la grilla por el volumen de visitas, aunque no se ha observado bloqueo en la práctica).
#    Pensado para correr en tandas vía cron, no de una sola sentada
#    (ver LIMITE_POR_CORRIDA y COOLDOWN_TRAS_CAPTCHA_MINUTOS en el script).
python investigacion/01_obtener_datos/02_scraper_detalle.py

# 2b. Solo si la ruta principal empezara a bloquearse de forma persistente (no
#     observado hasta ahora) o necesitas resolver un CAPTCHA a mano: ruta de
#     respaldo con Playwright (requiere pip install playwright playwright-stealth
#     && playwright install chromium - ver sección 1).
# python investigacion/01_obtener_datos/02_scraper_detalle.py --fallback-playwright

# 3. Cruce geoespacial con el índice de vulnerabilidad socioterritorial (IGVUST).
#    Requiere el shapefile 202505_IGVUST_UV_cuartil.(shp/dbf/shx/prj) en
#    investigacion/01_obtener_datos/datos_vulnerabilidad/ — NO está incluido en el repo
#    (ver nota más abajo).
python investigacion/01_obtener_datos/03_vulnerabilidad_socioterritorial.py

# 4-6. Igual que el camino rápido (2.1)
```

> **Nota sobre el shapefile de vulnerabilidad**: la carpeta
> `investigacion/01_obtener_datos/datos_vulnerabilidad/` está excluida del repo vía `.gitignore`
> (dato pesado de origen externo). La base de datos ya incluye las tablas `vulnerabilidad_uv` y
> `avisos_igvust` resueltas de una corrida previa, así que solo necesitas el shapefile si quieres
> **regenerar ese cruce desde cero** (por ejemplo, tras scrapear avisos nuevos).

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
`investigacion/03_ingenieria_variables/01_ingenieria_variables.py`, que `data.py` importa
dinámicamente — sin ellas el deploy falla con `ModuleNotFoundError` al cargar ese módulo
(Streamlit Cloud solo instala lo declarado en este `requirements.txt`, no las dependencias de las
demás etapas del pipeline).

Requiere `BD_STRING` configurado (sección 3) y que la base ya tenga al menos un aviso en
`predicciones` (ver [README, sección 9](README.md#9-pipeline-de-producción-produccion01_modelo_produccion))
— si la base está vacía o el orquestador todavía no corrió la etapa de predicción, el dashboard
muestra un aviso de "sin datos" en vez de una tabla vacía silenciosa.
