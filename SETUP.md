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

## 3. Dashboard (`produccion/03_visualizacion/`)

```bash
pip install -r produccion/03_visualizacion/requirements.txt
cd produccion/03_visualizacion
streamlit run app.py
```

`requirements.txt` incluye, además de `streamlit`/`folium`/`streamlit-folium`/`pandas`,
`numpy`/`requests`/`joblib`/`scikit-learn`: son transitivas de
`investigacion/03_ingenieria_variables/01_ingenieria_variables.py`, que `data.py` importa
dinámicamente — sin ellas el deploy falla con `ModuleNotFoundError` al cargar ese módulo
(Streamlit Cloud solo instala lo declarado en este `requirements.txt`, no las dependencias de las
demás etapas del pipeline).

Requiere que `produccion/01_modelo_produccion/produccion_gran_concepcion.db` ya exista con al
menos un aviso en `predicciones` (ver [README, sección 9](README.md#9-pipeline-de-producción-produccion01_modelo_produccion))
— si la base está vacía o el orquestador todavía no corrió la etapa de predicción, el dashboard
muestra un aviso de "sin datos" en vez de una tabla vacía silenciosa.
