# Gran Concepción Rentals

Pipeline completo de datos, desde scraping hasta modelamiento, para estimar el precio de
arriendo de **departamentos** en las comunas del Gran Concepción (Chile) y detectar avisos
que están publicados por debajo o por encima de lo que el mercado local justifica.

El proyecto cubre cinco etapas encadenadas:

1. **Scraping** de avisos de arriendo (Portal Inmobiliario) + cruce con un índice público de
   vulnerabilidad socioterritorial.
2. **Análisis exploratorio** de los datos crudos.
3. **Ingeniería y selección de variables**: limpieza, imputación, variables derivadas y
   selección estadística de las features finales.
4. **Modelamiento**: comparación de XGBoost, LightGBM y Random Forest para predecir
   `precio_clp`, con **XGBoost como modelo final**, sobre el cual se construye además un
   sistema de etiquetado "oportunidad / caro / precio de mercado".
5. **Producción** (`05_modelo_produccion/`): pipeline separado, re-ejecutable vía cron, que
   entrena y versiona un modelo de producción, scrapea avisos nuevos de forma incremental,
   y genera predicciones + etiquetas sobre una base de datos propia. Ver sección 12.

> El pipeline de modelamiento trabaja exclusivamente sobre **departamentos**. Los scrapers sí
> recolectan casas, pero la etapa de ingeniería de variables filtra y trabaja solo con
> `tipo_propiedad = "departamento"`.

---

## 1. Arquitectura del pipeline

```
01_obtener_datos/
  01_scraper_grilla.py                  → tabla `avisos`               (requests + BeautifulSoup)
  02_scraper_detalle.py                 → tabla `avisos_detalle`       (Playwright + stealth)
  03_vulnerabilidad_socioterritorial.py → tablas `vulnerabilidad_uv`,
                                           `avisos_igvust`               (geopandas, cruce espacial)
        │  (todo persiste en avisos_gran_concepcion.db, SQLite)
        ▼
02_analisis_exploratorio/
  01_EDA.ipynb                          → exploración manual de los datos crudos
        ▼
03_ingenieria_variables/
  01_ingenieria_variables.py            → datos_ingenieria_variables.csv (1.476 filas × 42 features)
  02_seleccion_variables.py             → selected_features.csv          (20 features finales)
        ▼
04_modelamiento/
  01_xgboost.py        → modelo FINAL (bagging ×10) + etiquetado oportunidad/caro
  02_lightgbm.py        → modelo de comparación (bagging ×5)
  03_random_forest.py   → modelo de comparación (bagging ×5)
        ▼
05_modelo_produccion/   → pipeline de producción, separado e independiente (ver sección 12)
  entrenamiento/01_entrenar_modelo_produccion.py → modelo versionado (85/15 + calibración)
  00_orquestador.py                              → corre las etapas 1-5 de abajo en orden
  01_scraper_grilla_incremental.py               → tabla `avisos`          (produccion_gran_concepcion.db)
  02_scraper_detalle_incremental.py              → tabla `avisos_detalle` + estado_publicacion
  03_vulnerabilidad_produccion.py                → columnas de vulnerabilidad en `avisos_detalle`
  04_ingenieria_variables_produccion.py          → features de avisos nuevos (contra referencia histórica)
  05_prediccion.py                               → tabla `predicciones` (precio + etiqueta + confianza)
```

Cada script de cada etapa ancla sus rutas de entrada/salida a la ubicación del propio archivo
(no al directorio de trabajo actual), por lo que pueden ejecutarse desde la raíz del repo o
desde su propia carpeta indistintamente.

La base de datos SQLite (`01_obtener_datos/avisos_gran_concepcion.db`, ~4 MB) **está
versionada en el repo**, ya con los datos scrapeados y las tablas de vulnerabilidad
resueltas. Esto significa que para reproducir las etapas 3 y 4 (ingeniería de variables y
modelamiento) no es necesario correr los scrapers desde cero.

---

## 2. Cómo correrlo

### 2.1. Requisitos

Instala las dependencias de Python (ver sección [Dependencias](#10-dependencias)) y, si vas a
correr el scraper de detalle, los navegadores de Playwright:

```bash
pip install requests beautifulsoup4 lxml pandas playwright playwright-stealth \
            geopandas shapely scikit-learn joblib xgboost lightgbm optuna scipy

playwright install chromium
```

### 2.2. Camino rápido — usar los datos ya incluidos

Si solo quieres reproducir la ingeniería de variables y el modelamiento (sin volver a
scrapear), corre en orden desde la raíz del repo:

```bash
python 03_ingenieria_variables/01_ingenieria_variables.py
python 03_ingenieria_variables/02_seleccion_variables.py
python 04_modelamiento/01_xgboost.py        # modelo final + etiquetado
python 04_modelamiento/02_lightgbm.py       # comparación
python 04_modelamiento/03_random_forest.py  # comparación
```

### 2.3. Camino completo — scraping desde cero

```bash
# 1. Grilla de búsqueda (requests + BeautifulSoup, sin navegador)
python 01_obtener_datos/01_scraper_grilla.py

# 2. Detalle de cada aviso (Playwright, más lento y más sensible a bloqueo).
#    Pensado para correr en tandas vía cron, no de una sola sentada
#    (ver LIMITE_POR_CORRIDA y COOLDOWN_TRAS_CAPTCHA_MINUTOS en el script).
python 01_obtener_datos/02_scraper_detalle.py

# 3. Cruce geoespacial con el índice de vulnerabilidad socioterritorial (IGVUST).
#    Requiere el shapefile 202505_IGVUST_UV_cuartil.(shp/dbf/shx/prj) en
#    01_obtener_datos/datos_vulnerabilidad/ — NO está incluido en el repo
#    (ver nota más abajo).
python 01_obtener_datos/03_vulnerabilidad_socioterritorial.py

# 4-6. Igual que el camino rápido (2.2)
```

> **Nota sobre el shapefile de vulnerabilidad**: la carpeta `01_obtener_datos/datos_vulnerabilidad/`
> está excluida del repo vía `.gitignore` (dato pesado de origen externo). La base de datos ya
> incluye las tablas `vulnerabilidad_uv` y `avisos_igvust` resueltas de una corrida previa, así
> que solo necesitas el shapefile si quieres **regenerar ese cruce desde cero** (por ejemplo,
> tras scrapear avisos nuevos).

> **Nota sobre el scraping**: revisa el `robots.txt` / Términos de Uso del sitio antes de correr
> los scrapers a gran escala, y no redistribuyas contenido con derechos de terceros (fotos,
> descripciones) sin permiso.

---

## 3. Decisiones clave de calidad de datos

### 3.1. La lección más importante del proyecto: fuga de datos en `precio_m2`

El hallazgo más significativo del proyecto fue detectar **fuga de datos (data leakage)** en
una primera versión de la variable de precio/m² de sector: se calculaba usando el precio de
la **propia fila** del aviso, lo que producía un MAPE artificialmente bajo (~3.6%) — una señal
de alerta, no de buen desempeño, ya que el modelo estaba viendo (indirectamente) la respuesta
que debía predecir.

La corrección, implementada en `agregar_precio_m2_sector` (`03_ingenieria_variables/01_ingenieria_variables.py`),
calcula el precio/m² de cada aviso usando **solo comparables de OTRAS propiedades** dentro de
un radio de 300 metros (excluyendo la fila propia), con:
- filtro de outliers vía IQR (multiplicador ×3) sobre el precio/m² del sector antes de
  promediar,
- mediana (no promedio) de los vecinos válidos,
- fallback a la mediana general del grupo cuando no hay vecinos cercanos válidos, marcado
  explícitamente en la columna `tiene_comparables_cercanos` para distinguir ese caso de un
  vecindario real con valor bajo.

Esta variable corregida (`precio_m2_sector_departamento`) sí es una de las 20 features
finales del modelo — la diferencia crítica es que su fuente son los vecinos, nunca la propia
fila.

### 3.2. Scraping

- Arquitectura de **dos scrapers separados**: `01_scraper_grilla.py` recorre las páginas de
  resultados de búsqueda (requests + BeautifulSoup, sin navegador), y `02_scraper_detalle.py`
  visita cada aviso individual (Playwright) para extraer descripción, características y
  puntos de interés cercanos.
- **Guardado incremental y diseño reanudable**: cada página/aviso se persiste con commit
  inmediato en SQLite; el scraper de grilla usa `INSERT OR IGNORE` sobre `id_aviso`, y el de
  detalle retoma solo los avisos pendientes vía un `LEFT JOIN` entre `avisos` y
  `avisos_detalle`.
- **Corte temprano por duplicados**: si una página completa de resultados ya existía en la
  base, se corta esa búsqueda y se pasa a la siguiente combinación comuna/tipo.
- **POIs vía JSON embebido**: los puntos de interés (colegios, paraderos, plazas, etc.) se
  extraen de un bloque JSON de configuración embebido en la página (`window._n.ctx.r`), no del
  HTML visible, porque ese JSON trae todas las categorías completas aunque el usuario no haya
  abierto esa pestaña. Se consideran "cercanos" solo los que están dentro de un **radio de
  500 metros**.
- **Mitigaciones de bloqueo**: `playwright-stealth`, delays variables entre requests, y
  detección de CAPTCHA por **doble condición** (la palabra "captcha" aparece en el HTML **Y**
  el contenido normal del aviso no cargó) para no confundir el script de reCAPTCHA de fondo
  (presente casi siempre) con un bloqueo real.
- **Bug corregido: `avisos_detalle.banos` capturaba la superficie, no los baños.** El regex
  original (`RE_BANOS`, con `re.IGNORECASE`) encontraba primero la insignia superior de la
  página ("2 baños\n75 m² totales", número **antes** de la palabra, en minúscula) en vez de la
  sección de características ("Baños\n2", número **después**, con mayúscula), capturando la
  superficie en vez de los baños reales. Afectaba ~48% de las filas (851 de 1782). Corregido
  quitando `re.IGNORECASE` (ahora solo matchea la "Baños" con mayúscula de la sección de
  características), y se corrigieron retroactivamente los valores ya guardados en la base
  usando `avisos.banos` (grilla, con un regex distinto que nunca tuvo este problema) como
  fuente confiable. El feature `banos` que usa el modelo siempre vino de `avisos.banos`
  (grilla), así que ningún modelo entrenado quedó afectado por este bug.

### 3.3. Limpieza y conversión de datos

- **Conversión UF → CLP** vía la API de `mindicador.cl`, con caché de valores de UF en una
  tabla SQLite (`valores_uf`) para no repetir consultas entre corridas.
- Corrección de formato numérico chileno (separador de miles) al parsear distancias y precios.
- Filtros de valores imposibles en dormitorios, baños y estacionamientos (probables errores de
  digitación).
- **Imputación de superficie corrupta** (bajo un umbral mínimo o faltante) con un
  `RandomForestRegressor` entrenado por tipo de propiedad, persistido en `.pkl` para
  reutilizarse sin reentrenar.
- **Imputación de antigüedad** por cercanía geográfica (mediana de vecinos dentro de 200 m del
  mismo tipo de propiedad), con una cascada de fallbacks (mediana por tipo → por comuna →
  global) calculados siempre sobre los valores originales, nunca sobre estimaciones previas.
- **Filtro de outliers de precio** vía un tope máximo de `precio_clp` (8.000.000 CLP): por
  encima de ese nivel se asume que son ventas mal clasificadas como arriendo, no arriendos
  reales.

### 3.4. Ingeniería de variables

- Distancias vía **fórmula de Haversine** (al centro de la propia comuna y al centro de
  Concepción).
- **`nivel_barrio`**: precio/m² promedio por barrio, suavizado hacia la media general (k=20
  "avisos virtuales") y agrupado en 5 niveles usando **cuantiles ponderados por cantidad de
  avisos** (no por cantidad de barrios), para que "alto" represente realmente ~20% de las
  propiedades más caras.
- **`precio_m2_sector_departamento`**: ver sección 3.1 — comparables reales cercanos (300 m),
  excluyendo la propia fila.
- **`ratio_total_util`**: superficie total / superficie útil, como proxy de cuánta superficie
  común/no habitable tiene la propiedad.

---

## 4. Features del modelo final (20)

Seleccionadas por `03_ingenieria_variables/02_seleccion_variables.py` a partir de 42 features
candidatas (ver sección 6 para la metodología de selección).

**Características físicas de la propiedad**
- `superficie_util_m2`, `superficie_total_m2`, `ratio_total_util`
- `banos`, `estacionamientos`, `piso_unidad`, `ascensor`, `piscina`, `amoblado`
- `antiguedad_anos`

**Costos asociados**
- `gastos_comunes`

**Ubicación y mercado local**
- `precio_m2_sector_departamento` (comparables cercanos, ver sección 3.1)
- `tiene_comparables_cercanos` (flag de confiabilidad del anterior)
- `distancia_centro_concepcion_m`, `distancia_centro_comuna_m`
- `cantidad_paraderos`, `cantidad_colegios`

**Contexto socioeconómico del sector (índice IGVUST / Registro Social de Hogares, por
Unidad Vecinal)**
- `rank_nac` (ranking nacional de vulnerabilidad de la Unidad Vecinal)
- `pob_rsh_uv` (población registrada en el RSH de la Unidad Vecinal)
- `p_urbano` (porcentaje urbano de la Unidad Vecinal)

`nivel_barrio` no quedó entre las 20 finales (fue descartada en la selección por estabilidad),
pese a ser una variable derivada relevante en la etapa de ingeniería.

---

## 5. Evolución del modelamiento

El proyecto pasó por dos grandes etapas de modelamiento:

**Etapa inicial** — Random Forest como modelo base, evaluado con MAE/RMSE/R²/MAPE y la razón
RMSE/MAE como diagnóstico de concentración de error en casos extremos (mejoró de ~2.7 a ~2.1
tras la limpieza de datos). En esta etapa también se probó `log(precio_clp)` como target y se
descartó por no aportar mejora, y se revisaron manualmente (URL por URL) los casos de error
más extremo para descartar errores de datos frente a variabilidad genuina de mercado.

**Etapa de refinamiento** (la que definió la versión final, implementada en los scripts
actuales de `04_modelamiento/`):

- **Split estratificado por quintil de precio** (`construir_estratos_precio` + `stratify=` en
  `train_test_split`), en reemplazo de un split aleatorio simple que mostraba inestabilidad
  severa entre corridas.
- **Selección de variables por estabilidad**: SHAP + 30 modelos + K-Fold + regla de 1 error
  estándar (ver sección 6), en vez de depender de la importancia de un único modelo.
- **Objective/criterion del modelo fijado manualmente** (no como hiperparámetro de Optuna) en
  los tres algoritmos, tras detectar que dejarlo como parte del espacio de búsqueda generaba
  inestabilidad entre corridas (el "ganador" cambiaba por ruido de los folds, no por
  diferencia real de desempeño).
- **Validación multi-semilla** (`run_stability_across_seeds`, 8 particiones con split
  estratificado, Optuna completo en cada una) para confirmar que las métricas del modelo
  final no dependen de una partición con suerte. No se ejecuta automáticamente por su costo
  computacional; está disponible para correrse aparte.
- Se **volvió a probar `target_transform='log'`** en esta etapa, con el pipeline ya mejorado,
  y se descartó otra vez por empeorar todas las métricas, incluida la kurtosis de los
  residuos — confirmando con metodología más rigurosa la misma conclusión de la etapa inicial.
- **Comparación de tres arquitecturas** (XGBoost, LightGBM, Random Forest) sobre las mismas 20
  features, mismo split y misma seed (ver sección 7).
- Se **descartó separar el modelo en "premium vs. resto"**: el MAPE del quintil más caro (Q5)
  no es dramáticamente peor que el resto — los datos de test lo confirman (Q5 MAPE=11.77% vs.
  Q1 MAPE=11.96%, ver sección 7) — por lo que el problema real es volumen de datos en ese
  segmento, no una estructura de precios distinta.

> **Nota sobre variables derivadas adicionales**: `03_ingenieria_variables/01_ingenieria_variables.py`
> incluye una función `crear_variables_derivadas` (ratios por dormitorio, índice de amenities,
> interacciones ubicación-superficie, `valor_mercado_estimado_sector`, etc.), pero está
> **deshabilitada** en el pipeline actual (`ejecutar_pipeline`, la llamada está comentada). Es
> decir: estas variables se probaron pero, a la fecha, **no está confirmado en el código que
> aporten mejora real** frente al ruido de partición — quedan documentadas como una línea de
> trabajo pendiente, no como parte del modelo final.

Del target transform log en la etapa de refinamiento tampoco queda rastro activo en el
código actual (los scripts de `04_modelamiento/` entrenan directamente sobre `precio_clp`),
consistente con que la conclusión fue descartarlo.

---

## 6. Selección de variables (metodología)

`03_ingenieria_variables/02_seleccion_variables.py` reduce 42 features candidatas a las 20
finales en 4 pasos:

1. **Eliminación de constantes**: features con varianza ≈ 0 sobre train. En la corrida
   registrada, se eliminó `quincho` (varianza cero: prácticamente ningún departamento
   reportaba esa amenity).
2. **Selección por estabilidad**: se entrenan 30 modelos XGBoost con K-Fold aleatorio (5
   folds, semillas distintas), midiendo importancia SHAP (TreeSHAP nativo) de cada feature en
   cada fold. `stability_score = (1 / (1 + CV)) × presence_pct`, donde CV es el coeficiente de
   variación de la importancia entre modelos y `presence_pct` el % de modelos donde la
   importancia fue > 0.
3. **Selección de k óptimo vía MAE de validación**: se evalúa la curva MAE/RMSE/R² en el set
   de validación para k features (ordenadas por `stability_score`), promediando sobre 5
   semillas, y se aplica la **regla de 1 error estándar** (el k más chico cuyo MAE promedio
   cae dentro de 1 SE del mínimo) — resultado: **k=20**.
4. **Red de seguridad de correlación**: elimina pares de features con correlación > 0.95 entre
   las ya seleccionadas, quedándose con la de mayor `stability_score`. En la corrida
   registrada no eliminó ninguna (0 de 20).

---

## 7. Modelos comparados y modelo final

Los tres modelos comparten exactamente las mismas 20 features, el mismo split estratificado
(seed=42; 1.032 train / 222 val / 222 test) y el mismo esquema de optimización (Optuna, 50
trials, KFold=5, CV solo sobre train) — la única diferencia estructural es el algoritmo y su
objective/criterion fijo.

| Métrica (Test, n=222)      | **XGBoost (final)** | LightGBM      | Random Forest |
|-----------------------------|:--------------------:|:-------------:|:-------------:|
| MAE                         | 54.132               | **53.388**    | 57.445        |
| RMSE                        | **80.644**           | 83.079        | 92.556        |
| R²                          | **0.8319**           | 0.8216        | 0.7786        |
| MAPE                        | 9.44%                | **9.28%**     | 9.87%         |
| MdAPE                       | 6.72%                | **6.87%** ⁽¹⁾ | 7.82%         |
| Skewness de residuos        | **1.02**              | 1.46          | 1.75          |
| Kurtosis de residuos        | **5.25**              | 8.41          | 11.11         |
| Bagging (nº modelos)        | 10                    | 5             | 5             |

⁽¹⁾ MdAPE de XGBoost (6.72%) es levemente mejor que LightGBM (6.87%); la negrita en esa fila
marca el valor más bajo salvo por ese caso, donde XGBoost gana.

**Objective/criterion fijo por modelo**: XGBoost `reg:squarederror`, LightGBM `regression`
(L2), Random Forest `squared_error` — en los tres casos, fijado manualmente en vez de dejarlo
como hiperparámetro de Optuna (ver sección 5).

**Baselines de comparación (Test)**: predecir siempre la media de train → MAE=130.177;
precio de mercado ingenuo (`precio_m2_sector_departamento × superficie_util_m2`) →
MAE=86.280. Los tres modelos superan ampliamente ambos baselines.

**MAE/MAPE por quintil de precio real (XGBoost, Test)**:

| Quintil | Rango (CLP)         | MAE     | MAPE   |
|---------|----------------------|---------|--------|
| Q1      | 250.000 – 430.000    | 45.010  | 11.96% |
| Q2      | 440.000 – 500.000    | 41.996  | 8.76%  |
| Q3      | 520.000 – 566.186    | 37.861  | 7.00%  |
| Q4      | 570.000 – 650.000    | 45.711  | 7.34%  |
| Q5      | 660.000 – 1.850.000  | 104.415 | 11.77% |

El quintil más caro (Q5) concentra la mayor parte del error **absoluto** (MAE), pero su error
**relativo** (MAPE) no es peor que el del quintil más barato — de ahí la decisión documentada
en la sección 5 de no separar el modelo en "premium vs. resto".

### Modelo final: XGBoost

Se eligió **XGBoost** por el mejor balance de métricas (mejor R² y RMSE) y, sobre todo, por
tener la **menor asimetría y kurtosis de residuos** de los tres — es decir, comete menos
errores extremos y su distribución de error es más estable — aunque LightGBM queda muy cerca
en MAE/MAPE. Random Forest quedó claramente por debajo en todas las métricas.

El modelo final se entrena como un **ensamble de bagging de 10 modelos** (mismos
hiperparámetros de Optuna, 10 semillas distintas; LightGBM y Random Forest usan 5 semillas
cada uno) y las predicciones finales son el promedio del ensamble — esto es, además, la base
del sistema de etiquetado de la sección 8.

---

## 8. Sistema de etiquetado "oportunidad / caro"

Implementado exclusivamente en `04_modelamiento/01_xgboost.py`, sobre el ensamble de bagging
de XGBoost (los 10 modelos, no un modelo único) y solo para el set de test.

**Lógica**:

1. Para cada aviso de test, se calcula el error `precio_real − precio_predicho` (predicho =
   promedio del ensamble de 10 modelos).
2. Ese error se normaliza de forma robusta **dentro de su propio decil de `precio_clp` real**
   (no del precio predicho): `z_robusto = (error − mediana_error_decil) / (MAD_error_decil ×
   1.4826)`. Se usa mediana/MAD en vez de media/desviación estándar porque cada decil tiene
   pocas filas en test (~20-30) y la mediana/MAD es menos sensible a outliers.
3. Etiqueta según umbral (`±1.0` en `z_robusto`):
   - **`oportunidad`**: precio real muy por debajo de lo esperado para su decil (z < −1.0).
   - **`caro`**: precio real muy por encima de lo esperado (z > 1.0).
   - **`precio_de_mercado`**: dentro del rango normal.
4. **Nivel de confianza** por fila, según el coeficiente de variación (std/mean) de las 10
   predicciones individuales del ensamble para esa fila: si los 10 modelos discrepan mucho
   entre sí, la etiqueta es menos confiable aunque el z_robusto sea grande. Se reporta en 3
   niveles (alta / media / baja confianza) según terciles del CV sobre el propio set de test.

**Distribución resultante (Test, n=222)**:

| Etiqueta            | Total | Alta confianza | Confianza media | Baja confianza |
|----------------------|:-----:|:---------------:|:-----------------:|:-----------------:|
| `precio_de_mercado`  | 145 (65,3%) | 52 | 52 | 41 |
| `oportunidad`        | 44 (19,8%)  | 6  | 14 | 24 |
| `caro`               | 33 (14,9%)  | 16 | 8  | 9  |

Nótese que las etiquetas `oportunidad` concentran proporcionalmente más casos de **baja**
confianza (24 de 44) que `caro` (9 de 33) — es decir, el modelo es menos consistente
identificando gangas que sobreprecios en este dataset.

Los resultados se exportan a `04_modelamiento/save/model/`:
- `xgboost_regression_precio_oportunidades_test.csv` (detalle fila por fila)
- `xgboost_regression_precio_oportunidades_resumen_decil.csv` (conteo por decil de precio)
- `xgboost_regression_precio_oportunidades_resumen_etiqueta_confianza.csv` (tabla de arriba)

---

## 9. Estructura de carpetas

```
gran-concepcion-rentals/
├── 01_obtener_datos/
│   ├── 01_scraper_grilla.py
│   ├── 02_scraper_detalle.py
│   ├── 03_vulnerabilidad_socioterritorial.py
│   ├── avisos_gran_concepcion.db          # SQLite, versionado en el repo
│   └── datos_vulnerabilidad/              # shapefile IGVUST, NO versionado (.gitignore)
│
├── 02_analisis_exploratorio/
│   └── 01_EDA.ipynb
│
├── 03_ingenieria_variables/
│   ├── 01_ingenieria_variables.py
│   ├── 02_seleccion_variables.py
│   └── save/
│       ├── ingeniaria_variables/
│       │   ├── datos_ingenieria_variables.csv
│       │   ├── niveles_barrio.json
│       │   └── modelos_superficie/*.pkl   # RandomForest de imputación de superficie
│       └── seleccion_variables/
│           ├── selected_features.csv
│           └── seleccion_variables_reporte.json
│
├── 04_modelamiento/
│   ├── 01_xgboost.py
│   ├── 02_lightgbm.py
│   ├── 03_random_forest.py
│   └── save/model/
│       ├── xgboost_regression_precio.pkl               # ensamble de 10 modelos
│       ├── xgboost_regression_precio_metrics.json
│       ├── xgboost_regression_precio_oportunidades_*.csv
│       ├── lightgbm_regression_precio.pkl
│       ├── lightgbm_regression_precio_metrics.json
│       ├── random_forest_regression_precio.pkl
│       └── random_forest_regression_precio_metrics.json
│
└── 05_modelo_produccion/          # pipeline de producción, ver sección 12
    ├── db.py                                    # esquema + conexión a produccion_gran_concepcion.db
    ├── 00_orquestador.py                        # corre las etapas de abajo en orden, logging + alertas
    ├── 01_scraper_grilla_incremental.py
    ├── 02_scraper_detalle_incremental.py
    ├── 03_vulnerabilidad_produccion.py
    ├── 04_ingenieria_variables_produccion.py
    ├── 05_prediccion.py
    ├── produccion_gran_concepcion.db            # SQLite, propia de este pipeline
    ├── logs/orquestador.log                     # log rotativo (RotatingFileHandler)
    └── entrenamiento/
        ├── 01_entrenar_modelo_produccion.py
        ├── version_modelo.json                  # contador + historial de versiones
        └── versiones/{version}/
            ├── modelo_produccion.pkl
            └── parametros_produccion.json
```

---

## 10. Dependencias

El repo no incluye un `requirements.txt`; estas son las dependencias reales, inferidas de los
`import` de cada etapa (probado con Python 3.11):

| Etapa                          | Librerías                                                        |
|----------------------------------|-------------------------------------------------------------------|
| Scraping (grilla)                | `requests`, `beautifulsoup4`, `lxml`, `pandas`                    |
| Scraping (detalle)                | `playwright`, `playwright-stealth` (opcional), `pandas`            |
| Vulnerabilidad socioterritorial   | `geopandas`, `shapely`, `pandas`                                   |
| Ingeniería de variables           | `pandas`, `numpy`, `requests`, `joblib`, `scikit-learn`            |
| Selección de variables            | `pandas`, `numpy`, `xgboost`, `optuna`, `scikit-learn`             |
| Modelamiento                      | `pandas`, `numpy`, `xgboost`, `lightgbm`, `optuna`, `scikit-learn`, `scipy` |

Instalación sugerida (sin versiones pineadas, ya que no existen en el repo):

```bash
pip install requests beautifulsoup4 lxml pandas playwright playwright-stealth \
            geopandas shapely scikit-learn joblib xgboost lightgbm optuna scipy
playwright install chromium
```

---

## 11. Limitaciones conocidas

**Del modelo:**
- El segmento premium (quintil más caro) tiene menos datos que el resto y concentra la mayor
  parte del error absoluto (MAE), aunque su error relativo (MAPE) esté en línea con el resto —
  ver sección 7. Con más datos en ese segmento el modelo probablemente mejoraría ahí en
  términos absolutos.
- No existen en el dataset variables de calidad, vista, terminaciones o estado de conservación
  real de la propiedad — el modelo infiere precio a partir de estructura, ubicación y contexto
  socioeconómico del sector, pero no "ve" fotos ni descripciones cualitativas.
- El dataset es de corte transversal y relativamente chico (1.476 filas tras limpieza); la
  validación multi-semilla existe para cuantificar esta sensibilidad al split, pero no se
  ejecuta por defecto en cada corrida por su costo computacional.
- Las variables derivadas adicionales (`crear_variables_derivadas`) están implementadas pero
  deshabilitadas: no hay evidencia confirmada en el código de que aporten mejora real sobre el
  ruido de partición.

**Del scraping:**
- Depende de la estructura HTML y las clases CSS del sitio, que pueden cambiar sin aviso —
  varios extractores usan regex sobre texto en español como respaldo, pero no eliminan el
  riesgo por completo.
- Riesgo de bloqueo/CAPTCHA, mitigado pero no eliminado por `playwright-stealth`, delays
  variables y ejecución en tandas pequeñas vía cron; existe un modo manual de resolución de
  CAPTCHA como respaldo.
- Las coordenadas geográficas de cada aviso suelen ser aproximadas al sector (no la dirección
  exacta), por diseño del sitio de origen — esto acota la precisión de las variables espaciales
  derivadas (distancias, comparables cercanos, cruce con Unidad Vecinal).
- El shapefile de vulnerabilidad socioterritorial (IGVUST) es un dato externo no versionado en
  el repo; regenerar ese cruce desde cero requiere obtenerlo por separado.

---

## 12. Pipeline de producción (`05_modelo_produccion/`)

Sistema separado e independiente de `01_obtener_datos/` a `04_modelamiento/`: usa su **propia
base de datos** (`produccion_gran_concepcion.db`), nunca escribe en `avisos_gran_concepcion.db`
(la trata como fuente de solo lectura), y está pensado para correr sin intervención manual vía
cron, agregando avisos nuevos y sus predicciones día a día.

### 12.1. Parte 1 — Entrenamiento y versionado (`entrenamiento/01_entrenar_modelo_produccion.py`)

- Reutiliza `04_modelamiento/01_xgboost.py` (cargado vía `importlib`, ya que su nombre empieza
  con dígito) — no duplica la lógica de optimización de hiperparámetros, bagging ni SHAP.
- **Split 85/15** (train/test) en vez de 70/15/15: el ensamble de bagging igual necesita un set
  para early stopping, así que se separa internamente un 10% del 85% de train solo para eso
  (~76.5% train_fit / ~8.5% early-stopping / 15% test) — un detalle interno de entrenamiento,
  no una tercera partición pública.
- **Versionado**: cada corrida genera un identificador `v{contador:04d}_{timestamp}_{hash8}`
  (contador incremental + hash sha256 de los hiperparámetros ganadores), registrado en
  `version_modelo.json` (contador + historial). El modelo y sus parámetros se **archivan por
  versión** en `versiones/{version}/` (no se sobrescriben) — así se puede recuperar el modelo
  exacto usado en cualquier predicción pasada.
- **Calibración de oportunidad/confianza**: además de las métricas de evaluación estándar,
  calcula y persiste (sobre el set de test) los bordes de deciles de precio, la mediana/MAD del
  error por decil, y los terciles del coeficiente de variación del ensamble — todo guardado en
  `parametros_produccion.json` bajo `calibracion_oportunidad`, para poder etiquetar avisos
  nuevos en la Parte 2 sin recalcular una distribución con una sola fila (imposible con qcut).
- Dataset de entrada: por ahora, el mismo CSV curado que usa el modelo de investigación
  (`datos_ingenieria_variables.csv` + `selected_features.csv`).

### 12.2. Parte 2 — Pipeline incremental

**Esquema de `produccion_gran_concepcion.db`** (definido en `db.py`, normalizado sin llevarlo
al extremo):

| Tabla | Contenido |
|---|---|
| `avisos` | Nivel grilla + `estado_publicacion` (activo/pausado/finalizado) + `fecha_ultimo_chequeo_estado` |
| `avisos_detalle` | Nivel detalle (1:1 con `avisos`) + columnas de vulnerabilidad IGVUST resueltas directo (sin las tablas separadas `vulnerabilidad_uv`/`avisos_igvust` de la base original) |
| `predicciones` | Una fila por `(id_aviso, version_modelo)` — precio predicho, z_robusto, decil, etiqueta, confianza |
| `corridas` | Metadatos de cada corrida del orquestador (contadores, motivo de corte, resultado) |
| `logs_ejecucion` | Log persistente por etapa, espejo de `logs/orquestador.log` pero consultable con SQL |
| `control` | Clave/valor genérico (ej. cooldown tras CAPTCHA del scraper de detalle) |

**Etapas** (cada una reutiliza el script equivalente de `01_obtener_datos/`/`03_ingenieria_variables/`
vía `importlib`, sin duplicar lógica de parsing/extracción):

1. **`01_scraper_grilla_incremental.py`** — recorre la grilla de búsqueda guardando solo avisos
   cuyo `id_aviso` no exista en la base original NI en producción. Corta por
   `MAX_PAGINAS_VACIAS_CONSECUTIVAS = 10` páginas seguidas sin avisos nuevos (por combinación
   comuna×tipo), o por techo de presupuesto (`MAX_PAGINAS_POR_CORRIDA = 200`,
   `MAX_MINUTOS_POR_CORRIDA = 30`) si se alcanza antes — el motivo de corte queda registrado.
2. **`02_scraper_detalle_incremental.py`** — visita avisos nuevos sin detalle, y además
   **re-chequea** avisos `activo` con más de `DIAS_MIN_ENTRE_RECHEQUEOS = 7` días desde su
   último chequeo (batch de `MAX_AVISOS_RECHEQUEO_POR_CORRIDA = 50`, los más antiguos primero).
   Extrae `estado_publicacion` del mismo JSON embebido que ya se usa para los puntos de interés
   (busca el componente `item_status_message`/`item_status_short_description_message` dentro de
   `components.head`/`components.short_description`; si no aparece, el aviso está activo). El
   guardado usa **UPSERT** (no `INSERT OR REPLACE`) para que un re-chequeo nunca borre las
   columnas de vulnerabilidad que llena la etapa siguiente.
3. **`03_vulnerabilidad_produccion.py`** — cruce punto-en-polígono contra el mismo shapefile
   IGVUST, resuelto directo a columnas de `avisos_detalle` (no a tablas de referencia
   separadas). Solo procesa avisos con coordenadas y `uv_rsh` todavía `NULL` (incremental).
4. **`04_ingenieria_variables_produccion.py`** — calcula las 20 features del modelo para avisos
   nuevos, pero **sin recalcular nada en modo batch** (a diferencia del pipeline de
   investigación): compara cada aviso contra una **población de referencia fija**
   (`datos_ingenieria_variables.csv` + coordenadas/comuna recuperadas con un `SELECT` de solo
   lectura contra la base original) vía `BallTree` para `precio_m2_sector_departamento` y la
   imputación de `antiguedad_anos`, y reutiliza los modelos de imputación de superficie ya
   entrenados (`aplicar_modelo_guardado`) sin reentrenar nada.
5. **`05_prediccion.py`** — predice con el ensamble vigente, convierte el precio real del aviso
   a CLP (reutiliza `convertir_precios_uf_a_clp` si `moneda='UF'`), aplica la calibración
   guardada en la Parte 1 para calcular `z_robusto`/etiqueta/confianza, e inserta en
   `predicciones` vía UPSERT sobre `(id_aviso, version_modelo)`.

### 12.3. Orquestador (`00_orquestador.py`)

Corre las 5 etapas en orden. Por cada corrida:
- Crea una fila en `corridas` (resultado inicial `'parcial'`, se actualiza al final).
- Loggea a un archivo rotativo (`logs/orquestador.log`, `RotatingFileHandler`, 5 MB × 5
  backups) **y** a la tabla `logs_ejecucion`, etiquetando cada mensaje con la etapa vigente
  (una variable global simple que lee un `logging.Handler` propio — funciona sin importar desde
  qué submódulo anidado venga el log).
- Si una etapa lanza una excepción, se detiene ahí mismo (las etapas siguientes NO corren),
  registra `ERROR` con traceback completo en archivo y BD, y marca `corridas.resultado='error'`
  con `etapa_fallida`/`mensaje_error`.
- Si el scraper de detalle se detiene por CAPTCHA (no es una excepción, es un corte limpio ya
  manejado por esa etapa), se sigue igual con las etapas siguientes sobre lo ya procesado, y la
  corrida queda como `'parcial'` en vez de `'ok'`.
- **Alerta de fallo silencioso**: al final de cada corrida exitosa/parcial, revisa si las
  últimas `N_CORRIDAS_CONSECUTIVAS_SIN_NUEVOS = 5` corridas tuvieron 0 avisos nuevos en la
  grilla, y deja un `WARNING` explícito — podría ser un cambio en la estructura del sitio o un
  bloqueo no detectado, no necesariamente falta de contenido nuevo.

Pensado para cron, ej.:
```bash
0 */4 * * * cd /ruta/al/proyecto/05_modelo_produccion && /ruta/al/python 00_orquestador.py
```

### 12.4. Notas y limitaciones conocidas de esta sección

- `gastos_comunes` en producción replica a propósito el mismo quirk del pipeline de
  investigación (pierde el separador de miles chileno: "$120.000" → `120.0`, no `120000.0`),
  para que el dato sea comparable con el histórico que ya vio el modelo. Corregirlo requiere
  hacerlo en ambos lados a la vez y reentrenar.
- La población de referencia de la etapa de variables (comparables de precio/m² e imputación de
  antigüedad) usa **solo** el dataset histórico de investigación, no los avisos de producción ya
  procesados — más simple y estable, a costa de no reflejar aún datos más recientes del mercado.
- Los bordes `-inf`/`inf` en `calibracion_oportunidad` (para clasificar avisos fuera del rango
  de precio/CV visto en test) se serializan como `-Infinity`/`Infinity`, válido para Python pero
  no JSON estándar — una herramienta no-Python que lea ese archivo directo fallaría en esos dos
  campos.
- El pipeline de producción solo predice `tipo_propiedad='departamento'` (igual que el modelo);
  las casas se scrapean pero nunca se puntúan.
