# Gran Concepción Rentals

Pipeline completo de datos, desde scraping hasta modelamiento, para estimar el precio de
arriendo de **departamentos** en las comunas del Gran Concepción (Chile) y detectar avisos
que están publicados por debajo o por encima de lo que el mercado local justifica.

El proyecto cubre seis etapas encadenadas:

1. **Scraping** de avisos de arriendo (Portal Inmobiliario) + cruce con un índice público de
   vulnerabilidad socioterritorial.
2. **Análisis exploratorio** de los datos crudos.
3. **Ingeniería y selección de variables**: limpieza, imputación, variables derivadas y
   selección estadística de las features finales.
4. **Modelamiento**: comparación de XGBoost y LightGBM para predecir `precio_clp`, cada uno con
   su propio sistema de etiquetado "oportunidad / caro / precio de mercado" construido sobre su
   ensamble de bagging. El algoritmo que entrena el modelo de **producción** se decide de forma
   automática (ver sección 12.1); a la fecha es **LightGBM**.
5. **Producción** (`05_modelo_produccion/`): pipeline separado, re-ejecutable vía cron, que
   entrena y versiona un modelo de producción, scrapea avisos nuevos de forma incremental,
   y genera predicciones + etiquetas sobre una base de datos propia. Ver sección 12.
6. **Visualización** (`06_visualizacion/`): dashboard Streamlit que lee directo de
   `produccion_gran_concepcion.db` (sin escribir en ella) y muestra los avisos con predicción
   como tarjetas filtrables + mapa, para explorar oportunidades sin tocar SQL. Ver sección 14.

> El pipeline de modelamiento trabaja exclusivamente sobre **departamentos**. Los scrapers de
> **investigación** (`01_obtener_datos/`) sí recolectan casas, pero la etapa de ingeniería de
> variables filtra y trabaja solo con `tipo_propiedad = "departamento"`. El scraper de grilla de
> **producción** (`05_modelo_produccion/01_scraper_grilla_incremental.py`) va un paso más allá y
> directamente **no recorre casas** (`TIPOS_PROPIEDAD_PRODUCCION = ["departamento"]`), ya que el
> resto del pipeline de producción las descartaría de todas formas — evita gastar presupuesto de
> scraping en avisos que nunca generan features ni predicción (ver sección 13).

---

### 🔎 Visualización interactiva

Puedes explorar los resultados del modelo y las predicciones sobre los avisos vigentes mediante
la aplicación desplegada en Streamlit:

**[gran-concepcion-arriendos-departamentos.streamlit.app](https://gran-concepcion-arriendos-departamentos.streamlit.app)**

---

## 1. Arquitectura del pipeline

```
01_obtener_datos/
  01_scraper_grilla.py                  → tabla `avisos`               (requests + BeautifulSoup)
  02_scraper_detalle.py                 → tabla `avisos_detalle`       (requests; Playwright solo como respaldo)
  03_vulnerabilidad_socioterritorial.py → tablas `vulnerabilidad_uv`,
                                           `avisos_igvust`               (geopandas, cruce espacial)
        │  (todo persiste en avisos_gran_concepcion.db, SQLite)
        ▼
02_analisis_exploratorio/
  01_EDA.ipynb                          → exploración manual de los datos crudos
        ▼
03_ingenieria_variables/
  01_ingenieria_variables.py            → datos_ingenieria_variables.csv (1.628 filas × 42 features)
  02_seleccion_variables.py             → selected_features.csv          (32 features finales)
        ▼
04_modelamiento/
  01_xgboost.py          → bagging ×10 + etiquetado oportunidad/caro
  02_lightgbm.py         → bagging ×10 + etiquetado oportunidad/caro (misma API que 01_xgboost.py)
        ▼
05_modelo_produccion/   → pipeline de producción, separado e independiente (ver sección 12)
  entrenamiento/seleccionar_algoritmo.py         → compara xgboost vs lightgbm (JSON de métricas
                                                    de investigación) y elige el algoritmo ganador
  entrenamiento/01_entrenar_modelo_produccion.py → entrena el algoritmo ganador, modelo versionado
                                                    (85/15 + calibración)
  00_orquestador.py                              → corre las etapas 1-5 de abajo en orden
  01_scraper_grilla_incremental.py               → tabla `avisos`          (produccion_gran_concepcion.db)
  02_scraper_detalle_incremental.py              → tabla `avisos_detalle` + estado_publicacion
  03_vulnerabilidad_produccion.py                → columnas de vulnerabilidad en `avisos_detalle`
  04_ingenieria_variables_produccion.py          → features de avisos nuevos (contra referencia histórica)
  05_prediccion.py                               → tabla `predicciones` (precio + etiqueta + confianza)
        │  (produccion_gran_concepcion.db, solo lectura desde acá en adelante)
        ▼
06_visualizacion/       → dashboard Streamlit, ver sección 14
  app.py                → tarjetas filtrables + mapa (st.tabs: Buscador / Cómo funciona)
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

Instala las dependencias de Python (ver sección [Dependencias](#10-dependencias)):

```bash
pip install requests beautifulsoup4 lxml pandas \
            geopandas shapely scikit-learn joblib xgboost lightgbm optuna scipy
```

Playwright **no es necesario** para el camino normal (ambos scrapers usan `requests`, sin
navegador). Solo instálalo si vas a usar la ruta de respaldo de `02_scraper_detalle.py`
(`--fallback-playwright`, ver sección 3.2):

```bash
pip install playwright playwright-stealth
playwright install chromium
```

### 2.2. Camino rápido — usar los datos ya incluidos

Si solo quieres reproducir la ingeniería de variables y el modelamiento (sin volver a
scrapear), corre en orden desde la raíz del repo:

```bash
python 03_ingenieria_variables/01_ingenieria_variables.py
python 03_ingenieria_variables/02_seleccion_variables.py
python 04_modelamiento/01_xgboost.py   # entrenamiento + etiquetado oportunidad/caro
python 04_modelamiento/02_lightgbm.py  # entrenamiento + etiquetado oportunidad/caro
```

### 2.3. Camino completo — scraping desde cero

```bash
# 1. Grilla de búsqueda (requests + BeautifulSoup, sin navegador)
python 01_obtener_datos/01_scraper_grilla.py

# 2. Detalle de cada aviso (requests, sin navegador - más sensible a bloqueo que
#    la grilla por el volumen de visitas, aunque no se ha observado bloqueo en la práctica).
#    Pensado para correr en tandas vía cron, no de una sola sentada
#    (ver LIMITE_POR_CORRIDA y COOLDOWN_TRAS_CAPTCHA_MINUTOS en el script).
python 01_obtener_datos/02_scraper_detalle.py

# 2b. Solo si la ruta principal empezara a bloquearse de forma persistente (no
#     observado hasta ahora) o necesitas resolver un CAPTCHA a mano: ruta de
#     respaldo con Playwright (requiere pip install playwright playwright-stealth
#     && playwright install chromium - ver sección 2.1).
# python 01_obtener_datos/02_scraper_detalle.py --fallback-playwright

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

Esta variable corregida (`precio_m2_sector_departamento`) sí es una de las 32 features
finales del modelo — la diferencia crítica es que su fuente son los vecinos, nunca la propia
fila.

### 3.2. Scraping

- Arquitectura de **dos scrapers separados**: `01_scraper_grilla.py` recorre las páginas de
  resultados de búsqueda (requests + BeautifulSoup, sin navegador), y `02_scraper_detalle.py`
  visita cada aviso individual (también requests, sin navegador) para extraer descripción,
  características y puntos de interés cercanos.
- **Migración de Playwright a `requests`** (confirmada equivalente con pruebas de 6 URLs
  comparadas 1:1 y una corrida de volumen de 150 requests seguidas, sin señales de bloqueo):
  el HTML servido por el sitio ya trae server-side el mismo JSON embebido y el mismo texto de
  características que renderizaba Playwright, así que dejó de ser necesario un navegador para
  la ruta normal. La lógica de Playwright se conservó como **ruta de respaldo**
  (`main_fallback_playwright()`, invocable con `--fallback-playwright`) para el caso de que el
  sitio empiece a bloquear las requests simples, o para resolver un CAPTCHA a mano. Su import
  es **perezoso** (ocurre recién al llamar esa función), así que el resto del script - incluida
  la ruta principal - funciona sin problema en entornos donde Playwright no está instalado (ej.
  GitHub Actions).
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
- **Mitigaciones de bloqueo**: delays variables entre requests, reintento automático (con
  backoff corto) ante un fallo de red/HTTP aislado dentro de la misma corrida - confirmado que
  un 404 puntual puede ser transitorio y no reproducirse al reintentar segundos después - y
  detección de CAPTCHA por **doble condición** (la palabra "captcha" aparece en el HTML **Y**
  el contenido normal del aviso no cargó) para no confundir el script de reCAPTCHA de fondo
  (presente casi siempre) con un bloqueo real.
- **Fallos persistentes entre corridas** (`05_modelo_produccion/02_scraper_detalle_incremental.py`):
  un aviso que falla incluso tras los reintentos nunca queda en `avisos_detalle`, así que sin
  más el `LEFT JOIN` de pendientes lo volvería a traer en cada corrida futura sin límite -
  incluso si el aviso fue realmente eliminado del sitio. El contador
  `avisos.intentos_fallidos_detalle` suma 1 en cada fallo y se resetea a 0 en cuanto el aviso
  se scrapea con éxito; al superar `MAX_INTENTOS_FALLIDOS_DETALLE = 5` fallos consecutivos, el
  aviso se marca `estado_publicacion = 'no_disponible'` (estado distinto de `'finalizado'`, que
  significa que el arriendo terminó con normalidad) y sale de la cola de pendientes.
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
- **Bug corregido: `gastos_comunes` perdía el separador de miles chileno** (`"82.000"` se
  interpretaba como `82.0` en vez de `82000.0`) — afectaba 1411 de 1956 avisos del histórico
  (72%). La causa: `float()` directo sobre el texto crudo, sin manejar el punto como separador
  de miles. Corregido en `RE_GASTOS_COMUNES`/`_a_gastos_comunes()`
  (`05_modelo_produccion/02_scraper_detalle_incremental.py`), que ahora distingue tres casos
  según el texto crudo:
  - **con punto** (`"N.NNN"`): separador de miles real → se elimina y se convierte
    (`"82.000"` → `82000.0`).
  - **dígito suelto sin punto y < 1.000** (ej. `"1"`, `"10"`): no es un monto real — es un
    **placeholder de "gastos comunes incluidos en el arriendo"** que el sitio muestra cuando el
    campo no puede quedar vacío (confirmado contra el HTML en vivo: la tabla de características
    dice "1 CLP" mientras la descripción del aviso dice textualmente "Gastos comunes
    incluidos"). Se mapea a `0.0` (mismo tratamiento que un `$0` explícito), sin agregar una
    columna booleana aparte — la distinción no cambia nada para el modelo (carga mensual
    adicional = cero en ambos casos) y siempre se puede reconstruir desde el texto crudo si hace
    falta para otro fin.
  - **sin punto y ≥ 500.000** (visto una vez en 1766 avisos, ej. `"1.111.111"`): outlier
    implausible — se descarta a `NULL` en vez de adivinar una transformación.

  De paso, se agregó un **fallback de extracción** (`RE_GASTOS_COMUNES_RESUMEN`): el campo
  también aparece en la insignia de resumen superior de la página ("Gastos comunes desde $X"),
  formato que el regex original nunca lograba matchear por la palabra "desde" — se usa como
  respaldo cuando la tabla de características no trae el dato. Verificado en vivo sobre los 190
  avisos históricos con `gastos_comunes NULL`: **102 (54%) sí tenían el dato en el resumen** y
  se recuperaron; los 88 restantes son ausencia genuina (52 de ellos son `casa`, que en general
  no reportan gastos comunes en ninguna parte de la página — consistente con que "gastos
  comunes" es un concepto de edificios/condominios).

  El histórico (`avisos_gran_concepcion.db`) se corrigió con un backfill de una sola vez
  (backup previo vía `.bak-pre-backfill-gastos-comunes`), y `03_ingenieria_variables/01_ingenieria_variables.py`
  + `04_modelamiento/01_xgboost.py`/`02_lightgbm.py` se re-corrieron sobre los datos limpios
  (ver sección 13).

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

## 4. Features del modelo final (32)

Seleccionadas por `03_ingenieria_variables/02_seleccion_variables.py` a partir de 42 features
candidatas (ver sección 6 para la metodología de selección). Mismas 32 features para XGBoost y
LightGBM (ambos se comparan sobre exactamente el mismo set, ver sección 7).

**Características físicas de la propiedad**
- `superficie_util_m2`, `superficie_total_m2`, `ratio_total_util`
- `banos`, `estacionamientos`, `estacionamiento_visitas`, `bodegas`, `piso_unidad`, `ascensor`,
  `piscina`, `amoblado`, `conserjeria`, `condominio_cerrado`
- `antiguedad_anos`

**Costos asociados**
- `gastos_comunes`

**Ubicación y mercado local**
- `precio_m2_sector_departamento` (comparables cercanos, ver sección 3.1)
- `tiene_comparables_cercanos` (flag de confiabilidad del anterior)
- `nivel_barrio` (precio/m² suavizado por barrio, ver sección 3.4)
- `distancia_centro_concepcion_m`, `distancia_centro_comuna_m`
- `cantidad_paraderos`, `cantidad_colegios`, `cantidad_supermercados`,
  `cantidad_jardines_infantiles`, `cantidad_centros_comerciales`, `cantidad_plazas`,
  `cantidad_farmacias`, `cantidad_clinicas`

**Contexto socioeconómico del sector (índice IGVUST / Registro Social de Hogares, por
Unidad Vecinal)**
- `rank_nac` (ranking nacional de vulnerabilidad de la Unidad Vecinal)
- `pob_rsh_uv` (población registrada en el RSH de la Unidad Vecinal)
- `p_urbano` (porcentaje urbano de la Unidad Vecinal)
- `c_ig_com` (índice de vulnerabilidad comunal IGVUST)

---

## 5. Evolución del modelamiento

El proyecto pasó por dos grandes etapas de modelamiento:

**Etapa inicial** — modelo base de árboles, evaluado con MAE/RMSE/R²/MAPE y la razón RMSE/MAE
como diagnóstico de concentración de error en casos extremos (mejoró de ~2.7 a ~2.1 tras la
limpieza de datos). En esta etapa también se probó `log(precio_clp)` como target y se descartó
por no aportar mejora, y se revisaron manualmente (URL por URL) los casos de error más extremo
para descartar errores de datos frente a variabilidad genuina de mercado.

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
- **Comparación de dos arquitecturas** (XGBoost, LightGBM) sobre las mismas 32 features, mismo
  split y misma seed (ver sección 7). Random Forest se probó en una etapa anterior del proyecto
  (con menos features y datos sin corregir) y quedó consistentemente por debajo de las otras
  dos arquitecturas, por lo que se eliminó del pipeline de modelamiento para simplificarlo — el
  código actual de `04_modelamiento/` solo compara XGBoost y LightGBM.
- Se **descartó separar el modelo en "premium vs. resto"**: el MAPE del quintil más caro (Q5)
  no es dramáticamente peor que el resto de los quintiles — ver sección 7 — por lo que el
  problema real es volumen de datos en ese segmento, no una estructura de precios distinta.

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

`03_ingenieria_variables/02_seleccion_variables.py` reduce 42 features candidatas a las 32
finales en 4 pasos:

1. **Eliminación de constantes**: features con varianza ≈ 0 sobre train. En la corrida
   registrada, se eliminó `quincho` (varianza cero: prácticamente ningún departamento
   reportaba esa amenity) — quedan 41 candidatas.
2. **Selección por estabilidad**: se entrenan 30 modelos XGBoost con K-Fold aleatorio (5
   folds, semillas distintas), midiendo importancia SHAP (TreeSHAP nativo) de cada feature en
   cada fold. `stability_score = (1 / (1 + CV)) × presence_pct`, donde CV es el coeficiente de
   variación de la importancia entre modelos y `presence_pct` el % de modelos donde la
   importancia fue > 0.
3. **Selección de k óptimo vía MAE de validación**: se evalúa la curva MAE/RMSE/R² en el set
   de validación para k features (ordenadas por `stability_score`), promediando sobre 5
   semillas, y se aplica la **regla de 1 error estándar** (el k más chico cuyo MAE promedio
   cae dentro de 1 SE del mínimo) — resultado: **k=33**.
4. **Red de seguridad de correlación**: elimina pares de features con correlación > 0.95 entre
   las ya seleccionadas, quedándose con la de mayor `stability_score`. En la corrida registrada
   eliminó `hog_uv` (1 de 33), dejando las **32 features finales**.

---

## 7. Modelos comparados y selección de algoritmo

Los dos modelos comparten exactamente las mismas 32 features, el mismo split estratificado
(seed=42; 1.138 train / 245 val / 245 test) y el mismo esquema de optimización (Optuna, 50
trials, KFold=5, CV solo sobre train) — la única diferencia estructural es el algoritmo y su
objective/criterion fijo.

Se incluyen además dos baselines *naive* (sin aprendizaje, solo aritmética simple) como piso de
comparación: predecir siempre la media de `precio_clp` de train, y un "precio de mercado
ingenuo" (`precio_m2_sector_departamento × superficie_util_m2`, sin ajuste ninguno). Estos
baselines solo reportan MAE (no se les calculan el resto de las métricas, al no ser modelos
entrenados) — el resto de las celdas queda como "—".

| Métrica (Test, n=245)         | Media de train (naive) | Precio × m² ingenuo (naive) | XGBoost | **LightGBM** |
|--------------------------------|:-----------------------:|:-----------------------------:|:-------:|:------------:|
| MAE                             | 125.949                 | 79.200                        | 49.237  | **48.901**   |
| RMSE                            | —                        | —                              | 74.455  | **73.890**   |
| R²                              | —                        | —                              | 0.8391  | **0.8416**   |
| MAPE                            | —                        | —                              | 8.65%   | **8.57%**    |
| MdAPE                           | —                        | —                              | 6.50%   | **6.16%**    |
| Skewness de residuos            | —                        | —                              | -0.83   | **-0.60**    |
| Kurtosis de residuos            | —                        | —                              | 9.79    | **8.68**     |
| Bagging (nº modelos)            | —                        | —                              | 10      | 10           |

Ambos modelos superan ampliamente los dos baselines: LightGBM reduce el MAE en **61%** frente
al baseline de media de train y en **38%** frente al de precio/m² ingenuo.

**Objective/criterion fijo por modelo**: XGBoost `reg:squarederror`, LightGBM `regression`
(L2) — fijado manualmente en vez de dejarlo como hiperparámetro de Optuna (ver sección 5).

**MAE/MAPE por quintil de precio real (Test)**:

| Quintil | Rango (CLP)         | MAE XGBoost | MAPE XGBoost | MAE LightGBM | MAPE LightGBM |
|---------|----------------------|:-----------:|:------------:|:------------:|:--------------:|
| Q1      | 250.000 – 430.000    | 35.624      | 9.62%        | **34.910**   | **9.46%**      |
| Q2      | 435.000 – 500.000    | 36.145      | 7.51%        | **35.384**   | **7.35%**      |
| Q3      | 520.000 – 565.000    | 37.525      | 6.97%        | **38.680**   | 7.19%          |
| Q4      | 570.000 – 650.000    | 53.789      | 8.74%        | **52.252**   | **8.49%**      |
| Q5      | 670.000 – 1.850.000  | **86.934**  | **10.21%**   | 87.909       | 10.30%         |

El quintil más caro (Q5) concentra la mayor parte del error **absoluto** (MAE) en ambos
modelos, pero su error **relativo** (MAPE) no es peor que el de los quintiles más baratos — de
ahí la decisión documentada en la sección 5 de no separar el modelo en "premium vs. resto".
Nótese que XGBoost es levemente mejor que LightGBM justo en Q5 (el segmento más caro), aunque
pierda en todas las demás métricas globales — un patrón que `seleccionar_algoritmo.py` detecta
y reporta como advertencia (ver sección 12.1), sin que cambie la decisión final.

### Selección de algoritmo para producción

A diferencia de etapas anteriores del proyecto (con Random Forest como tercera opción, ver
sección 5), el pipeline actual **no fija editorialmente un "modelo final" único** en
investigación: los dos scripts de `04_modelamiento/` se entrenan y evalúan en paralelo sobre
las mismas 32 features, y `05_modelo_produccion/entrenamiento/seleccionar_algoritmo.py` decide
de forma automática cuál entrena el modelo de **producción**, comparando los JSON de métricas
de test más recientes con un criterio ponderado (50% MAE + 50% RMSE, normalizado). Con la
corrida vigente, el ganador es **LightGBM**, por un margen relativo de apenas **0.72%** — ver
sección 12.1 para el detalle del mecanismo.

Cada modelo se entrena como un **ensamble de bagging de 10 modelos** (mismos hiperparámetros
de Optuna, 10 semillas distintas) y las predicciones finales son el promedio del ensamble —
esto es, además, la base del sistema de etiquetado de la sección 8.

---

## 8. Sistema de etiquetado "oportunidad / caro"

Implementado en ambos scripts de `04_modelamiento/` (`01_xgboost.py` y `02_lightgbm.py`, misma
lógica en los dos), sobre el ensamble de bagging propio de cada uno (los 10 modelos, no un
modelo único) y solo para el set de test. Los números de esta sección corresponden a
**LightGBM** (el algoritmo vigente en producción, ver sección 7).

**Lógica**:

1. Para cada aviso de test, se calcula el error `precio_real − precio_predicho` (predicho =
   promedio del ensamble de 10 modelos).
2. Ese error se normaliza de forma robusta **dentro de su propio decil de `precio_clp` real**
   (no del precio predicho): `z_robusto = (error − mediana_error_decil) / (MAD_error_decil ×
   1.4826)`. Se usa mediana/MAD en vez de media/desviación estándar porque cada decil tiene
   pocas filas en test (~24-30) y la mediana/MAD es menos sensible a outliers.
3. Etiqueta según umbral (`±1.0` en `z_robusto`):
   - **`oportunidad`**: precio real muy por debajo de lo esperado para su decil (z < −1.0).
   - **`caro`**: precio real muy por encima de lo esperado (z > 1.0).
   - **`precio_de_mercado`**: dentro del rango normal.
4. **Nivel de confianza** por fila, según el coeficiente de variación (std/mean) de las 10
   predicciones individuales del ensamble para esa fila: si los 10 modelos discrepan mucho
   entre sí, la etiqueta es menos confiable aunque el z_robusto sea grande. Se reporta en 3
   niveles (alta / media / baja confianza) según terciles del CV sobre el propio set de test.

**Distribución resultante (LightGBM, Test, n=245)**:

| Etiqueta            | Total | Alta confianza | Confianza media | Baja confianza |
|----------------------|:-----:|:---------------:|:-----------------:|:-----------------:|
| `precio_de_mercado`  | 168 (68,6%) | 68 | 59 | 41 |
| `oportunidad`        | 47 (19,2%)  | 7  | 15 | 25 |
| `caro`               | 30 (12,2%)  | 7  | 7  | 16 |

Nótese que tanto `oportunidad` como `caro` concentran proporcionalmente la misma fracción de
casos de **baja** confianza (25 de 47 ≈ 53% vs. 16 de 30 ≈ 53%) — a diferencia de corridas
anteriores del proyecto, en esta el modelo es igual de consistente identificando gangas que
sobreprecios.

Los resultados se exportan a `04_modelamiento/save/model/` con un prefijo por algoritmo
(`xgboost_regression_precio_*` y `lightgbm_regression_precio_*`, cada script genera el suyo):
- `..._oportunidades_test.csv` (detalle fila por fila)
- `..._oportunidades_resumen_decil.csv` (conteo por decil de precio)
- `..._oportunidades_resumen_etiqueta_confianza.csv` (tabla de arriba)

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
│       │   └── modelos_superficie/*.pkl   # RandomForest de IMPUTACIÓN de superficie (no es el
│       │                                  # modelo de precio, ver sección 3.3)
│       └── seleccion_variables/
│           ├── selected_features.csv
│           └── seleccion_variables_reporte.json
│
├── 04_modelamiento/
│   ├── 01_xgboost.py
│   ├── 02_lightgbm.py
│   └── save/model/
│       ├── xgboost_regression_precio.pkl               # ensamble de 10 modelos
│       ├── xgboost_regression_precio_metrics.json
│       ├── xgboost_regression_precio_oportunidades_*.csv
│       ├── lightgbm_regression_precio.pkl
│       ├── lightgbm_regression_precio_metrics.json
│       └── lightgbm_regression_precio_oportunidades_*.csv
│
├── 05_modelo_produccion/          # pipeline de producción, ver sección 12
│   ├── db.py                                    # esquema + conexión a produccion_gran_concepcion.db
│   ├── 00_orquestador.py                        # corre las etapas de abajo en orden, logging + alertas
│   ├── 01_scraper_grilla_incremental.py
│   ├── 02_scraper_detalle_incremental.py
│   ├── 03_vulnerabilidad_produccion.py
│   ├── migrar_poligonos_vulnerabilidad.py       # migración manual/local: shapefile -> tabla poligonos_vulnerabilidad_uv
│   ├── 04_ingenieria_variables_produccion.py
│   ├── 05_prediccion.py
│   ├── requirements.txt                         # dependencias pineadas para GitHub Actions (sin geopandas ni playwright)
│   ├── produccion_gran_concepcion.db            # SQLite, propia de este pipeline
│   ├── logs/orquestador.log                     # log rotativo (RotatingFileHandler)
│   └── entrenamiento/
│       ├── seleccionar_algoritmo.py             # compara xgboost vs lightgbm, elige ganador
│       ├── algoritmo_seleccionado.json          # decisión persistida (algoritmo + métricas)
│       ├── 01_entrenar_modelo_produccion.py     # entrena el algoritmo ganador
│       ├── version_modelo.json                  # contador + historial de versiones
│       └── versiones/{version}/
│           ├── modelo_produccion.pkl
│           └── parametros_produccion.json
│
└── 06_visualizacion/               # dashboard Streamlit, ver sección 14
    ├── app.py                                   # entrypoint: st.tabs (Buscador / Cómo funciona)
    ├── data.py                                  # query + join + estandarización de precio
    ├── filters.py                               # sidebar de filtros + lógica de filtrado
    ├── components.py                            # tarjetas de aviso + mapa folium
    ├── explicacion.py                           # contenido de la pestaña "Cómo funciona"
    ├── styles.py                                # paleta de colores + CSS compartido
    ├── requirements.txt
    └── .streamlit/config.toml                   # tema forzado a claro
```

---

## 10. Dependencias

El repo no incluye un `requirements.txt` en la raíz (sí uno propio en `06_visualizacion/`, ver
sección 14.5); estas son las dependencias reales de las demás etapas, inferidas de los
`import` de cada una (probado con Python 3.11):

| Etapa                          | Librerías                                                        |
|----------------------------------|-------------------------------------------------------------------|
| Scraping (grilla)                | `requests`, `beautifulsoup4`, `lxml`, `pandas`                    |
| Scraping (detalle)                | `requests`, `beautifulsoup4`, `lxml`, `pandas` — `playwright`/`playwright-stealth` opcionales, solo para la ruta de respaldo (`--fallback-playwright`) |
| Vulnerabilidad socioterritorial   | `geopandas`, `shapely`, `pandas`                                   |
| Ingeniería de variables           | `pandas`, `numpy`, `requests`, `joblib`, `scikit-learn`            |
| Selección de variables            | `pandas`, `numpy`, `xgboost`, `optuna`, `scikit-learn`             |
| Modelamiento                      | `pandas`, `numpy`, `xgboost`, `lightgbm`, `optuna`, `scikit-learn`, `scipy` |
| Visualización (`06_visualizacion/`) | `streamlit`, `folium`, `streamlit-folium`, `pandas`, `numpy`, `requests`, `joblib`, `scikit-learn` — las últimas cuatro porque `data.py` importa dinámicamente `03_ingenieria_variables/01_ingenieria_variables.py` (ver sección 14.2) |

Instalación sugerida (sin versiones pineadas, ya que no existen en el repo):

```bash
pip install requests beautifulsoup4 lxml pandas \
            geopandas shapely scikit-learn joblib xgboost lightgbm optuna scipy

# Opcional, solo para la ruta de respaldo de 02_scraper_detalle.py:
pip install playwright playwright-stealth
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
- El dataset es de corte transversal y relativamente chico (1.628 filas tras limpieza); la
  validación multi-semilla existe para cuantificar esta sensibilidad al split, pero no se
  ejecuta por defecto en cada corrida por su costo computacional.
- Las variables derivadas adicionales (`crear_variables_derivadas`) están implementadas pero
  deshabilitadas: no hay evidencia confirmada en el código de que aporten mejora real sobre el
  ruido de partición.

**Del scraping:**
- Depende de la estructura HTML y las clases CSS del sitio, que pueden cambiar sin aviso —
  varios extractores usan regex sobre texto en español como respaldo, pero no eliminan el
  riesgo por completo.
- Riesgo de bloqueo/CAPTCHA, mitigado pero no eliminado por delays variables, reintentos con
  backoff y ejecución en tandas pequeñas vía cron; existe una ruta de respaldo con Playwright
  (incluido un modo manual de resolución de CAPTCHA) por si la ruta principal con `requests`
  empezara a bloquearse.
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

### 12.1. Parte 1 — Selección de algoritmo y entrenamiento

**`entrenamiento/seleccionar_algoritmo.py`** — no reentrena nada: lee los JSON de métricas de
test más recientes de `04_modelamiento/01_xgboost.py` y `02_lightgbm.py`, valida que sean
comparables (mismas features, misma seed, mismo tamaño de test — si no, lanza error en vez de
comparar corridas desalineadas), y elige un ganador según un criterio configurable (`ponderado`
por defecto: 50% MAE test + 50% RMSE test, cada métrica normalizada por su propio promedio
entre los dos algoritmos antes de ponderar). También reporta una advertencia si el ganador
global no es el mejor en el quintil más caro (Q5, ver sección 7). La decisión queda persistida
en `algoritmo_seleccionado.json`.

**`entrenamiento/01_entrenar_modelo_produccion.py`** — carga como módulo (vía `importlib`, ya
que su nombre empieza con dígito) el script de investigación del algoritmo **ganador**
(`04_modelamiento/01_xgboost.py` o `02_lightgbm.py`, según `algoritmo_seleccionado.json`) y
reutiliza sus funciones (optimización de hiperparámetros, bagging, evaluación, SHAP); solo se
entrena el algoritmo elegido, no ambos. Los dos scripts de investigación exponen la misma API
(mismos nombres de función/constante), así que este script es agnóstico a cuál de los dos se
cargó.
- **Split 85/15** (train/test) en vez de 70/15/15: el ensamble de bagging igual necesita un set
  para early stopping, así que se separa internamente un 10% del 85% de train solo para eso
  (~76.5% train_fit / ~8.5% early-stopping / 15% test) — un detalle interno de entrenamiento,
  no una tercera partición pública.
- **Versionado**: cada corrida genera un identificador
  `v{contador:04d}_{timestamp}_{algoritmo}_{hash8}` (contador incremental + algoritmo + hash
  sha256 de algoritmo+hiperparámetros ganadores), registrado en `version_modelo.json` (contador
  + historial). El algoritmo queda explícito en el propio identificador de versión, para poder
  distinguir de un vistazo qué algoritmo generó cada predicción histórica. El modelo y sus
  parámetros se **archivan por versión** en `versiones/{version}/` (no se sobrescriben) — así
  se puede recuperar el modelo exacto usado en cualquier predicción pasada.
- **Calibración de oportunidad/confianza**: además de las métricas de evaluación estándar,
  calcula y persiste (sobre el set de test) los bordes de deciles de precio, la mediana/MAD del
  error por decil, y los terciles del coeficiente de variación del ensamble — todo guardado en
  `parametros_produccion.json` bajo `calibracion_oportunidad`, para poder etiquetar avisos
  nuevos en la Parte 2 sin recalcular una distribución con una sola fila (imposible con qcut).
- Dataset de entrada: por ahora, el mismo CSV curado que usa el modelo de investigación
  (`datos_ingenieria_variables.csv` + `selected_features.csv`).
- Versión vigente a la fecha: `v0005_20260712085625_lightgbm_03cb22af`.

### 12.2. Parte 2 — Pipeline incremental

**Esquema de `produccion_gran_concepcion.db`** (definido en `db.py`, normalizado sin llevarlo
al extremo):

| Tabla | Contenido |
|---|---|
| `avisos` | Nivel grilla + `estado_publicacion` (activo/pausado/finalizado/no_disponible) + `fecha_ultimo_chequeo_estado` + `intentos_fallidos_detalle` |
| `avisos_detalle` | Nivel detalle (1:1 con `avisos`) + columnas de vulnerabilidad IGVUST resueltas directo (sin las tablas separadas `vulnerabilidad_uv`/`avisos_igvust` de la base original) |
| `poligonos_vulnerabilidad_uv` | Polígonos de Unidad Vecinal (IGVUST) de las 10 comunas analizadas, precalculados una vez desde el shapefile y guardados como WKT (EPSG:4326) — ver etapa 3 |
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
2. **`02_scraper_detalle_incremental.py`** — visita avisos nuevos sin detalle (vía
   `sd.obtener_detalle_aviso`, ruta `requests` de `01_obtener_datos/02_scraper_detalle.py`, sin
   navegador), y además **re-chequea** avisos `activo` con más de `DIAS_MIN_ENTRE_RECHEQUEOS = 7`
   días desde su último chequeo, o que nunca se chequearon (`fecha_ultimo_chequeo_estado IS
   NULL` — ej. avisos recién migrados del histórico, ver sección 13), en batches de
   `MAX_AVISOS_RECHEQUEO_POR_CORRIDA = 200`. Los nunca-chequeados van primero (son los más
   urgentes: nunca se confirmó que sigan activos) y después los vencidos por antigüedad de más
   antiguo a más reciente — SQLite ordena `NULL` antes que cualquier fecha en `ORDER BY ... ASC`,
   así que un único `ORDER BY fecha_ultimo_chequeo_estado ASC` alcanza para ambos criterios sin
   necesitar un `CASE` aparte. Extrae `estado_publicacion` del mismo JSON embebido que ya se usa para los
   puntos de interés (busca el componente `item_status_message`/`item_status_short_description_message`
   dentro de `components.head`/`components.short_description`; si no aparece, el aviso está
   activo). El guardado usa **UPSERT** (no `INSERT OR REPLACE`) para que un re-chequeo nunca
   borre las columnas de vulnerabilidad que llena la etapa siguiente. **Fallos persistentes**:
   cada fallo de scraping (agotados los reintentos de esa corrida) suma 1 a
   `avisos.intentos_fallidos_detalle`; al superar `MAX_INTENTOS_FALLIDOS_DETALLE = 5` fallos
   consecutivos, el aviso se marca `estado_publicacion = 'no_disponible'` y sale de la cola de
   pendientes (un éxito antes de llegar al umbral resetea el contador a 0).
3. **`03_vulnerabilidad_produccion.py`** — cruce punto-en-polígono contra los polígonos IGVUST ya
   precalculados en la tabla `poligonos_vulnerabilidad_uv` (WKT, EPSG:4326), resuelto con
   `shapely` puro (sin `geopandas`/GDAL) directo a columnas de `avisos_detalle` (`uv_rsh`,
   `rank_nac`, `pob_rsh_uv`, `p_urbano`, `c_ig_com`) — no a tablas de referencia separadas. Solo
   procesa avisos con coordenadas y `uv_rsh` todavía `NULL` (incremental). Esta etapa **ya no lee
   el shapefile en cada corrida** (no está versionado en el repo, así que en GitHub Actions no
   existiría): la tabla se llena una única vez —o cuando el shapefile se actualiza— corriendo a
   mano `migrar_poligonos_vulnerabilidad.py` localmente, con geopandas instalado.
4. **`04_ingenieria_variables_produccion.py`** — calcula las features seleccionadas del modelo
   para avisos nuevos (32 actualmente, leídas dinámicamente desde `selected_features.csv`; ver
   sección 13), pero **sin recalcular nada en modo batch** (a diferencia del pipeline de
   investigación): compara cada aviso contra una **población de referencia fija**
   (`datos_ingenieria_variables.csv` + coordenadas/comuna recuperadas con un `SELECT` de solo
   lectura contra la base original) vía `BallTree` para `precio_m2_sector_departamento` y la
   imputación de `antiguedad_anos`, y reutiliza los modelos de imputación de superficie ya
   entrenados (`aplicar_modelo_guardado`) sin reentrenar nada.
5. **`05_prediccion.py`** — predice con el ensamble vigente, convierte el precio real del aviso
   a CLP (reutiliza `convertir_precios_uf_a_clp`: UF con el valor del día vía mindicador.cl,
   US$ con una tasa fija), aplica la calibración
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
- **Código de salida distinto de cero si `corridas.resultado == 'error'`** (`sys.exit(1)` en
  `main()`): para que un runner externo (ej. GitHub Actions) marque la corrida como fallida y
  avise, en vez de un check verde silencioso mientras el detalle del error solo queda en
  `corridas`/`logs_ejecucion`.

Pensado para cron, ej.:
```bash
0 */4 * * * cd /ruta/al/proyecto/05_modelo_produccion && /ruta/al/python 00_orquestador.py
```

**Despliegue vigente vía GitHub Actions** (`.github/workflows/orquestador.yml`): corre cada 6
horas (4 corridas/día, `cron: '0 */6 * * *'`; antes cada 12h/2 corridas al día) más
`workflow_dispatch` para disparo manual desde la pestaña Actions. El paso de commit/push de
`produccion_gran_concepcion.db` corre con `if: always()`, así que el diagnóstico (tabla
`corridas`/`logs_ejecucion`) se persiste en el repo incluso si el orquestador terminó en
`sys.exit(1)` — solo el job queda en rojo, el commit de la BD no se salta.

### 12.4. Notas y limitaciones conocidas de esta sección

- La población de referencia de la etapa de variables (comparables de precio/m² e imputación de
  antigüedad) usa **solo** el dataset histórico de investigación, no los avisos de producción ya
  procesados — más simple y estable, a costa de no reflejar aún datos más recientes del mercado.
- Los bordes `-inf`/`inf` en `calibracion_oportunidad` (para clasificar avisos fuera del rango
  de precio/CV visto en test) se serializan como `-Infinity`/`Infinity`, válido para Python pero
  no JSON estándar — una herramienta no-Python que lea ese archivo directo fallaría en esos dos
  campos.
- El pipeline de producción solo predice `tipo_propiedad='departamento'` (igual que el modelo);
  las casas se scrapean pero nunca se puntúan.

---

## 13. Historial de correcciones y mejoras (sesión de mantenimiento, jul-2026)

Sesión enfocada en tres frentes: poblar la base de producción con el histórico de
investigación, corregir un bug de escala en `gastos_comunes` (extracción → base → modelo →
producción), y ajustar el ritmo del re-chequeo. Quedan documentados acá los cambios que no
encajan en las secciones anteriores (que describen el estado ya corregido del código) por ser
más eventos puntuales que arquitectura permanente.

### 13.1. Mejora: producción solo scrapea departamentos

Ver nota en la sección 1. `01_scraper_grilla_incremental.py` ahora usa
`TIPOS_PROPIEDAD_PRODUCCION = ["departamento"]` como default (configurable, no hardcodeado) en
vez de `sg.TIPOS_PROPIEDAD` (`["casa", "departamento"]`) — evita gastar presupuesto de scraping
en avisos que el resto del pipeline de producción descartaría de todas formas.
`02_scraper_detalle_incremental.py` no necesitó cambios (no filtra por tipo, trabaja directo
sobre lo que ya está en `avisos`).

### 13.2. Mejora: migración del histórico de investigación a producción

Nuevo script de una sola vez, `05_modelo_produccion/migracion_historico_a_produccion.py` (no
forma parte del orquestador). Migró **1628 avisos** (`departamento`/`arriendo`) desde
`avisos_gran_concepcion.db` — exactamente los IDs de `datos_ingenieria_variables.csv` (el
dataset que entrenó el modelo), no los ~1956 avisos crudos completos, ya que las ~298 casas del
histórico nunca generarían features ni predicción en producción. Quedan con
`estado_publicacion='activo'` y `fecha_ultimo_chequeo_estado=NULL` (califican de inmediato para
el primer re-chequeo real). Colisiones: se conserva siempre la fila ya existente en producción,
nunca se sobreescribe con la versión del histórico. Backup previo
(`produccion_gran_concepcion.db.bak-pre-migracion-historico`), idempotente.

**Incidente durante la migración**: la primera corrida reportó 1628 migrados pero la base
terminó con solo 1606 avisos nuevos — 25 filas se perdieron de forma silenciosa (sin excepción,
`PRAGMA integrity_check` seguía en `ok`). Causa más probable: el repo vive dentro de una carpeta
sincronizada por Google Drive, y la migración hizo ~1628 commits SQLite individuales a lo largo
de ~3 minutos — ventana larga donde el sincronizador pudo interferir con el archivo. Al ser el
script idempotente, una segunda corrida completó exactamente las 25 filas faltantes sin
duplicar nada. **Recomendación para scripts futuros** que hagan muchas escrituras seguidas a
estas bases: verificar conteos después de corridas largas, dado que el repo vive en una carpeta
sincronizada.

### 13.3. Bug corregido: escala de `gastos_comunes` (ver también sección 3.2)

Backfill de una sola vez sobre `avisos_gran_concepcion.db` (backup
`.bak-pre-backfill-gastos-comunes`), reclasificando las 1956 filas de `avisos_detalle` según el
texto crudo:

| Categoría | Filas | Acción |
|---|---:|---|
| Bug de escala corregido (`"N.NNN"` → valor real) | 1411 | `×1000` efectivo |
| Placeholder "incluido" → `0` | 45 | incluye 3 casos fuera del rango 1-14 que la inspección manual no había detectado (`"90"`, `"60"`, `"10"`) |
| Ya era `0` explícito | 392 | sin cambios |
| NULL sin resolver (ausencia genuina) | 88 | se dejan `NULL`, no se fuerzan a `0` |
| No parseable (basura de extracción: `","`, `"."`) | 18 | sin cambios, fuera de alcance |
| Outlier sin punto ($35.000, `MLC-2055388319`) | 1 | sin cambios, señalado |
| Outlier implausible (\$1.111.111, `MLC-4068110790`) | 1 | descartado a `NULL` (supera el techo de sanidad de \$500.000) |

De los 190 `NULL` originales, **102 (54%) se resolvieron** vía el fallback de extracción (ver
sección 3.2) verificado en vivo contra el sitio real.

`produccion_gran_concepcion.db` se corrigió por separado con el mismo criterio: los 1628 avisos
migrados ya venían limpios (heredan el texto ya corregido del histórico), y los 3 avisos
preexistentes (scrapeados antes del fix del scraper) se corrigieron directo, respaldados en una
verificación en vivo contra el HTML real.

### 13.4. Reentrenamiento con datos corregidos

Dataset regenerado desde cero (`01_ingenieria_variables.py` sobre la base ya corregida, no un
parche del CSV): **1628 filas × 44 columnas** (igual tamaño que antes — la corrección de escala
no elimina filas, solo corrige valores). Selección de variables re-corrida:
**32 features finales** (antes 30) — nuevas: `c_ig_com`, `cantidad_clinicas`,
`cantidad_centros_comerciales`; sacada: `cantidad_universidades`. `gastos_comunes` se mantuvo
entre las más estables (`stability_score=0.97`, top 6).

Métricas de test, antes/después de la corrección:

| Métrica | XGBoost antes | XGBoost después | LightGBM antes | LightGBM después |
|---|---:|---:|---:|---:|
| MAE | 50.981 | 49.237 | 50.058 | 48.901 |
| RMSE | 77.646 | 74.455 | 74.359 | 73.890 |
| R² | 0.8250 | 0.8391 | 0.8395 | 0.8416 |
| MAPE | 8.94% | 8.65% | 8.74% | 8.57% |
| MdAPE | 6.85% | 6.50% | 6.67% | 6.16% |

Ambos algoritmos mejoraron en las cinco métricas de test. `seleccionar_algoritmo.py` (criterio
ponderado 50% MAE + 50% RMSE test) sigue eligiendo **LightGBM** como antes, pero el margen se
achicó de 3.03% a 0.72% relativo — XGBoost cerró bastante la brecha con datos corregidos. Las
secciones 5-7 ya reflejan este estado corregido (32 features, LightGBM vigente en producción);
esta subsección documenta el delta puntual frente a la corrida anterior, no una comparación
aparte.

### 13.5. Producción: pipeline extendido para 3 features nuevas + reentrenamiento

El modelo ganador (LightGBM, 32 features) usa 3 features que el pipeline de producción no
calculaba. Se extendió:
- **`db.py`**: nueva columna `avisos_detalle.c_ig_com` (migración `ALTER TABLE` para bases ya
  existentes, mismo patrón que `intentos_fallidos_detalle`).
- **`03_vulnerabilidad_produccion.py`**: ahora también resuelve `c_ig_com` desde el shapefile
  IGVUST (antes solo `uv_rsh`/`rank_nac`/`pob_rsh_uv`/`p_urbano`).
- **`04_ingenieria_variables_produccion.py`**: agregadas `cantidad_centros_comerciales` y
  `cantidad_clinicas` (mismo patrón que los demás conteos de POI) y `c_ig_com` (mismo fallback
  por comuna que ya usan `rank_nac`/`p_urbano`/`pob_rsh_uv`).

Modelo de producción reentrenado sobre datos corregidos: nueva versión
`v0005_..._lightgbm_...` (algoritmo ganador de la sección 13.4). Las 3 predicciones ya
existentes en `predicciones` (hechas con el modelo y el `gastos_comunes` sin corregir) se
recalcularon con la versión nueva **sin pisar las anteriores** — `UNIQUE(id_aviso,
version_modelo)` permite que ambas convivan, tal como está pensado el esquema.

---

## 14. Visualización (`06_visualizacion/`)

Dashboard Streamlit de solo lectura sobre `produccion_gran_concepcion.db`: **nunca escribe**
en las tablas de negocio (`avisos`, `avisos_detalle`, `predicciones`) — la única escritura que
hace es indirecta, vía la caché compartida `valores_uf` (ver 14.2), la misma tabla que ya usa
`05_prediccion.py`. Hecho **a modo de prueba y aprendizaje**, no es un servicio de tasación.

### 14.1. Estructura de archivos

| Archivo | Responsabilidad |
|---|---|
| `app.py` | Entrypoint: `st.set_page_config`, `st.tabs(["🏠 Buscador", "ℹ️ Cómo funciona"])`, orden de resultados |
| `data.py` | Query SQL (join + predicción más reciente) + estandarización de precio, cacheada con `st.cache_data` |
| `filters.py` | Sidebar de filtros (precio, dormitorios, baños, comuna, superficie, etiqueta, confianza, amenities, estado) + lógica de filtrado |
| `components.py` | Tarjeta de aviso (HTML) y mapa (`folium` + `streamlit-folium`) |
| `explicacion.py` | Contenido de la pestaña "Cómo funciona" (metodología, error del modelo, deciles, advertencias) |
| `styles.py` | Paleta de colores + CSS inyectado (tarjetas, badges, tema) |
| `.streamlit/config.toml` | Tema forzado a claro (ver 14.4) |

Las rutas a `produccion_gran_concepcion.db` y a `03_ingenieria_variables/01_ingenieria_variables.py`
se resuelven con `Path(__file__).resolve().parent.parent` en `data.py`, no hardcodeadas, para
que `streamlit run app.py` funcione sin importar desde qué directorio se invoque.

### 14.2. De dónde salen los datos que se muestran

`data.py` hace `INNER JOIN` entre `avisos`, `avisos_detalle` y `predicciones` — un aviso sin
predicción simplemente no aparece, porque la etiqueta/confianza/z_robusto son el centro del
diseño (no tiene sentido mostrar una tarjeta sin ellas). En la práctica esto significa que el
dashboard solo muestra una fracción de `avisos` en un momento dado: el resto son avisos que el
orquestador todavía no llegó a puntuar.

**Predicción "más reciente" por aviso**: en vez de filtrar por el `version_actual` fijo de
`entrenamiento/version_modelo.json`, usa `ROW_NUMBER() OVER (PARTITION BY id_aviso ORDER BY
fecha_prediccion DESC)` — la predicción con fecha más nueva de cada aviso, sin importar su
versión. Es más robusto ante un rollout parcial: si un aviso fue puntuado con la versión
anterior y todavía no se ha vuelto a puntuar con la vigente, igual se muestra (con su
predicción más nueva disponible) en vez de desaparecer del dashboard.

**Estandarización de precio**: reutiliza `convertir_precios_uf_a_clp` y
`PRECIO_MAXIMO_ARRIENDO_CLP`/`filtrar_precio_maximo` de
`03_ingenieria_variables/01_ingenieria_variables.py` vía el mismo truco de import dinámico que
usa `05_prediccion.py` (el directorio empieza con dígitos, no es importable como paquete
normal) — así el precio que se muestra en cada tarjeta es *literalmente* el mismo
`precio_clp` que vio el modelo al calcular `z_robusto`, no una reimplementación aparte que se
pueda desincronizar. Esto de paso significó extender esa función compartida (beneficiando
también a entrenamiento y producción, no solo al dashboard):
- **Conversión US$ → CLP con tasa fija** (`TASA_USD_CLP = 930`): antes la función solo
  convertía UF, y trataba cualquier otra moneda como si ya fuera CLP. Con el único aviso real
  en US$ de la base (`precio=350000`), eso daba `$325.500.000/mes` — implausible. Se optó por
  agregar la tasa fija (no vale la pena ir a buscar el valor histórico por fecha a una API para
  un puñado de avisos) **y además** aplicar el mismo tope de precio máximo que ya usaba
  entrenamiento (`PRECIO_MAXIMO_ARRIENDO_CLP = 8.000.000`, antes un default hardcodeado dentro
  de `filtrar_precio_maximo`, ahora una constante nombrada) también en `05_prediccion.py`
  (avisos sobre el tope simplemente no generan predicción) y en `data.py` (defensa adicional en
  la vista).
- La tabla de caché `valores_uf` (fecha → valor UF, consultado a `mindicador.cl`) no existía
  todavía en `produccion_gran_concepcion.db` cuando se construyó el dashboard — la creó y
  pobló la primera corrida real, mismo mecanismo que ya usaba `05_prediccion.py`.

**`gastos_comunes`**: se aplica el mismo criterio de imputación que usa el pipeline al armar
el feature del modelo (`fillna(0)`, ver secciones 3.3/12.2) — un aviso sin gastos comunes
informados se muestra como `$0`, no como `nan`. La columna cruda de `avisos_detalle` sí puede
tener `NULL` genuino (no se scrapeó/no se informó); mostrarlo tal cual sin imputar producía
literalmente el string `"nan"` en la tarjeta.

**Conexión SQLite con `timeout=30`** (en vez del default de 5s) en `data.py` y en la función
compartida de conversión de moneda: el orquestador puede estar escribiendo en la misma base en
paralelo (una fila cada ~0.5s durante la etapa de predicción), y con el timeout default el
dashboard podía toparse con `sqlite3.OperationalError: database is locked` si alguien lo abría
justo durante una corrida.

Todo el resultado se cachea con `st.cache_data(ttl=600)` — 10 minutos, ya que la base la
actualiza el orquestador en segundo plano (vía cron), no en cada request del usuario.

`fecha_publicacion_aprox` (aproximada, calculada por el scraper a partir de texto relativo tipo
"hace 3 meses") se conserva en el DataFrame que llega a la UI — antes se descartaba en
`load_data()` junto con `precio_clp`/`moneda`, ya que hasta ahora nada la consumía; ahora la
usan tanto el orden "Más recientes primero" como la fecha que se muestra en cada tarjeta (ver
14.3).

### 14.3. Filtros y diseño visual

Sidebar con precio/superficie (sliders de rango), dormitorios/baños (checkboxes con conteo,
agrupando "4+"), comuna (multiselect), etiqueta del modelo y nivel de confianza (checkboxes,
todo marcado por defecto), amenities en una sección colapsable, y un filtro de
`estado_publicacion` (activo por defecto, con opción de incluir pausados). Botón "Limpiar
filtros" vía `st.session_state` (un callback que resetea las keys a sus valores por defecto
antes del siguiente render).

**Orden de resultados** (`app.py`, selectbox sobre `filtered_df`): "Más relevantes" (sin
reordenar), "Mejor oportunidad" (`z_robusto` ascendente), "Precio: menor a mayor/mayor a menor",
y **"Más recientes primero"** (`fecha_publicacion_aprox` descendente,
`na_position="last"` — alrededor de 1 de cada 5 avisos activos no trae esa fecha porque la
página de detalle no siempre la informa, y esos avisos van al final en vez de asumirles una
fecha). Cada tarjeta (`components.py`) muestra ahora también "Publicado: hace N días/meses/años"
(o "fecha no disponible" si es nula) debajo del precio, con el mismo criterio de redondeo que
`_format_frescura` usa para "Datos verificados: hace N días".

El contador de resultados ("N departamentos encontrados") pasó de un `<p>` de texto plano a una
tarjeta propia (`.result-count-card`, texto en negro y fondo alineado con el selectbox de
orden) para que ambos elementos se vean como un bloque visual único a la altura del selector de
orden, en vez de texto suelto flotando junto a un dropdown con su propio fondo.

**Tema forzado a claro** (`.streamlit/config.toml`, `theme.base = "light"`): el diseño usa una
paleta fija (fondo gris claro, tarjetas blancas, semáforo verde/gris/rojo para las etiquetas) —
sin forzar el tema, Streamlit adapta colores al modo oscuro del navegador/SO, lo que dejaba el
título del header en blanco sobre fondo claro (ilegible) y las tarjetas sin fondo/borde
visible. Se combinó con `!important` en los colores de texto de `styles.py` como defensa
adicional, ya que Streamlit reinyecta su propio color de tema con selectores más específicos
que los del CSS propio.

Las tarjetas son un único `st.markdown(unsafe_allow_html=True)` por aviso, envuelto en un
`<div class="app-card">` propio (fondo blanco, sombra, radio, margen) — no
`st.container(border=True)`: su borde por defecto es casi invisible (`rgba(…, 0.2)`) y no trae
fondo propio, así que no se distinguía como "tarjeta" contra el fondo gris. El mismo tratamiento
visual (`app-card`) se le dio también al bloque completo de filtros del sidebar, para que use
el mismo lenguaje visual que las tarjetas de aviso.

### 14.4. Pestañas en vez de multipágina

La pestaña "Cómo funciona" (metodología, error del modelo, deciles, advertencias de uso — con
cifras reales tomadas de `parametros_produccion.json`/`lightgbm_regression_precio_metrics.json`
del modelo vigente) se implementó primero como una página aparte del sistema clásico de
multipágina de Streamlit (`pages/`), pero **`st.page_link()` con ese mecanismo tiene un bug
real** en la versión instalada (1.58): itera sobre todas las páginas registradas leyendo
`page_data["url_pathname"]` sin `.get()`, y esa clave es opcional (`NotRequired`) para páginas
descubiertas vía la carpeta `pages/` clásica — solo la completa la API moderna
`st.navigation`/`st.Page`. Resultado: `KeyError: 'url_pathname'` al hacer clic en cualquier
link generado así.

Se probó migrar a `st.navigation`/`st.Page` (que sí completa esa clave), pero introdujo otro
problema: con `st.navigation`, Streamlit ejecuta un router aparte y la pestaña quedaba en
blanco/rota en la práctica. Se optó, en cambio, por la solución más simple y sin las
dependencias frágiles de routing entre páginas: **una sola página con `st.tabs`** — ambas
vistas (`app.py` y `explicacion.py`) se renderizan en el mismo script run, sin URLs ni
navegación de por medio.

### 14.5. Cómo correrlo

```bash
pip install -r 06_visualizacion/requirements.txt
cd 06_visualizacion
streamlit run app.py
```

`requirements.txt` incluye, además de `streamlit`/`folium`/`streamlit-folium`/`pandas`,
`numpy`/`requests`/`joblib`/`scikit-learn`: son transitivas de `03_ingenieria_variables/01_ingenieria_variables.py`,
que `data.py` importa dinámicamente (ver 14.2) — sin ellas el deploy falla con
`ModuleNotFoundError` al cargar ese módulo (Streamlit Cloud solo instala lo declarado en este
`requirements.txt`, no las dependencias de las demás etapas del pipeline).

Requiere que `05_modelo_produccion/produccion_gran_concepcion.db` ya exista con al menos un
aviso en `predicciones` (ver sección 12) — si la base está vacía o el orquestador todavía no
corrió la etapa de predicción, el dashboard muestra un aviso de "sin datos" en vez de una
tabla vacía silenciosa.

### 14.6. Limitaciones conocidas

- Solo muestra avisos ya puntuados por el orquestador — no es un catastro completo de
  `avisos`, y el conteo de resultados fluctúa corrida a corrida según cuánto haya avanzado la
  etapa de predicción.
- La conversión US$ → CLP usa una tasa fija (`930`), no un valor de mercado por fecha — pensada
  para el puñado de avisos en esa moneda visto hasta ahora, no para volúmenes grandes.
- No hay autenticación ni control de acceso: pensado para uso local/personal, no para
  exponerse como servicio público sin revisión adicional.
