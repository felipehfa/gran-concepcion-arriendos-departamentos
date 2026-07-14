"""Contenido de la pestaña 'Cómo funciona': metodología, precisión y advertencias."""

import streamlit as st

from styles import COLOR_TEXT_TITLE

# Las 32 features que usa el modelo vigente (ver 03_ingenieria_variables/save/seleccion_variables/
# selected_features.csv y sección 4 del README), agrupadas igual que ahí, con una explicación en
# lenguaje simple de qué mide cada una. Contenido estático (no se lee en vivo), igual que el resto
# de las cifras de este archivo.
_FEATURE_GROUPS = [
    (
        "Características físicas del departamento",
        [
            ("Superficie útil", "El espacio interior habitable del departamento, en m²."),
            ("Superficie total", "Superficie útil más terrazas, bodega y otras áreas no habitables, en m²."),
            ("Relación superficie total/útil", "Qué tanta superficie \"extra\" (no habitable) tiene el departamento respecto a la habitable."),
            ("Número de baños", "Cantidad de baños del departamento."),
            ("Número de estacionamientos", "Cantidad de estacionamientos incluidos con el departamento."),
            ("Estacionamientos de visita", "Si el edificio cuenta con estacionamientos para visitas."),
            ("Bodega", "Si el departamento incluye bodega."),
            ("Piso", "En qué piso del edificio está ubicado."),
            ("Ascensor", "Si el edificio tiene ascensor."),
            ("Piscina", "Si el edificio o condominio tiene piscina."),
            ("Amoblado", "Si el departamento se arrienda amoblado."),
            ("Conserjería", "Si el edificio cuenta con conserjería."),
            ("Condominio cerrado", "Si el edificio pertenece a un condominio con acceso controlado."),
            ("Antigüedad del edificio", "Años transcurridos desde la construcción del edificio."),
        ],
    ),
    (
        "Costos asociados",
        [
            ("Gastos comunes", "Monto mensual de gastos comunes informado en el aviso."),
        ],
    ),
    (
        "Ubicación y mercado local",
        [
            ("Precio/m² de comparables cercanos", "Precio promedio por m² de otros departamentos similares dentro de 300 metros — nunca del propio aviso, para no sesgar la comparación."),
            ("Confiabilidad de ese precio comparable", "Si hubo suficientes departamentos cercanos para calcular el precio/m² anterior de forma confiable."),
            ("Nivel de precio del barrio", "Qué tan caro es el barrio en relación al resto de la ciudad, en 5 niveles."),
            ("Distancia al centro de Concepción", "Distancia en línea recta al centro de la ciudad."),
            ("Distancia al centro de la comuna", "Distancia en línea recta al centro de la propia comuna del departamento."),
            ("Paraderos cercanos", "Cantidad de paraderos de locomoción colectiva a menos de 500 metros."),
            ("Colegios cercanos", "Cantidad de colegios a menos de 500 metros."),
            ("Supermercados cercanos", "Cantidad de supermercados a menos de 500 metros."),
            ("Jardines infantiles cercanos", "Cantidad de jardines infantiles a menos de 500 metros."),
            ("Centros comerciales cercanos", "Cantidad de centros comerciales/malls a menos de 500 metros."),
            ("Plazas cercanas", "Cantidad de plazas o áreas verdes a menos de 500 metros."),
            ("Farmacias cercanas", "Cantidad de farmacias a menos de 500 metros."),
            ("Clínicas u hospitales cercanos", "Cantidad de centros de salud a menos de 500 metros."),
        ],
    ),
    (
        "Contexto socioeconómico del sector",
        [
            ("Vulnerabilidad del sector (ranking nacional)", "Ranking nacional de vulnerabilidad social de la Unidad Vecinal, según el índice IGVUST del Registro Social de Hogares."),
            ("Vulnerabilidad de la comuna", "El mismo índice de vulnerabilidad IGVUST, pero calculado a nivel de toda la comuna."),
            ("Población en el Registro Social de Hogares", "Cantidad de personas registradas en el RSH dentro de la Unidad Vecinal del departamento."),
            ("Porcentaje urbano del sector", "Qué tan urbanizada (vs. rural) es la Unidad Vecinal donde está el departamento."),
        ],
    ),
]


def _render_grupos_features() -> str:
    bloques = []
    for titulo, features in _FEATURE_GROUPS:
        filas = "".join(
            f"<tr><td style='padding:4px 10px 4px 0;font-weight:600;white-space:nowrap;vertical-align:top;'>{nombre}</td>"
            f"<td style='padding:4px 0;color:#4A4A4A;'>{descripcion}</td></tr>"
            for nombre, descripcion in features
        )
        bloques.append(
            f"""
<p style="color:{COLOR_TEXT_TITLE};font-weight:600;margin:14px 0 6px 0;">{titulo} ({len(features)})</p>
<table style="width:100%;border-collapse:collapse;font-size:0.85rem;">{filas}</table>"""
        )
    return "".join(bloques)


def render() -> None:
    st.markdown(
        """
<div class="app-card">

### Qué es esto

Este buscador usa un **modelo predictivo** (no una tasación ni un servicio profesional)
entrenado con avisos de arriendo de departamentos del Gran Concepción, recolectados por un
scraper propio desde Portal Inmobiliario. El modelo aprende, a partir de las características
de cada aviso (superficie, dormitorios, baños, comuna, cercanía a servicios, antigüedad,
amenities, etc.), a estimar cuál **debería** ser el precio de arriendo — y luego compara ese
precio estimado contra el precio real publicado para detectar si un aviso está por debajo,
en línea, o por sobre lo esperado.

</div>

<div class="app-card">

### Cantidad de datos usados

El modelo vigente (`v0005_20260712085625_lightgbm_03cb22af`) se entrenó con **1.628 avisos**
de departamentos en arriendo, divididos en:

- **1.383 avisos** para entrenar el modelo (ajuste + early stopping).
- **245 avisos** que el modelo nunca vio durante el entrenamiento, usados solo para medir
  qué tan bien predice (conjunto de test).

Es un dataset acotado a una sola ciudad y a un momento puntual en el tiempo — no es un
catastro exhaustivo del mercado.

</div>

<div class="app-card">

### Qué variables usa el modelo para estimar el precio

El modelo llega a un precio combinando **32 variables** de cada aviso, agrupadas en cuatro
categorías. Ninguna variable por sí sola determina el precio — es la combinación de todas la
que produce la estimación final.
"""
        + _render_grupos_features()
        + """

</div>

<div class="app-card">

### Qué tan preciso es (el error del modelo)

Medido sobre los 245 avisos de test (que el modelo no usó para entrenar):

| Métrica | Valor | Qué significa |
|---|---|---|
| Error promedio (MAE) | **≈ \\$48.700** | En promedio, la predicción se equivoca por unos \\$48.700 respecto al precio real |
| Error porcentual (MAPE) | **≈ 8,6%** | En promedio, la predicción se desvía ~8,6% del precio real |
| R² | **≈ 0,84** | El modelo explica ~84% de la variación de precios entre avisos — queda ~16% sin explicar |

El error **no es parejo en todo el rango de precios**: en departamentos más caros (sobre
~\\$670.000) el error típico sube a más de \\$85.000, porque hay menos avisos en ese rango para
aprender de ellos y los precios son más variables. Por eso la calificación de "oportunidad"
no compara contra un error único, sino contra el error típico de avisos de precio similar
(ver estratos más abajo).

**¿Es esto bueno?** Como referencia, se compara contra dos formas "ingenuas" de estimar un
precio sin ningún modelo:

| Cómo se estima el precio | Error promedio (MAE) |
|---|---|
| Usar siempre el precio promedio del mercado, sin distinguir entre avisos | ≈ \\$125.900 |
| Precio/m² del sector × superficie del departamento, sin ajustar nada más | ≈ \\$79.200 |
| **Este modelo** | **≈ \\$48.700** |

El modelo reduce el error a menos de la mitad frente a usar solo el precio/m² del sector, y a
poco más de un tercio frente a asumir un precio promedio único para todo el mercado.

</div>

<div class="app-card">

### Los estratos (deciles de precio)

Para calibrar cuándo un precio es "raro", los 245 avisos de test se agruparon en **10
estratos según su precio** (deciles), y para cada estrato se calculó su propio error típico
(mediana del error y MAD — desviación absoluta mediana, una versión del error típico menos
sensible a valores extremos que la desviación estándar). Los bordes de precio de cada
estrato en el modelo vigente son:

`< $390.000` · `$390.000–430.000` · `$430.000–480.000` · `$480.000–500.000` ·
`$500.000–530.000` · `$530.000–567.000` · `$567.000–600.000` · `$600.000–650.000` ·
`$650.000–750.000` · `> $750.000`

Cada aviso se compara solo contra el error típico de su propio estrato, no contra un
promedio general del mercado.

</div>

<div class="app-card">

### Cuándo es "Oportunidad", "Precio de mercado" o "Caro"

Para cada aviso se calcula un **z-score robusto**: qué tan lejos está su error (precio real
− precio predicho) de la mediana de error de su estrato, medido en unidades de MAD de ese
mismo estrato. Con eso:

- **Oportunidad**: el precio real está bastante más bajo de lo esperado para avisos
  similares (z-score robusto menor a −1).
- **Caro**: el precio real está bastante más alto de lo esperado (z-score robusto mayor a 1).
- **Precio de mercado**: el precio real está dentro de lo esperado para avisos similares.

La app solo muestra la etiqueta en palabras — a propósito no se expone el número del
z-score ni el decil, para no dar una falsa sensación de precisión numérica.

</div>

<div class="app-card">

### Qué significa el nivel de confianza

El modelo no es un solo modelo: es un **ensamble de 10 modelos** entrenados con distintas
semillas aleatorias sobre los mismos datos. El nivel de confianza mide **qué tan de acuerdo
están esos 10 modelos entre sí** para un aviso en particular (su coeficiente de variación):

- **Alta**: los 10 modelos predicen precios muy parecidos entre sí.
- **Media**: hay algo de desacuerdo entre los 10 modelos.
- **Baja**: los 10 modelos predicen precios bastante distintos entre sí — es una señal de
  que ese aviso tiene una combinación de características poco común en los datos de
  entrenamiento, así que la predicción es menos confiable.

Confianza alta **no garantiza que la predicción sea correcta** — solo dice que el modelo es
consistente consigo mismo para ese aviso.

</div>

<div class="app-card">

### ⚠️ Usar con precaución

Este buscador se hizo **solo a modo de prueba y aprendizaje**, no es un servicio de tasación
ni una recomendación de inversión. Antes de tomar cualquier decisión con esta información,
ten en cuenta:

- El modelo **no captura ni considera todas las variables** que afectan el precio real de un
  arriendo: estado de conservación interno, luminosidad, ruido, calidad de terminaciones,
  vista, estado del edificio/administración, condiciones del contrato, urgencia del
  arrendador, entre otras cosas que no están en los datos scrapeados.
- Los datos vienen de avisos publicados — pueden tener errores de tipeo, precios
  desactualizados, o información incompleta que el scraper no pudo capturar bien.
- El error típico del modelo (~8,6%, más alto en tramos de precio poco frecuentes) significa
  que **puede fallar**, incluso en avisos marcados con confianza alta.
- Revisa siempre la publicación original y otras variables no capturadas por el modelo antes
  de sacar conclusiones sobre un aviso puntual.

</div>
        """,
        unsafe_allow_html=True,
    )
