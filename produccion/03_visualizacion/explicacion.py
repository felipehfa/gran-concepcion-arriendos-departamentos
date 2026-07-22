"""Contenido de la pestaña 'Cómo funciona': metodología, precisión y advertencias."""

import base64
from pathlib import Path

import streamlit as st

from styles import COLOR_CARO, COLOR_OPORTUNIDAD, COLOR_TEXT_TITLE

# Gráfico de importancia SHAP, generado por produccion/02_pruebas/analisis_shap.py
# (script de validación manual, no forma parte del pipeline) contra el modelo de
# producción vigente, y copiado a docs/images/ para que tanto el README como esta
# app lo usen sin duplicar el archivo. Se embebe como data URI (en vez de una URL)
# para no depender de cómo Streamlit Cloud sirve archivos estáticos fuera de este
# directorio.
_SHAP_IMG_PATH = Path(__file__).resolve().parent.parent.parent / "docs" / "images" / "shap_importancia.png"


def _shap_img_data_uri() -> str:
    data = _SHAP_IMG_PATH.read_bytes()
    return "data:image/png;base64," + base64.b64encode(data).decode("ascii")


# Top 10 features por importancia SHAP (magnitud + dirección) del modelo vigente
# (v0006_20260721211302_lightgbm_5fbde493), calculadas por
# produccion/02_pruebas/analisis_shap.py sobre los 245 avisos de test — ver sección 4.1
# del README para la metodología y la tabla completa (29 features). Contenido estático
# (no se lee en vivo), igual que el resto de las cifras de este archivo: hay que
# actualizarlo a mano si se reentrena el modelo y cambia el ranking.
_SHAP_TOP10 = [
    ("Superficie útil", "18,1%", "sube"),
    ("Número de baños", "17,5%", "sube"),
    ("Superficie total", "14,5%", "sube"),
    ("Número de estacionamientos", "6,9%", "sube"),
    ("Costo total/m² de comparables cercanos", "5,5%", "sube"),
    ("Amoblado", "3,9%", "sube"),
    ("Distancia al centro de Concepción", "3,6%", "baja"),
    ("Vulnerabilidad del sector (ranking nacional)", "3,4%", "sube"),
    ("Porcentaje urbano del sector", "3,0%", "baja"),
    ("Ascensor", "2,9%", "sube"),
]

_SHAP_DIRECCION_ICONO = {"sube": ("↑", COLOR_OPORTUNIDAD), "baja": ("↓", COLOR_CARO)}


def _render_shap_tabla() -> str:
    filas = "".join(
        f"<tr>"
        f"<td style='padding:4px 10px 4px 0;'>{i}</td>"
        f"<td style='padding:4px 10px 4px 0;font-weight:600;'>{nombre}</td>"
        f"<td style='padding:4px 10px 4px 0;text-align:right;'>{pct}</td>"
        f"<td style='padding:4px 0;color:{_SHAP_DIRECCION_ICONO[direccion][1]};font-weight:700;'>"
        f"{_SHAP_DIRECCION_ICONO[direccion][0]} {'sube el costo' if direccion == 'sube' else 'baja el costo'}</td>"
        f"</tr>"
        for i, (nombre, pct, direccion) in enumerate(_SHAP_TOP10, start=1)
    )
    return f"""<table style="width:100%;border-collapse:collapse;font-size:0.85rem;">
<tr style="color:{COLOR_TEXT_TITLE};font-weight:700;">
<td style='padding:4px 10px 4px 0;'>#</td><td>Variable</td>
<td style='text-align:right;padding:4px 10px 4px 0;'>% del total</td><td>Dirección</td>
</tr>{filas}</table>"""

# Las 29 features que usa el modelo vigente (ver 03_ingenieria_variables/save/seleccion_variables/
# selected_features.csv y sección 4 del README), agrupadas igual que ahí, con una explicación en
# lenguaje simple de qué mide cada una. Contenido estático (no se lee en vivo), igual que el resto
# de las cifras de este archivo.
#
# 'gastos_comunes' YA NO es una feature de entrada: el modelo predice el costo
# total mensual (arriendo + gastos comunes, ver costo_total_clp), así que
# usar gastos_comunes como input sería fuga de datos (sería entrenar con
# parte de la respuesta ya en la pregunta). Los gastos comunes del aviso se
# siguen mostrando en la tarjeta y se siguen usando para calcular el costo
# total real contra el que se compara la predicción — solo dejaron de
# entrarle al modelo como variable predictora.
_FEATURE_GROUPS = [
    (
        "Características físicas del departamento",
        [
            ("Superficie útil", "El espacio interior habitable del departamento, en m²."),
            ("Superficie total", "Superficie útil más terrazas, bodega y otras áreas no habitables, en m²."),
            ("Relación superficie total/útil", "Qué tanta superficie \"extra\" (no habitable) tiene el departamento respecto a la habitable."),
            ("Número de baños", "Cantidad de baños del departamento."),
            ("Número de estacionamientos", "Cantidad de estacionamientos incluidos con el departamento."),
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
        "Ubicación y mercado local",
        [
            ("Costo total/m² de comparables cercanos", "Costo total mensual promedio por m² (arriendo + gastos comunes) de otros departamentos similares dentro de 300 metros — nunca del propio aviso, para no sesgar la comparación."),
            ("Confiabilidad de ese costo comparable", "Si hubo suficientes departamentos cercanos para calcular el costo/m² anterior de forma confiable."),
            ("Nivel de precio del barrio", "Qué tan caro es el barrio en relación al resto de la ciudad (según costo total mensual), en 5 niveles."),
            ("Distancia al centro de Concepción", "Distancia en línea recta al centro de la ciudad."),
            ("Distancia al centro de la comuna", "Distancia en línea recta al centro de la propia comuna del departamento."),
            ("Paraderos cercanos", "Cantidad de paraderos de locomoción colectiva a menos de 500 metros."),
            ("Colegios cercanos", "Cantidad de colegios a menos de 500 metros."),
            ("Supermercados cercanos", "Cantidad de supermercados a menos de 500 metros."),
            ("Jardines infantiles cercanos", "Cantidad de jardines infantiles a menos de 500 metros."),
            ("Universidades cercanas", "Cantidad de universidades a menos de 500 metros."),
            ("Plazas cercanas", "Cantidad de plazas o áreas verdes a menos de 500 metros."),
            ("Farmacias cercanas", "Cantidad de farmacias a menos de 500 metros."),
            ("Clínicas u hospitales cercanos", "Cantidad de centros de salud a menos de 500 metros."),
        ],
    ),
    (
        "Contexto socioeconómico del sector",
        [
            ("Vulnerabilidad del sector (ranking nacional)", "Ranking nacional de vulnerabilidad social de la Unidad Vecinal, según el índice IGVUST del Registro Social de Hogares."),
            ("Hogares en el Registro Social de Hogares", "Cantidad de hogares registrados en el RSH dentro de la Unidad Vecinal del departamento."),
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
amenities, etc.), a estimar cuál **debería** ser el costo total mensual (arriendo + gastos
comunes) — y luego compara ese costo estimado contra el costo total real (arriendo publicado
más gastos comunes informados) para detectar si un aviso está por debajo, en línea, o por
sobre lo esperado.

</div>

<div class="app-card">

### Cantidad de datos usados

El modelo vigente (`v0006_20260721211302_lightgbm_5fbde493`) se entrenó con **1.627 avisos**
de departamentos en arriendo, divididos en:

- **1.382 avisos** para entrenar el modelo (ajuste + early stopping).
- **245 avisos** que el modelo nunca vio durante el entrenamiento, usados solo para medir
  qué tan bien predice (conjunto de test).

Es un dataset acotado a una sola ciudad y a un momento puntual en el tiempo — no es un
catastro exhaustivo del mercado.

</div>

<div class="app-card">

### Qué variables usa el modelo para estimar el costo total

El modelo llega a un costo total mensual combinando **29 variables** de cada aviso, agrupadas
en tres categorías. Ninguna variable por sí sola determina el costo — es la combinación de
todas la que produce la estimación final. Los gastos comunes del propio aviso **no** son una
de esas variables (ver nota más abajo): se usan solo al comparar el resultado, no como input
del modelo.
"""
        + _render_grupos_features()
        + """

</div>

<div class="app-card">

### Qué variables pesan más en la predicción (SHAP)

No todas las 29 variables influyen por igual. Para medir cuánto pesa cada una, este modelo usa
**valores SHAP**: por cada aviso, el modelo reparte la diferencia entre su predicción y el
promedio general del mercado entre las variables que la explican — cada variable recibe una
contribución en pesos chilenos, positiva (empuja el costo hacia arriba) o negativa (lo empuja
hacia abajo), y la suma de todas esas contribuciones más el promedio general da la predicción
final de ese aviso. Es una técnica exacta para modelos de árboles como este (no una
aproximación), y es la misma que ya se usó para elegir estas 29 variables entre 41 candidatas.

Promediando esas contribuciones (en valor absoluto) sobre los 245 avisos de test se obtiene qué
tan determinante es cada variable **en general** — no para un aviso puntual:

"""
        + f'<img src="{_shap_img_data_uri()}" alt="Importancia SHAP de las variables" style="width:100%;max-width:640px;display:block;margin:6px auto;" />'
        + _render_shap_tabla()
        + """

Las **características físicas** (superficie, baños, estacionamientos y el resto de las
amenities) concentran en conjunto **~75%** de la importancia total, ubicación y mercado local
**~16%**, y el contexto socioeconómico del sector **~9%**. Superficie útil y número de baños,
por sí solas, ya explican cerca de un tercio de cada predicción — son, con diferencia, lo que
más mueve el costo total estimado.

</div>

<div class="app-card">

### Qué tan preciso es (el error del modelo)

Medido sobre los 245 avisos de test (que el modelo no usó para entrenar), comparando el costo
total predicho contra el costo total real (arriendo + gastos comunes):

| Métrica | Valor | Qué significa |
|---|---|---|
| Error promedio (MAE) | **≈ \\$61.500** | En promedio, la predicción se equivoca por unos \\$61.500 respecto al costo total real |
| Error porcentual (MAPE) | **≈ 10,0%** | En promedio, la predicción se desvía ~10,0% del costo total real |
| R² | **≈ 0,80** | El modelo explica ~80% de la variación de costo total entre avisos — queda ~20% sin explicar |

El error **no es parejo en todo el rango de precios**: en departamentos más caros (sobre
~\\$740.000 de costo total) el error típico sube a más de \\$114.000, porque hay menos avisos en
ese rango para aprender de ellos y los precios son más variables. Por eso la calificación de
"oportunidad" no compara contra un error único, sino contra el error típico de avisos de costo
similar (ver estratos más abajo).

**¿Es esto bueno?** Como referencia, se compara contra dos formas "ingenuas" de estimar un
costo total sin ningún modelo:

| Cómo se estima el costo total | Error promedio (MAE) |
|---|---|
| Usar siempre el costo total promedio del mercado, sin distinguir entre avisos | ≈ \\$141.900 |
| Costo total/m² del sector × superficie del departamento, sin ajustar nada más | ≈ \\$93.200 |
| **Este modelo** | **≈ \\$61.500** |

El modelo reduce el error en un tercio frente a usar solo el costo total/m² del sector, y en
más de la mitad frente a asumir un costo total promedio único para todo el mercado.

</div>

<div class="app-card">

### Los estratos (deciles de costo total)

Para calibrar cuándo un costo total es "raro", los 245 avisos de test se agruparon en **10
estratos según su costo total mensual** (deciles), y para cada estrato se calculó su propio
error típico (mediana del error y MAD — desviación absoluta mediana, una versión del error
típico menos sensible a valores extremos que la desviación estándar). Los bordes de costo
total de cada estrato en el modelo vigente son:

`< $417.000` · `$417.000–460.000` · `$460.000–495.200` · `$495.200–548.000` ·
`$548.000–580.075` · `$580.075–630.000` · `$630.000–664.000` · `$664.000–730.000` ·
`$730.000–796.060` · `> $796.060`

Cada aviso se compara solo contra el error típico de su propio estrato, no contra un
promedio general del mercado.

</div>

<div class="app-card">

### Cuándo es "Oportunidad", "Precio de mercado" o "Caro"

Para cada aviso se calcula un **z-score robusto**: qué tan lejos está su error (costo total
real − costo total predicho) de la mediana de error de su estrato, medido en unidades de MAD
de ese mismo estrato. Con eso:

- **Oportunidad**: el costo total real está bastante más bajo de lo esperado para avisos
  similares (z-score robusto menor a −1).
- **Caro**: el costo total real está bastante más alto de lo esperado (z-score robusto mayor
  a 1).
- **Precio de mercado**: el costo total real está dentro de lo esperado para avisos similares.

La app solo muestra la etiqueta en palabras — a propósito no se expone el número del
z-score ni el decil, para no dar una falsa sensación de precisión numérica.

</div>

<div class="app-card">

### Qué significa el nivel de confianza

El modelo no es un solo modelo: es un **ensamble de 10 modelos** entrenados con distintas
semillas aleatorias sobre los mismos datos. El nivel de confianza mide **qué tan de acuerdo
están esos 10 modelos entre sí** para un aviso en particular (su coeficiente de variación):

- **Alta**: los 10 modelos predicen costos totales muy parecidos entre sí.
- **Media**: hay algo de desacuerdo entre los 10 modelos.
- **Baja**: los 10 modelos predicen costos totales bastante distintos entre sí — es una señal de
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

- El modelo **no captura ni considera todas las variables** que afectan el costo total real de
  un arriendo: estado de conservación interno, luminosidad, ruido, calidad de terminaciones,
  vista, estado del edificio/administración, condiciones del contrato, urgencia del
  arrendador, entre otras cosas que no están en los datos scrapeados.
- Los datos vienen de avisos publicados — pueden tener errores de tipeo, precios o gastos
  comunes desactualizados, o información incompleta que el scraper no pudo capturar bien.
- El error típico del modelo (~10,0%, más alto en tramos de costo poco frecuentes) significa
  que **puede fallar**, incluso en avisos marcados con confianza alta.
- Revisa siempre la publicación original y otras variables no capturadas por el modelo antes
  de sacar conclusiones sobre un aviso puntual.

</div>
        """,
        unsafe_allow_html=True,
    )
