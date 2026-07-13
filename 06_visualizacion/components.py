"""Tarjetas de avisos y mapa interactivo."""

import html
import json
from datetime import date

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from styles import (
    COLOR_PRIMARY,
    COLOR_TEXT_BODY,
    COLOR_TEXT_TITLE,
    CONFIANZA_COLORS,
    CONFIANZA_LABELS,
    CONFIANZA_SYMBOLS,
    ETIQUETA_COLORS,
    ETIQUETA_LABELS,
)

# Tiles Positron de CARTO: mismo estilo visual que pdk.map_styles.LIGHT, sin
# requerir token (a diferencia de Mapbox).
_BASEMAP_TILES_URL = "https://basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png"

def _clp(value: float) -> str:
    return f"{value:,.0f}".replace(",", ".")


def _format_ubicacion(comuna: str, barrio) -> str:
    comuna_fmt = html.escape(comuna.replace("-biobio", "").replace("-", " ").title())
    if isinstance(barrio, str) and barrio.strip():
        return f"{html.escape(barrio)}, {comuna_fmt}"
    return comuna_fmt


def _format_frescura(fecha_str) -> str:
    if not isinstance(fecha_str, str) or not fecha_str:
        return "Datos verificados: fecha desconocida"
    try:
        fecha = date.fromisoformat(fecha_str[:10])
    except ValueError:
        return "Datos verificados: fecha desconocida"
    dias = (date.today() - fecha).days
    if dias <= 0:
        return "Datos verificados: hoy"
    if dias == 1:
        return "Datos verificados: hace 1 día"
    return f"Datos verificados: hace {dias} días"


def _format_publicacion(fecha_val) -> str:
    # fecha_publicacion_aprox es una aproximación calculada por el scraper a
    # partir de texto relativo ("hace 3 meses"), y queda nula en ~1 de cada 5
    # avisos activos porque esa página de detalle no siempre trae esa info:
    # se muestra "no disponible" en vez de inventarle una fecha al aviso.
    # Llega como Timestamp/NaT (convertir_precios_uf_a_clp la normaliza con
    # pd.to_datetime como parte de su propia lógica), no como string.
    if pd.isna(fecha_val):
        return "Publicado: fecha no disponible"
    fecha = pd.Timestamp(fecha_val).date()
    dias = (date.today() - fecha).days
    if dias <= 0:
        return "Publicado: hoy"
    if dias == 1:
        return "Publicado: hace 1 día"
    if dias < 30:
        return f"Publicado: hace {dias} días"
    if dias < 365:
        meses = dias // 30
        return f"Publicado: hace {meses} mes{'es' if meses != 1 else ''}"
    anos = dias // 365
    return f"Publicado: hace {anos} año{'s' if anos != 1 else ''}"


def _card_html(row: pd.Series) -> str:
    etiqueta = row["etiqueta"]
    confianza = row["nivel_confianza"]
    titulo = html.escape(str(row["titulo"]))

    return f"""
        <div class="app-card">
            <div class="badge-row">
                <span class="badge" style="background-color:{ETIQUETA_COLORS.get(etiqueta, '#888')}">
                    {ETIQUETA_LABELS.get(etiqueta, etiqueta)}
                </span>
                <span class="badge" style="background-color:{CONFIANZA_COLORS.get(confianza, '#888')}">
                    {CONFIANZA_SYMBOLS.get(confianza, '')} Confianza {CONFIANZA_LABELS.get(confianza, confianza)}
                </span>
            </div>
            <p class="card-title">{titulo}</p>
            <p class="card-location">{_format_ubicacion(row['comuna'], row.get('barrio'))}</p>
            <p class="card-facts">🛏️ {int(row['dormitorios'])} dorm. &nbsp;·&nbsp;
                🚿 {int(row['banos'])} baños &nbsp;·&nbsp;
                📐 {row['superficie_util_m2']:.0f} m²</p>
            <p class="card-price">${_clp(row['precio'])}</p>
            <p class="card-expenses">Gastos comunes: ${_clp(row['gastos_comunes'])}</p>
            <p class="card-published">{_format_publicacion(row.get('fecha_publicacion_aprox'))}</p>
            <p class="card-freshness">{_format_frescura(row.get('fecha_ultimo_chequeo_estado'))}</p>
            <p class="card-link"><a href="{html.escape(row['url'])}" target="_blank" rel="noopener noreferrer">
                Ver publicación original →</a></p>
        </div>
        """


def render_cards(df: pd.DataFrame) -> None:
    # Un solo st.markdown para todas las tarjetas de la página en vez de uno
    # por fila: cada llamada a st.markdown tiene overhead de protocolo
    # (serialización + mensaje al frontend + nodo DOM propio) que con
    # cientos de tarjetas se vuelve el costo dominante del rerun.
    cards_html = "".join(_card_html(row) for _, row in df.iterrows())
    st.markdown(cards_html, unsafe_allow_html=True)


def _hex_to_rgb(hex_color: str) -> list:
    hex_color = hex_color.lstrip("#")
    return [int(hex_color[i : i + 2], 16) for i in (0, 2, 4)]


def _map_point(row: pd.Series) -> dict:
    etiqueta = row["etiqueta"]
    confianza = row["nivel_confianza"]
    # Solo los campos crudos (ya escapados/formateados) viajan al cliente:
    # la tarjeta completa se arma en JS (ver buildCardHtml) en vez de mandar
    # el HTML de _card_html ya renderizado para cada una de las ~1700 filas,
    # que infla el payload del iframe ~1.5MB para mostrar como máximo una
    # tarjeta a la vez.
    return {
        "lon": row["longitud"],
        "lat": row["latitud"],
        "color": _hex_to_rgb(ETIQUETA_COLORS.get(etiqueta, "#888888")),
        "tooltip": (
            f"<b>{html.escape(str(row['titulo']))}</b><br/>"
            f"${_clp(row['precio'])}.-<br/>"
            f"{int(row['dormitorios'])} dorm. · {int(row['banos'])} baños"
        ),
        "etiquetaLabel": ETIQUETA_LABELS.get(etiqueta, etiqueta),
        "etiquetaColor": ETIQUETA_COLORS.get(etiqueta, "#888"),
        "confianzaLabel": CONFIANZA_LABELS.get(confianza, confianza),
        "confianzaColor": CONFIANZA_COLORS.get(confianza, "#888"),
        "confianzaSymbol": CONFIANZA_SYMBOLS.get(confianza, ""),
        "titulo": html.escape(str(row["titulo"])),
        "ubicacion": _format_ubicacion(row["comuna"], row.get("barrio")),
        "dormitorios": int(row["dormitorios"]),
        "banos": int(row["banos"]),
        "superficie": f"{row['superficie_util_m2']:.0f}",
        "precio": _clp(row["precio"]),
        "gastosComunes": _clp(row["gastos_comunes"]),
        "publicado": _format_publicacion(row.get("fecha_publicacion_aprox")),
        "frescura": _format_frescura(row.get("fecha_ultimo_chequeo_estado")),
        "url": html.escape(row["url"]),
    }


def render_map(df: pd.DataFrame) -> None:
    if df.empty or df["latitud"].isna().all():
        st.info("No hay avisos con ubicación para mostrar en el mapa.")
        return

    center_lat, center_lon = df["latitud"].mean(), df["longitud"].mean()
    zoom = 12 if df["latitud"].std() < 0.1 else 10

    points = [_map_point(row) for _, row in df.iterrows()]
    # Los campos de texto ya vienen html.escape()-ados desde Python, así que
    # un "</script" literal requeriría un "<" sin escapar; el replace es solo
    # defensa en profundidad.
    data_json = json.dumps(points, ensure_ascii=False).replace("</script", "<\\/script")

    map_height = 720
    html_doc = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8" />
<script src="https://unpkg.com/deck.gl@^8.9.0/dist.min.js"></script>
<style>
  html, body {{ margin: 0; padding: 0; overflow: hidden; font-family: "Source Sans Pro", sans-serif; }}
  #container {{ position: relative; width: 100%; height: {map_height}px; }}
  #map {{ position: absolute; inset: 0; }}
  .app-card {{
      background-color: #FFFFFF;
      box-shadow: 0 1px 4px rgba(0,0,0,0.12);
      border-radius: 10px;
      padding: 16px 18px 14px 18px;
  }}
  .badge-row {{ display: flex; gap: 6px; margin-bottom: 6px; }}
  .badge {{
      display: inline-block; padding: 3px 10px; border-radius: 6px;
      color: white; font-size: 0.78rem; font-weight: 600;
  }}
  .card-title {{ color: {COLOR_TEXT_TITLE}; font-size: 1.02rem; font-weight: 600; margin: 2px 0 0 0; }}
  .card-location {{ color: {COLOR_TEXT_BODY}; font-size: 0.85rem; margin-bottom: 6px; }}
  .card-facts {{ color: {COLOR_TEXT_BODY}; font-size: 0.88rem; margin-bottom: 6px; }}
  .card-price {{ color: {COLOR_TEXT_TITLE}; font-size: 1.35rem; font-weight: 700; margin-bottom: 0; }}
  .card-expenses {{ color: {COLOR_TEXT_BODY}; font-size: 0.8rem; margin-bottom: 6px; }}
  .card-published {{ color: #9A9A9A; font-size: 0.75rem; margin-bottom: 2px; }}
  .card-freshness {{ color: #9A9A9A; font-size: 0.75rem; margin-bottom: 8px; }}
  .card-link a {{ color: {COLOR_PRIMARY}; font-size: 0.85rem; font-weight: 600; text-decoration: none; }}
  .card-link a:hover {{ text-decoration: underline; }}
  .map-popup {{ position: absolute; z-index: 20; width: 300px; display: none; }}
  .map-popup-close {{
      position: absolute; top: 6px; right: 8px; width: 22px; height: 22px;
      border-radius: 50%; border: none; background: rgba(0,0,0,0.08);
      color: {COLOR_TEXT_BODY}; font-size: 15px; line-height: 22px; text-align: center;
      cursor: pointer; padding: 0;
  }}
  .map-popup-close:hover {{ background: rgba(0,0,0,0.16); }}
</style>
</head>
<body>
<div id="container">
  <div id="map"></div>
  <div id="popup" class="map-popup">
    <button id="popup-close" class="map-popup-close" aria-label="Cerrar">×</button>
    <div id="popup-content"></div>
  </div>
</div>
<script>
const DATA = {data_json};

function buildCardHtml(d) {{
  return `
    <div class="app-card">
        <div class="badge-row">
            <span class="badge" style="background-color:${{d.etiquetaColor}}">${{d.etiquetaLabel}}</span>
            <span class="badge" style="background-color:${{d.confianzaColor}}">${{d.confianzaSymbol}} Confianza ${{d.confianzaLabel}}</span>
        </div>
        <p class="card-title">${{d.titulo}}</p>
        <p class="card-location">${{d.ubicacion}}</p>
        <p class="card-facts">🛏️ ${{d.dormitorios}} dorm. &nbsp;·&nbsp; 🚿 ${{d.banos}} baños &nbsp;·&nbsp; 📐 ${{d.superficie}} m²</p>
        <p class="card-price">$${{d.precio}}</p>
        <p class="card-expenses">Gastos comunes: $${{d.gastosComunes}}</p>
        <p class="card-published">${{d.publicado}}</p>
        <p class="card-freshness">${{d.frescura}}</p>
        <p class="card-link"><a href="${{d.url}}" target="_blank" rel="noopener noreferrer">Ver publicación original →</a></p>
    </div>`;
}}

const container = document.getElementById('container');
const popup = document.getElementById('popup');
const popupContent = document.getElementById('popup-content');
const popupClose = document.getElementById('popup-close');

let selected = null;

function clampedPosition(x, y) {{
  const rect = container.getBoundingClientRect();
  const popupW = 300;
  const popupHMax = 420;
  let left = x + 12;
  let top = y + 12;
  if (left + popupW > rect.width) left = x - popupW - 12;
  if (left < 0) left = 8;
  if (top + popupHMax > rect.height) top = y - popupHMax - 12;
  if (top < 0) top = 8;
  return {{ left, top }};
}}

function showPopup(object, x, y) {{
  selected = object;
  popupContent.innerHTML = buildCardHtml(object);
  const pos = clampedPosition(x, y);
  popup.style.left = pos.left + 'px';
  popup.style.top = pos.top + 'px';
  popup.style.display = 'block';
}}

function hidePopup() {{
  selected = null;
  popup.style.display = 'none';
}}

popupClose.addEventListener('click', (e) => {{
  e.stopPropagation();
  hidePopup();
}});

const {{ DeckGL, ScatterplotLayer, TileLayer, BitmapLayer, WebMercatorViewport }} = deck;

const basemap = new TileLayer({{
  id: 'basemap',
  data: '{_BASEMAP_TILES_URL}',
  minZoom: 0,
  maxZoom: 19,
  tileSize: 256,
  renderSubLayers: props => {{
    const {{ bbox: {{ west, south, east, north }} }} = props.tile;
    return new BitmapLayer(props, {{ data: null, image: props.data, bounds: [west, south, east, north] }});
  }},
}});

const avisos = new ScatterplotLayer({{
  id: 'avisos',
  data: DATA,
  getPosition: d => [d.lon, d.lat],
  getFillColor: d => d.color,
  getLineColor: [255, 255, 255],
  lineWidthMinPixels: 1,
  stroked: true,
  filled: true,
  radiusMinPixels: 7.5,
  radiusMaxPixels: 13.5,
  pickable: true,
  autoHighlight: true,
}});

new DeckGL({{
  container: 'map',
  initialViewState: {{ longitude: {center_lon}, latitude: {center_lat}, zoom: {zoom}, pitch: 0, bearing: 0 }},
  controller: {{ doubleClickZoom: false }},
  layers: [basemap, avisos],
  getTooltip: ({{ object }}) => object && !selected ? {{
    html: object.tooltip,
    style: {{ backgroundColor: 'white', color: '{COLOR_TEXT_TITLE}', fontSize: '13px' }}
  }} : null,
  onClick: (info) => {{
    if (info.object) {{
      showPopup(info.object, info.x, info.y);
    }} else {{
      hidePopup();
    }}
  }},
  onViewStateChange: ({{ viewState }}) => {{
    if (selected) {{
      const viewport = new WebMercatorViewport(viewState);
      const [x, y] = viewport.project([selected.lon, selected.lat]);
      const pos = clampedPosition(x, y);
      popup.style.left = pos.left + 'px';
      popup.style.top = pos.top + 'px';
    }}
  }},
}});
</script>
</body>
</html>
"""
    components.html(html_doc, height=map_height, scrolling=False)
