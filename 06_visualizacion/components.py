"""Tarjetas de avisos y mapa interactivo."""

import html
from datetime import date

import folium
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from styles import (
    CONFIANZA_COLORS,
    CONFIANZA_LABELS,
    CONFIANZA_SYMBOLS,
    ETIQUETA_COLORS,
    ETIQUETA_LABELS,
)

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


def render_card(row: pd.Series) -> None:
    etiqueta = row["etiqueta"]
    confianza = row["nivel_confianza"]
    titulo = html.escape(str(row["titulo"]))

    st.markdown(
        f"""
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
        """,
        unsafe_allow_html=True,
    )


def _pin_icon(color: str) -> folium.DivIcon:
    # Pin propio (círculo + punta) en vez de folium.Icon: su paleta fija de
    # colores con nombre no permite igualar los hex exactos de las tarjetas.
    # Sin trazo (antes blanco): con muchos pines juntos el borde recargaba el
    # mapa. Tamaño al 60% del original (26x38 -> 16x23).
    svg = f"""
    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="23" viewBox="0 0 26 38">
        <path d="M13 0C5.8 0 0 5.8 0 13c0 9.7 13 25 13 25s13-15.3 13-25C26 5.8 20.2 0 13 0z"
              fill="{color}"/>
        <circle cx="13" cy="13" r="5" fill="#FFFFFF"/>
    </svg>
    """
    return folium.DivIcon(html=svg, icon_size=(16, 23), icon_anchor=(8, 23))


def render_map(df: pd.DataFrame) -> None:
    if df.empty or df["latitud"].isna().all():
        st.info("No hay avisos con ubicación para mostrar en el mapa.")
        return

    center = (df["latitud"].mean(), df["longitud"].mean())
    zoom = 12 if df["latitud"].std() < 0.1 else 10

    m = folium.Map(location=center, zoom_start=zoom, tiles="cartodbpositron")

    for _, row in df.iterrows():
        titulo = html.escape(str(row["titulo"]))
        popup_html = f"""
        <div style="font-family: sans-serif; font-size: 13px; min-width: 180px;">
            <b>{titulo}</b><br>
            ${_clp(row['precio'])}.-<br>
            {int(row['dormitorios'])} dorm. · {int(row['banos'])} baños<br>
            <a href="{html.escape(row['url'])}" target="_blank" rel="noopener noreferrer">Ver publicación</a>
        </div>
        """
        # Leaflet por defecto da más z-index a los pines más abajo en pantalla
        # (para simular profundidad); ese offset alto fuerza a "oportunidad" a
        # quedar siempre por encima de precio_de_mercado/caro sin importar su
        # posición.
        z_index_offset = 1000 if row["etiqueta"] == "oportunidad" else 0
        folium.Marker(
            location=(row["latitud"], row["longitud"]),
            popup=folium.Popup(popup_html, max_width=250),
            tooltip=titulo,
            icon=_pin_icon(ETIQUETA_COLORS.get(row["etiqueta"], "#888")),
            z_index_offset=z_index_offset,
        ).add_to(m)

    st_folium(m, use_container_width=True, height=720, returned_objects=[])
