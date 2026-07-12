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
    ETIQUETA_COLORS,
    ETIQUETA_LABELS,
    FOLIUM_ICON_COLOR,
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
                    Confianza {CONFIANZA_LABELS.get(confianza, confianza)}
                </span>
            </div>
            <p class="card-title">{titulo}</p>
            <p class="card-location">{_format_ubicacion(row['comuna'], row.get('barrio'))}</p>
            <p class="card-facts">🛏️ {int(row['dormitorios'])} dorm. &nbsp;·&nbsp;
                🚿 {int(row['banos'])} baños &nbsp;·&nbsp;
                📐 {row['superficie_util_m2']:.0f} m²</p>
            <p class="card-price">${_clp(row['precio'])}</p>
            <p class="card-expenses">Gastos comunes: ${_clp(row['gastos_comunes'])}</p>
            <p class="card-freshness">{_format_frescura(row.get('fecha_ultimo_chequeo_estado'))}</p>
            <p class="card-link"><a href="{html.escape(row['url'])}" target="_blank" rel="noopener noreferrer">
                Ver publicación original →</a></p>
        </div>
        """,
        unsafe_allow_html=True,
    )


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
        folium.Marker(
            location=(row["latitud"], row["longitud"]),
            popup=folium.Popup(popup_html, max_width=250),
            tooltip=titulo,
            icon=folium.Icon(color=FOLIUM_ICON_COLOR.get(row["etiqueta"], "gray")),
        ).add_to(m)

    st_folium(m, use_container_width=True, height=720, returned_objects=[])
