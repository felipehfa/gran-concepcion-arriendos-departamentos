"""Paleta de colores y CSS compartido de la app."""

import streamlit as st

COLOR_PRIMARY = "#2968C8"
COLOR_BACKGROUND = "#F5F5F5"
COLOR_CARD_BG = "#FFFFFF"
COLOR_TEXT_BODY = "#4A4A4A"
COLOR_TEXT_TITLE = "#1A2B4C"

COLOR_OPORTUNIDAD = "#2E7D32"
COLOR_PRECIO_MERCADO = "#5A6B7D"
COLOR_CARO = "#C62828"

COLOR_CONFIANZA_ALTA = "#2E7D32"
COLOR_CONFIANZA_MEDIA = "#B8860B"
COLOR_CONFIANZA_BAJA = "#8A8A8A"

ETIQUETA_LABELS = {
    "oportunidad": "Oportunidad",
    "precio_de_mercado": "Precio de mercado",
    "caro": "Caro",
}

ETIQUETA_COLORS = {
    "oportunidad": COLOR_OPORTUNIDAD,
    "precio_de_mercado": COLOR_PRECIO_MERCADO,
    "caro": COLOR_CARO,
}

ETIQUETA_ORDER = {"oportunidad": 0, "precio_de_mercado": 1, "caro": 2}

CONFIANZA_LABELS = {
    "alta confianza": "Alta",
    "confianza media": "Media",
    "confianza baja": "Baja",
}

CONFIANZA_COLORS = {
    "alta confianza": COLOR_CONFIANZA_ALTA,
    "confianza media": COLOR_CONFIANZA_MEDIA,
    "confianza baja": COLOR_CONFIANZA_BAJA,
}

CONFIANZA_ORDER = {"alta confianza": 0, "confianza media": 1, "confianza baja": 2}

FOLIUM_ICON_COLOR = {
    "oportunidad": "green",
    "precio_de_mercado": "blue",
    "caro": "red",
}


def inject_css() -> None:
    # !important en los colores de texto: Streamlit reinyecta su propio color
    # de tema (blanco si el navegador está en modo oscuro) con selectores más
    # específicos que los nuestros, así que sin !important queda pisado.
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-color: {COLOR_BACKGROUND};
        }}
        [data-testid="stSidebar"] {{
            background-color: {COLOR_BACKGROUND};
        }}
        .app-title {{
            color: {COLOR_TEXT_TITLE} !important;
            font-size: 2.2rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }}
        .subtitle {{
            color: {COLOR_TEXT_BODY} !important;
            font-size: 1.05rem;
            margin-top: 0;
        }}
        .result-count {{
            color: {COLOR_TEXT_BODY} !important;
            font-size: 0.95rem;
        }}
        .app-card, [data-testid="stSidebarUserContent"] {{
            background-color: {COLOR_CARD_BG};
            box-shadow: 0 1px 4px rgba(0,0,0,0.12);
            border-radius: 10px;
            padding: 16px 18px 14px 18px;
        }}
        .app-card {{
            margin-bottom: 14px;
        }}
        [data-testid="stSidebarUserContent"] {{
            margin: 14px;
        }}
        .badge-row {{
            display: flex;
            gap: 6px;
            margin-bottom: 6px;
        }}
        .badge {{
            display: inline-block;
            padding: 3px 10px;
            border-radius: 6px;
            color: white !important;
            font-size: 0.78rem;
            font-weight: 600;
        }}
        .card-title {{
            color: {COLOR_TEXT_TITLE} !important;
            font-size: 1.02rem;
            font-weight: 600;
            margin: 2px 0 0 0;
        }}
        .card-location {{
            color: {COLOR_TEXT_BODY} !important;
            font-size: 0.85rem;
            margin-bottom: 6px;
        }}
        .card-facts {{
            color: {COLOR_TEXT_BODY} !important;
            font-size: 0.88rem;
            margin-bottom: 6px;
        }}
        .card-price {{
            color: {COLOR_TEXT_TITLE} !important;
            font-size: 1.35rem;
            font-weight: 700;
            margin-bottom: 0;
        }}
        .card-expenses {{
            color: {COLOR_TEXT_BODY} !important;
            font-size: 0.8rem;
            margin-bottom: 6px;
        }}
        .card-published {{
            color: #9A9A9A !important;
            font-size: 0.75rem;
            margin-bottom: 2px;
        }}
        .card-freshness {{
            color: #9A9A9A !important;
            font-size: 0.75rem;
            margin-bottom: 8px;
        }}
        .card-link a {{
            color: {COLOR_PRIMARY} !important;
            font-size: 0.85rem;
            font-weight: 600;
            text-decoration: none;
        }}
        .card-link a:hover {{
            text-decoration: underline;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
