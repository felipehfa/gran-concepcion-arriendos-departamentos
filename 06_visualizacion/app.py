"""Gran Concepción Rentals — visualizador de departamentos en arriendo."""

import streamlit as st

from components import render_card, render_map
from data import load_data
from explicacion import render as render_explicacion
from filters import render_sidebar
from styles import CONFIANZA_ORDER, ETIQUETA_ORDER, inject_css

ORDER_OPTIONS = [
    "Más recientes primero",
    "Más relevantes",
    "Mejor oportunidad",
    "Precio: menor a mayor",
    "Precio: mayor a menor",
]


def sort_listings(df, order: str):
    if df.empty:
        return df
    if order == "Mejor oportunidad":
        return df.sort_values("z_robusto", ascending=True)
    if order == "Más recientes primero":
        # fecha_publicacion_aprox es aproximada (ver explicacion.py) y puede ser
        # nula (~1 de cada 5 avisos activos, el scraper no siempre la captura):
        # esos avisos van al final en vez de asumirles una fecha.
        return df.sort_values("fecha_publicacion_aprox", ascending=False, na_position="last")
    if order == "Precio: menor a mayor":
        return df.sort_values("precio", ascending=True)
    if order == "Precio: mayor a menor":
        return df.sort_values("precio", ascending=False)
    # "Más relevantes": etiqueta (oportunidad > precio_de_mercado > caro),
    # luego nivel_confianza (alta > media > baja) dentro de cada grupo.
    out = df.copy()
    out["_etiqueta_rank"] = out["etiqueta"].map(ETIQUETA_ORDER).fillna(99)
    out["_confianza_rank"] = out["nivel_confianza"].map(CONFIANZA_ORDER).fillna(99)
    return out.sort_values(["_etiqueta_rank", "_confianza_rank"]).drop(
        columns=["_etiqueta_rank", "_confianza_rank"]
    )


def render_buscador() -> None:
    df = load_data()

    st.markdown(
        '<h1 class="app-title">Buscador de Oportunidades de Arriendo — Gran Concepción</h1>'
        '<p class="subtitle">Departamentos evaluados por un modelo de precios para detectar '
        'cuáles están bajo o sobre su valor esperado</p>',
        unsafe_allow_html=True,
    )

    if df.empty:
        st.warning("No hay avisos con predicción disponible en la base de datos todavía.")
        return

    filtered_df = render_sidebar(df)

    order_col, count_col = st.columns([1, 3])
    with order_col:
        order = st.selectbox("Ordenar por", ORDER_OPTIONS, label_visibility="collapsed")
    with count_col:
        st.markdown(
            f'<div class="result-count-card">{len(filtered_df)} departamentos encontrados</div>',
            unsafe_allow_html=True,
        )

    filtered_df = sort_listings(filtered_df, order)

    if filtered_df.empty:
        st.info("Ningún aviso coincide con los filtros seleccionados.")
        return

    list_col, map_col = st.columns([1, 1])

    with list_col:
        with st.container(height=800, border=False):
            for _, row in filtered_df.iterrows():
                render_card(row)

    with map_col:
        render_map(filtered_df)


def main() -> None:
    st.set_page_config(page_title="Gran Concepción Rentals", layout="wide")
    inject_css()

    tab_buscador, tab_explicacion = st.tabs(["🏠 Buscador", "ℹ️ Cómo funciona"])

    with tab_buscador:
        render_buscador()

    with tab_explicacion:
        st.markdown(
            '<h1 class="app-title">Cómo funciona este buscador</h1>',
            unsafe_allow_html=True,
        )
        render_explicacion()


if __name__ == "__main__":
    main()
