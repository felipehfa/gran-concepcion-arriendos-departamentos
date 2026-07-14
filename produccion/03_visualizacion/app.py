"""Gran Concepción Rentals — visualizador de departamentos en arriendo."""

import math

import streamlit as st

from components import render_cards, render_map
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

# Tarjetas por página: sin paginar, un rerun (cualquier filtro/orden) montaba
# un st.markdown y cientos de nodos DOM por cada aviso filtrado, la mayoría
# fuera del área visible del contenedor con scroll.
PAGE_SIZE = 24


def _paginate(df, page_key: str = "f_page"):
    st.session_state.setdefault(page_key, 1)
    total_pages = max(1, math.ceil(len(df) / PAGE_SIZE))
    if st.session_state[page_key] > total_pages:
        st.session_state[page_key] = 1
    page = st.session_state[page_key]

    def _go_prev():
        st.session_state[page_key] -= 1

    def _go_next():
        st.session_state[page_key] += 1

    prev_col, label_col, next_col = st.columns([1, 2, 1])
    with prev_col:
        st.button("← Anterior", disabled=page <= 1, on_click=_go_prev, use_container_width=True)
    with label_col:
        st.markdown(f'<div class="page-label">Página {page} de {total_pages}</div>', unsafe_allow_html=True)
    with next_col:
        st.button("Siguiente →", disabled=page >= total_pages, on_click=_go_next, use_container_width=True)

    start = (page - 1) * PAGE_SIZE
    return df.iloc[start : start + PAGE_SIZE]


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

    # Si cambió algún filtro o el orden desde el último rerun, se vuelve a la
    # página 1 (si no, una tarjeta en la página 3 del filtro anterior no
    # tiene relación con el nuevo resultado filtrado).
    filter_signature = (
        order,
        tuple(sorted((k, str(v)) for k, v in st.session_state.items() if k.startswith("f_") and k != "f_page")),
    )
    if st.session_state.get("_last_filter_signature") != filter_signature:
        st.session_state["f_page"] = 1
        st.session_state["_last_filter_signature"] = filter_signature

    list_col, map_col = st.columns([1, 1])

    with list_col:
        page_df = _paginate(filtered_df)
        with st.container(height=800, border=False):
            render_cards(page_df)

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
