"""Sidebar de filtros y lógica de filtrado del dataframe."""

import pandas as pd
import streamlit as st

from data import AMENITY_COLUMNS

_ROOM_BUCKETS = ["1", "2", "3", "4+"]


def _room_bucket(n: int) -> str:
    return "4+" if n >= 4 else str(n)


def _default_filters(df: pd.DataFrame) -> dict:
    price_min, price_max = int(df["precio"].min()), int(df["precio"].max())
    sup_min, sup_max = int(df["superficie_util_m2"].min()), int(df["superficie_util_m2"].max())
    return {
        "f_price": (price_min, price_max),
        "f_superficie": (sup_min, sup_max),
        "f_comuna": sorted(df["comuna"].dropna().unique().tolist()),
        **{f"f_dorm_{b}": True for b in _ROOM_BUCKETS},
        **{f"f_banos_{b}": True for b in _ROOM_BUCKETS},
        "f_etiqueta_oportunidad": True,
        "f_etiqueta_precio_de_mercado": True,
        "f_etiqueta_caro": True,
        "f_confianza_alta confianza": True,
        "f_confianza_confianza media": True,
        "f_confianza_confianza baja": True,
        "f_amenity_amoblado": False,
        "f_amenity_piscina": False,
        "f_amenity_ascensor": False,
        "f_amenity_estacionamiento": False,
        "f_amenity_conserjeria": False,
        "f_incluir_pausados": False,
    }


def _reset_filters(df: pd.DataFrame) -> None:
    for key, value in _default_filters(df).items():
        st.session_state[key] = value


def render_sidebar(df: pd.DataFrame) -> pd.DataFrame:
    defaults = _default_filters(df)
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)

    with st.sidebar:
        st.markdown("### Filtros")
        st.button("Limpiar filtros", on_click=_reset_filters, args=(df,), use_container_width=True)

        st.markdown("**Precio**")
        price_min, price_max = int(df["precio"].min()), int(df["precio"].max())
        st.slider(
            "Rango de precio ($)",
            min_value=price_min,
            max_value=price_max,
            step=10_000,
            key="f_price",
            label_visibility="collapsed",
        )

        st.markdown("**Dormitorios**")
        dorm_counts = df["dormitorios"].apply(_room_bucket).value_counts()
        for b in _ROOM_BUCKETS:
            st.checkbox(f"{b} ({dorm_counts.get(b, 0)})", key=f"f_dorm_{b}")

        st.markdown("**Baños**")
        banos_counts = df["banos"].apply(_room_bucket).value_counts()
        for b in _ROOM_BUCKETS:
            st.checkbox(f"{b} ({banos_counts.get(b, 0)})", key=f"f_banos_{b}")

        st.markdown("**Comuna**")
        st.multiselect(
            "Comuna",
            options=sorted(df["comuna"].dropna().unique().tolist()),
            key="f_comuna",
            label_visibility="collapsed",
        )

        st.markdown("**Superficie útil (m²)**")
        sup_min, sup_max = int(df["superficie_util_m2"].min()), int(df["superficie_util_m2"].max())
        st.slider(
            "Superficie útil (m²)",
            min_value=sup_min,
            max_value=sup_max,
            key="f_superficie",
            label_visibility="collapsed",
        )

        st.markdown("**Etiqueta del modelo**")
        st.checkbox("Oportunidad", key="f_etiqueta_oportunidad")
        st.checkbox("Precio de mercado", key="f_etiqueta_precio_de_mercado")
        st.checkbox("Caro", key="f_etiqueta_caro")

        st.markdown("**Nivel de confianza**")
        st.checkbox("Alta", key="f_confianza_alta confianza")
        st.checkbox("Media", key="f_confianza_confianza media")
        st.checkbox("Baja", key="f_confianza_confianza baja")

        with st.expander("Más filtros"):
            for col, label in AMENITY_COLUMNS.items():
                amenity_key = "estacionamiento" if col == "estacionamientos" else col
                st.checkbox(label, key=f"f_amenity_{amenity_key}")

        st.markdown("**Estado**")
        st.checkbox("Incluir pausados", key="f_incluir_pausados")

    return _apply_filters(df)


def _apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    s = st.session_state
    out = df.copy()

    if not s["f_incluir_pausados"]:
        out = out[out["estado_publicacion"] == "activo"]

    price_min, price_max = s["f_price"]
    out = out[out["precio"].between(price_min, price_max)]

    sup_min, sup_max = s["f_superficie"]
    out = out[out["superficie_util_m2"].between(sup_min, sup_max)]

    if s["f_comuna"]:
        out = out[out["comuna"].isin(s["f_comuna"])]

    dorm_selected = {b for b in _ROOM_BUCKETS if s[f"f_dorm_{b}"]}
    out = out[out["dormitorios"].apply(_room_bucket).isin(dorm_selected)]

    banos_selected = {b for b in _ROOM_BUCKETS if s[f"f_banos_{b}"]}
    out = out[out["banos"].apply(_room_bucket).isin(banos_selected)]

    etiquetas_selected = {
        etiqueta
        for etiqueta in ["oportunidad", "precio_de_mercado", "caro"]
        if s[f"f_etiqueta_{etiqueta}"]
    }
    out = out[out["etiqueta"].isin(etiquetas_selected)]

    confianzas_selected = {
        c for c in ["alta confianza", "confianza media", "confianza baja"] if s[f"f_confianza_{c}"]
    }
    out = out[out["nivel_confianza"].isin(confianzas_selected)]

    for col in ["amoblado", "piscina", "ascensor", "estacionamiento", "conserjeria"]:
        if s[f"f_amenity_{col}"]:
            out = out[out[col]]

    return out
