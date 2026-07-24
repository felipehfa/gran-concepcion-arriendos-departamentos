"""Pestaña de estadísticas diarias: avisos que entran y salen del estado
activo día a día, con los mismos filtros del buscador (sidebar compartida)."""

import altair as alt
import pandas as pd
import streamlit as st

from data import load_data_todos_los_estados, load_historial_diario
from filters import apply_atributo_filters
from styles import COLOR_CARO, COLOR_OPORTUNIDAD, COLOR_PRIMARY, COLOR_SERIE_2, COLOR_SERIE_3

FECHA_INICIO_HISTORIAL = "23/07/2026"


def _serie_diaria(historial: pd.DataFrame, ids_filtrados: set, incluir_pausados: bool) -> pd.DataFrame:
    """Una fila por día con el total de avisos disponibles ese día (según
    incluir_pausados) y cuántos entraron/salieron del estado 'activo' desde
    el día anterior. entran/salen quedan en None para el primer día del
    historial (no hay día anterior con el que comparar)."""
    hist = historial[historial["id_aviso"].isin(ids_filtrados)]
    if hist.empty:
        return pd.DataFrame(columns=["fecha", "total", "entran", "salen"])

    estados_total = {"activo", "pausado"} if incluir_pausados else {"activo"}

    filas = []
    ids_activos_ayer = None
    for fecha, dia in hist.groupby("fecha", sort=True):
        ids_total_hoy = set(dia.loc[dia["estado_publicacion"].isin(estados_total), "id_aviso"])
        ids_activos_hoy = set(dia.loc[dia["estado_publicacion"] == "activo", "id_aviso"])

        entran = len(ids_activos_hoy - ids_activos_ayer) if ids_activos_ayer is not None else None
        salen = len(ids_activos_ayer - ids_activos_hoy) if ids_activos_ayer is not None else None

        filas.append({"fecha": fecha, "total": len(ids_total_hoy), "entran": entran, "salen": salen})
        ids_activos_ayer = ids_activos_hoy

    return pd.DataFrame(filas)


def _serie_valor_m2(historial: pd.DataFrame, df_todos: pd.DataFrame, ids_filtrados: set, incluir_pausados: bool) -> pd.DataFrame:
    """Una fila por día con media/mediana/desviación estándar del valor
    ($/m² útil) de los avisos disponibles ese día (mismo criterio de
    "disponible" que `_serie_diaria`: activo, o activo+pausado si
    incluir_pausados). El precio y la superficie se toman del estado ACTUAL
    del aviso (igual que `apply_atributo_filters` para los filtros): rara
    vez cambian día a día, así que no hace falta reconstruirlos históricos."""
    hist = historial[historial["id_aviso"].isin(ids_filtrados)]
    if hist.empty:
        return pd.DataFrame(columns=["fecha", "media", "mediana", "desviacion_estandar"])

    valor_m2 = (df_todos["precio"] / df_todos["superficie_util_m2"]).replace([float("inf"), float("-inf")], pd.NA)
    valor_m2_por_id = pd.Series(valor_m2.values, index=df_todos["id_aviso"]).dropna()

    estados_total = {"activo", "pausado"} if incluir_pausados else {"activo"}

    filas = []
    for fecha, dia in hist.groupby("fecha", sort=True):
        ids_hoy = dia.loc[dia["estado_publicacion"].isin(estados_total), "id_aviso"]
        valores = valor_m2_por_id.reindex(ids_hoy).dropna()
        if valores.empty:
            continue
        filas.append({
            "fecha": fecha,
            "media": valores.mean(),
            "mediana": valores.median(),
            "desviacion_estandar": valores.std(),
        })

    return pd.DataFrame(filas)


def _grafico_total(serie: pd.DataFrame) -> alt.Chart:
    return (
        alt.Chart(serie)
        .mark_line(point=alt.OverlayMarkDef(size=55, filled=True), strokeWidth=2, color=COLOR_PRIMARY)
        .encode(
            # :O (ordinal), no :T (temporal continuo): son snapshots diarios
            # discretos, uno por día - con escala temporal continua, Vega-Lite
            # generaba varios ticks por día (subdivisiones de hora) en vez de
            # uno solo por fecha. formatType="time" mantiene el label "23 Jul"
            # aunque el campo ya no sea temporal.
            x=alt.X("fecha:O", title=None, axis=alt.Axis(format="%d %b", formatType="time")),
            y=alt.Y("total:Q", title="Avisos disponibles", scale=alt.Scale(zero=False)),
            tooltip=[
                alt.Tooltip("fecha:O", title="Día", format="%d/%m/%Y", formatType="time"),
                alt.Tooltip("total:Q", title="Avisos disponibles"),
            ],
        )
        .properties(height=220)
    )


def _grafico_valor_m2(serie_m2: pd.DataFrame) -> alt.Chart:
    largo = serie_m2.melt(
        id_vars="fecha", value_vars=["media", "mediana", "desviacion_estandar"],
        var_name="metrica", value_name="valor",
    )
    largo["metrica"] = largo["metrica"].map(
        {"media": "Media", "mediana": "Mediana", "desviacion_estandar": "Desviación estándar"}
    )

    return (
        alt.Chart(largo)
        .mark_line(point=alt.OverlayMarkDef(size=55, filled=True), strokeWidth=2)
        .encode(
            x=alt.X("fecha:O", title=None, axis=alt.Axis(format="%d %b", formatType="time")),
            y=alt.Y("valor:Q", title="$/m² útil", scale=alt.Scale(zero=False)),
            color=alt.Color(
                "metrica:N",
                scale=alt.Scale(
                    domain=["Media", "Mediana", "Desviación estándar"],
                    range=[COLOR_PRIMARY, COLOR_SERIE_2, COLOR_SERIE_3],
                ),
                legend=alt.Legend(orient="top", direction="horizontal", title=None),
            ),
            tooltip=[
                alt.Tooltip("fecha:O", title="Día", format="%d/%m/%Y", formatType="time"),
                alt.Tooltip("metrica:N", title="Métrica"),
                alt.Tooltip("valor:Q", title="$/m²", format=",.0f"),
            ],
        )
        .properties(height=220)
    )


def _grafico_entran_salen(serie: pd.DataFrame) -> alt.Chart:
    datos = serie.dropna(subset=["entran", "salen"])
    largo = pd.concat(
        [
            pd.DataFrame(
                {"fecha": datos["fecha"], "tipo": "Entran", "valor": datos["entran"], "valor_abs": datos["entran"]}
            ),
            pd.DataFrame(
                {"fecha": datos["fecha"], "tipo": "Salen", "valor": -datos["salen"], "valor_abs": datos["salen"]}
            ),
        ],
        ignore_index=True,
    )

    return (
        alt.Chart(largo)
        .mark_bar(size=18)
        .encode(
            x=alt.X("fecha:O", title=None, axis=alt.Axis(format="%d %b", formatType="time")),
            y=alt.Y("valor:Q", title="Avisos"),
            color=alt.Color(
                "tipo:N",
                scale=alt.Scale(domain=["Entran", "Salen"], range=[COLOR_OPORTUNIDAD, COLOR_CARO]),
                legend=alt.Legend(orient="top", direction="horizontal", title=None),
            ),
            tooltip=[
                alt.Tooltip("fecha:O", title="Día", format="%d/%m/%Y", formatType="time"),
                alt.Tooltip("tipo:N", title="Tipo"),
                alt.Tooltip("valor_abs:Q", title="Avisos"),
            ],
        )
        .properties(height=220)
    )


def render() -> None:
    st.markdown(
        '<h1 class="app-title">Estadísticas diarias de avisos</h1>'
        f'<p class="subtitle">Avisos que entran (verde) y salen (rojo) del estado activo día a día '
        f'&mdash; salen incluye tanto pausados como eliminados/finalizados &mdash; '
        f'con los mismos filtros del buscador. Historial disponible desde el {FECHA_INICIO_HISTORIAL}.</p>',
        unsafe_allow_html=True,
    )

    historial = load_historial_diario()
    if historial.empty:
        st.info(f"Todavía no hay historial diario registrado. Se empezó a capturar el {FECHA_INICIO_HISTORIAL}.")
        return

    df_todos = load_data_todos_los_estados()
    if df_todos.empty:
        st.warning("No hay avisos con predicción disponible en la base de datos todavía.")
        return

    ids_filtrados = set(apply_atributo_filters(df_todos)["id_aviso"])
    incluir_pausados = st.session_state.get("f_incluir_pausados", False)

    serie = _serie_diaria(historial, ids_filtrados, incluir_pausados)
    if serie.empty:
        st.info("Ningún aviso del historial coincide con los filtros seleccionados.")
        return

    with st.container(key="chart-card-total"):
        st.markdown("**Avisos disponibles**")
        st.altair_chart(_grafico_total(serie), use_container_width=True)

    with st.container(key="chart-card-entran-salen"):
        st.markdown("**Entradas y salidas del estado activo**")
        if serie["entran"].notna().sum() == 0:
            st.info(
                "Todavía hay un solo día de historial con estos filtros: las entradas y salidas se "
                "podrán ver a partir de mañana."
            )
        else:
            st.altair_chart(_grafico_entran_salen(serie), use_container_width=True)

    serie_m2 = _serie_valor_m2(historial, df_todos, ids_filtrados, incluir_pausados)
    if not serie_m2.empty:
        with st.container(key="chart-card-m2"):
            st.markdown("**Valor por m² útil (CLP)**")
            st.altair_chart(_grafico_valor_m2(serie_m2), use_container_width=True)
