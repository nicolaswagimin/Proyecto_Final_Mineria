"""Panel de campañas: sidebar en degradado, KPI cards, paneles y gráficos."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from . import analytics, config, experiment, theme, ui
from .data import load_dataset
from .modeling import MODEL_LABELS, load_predictions, load_report, report_is_compatible

TEMPLATE = theme.TEMPLATE_NAME
SECTIONS = ["Resumen", "Segmentos", "Modelos", "Recomendaciones", "ETL"]
ICONS = {"Resumen": "dashboard", "Segmentos": "group", "Modelos": "insights",
         "Recomendaciones": "lightbulb", "ETL": "account_tree"}


# ------------------------------------------------------------------ formato --
def _es(text: str) -> str:
    return text.replace(",", "\x00").replace(".", ",").replace("\x00", ".")


def money(value: float | None, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "No calculable"
    return _es(f"{value:,.{digits}f}") + " u.m."


def number(value: float | None, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "No calculable"
    return _es(f"{value:,.{digits}f}")


def pct(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "No calculable"
    digits = 3 if abs(value) < 0.1 else 2
    return _es(f"{value:.{digits}f}") + " %"


def integer(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "0"
    return _es(f"{int(round(value)):,}")


def pct_num(value: float | None, digits: int = 0) -> str:
    if value is None or pd.isna(value):
        return "No calculable"
    return _es(f"{value * 100:.{digits}f}") + " %"


def chart(fig) -> None:
    fig.update_layout(separators=",.")
    st.plotly_chart(fig, width="stretch")


# -------------------------------------------------------------------- carga --
@st.cache_data(show_spinner=False)
def _load_cached(path: str, mtime: float, size: int):
    df, quality = load_dataset(Path(path))
    return df, quality.to_dict()


def load_everything():
    path = config.RAW_CSV
    if not path.exists():
        return None, None
    stat = path.stat()
    try:
        df, quality = _load_cached(str(path), stat.st_mtime, stat.st_size)
    except Exception as exc:
        st.error(f"No se pudo leer el CSV: {exc}")
        return None, None
    return df, quality


# ---------------------------------------------------------------- navegación --
def sidebar(df: pd.DataFrame) -> tuple[str, str, dict, analytics.EvidenceThreshold]:
    with st.sidebar:
        section = st.radio(
            "Sección", SECTIONS,
            format_func=lambda s: f":material/{ICONS[s]}: {s}",
            label_visibility="collapsed",
        )
        st.markdown("### Filtros")
        campaigns = sorted(df["campaign_id"].dropna().unique())
        sel_campaigns = st.multiselect(
            "Campaña", options=campaigns, default=list(campaigns),
            format_func=lambda c: f"Campaña {int(c)}",
        )
        ages = sorted(df["age"].dropna().unique())
        sel_ages = st.multiselect("Edad", options=ages, default=list(ages))
        genders = sorted(df["gender"].dropna().unique())
        sel_genders = st.multiselect("Género", options=genders, default=list(genders))
        interests = sorted(df["interest1"].dropna().unique())
        sel_interests = st.multiselect(
            "Interés", options=interests, default=list(interests),
            format_func=lambda i: str(int(i)),
        )
        if st.button("Restablecer filtros", use_container_width=True):
            st.rerun()

        st.markdown("### Evidencia")
        min_ads = st.slider("Anuncios mínimos", 1, 10, config.EVIDENCE_MIN_ADS)
        min_approved = st.slider("Conversiones aprobadas", 1, 10, config.EVIDENCE_MIN_APPROVED)
        improvement_pct = st.slider(
            "Mejora mínima", 0, 100, int(round(config.EVIDENCE_MIN_IMPROVEMENT * 100)),
            step=5, format="%d%%",
        )
        spend_pct = st.slider(
            "Cuota de gasto", 0, 25, int(round(config.EVIDENCE_MIN_SPEND_RATIO * 100)),
            step=1, format="%d%%",
        )

    filters = {
        "campaigns": sel_campaigns, "ages": sel_ages,
        "genders": sel_genders, "interests": sel_interests,
    }
    threshold = analytics.EvidenceThreshold(
        min_ads=min_ads, min_approved=min_approved,
        min_improvement=improvement_pct / 100, min_spend_ratio=spend_pct / 100,
    )
    search = st.text_input(
        "Buscar", placeholder="Buscar campaña, edad, género o interés…",
        label_visibility="collapsed",
    )
    return section, search, filters, threshold


def sidebar_status(quality: dict, report: dict | None) -> None:
    with st.sidebar:
        st.markdown("### Datos")
        estado = (
            "Modelos entrenados" if report and report_is_compatible(report, quality)
            else "Sin resultados compatibles"
        )
        st.markdown(
            ui.group(
                [
                    ("Registros válidos", f"<strong>{integer(quality['n_rows_clean'])}</strong>"),
                    ("En el archivo", integer(quality["n_rows_raw"])),
                    ("Huella", f"<code>{quality['dataset_id']}</code>"),
                    ("Estado", estado),
                ]
            ),
            unsafe_allow_html=True,
        )


# ------------------------------------------------------------------- vistas --
def view_summary(df: pd.DataFrame) -> None:
    m = analytics.totals(df)
    st.markdown(
        ui.toolbar("Resumen", "Gasto, volumen y eficiencia del filtro actual"),
        unsafe_allow_html=True,
    )
    st.markdown(
        ui.kpis(
            [
                {"label": "Gasto", "value": money(m.spend, 0), "note": f"{integer(m.n_ads)} anuncios"},
                {"label": "Conversiones aprobadas", "value": integer(m.approved_conversions),
                 "note": f"{integer(m.total_conversions)} registradas"},
                {"label": "Coste por conversión", "value": money(m.cost_per_approved), "note": "por aprobada"},
            ],
            highlight=1,
        ),
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns([2, 1])
    with c1:
        fechas = df["reporting_start"].dropna()
        if not fechas.empty:
            by_day = (
                df.groupby(df["reporting_start"].dt.date)
                .agg(spend=("spend", "sum"), approved=("approved_conversions", "sum"))
                .reset_index()
            )
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(
                go.Scatter(
                    x=by_day["reporting_start"], y=by_day["spend"], name="Gasto",
                    fill="tozeroy", fillgradient=theme.AREA_CYAN,
                    line=dict(color=config.DASH_CYAN, width=2),
                ),
                secondary_y=False,
            )
            fig.add_trace(
                go.Scatter(
                    x=by_day["reporting_start"], y=by_day["approved"], name="Aprobadas",
                    fill="tozeroy", fillgradient=theme.AREA_GREEN,
                    line=dict(color=config.DASH_GREEN, width=2),
                ),
                secondary_y=True,
            )
            fig.update_layout(
                height=300, template=TEMPLATE, margin=dict(t=44, l=4, r=4, b=4),
                legend=dict(orientation="h", y=-0.2), hovermode="x unified",
            )
            fig.update_yaxes(title_text=None, secondary_y=False, tickformat="d")
            fig.update_yaxes(title_text=None, secondary_y=True, showgrid=False, tickformat="d")
            fig.update_xaxes(tickformat="%d/%m")
            fig.update_layout(title=dict(text="Actividad por día", x=0.01, font=dict(size=12)))
            chart(fig)

    with c2:
        rate = 100 * m.approved_conversions / m.total_conversions if m.total_conversions else 0
        donut = go.Figure(
            go.Pie(
                values=[m.approved_conversions, max(m.total_conversions - m.approved_conversions, 0)],
                hole=0.72, sort=False, textinfo="none",
                marker=dict(colors=[config.DASH_GREEN, "rgba(255,255,255,0.08)"], line=dict(width=0)),
            )
        )
        donut.add_annotation(
            text=f"<b>{rate:.0f}%</b>", showarrow=False,
            font=dict(size=26, color=config.DASH_TEXT),
        )
        donut.update_layout(height=300, template=TEMPLATE, showlegend=False,
                            margin=dict(t=44, l=10, r=10, b=10))
        donut.update_layout(title=dict(text="Tasa de aprobación", x=0.02, font=dict(size=12)))
        chart(donut)

    c3, c4 = st.columns([2, 1])
    with c3:
        seg = analytics.segment_summary(df).dropna(subset=["approved_conversions"])
        top = seg.sort_values("approved_conversions", ascending=False).head(4)
        total_appr = max(m.approved_conversions, 1)
        rows = [
            (
                f"Campaña {int(r.campaign_id)} · {r.age} · {r.gender} · int. {int(r.interest1)}",
                f"{integer(r.approved_conversions)} aprobadas",
                100 * r.approved_conversions / total_appr,
            )
            for r in top.itertuples()
        ]
        st.markdown(ui.panel("Segmentos con más conversiones", ui.bars(rows), "top 4 del filtro"),
                    unsafe_allow_html=True)
    with c4:
        fechas = df["reporting_start"].dropna()
        if not fechas.empty:
            first = fechas.min()
            active = {d.day for d in fechas.dt.date.unique() if d.month == first.month}
            st.markdown(ui.panel("Periodo", ui.month_grid(first.year, first.month, active)),
                        unsafe_allow_html=True)

    table = analytics.campaign_summary(df)
    if not table.empty:
        rows = [
            [f"Campaña {int(r.campaign_id)}", integer(r.n_ads), money(r.spend),
             integer(r.clicks), integer(r.approved_conversions), pct(r.ctr), money(r.cost_per_approved)]
            for r in table.itertuples()
        ]
        st.markdown(
            ui.panel(
                "Campañas",
                ui.table(["Campaña", "Anuncios", "Gasto", "Clics", "Aprobadas", "CTR", "Coste/conv."],
                         rows, numeric={1, 2, 3, 4, 5, 6}),
            ),
            unsafe_allow_html=True,
        )


def view_segments(df: pd.DataFrame) -> None:
    st.markdown(ui.toolbar("Segmentos", "Eficiencia dentro de una campaña"), unsafe_allow_html=True)
    campaigns = sorted(df["campaign_id"].dropna().unique())
    c1, c2 = st.columns([1, 2])
    with c1:
        selected = st.selectbox("Campaña", campaigns, format_func=lambda c: f"Campaña {int(c)}",
                                key="segment_campaign")
    with c2:
        dims = st.multiselect(
            "Segmentar por", ["age", "gender", "interest1"],
            default=["age", "gender", "interest1"],
            format_func=lambda d: {"age": "Edad", "gender": "Género", "interest1": "Interés"}[d],
            key="segment_dims",
        )
    subset = df[df["campaign_id"] == selected]
    if subset.empty or not dims:
        st.warning("Sin registros o sin dimensiones seleccionadas.")
        return

    camp = analytics.totals(subset)
    st.markdown(
        ui.kpis(
            [
                {"label": "Gasto", "value": money(camp.spend, 0), "note": f"{integer(camp.n_ads)} anuncios"},
                {"label": "Aprobadas", "value": integer(camp.approved_conversions),
                 "note": f"{integer(camp.total_conversions)} registradas"},
                {"label": "Coste por conversión", "value": money(camp.cost_per_approved), "note": "referencia"},
            ],
            highlight=2,
        ),
        unsafe_allow_html=True,
    )

    table = analytics.segment_summary(subset, dims=dims)
    if table.empty:
        st.warning("No hay segmentos comparables.")
        return
    zero = table[table["sin_conversiones_aprobadas"]]
    if not zero.empty:
        st.warning(f"{len(zero)} segmento(s) con gasto y cero conversiones aprobadas.")
    rows = []
    for r in table.itertuples():
        estado = ui.badge("sin aprobadas", "warn") if r.sin_conversiones_aprobadas else ui.badge("con evidencia", "ok")
        rows.append([r.segmento, integer(r.n_ads), money(r.spend), integer(r.total_conversions),
                     integer(r.approved_conversions), pct(r.ctr), money(r.cost_per_approved), estado])
    st.markdown(
        ui.panel(
            f"Campaña {int(selected)} · {len(table)} segmentos",
            ui.table(["Segmento", "Anuncios", "Gasto", "Conv.", "Aprobadas", "CTR", "Coste/conv.", "Estado"],
                     rows, numeric={1, 2, 3, 4, 5, 6}),
        ),
        unsafe_allow_html=True,
    )

    valid = table.dropna(subset=["cost_per_approved"]).sort_values("cost_per_approved").head(15)
    if not valid.empty:
        fig = px.bar(valid, x="cost_per_approved", y="segmento", orientation="h",
                     labels={"cost_per_approved": "", "segmento": ""}, template=TEMPLATE)
        fig.update_traces(marker_color=config.DASH_CYAN)
        fig.update_layout(height=420, yaxis={"categoryorder": "total ascending"},
                          margin=dict(t=10, l=4, r=4, b=4))
        st.markdown(ui.panel("15 segmentos más eficientes", ""), unsafe_allow_html=True)
        chart(fig)


def _metrics_rows(report: dict) -> list[list[str]]:
    return [
        [config.TARGET_LABELS.get(m["objetivo"], m["objetivo"]),
         MODEL_LABELS.get(m["modelo"], m["modelo"]), integer(m["n"]),
         number(m["mae"], 3), number(m["rmse"], 3), number(m.get("r2") or 0, 2)]
        for m in report.get("metricas", [])
    ]


def view_models(df: pd.DataFrame, quality: dict, report: dict | None, predictions) -> None:
    st.markdown(ui.toolbar("Modelos", "Error fuera de muestra"), unsafe_allow_html=True)
    if report is None:
        st.warning("Todavía no hay resultados de entrenamiento.")
        st.code("python -m campaign_optimizer.train", language="bash")
        return
    if not report_is_compatible(report, quality):
        st.warning("Los resultados pertenecen a otra versión del CSV. Vuelve a entrenar.")
        return

    metricas = {(m["objetivo"], m["modelo"]): m for m in report.get("metricas", [])}
    rf = metricas.get(("conversiones_aprobadas", "random_forest"), {})
    dummy = metricas.get(("conversiones_aprobadas", "dummy"), {})
    reduccion = (1 - rf["mae"] / dummy["mae"]) if rf.get("mae") and dummy.get("mae") else None
    st.markdown(
        ui.kpis(
            [
                {"label": "MAE Random Forest", "value": number(rf.get("mae"), 3), "note": "conversiones/registro"},
                {"label": "MAE referencia", "value": number(dummy.get("mae"), 3), "note": "media"},
                {"label": "Reducción", "value": pct_num(reduccion), "note": f"R² {number(rf.get('r2') or 0, 2)}"},
            ],
            highlight=2,
        ),
        unsafe_allow_html=True,
    )
    st.markdown(
        ui.panel("Error por objetivo y modelo",
                 ui.table(["Objetivo", "Modelo", "n", "MAE", "RMSE", "R²"],
                          _metrics_rows(report), numeric={2, 3, 4, 5}),
                 f"partición {report['particion']}"),
        unsafe_allow_html=True,
    )

    table = pd.DataFrame(report.get("metricas", []))
    if not table.empty:
        plot_df = table.copy()
        plot_df["objetivo"] = plot_df["objetivo"].map(config.TARGET_LABELS).fillna(plot_df["objetivo"])
        plot_df["modelo"] = plot_df["modelo"].map(MODEL_LABELS).fillna(plot_df["modelo"])
        fig = px.bar(plot_df, x="objetivo", y="mae", color="modelo", barmode="group",
                     labels={"objetivo": "", "mae": "", "modelo": ""}, template=TEMPLATE)
        fig.update_layout(height=300, margin=dict(t=10), legend=dict(orientation="h", y=-0.2))
        chart(fig)

    if predictions is not None and not predictions.empty:
        targets = list(predictions["objetivo"].unique())
        cols = st.columns(len(targets))
        for col, target in zip(cols, targets):
            with col:
                key = st.radio("Modelo", ["random_forest", "adaboost", "dummy"],
                               format_func=lambda k: MODEL_LABELS[k], key=f"pick_{target}",
                               horizontal=True, label_visibility="collapsed")
                sub = predictions[predictions["objetivo"] == target]
                fig = px.scatter(sub, x="observed", y=f"pred_{key}",
                                 labels={"observed": "", f"pred_{key}": ""}, template=TEMPLATE,
                                 opacity=0.7)
                lim = max(sub["observed"].max(), sub[f"pred_{key}"].max(), 1.0) * 1.05
                fig.add_shape(type="line", x0=0, y0=0, x1=lim, y1=lim,
                              line=dict(color=config.RC_MUTE, dash="dash"))
                fig.update_layout(height=280, title=config.TARGET_LABELS.get(target, target),
                                  showlegend=False, margin=dict(t=30, l=4, r=4, b=4))
                chart(fig)


def view_recommendations(df: pd.DataFrame, threshold: analytics.EvidenceThreshold, quality: dict) -> None:
    st.markdown(ui.toolbar("Recomendaciones", "Qué segmento investigar y con cuánta evidencia"),
                unsafe_allow_html=True)
    recs = analytics.recommend(df, threshold)
    if not recs:
        st.warning("Evidencia insuficiente: ningún segmento cumple el umbral.")
        motivos = analytics.discarded_segment_reasons(df, threshold)
        if not motivos.empty:
            st.markdown(ui.panel("Motivos de descarte",
                                 ui.table(["Motivo", "Segmentos"],
                                          [[r.motivo, integer(r.segmentos)] for r in motivos.itertuples()],
                                          numeric={1})), unsafe_allow_html=True)
    else:
        for rec in recs:
            st.markdown(
                ui.panel(
                    f"Campaña {rec.campaign_id} · {rec.segment_label}",
                    ui.group([
                        ("Coste por conversión", f"<strong>{money(rec.segment.cost_per_approved)}</strong>"),
                        ("Resto de la campaña", money(rec.reference.cost_per_approved)),
                        ("Diferencia histórica", pct_num(rec.improvement)),
                        ("Evidencia", f"{integer(rec.segment.approved_conversions)} aprobadas · "
                                      f"{integer(rec.segment.n_ads)} anuncios"),
                    ]),
                    rec.action,
                ),
                unsafe_allow_html=True,
            )

    top = recs[0] if recs else next(iter(analytics.recommend(df, threshold, include_weak=True)), None)
    if top is not None:
        st.markdown("## Prueba A/B")
        window = 14
        fechas = quality.get("date_range") or {}
        if fechas.get("min") and fechas.get("max"):
            window = (pd.Timestamp(fechas["max"]) - pd.Timestamp(fechas["min"])).days + 1
        mde = st.selectbox("Efecto mínimo a detectar", [0.25, 0.40, 0.50, 0.75],
                           format_func=lambda v: pct_num(v), index=0)
        plan = experiment.plan_ab_test(top.segment, top.campaign_id, top.segment_label, window, delta=mde)
        st.markdown(
            ui.kpis(
                [
                    {"label": "Aprobadas por brazo", "value": integer(plan.n_per_arm), "note": "α 0,05 · potencia 80 %"},
                    {"label": "Días de calendario", "value": integer(plan.calendar_days_ceiled),
                     "note": f"{number(plan.arm_days, 2)} de exposición × 2"},
                    {"label": "Presupuesto", "value": money(plan.test_budget, 0), "note": "control + variante 1,25"},
                ],
                highlight=1,
            ),
            unsafe_allow_html=True,
        )
        st.markdown(
            ui.panel("Escenarios",
                     ui.table(["Mejora", "Aprobadas/brazo", "Exposición", "Calendario", "Presupuesto"],
                              [[pct_num(s["delta"]), integer(s["n_por_brazo"]),
                                number(s["dias_exposicion"], 1), integer(s["dias_calendario_plan"]),
                                money(s["presupuesto"], 0)] for s in plan.scenarios],
                              numeric={1, 2, 3, 4}),
                     "calendario redondeado hacia arriba"),
            unsafe_allow_html=True,
        )

    share = analytics.conversion_share_by_campaign(df)
    if not share.empty:
        st.markdown(
            ui.panel("Atribución por campaña",
                     ui.table(["Campaña", "Conv.", "Aprobadas", "Cuota"],
                              [[f"Campaña {int(r.campaign_id)}", integer(r.total_conversions),
                                integer(r.approved_conversions), pct(r.cuota_conversiones_aprobadas)]
                               for r in share.itertuples()], numeric={1, 2, 3})),
            unsafe_allow_html=True,
        )
    st.caption(
        "Facebook es el único canal del archivo: la comparación es entre sus campañas. Sin "
        "recorridos de usuario no hay atribución de primer/último contacto; sin ingresos no hay ROI."
    )


def view_etl(df: pd.DataFrame, quality: dict, report: dict | None) -> None:
    """Fases del ETL, decisiones de limpieza y tratamiento de datos incompletos."""
    raw = int(quality["n_rows_raw"])
    clean = int(quality["n_rows_clean"])
    descartadas = raw - clean
    st.markdown(
        ui.toolbar(
            "ETL",
            "Extracción, transformación y carga: qué entró, qué se descartó y por qué",
        ),
        unsafe_allow_html=True,
    )
    st.markdown(
        ui.kpis(
            [
                {"label": "Filas extraídas", "value": integer(raw), "note": "CSV original intacto"},
                {"label": "Filas cargadas al modelo", "value": integer(clean),
                 "note": f"{pct_num(clean / raw, 1)} retenido"},
                {"label": "Filas descartadas", "value": integer(descartadas),
                 "note": "desalineación estructural"},
            ],
            highlight=1,
        ),
        unsafe_allow_html=True,
    )

    st.markdown(
        ui.panel(
            "Fases",
            ui.table(
                ["Fase", "Acción", "Filas", "Resultado"],
                [
                    ["E — Extract", "Descarga de Kaggle (.zip) y lectura del CSV", integer(raw),
                     "15 columnas · sha256 registrado"],
                    ["T — Transform", "Normalización de esquema y tipado", integer(raw),
                     "numéricos, fechas y texto"],
                    ["T — Transform", "Validación estructural por dominio", integer(quality["desalineadas"]),
                     "excluidas: columnas desplazadas"],
                    ["T — Transform", "Duplicados exactos y valores negativos", "0",
                     "sin cambios"],
                    ["L — Load", "Artefactos y dataset modelable", integer(clean),
                     "predictions · report · quality"],
                ],
            ),
            "reproducible con python -m campaign_optimizer.train",
        ),
        unsafe_allow_html=True,
    )

    decision_rows = [
        ["Duplicados exactos", "misma fila repetida", integer(quality["duplicate_rows"]), "se conserva la primera"],
        ["Desalineación de columnas", "≥2 violaciones de dominio", integer(quality["desalineadas"]),
         "excluidas: sin campaña atribuible"],
        ["Nulos en el objetivo", "conversiones vacías", integer(quality["dropped"].get("nulos_en_objetivos", 0)),
         "coinciden con las desalineadas"],
        ["Valores negativos", "gasto/impresiones/clics < 0", integer(sum(quality["negatives"].values())),
         "excluidas"],
        ["Aprobadas > totales", "inconsistencia contable", integer(quality["approved_gt_total"]),
         "conservadas marcadas"],
        ["Conversiones sin clics", "posible view-through", integer(quality["conversions_without_clicks"]),
         "conservadas marcadas"],
        ["Gasto cero", "anuncios sin inversión", integer(quality["zero_spend_rows"]),
         "conservadas: no son error"],
    ]
    st.markdown(
        ui.panel(
            "Decisiones de transformación",
            ui.table(["Paso", "Criterio", "Filas", "Acción"], decision_rows, numeric={2}),
            "cada decisión queda contada en artifacts/data_quality.json",
        ),
        unsafe_allow_html=True,
    )

    cobertura = (
        df.groupby("campaign_id").size().rename("filas").reset_index()
        .assign(cuota=lambda d: 100 * d["filas"] / d["filas"].sum())
    )
    st.markdown(
        ui.panel(
            "Cobertura del dataset modelable",
            ui.table(
                ["Campaña", "Filas", "Cuota"],
                [[f"Campaña {int(r.campaign_id)}", integer(r.filas), pct(r.cuota)]
                 for r in cobertura.itertuples()],
                numeric={1, 2},
            ),
            f"{df['campaign_id'].nunique()} campañas · {integer(len(df))} filas",
        ),
        unsafe_allow_html=True,
    )

    st.markdown(
        ui.panel(
            "Datos incompletos: mecanismo y decisión",
            f"""
            <p>Los {integer(descartadas)} registros descartados no faltan al azar: su estructura
            está desplazada y pierden <code>campaign_id</code> y <code>fb_campaign_id</code>.
            En términos de minería de datos el mecanismo es <strong>MNAR estructural</strong>
            (no aleatorio): no depende del valor de una variable, sino de cómo se generó el
            archivo.</p>
            <p><strong>No se imputa.</strong> Imputar la campaña con la moda o con un modelo
            fabricaría atribución: los totales por campaña dejarían de ser un hecho del archivo.
            Se aplica <strong>análisis de casos completos</strong> sobre los datos atribuibles y se
            documenta el <strong>sesgo de selección</strong> que introduce: el panel describe
            {integer(clean)} registros, no los {integer(raw)} del archivo.</p>
            <p>Las conversiones sin clics ({integer(quality["conversions_without_clicks"])}) sí se
            conservan marcadas: descartarlas eliminaría gasto real y convertiría un problema de
            medición en una pérdida de datos.</p>
            """,
            "decisión documentada, no silenciosa",
        ),
        unsafe_allow_html=True,
    )

    st.markdown(
        ui.panel(
            "Trazabilidad",
            ui.group(
                [
                    ("SHA-256 del CSV", f"<code>{quality['source_sha256'][:24]}…</code>"),
                    ("Huella del conjunto", f"<code>{quality['dataset_id']}</code>"),
                    ("Ventana de datos", f"{quality['date_range'].get('min', 'n/d')} → "
                                         f"{quality['date_range'].get('max', 'n/d')}"),
                    ("Modelos", "entrenados" if report else "pendientes"),
                    ("Reproducir", "<code>python -m campaign_optimizer.train</code>"),
                ]
            ),
        ),
        unsafe_allow_html=True,
    )

# --------------------------------------------------------------------- main --
def main() -> None:
    st.set_page_config(page_title="Campañas", page_icon=":material/bar_chart:", layout="wide")
    theme.apply()

    df, quality = load_everything()
    if df is None:
        st.error(f"Falta el CSV de Kaggle en `{config.RAW_CSV}`.")
        return
    report = load_report()
    predictions = load_predictions()

    section, search, filters, threshold = sidebar(df)
    sidebar_status(quality, report)

    filtered = analytics.filter_records(df, **filters)
    if search.strip():
        needle = search.strip().lower()
        mask = (
            filtered["campaign_id"].astype(str).str.contains(needle, na=False)
            | filtered["age"].str.lower().str.contains(needle, na=False)
            | filtered["gender"].str.lower().str.contains(needle, na=False)
            | filtered["interest1"].astype(str).str.contains(needle, na=False)
        )
        filtered = filtered[mask]

    if filtered.empty:
        st.warning("No hay registros con los filtros actuales.")
        return

    if section == "Resumen":
        view_summary(filtered)
    elif section == "Segmentos":
        view_segments(filtered)
    elif section == "Modelos":
        view_models(filtered, quality, report, predictions)
    elif section == "Recomendaciones":
        view_recommendations(filtered, threshold, quality)
    else:
        view_etl(df, quality, report)


if __name__ == "__main__":
    main()
