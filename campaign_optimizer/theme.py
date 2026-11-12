"""Tema del panel: pizarra oscura, sidebar en degradado y acentos cian/verde.

Sigue la referencia visual aportada: sidebar con degradado teal→verde, lienzo
pizarra, paneles planos con hairline, tarjetas KPI (una en degradado), áreas con
relleno degradado, donut, barras de progreso y calendario.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import plotly.graph_objects as go
import plotly.io as pio
import yaml

from . import config

FONT_TEXT = '"Inter", "Segoe UI Variable Text", "Segoe UI", system-ui, sans-serif'
FONT_DISPLAY = '"Inter", "Segoe UI Variable Display", "Segoe UI", system-ui, sans-serif'
FEATURES = '"calt", "kern", "liga", "ss03"'

TEMPLATE_NAME = "design_md"


@dataclass(frozen=True)
class Theme:
    slug: str
    name: str
    bg: str
    panel: str
    panel2: str
    card: str
    line: str
    text: str
    body: str
    mute: str
    cyan: str
    green: str
    red: str
    yellow: str
    sidebar_top: str
    sidebar_bottom: str

    def css(self) -> str:
        return f"""
<style>
@import url("https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap");
:root {{
  --d-bg: {self.bg};
  --d-panel: {self.panel};
  --d-panel2: {self.panel2};
  --d-card: {self.card};
  --d-line: {self.line};
  --d-text: {self.text};
  --d-body: {self.body};
  --d-mute: {self.mute};
  --d-cyan: {self.cyan};
  --d-green: {self.green};
  --d-red: {self.red};
  --d-yellow: {self.yellow};
  --d-grad: linear-gradient(135deg, {self.cyan} 0%, {self.green} 100%);
  --d-sidebar: linear-gradient(180deg, {self.sidebar_top} 0%, {self.sidebar_bottom} 100%);
}}

/* ============================================================== base ==== */
.stApp {{
  background: var(--d-bg);
  color: var(--d-body);
  font-family: {FONT_TEXT};
  font-size: 15px;
  line-height: 1.55;
  font-feature-settings: {FEATURES};
}}
.stApp p, .stApp li, .stApp label, .stApp input, .stApp textarea, .stApp select,
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp button {{
  font-family: {FONT_TEXT}; font-feature-settings: {FEATURES};
}}
::selection {{ background: rgba(62, 214, 196, 0.35); }}
.block-container {{ max-width: 1280px; padding-top: 14px !important; padding-bottom: 3rem; }}

/* ======================================================= barra superior == */
header[data-testid="stHeader"] {{
  background: #26313A !important;
  height: 56px; min-height: 56px;
  border-bottom: 1px solid var(--d-line);
  position: relative;
}}
header[data-testid="stHeader"]::after {{
  content: "CAMPAÑAS";
  position: absolute; left: 58px; top: 0;
  font-size: 15px; font-weight: 700; letter-spacing: 0.14em;
  line-height: 56px; color: var(--d-text);
  pointer-events: none;
}}
#MainMenu, footer {{ visibility: hidden; }}
div[data-testid="stToolbarActions"],
div[data-testid="stAppDeployButton"],
div[data-testid="stStatusWidget"] {{ display: none !important; }}
button[data-testid="stExpandSidebarButton"] {{
  background: rgba(255,255,255,0.12) !important;
  border: 0 !important; border-radius: 4px !important;
  width: 28px; height: 28px; margin: 14px 0 0 12px;
  color: #ffffff !important;
}}
button[data-testid="stExpandSidebarButton"] span {{ color: #ffffff !important; }}

/* ======================================================== sidebar ======= */
section[data-testid="stSidebar"] {{
  background: var(--d-sidebar);
  border-right: 1px solid rgba(255,255,255,0.10);
}}
section[data-testid="stSidebar"] > div {{ padding-top: 10px; }}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {{
  font-size: 11px !important; font-weight: 600 !important;
  letter-spacing: 0.12em; text-transform: uppercase;
  color: rgba(255,255,255,0.72) !important; margin: 16px 0 4px !important;
}}
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] label p,
section[data-testid="stSidebar"] span {{ color: #ffffff !important; }}
section[data-testid="stSidebar"] hr {{ border: 0; border-top: 1px solid rgba(255,255,255,0.16); margin: 12px 0; }}

/* Navegación: icono + etiqueta, fila activa con velo blanco. */
section[data-testid="stSidebar"] div[role="radiogroup"] {{ gap: 2px; }}
section[data-testid="stSidebar"] label[data-testid="stRadioOption"] {{
  display: flex; align-items: center; gap: 12px;
  padding: 10px 12px;
  border-radius: 4px;
  cursor: pointer;
  transition: background 140ms ease;
}}
section[data-testid="stSidebar"] label[data-testid="stRadioOption"]:hover {{
  background: rgba(255,255,255,0.10);
}}
section[data-testid="stSidebar"] label[data-testid="stRadioOption"][data-selected="true"] {{
  background: rgba(255,255,255,0.18);
}}
section[data-testid="stSidebar"] label[data-testid="stRadioOption"] > div > div:first-child {{
  display: none !important;
}}
section[data-testid="stSidebar"] label[data-testid="stRadioOption"] p {{
  font-size: 14px !important; font-weight: 400 !important; color: #ffffff !important;
}}
section[data-testid="stSidebar"] label[data-testid="stRadioOption"][data-selected="true"] p {{
  font-weight: 600 !important;
}}
section[data-testid="stSidebar"] label[data-testid="stRadioOption"] [data-testid="stIconMaterial"] {{
  color: #ffffff !important; font-size: 20px;
}}

/* Filtros: tarjeta translúcida oscura sobre el degradado. */
section[data-testid="stSidebar"] div[data-baseweb="select"] > div,
section[data-testid="stSidebar"] div[data-baseweb="input"] > div {{
  background: rgba(0,0,0,0.22) !important;
  border: 1px solid rgba(255,255,255,0.16) !important;
  border-radius: 4px !important;
  min-height: 34px;
}}
section[data-testid="stSidebar"] [data-testid="stMultiSelectTagsContainer"] span span {{
  background: rgba(0,0,0,0.28) !important;
  border: 1px solid rgba(255,255,255,0.18) !important;
  border-radius: 4px !important;
  color: #ffffff !important;
}}
section[data-testid="stSidebar"] .stButton > button {{
  background: rgba(0,0,0,0.22) !important;
  border: 1px solid rgba(255,255,255,0.20) !important;
  color: #ffffff !important;
}}

/* ====================================================== tarjetas KPI ==== */
.rc-kpis {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 20px; margin: 8px 0 22px; }}
.rc-kpi {{
  background: var(--d-panel);
  border: 1px solid var(--d-line);
  border-radius: 4px;
  padding: 22px 24px;
}}
.rc-kpi-value {{
  font-family: {FONT_DISPLAY}; font-size: 34px; font-weight: 700;
  letter-spacing: -0.02em; color: var(--d-text); font-variant-numeric: tabular-nums;
}}
.rc-kpi-label {{ margin-top: 6px; font-size: 14px; color: var(--d-body); }}
.rc-kpi-hl {{ background: var(--d-grad); border-color: transparent; }}
.rc-kpi-hl .rc-kpi-value, .rc-kpi-hl .rc-kpi-label {{ color: #06302f; }}
.rc-kpi-note {{ margin-top: 2px; font-size: 12px; color: var(--d-mute); }}
.rc-kpi-hl .rc-kpi-note {{ color: rgba(6,48,47,0.72); }}

/* ============================================================ paneles === */
.rc-panel {{
  background: var(--d-panel);
  border: 1px solid var(--d-line);
  border-radius: 4px;
  padding: 18px 20px 20px;
  margin-bottom: 20px;
}}
.rc-panel-head {{
  display: flex; align-items: baseline; justify-content: space-between;
  gap: 12px; margin-bottom: 14px; padding-bottom: 10px;
  border-bottom: 1px solid var(--d-line);
}}
.rc-panel-title {{ font-size: 13px; font-weight: 600; letter-spacing: 0.10em; text-transform: uppercase; color: var(--d-text); }}
.rc-panel-sub {{ font-size: 12px; color: var(--d-mute); }}
.rc-panel-body {{ font-size: 14px; color: var(--d-body); }}

.rc-group {{ background: var(--d-panel); border: 1px solid var(--d-line); border-radius: 4px; overflow: hidden; margin-bottom: 20px; }}
.rc-row {{ display: flex; align-items: baseline; justify-content: space-between; gap: 16px; padding: 11px 16px; border-top: 1px solid var(--d-line); }}
.rc-row:first-child {{ border-top: 0; }}
.rc-row-label {{ color: var(--d-body); font-size: 14px; }}
.rc-row-value {{ color: var(--d-text); font-size: 14px; font-variant-numeric: tabular-nums; }}
.rc-row-value strong {{ color: #ffffff; font-weight: 600; }}

/* Barras de progreso con degradado. */
.rc-bars {{ display: flex; flex-direction: column; gap: 12px; }}
.rc-bar-row {{ display: grid; grid-template-columns: 1fr auto; gap: 4px 12px; }}
.rc-bar-label {{ font-size: 13px; color: var(--d-body); }}
.rc-bar-value {{ font-size: 13px; color: var(--d-text); font-variant-numeric: tabular-nums; }}
.rc-bar-track {{ grid-column: 1 / -1; height: 8px; background: rgba(255,255,255,0.08); border-radius: 9999px; overflow: hidden; }}
.rc-bar-fill {{ height: 100%; background: var(--d-grad); border-radius: 9999px; }}

/* Calendario del periodo. */
.rc-cal {{ display: grid; grid-template-columns: repeat(7, 1fr); gap: 4px; }}
.rc-cal-dow {{ font-size: 11px; color: var(--d-mute); text-align: center; padding-bottom: 4px; }}
.rc-cal-day {{ text-align: center; font-size: 12px; color: var(--d-mute); padding: 5px 0; border-radius: 3px; }}
.rc-cal-day.is-on {{ background: rgba(62,214,196,0.18); color: var(--d-text); font-weight: 600; }}
.rc-cal-day.is-empty {{ visibility: hidden; }}

/* ============================================================= tablas === */
.rc-table-wrap {{ background: var(--d-panel); border: 1px solid var(--d-line); border-radius: 4px; overflow: auto; margin-bottom: 20px; }}
table.rc-table {{ width: 100%; border-collapse: collapse; font-size: 13.5px; }}
table.rc-table th {{
  text-align: left; font-weight: 600; font-size: 11px; letter-spacing: 0.08em;
  text-transform: uppercase; color: var(--d-mute);
  padding: 10px 14px; border-bottom: 1px solid var(--d-line);
  background: var(--d-panel); position: sticky; top: 0;
}}
table.rc-table td {{ padding: 10px 14px; border-bottom: 1px solid rgba(255,255,255,0.05); color: var(--d-body); font-variant-numeric: tabular-nums; }}
table.rc-table tr:last-child td {{ border-bottom: 0; }}
table.rc-table td.num, table.rc-table th.num {{ text-align: right; }}
table.rc-table tbody tr:hover td {{ background: var(--d-panel2); }}

/* Badges. */
.rc-badge {{ display: inline-block; padding: 2px 8px; border-radius: 3px; font-size: 11px; font-weight: 600; }}
.rc-badge-ok {{ background: rgba(46,229,157,0.16); color: var(--d-green); }}
.rc-badge-warn {{ background: rgba(255,197,51,0.16); color: var(--d-yellow); }}
.rc-badge-bad {{ background: rgba(255,97,97,0.16); color: var(--d-red); }}
.rc-badge-neutral {{ background: rgba(255,255,255,0.08); color: var(--d-mute); }}

/* ======================================================= herramientas == */
.rc-toolbar {{ display: flex; align-items: flex-end; justify-content: space-between; gap: 16px; padding: 6px 0 14px; }}
.rc-toolbar h1 {{ margin: 0; font-family: {FONT_DISPLAY}; font-size: 26px; font-weight: 700; letter-spacing: -0.01em; color: var(--d-text); }}
.rc-toolbar p {{ margin: 4px 0 0; font-size: 14px; color: var(--d-mute); }}
.rc-toolbar-right {{ font-size: 12px; color: var(--d-mute); }}
.rc-chip {{ display: inline-flex; align-items: center; gap: 6px; background: var(--d-panel2); border: 1px solid var(--d-line); border-radius: 4px; padding: 4px 10px; font-size: 12px; color: var(--d-body); }}

/* ============================================================ avisos ==== */
div[data-testid="stAlert"], div[data-testid="stAlert"] > div {{
  background: var(--d-panel2) !important;
  border: 1px solid var(--d-line) !important;
  border-left: 3px solid var(--d-cyan) !important;
  border-radius: 4px !important;
  color: var(--d-body) !important;
  font-size: 14px;
}}
div[data-testid="stAlert"] p, div[data-testid="stAlert"] span {{ color: var(--d-body) !important; }}
div[data-testid="stAlert"] span[data-testid="stIconMaterial"] {{ color: var(--d-cyan) !important; }}

/* ========================================================= controles ==== */
.stButton > button, [data-testid="stPopover"] button {{
  background: var(--d-cyan) !important;
  border: 0 !important;
  border-radius: 4px !important;
  color: #06302f !important;
  font-size: 13px !important; font-weight: 600 !important;
  min-height: 34px; padding: 8px 16px !important;
  transition: filter 120ms ease, transform 90ms ease;
}}
.stButton > button:hover {{ filter: brightness(1.08); }}
.stButton > button:active {{ transform: scale(0.97); }}
div[data-testid="stExpander"] {{ background: var(--d-panel); border: 1px solid var(--d-line) !important; border-radius: 4px; }}
div[data-testid="stExpander"] summary p {{ font-size: 13px; color: var(--d-body); }}

/* Campo de búsqueda superior. */
div[data-testid="stTextInput"] {{ max-width: 440px; }}\ndiv[data-testid="stTextInput"] input {{
  background: var(--d-panel2) !important;
  border: 1px solid var(--d-line) !important;
  border-radius: 9999px !important;
  color: var(--d-text) !important;
  padding: 8px 16px !important;
}}

/* ============================================================ gráficos == */
div[data-testid="stPlotlyChart"] {{
  background: var(--d-panel);
  border: 1px solid var(--d-line);
  border-radius: 4px;
  padding: 6px 8px 2px;
}}

/* ============================================================== texto === */
a {{ color: var(--d-cyan); text-decoration: none; }}
a:hover {{ text-decoration: underline; }}
div[data-testid="stCaptionContainer"] p {{ color: var(--d-mute); font-size: 12px; }}
.stApp h2 {{ font-family: {FONT_DISPLAY}; font-size: 20px; font-weight: 600; color: var(--d-text); margin: 22px 0 8px; }}
.stApp h3 {{ font-family: {FONT_DISPLAY}; font-size: 17px; font-weight: 600; color: var(--d-text); margin: 18px 0 6px; }}
.stApp strong {{ color: var(--d-text); font-weight: 600; }}

/* ========================================================= movimiento == */
@keyframes rc-fade {{ from {{ opacity: 0; transform: translateY(4px); }} to {{ opacity: 1; transform: translateY(0); }} }}
.rc-panel, .rc-kpi, .rc-table-wrap, div[data-testid="stPlotlyChart"] {{ animation: rc-fade 0.22s ease-out both; }}
@media (prefers-reduced-motion: reduce) {{
  .rc-panel, .rc-kpi, .rc-table-wrap, div[data-testid="stPlotlyChart"] {{ animation: none; }}
}}

/* ========================================================== responsive = */
@media (max-width: 1000px) {{
  .rc-kpis {{ grid-template-columns: 1fr; gap: 12px; }}
  .rc-toolbar {{ flex-direction: column; align-items: flex-start; gap: 6px; }}
}}
@media (max-width: 600px) {{
  header[data-testid="stHeader"]::after {{ content: ""; }}
}}
</style>
"""


def load(design_file: Path | None = None) -> Theme:
    path = Path(design_file or config.DESIGN_FILE)
    name = config.DESIGN_SLUG
    if path.exists():
        text = path.read_text(encoding="utf-8")
        if text.startswith("---"):
            try:
                _, front, _ = text.split("---", 2)
                name = str((yaml.safe_load(front) or {}).get("name", name))
            except yaml.YAMLError:
                pass
    return Theme(
        slug="dashboard",
        name=name,
        bg=config.DASH_BG,
        panel=config.DASH_PANEL,
        panel2=config.DASH_PANEL_2,
        card=config.RC_CARD,
        line=config.RC_HAIRLINE,
        text=config.DASH_TEXT,
        body=config.DASH_BODY,
        mute=config.DASH_MUTE,
        cyan=config.DASH_CYAN,
        green=config.DASH_GREEN,
        red=config.RC_ACCENT_RED,
        yellow=config.RC_ACCENT_YELLOW,
        sidebar_top=config.DASH_SIDEBAR_TOP,
        sidebar_bottom=config.DASH_SIDEBAR_BOTTOM,
    )


def register_plotly(theme: Theme) -> str:
    template = go.layout.Template(
        layout=go.Layout(
            font=dict(family=FONT_TEXT, size=12, color=theme.body),
            title=dict(font=dict(family=FONT_TEXT, size=12, color=theme.mute)),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            colorway=[theme.cyan, theme.green, "#8FD8E8", theme.mute],
            xaxis=dict(gridcolor="rgba(255,255,255,0.06)", linecolor=theme.line, zeroline=False,
                       tickfont=dict(size=11, color=theme.mute), title=dict(font=dict(size=11, color=theme.mute))),
            yaxis=dict(gridcolor="rgba(255,255,255,0.06)", linecolor=theme.line, zeroline=False,
                       tickfont=dict(size=11, color=theme.mute), title=dict(font=dict(size=11, color=theme.mute))),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11, color=theme.body)),
            hoverlabel=dict(bgcolor=theme.panel2, bordercolor=theme.line,
                            font=dict(family=FONT_TEXT, color=theme.text)),
            margin=dict(t=30, l=6, r=6, b=6),
            bargap=0.45,
        )
    )
    pio.templates[TEMPLATE_NAME] = template
    return TEMPLATE_NAME


# Rellenos degradados para áreas (Plotly 7 admite fillgradient).
AREA_CYAN = dict(type="vertical", colorscale=[[0, "rgba(62,214,196,0.0)"], [1, "rgba(62,214,196,0.45)"]])
AREA_GREEN = dict(type="vertical", colorscale=[[0, "rgba(46,229,157,0.0)"], [1, "rgba(46,229,157,0.40)"]])
SEQUENTIAL = [[0.0, "#33414D"], [0.5, "#3ED6C4"], [1.0, "#2EE59D"]]
DIVERGING = [[0.0, "#3ED6C4"], [0.5, "#33414D"], [1.0, "#FF6161"]]


def apply(theme: Theme | None = None) -> Theme:
    import streamlit as st

    theme = theme or load()
    register_plotly(theme)
    st.markdown(theme.css(), unsafe_allow_html=True)
    return theme
