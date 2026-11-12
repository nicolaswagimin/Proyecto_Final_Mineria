"""Tests del tema: pizarra oscura, degradados y componentes del panel."""

from __future__ import annotations

from pathlib import Path

import plotly.io as pio

from campaign_optimizer import config, theme


def test_paleta_del_panel():
    t = theme.load()
    assert t.bg.lower() == "#2a353f"
    assert t.panel.lower() == "#33414d"
    assert t.text.lower() == "#f2f7fa"
    assert t.cyan.lower() == "#3ed6c4"
    assert t.green.lower() == "#2ee59d"
    assert t.sidebar_top.lower() == config.DASH_SIDEBAR_TOP.lower()
    assert t.sidebar_bottom.lower() == config.DASH_SIDEBAR_BOTTOM.lower()
    assert config.PALETTE == ("#021C1E", "#004445", "#2C7873", "#6FB98F")


def test_css_aplica_la_referencia():
    css = theme.load().css()
    # sidebar con degradado teal → verde
    assert "linear-gradient(180deg, #2C7873 0%, #6FB98F 100%)" in css
    # tarjeta KPI en degradado cian → verde
    assert "linear-gradient(135deg, #3ED6C4 0%, #2EE59D 100%)" in css
    assert ".rc-kpi-hl" in css
    # barras de progreso y calendario
    assert ".rc-bar-fill" in css
    assert ".rc-cal-day.is-on" in css
    # sin sombras
    assert "box-shadow" not in css
    # título desplazado del botón de filtros
    assert "left: 58px" in css


def test_componentes_presentes():
    css = theme.load().css()
    for selector in (".rc-kpi", ".rc-panel", ".rc-panel-title", ".rc-bar-track", "table.rc-table",
                     ".rc-badge", ".rc-cal"):
        assert selector in css


def test_relleno_degradado_en_areas():
    # Plotly 7 admite fillgradient: los degradados de área son reales, no una imagen
    assert theme.AREA_CYAN["type"] == "vertical"
    assert theme.AREA_GREEN["type"] == "vertical"
    colors = dict(theme.AREA_CYAN["colorscale"])
    assert "rgba(62,214,196,0.45)" in colors.values()


def test_plantilla_plotly_cian_y_verde():
    t = theme.load()
    name = theme.register_plotly(t)
    layout = pio.templates[name].layout
    assert layout.paper_bgcolor == "rgba(0,0,0,0)"
    assert t.cyan in list(layout.colorway)
    assert t.green in list(layout.colorway)


def test_sin_design_md_cae_en_el_tema():
    t = theme.load(Path("no-existe.md"))
    assert t.bg.startswith("#")
