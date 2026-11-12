"""Tests del diseño experimental: muestra, techo de coste y elección de diseño."""

from __future__ import annotations

import math

import pytest

from campaign_optimizer import analytics
from campaign_optimizer.experiment import plan_ab_test, required_sample


def _metrics(approved=37, spend=542.11, ads=4):
    return analytics.Metrics(
        n_rows=ads, impressions=1000, clicks=50, spend=spend,
        total_conversions=50, approved_conversions=approved, n_ads=ads, n_campaigns=1,
    )


def test_muestra_sigue_la_formula_de_poisson():
    # 2·(1,959964 + 0,841621)² / 0,25² = 251,16 → 252
    assert required_sample(0.25) == 252
    assert required_sample(0.50) == 63
    assert required_sample(0.25) > required_sample(0.50) > required_sample(0.75)


def test_plan_traduce_conversiones_a_dias_y_costeg():
    plan = plan_ab_test(
        _metrics(), campaign_id=1178, segment_label="30-34 · M · interés 16",
        window_days=14,
    )
    assert plan.n_per_arm == 252
    assert plan.approved_per_day == pytest.approx(37 / 14)
    assert plan.arm_days == pytest.approx(252 / (37 / 14), rel=1e-6)
    assert plan.calendar_days == pytest.approx(plan.arm_days / 0.5, rel=1e-6)
    assert plan.calendar_days_ceiled == 191
    assert plan.calendar_days_ceiled == math.ceil(plan.calendar_days)
    assert plan.cost_ceiling == pytest.approx(plan.cost_per_approved * 1.30)
    assert plan.daily_spend == pytest.approx(542.11 / 14)
    # presupuesto de ejecución: control a gasto actual + variante con +25 %
    assert plan.test_budget == pytest.approx(
        plan.arm_days * plan.daily_spend * (1 + (1 + 0.25))
    )
    assert plan.exposure_share == 0.5
    # 95,4 días de exposición → 191 de calendario: no cabe en 60 días
    assert plan.arm_days == pytest.approx(95.35, abs=0.01)
    assert plan.feasible is False


def test_diseno_switchback_con_pocos_anuncios():
    plan = plan_ab_test(_metrics(ads=4), campaign_id=1, segment_label="x", window_days=14)
    assert "Switchback" in plan.design
    assert "antigüedad" in plan.design_reason


def test_diseno_por_anuncio_con_muchos_anuncios():
    plan = plan_ab_test(_metrics(ads=12), campaign_id=1, segment_label="x", window_days=14)
    assert "Aleatorización por anuncio" in plan.design
    assert "estratos" in plan.design_reason


def test_escenarios_ordenados_y_marcados_como_viables():
    plan = plan_ab_test(_metrics(), campaign_id=1, segment_label="x", window_days=14)
    deltas = [s["delta"] for s in plan.scenarios]
    assert deltas == sorted(deltas)
    assert plan.scenarios[0]["n_por_brazo"] > plan.scenarios[-1]["n_por_brazo"]
    assert plan.scenarios[0]["dias_calendario"] == pytest.approx(
        plan.scenarios[0]["dias_exposicion"] / 0.5, rel=1e-6
    )
    assert all(isinstance(s['cabe_en_plazo'], bool) for s in plan.scenarios)
    assert all(s['dias_calendario_plan'] == math.ceil(s['dias_calendario']) for s in plan.scenarios)


def test_no_acepta_segmento_sin_conversiones():
    with pytest.raises(ValueError):
        plan_ab_test(_metrics(approved=0), campaign_id=1, segment_label="x", window_days=14)
