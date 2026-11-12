"""Tests de la capa analítica: una sola fuente de verdad para las cifras."""

from __future__ import annotations

import pandas as pd
import pytest

from campaign_optimizer import analytics, data


@pytest.fixture
def clean(raw_df):
    normalized, _ = data.normalize(raw_df)
    df, _ = data.validate_and_clean(normalized)
    return df


def test_totales_coinciden_con_calculo_independiente(clean):
    m = analytics.totals(clean)
    assert m.spend == pytest.approx(clean["spend"].sum())
    assert m.impressions == int(clean["impressions"].sum())
    assert m.clicks == int(clean["clicks"].sum())
    assert m.total_conversions == int(clean["total_conversions"].sum())
    assert m.approved_conversions == int(clean["approved_conversions"].sum())
    assert m.n_ads == clean["ad_id"].nunique()

    # Fórmulas del encargo, calculadas a mano
    assert m.ctr == pytest.approx(100 * clean["clicks"].sum() / clean["impressions"].sum())
    assert m.cpc == pytest.approx(clean["spend"].sum() / clean["clicks"].sum())
    assert m.cost_per_approved == pytest.approx(
        clean["spend"].sum() / clean["approved_conversions"].sum()
    )


def test_denominadores_cero_no_revientan(clean):
    vacio = clean.iloc[0:0]
    m = analytics.totals(vacio)
    assert m.ctr is None
    assert m.cpc is None
    assert m.cost_per_approved is None
    assert m.spend == 0

    # Recorte sin conversiones aprobadas
    sin_aprobadas = clean[clean["approved_conversions"] == 0]
    if not sin_aprobadas.empty:
        m2 = analytics.totals(sin_aprobadas)
        assert m2.cost_per_approved is None
        assert m2.spend > 0  # el gasto sigue visible


def test_filtro_vacio_devuelve_metrics_vacio(clean):
    filtrado = analytics.filter_records(clean, campaigns=[999])
    assert filtrado.empty
    assert analytics.totals(filtrado).n_rows == 0
    assert analytics.campaign_summary(filtrado).empty
    assert analytics.segment_summary(filtrado).empty


def test_segment_summary_calcula_ratio_desde_sumas(clean):
    tabla = analytics.segment_summary(clean)
    assert not tabla.empty
    for row in tabla.itertuples():
        esperado = row.spend / row.approved_conversions if row.approved_conversions else None
        if esperado is None:
            assert row.cost_per_approved is None
        else:
            assert row.cost_per_approved == pytest.approx(esperado)


def test_segmento_con_cero_conversiones_queda_marcado(raw_df):
    normalized, _ = data.normalize(raw_df)
    df, _ = data.validate_and_clean(normalized)
    tabla = analytics.segment_summary(df)
    con_cero = tabla[tabla["approved_conversions"] == 0]
    if not con_cero.empty:
        assert con_cero["sin_conversiones_aprobadas"].all()
        assert con_cero["cost_per_approved"].isna().all()


def test_recomendacion_compara_con_el_resto_de_su_campana(clean):
    recs = analytics.recommend(clean, analytics.EvidenceThreshold(min_ads=1, min_approved=1))
    assert recs, "con estos datos debería existir al menos un candidato"
    for rec in recs:
        # la referencia no contiene las filas del segmento
        assert rec.reference.n_rows == rec.trace["filas_referencia"]
        assert rec.reference.n_rows >= 0
        assert rec.segment.spend == pytest.approx(rec.trace["gasto_segmento"])
        assert rec.segment.approved_conversions == rec.trace["aprobadas_segmento"]
        # y la mejora se puede recalcular
        if rec.improvement is not None:
            esperado = (rec.reference.cost_per_approved - rec.segment.cost_per_approved) / (
                rec.reference.cost_per_approved
            )
            assert rec.improvement == pytest.approx(esperado)


def test_umbral_estricto_descarta_todo(clean):
    estricto = analytics.EvidenceThreshold(min_ads=99, min_approved=99)
    assert analytics.recommend(clean, estricto) == []
    motivos = analytics.discarded_segment_reasons(clean, estricto)
    assert not motivos.empty
    assert "Sin conversiones aprobadas suficientes" in set(motivos["motivo"])


def test_cuotas_de_conversion_suman_cien(clean):
    share = analytics.conversion_share_by_campaign(clean)
    if not share.empty and share["cuota_conversiones_totales"].notna().all():
        assert share["cuota_conversiones_totales"].sum() == pytest.approx(100.0)
