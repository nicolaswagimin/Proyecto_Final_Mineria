"""Tests de modelado: partición honesta, baseline y reproducibilidad."""

from __future__ import annotations

import numpy as np
import pytest

from campaign_optimizer import data, modeling


@pytest.fixture
def clean(raw_df):
    normalized, _ = data.normalize(raw_df)
    df, _ = data.validate_and_clean(normalized)
    return df


def test_particion_deja_fuera_una_campana_completa(clean):
    folds = modeling.campaign_folds(clean)
    assert len(folds) == clean["campaign_id"].nunique()
    for train_idx, test_idx, campaign in folds:
        train_campaigns = set(clean["campaign_id"].iloc[train_idx])
        test_campaigns = set(clean["campaign_id"].iloc[test_idx])
        assert test_campaigns == {campaign}
        assert campaign not in train_campaigns, "la campaña evaluada no puede estar en entrenamiento"
        assert len(train_idx) + len(test_idx) == len(clean)


def test_entrenamiento_produce_predicciones_fuera_de_muestra(clean):
    out = modeling.train(clean, quality={}, tune=False, seed=7)
    preds = out.predictions
    assert set(preds["objetivo"]) == set(modeling.config.TARGETS)

    for target in modeling.config.TARGETS:
        sub = preds[preds["objetivo"] == target]
        # una fila por registro y objetivo
        assert len(sub) == len(clean)
        # cada campaña aparece exactamente una vez como campaña evaluada
        assert set(sub["campaña_evaluada"]) == set(clean["campaign_id"].unique())
        for camp, group in sub.groupby("campaña_evaluada"):
            assert len(group) == int((clean["campaign_id"] == camp).sum())
        for col in ("pred_random_forest", "pred_adaboost", "pred_dummy"):
            assert sub[col].notna().all()


def test_metricas_incluyen_los_tres_modelos(clean):
    out = modeling.train(clean, quality={}, tune=False, seed=7)
    assert set(out.metrics["modelo"]) == {"random_forest", "adaboost", "dummy"}
    assert set(out.metrics["objetivo"]) == set(modeling.config.TARGETS)
    assert (out.metrics["n"] == len(clean)).all()
    assert (out.metrics["mae"] >= 0).all()


def test_es_reproducible_con_la_misma_semilla(clean):
    a = modeling.train(clean, quality={}, tune=False, seed=7)
    b = modeling.train(clean, quality={}, tune=False, seed=7)
    columnas = ["objetivo", "modelo", "mae", "rmse"]
    pd_a = a.metrics[columnas].sort_values(["objetivo", "modelo"]).reset_index(drop=True)
    pd_b = b.metrics[columnas].sort_values(["objetivo", "modelo"]).reset_index(drop=True)
    assert pd_a.equals(pd_b)


def test_la_referencia_predice_la_media_del_entrenamiento(clean):
    out = modeling.train(clean, quality={}, tune=False, seed=7)
    preds = out.predictions
    folds = modeling.campaign_folds(clean)
    for train_idx, test_idx, campaign in folds:
        media = clean["approved_conversions"].iloc[train_idx].mean()
        sub = preds[
            (preds["objetivo"] == "conversiones_aprobadas")
            & (preds["campaña_evaluada"] == campaign)
        ]
        assert np.allclose(sub["pred_dummy"].to_numpy(), media)


def test_no_se_usan_identificadores_ni_objetivos_como_entrada(clean):
    X, targets = modeling.build_features(clean)
    for prohibida in ("ad_id", "fb_campaign_id", "campaign_id",
                      "total_conversions", "approved_conversions", "spend_x"):
        assert prohibida not in X.columns
    assert set(targets) == set(modeling.config.TARGETS)
    # sí están las tres variables del escenario
    for requerida in modeling.config.FEATURE_NUMERIC:
        assert requerida in X.columns
