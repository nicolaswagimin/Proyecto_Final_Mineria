"""Tests de la capa de datos: esquema, limpieza y trazabilidad."""

from __future__ import annotations

import pytest

from campaign_optimizer import data


def test_resolve_columns_acepta_alias_mayusculas_y_espacios(messy_text_raw_df):
    mapping = data.resolve_columns(messy_text_raw_df)
    assert mapping["ad_id"] == "Ad_ID"
    assert mapping["campaign_id"] == "XYZ_Campaign_ID"
    assert mapping["spend"] == "Spent"
    assert mapping["approved_conversions"] == "Approved_Conversion"


def test_schema_error_lista_columnas_faltantes(raw_df):
    roto = raw_df.drop(columns=["spent"])
    with pytest.raises(data.SchemaError) as exc:
        data.resolve_columns(roto)
    assert "spend" in str(exc.value)
    assert "Columnas encontradas" in str(exc.value)


def test_limpieza_cuenta_cada_decision(dirty_raw_df):
    normalized, _ = data.normalize(dirty_raw_df)
    clean, q = data.validate_and_clean(normalized)

    assert q.n_rows_raw == 13
    assert q.duplicate_rows == 1
    assert q.dropped["valores_negativos"] == 1
    # los registros válidos: 13 - 1 duplicado - 1 negativo
    assert q.n_rows_clean == 11
    assert len(clean) == q.n_rows_clean

    # Las inconsistencias se marcan, no se borran
    assert q.approved_gt_total == 1
    assert clean["flag_aprobadas_mayor_que_totales"].sum() == 1
    assert q.conversions_without_clicks == 1
    assert clean["flag_conversion_sin_clics"].sum() == 1
    assert q.notes, "las decisiones deben quedar documentadas"


def test_normalize_no_modifica_el_original(dirty_raw_df):
    copia = dirty_raw_df.copy(deep=True)
    data.normalize(dirty_raw_df)
    assert dirty_raw_df.equals(copia)


def test_dataset_id_estable_y_sensible(raw_df):
    normalized, _ = data.normalize(raw_df)
    a, _ = data.validate_and_clean(normalized)
    b, _ = data.validate_and_clean(normalized)
    assert data.dataset_id(a) == data.dataset_id(b)

    altered = raw_df.copy()
    altered.loc[0, "Spent"] = 999.0
    normalized2, _ = data.normalize(altered)
    c, _ = data.validate_and_clean(normalized2)
    assert data.dataset_id(a) != data.dataset_id(c)


def test_totales_de_limpieza_suman(raw_df):
    """Ninguna fila desaparece sin quedar contada en el informe."""
    normalized, _ = data.normalize(raw_df)
    clean, q = data.validate_and_clean(normalized)
    descartadas = sum(q.dropped.values())
    assert q.n_rows_raw - descartadas - q.duplicate_rows == q.n_rows_clean


def test_dataset_id_no_depende_del_sistema_operativo(raw_df, monkeypatch):
    import os

    normalized, _ = data.normalize(raw_df)
    clean, _ = data.validate_and_clean(normalized)
    monkeypatch.setattr(os, "linesep", "\n")
    linux_id = data.dataset_id(clean)
    monkeypatch.setattr(os, "linesep", "\r\n")
    assert data.dataset_id(clean) == linux_id


def test_detecta_filas_desalineadas(shifted_raw_df):
    """El bloque corrupto del archivo real no debe colarse como dato válido."""
    normalized, _ = data.normalize(shifted_raw_df)
    clean, q = data.validate_and_clean(normalized)
    assert q.desalineadas == 1
    assert q.dropped["filas_desalineadas"] == 1
    assert 999 not in set(clean["ad_id"])
    assert any("desplazadas" in n for n in q.notes)


def test_esquema_antiguo_con_interest_simple(messy_text_raw_df):
    normalized, mapping = data.normalize(messy_text_raw_df)
    assert mapping["interest1"] == "Interest"
    clean, q = data.validate_and_clean(normalized)
    clean = data.clean_case(clean)
    assert len(clean) == 1
    assert clean["interest1"].iloc[0] == 10
    assert clean["gender"].iloc[0] == "M"


def test_campos_opcionales_ausentes_no_rompen(raw_df):
    """Sin fechas ni interest2/3 el proyecto sigue funcionando."""
    minimal = raw_df[["ad_id", "campaign_id", "fb_campaign_id", "age", "gender",
                      "interest1", "impressions", "clicks", "spent",
                      "total_conversion", "approved_conversion"]]
    normalized, mapping = data.normalize(minimal)
    clean, q = data.validate_and_clean(normalized)
    assert len(clean) == len(minimal)
    assert "reporting_start" not in mapping
    assert q.date_range == {}
