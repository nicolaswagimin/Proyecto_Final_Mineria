"""Fixtures compartidas: datos sintéticos mínimos, nunca el dataset real.

Los tests no dependen del CSV de Kaggle (que no se versiona). Construyen un
DataFrame con el MISMO esquema del archivo real: 15 columnas, tres intereses y
dos fechas de reporte.
"""

from __future__ import annotations

import pandas as pd
import pytest

ORIGINAL_COLUMNS = [
    "ad_id",
    "reporting_start",
    "reporting_end",
    "campaign_id",
    "fb_campaign_id",
    "age",
    "gender",
    "interest1",
    "interest2",
    "interest3",
    "impressions",
    "clicks",
    "spent",
    "total_conversion",
    "approved_conversion",
]

START = "17/08/2017"
END = "17/08/2017"


def raw_row(ad_id, campaign, fb_id, age, gender, interest1, impressions, clicks,
            spent, total, approved, interest2=None, interest3=None):
    return {
        "ad_id": ad_id,
        "reporting_start": START,
        "reporting_end": END,
        "campaign_id": campaign,
        "fb_campaign_id": fb_id,
        "age": age,
        "gender": gender,
        "interest1": interest1,
        "interest2": interest1 if interest2 is None else interest2,
        "interest3": interest1 if interest3 is None else interest3,
        "impressions": impressions,
        "clicks": clicks,
        "spent": spent,
        "total_conversion": total,
        "approved_conversion": approved,
    }


@pytest.fixture
def raw_df() -> pd.DataFrame:
    """Dos campañas con segmentos distinguibles."""
    rows = [
        # campaña 1: segmento barato (30-34, M, interés 10) y caro (35-39, F, 20)
        raw_row(1, 1, 1001, "30-34", "M", 10, 1000, 50, 20.0, 10, 4),
        raw_row(2, 1, 1001, "30-34", "M", 10, 1200, 60, 24.0, 12, 5),
        raw_row(3, 1, 1002, "35-39", "F", 20, 800, 20, 40.0, 2, 1),
        raw_row(4, 1, 1002, "35-39", "F", 20, 900, 25, 45.0, 3, 1),
        # campaña 2
        raw_row(5, 2, 2001, "30-34", "M", 10, 500, 10, 15.0, 3, 1),
        raw_row(6, 2, 2001, "30-34", "M", 10, 600, 12, 18.0, 4, 2),
        raw_row(7, 2, 2002, "40-44", "F", 30, 700, 14, 30.0, 1, 0),
        raw_row(8, 2, 2002, "40-44", "F", 30, 750, 16, 33.0, 2, 1),
    ]
    return pd.DataFrame(rows, columns=ORIGINAL_COLUMNS)


@pytest.fixture
def dirty_raw_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Con duplicado, negativo, aprobadas > totales, conv. sin clics y desalineada."""
    extra = pd.DataFrame(
        [
            # duplicado exacto de la primera fila
            raw_df.iloc[0].to_dict(),
            # gasto negativo (corrupción)
            raw_row(9, 1, 1001, "30-34", "M", 10, 100, 5, -3.0, 1, 1),
            # aprobadas mayores que totales
            raw_row(10, 1, 1001, "30-34", "M", 10, 400, 8, 12.0, 1, 3),
            # conversiones con cero clics
            raw_row(11, 2, 2001, "30-34", "M", 10, 300, 0, 7.0, 2, 1),
            # cero impresiones
            raw_row(12, 2, 2001, "30-34", "M", 10, 0, 0, 0.0, 0, 0),
        ],
        columns=ORIGINAL_COLUMNS,
    )
    return pd.concat([raw_df, extra], ignore_index=True)


@pytest.fixture
def shifted_raw_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Reproduce el bloque corrupto del archivo real: desplazado dos columnas.

    Las dos primeras columnas de datos (campaign_id y fb_campaign_id) se pierden
    y todo lo demás se corre a la izquierda; las dos últimas quedan vacías.
    """
    base = raw_df.iloc[0].to_dict()
    shifted = {
        "ad_id": 999,
        "reporting_start": base["reporting_start"],
        "reporting_end": base["reporting_end"],
        "campaign_id": "45-49",       # donde debería ir la edad
        "fb_campaign_id": "M",        # donde debería ir el género
        "age": 10,                    # donde debería ir interest1
        "gender": 14,                 # interest2
        "interest1": 14,              # interest3
        "interest2": 426500,          # impressions
        "interest3": 72,              # clicks
        "impressions": 128.2799988,   # spent
        "clicks": 4,                  # total_conversion
        "spent": 1,                   # approved_conversion
        "total_conversion": None,
        "approved_conversion": None,
    }
    return pd.concat([raw_df, pd.DataFrame([shifted])], ignore_index=True)


@pytest.fixture
def messy_text_raw_df() -> pd.DataFrame:
    """Esquema antiguo con un solo 'Interest' y nombres con mayúsculas."""
    return pd.DataFrame(
        [
            {
                "Ad_ID": "1",
                "XYZ_Campaign_ID": "1",
                "Fb_Campaign_ID": "1001",
                "Age": " 30-34 ",
                "Gender": "m",
                "Interest": "10",
                "Impressions": "100",
                "Clicks": "10",
                "Spent": "5.5",
                "Total_Conversion": "2",
                "Approved_Conversion": "1",
            }
        ]
    )
