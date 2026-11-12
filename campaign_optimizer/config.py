"""Configuración central del proyecto.

Un único sitio para rutas, semilla, contrato de columnas y umbrales de evidencia.
Ningún módulo debe hardcodear nombres de columnas: todos pasan por aquí.
"""

from __future__ import annotations

import os
from pathlib import Path

# --- Rutas -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# Permite apuntar a otro CSV (CI, pruebas de integración) sin tocar el código.
RAW_CSV = Path(os.environ.get("CO_RAW_CSV", DATA_DIR / "raw" / "facebook_ads.csv"))

# Permite redirigir artefactos (tests de integración, CI) sin tocar el código.
ARTIFACTS_DIR = Path(os.environ.get("CO_ARTIFACTS_DIR", PROJECT_ROOT / "artifacts"))
MODELS_DIR = ARTIFACTS_DIR / "models"
TRAIN_REPORT = ARTIFACTS_DIR / "train_report.json"
PREDICTIONS_CSV = ARTIFACTS_DIR / "predictions.csv"
QUALITY_REPORT = ARTIFACTS_DIR / "data_quality.json"

# --- Reproducibilidad --------------------------------------------------------
SEED = 42

# --- Contrato de datos -------------------------------------------------------
# Nombres canónicos internos (los que usa todo el código) y alias aceptados del
# CSV original. El mapeo se resuelve por nombre, ignorando mayúsculas y espacios.
#
# El archivo real de Kaggle trae TRES columnas de interés por anuncio
# (interest1/interest2/interest3) y dos fechas de reporte. `interest2/3` y las
# fechas son opcionales: si faltan, el proyecto sigue funcionando.
COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "ad_id": ("ad_id", "adid"),
    "campaign_id": ("campaign_id", "xyz_campaign_id", "xyz_campaign"),
    "fb_campaign_id": ("fb_campaign_id", "fbcampaignid"),
    "age": ("age", "age_group"),
    "gender": ("gender", "sex"),
    "interest1": ("interest1", "interest", "interest_1"),
    "interest2": ("interest2", "interest_2"),
    "interest3": ("interest3", "interest_3"),
    "reporting_start": ("reporting_start", "date_start"),
    "reporting_end": ("reporting_end", "date_end"),
    "impressions": ("impressions", "impression"),
    "clicks": ("clicks", "click"),
    "spend": ("spent", "spend", "cost", "cost_usd"),
    "total_conversions": ("total_conversion", "total_conversions"),
    "approved_conversions": ("approved_conversion", "approved_conversions"),
}

CANONICAL_COLUMNS = tuple(COLUMN_ALIASES)

# Sin estas columnas no hay análisis posible.
REQUIRED_FIELDS = (
    "ad_id",
    "campaign_id",
    "age",
    "gender",
    "interest1",
    "impressions",
    "clicks",
    "spend",
    "total_conversions",
    "approved_conversions",
)

NUMERIC_FIELDS = (
    "impressions",
    "clicks",
    "spend",
    "total_conversions",
    "approved_conversions",
)

# Campos que nunca pueden ser negativos (un gasto o unas impresiones negativas
# indican corrupción, no una corrección contable).
NON_NEGATIVE_FIELDS = NUMERIC_FIELDS

# Identificadores: sirven para agrupar y rastrear, jamás como variable de entrada.
ID_FIELDS = ("ad_id", "campaign_id", "fb_campaign_id")

# --- Objetivos de modelado ---------------------------------------------------
TARGETS: dict[str, str] = {
    "conversiones_totales": "total_conversions",
    "conversiones_aprobadas": "approved_conversions",
}

TARGET_LABELS: dict[str, str] = {
    "conversiones_totales": "Conversiones totales",
    "conversiones_aprobadas": "Conversiones aprobadas",
}

# --- Escenario de estimación -------------------------------------------------
# Momento elegido: se predice DESPUÉS de conocer gasto, impresiones y clics
# (lo que un analista sabe al cerrar el día de una campaña). Por eso esas tres
# variables son entradas y no se usan conversiones ni sus derivados.
#
# `interest1` es el interés principal del anuncio. interest2/3 existen en el
# archivo pero tienen cientos de valores: usarlos como dimensión partiría los
# segmentos en celdas de uno o dos anuncios, sin evidencia posible.
FEATURE_NUMERIC = ("spend", "impressions", "clicks")
FEATURE_CATEGORICAL = ("age", "gender", "interest1")

SCENARIO_TEXT = (
    "Estimación posterior a la interacción: se asume conocido el gasto, las "
    "impresiones y los clics del anuncio. No se usan conversiones ni variables "
    "derivadas de ellas (CTR, CPC, coste por conversión)."
)

# --- Umbral de evidencia para recomendar ------------------------------------
# Se justifica con la distribución observada en el README (se recalcula al
# entrenar y queda en el informe de calidad).
EVIDENCE_MIN_ADS = 3          # anuncios distintos que sostienen el segmento
EVIDENCE_MIN_APPROVED = 1     # al menos una conversión aprobada en el segmento
EVIDENCE_MIN_IMPROVEMENT = 0.20   # 20 % mejor coste/conversión que su campaña
EVIDENCE_MIN_SPEND_RATIO = 0.02   # el segmento debe ser >=2 % del gasto de la campaña

# --- Identidad visual --------------------------------------------------------
# DESIGN.md copiado desde devkit/resources/design-md/raycast/DESIGN.md.
DESIGN_SLUG = "raycast"
DESIGN_FILE = PROJECT_ROOT / "DESIGN.md"

# Paleta del proyecto: la gama teal/verde indicada por el usuario, aplicada
# sobre la estructura de Raycast. Las superficies usan alfa de la propia gama
# para conservar el escalón de elevación sin introducir colores nuevos.
PALETTE = ("#021C1E", "#004445", "#2C7873", "#6FB98F")

# Lienzo pizarra y paneles (referencia del panel de administración).
DASH_BG = "#2A353F"
DASH_PANEL = "#33414D"
DASH_PANEL_2 = "#3B4A57"
DASH_TEXT = "#F2F7FA"
DASH_BODY = "#C9D6DF"
DASH_MUTE = "#93A7B4"
DASH_CYAN = "#3ED6C4"
DASH_GREEN = "#2EE59D"
DASH_SIDEBAR_TOP = "#2C7873"
DASH_SIDEBAR_BOTTOM = "#6FB98F"

# Tokens que consume theme.py
RC_CANVAS = DASH_BG
RC_SURFACE = DASH_PANEL
RC_ELEVATED = DASH_PANEL_2
RC_CARD = "#3A4855"
RC_HAIRLINE = "rgba(255, 255, 255, 0.08)"
RC_INK = DASH_TEXT
RC_BODY = DASH_BODY
RC_MUTE = DASH_MUTE
RC_ASH = "#74899A"
RC_PRIMARY = DASH_CYAN
RC_ON_PRIMARY = "#08343A"
RC_ACCENT_BLUE = DASH_CYAN
RC_ACCENT_GREEN = DASH_GREEN
RC_ACCENT_RED = "#FF6161"
RC_ACCENT_YELLOW = "#FFC533"
RC_STRIPE_START = DASH_CYAN
RC_STRIPE_END = DASH_GREEN
# --- Etiquetas de presentación ------------------------------------------------
SPEND_LABEL = "Gasto"
CURRENCY_NOTE = (
    "La fuente no documenta la moneda. Los importes se muestran en las unidades "
    "del archivo original; no se afirma que sean USD."
)
