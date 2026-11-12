"""Capa de datos: leer, validar, normalizar y auditar el CSV original.

Reglas de diseño:
  * El archivo original nunca se modifica ni se sobrescribe.
  * Cada decisión de limpieza devuelve cuántas filas afecta.
  * Las combinaciones raras (conversiones sin clics) se marcan, no se borran.
  * Las filas estructuralmente corruptas (columnas desplazadas) se detectan por
    dominio de valores y se excluyen con su propio contador, nunca en silencio.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

from . import config

CANONICAL_COLUMNS = list(config.CANONICAL_COLUMNS)

# Un rango de edad válido en esta fuente: "30-34". Se usa para detectar filas
# con columnas desplazadas, no para descartar valores raros por capricho.
AGE_PATTERN = r"^\d{1,2}-\d{1,2}$"
VALID_GENDERS = {"M", "F"}


class SchemaError(ValueError):
    """El CSV no contiene las columnas mínimas del contrato de datos."""


@dataclass
class DataQuality:
    """Informe de calidad reproducible, serializable a JSON."""

    source_path: str = ""
    source_sha256: str = ""
    dataset_id: str = ""
    n_rows_raw: int = 0
    n_rows_clean: int = 0
    columns_found: list[str] = field(default_factory=list)
    columns_missing: list[str] = field(default_factory=list)
    columns_extra: list[str] = field(default_factory=list)
    column_mapping: dict[str, str] = field(default_factory=dict)
    duplicate_rows: int = 0
    duplicate_ad_ids: int = 0
    desalineadas: int = 0
    nulls: dict[str, int] = field(default_factory=dict)
    unparsable: dict[str, int] = field(default_factory=dict)
    negatives: dict[str, int] = field(default_factory=dict)
    approved_gt_total: int = 0
    conversions_without_clicks: int = 0
    zero_impressions_rows: int = 0
    zero_spend_rows: int = 0
    date_range: dict[str, str] = field(default_factory=dict)
    dropped: dict[str, int] = field(default_factory=dict)
    flags: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _norm(name: str) -> str:
    return "".join(ch for ch in str(name).strip().lower() if ch.isalnum())


def resolve_columns(raw: pd.DataFrame) -> dict[str, str]:
    """Mapea campo canónico -> nombre real en el CSV.

    Solo los campos de `REQUIRED_FIELDS` son obligatorios; el resto se resuelve
    si existe y se ignora si no. Lanza SchemaError con mensaje accionable.
    """
    lookup = {_norm(c): c for c in raw.columns}
    mapping: dict[str, str] = {}
    for canonical, aliases in config.COLUMN_ALIASES.items():
        for alias in aliases:
            if _norm(alias) in lookup:
                mapping[canonical] = lookup[_norm(alias)]
                break
    missing = [c for c in config.REQUIRED_FIELDS if c not in mapping]
    if missing:
        raise SchemaError(
            "Faltan columnas obligatorias en el CSV: "
            + ", ".join(missing)
            + ".\nColumnas encontradas: "
            + ", ".join(map(str, raw.columns))
        )
    return mapping


def dataset_id(df: pd.DataFrame) -> str:
    """Huella estable del conjunto limpio (ordena filas y columnas)."""
    ordered = df[CANONICAL_COLUMNS].copy()
    for col in ("campaign_id", "fb_campaign_id", "ad_id"):
        ordered[col] = ordered[col].astype("string")
    ordered = ordered.sort_values(
        ["campaign_id", "fb_campaign_id", "ad_id"], kind="mergesort", na_position="first"
    )
    payload = ordered.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def normalize(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    """Renombra al contrato interno y fija tipos; no elimina nada todavía.

    Devuelve (dataframe_canónico, mapeo campo_interno -> columna_original).
    """
    mapping = resolve_columns(raw)
    df = pd.DataFrame(index=raw.index)
    for canonical in CANONICAL_COLUMNS:
        origin = mapping.get(canonical)
        df[canonical] = raw[origin] if origin else pd.NA

    for col in ("ad_id", "campaign_id", "fb_campaign_id", "interest1", "interest2",
                "interest3", *config.NUMERIC_FIELDS):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ("age", "gender"):
        df[col] = df[col].astype("string").str.strip()

    for col in ("reporting_start", "reporting_end"):
        df[col] = pd.to_datetime(df[col], errors="coerce", format="mixed", dayfirst=True)

    for col in ("ad_id", "campaign_id", "fb_campaign_id", "interest1", "interest2",
                "interest3", "impressions", "clicks",
                "total_conversions", "approved_conversions"):
        df[col] = df[col].round().astype("Int64")
    return df, mapping


def _structural_anomalies(df: pd.DataFrame) -> pd.Series:
    """Filas cuyo contenido no encaja con el dominio de su columna.

    El archivo real trae un bloque con las columnas desplazadas dos posiciones:
    la edad aparece en `campaign_id`, el género en `fb_campaign_id`, etc. Se
    detecta por dominio (campaña numérica, edad tipo '30-34', género M/F) y se
    exige más de una violación para no marcar un simple dato atípico.
    """
    campaign_ok = df["campaign_id"].notna()
    age_ok = df["age"].str.match(AGE_PATTERN).fillna(False).astype(bool)
    gender_ok = df["gender"].str.upper().isin(VALID_GENDERS).fillna(False).astype(bool)
    violations = (
        (~campaign_ok).astype(int) + (~age_ok).astype(int) + (~gender_ok).astype(int)
    )
    return violations >= 2


def validate_and_clean(normalized: pd.DataFrame) -> tuple[pd.DataFrame, DataQuality]:
    """Audita y limpia. Devuelve (datos_válidos, informe de calidad)."""
    q = DataQuality()
    q.n_rows_raw = int(len(normalized))
    q.columns_found = [c for c in CANONICAL_COLUMNS]
    q.nulls = {c: int(normalized[c].isna().sum()) for c in CANONICAL_COLUMNS}

    df = normalized

    # 1. Filas estructuralmente corruptas (columnas desplazadas)
    shifted = _structural_anomalies(df)
    q.desalineadas = int(shifted.sum())
    if q.desalineadas:
        q.dropped["filas_desalineadas"] = q.desalineadas
        df = df.loc[~shifted].copy()
        q.notes.append(
            f"{q.desalineadas} filas tienen las columnas desplazadas (la edad aparece "
            "en campaign_id, el género en fb_campaign_id, etc.): se pierden "
            "campaign_id y fb_campaign_id, así que su gasto y sus conversiones no son "
            "atribuibles a ninguna campaña. Se excluyen del análisis, no del archivo. "
            "Verifica que el CSV descargado no esté corrupto."
        )

    # 2. Duplicados exactos
    dup_mask = df.duplicated(keep="first")
    q.duplicate_rows = int(dup_mask.sum())
    df = df.loc[~dup_mask].copy()

    # 3. Identificadores de anuncio repetidos (no se borran: el anuncio puede
    # repartirse en varios cortes de audiencia, pero el dato debe quedar visible)
    q.duplicate_ad_ids = int(df["ad_id"].duplicated(keep=False).sum())

    # 4. Nulos en campos obligatorios y en los objetivos
    q.unparsable = {c: int(normalized[c].isna().sum()) for c in config.NUMERIC_FIELDS}
    base = ["ad_id", "campaign_id", "age", "gender", "interest1",
            "impressions", "clicks", "spend"]
    before = len(df)
    df = df.dropna(subset=base)
    q.dropped["nulos_en_campos_obligatorios"] = before - len(df)

    before = len(df)
    df = df.dropna(subset=["total_conversions", "approved_conversions"])
    q.dropped["nulos_en_objetivos"] = before - len(df)
    if q.dropped["nulos_en_objetivos"]:
        q.notes.append(
            f"{q.dropped['nulos_en_objetivos']} filas no registran conversiones (NaN). "
            "Sin objetivo no se pueden evaluar ni entrenar modelos; se excluyen."
        )

    # 5. Negativos (corrupción, no corrección contable)
    q.negatives = {
        col: int((df[col] < 0).sum()) for col in config.NON_NEGATIVE_FIELDS
    }
    q.negatives = {k: v for k, v in q.negatives.items() if v}
    neg_mask = pd.Series(False, index=df.index)
    for col in config.NON_NEGATIVE_FIELDS:
        neg_mask |= df[col] < 0
    q.dropped["valores_negativos"] = int(neg_mask.sum())
    df = df.loc[~neg_mask].copy()

    # 6. Inconsistencias de negocio: se marcan, no se borran
    q.approved_gt_total = int((df["approved_conversions"] > df["total_conversions"]).sum())
    sin_clics = (
        (df["total_conversions"] > 0) | (df["approved_conversions"] > 0)
    ) & (df["clicks"] == 0)
    q.conversions_without_clicks = int(sin_clics.sum())
    q.zero_impressions_rows = int((df["impressions"] == 0).sum())
    q.zero_spend_rows = int((df["spend"] == 0).sum())

    df["flag_conversion_sin_clics"] = sin_clics
    df["flag_aprobadas_mayor_que_totales"] = (
        df["approved_conversions"] > df["total_conversions"]
    )

    if q.conversions_without_clicks:
        q.notes.append(
            f"{q.conversions_without_clicks} filas tienen conversiones con cero clics. "
            "Se conservan marcadas (flag_conversion_sin_clics): pueden reflejar "
            "conversiones view-through o un error de registro, y descartarlas "
            "eliminaría gasto real."
        )
    if q.approved_gt_total:
        q.notes.append(
            f"{q.approved_gt_total} filas tienen conversiones aprobadas mayores que las "
            "totales (imposible contablemente). Se conservan marcadas: el gasto y las "
            "impresiones de esas filas siguen siendo válidos."
        )
    if q.zero_impressions_rows:
        q.notes.append(
            f"{q.zero_impressions_rows} filas tienen cero impresiones: el CTR no es "
            "calculable para ellas (denominador cero), pero sus otros totales sí."
        )

    # 7. Tipos finales y rango temporal
    df["spend"] = df["spend"].astype("float64")
    df = df.reset_index(drop=True)
    fechas = df["reporting_start"].dropna()
    if len(fechas):
        q.date_range = {
            "min": str(df["reporting_start"].min().date()),
            "max": str(df["reporting_end"].max().date()),
        }

    q.n_rows_clean = int(len(df))
    q.flags = {
        "conversion_sin_clics": int(df["flag_conversion_sin_clics"].sum()),
        "aprobadas_mayor_que_totales": int(df["flag_aprobadas_mayor_que_totales"].sum()),
    }
    return df, q


def clean_case(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza el texto de los segmentos sin alterar los valores numéricos."""
    out = df.copy()
    if "gender" in out:
        out["gender"] = out["gender"].str.strip().str.upper()
    if "age" in out:
        out["age"] = out["age"].str.strip()
    return out


def _ensure_csv(path: Path) -> tuple[Path, bool]:
    """Devuelve una ruta CSV utilizable; si falta, intenta el .zip de Kaggle.

    Solo lee y extrae: el zip original queda intacto.
    """
    if path.exists():
        return path, False
    for zpath in sorted(path.parent.glob("*.zip")):
        try:
            with zipfile.ZipFile(zpath) as zf:
                names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
                if not names:
                    continue
                with zf.open(names[0]) as src, open(path, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                return path, True
        except zipfile.BadZipFile:
            continue
    raise FileNotFoundError(
        f"No existe el CSV en {path}. Descarga el dataset de Kaggle "
        "(madislemsalu/facebook-ad-campaign), descomprímelo y guárdalo en esa "
        "ruta (o deja el .zip en la misma carpeta y se extrae solo)."
    )


def load_dataset(path: Path | None = None) -> tuple[pd.DataFrame, DataQuality]:
    """Flujo completo: leer -> normalizar -> validar/limpiar."""
    requested = Path(path or config.RAW_CSV)
    path, extracted = _ensure_csv(requested)
    raw = pd.read_csv(path, dtype=str, keep_default_na=True)
    normalized, mapping = normalize(raw)
    clean, quality = validate_and_clean(normalized)
    clean = clean_case(clean)
    quality.column_mapping = mapping
    quality.source_path = str(path)
    quality.columns_missing = [
        c for c in CANONICAL_COLUMNS if c not in mapping
    ]
    quality.columns_extra = [
        c for c in raw.columns if c not in mapping.values()
    ]
    quality.source_sha256 = file_sha256(path)
    quality.dataset_id = dataset_id(clean)
    if extracted:
        quality.notes.append(
            f"El CSV se extrajo automáticamente de un .zip en {path.parent}; "
            "el zip original no se ha modificado."
        )
    quality.notes.append(
        f"Archivo original intacto: {path.name} ({quality.source_sha256[:12]}…)."
    )
    return clean, quality
