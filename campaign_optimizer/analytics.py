"""Capa analítica: única fuente de verdad para tarjetas, tablas y gráficos.

Toda métrica se calcula sobre sumas del conjunto filtrado. Nunca se promedian
ratios por fila: eso produciría cifras que no coinciden con los totales.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import pandas as pd

from . import config

DATASET_CHANNEL = "Facebook"


@dataclass(frozen=True)
class Metrics:
    """Indicadores de un conjunto de filas. None = no calculable (denominador 0)."""

    n_rows: int = 0
    impressions: int = 0
    clicks: int = 0
    spend: float = 0.0
    total_conversions: int = 0
    approved_conversions: int = 0
    n_ads: int = 0
    n_campaigns: int = 0

    @property
    def ctr(self) -> float | None:
        return 100.0 * self.clicks / self.impressions if self.impressions else None

    @property
    def cpc(self) -> float | None:
        return self.spend / self.clicks if self.clicks else None

    @property
    def cost_per_approved(self) -> float | None:
        return self.spend / self.approved_conversions if self.approved_conversions else None

    @property
    def cost_per_total(self) -> float | None:
        return self.spend / self.total_conversions if self.total_conversions else None

    @property
    def approval_rate(self) -> float | None:
        return 100.0 * self.approved_conversions / self.total_conversions if self.total_conversions else None

    def to_dict(self) -> dict:
        data = asdict(self)
        data.update(
            ctr=self.ctr,
            cpc=self.cpc,
            cost_per_approved=self.cost_per_approved,
            cost_per_total=self.cost_per_total,
            approval_rate=self.approval_rate,
        )
        return data


def metrics_from_sums(
    n_rows: int,
    impressions: int,
    clicks: int,
    spend: float,
    total_conversions: int,
    approved_conversions: int,
    n_ads: int,
    n_campaigns: int = 1,
) -> Metrics:
    return Metrics(
        n_rows=int(n_rows),
        impressions=int(impressions),
        clicks=int(clicks),
        spend=float(spend),
        total_conversions=int(total_conversions),
        approved_conversions=int(approved_conversions),
        n_ads=int(n_ads),
        n_campaigns=int(n_campaigns),
    )


def filter_records(
    df: pd.DataFrame,
    campaigns: list | None = None,
    ages: list | None = None,
    genders: list | None = None,
    interests: list | None = None,
) -> pd.DataFrame:
    """Aplica filtros de segmentación. Lista vacía o None = sin restricción."""
    out = df
    if campaigns:
        out = out[out["campaign_id"].isin(campaigns)]
    if ages:
        out = out[out["age"].isin(ages)]
    if genders:
        out = out[out["gender"].isin(genders)]
    if interests:
        out = out[out["interest1"].isin(interests)]
    return out


def totals(df: pd.DataFrame) -> Metrics:
    """Totales del conjunto filtrado (alimenta las tarjetas del resumen)."""
    if df.empty:
        return Metrics()
    return metrics_from_sums(
        n_rows=len(df),
        impressions=df["impressions"].sum(),
        clicks=df["clicks"].sum(),
        spend=df["spend"].sum(),
        total_conversions=df["total_conversions"].sum(),
        approved_conversions=df["approved_conversions"].sum(),
        n_ads=df["ad_id"].nunique(),
        n_campaigns=df["campaign_id"].nunique(),
    )


def _aggregate(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    agg = (
        df.groupby(by, dropna=False)
        .agg(
            n_rows=("ad_id", "size"),
            impressions=("impressions", "sum"),
            clicks=("clicks", "sum"),
            spend=("spend", "sum"),
            total_conversions=("total_conversions", "sum"),
            approved_conversions=("approved_conversions", "sum"),
            n_ads=("ad_id", "nunique"),
        )
        .reset_index()
    )
    for col in ("impressions", "clicks", "spend", "total_conversions",
                "approved_conversions", "n_ads", "n_rows"):
        agg[col] = agg[col].astype(float)

    agg["ctr"] = agg.apply(
        lambda r: 100.0 * r.clicks / r.impressions if r.impressions else None, axis=1
    )
    agg["cpc"] = agg.apply(
        lambda r: r.spend / r.clicks if r.clicks else None, axis=1
    )
    agg["cost_per_approved"] = agg.apply(
        lambda r: r.spend / r.approved_conversions if r.approved_conversions else None, axis=1
    )
    agg["cost_per_total"] = agg.apply(
        lambda r: r.spend / r.total_conversions if r.total_conversions else None, axis=1
    )
    agg["approval_rate"] = agg.apply(
        lambda r: 100.0 * r.approved_conversions / r.total_conversions if r.total_conversions else None,
        axis=1,
    )
    agg["sin_conversiones_aprobadas"] = agg["approved_conversions"] <= 0
    return agg


def campaign_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Tabla comparable de campañas (todas las métricas en las mismas columnas)."""
    if df.empty:
        return pd.DataFrame()
    return _aggregate(df, ["campaign_id"]).sort_values("campaign_id").reset_index(drop=True)


def segment_summary(
    df: pd.DataFrame, dims: list[str] | None = None, include_campaign: bool = True
) -> pd.DataFrame:
    """Eficiencia por segmento de audiencia (edad, género, interés).

    `campaign_id` va siempre en la agrupación: el mismo segmento en dos campañas
    distintas no es comparable (presupuestos y objetivos distintos). Mantiene el
    número de anuncios y las conversiones que sostienen el ratio: un coste por
    conversión sin su volumen no es evidencia.
    """
    if df.empty:
        return pd.DataFrame()
    dims = dims or ["age", "gender", "interest1"]
    group_cols = (["campaign_id"] if include_campaign else []) + list(dims)
    agg = _aggregate(df, group_cols)
    labels = {"age": "Edad", "gender": "Género", "interest1": "Interés"}
    agg["segmento"] = agg.apply(
        lambda r: " · ".join(
            f"{labels[d]} {int(r[d])}" if d == "interest1" else f"{labels[d]} {r[d]}"
            for d in dims
        ),
        axis=1,
    )
    return agg.sort_values("cost_per_approved", na_position="last").reset_index(drop=True)


def reference_for_segment(
    df: pd.DataFrame, segment_row: pd.Series, dims: list[str]
) -> Metrics:
    """Resto de la campaña del segmento, excluyéndolo (referencia comparable)."""
    campaign_rows = df[df["campaign_id"] == segment_row["campaign_id"]]
    mask = pd.Series(True, index=campaign_rows.index)
    for dim in dims:
        mask &= campaign_rows[dim] == segment_row[dim]
    rest = campaign_rows.loc[~mask]
    return totals(rest)


@dataclass
class EvidenceThreshold:
    """Umbral configurable de evidencia para recomendar."""

    min_ads: int = config.EVIDENCE_MIN_ADS
    min_approved: int = config.EVIDENCE_MIN_APPROVED
    min_improvement: float = config.EVIDENCE_MIN_IMPROVEMENT
    min_spend_ratio: float = config.EVIDENCE_MIN_SPEND_RATIO


@dataclass
class Recommendation:
    campaign_id: int
    segment_label: str
    dims: list[str]
    segment_values: dict
    segment: Metrics
    reference: Metrics
    improvement: float | None
    spend_share: float
    reasons: list[str] = field(default_factory=list)
    action: str = ""
    trace: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "campaign_id": self.campaign_id,
            "segmento": self.segment_label,
            "dimensiones": self.dims,
            "valores": self.segment_values,
            "segmento_totales": self.segment.to_dict(),
            "referencia_totales": self.reference.to_dict(),
            "mejora_relativa": self.improvement,
            "cuota_de_gasto": self.spend_share,
            "motivos": self.reasons,
            "accion": self.action,
            "trazabilidad": self.trace,
        }


def recommend(
    df: pd.DataFrame,
    threshold: EvidenceThreshold | None = None,
    include_weak: bool = False,
) -> list[Recommendation]:
    """Segmentos candidatos por campaña, con su referencia y su evidencia.

    Reglas (documentadas en el README):
      1. Solo se comparan segmentos dentro de su propia campaña.
      2. Debe existir referencia válida: el resto de la campaña con conversiones.
      3. El segmento debe superar el umbral de evidencia configurado.
      4. Se ordena por coste por conversión aprobada ascendente.

    `include_weak=True` devuelve también los que no superan el umbral, siempre
    que su mejora sea calculable. Se usa para poder plantear una prueba A/B
    cuando no hay candidatos sólidos, declarándolos como evidencia débil.
    """
    threshold = threshold or EvidenceThreshold()
    if df.empty:
        return []
    dims = ["age", "gender", "interest1"]
    segments = _aggregate(df, ["campaign_id"] + dims)
    campaign_totals = totals(df)
    recs: list[Recommendation] = []

    for _, row in segments.iterrows():
        ref = reference_for_segment(df, row, dims)
        cost = row["cost_per_approved"]
        ref_cost = (ref.spend / ref.approved_conversions) if ref.approved_conversions else None
        improvement = (
            (ref_cost - cost) / ref_cost
            if (cost is not None and ref_cost not in (None, 0))
            else None
        )
        campaign_spend = campaign_totals.spend
        spend_share = float(row["spend"] / campaign_spend) if campaign_spend else 0.0

        reasons: list[str] = []
        if row["approved_conversions"] < threshold.min_approved:
            reasons.append(
                f"evidencia insuficiente: {int(row['approved_conversions'])} conversiones aprobadas"
            )
        if row["n_ads"] < threshold.min_ads:
            reasons.append(f"evidencia insuficiente: {int(row['n_ads'])} anuncios (< {threshold.min_ads})")
        if spend_share < threshold.min_spend_ratio:
            reasons.append(
                f"gasto relativo bajo: {spend_share:.1%} del gasto de la campaña "
                f"(< {threshold.min_spend_ratio:.0%})"
            )
        if ref_cost is None:
            reasons.append("sin referencia válida: el resto de la campaña no tiene conversiones aprobadas")
        elif improvement is None:
            reasons.append("relación no calculable")
        elif improvement < threshold.min_improvement:
            reasons.append(
                f"mejora insuficiente: {improvement:.0%} frente al resto de la campaña "
                f"(< {threshold.min_improvement:.0%})"
            )

        label = f"{row['age']} · {row['gender']} · interés {int(row['interest1'])}"
        segment_metrics = metrics_from_sums(
            n_rows=row["n_rows"],
            impressions=row["impressions"],
            clicks=row["clicks"],
            spend=row["spend"],
            total_conversions=row["total_conversions"],
            approved_conversions=row["approved_conversions"],
            n_ads=row["n_ads"],
        )
        rec = Recommendation(
            campaign_id=int(row["campaign_id"]),
            segment_label=label,
            dims=dims,
            segment_values={"age": row["age"], "gender": row["gender"], "interest1": int(row["interest1"])},
            segment=segment_metrics,
            reference=ref,
            improvement=improvement,
            spend_share=spend_share,
            reasons=reasons,
            trace={
                "filas_del_segmento": int(row["n_rows"]),
                "filas_referencia": ref.n_rows,
                "gasto_segmento": row["spend"],
                "gasto_referencia": ref.spend,
                "aprobadas_segmento": int(row["approved_conversions"]),
                "aprobadas_referencia": ref.approved_conversions,
                "campaign_id": int(row["campaign_id"]),
            },
        )
        rec.action = (
            f"Priorizar «{label}» de la campaña {int(row['campaign_id'])} para una "
            f"prueba de inversión controlada."
        )
        recs.append(rec)

    if include_weak:
        valid = [
            r for r in recs
            if r.improvement is not None and r.segment.cost_per_approved is not None
        ]
    else:
        valid = [r for r in recs if not r.reasons]
    valid.sort(key=lambda r: (r.segment.cost_per_approved or float("inf")))
    return valid


def discarded_segment_reasons(
    df: pd.DataFrame, threshold: EvidenceThreshold | None = None
) -> pd.DataFrame:
    """Cuenta los segmentos descartados por cada motivo.

    Sirve para que la vista de recomendaciones explique *por qué* no hay
    candidatos, en vez de solo mostrar "evidencia insuficiente".
    """
    threshold = threshold or EvidenceThreshold()
    if df.empty:
        return pd.DataFrame(columns=["motivo", "segmentos"])
    dims = ["age", "gender", "interest1"]
    segments = _aggregate(df, ["campaign_id"] + dims)
    campaign_totals = totals(df)
    reasons: list[str] = []
    for _, row in segments.iterrows():
        ref = reference_for_segment(df, row, dims)
        ref_cost = (ref.spend / ref.approved_conversions) if ref.approved_conversions else None
        cost = row["cost_per_approved"]
        improvement = (ref_cost - cost) / ref_cost if (cost is not None and ref_cost) else None
        spend_share = (row["spend"] / campaign_totals.spend) if campaign_totals.spend else 0.0

        if row["approved_conversions"] < threshold.min_approved:
            reasons.append("Sin conversiones aprobadas suficientes")
        elif row["n_ads"] < threshold.min_ads:
            reasons.append(f"Menos de {threshold.min_ads} anuncios")
        elif spend_share < threshold.min_spend_ratio:
            reasons.append("Gasto relativo por debajo del umbral")
        elif ref_cost is None:
            reasons.append("Sin referencia válida en su campaña")
        elif improvement is None or improvement < threshold.min_improvement:
            reasons.append(f"Mejora menor que {threshold.min_improvement:.0%}")
        else:
            reasons.append("Cumple el umbral")
    counts = pd.Series(reasons, name="motivo").value_counts()
    return counts.rename_axis("motivo").reset_index(name="segmentos")


def conversion_share_by_campaign(df: pd.DataFrame) -> pd.DataFrame:
    """Distribución descriptiva de conversiones registradas por campaña.

    Es descriptiva: el archivo no contiene recorridos de usuario, así que no
    permite atribución multicanal ni de primer/último contacto.
    """
    if df.empty:
        return pd.DataFrame()
    table = campaign_summary(df)
    total_conv = table["total_conversions"].sum()
    total_appr = table["approved_conversions"].sum()
    table["cuota_conversiones_totales"] = (
        100.0 * table["total_conversions"] / total_conv if total_conv else None
    )
    table["cuota_conversiones_aprobadas"] = (
        100.0 * table["approved_conversions"] / total_appr if total_appr else None
    )
    return table
