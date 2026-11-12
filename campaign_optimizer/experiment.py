"""Diseño de una prueba A/B a partir de un hallazgo histórico.

La distinción que este módulo hace explícita:

  * El **69 %** de una recomendación es una diferencia *transversal* (ese
    segmento frente al resto de su campaña, en el pasado).
  * La prueba A/B mide un **efecto de intervención**: qué cambia *dentro del
    segmento* al subir el presupuesto.

Son dos comparaciones distintas. Usar la primera como mejora exigida a la
segunda sería un error de diseño, así que aquí no se usa.

El tamaño de muestra se calcula para el efecto que interesa detectar, no para
el observado: se asumen conteos de Poisson (conversiones aprobadas), dos brazos
iguales, y un contraste bilateral con la potencia elegida.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from statistics import NormalDist

from .analytics import Metrics


@dataclass
class ABPlan:
    campaign_id: int
    segment_label: str
    comparable_ads: int
    window_days: int
    baseline_approved: int
    cost_per_approved: float
    cost_ceiling: float
    delta: float
    alpha: float
    power: float
    n_per_arm: int
    approved_per_day: float
    daily_spend: float
    uplift: float
    exposure_share: float
    arm_days: float
    calendar_days: float
    calendar_days_ceiled: int
    test_budget: float
    design: str
    design_reason: str
    feasible: bool
    max_days: int
    assumptions: list[str] = field(default_factory=list)
    scenarios: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _z(prob: float) -> float:
    return NormalDist().inv_cdf(prob)


def required_sample(delta: float, alpha: float = 0.05, power: float = 0.80) -> int:
    """Conversiones aprobadas necesarias POR BRAZO para detectar `delta`.

    Aproximación normal para conteos de Poisson con igual exposición:
        n = 2 · (z_{1-α/2} + z_potencia)² / δ²
    """
    if not 0 < delta < 1:
        raise ValueError("delta debe estar entre 0 y 1")
    z_alpha = _z(1 - alpha / 2)
    z_power = _z(power)
    return math.ceil(2 * (z_alpha + z_power) ** 2 / delta**2)


def plan_ab_test(
    segment: Metrics,
    campaign_id: int,
    segment_label: str,
    window_days: int,
    delta: float = 0.25,
    alpha: float = 0.05,
    power: float = 0.80,
    cost_tolerance: float = 0.30,
    max_days: int = 60,
    min_ads_for_randomization: int = 8,
    exposure_share: float = 0.50,
    uplift: float = 0.25,
) -> ABPlan:
    """Plan reproducible: unidades, techo de coste, muestra y calendario.

    `window_days` es la ventana observada de la que salen las conversiones del
    segmento. La conversión de "conversiones necesarias" a calendario depende
    del reparto de exposición: con 50/50, cada brazo recibe la mitad, así que los
    días de calendario son el doble de los días de exposición a tasa plena.

    El presupuesto de la prueba se calcula desde el gasto diario y los días de
    exposición: control a gasto actual y variante con el incremento `uplift`.
    No se valora ninguna conversión a coste histórico: el coste por conversión
    de la variante es justamente una de las incógnitas del experimento.
    """
    if segment.cost_per_approved is None or segment.approved_conversions <= 0:
        raise ValueError("el segmento necesita coste por conversión calculable")
    if window_days <= 0:
        raise ValueError("window_days debe ser positivo")
    if not 0 < exposure_share <= 1:
        raise ValueError("exposure_share debe estar en (0, 1]")
    if uplift <= 0:
        raise ValueError("uplift debe ser positivo")

    cost = float(segment.cost_per_approved)
    ceiling = cost * (1 + cost_tolerance)
    per_day = segment.approved_conversions / window_days
    daily_spend = segment.spend / window_days

    n_per_arm = required_sample(delta, alpha, power)
    arm_days = n_per_arm / per_day if per_day else float("inf")
    calendar_days = arm_days / exposure_share
    # Días completos necesarios: se redondea hacia arriba para planificar.
    calendar_days_ceiled = math.ceil(calendar_days)
    # Presupuesto: cada condición consume `arm_days` de gasto diario; la variante
    # lo hace con el incremento aplicado (uplift), el control a gasto actual.
    test_budget = arm_days * daily_spend * (1 + (1 + uplift))

    if segment.n_ads >= min_ads_for_randomization:
        design = "Aleatorización por anuncio, estratificada por gasto y antigüedad"
        reason = (
            f"El segmento tiene {segment.n_ads} anuncios comparables: permite sortear "
            "anuncios a control y variante dentro de estratos de gasto y antigüedad, "
            "de modo que ambos brazos contienen anuncios de la misma edad media. "
            "Cada brazo recibe ~50 % del volumen diario, así que la tasa por brazo es "
            "la mitad de la observada."
        )
    else:
        design = "Switchback por bloques temporales (mismos anuncios en ambos brazos)"
        reason = (
            f"El segmento solo tiene {segment.n_ads} anuncios: sortearlos dejaría "
            "brazos de 1-2 anuncios y mezclaría el efecto del presupuesto con la "
            "antigüedad del creativo. Se alterna el bloque completo entre control y "
            "variante: cada condición está activa ~50 % de los días a tasa plena, así "
            "que el calendario duplica los días de exposición."
        )

    scenarios = []
    for d in (0.25, 0.40, 0.50, 0.75):
        n = required_sample(d, alpha, power)
        exposure = n / per_day if per_day else float("inf")
        calendar = exposure / exposure_share
        budget = exposure * daily_spend * (1 + (1 + uplift))
        scenarios.append(
            {
                "delta": d,
                "n_por_brazo": n,
                "dias_exposicion": exposure,
                "dias_calendario": calendar,
                "dias_calendario_plan": math.ceil(calendar),
                "presupuesto": budget,
                "cabe_en_plazo": math.ceil(calendar) <= max_days,
            }
        )

    assumptions = [
        f"Reparto de exposición asumido: {exposure_share:.0%}/{1 - exposure_share:.0%} "
        "entre control y variante; los días de calendario ya lo incorporan.",
        f"Presupuesto = días de exposición × gasto diario del segmento × "
        f"(1 + {1 + uplift:.2f}): el control gasta {1:.0%} y la variante "
        f"{1 + uplift:.0%}. No es una valoración de conversiones: el coste por "
        "conversión de la variante es una de las incógnitas del experimento.",
        "Conteos de Poisson, dos brazos iguales y contraste bilateral: es una "
        "aproximación de potencia, no una simulación del día a día.",
        "No incluye la variación entre días ni el retardo de registro de "
        "conversiones: la ventana de medición debe extenderse hasta que la última "
        "conversión atribuible quede registrada (criterio a fijar antes de lanzar).",
        "El techo de coste es una propuesta (coste observado + margen); convertirlo "
        "en límite operativo es una decisión de negocio previa al lanzamiento.",
    ]

    return ABPlan(
        campaign_id=campaign_id,
        segment_label=segment_label,
        comparable_ads=int(segment.n_ads),
        window_days=int(window_days),
        baseline_approved=int(segment.approved_conversions),
        cost_per_approved=cost,
        cost_ceiling=ceiling,
        delta=delta,
        alpha=alpha,
        power=power,
        n_per_arm=n_per_arm,
        approved_per_day=per_day,
        daily_spend=daily_spend,
        uplift=uplift,
        exposure_share=exposure_share,
        arm_days=arm_days,
        calendar_days=calendar_days,
        calendar_days_ceiled=calendar_days_ceiled,
        test_budget=test_budget,
        design=design,
        design_reason=reason,
        feasible=calendar_days_ceiled <= max_days,
        max_days=max_days,
        assumptions=assumptions,
        scenarios=scenarios,
    )
