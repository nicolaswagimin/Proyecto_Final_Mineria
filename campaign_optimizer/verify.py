"""Verificación de los criterios de aceptación, con recálculo independiente.

No reutiliza `analytics` para comprobar las cifras: vuelve a calcularlas desde
el CSV original con pandas puro. Si las dos vías coinciden, la interfaz no puede
estar enseñando un número distinto al de las filas.

Uso:
    python -m campaign_optimizer.verify
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import analytics, config
from .data import load_dataset
from .modeling import (
    build_features,
    campaign_folds,
    load_predictions,
    load_report,
    report_is_compatible,
)

PASS, FAIL = "PASS", "FAIL"
results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))
    print(f"[{PASS if ok else FAIL}] {name}" + (f" — {detail}" if detail else ""))


def _independent_totals(raw: pd.DataFrame, campaign: int) -> dict:
    """Sumas directas sobre el CSV crudo, sin pasar por `analytics`.

    Usa el mapeo de columnas del contrato (no las fórmulas de la app), de modo
    que sigue siendo un cálculo independiente.
    """
    from .data import resolve_columns

    cols = resolve_columns(raw)
    camp = pd.to_numeric(raw[cols["campaign_id"]], errors="coerce")
    sub = raw[camp == campaign]
    spend = pd.to_numeric(sub[cols["spend"]], errors="coerce").sum()
    impressions = pd.to_numeric(sub[cols["impressions"]], errors="coerce").sum()
    clicks = pd.to_numeric(sub[cols["clicks"]], errors="coerce").sum()
    total = pd.to_numeric(sub[cols["total_conversions"]], errors="coerce").sum()
    approved = pd.to_numeric(sub[cols["approved_conversions"]], errors="coerce").sum()
    return {
        "spend": float(spend),
        "impressions": int(impressions),
        "clicks": int(clicks),
        "total_conversions": int(total),
        "approved_conversions": int(approved),
        "cost_per_approved": float(spend / approved) if approved else None,
    }


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    print("== Verificación de criterios de aceptación ==\n")

    # 1. Datos presentes y esquema
    try:
        df, quality = load_dataset(config.RAW_CSV)
    except FileNotFoundError as exc:
        print(f"[{FAIL}] No hay dataset: {exc}")
        return 2
    check(
        "CSV cargado y validado",
        quality.n_rows_clean > 0,
        f"{quality.n_rows_clean} filas válidas de {quality.n_rows_raw}, "
        f"huella {quality.dataset_id}",
    )

    raw = pd.read_csv(config.RAW_CSV, dtype=str)

    # 2. Totales por campaña coinciden con el cálculo independiente
    ok_all = True
    detail = []
    for campaign in sorted(df["campaign_id"].unique()):
        ui = analytics.totals(df[df["campaign_id"] == campaign])
        ind = _independent_totals(raw, int(campaign))
        same = (
            np.isclose(ui.spend, ind["spend"])
            and ui.impressions == ind["impressions"]
            and ui.clicks == ind["clicks"]
            and ui.total_conversions == ind["total_conversions"]
            and ui.approved_conversions == ind["approved_conversions"]
            and (
                (ui.cost_per_approved is None and ind["cost_per_approved"] is None)
                or np.isclose(ui.cost_per_approved, ind["cost_per_approved"] or 0)
            )
        )
        ok_all &= same
        detail.append(f"campaña {campaign}: {'ok' if same else 'DIFIERE'}")
    check("Totales y gasto/conversión por campaña", ok_all, "; ".join(detail))

    # 3. Filtro vacío y segmento sin conversiones
    vacio = analytics.filter_records(df, campaigns=[-1])
    m_vacio = analytics.totals(vacio)
    check(
        "Filtro vacío produce estado comprensible",
        vacio.empty and m_vacio.n_rows == 0 and m_vacio.cost_per_approved is None
        and m_vacio.ctr is None,
        "totales a cero y ratios 'No calculable'",
    )

    segmentos = analytics.segment_summary(df)
    sin_conv = segmentos[segmentos["approved_conversions"] == 0]
    check(
        "Segmentos con gasto y cero conversiones aprobadas",
        sin_conv["cost_per_approved"].isna().all() if not sin_conv.empty else True,
        f"{len(sin_conv)} segmento(s) afectados" if not sin_conv.empty else "no hay ninguno",
    )

    # 4. Artefactos de entrenamiento
    report = load_report()
    preds = load_predictions()
    if report is None or preds is None:
        check("Resultados de entrenamiento", False, "faltan; ejecuta el entrenamiento")
        _summary()
        return 1

    compatible = report_is_compatible(report, quality.to_dict())
    check("Los resultados corresponden a este CSV", compatible,
          f"informe {report.get('dataset_id')} vs datos {quality.dataset_id}")
    if not compatible:
        _summary()
        return 1

    esperados = {(t, m) for t in config.TARGETS for m in ("random_forest", "adaboost", "dummy")}
    obtenidos = {(r["objetivo"], r["modelo"]) for r in report["metricas"]}
    check(
        "4 ajustes pedidos + 2 referencias presentes",
        esperados <= obtenidos,
        f"{len(obtenidos)} combinaciones: " + ", ".join(sorted(f"{t}/{m}" for t, m in obtenidos)),
    )

    for target in config.TARGETS:
        sub = preds[preds["objetivo"] == target]
        check(
            f"Predicciones fuera de muestra completas ({target})",
            len(sub) == len(df),
            f"{len(sub)} de {len(df)} registros",
        )

    # 5. Cada punto proviene de un modelo que no vio su campaña
    folds = campaign_folds(df)
    sin_fuga = True
    for target in config.TARGETS:
        sub = preds[preds["objetivo"] == target]
        for train_idx, test_idx, camp in folds:
            evaluados = set(df["ad_id"].iloc[test_idx])
            entrenados = set(df["ad_id"].iloc[train_idx])
            sin_fuga &= evaluados.isdisjoint(entrenados)
            filas = sub[sub["campaña_evaluada"] == camp]
            sin_fuga &= set(filas["ad_id"]) == evaluados
    check("Sin fuga: cada predicción es de una campaña no vista", sin_fuga)

    # 6. Recomendaciones trazables hasta las filas
    recs = analytics.recommend(df)
    if not recs:
        check("Recomendaciones trazables", True, "sin candidatos: 'Evidencia insuficiente'")
    else:
        ok = True
        for rec in recs:
            filas = analytics.filter_records(
                df,
                campaigns=[rec.campaign_id],
                ages=[rec.segment_values["age"]],
                genders=[rec.segment_values["gender"]],
                interests=[rec.segment_values["interest1"]],
            )
            gasto = filas["spend"].sum()
            aprobadas = filas["approved_conversions"].sum()
            ok &= np.isclose(gasto, rec.trace["gasto_segmento"])
            ok &= aprobadas == rec.trace["aprobadas_segmento"]
            ok &= np.isclose(rec.segment.cost_per_approved, gasto / aprobadas)
        check("Cada recomendación se traza a gasto, volumen y conversiones", ok,
              f"{len(recs)} recomendación(es)")

    # 7. Plan experimental reproducible a partir del hallazgo
    if recs:
        from .experiment import plan_ab_test

        top = recs[0]
        fechas = quality.date_range or {}
        window = 14
        if fechas.get("min") and fechas.get("max"):
            window = (pd.Timestamp(fechas["max"]) - pd.Timestamp(fechas["min"])).days + 1

        plan = plan_ab_test(
            top.segment,
            campaign_id=top.campaign_id,
            segment_label=top.segment_label,
            window_days=window,
        )
        ok_plan = (
            abs(plan.cost_ceiling - plan.cost_per_approved * 1.30) < 1e-9
            and plan.n_per_arm > 0
            and abs(plan.calendar_days - plan.arm_days / plan.exposure_share) < 1e-6
            and plan.calendar_days_ceiled == math.ceil(plan.calendar_days)
            and plan.feasible == (plan.calendar_days_ceiled <= plan.max_days)
            and all(
                s["dias_calendario_plan"] == math.ceil(s["dias_calendario"])
                for s in plan.scenarios
            )
            and abs(
                plan.test_budget
                - plan.arm_days * plan.daily_spend * (1 + (1 + plan.uplift))
            ) < 1e-6
            and plan.window_days == window
        )
        check(
            "Plan A/B calculado desde las cifras del segmento",
            ok_plan,
            f"{plan.n_per_arm} aprobadas/brazo, {plan.calendar_days:.0f} días de calendario, "
            f"techo {plan.cost_ceiling:.2f}",
        )

    # 8. Reproducibilidad del informe
    check(
        "Partición registrada es la prometida",
        report["particion"].startswith("leave-one-campaign-out") if df["campaign_id"].nunique() >= 2
        else report["particion"].startswith("aleatoria"),
        report["particion"],
    )

    return _summary()


def _summary() -> int:
    fallos = [r for r in results if not r[1]]
    print(f"\n{len(results) - len(fallos)}/{len(results)} comprobaciones superadas.")
    if fallos:
        print("Fallos:")
        for name, _, detail in fallos:
            print(f"  - {name}: {detail}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
