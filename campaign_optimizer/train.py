"""Punto de entrada de entrenamiento.

Uso:
    python -m campaign_optimizer.train
    python -m campaign_optimizer.train --csv data/raw/facebook_ads.csv --sin-ajuste

Escribe en artifacts/: predictions.csv, train_report.json, data_quality.json,
y los modelos finales en artifacts/models/.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import config
from .data import load_dataset
from .modeling import fit_final_models, train, write_artifacts


def _best_params(report: dict) -> dict[str, dict]:
    """Hiperparámetros más votados entre las vueltas de validación."""
    out: dict[str, dict] = {}
    for key, entries in report.get("hiperparametros_elegidos", {}).items():
        votos: dict[str, list[dict]] = {}
        for entry in entries:
            firma = json.dumps(entry["params"], sort_keys=True)
            votos.setdefault(firma, []).append(entry["params"])
        out[key] = max(votos.values(), key=len)[0]
    return out


def _fmt(value: float | None, digits: int = 3) -> str:
    return "n/d" if value is None else f"{value:.{digits}f}"


def main(argv: list[str] | None = None) -> int:
    # Consola de Windows: evita mojibake con acentos al redirigir la salida.
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    parser = argparse.ArgumentParser(
        description="Entrena Random Forest y AdaBoost para conversiones totales y aprobadas."
    )
    parser.add_argument("--csv", type=Path, default=config.RAW_CSV, help="ruta del CSV original")
    parser.add_argument("--semilla", type=int, default=config.SEED, help="semilla aleatoria")
    parser.add_argument(
        "--sin-ajuste",
        action="store_true",
        help="no hace búsqueda de hiperparámetros (más rápido, menos honesto)",
    )
    parser.add_argument(
        "--sin-modelos-finales",
        action="store_true",
        help="no reentrena sobre todos los datos para guardar los .joblib",
    )
    args = parser.parse_args(argv)

    print("== 1/4 Leyendo y validando datos ==")
    try:
        df, quality = load_dataset(args.csv)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(f"   filas originales : {quality.n_rows_raw}")
    print(f"   filas válidas    : {quality.n_rows_clean}")
    print(f"   campañas         : {df['campaign_id'].nunique()}")
    print(f"   dataset_id       : {quality.dataset_id}")
    print(f"   duplicados       : {quality.duplicate_rows}")
    print(f"   negativos        : {quality.negatives}")
    print(f"   aprobadas > total: {quality.approved_gt_total}")
    print(f"   conv. sin clics  : {quality.conversions_without_clicks}")

    print("\n== 2/4 Entrenando (leave-one-campaign-out) ==")
    output = train(df, quality.to_dict(), seed=args.semilla, tune=not args.sin_ajuste)

    print("\n== 3/4 Resultados fuera de muestra ==")
    header = f"   {'objetivo':<24}{'modelo':<22}{'n':>6}{'MAE':>9}{'RMSE':>9}"
    print(header)
    print("   " + "-" * (len(header) - 3))
    for row in output.metrics.sort_values(["objetivo", "mae"]).itertuples():
        print(
            f"   {row.objetivo:<24}{row.modelo_etiqueta:<22}{row.n:>6}"
            f"{row.mae:>9.3f}{row.rmse:>9.3f}"
        )

    print("\n   Error por campaña (fuera de muestra; mide la incertidumbre):")
    for target in output.per_campaign["objetivo"].unique():
        sub = output.per_campaign[output.per_campaign["objetivo"] == target]
        for model_key in ("random_forest", "adaboost", "dummy"):
            vals = sub[sub["modelo"] == model_key]
            if vals.empty:
                continue
            mae_min, mae_max = vals["mae"].min(), vals["mae"].max()
            print(f"     {target:<24}{model_key:<15} MAE entre {mae_min:.3f} y {mae_max:.3f}")

    print("\n== 4/4 Guardando artefactos ==")
    if not args.sin_modelos_finales:
        saved = fit_final_models(
            df, params_by_key=_best_params(output.report), seed=args.semilla
        )
        for name, path in saved.items():
            print(f"   modelo final {name}: {path.name}")
    paths = write_artifacts(output, quality.to_dict())
    for name, path in paths.items():
        print(f"   {name}: {path}")
    print("\nListo. Abre el dashboard con:  streamlit run app.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
