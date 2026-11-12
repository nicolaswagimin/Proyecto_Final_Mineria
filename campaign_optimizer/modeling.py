"""Capa de modelado: preparar variables, entrenar, evaluar y guardar.

Decisiones que este módulo hace explícitas y registra en el informe:
  * Momento de la estimación: gasto, impresiones y clics ya conocidos.
  * Partición: dejar una campaña entera fuera (leave-one-campaign-out), porque
    la pregunta es "¿cómo rendirá en campañas nuevas?".
  * Ajuste de hiperparámetros solo con campañas de entrenamiento (CV interna).
"""

from __future__ import annotations

import json
import platform
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config

MODEL_KEYS = ("random_forest", "adaboost", "dummy")
MODEL_LABELS = {
    "random_forest": "Random Forest",
    "adaboost": "AdaBoost",
    "dummy": "Referencia (media)",
}

PARAM_GRIDS: dict[str, list[dict]] = {
    "random_forest": [
        {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 1},
        {"n_estimators": 300, "max_depth": 12, "min_samples_leaf": 1},
        {"n_estimators": 300, "max_depth": 12, "min_samples_leaf": 3},
    ],
    "adaboost": [
        {"n_estimators": 100, "learning_rate": 0.5},
        {"n_estimators": 200, "learning_rate": 0.5},
        {"n_estimators": 200, "learning_rate": 1.0},
    ],
}


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    """Matriz de entrada y objetivos.

    Se excluyen identificadores (ad_id, fb_campaign_id, campaign_id) y los dos
    objetivos. No se incluyen derivados de conversiones.
    """
    feature_cols = list(config.FEATURE_NUMERIC) + list(config.FEATURE_CATEGORICAL)
    X = df[feature_cols].copy()
    X["interest1"] = X["interest1"].astype("string")
    for col in config.FEATURE_CATEGORICAL:
        X[col] = X[col].astype("string").fillna("desconocido")
    for col in config.FEATURE_NUMERIC:
        X[col] = pd.to_numeric(X[col], errors="coerce").astype("float64")

    targets = {name: df[col].astype("float64") for name, col in config.TARGETS.items()}
    return X, targets


def make_pipeline(model_key: str, params: dict | None = None, seed: int = config.SEED) -> Pipeline:
    params = dict(params or {})
    numeric = list(config.FEATURE_NUMERIC)
    categorical = list(config.FEATURE_CATEGORICAL)
    pre = ColumnTransformer(
        [
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    if model_key == "random_forest":
        # n_jobs=1: con ~1.100 filas el paralelismo no aporta y evita depender de
        # named pipes (bloqueados en algunos entornos restringidos).
        est = RandomForestRegressor(random_state=seed, n_jobs=1, **params)
    elif model_key == "adaboost":
        est = AdaBoostRegressor(random_state=seed, **params)
    elif model_key == "dummy":
        est = DummyRegressor(strategy="mean")
    else:
        raise ValueError(f"modelo desconocido: {model_key}")
    return Pipeline([("pre", pre), ("model", est)])


def campaign_folds(df: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray, int]]:
    """Leave-one-campaign-out: una campaña completa fuera en cada vuelta.

    Si solo hay una campaña, no hay partición honesta posible: se devuelve una
    única partición aleatoria y el informe lo marca como limitación.
    """
    campaigns = sorted(df["campaign_id"].dropna().unique())
    if len(campaigns) >= 2:
        folds = []
        for camp in campaigns:
            test_mask = (df["campaign_id"] == camp).to_numpy()
            folds.append((np.where(~test_mask)[0], np.where(test_mask)[0], int(camp)))
        return folds
    rng = np.random.default_rng(config.SEED)
    idx = rng.permutation(len(df))
    split = int(len(df) * 0.8)
    return [(idx[:split], idx[split:], int(campaigns[0]) if campaigns else -1)]


def _inner_tune(
    model_key: str, X_train: pd.DataFrame, y_train: pd.Series, groups: np.ndarray,
    seed: int = config.SEED,
) -> tuple[dict, float]:
    """Elige hiperparámetros con CV agrupada SOLO en datos de entrenamiento."""
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        grid = PARAM_GRIDS[model_key][0]
        return grid, float("nan")

    n_splits = min(3, len(unique_groups))
    cv = GroupKFold(n_splits=n_splits)
    best_params, best_score = None, np.inf
    for candidate in PARAM_GRIDS[model_key]:
        scores = []
        for tr, va in cv.split(X_train, y_train, groups):
            pipe = make_pipeline(model_key, candidate, seed=seed)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pipe.fit(X_train.iloc[tr], y_train.iloc[tr])
                pred = pipe.predict(X_train.iloc[va])
            scores.append(mean_absolute_error(y_train.iloc[va], pred))
        score = float(np.mean(scores))
        if score < best_score:
            best_params, best_score = dict(candidate), score
    return dict(best_params), best_score


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else None,
        "n": int(len(y_true)),
    }


@dataclass
class TrainingOutput:
    predictions: pd.DataFrame
    metrics: pd.DataFrame
    per_campaign: pd.DataFrame
    report: dict


def train(
    df: pd.DataFrame,
    quality: dict | None = None,
    seed: int = config.SEED,
    tune: bool = True,
) -> TrainingOutput:
    """Entrena los dos algoritmos para los dos objetivos y evalúa fuera de muestra."""
    X, targets = build_features(df)
    folds = campaign_folds(df)

    pred_rows: list[dict] = []
    metric_rows: list[dict] = []
    per_campaign_rows: list[dict] = []
    chosen: dict[str, dict] = {}

    for target_name, y in targets.items():
        for train_idx, test_idx, heldout_campaign in folds:
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            groups = df["campaign_id"].iloc[train_idx].to_numpy()

            fold_preds: dict[str, np.ndarray] = {}
            for model_key in MODEL_KEYS:
                params: dict = {}
                inner_mae = float("nan")
                if tune and model_key in PARAM_GRIDS:
                    params, inner_mae = _inner_tune(model_key, X_train, y_train, groups, seed=seed)
                    chosen.setdefault(f"{target_name}/{model_key}", []).append(
                        {
                            "params": params,
                            "inner_mae": inner_mae,
                            "held_out_campaign": heldout_campaign,
                        }
                    )
                pipe = make_pipeline(model_key, params, seed=seed)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    pipe.fit(X_train, y_train)
                    pred = pipe.predict(X_test)
                fold_preds[model_key] = np.asarray(pred, dtype=float)

                m = _metrics(y_test.to_numpy(), fold_preds[model_key])
                metric_rows.append(
                    {
                        "objetivo": target_name,
                        "modelo": model_key,
                        "modelo_etiqueta": MODEL_LABELS[model_key],
                        "campaña_evaluada": heldout_campaign,
                        **m,
                    }
                )

            for row_i, pos in enumerate(test_idx):
                pred_rows.append(
                    {
                        "objetivo": target_name,
                        "campaña_evaluada": heldout_campaign,
                        "ad_id": int(df["ad_id"].iloc[pos]),
                        "campaign_id": int(df["campaign_id"].iloc[pos]),
                        "observed": float(y.iloc[pos]),
                        "pred_random_forest": fold_preds["random_forest"][row_i],
                        "pred_adaboost": fold_preds["adaboost"][row_i],
                        "pred_dummy": fold_preds["dummy"][row_i],
                    }
                )

    predictions = pd.DataFrame(pred_rows)
    metrics_raw = pd.DataFrame(metric_rows)

    # Métricas agregadas por objetivo y modelo (sobre predicciones fuera de muestra)
    agg_rows = []
    for (target_name, model_key), group in metrics_raw.groupby(["objetivo", "modelo"]):
        preds = predictions[
            (predictions["objetivo"] == target_name)
        ]
        observed = preds["observed"].to_numpy()
        model_pred = preds[f"pred_{model_key}"].to_numpy()
        m = _metrics(observed, model_pred)
        agg_rows.append(
            {
                "objetivo": target_name,
                "modelo": model_key,
                "modelo_etiqueta": MODEL_LABELS[model_key],
                **m,
            }
        )
    metrics_table = pd.DataFrame(agg_rows)

    # Errores por campaña (incertidumbre cuando hay pocas campañas)
    for target_name in targets:
        preds_t = predictions[predictions["objetivo"] == target_name]
        for model_key in MODEL_KEYS:
            for camp, g in preds_t.groupby("campaña_evaluada"):
                m = _metrics(g["observed"].to_numpy(), g[f"pred_{model_key}"].to_numpy())
                per_campaign_rows.append(
                    {
                        "objetivo": target_name,
                        "modelo": model_key,
                        "campaña_evaluada": int(camp),
                        "n": m["n"],
                        "mae": m["mae"],
                        "rmse": m["rmse"],
                    }
                )
    per_campaign = pd.DataFrame(per_campaign_rows)

    dummy_mae = {(r.objetivo, r.modelo): r.mae for r in metrics_table.itertuples()}
    baseline_mae = {target: dummy_mae.get((target, "dummy")) for target in config.TARGETS}
    reference_beaten = {}
    for r in metrics_table.itertuples():
        if r.modelo == "dummy":
            continue
        base = baseline_mae.get(r.objetivo)
        reference_beaten[f"{r.objetivo}/{r.modelo}"] = bool(base is not None and r.mae < base)
    report = {
        "generado_en": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset_id": (quality or {}).get("dataset_id"),
        "source_sha256": (quality or {}).get("source_sha256"),
        "n_rows": int(len(df)),
        "n_campaigns": int(df["campaign_id"].nunique()),
        "semilla": seed,
        "escenario": config.SCENARIO_TEXT,
        "variables_entrada": list(config.FEATURE_NUMERIC) + list(config.FEATURE_CATEGORICAL),
        "variables_excluidas": list(config.ID_FIELDS) + list(config.TARGETS.values()),
        "particion": "leave-one-campaign-out" if df["campaign_id"].nunique() >= 2 else "aleatoria 80/20 (limitación: una sola campaña)",
        "objetivos": list(config.TARGETS),
        "hiperparametros_elegidos": chosen,
        "metricas": metrics_table.to_dict(orient="records"),
        "metricas_por_campaña": per_campaign.to_dict(orient="records"),
        "referencia_superada": reference_beaten,
        "versiones": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__,
            "numpy": np.__version__,
        },
        "calidad_datos": quality or {},
    }
    return TrainingOutput(predictions, metrics_table, per_campaign, report)


def fit_final_models(
    df: pd.DataFrame,
    params_by_key: dict[str, dict] | None = None,
    seed: int = config.SEED,
) -> dict[str, Pipeline]:
    """Modelos finales sobre todos los datos (para uso futuro, no para evaluar).

    `params_by_key` permite reutilizar los hiperparámetros elegidos en la
    validación agrupada, en vez de los de partida.
    """
    params_by_key = params_by_key or {}
    X, targets = build_features(df)
    saved: dict[str, Pipeline] = {}
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for target_name, y in targets.items():
        for model_key in ("random_forest", "adaboost"):
            params = params_by_key.get(f"{target_name}/{model_key}") or PARAM_GRIDS[model_key][0]
            pipe = make_pipeline(model_key, params, seed=seed)
            pipe.fit(X, y)
            path = config.MODELS_DIR / f"{target_name}__{model_key}.joblib"
            joblib.dump(pipe, path)
            saved[f"{target_name}/{model_key}"] = path
    return saved


def write_artifacts(output: TrainingOutput, quality: dict) -> dict[str, str]:
    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    paths = {
        "predicciones": config.PREDICTIONS_CSV,
        "informe": config.TRAIN_REPORT,
        "calidad": config.QUALITY_REPORT,
    }
    output.predictions.to_csv(config.PREDICTIONS_CSV, index=False)
    config.TRAIN_REPORT.write_text(
        json.dumps(output.report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    config.QUALITY_REPORT.write_text(
        json.dumps(quality, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {k: str(v) for k, v in paths.items()}


def load_report() -> dict | None:
    if not config.TRAIN_REPORT.exists():
        return None
    try:
        return json.loads(config.TRAIN_REPORT.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def load_predictions() -> pd.DataFrame | None:
    if not config.PREDICTIONS_CSV.exists():
        return None
    try:
        return pd.read_csv(config.PREDICTIONS_CSV)
    except Exception:
        return None


def report_is_compatible(report: dict | None, quality: dict) -> bool:
    """Los resultados solo sirven si se generaron con este mismo CSV."""
    if not report:
        return False
    return (
        report.get("dataset_id") == quality.get("dataset_id")
        and report.get("source_sha256") == quality.get("source_sha256")
        and report.get("n_rows") == quality.get("n_rows_clean")
    )
