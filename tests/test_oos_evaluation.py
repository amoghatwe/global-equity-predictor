"""
Out-of-Sample Evaluation: Ensemble vs. Baselines
=================================================

Walk-forward evaluation with a 36-month train/test gap (no label leakage
from overlapping 3-year forward-return targets). Compares the system's
ensemble (Ridge + RF + XGBoost equal-weight average) against:

- Historical-mean baseline: constant prediction of the training-set mean
  ("no-skill" floor).
- ARIMA(1,1,1) baseline: univariate forecast on the target series itself.

Prints a fold-level comparison table and writes reports/oos_results.json.

Requires data/features/features.csv. Skips with a clear message otherwise.
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config.settings import DATA_FEATURES_PATH, MODEL_CONFIG, MARKETS  # noqa: E402
from src.models.base import TimeSeriesCrossValidator  # noqa: E402
from src.models.implementations import create_model  # noqa: E402

FEATURES_PATH = Path(DATA_FEATURES_PATH) / "features.csv"
RESULTS_PATH = PROJECT_ROOT / "reports" / "oos_results.json"

# Feature columns with <90% coverage are dropped before training; matches
# ModelTrainingPipeline._train_single_market.
MIN_COLUMN_COVERAGE = 0.9


def _load_matrix(market: str) -> "tuple[pd.DataFrame, pd.Series]":
    """Load real features, apply the pipeline's market-specific selection."""
    from src.models.pipeline import ModelTrainingPipeline

    pipe = ModelTrainingPipeline(model_selection="ensemble")
    pipe.load_feature_matrix()
    target_col = f"{market}_target_return"
    X = pipe._select_features_for_market(market)
    combined = X.join(pipe.feature_matrix[target_col])
    coverage = combined.drop(columns=[target_col]).notna().mean()
    combined = combined.drop(columns=coverage[coverage < MIN_COLUMN_COVERAGE].index).dropna()
    X = combined.drop(columns=[target_col])
    y = combined[target_col]
    return X, y


def _fold_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """RMSE, MAE, R2, directional accuracy, Pearson correlation."""
    err = y_true - y_pred
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    dir_acc = float(np.mean(np.sign(y_true) == np.sign(y_pred)))
    if np.std(y_true) > 0 and np.std(y_pred) > 0:
        corr = float(np.corrcoef(y_true, y_pred)[0, 1])
    else:
        corr = float("nan")
    return {"rmse": rmse, "mae": mae, "r2": r2, "dir_acc": dir_acc, "corr": corr}


def _predict_ensemble(X_train, y_train, X_test) -> np.ndarray:
    """Equal-weight average of Ridge, RF, and XGBoost predictions."""
    preds = []
    for m_type in ("linear", "rf", "xgb"):
        model = create_model(m_type)
        model.train(X_train, y_train)
        preds.append(np.asarray(model.predict(X_test), dtype=float))
    return np.mean(preds, axis=0)


def _predict_arima(y_train: pd.Series, n_steps: int) -> np.ndarray:
    """ARIMA(1,1,1) on the univariate target; forecast n_steps ahead."""
    from statsmodels.tsa.arima.model import ARIMA

    fit = ARIMA(y_train.to_numpy(), order=(1, 1, 1)).fit()
    return np.asarray(fit.forecast(steps=n_steps), dtype=float)


def _agg(fold_values: "list[float]") -> "tuple[float, float]":
    """Mean and population std across folds."""
    arr = np.asarray([v for v in fold_values if not math.isnan(v)], dtype=float)
    if len(arr) == 0:
        return float("nan"), float("nan")
    return float(arr.mean()), float(arr.std())


def test_oos_vs_baselines():
    if not FEATURES_PATH.exists():
        pytest.skip(f"Feature matrix not found at {FEATURES_PATH}; run data collection first")

    gap = MODEL_CONFIG.get("gap_months", 36)
    min_train_months = MODEL_CONFIG["min_train_years"] * 12
    n_splits = MODEL_CONFIG["n_splits"]

    results = {"protocol": {
        "gap_months": gap,
        "min_train_months": min_train_months,
        "n_splits_requested": n_splits,
        "feature_coverage_threshold": MIN_COLUMN_COVERAGE,
    }, "markets": {}}

    for market in MARKETS:
        X, y = _load_matrix(market)
        usable = len(X) - min_train_months - gap
        if len(X) == 0 or usable < 24:  # need >=2 years of testable data
            print(f"[{market}] insufficient data ({len(X)} rows after cleaning); skipping")
            continue

        cv = TimeSeriesCrossValidator(
            n_splits=min(n_splits, usable // 12),
            min_train_years=MODEL_CONFIG["min_train_years"],
            gap_months=gap,
        )
        folds = list(cv.split(X))
        if not folds:
            print(f"[{market}] no CV folds possible; skipping")
            continue

        market_folds = []
        for i, (train_idx, test_idx) in enumerate(folds):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            yt = y_test.to_numpy()

            preds = {"Ensemble": _predict_ensemble(X_train, y_train, X_test)}
            preds["Historical Mean"] = np.full(len(yt), y_train.mean())
            preds["ARIMA(1,1,1)"] = _predict_arima(y_train, len(yt))

            fold_rec = {"fold": i, "n_train": len(train_idx), "n_test": len(test_idx),
                        "metrics": {}}
            for name, p in preds.items():
                fold_rec["metrics"][name] = _fold_metrics(yt, p)
            market_folds.append(fold_rec)

        # Aggregate mean +/- std across folds
        agg = {}
        model_names = [k for k in market_folds[0]["metrics"]]
        for name in model_names:
            agg[name] = {}
            for metric in ("rmse", "mae", "r2", "dir_acc", "corr"):
                mean, std = _agg([f["metrics"][name][metric] for f in market_folds])
                agg[name][f"{metric}_mean"] = mean
                agg[name][f"{metric}_std"] = std

        results["markets"][market] = {
            "n_rows": len(X), "date_span": [str(X.index[0].date()), str(X.index[-1].date())],
            "n_folds": len(market_folds), "folds": market_folds, "aggregate": agg,
        }

        print(f"\n{'=' * 70}\n{market}: {len(X)} rows ({X.index[0].date()} → {X.index[-1].date()}), "
              f"{len(market_folds)} folds, gap={gap}m\n{'=' * 70}")
        header = f"{'Model':<18}{'RMSE':>14}{'MAE':>14}{'R2':>14}{'DirAcc':>14}{'Corr':>14}"
        print(header)
        print("-" * len(header))
        for name in model_names:
            a = agg[name]
            def cell(m, s):
                return f"{m:.3f}±{s:.3f}" if not math.isnan(m) else "N/A"
            print(f"{name:<18}"
                  f"{cell(a['rmse_mean'], a['rmse_std']):>14}"
                  f"{cell(a['mae_mean'], a['mae_std']):>14}"
                  f"{cell(a['r2_mean'], a['r2_std']):>14}"
                  f"{cell(a['dir_acc_mean'], a['dir_acc_std']):>14}"
                  f"{cell(a['corr_mean'], a['corr_std']):>14}")

    assert results["markets"], "No market produced evaluation results"

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {RESULTS_PATH}")
