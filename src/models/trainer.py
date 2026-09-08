"""
Model training and baseline evaluation module.
Compares candidate ML models against the Naive Random Walk benchmark.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.pipeline import FEATURE_COLS, create_features, time_series_split
from src.ingestion.fetcher import load_from_db

logger = logging.getLogger(__name__)
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_base: np.ndarray) -> Dict[str, float]:
    """
    Compute regression and financial forecasting metrics:
    - MAE: Mean Absolute Error in Rupiah
    - RMSE: Root Mean Squared Error
    - MAPE: Mean Absolute Percentage Error (%)
    - Directional Accuracy: % of days where predicted trend direction (up/down) matches actual
    """
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100.0)

    # Directional accuracy: sign of (pred - today) vs sign of (actual - today)
    actual_dir = np.sign(y_true - y_base)
    pred_dir = np.sign(y_pred - y_base)
    # Ignore zero movements
    valid_idx = actual_dir != 0
    if np.sum(valid_idx) > 0:
        dir_acc = float(np.mean(actual_dir[valid_idx] == pred_dir[valid_idx]) * 100.0)
    else:
        dir_acc = 50.0

    return {
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "MAPE_pct": round(mape, 3),
        "Directional_Accuracy_pct": round(dir_acc, 1),
    }


def train_and_evaluate(test_days: int = 14) -> Tuple[Dict[str, Any], Path]:
    """
    Train candidate model, compare against Naive Random Walk, and save artifact.
    """
    df = load_from_db()
    if len(df) < 30:
        raise ValueError(f"Insufficient historical data: {len(df)} rows found in DB.")

    featured = create_features(df)
    train_df, test_df = time_series_split(featured, test_size=test_days)

    X_train = train_df[FEATURE_COLS]
    y_train = train_df["target_jual"]
    X_test = test_df[FEATURE_COLS]
    y_test = test_df["target_jual"].values
    y_today = test_df["jual"].values  # today's rate as base for random walk

    # 1. Benchmark: Naive Random Walk (y_pred = y_today)
    y_pred_naive = y_today
    metrics_naive = calculate_metrics(y_test, y_pred_naive, y_today)

    # 2. Benchmark: Moving Average 7 Hari
    y_pred_ma = test_df["jual_ma_7"].values
    metrics_ma = calculate_metrics(y_test, y_pred_ma, y_today)

    # 3. Candidate ML Model: Ridge Regression with L2 Regularization
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    y_pred_ml = model.predict(X_test)
    metrics_ml = calculate_metrics(y_test, y_pred_ml, y_today)

    # Comparison summary
    report = {
        "evaluation_period_days": test_days,
        "test_start_date": str(test_df["tgl"].iloc[0].date()),
        "test_end_date": str(test_df["tgl"].iloc[-1].date()),
        "benchmarks": {
            "Naive_Random_Walk": metrics_naive,
            "Moving_Average_7D": metrics_ma,
        },
        "candidate_model": {
            "model_type": "RidgeRegression",
            "metrics": metrics_ml,
            "beats_random_walk_mae": metrics_ml["MAE"] < metrics_naive["MAE"],
        },
    }

    # Save model artifact and metadata
    artifact_path = MODEL_DIR / "model_champion.pkl"
    meta_path = MODEL_DIR / "model_metadata.json"

    joblib.dump(model, artifact_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info("Model saved to %s", artifact_path)
    return report, artifact_path


if __name__ == "__main__":
    report, path = train_and_evaluate(test_days=14)
    print("=" * 60)
    print("MLOPS EXPERIMENT EVALUATION REPORT (Kurs Jual USD/IDR)")
    print("=" * 60)
    print(f"Evaluation Window : {report['test_start_date']} to {report['test_end_date']} ({report['evaluation_period_days']} business days)")
    print("-" * 60)
    print(f"{'Model / Baseline':<24} | {'MAE (Rp)':<9} | {'RMSE':<7} | {'MAPE':<8} | {'Dir Acc':<7}")
    print("-" * 60)
    
    n = report["benchmarks"]["Naive_Random_Walk"]
    print(f"{'1. Naive Random Walk':<24} | Rp {n['MAE']:<6.2f} | {n['RMSE']:<7.2f} | {n['MAPE_pct']:<5.3f}% | {n['Directional_Accuracy_pct']:<5.1f}%")
    
    m = report["benchmarks"]["Moving_Average_7D"]
    print(f"{'2. Moving Average 7D':<24} | Rp {m['MAE']:<6.2f} | {m['RMSE']:<7.2f} | {m['MAPE_pct']:<5.3f}% | {m['Directional_Accuracy_pct']:<5.1f}%")
    
    c = report["candidate_model"]["metrics"]
    print(f"{'3. ML Ridge Model':<24} | Rp {c['MAE']:<6.2f} | {c['RMSE']:<7.2f} | {c['MAPE_pct']:<5.3f}% | {c['Directional_Accuracy_pct']:<5.1f}%")
    print("=" * 60)
    print(f"Beats Random Walk Hurdle: {report['candidate_model']['beats_random_walk_mae']}")
