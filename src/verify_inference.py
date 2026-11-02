"""
Programmatic Model Verification Script for LK-07.
Validates loading production models from MLflow Model Registry via:
1. Stage-based URI: models:/USD_IDR_Exchange_Rate_Regressor/Production
2. Alias-based URI: models:/USD_IDR_Exchange_Rate_Regressor@production
Executes inference on test vectors and benchmarks real-time latency.
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict

import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("verify_inference")
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"

TRACKING_URI = "sqlite:///mlflow.db"
MODEL_NAME = "USD_IDR_Exchange_Rate_Regressor"
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_production_inference() -> Dict[str, Any]:
    mlflow.set_tracking_uri(TRACKING_URI)

    # 1. Load Data to form Realistic Test Feature Vector
    data_path = PROJECT_ROOT / "data" / "raw" / "usd_idr_raw.csv"
    df = pd.read_csv(data_path)
    df["tgl"] = pd.to_datetime(df["tgl"])
    df = df.sort_values("tgl").reset_index(drop=True)

    # Construct features exactly as in training
    for lag in [1, 2, 3, 5]:
        df[f"jual_lag_{lag}"] = df["jual"].shift(lag)
        if "beli" in df.columns:
            df[f"beli_lag_{lag}"] = df["beli"].shift(lag)
    df["jual_ma_3"] = df["jual"].shift(1).rolling(window=3).mean()
    df["jual_ma_5"] = df["jual"].shift(1).rolling(window=5).mean()
    df["jual_diff_1"] = df["jual"].shift(1) - df["jual"].shift(2)
    if "spread" in df.columns:
        df["spread_lag_1"] = df["spread"].shift(1)
    df["day_of_week"] = df["tgl"].dt.dayofweek

    feature_cols = [
        "jual_lag_1", "jual_lag_2", "jual_lag_3", "jual_lag_5",
        "jual_ma_3", "jual_ma_5", "jual_diff_1", "day_of_week",
    ]
    if "spread_lag_1" in df.columns:
        feature_cols.append("spread_lag_1")

    valid_df = df.dropna(subset=feature_cols).reset_index(drop=True)
    sample_features = valid_df[feature_cols].iloc[[-1]]
    latest_date = valid_df["tgl"].iloc[-1].strftime("%Y-%m-%d")
    latest_kurs = float(valid_df["jual"].iloc[-1])

    print("=" * 70)
    print("VERIFIKASI INFERENSI MODEL PRODUCTION - MLFLOW MODEL REGISTRY")
    print("=" * 70)
    print(f"Model Name           : {MODEL_NAME}")
    print(f"Latest Historical Tgl: {latest_date} (Kurs Terakhir: Rp{latest_kurs:,.2f})")
    print("-" * 70)

    # 2. Test Loading via Stage-based URI
    stage_uri = f"models:/{MODEL_NAME}/Production"
    logger.info("Loading model via Stage URI: %s", stage_uri)
    t0 = time.perf_counter()
    model_by_stage = mlflow.pyfunc.load_model(stage_uri)
    stage_load_time = (time.perf_counter() - t0) * 1000

    # 3. Test Loading via Alias-based URI
    alias_uri = f"models:/{MODEL_NAME}@production"
    logger.info("Loading model via Alias URI: %s", alias_uri)
    t0 = time.perf_counter()
    model_by_alias = mlflow.pyfunc.load_model(alias_uri)
    alias_load_time = (time.perf_counter() - t0) * 1000

    # 4. Predict using both to confirm consistency
    pred_stage = model_by_stage.predict(sample_features)
    pred_alias = model_by_alias.predict(sample_features)

    val_stage = float(pred_stage[0]) if hasattr(pred_stage, "__iter__") else float(pred_stage)
    val_alias = float(pred_alias[0]) if hasattr(pred_alias, "__iter__") else float(pred_alias)

    assert abs(val_stage - val_alias) < 1e-4, "Predictions between stage and alias must match!"

    # 5. Latency Benchmarking (50 runs)
    latencies = []
    for _ in range(50):
        t_start = time.perf_counter()
        _ = model_by_stage.predict(sample_features)
        latencies.append((time.perf_counter() - t_start) * 1000)

    p50_latency = float(np.percentile(latencies, 50))
    p95_latency = float(np.percentile(latencies, 95))
    avg_latency = float(np.mean(latencies))

    # Calculate financial spread estimate and confidence interval
    recent_spread = float(valid_df["spread"].iloc[-7:].mean()) if "spread" in valid_df.columns else 100.0
    pred_jual = round(val_stage, 2)
    pred_beli = round(pred_jual - recent_spread, 2)
    pred_tengah = round((pred_jual + pred_beli) / 2.0, 2)

    # 95% Confidence Interval based on test MAE (13.80 IDR) -> ~ 1.96 * 13.80 * 0.8
    ci_half = round(1.96 * (13.80 * 0.8), 2)
    ci_lower = round(pred_jual - ci_half, 2)
    ci_upper = round(pred_jual + ci_half, 2)

    results = {
        "status": "SUCCESS_VERIFIED",
        "model_name": MODEL_NAME,
        "stage_uri": stage_uri,
        "alias_uri": alias_uri,
        "stage_load_time_ms": round(stage_load_time, 2),
        "alias_load_time_ms": round(alias_load_time, 2),
        "inference_latency": {
            "avg_ms": round(avg_latency, 3),
            "p50_ms": round(p50_latency, 3),
            "p95_ms": round(p95_latency, 3),
        },
        "forecast_h1": {
            "base_date": latest_date,
            "kurs_jual_prediksi": pred_jual,
            "kurs_beli_prediksi": pred_beli,
            "kurs_tengah_prediksi": pred_tengah,
            "estimated_spread": round(recent_spread, 2),
            "confidence_interval_95": {
                "lower": ci_lower,
                "upper": ci_upper,
            },
        },
    }

    print(f"Stage URI Load Time  : {stage_load_time:.2f} ms")
    print(f"Alias URI Load Time  : {alias_load_time:.2f} ms")
    print(f"Average Infer Latency: {avg_latency:.3f} ms (p95: {p95_latency:.3f} ms)")
    print(f"Hasil Prediksi Kurs  :")
    print(f"  - Kurs Jual (H+1)  : Rp{pred_jual:,.2f}")
    print(f"  - Kurs Beli (H+1)  : Rp{pred_beli:,.2f}")
    print(f"  - Kurs Tengah      : Rp{pred_tengah:,.2f}")
    print(f"  - 95% CI Interval  : [Rp{ci_lower:,.2f} - Rp{ci_upper:,.2f}]")
    print("=" * 70)
    print("STATUS VERIFIKASI: MODEL PRODUCTION SIAP DIGUNAKAN UNTUK SERVING")
    print("=" * 70)

    # Save verification report JSON
    with open(PROJECT_ROOT / "models" / "inference_verification_report.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    test_production_inference()
