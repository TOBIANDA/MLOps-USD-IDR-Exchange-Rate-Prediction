"""
Automated Model Evaluation & Validation Gate against LK-01 Thresholds (LK-08 Stage 3).

Validates the best trained candidate model metrics against:
  1. Mean Absolute Error (MAE < 75.0 IDR)
  2. Mean Absolute Percentage Error (MAPE < 0.5%)
  3. Directional Accuracy (> 60.0%)
  4. Hurdle Rate Baseline (Candidate MAE < Naive Random Walk MAE on same 30-day OOS window)
"""

import argparse
import datetime
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.train import build_features, get_dvc_hash, load_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("evaluate_model")

DEFAULT_METADATA_PATH = PROJECT_ROOT / "models" / "best_model_metadata.json"
DEFAULT_RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "usd_idr_raw.csv"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "models" / "evaluation_gate_report.json"

LK01_MAX_MAE_IDR = 75.0
LK01_MAX_MAPE_PCT = 0.5
LK01_MIN_DIR_ACC_PCT = 60.0


def compute_naive_baseline(
    data_path: Path = DEFAULT_RAW_DATA_PATH, test_size: int = 30
) -> Dict[str, Any]:
    """Compute Naive Random Walk baseline metrics on the exact out-of-sample test window."""
    raw_df = load_dataset(data_path, use_db_fallback=True)
    featured_df, _ = build_features(raw_df)
    test_df = featured_df.iloc[-test_size:].copy()

    y_true = test_df["target_jual"].values
    y_naive = test_df["jual"].values

    naive_mae = float(mean_absolute_error(y_true, y_naive))
    naive_rmse = float(np.sqrt(mean_squared_error(y_true, y_naive)))
    naive_mape = float(np.mean(np.abs((y_true - y_naive) / y_true)) * 100.0)

    return {
        "MAE": round(naive_mae, 2),
        "RMSE": round(naive_rmse, 2),
        "MAPE_pct": round(naive_mape, 3),
        "test_size": len(test_df),
        "start_date": str(test_df["tgl"].iloc[0].date()),
        "end_date": str(test_df["tgl"].iloc[-1].date()),
    }


def evaluate_against_thresholds(
    metadata_path: Path = DEFAULT_METADATA_PATH,
    data_path: Path = DEFAULT_RAW_DATA_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    test_size: int = 30,
) -> Dict[str, Any]:
    """Validate candidate model metrics against LK-01 thresholds and Naive Random Walk baseline."""
    metadata_path = Path(metadata_path)
    data_path = Path(data_path)
    output_path = Path(output_path)

    if not metadata_path.exists():
        raise FileNotFoundError(f"Best model metadata not found: {metadata_path}")

    with open(metadata_path, "r", encoding="utf-8") as f:
        best_meta = json.load(f)

    metrics = best_meta.get("metrics", {})
    cand_mae = float(metrics["MAE"])
    cand_rmse = float(metrics["RMSE"])
    cand_mape = float(metrics["MAPE_pct"])
    cand_dir_acc = float(metrics["Directional_Accuracy_pct"])

    baseline = compute_naive_baseline(data_path=data_path, test_size=test_size)
    naive_mae = float(baseline["MAE"])

    checks = {
        "mae_below_75_idr": {
            "actual": cand_mae,
            "threshold": f"< {LK01_MAX_MAE_IDR}",
            "passed": bool(cand_mae < LK01_MAX_MAE_IDR),
        },
        "mape_below_0_5_pct": {
            "actual": cand_mape,
            "threshold": f"< {LK01_MAX_MAPE_PCT}",
            "passed": bool(cand_mape < LK01_MAX_MAPE_PCT),
        },
        "directional_accuracy_above_60_pct": {
            "actual": cand_dir_acc,
            "threshold": f"> {LK01_MIN_DIR_ACC_PCT}",
            "passed": bool(cand_dir_acc > LK01_MIN_DIR_ACC_PCT),
        },
        "beats_naive_random_walk_mae": {
            "actual": cand_mae,
            "threshold": f"< {naive_mae}",
            "passed": bool(cand_mae < naive_mae),
        },
    }

    all_passed = all(c["passed"] for c in checks.values())
    gate_status = "PASSED" if all_passed else "FAILED"

    dvc_hash = best_meta.get("dvc_hash") or get_dvc_hash()
    data_version = best_meta.get("data_version", "v3.0-bi-real")

    report = {
        "gate_status": gate_status,
        "evaluated_at_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "test_window": {
            "test_size": baseline["test_size"],
            "start_date": baseline["start_date"],
            "end_date": baseline["end_date"],
        },
        "candidate": {
            "run_id": best_meta.get("run_id", ""),
            "run_name": best_meta.get("run_name", ""),
            "model_type": best_meta.get("model_type", ""),
            "dvc_hash": dvc_hash,
            "data_version": data_version,
            "metrics": metrics,
            "params": best_meta.get("params", {}),
        },
        "baseline_naive_random_walk": {
            "MAE": baseline["MAE"],
            "RMSE": baseline["RMSE"],
            "MAPE_pct": baseline["MAPE_pct"],
        },
        "thresholds_lk01": {
            "max_mae_idr": LK01_MAX_MAE_IDR,
            "max_mape_pct": LK01_MAX_MAPE_PCT,
            "min_directional_accuracy_pct": LK01_MIN_DIR_ACC_PCT,
            "must_beat_naive_mae": True,
        },
        "checks": checks,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("=" * 78)
    print("STAGE 3: AUTOMATED MODEL EVALUATION & LK-01 THRESHOLD VALIDATION GATE")
    print("=" * 78)
    print(f"Candidate Run : {report['candidate']['run_name']} ({report['candidate']['run_id']})")
    print(f"Model Type    : {report['candidate']['model_type']}")
    print(f"DVC Data Hash : {dvc_hash}")
    print(f"Test Window   : {baseline['start_date']} to {baseline['end_date']} ({baseline['test_size']} business days)")
    print(f"Test RMSE     : {cand_rmse:.2f} IDR (Naive RMSE: {baseline['RMSE']:.2f} IDR)")
    print("-" * 78)
    print(f"{'Criterion':<38} | {'Actual':<12} | {'Threshold':<12} | {'Status'}")
    print("-" * 78)
    for name, info in checks.items():
        status_str = "PASS" if info["passed"] else "FAIL"
        print(f"{name:<38} | {str(info['actual']):<12} | {info['threshold']:<12} | {status_str}")
    print("-" * 78)
    print(f"OVERALL VALIDATION GATE: {gate_status}")
    print(f"Report saved to: {output_path}")
    print("=" * 78)

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained champion model against LK-01 thresholds.")
    parser.add_argument("--metadata-path", type=str, default=str(DEFAULT_METADATA_PATH))
    parser.add_argument("--data-path", type=str, default=str(DEFAULT_RAW_DATA_PATH))
    parser.add_argument("--output-path", type=str, default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--test-size", type=int, default=30)
    args = parser.parse_args()

    report = evaluate_against_thresholds(
        metadata_path=Path(args.metadata_path),
        data_path=Path(args.data_path),
        output_path=Path(args.output_path),
        test_size=args.test_size,
    )
    if report["gate_status"] != "PASSED":
        logger.error("Model failed LK-01 validation gate. Blocking MLflow Registry update.")
        sys.exit(1)


if __name__ == "__main__":
    main()
