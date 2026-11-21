"""
Unit tests for LK-08 CI/CD evaluation gate logic against LK-01 thresholds.
"""

import json
from pathlib import Path
import tempfile

from src.evaluate_model import DEFAULT_RAW_DATA_PATH, evaluate_against_thresholds


def _write_meta(path: Path, mae: float, mape: float, dir_acc: float) -> None:
    payload = {
        "run_id": "test-run-123",
        "run_name": "Test_Candidate",
        "model_type": "GradientBoostingRegressor",
        "dvc_hash": "a05285c9f8b3421595311897eecf5861",
        "data_version": "v2.0-dataset-500d",
        "metrics": {
            "MAE": mae,
            "RMSE": mae + 2.0,
            "MAPE_pct": mape,
            "R2_Score": 0.99,
            "Directional_Accuracy_pct": dir_acc,
        },
        "params": {"n_estimators": 100},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_evaluate_gate_passes_when_beating_lk01_and_naive():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        meta = tmp_path / "meta.json"
        out = tmp_path / "eval.json"
        _write_meta(meta, mae=16.73, mape=0.091, dir_acc=72.4)

        report = evaluate_against_thresholds(
            metadata_path=meta,
            data_path=DEFAULT_RAW_DATA_PATH,
            output_path=out,
            test_size=30,
        )
        assert report["gate_status"] == "PASSED"
        assert all(c["passed"] for c in report["checks"].values())
        assert out.exists()


def test_evaluate_gate_fails_on_high_mae_or_worse_than_naive():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        meta = tmp_path / "meta_fail.json"
        out = tmp_path / "eval_fail.json"
        # Exceeds both Naive Random Walk MAE (~22.45 IDR) and LK-01 75 IDR ceiling
        _write_meta(meta, mae=82.5, mape=0.45, dir_acc=65.0)

        report = evaluate_against_thresholds(
            metadata_path=meta,
            data_path=DEFAULT_RAW_DATA_PATH,
            output_path=out,
            test_size=30,
        )
        assert report["gate_status"] == "FAILED"
        assert report["checks"]["mae_below_75_idr"]["passed"] is False
        assert report["checks"]["beats_naive_random_walk_mae"]["passed"] is False
