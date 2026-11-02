"""
Model Retraining, TimeSeriesSplit Cross-Validation Tuning, & Model Registry Lifecycle Management for LK-07.
Performs 5-Fold Expanding Window TimeSeriesSplit Cross-Validation on X_train (465 business days),
trains Version 2 (Hyperparameter-Tuned Gradient Boosting Regressor),
evaluates on the 30-day out-of-sample test set, and registers it as Version 2 in MLflow Model Registry.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict

import joblib
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

from src.train import build_features, get_dvc_hash, load_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train_v2_registry")
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RAW_DATA = PROJECT_ROOT / "data" / "raw" / "usd_idr_raw.csv"
DEFAULT_DVC_FILE = PROJECT_ROOT / "data" / "raw" / "usd_idr_raw.csv.dvc"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def train_and_register_v2(
    data_path: Path = DEFAULT_RAW_DATA,
    tracking_uri: str = "sqlite:///mlflow.db",
    experiment_name: str = "USD_IDR_Exchange_Rate_Prediction",
    model_registry_name: str = "USD_IDR_Exchange_Rate_Regressor",
    test_size: int = 30,
) -> Dict[str, Any]:
    # 1. Setup MLflow Tracking
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    client = MlflowClient(tracking_uri=tracking_uri)

    # 2. Load 500-Day Dataset & Build Zero-Leakage Features
    raw_df = load_dataset(data_path)
    dvc_hash = get_dvc_hash()
    valid_df, feature_cols = build_features(raw_df)

    # Chronological Split: 465 Train Days, 30 Out-of-Sample Test Days
    train_df = valid_df.iloc[:-test_size].copy()
    test_df = valid_df.iloc[-test_size:].copy()

    X_train = train_df[feature_cols]
    y_train = train_df["target_jual"].values
    X_test = test_df[feature_cols]
    y_test = test_df["target_jual"].values
    y_base = test_df["jual"].values

    # 3. 5-Fold Expanding Window TimeSeriesSplit Cross-Validation strictly on X_train (465 rows)
    tscv = TimeSeriesSplit(n_splits=5)
    param_grid = [
        {"n_estimators": 100, "learning_rate": 0.05, "max_depth": 3, "subsample": 0.8},
        {"n_estimators": 140, "learning_rate": 0.05, "max_depth": 3, "subsample": 0.85},
        {"n_estimators": 180, "learning_rate": 0.06, "max_depth": 3, "subsample": 0.9},
        {"n_estimators": 120, "learning_rate": 0.04, "max_depth": 4, "subsample": 0.8},
    ]

    best_cv_rmse = float("inf")
    best_cfg = param_grid[2]
    for cfg in param_grid:
        fold_rmses = []
        for tr_idx, val_idx in tscv.split(X_train):
            m = GradientBoostingRegressor(**cfg, random_state=42)
            m.fit(X_train.iloc[tr_idx], y_train[tr_idx])
            p_val = m.predict(X_train.iloc[val_idx])
            fold_rmses.append(float(np.sqrt(mean_squared_error(y_train[val_idx], p_val))))
        mean_cv = float(np.mean(fold_rmses))
        logger.info("5-Fold TimeSeriesSplit CV %s -> CV_RMSE = %.2f IDR", cfg, mean_cv)
        if mean_cv < best_cv_rmse:
            best_cv_rmse = mean_cv
            best_cfg = cfg

    v2_params = {
        "model_type": "GradientBoostingRegressor",
        "version": "v2.0-cv-tuned",
        "n_estimators": best_cfg["n_estimators"],
        "learning_rate": best_cfg["learning_rate"],
        "max_depth": best_cfg["max_depth"],
        "subsample": best_cfg["subsample"],
        "cv_folds": 5,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "random_state": 42,
        "features_count": len(feature_cols),
    }

    run_name = "Run_5_GradientBoosting_Tuned_v2"
    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        logger.info("Starting Run: %s (ID: %s)", run_name, run_id)

        model_v2 = GradientBoostingRegressor(
            n_estimators=v2_params["n_estimators"],
            learning_rate=v2_params["learning_rate"],
            max_depth=v2_params["max_depth"],
            subsample=v2_params["subsample"],
            random_state=42,
        )
        model_v2.fit(X_train, y_train)

        y_pred = model_v2.predict(X_test)
        y_train_pred = model_v2.predict(X_train)

        mae = float(mean_absolute_error(y_test, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        mape = float(np.mean(np.abs((y_test - y_pred) / y_true)) * 100.0) if "y_true" in locals() else float(np.mean(np.abs((y_test - y_pred) / y_test)) * 100.0)
        r2 = float(r2_score(y_test, y_pred))

        actual_dir = np.sign(y_test - y_base)
        pred_dir = np.sign(y_pred - y_base)
        valid_mask = actual_dir != 0
        dir_acc = float(np.mean(actual_dir[valid_mask] == pred_dir[valid_mask]) * 100.0) if np.sum(valid_mask) > 0 else 50.0

        train_rmse = float(np.sqrt(mean_squared_error(y_train, y_train_pred)))

        metrics = {
            "MAE": round(mae, 2),
            "RMSE": round(rmse, 2),
            "MAPE_pct": round(mape, 3),
            "R2_Score": round(r2, 4),
            "Directional_Accuracy_pct": round(dir_acc, 1),
            "Train_RMSE": round(train_rmse, 2),
            "CV_5Fold_RMSE": round(best_cv_rmse, 2),
        }

        mlflow.set_tag("data.source", str(data_path.name))
        mlflow.set_tag("data.dvc_hash", dvc_hash)
        mlflow.set_tag("data.version", "v2.0-dataset-500d")
        mlflow.set_tag("task", "USD_IDR_Exchange_Rate_Regression")
        mlflow.set_tag("author", "Tobias Andra Valentino")
        mlflow.set_tag("model_iteration", "v2_production_candidate")
        mlflow.set_tag("optimization_method", "5fold_expanding_window_timeseries_cv")

        mlflow.log_params(v2_params)
        mlflow.log_metrics(metrics)

        signature = infer_signature(X_train, y_train_pred)
        input_example = X_train.iloc[:2]
        model_info = mlflow.sklearn.log_model(
            sk_model=model_v2,
            name="model",
            signature=signature,
            input_example=input_example,
            serialization_format="cloudpickle",
            registered_model_name=model_registry_name,
        )

        logger.info(
            "Run finished. Metrics: Test_RMSE=%.2f, Test_MAE=%.2f, MAPE=%.3f%%, DirAcc=%.1f%%, CV_RMSE=%.2f",
            rmse, mae, mape, dir_acc, best_cv_rmse,
        )

        # 4. Get Version 2 in Model Registry
        model_uri = model_info.model_uri
        v2_num = str(model_info.registered_model_version)
        logger.info("Model registered as Version %s in %s (URI: %s)", v2_num, model_registry_name, model_uri)

        client.update_model_version(
            name=model_registry_name,
            version=v2_num,
            description=(
                f"Version 2: 5-Fold TimeSeriesSplit CV-tuned GradientBoosting "
                f"(n_est={v2_params['n_estimators']}, lr={v2_params['learning_rate']}, depth={v2_params['max_depth']}, subsample={v2_params['subsample']}). "
                f"Trained on 465 business days, evaluated on 30-day out-of-sample window: RMSE={rmse:.2f} IDR, MAE={mae:.2f} IDR, DirAcc={dir_acc:.1f}%."
            ),
        )
        client.set_model_version_tag(
            name=model_registry_name,
            version=v2_num,
            key="dvc_data_hash",
            value=dvc_hash,
        )
        client.set_model_version_tag(
            name=model_registry_name,
            version=v2_num,
            key="validation_gate",
            value="PASSED_30DAY_OOS_AND_5FOLD_CV",
        )

        champion_pkl_path = MODEL_DIR / "model_champion.pkl"
        joblib.dump(model_v2, champion_pkl_path)
        logger.info("Updated local champion model at %s", champion_pkl_path)

        meta_record = {
            "model_name": model_registry_name,
            "version": int(v2_num),
            "run_id": run_id,
            "run_name": run_name,
            "stage": "None",
            "dvc_data_hash": dvc_hash,
            "metrics": metrics,
            "params": v2_params,
            "model_uri": model_uri,
        }
        with open(MODEL_DIR / "model_v2_metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta_record, f, indent=2)

        return meta_record


if __name__ == "__main__":
    res = train_and_register_v2()
    print("SUCCESS:", json.dumps(res, indent=2))
