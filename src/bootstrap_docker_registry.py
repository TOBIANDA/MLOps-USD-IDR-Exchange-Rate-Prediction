"""
Bootstrap script for LK-09 Docker Compose MLflow Registry.

Executes inside the containerized environment (connecting to http://mlflow-server:5000
over mlops-bridge-network) to log and register Version 1 (Archived) and Version 2
(Production / @production, @champion) directly into the MLflow server's persistent volumes.
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict

import mlflow
import mlflow.pyfunc
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.train import build_features, get_dvc_hash, load_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("bootstrap_docker_registry")
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-server:5000")
EXPERIMENT_NAME = "USD_IDR_Exchange_Rate_Prediction"
MODEL_NAME = os.getenv("MODEL_NAME", "USD_IDR_Exchange_Rate_Regressor")
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "usd_idr_raw.csv"


def _compute_metrics(model: GradientBoostingRegressor, X_train, y_train, X_test, y_test, y_base) -> Dict[str, float]:
    y_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train)
    mae = float(mean_absolute_error(y_test, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mape = float(np.mean(np.abs((y_test - y_pred) / y_test)) * 100.0)
    r2 = float(r2_score(y_test, y_pred))

    actual_dir = np.sign(y_test - y_base)
    pred_dir = np.sign(y_pred - y_base)
    valid_mask = actual_dir != 0
    dir_acc = float(np.mean(actual_dir[valid_mask] == pred_dir[valid_mask]) * 100.0) if np.sum(valid_mask) > 0 else 50.0

    tscv = TimeSeriesSplit(n_splits=5)
    cv_rmses = []
    for tr_idx, val_idx in tscv.split(X_train):
        m_cv = GradientBoostingRegressor(**model.get_params())
        m_cv.fit(X_train.iloc[tr_idx], y_train[tr_idx])
        p_val = m_cv.predict(X_train.iloc[val_idx])
        cv_rmses.append(float(np.sqrt(mean_squared_error(y_train[val_idx], p_val))))

    return {
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "MAPE_pct": round(mape, 3),
        "R2_Score": round(r2, 4),
        "Directional_Accuracy_pct": round(dir_acc, 1),
        "Train_RMSE": round(float(np.sqrt(mean_squared_error(y_train, y_train_pred))), 2),
        "CV_5Fold_RMSE": round(float(np.mean(cv_rmses)), 2),
    }


def bootstrap_registry() -> Dict[str, Any]:
    logger.info("Connecting to containerized MLflow server at %s", TRACKING_URI)
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    client = MlflowClient(tracking_uri=TRACKING_URI)

    # Idempotency check: skip duplicate registration if Production alias already exists
    try:
        existing_prod = client.get_model_version_by_alias(MODEL_NAME, "production")
        logger.info("Model %s@production already exists (Version %s). Skipping duplicate bootstrap.", MODEL_NAME, existing_prod.version)
        return {
            "status": "ALREADY_BOOTSTRAPPED",
            "tracking_uri": TRACKING_URI,
            "model_name": MODEL_NAME,
            "production_version": int(existing_prod.version),
            "run_id": existing_prod.run_id,
        }
    except Exception:
        pass

    raw_df = load_dataset(DATA_PATH, use_db_fallback=True)
    dvc_hash = get_dvc_hash()
    featured_df, feature_cols = build_features(raw_df)

    test_size = 30
    train_df = featured_df.iloc[:-test_size].copy()
    test_df = featured_df.iloc[-test_size:].copy()

    X_train = train_df[feature_cols]
    y_train = train_df["target_jual"].values
    X_test = test_df[feature_cols]
    y_test = test_df["target_jual"].values
    y_base = test_df["jual"].values

    candidates = [
        {
            "run_name": "Run_4_GradientBoosting_Champion",
            "params": {"n_estimators": 100, "learning_rate": 0.05, "max_depth": 3, "subsample": 0.8, "random_state": 42},
            "target_stage": "Archived",
            "alias": "archived",
            "desc": "Version 1 Baseline GradientBoosting (100 trees, lr=0.05, depth=3). Superseded by tuned Version 2.",
        },
        {
            "run_name": "Run_5_GradientBoosting_Tuned_v2",
            "params": {"n_estimators": 180, "learning_rate": 0.06, "max_depth": 3, "subsample": 0.9, "random_state": 42},
            "target_stage": "Production",
            "alias": "production",
            "desc": "Version 2 Tuned 5-Fold TimeSeriesSplit CV Champion GradientBoosting (180 trees, lr=0.06, depth=3, subsample=0.9).",
        },
    ]

    registered_versions = []
    for cfg in candidates:
        model = GradientBoostingRegressor(**cfg["params"])
        model.fit(X_train, y_train)
        metrics = _compute_metrics(model, X_train, y_train, X_test, y_test, y_base)
        y_train_pred = model.predict(X_train)
        signature = infer_signature(X_train, y_train_pred)

        with mlflow.start_run(run_name=cfg["run_name"]) as run:
            mlflow.set_tag("data.dvc_hash", dvc_hash)
            mlflow.set_tag("data.version", "v2.0-dataset-500d")
            mlflow.set_tag("runtime", "docker-compose-orchestrated")
            mlflow.log_params({
                "model_type": "GradientBoostingRegressor",
                "train_samples": len(X_train),
                "test_samples": len(X_test),
                "features_count": len(feature_cols),
                **cfg["params"],
            })
            mlflow.log_metrics(metrics)
            info = mlflow.sklearn.log_model(
                sk_model=model,
                name="model",
                signature=signature,
                input_example=X_train.iloc[:2],
                serialization_format="cloudpickle",
                registered_model_name=MODEL_NAME,
            )
            ver = str(info.registered_model_version)
            client.update_model_version(name=MODEL_NAME, version=ver, description=cfg["desc"])
            client.transition_model_version_stage(
                name=MODEL_NAME,
                version=ver,
                stage=cfg["target_stage"],
                archive_existing_versions=False,
            )
            client.set_registered_model_alias(MODEL_NAME, cfg["alias"], int(ver))
            if cfg["target_stage"] == "Production":
                client.set_registered_model_alias(MODEL_NAME, "champion", int(ver))

            registered_versions.append({
                "version": int(ver),
                "run_id": run.info.run_id,
                "run_name": cfg["run_name"],
                "stage": cfg["target_stage"],
                "alias": cfg["alias"],
                "metrics": metrics,
            })
            logger.info("Registered Version %s (%s) -> Stage %s (@%s)", ver, cfg["run_name"], cfg["target_stage"], cfg["alias"])

    client.update_registered_model(
        name=MODEL_NAME,
        description="Model Registry Resmi Sistem Peramalan Kurs USD/IDR Bank Indonesia pada Arsitektur Docker Compose Terorkestrasi.",
    )

    # Verify HTTP pyfunc load directly from mlflow-server:5000
    loaded = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@production")
    sample_pred = float(loaded.predict(X_test.iloc[[-1]])[0])
    logger.info("Verified remote pyfunc load from %s -> sample pred: %.2f IDR", TRACKING_URI, sample_pred)

    return {
        "status": "BOOTSTRAPPED",
        "tracking_uri": TRACKING_URI,
        "model_name": MODEL_NAME,
        "versions": registered_versions,
        "verified_sample_prediction_idr": round(sample_pred, 2),
    }


if __name__ == "__main__":
    summary = bootstrap_registry()
    print(json.dumps(summary, indent=2))
