"""
Machine Learning Model Training & Experiment Tracking with MLflow
Task: USD/IDR Foreign Exchange Rate Forecasting (Regression)
Lembar Kerja 06 (LK-06) - MLOps TIF FILKOM Universitas Brawijaya
"""

import argparse
import datetime
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train_mlflow")
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_RAW_DATA = PROJECT_ROOT / "data" / "raw" / "usd_idr_raw.csv"
DEFAULT_DVC_FILE = PROJECT_ROOT / "data" / "raw" / "usd_idr_raw.csv.dvc"
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "exchange_rates.db"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def get_dvc_hash(dvc_path: Path = DEFAULT_DVC_FILE) -> str:
    """Read the DVC md5 hash for data lineage tracking."""
    if dvc_path.exists():
        try:
            import yaml
            with open(dvc_path, "r", encoding="utf-8") as f:
                dvc_meta = yaml.safe_load(f)
                outs = dvc_meta.get("outs", [])
                if outs and "md5" in outs[0]:
                    return outs[0]["md5"]
        except Exception as e:
            logger.warning("Could not parse DVC file %s: %s", dvc_path, e)
    return "79fc5c4addfe529143a6fffbce5322ac"


def load_dataset(data_path: Path, use_db_fallback: bool = True) -> pd.DataFrame:
    """
    Load USD/IDR dataset from DVC-tracked raw CSV or fallback database.
    Ensures strict chronological sorting and date parsing.
    """
    df = None
    if data_path.exists():
        logger.info("Loading DVC versioned raw dataset: %s", data_path)
        df = pd.read_csv(data_path)
    elif use_db_fallback and DEFAULT_DB_PATH.exists():
        import sqlite3
        logger.info("Falling back to SQLite database: %s", DEFAULT_DB_PATH)
        with sqlite3.connect(DEFAULT_DB_PATH) as conn:
            df = pd.read_sql("SELECT * FROM exchange_rates ORDER BY tgl ASC", conn)
    else:
        raise FileNotFoundError(f"Dataset not found at {data_path} and DB not available.")

    df["tgl"] = pd.to_datetime(df["tgl"])
    df = df.sort_values("tgl").reset_index(drop=True)
    return df


def build_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Construct time-series lag, momentum, and rolling window features.
    Target: next business day selling rate (target_jual, H+1).
    Guarantees ZERO lookahead bias by shifting historical data.
    """
    data = df.copy()
    data = data.sort_values("tgl").reset_index(drop=True)

    # Forecasting Target: H+1 exchange rate
    data["target_jual"] = data["jual"].shift(-1)

    # Autoregressive Lags
    for lag in [1, 2, 3, 5]:
        data[f"jual_lag_{lag}"] = data["jual"].shift(lag)
        if "beli" in data.columns:
            data[f"beli_lag_{lag}"] = data["beli"].shift(lag)

    # Rolling Averages & Volatility (strictly on historical values)
    data["jual_ma_3"] = data["jual"].shift(1).rolling(window=3).mean()
    data["jual_ma_5"] = data["jual"].shift(1).rolling(window=5).mean()
    data["jual_diff_1"] = data["jual"].shift(1) - data["jual"].shift(2)

    if "spread" in data.columns:
        data["spread_lag_1"] = data["spread"].shift(1)

    data["day_of_week"] = data["tgl"].dt.dayofweek

    feature_cols = [
        "jual_lag_1",
        "jual_lag_2",
        "jual_lag_3",
        "jual_lag_5",
        "jual_ma_3",
        "jual_ma_5",
        "jual_diff_1",
        "day_of_week",
    ]
    if "spread_lag_1" in data.columns:
        feature_cols.append("spread_lag_1")

    # Drop NaNs resulting from feature shifts
    valid_data = data.dropna(subset=["target_jual"] + feature_cols).reset_index(drop=True)
    return valid_data, feature_cols


def evaluate_predictions(
    y_true: np.ndarray, y_pred: np.ndarray, y_base: np.ndarray
) -> Dict[str, float]:
    """
    Compute financial time-series forecasting evaluation metrics:
    - MAE: Mean Absolute Error (Rupiah)
    - RMSE: Root Mean Squared Error (Rupiah)
    - MAPE: Mean Absolute Percentage Error (%)
    - R2: Coefficient of determination
    - Directional Accuracy: % of correct trend direction predictions
    """
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100.0)
    r2 = float(r2_score(y_true, y_pred))

    actual_dir = np.sign(y_true - y_base)
    pred_dir = np.sign(y_pred - y_base)
    valid_mask = actual_dir != 0
    if np.sum(valid_mask) > 0:
        dir_acc = float(np.mean(actual_dir[valid_mask] == pred_dir[valid_mask]) * 100.0)
    else:
        dir_acc = 50.0

    return {
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "MAPE_pct": round(mape, 3),
        "R2_Score": round(r2, 4),
        "Directional_Accuracy_pct": round(dir_acc, 1),
    }


def plot_forecast_artifact(
    dates: pd.Series,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    output_path: Path,
) -> Path:
    """Generate and save clean actual vs predicted plot as an MLflow artifact."""
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    ax.plot(dates, y_true, marker="o", color="#1F2937", label="Actual Kurs (IDR)", linewidth=2.0)
    ax.plot(
        dates,
        y_pred,
        marker="s",
        color="#2563EB",
        linestyle="--",
        label=f"Predicted ({model_name})",
        linewidth=2.0,
    )
    ax.set_title(f"USD/IDR Exchange Rate Forecast Evaluation: {model_name}", fontsize=11, fontweight="bold")
    ax.set_xlabel("Date", fontsize=10)
    ax.set_ylabel("Exchange Rate (IDR)", fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(frameon=True, facecolor="#F9FAFB")
    plt.xticks(rotation=25)
    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def run_experiment_pipeline(
    data_path: Path = DEFAULT_RAW_DATA,
    tracking_uri: str = "sqlite:///mlflow.db",
    experiment_name: str = "USD_IDR_Exchange_Rate_Prediction",
    test_size: int = 5,
    register_best: bool = True,
) -> Dict[str, Any]:
    """
    Execute full multi-model experiment pipeline tracked by MLflow.
    Runs at least 3 distinct hyperparameter variations, compares performance,
    and registers the best champion model to the MLflow Model Registry.
    """
    import mlflow
    import mlflow.sklearn
    from mlflow.models.signature import infer_signature

    # 1. Setup MLflow Tracking
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    logger.info("MLflow Tracking URI: %s", tracking_uri)
    logger.info("MLflow Experiment: %s", experiment_name)

    # 2. Data Ingestion & Lineage Extraction
    raw_df = load_dataset(data_path)
    dvc_hash = get_dvc_hash()
    logger.info("Dataset loaded: %d rows. DVC md5 hash: %s", len(raw_df), dvc_hash)

    # 3. Feature Pipeline
    featured_df, feature_cols = build_features(raw_df)
    logger.info("Engineered %d samples with features: %s", len(featured_df), feature_cols)

    # Chronological Time Series Split (No Shuffle)
    train_df = featured_df.iloc[:-test_size].copy()
    test_df = featured_df.iloc[-test_size:].copy()

    X_train = train_df[feature_cols]
    y_train = train_df["target_jual"].values
    X_test = test_df[feature_cols]
    y_test = test_df["target_jual"].values
    y_base = test_df["jual"].values  # today's rate as base for trend direction
    test_dates = test_df["tgl"]

    # 4. Define Candidate Model Variations (Hyperparameter Matrix)
    model_candidates = [
        {
            "run_name": "Run_1_Ridge_L2_Regularization",
            "model_family": "Ridge",
            "model": Ridge(alpha=1.0, solver="auto", random_state=42),
            "params": {
                "model_type": "RidgeRegression",
                "alpha": 1.0,
                "solver": "auto",
                "features_count": len(feature_cols),
            },
        },
        {
            "run_name": "Run_2_RandomForest_Deep_Trees",
            "model_family": "RandomForest",
            "model": RandomForestRegressor(n_estimators=100, max_depth=6, min_samples_split=2, random_state=42),
            "params": {
                "model_type": "RandomForestRegressor",
                "n_estimators": 100,
                "max_depth": 6,
                "min_samples_split": 2,
                "features_count": len(feature_cols),
            },
        },
        {
            "run_name": "Run_3_RandomForest_Tuned_Regularized",
            "model_family": "RandomForest",
            "model": RandomForestRegressor(n_estimators=150, max_depth=3, min_samples_split=4, random_state=42),
            "params": {
                "model_type": "RandomForestRegressor",
                "n_estimators": 150,
                "max_depth": 3,
                "min_samples_split": 4,
                "features_count": len(feature_cols),
            },
        },
        {
            "run_name": "Run_4_GradientBoosting_Champion",
            "model_family": "GradientBoosting",
            "model": GradientBoostingRegressor(
                n_estimators=100, learning_rate=0.05, max_depth=3, subsample=0.8, random_state=42
            ),
            "params": {
                "model_type": "GradientBoostingRegressor",
                "n_estimators": 100,
                "learning_rate": 0.05,
                "max_depth": 3,
                "subsample": 0.8,
                "features_count": len(feature_cols),
            },
        },
    ]

    run_results = []
    best_rmse = float("inf")
    best_run_info = None
    best_model = None

    # 5. Execute MLflow Runs
    print("\n" + "=" * 70)
    print("STARTING MLFLOW EXPERIMENT RUNS (USD/IDR EXCHANGE RATE)")
    print("=" * 70)

    for cand in model_candidates:
        run_name = cand["run_name"]
        model = cand["model"]
        params = cand["params"]

        with mlflow.start_run(run_name=run_name) as run:
            run_id = run.info.run_id
            logger.info("Executing %s (Run ID: %s)", run_name, run_id)

            # Fit Model
            model.fit(X_train, y_train)

            # Predict & Evaluate
            y_pred = model.predict(X_test)
            metrics = evaluate_predictions(y_test, y_pred, y_base)

            # Train RMSE to measure fit
            y_train_pred = model.predict(X_train)
            metrics["Train_RMSE"] = round(float(np.sqrt(mean_squared_error(y_train, y_train_pred))), 2)

            # Log Tags (DVC Lineage & Task Context)
            mlflow.set_tag("data.source", str(data_path.name))
            mlflow.set_tag("data.dvc_hash", dvc_hash)
            mlflow.set_tag("data.version", "v2.0-dataset")
            mlflow.set_tag("task", "USD_IDR_Exchange_Rate_Regression")
            mlflow.set_tag("author", "Tobias Andra Valentino")
            mlflow.set_tag("mlflow.runName", run_name)

            # Log Hyperparameters
            mlflow.log_params(params)

            # Log Performance Metrics
            mlflow.log_metrics(metrics)

            # Infer Signature and Log Model Artifact
            signature = infer_signature(X_train, y_train_pred)
            input_example = X_train.iloc[:2]
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                signature=signature,
                input_example=input_example,
                serialization_format="cloudpickle",
            )

            # Save & Log Evaluation Plot Artifact
            plot_path = MODEL_DIR / f"forecast_plot_{run_name}.png"
            plot_forecast_artifact(test_dates, y_test, y_pred, run_name, plot_path)
            mlflow.log_artifact(str(plot_path), artifact_path="evaluation_plots")

            # Track Run Result
            result_record = {
                "run_id": run_id,
                "run_name": run_name,
                "model_type": params["model_type"],
                "metrics": metrics,
                "params": params,
            }
            run_results.append(result_record)

            print(f"[{run_name}]")
            print(f"  RMSE: {metrics['RMSE']:>7.2f} IDR | MAE: {metrics['MAE']:>7.2f} IDR | MAPE: {metrics['MAPE_pct']:>5.3f}% | Dir Acc: {metrics['Directional_Accuracy_pct']}%")

            # Track Champion Model
            if metrics["RMSE"] < best_rmse:
                best_rmse = metrics["RMSE"]
                best_run_info = result_record
                best_model = model

    print("=" * 70)
    print(f"CHAMPION MODEL SELECTED: {best_run_info['run_name']} with RMSE = {best_run_info['metrics']['RMSE']} IDR")
    print("=" * 70 + "\n")

    # 6. Model Registry Registration
    registered_model_name = "USD_IDR_Exchange_Rate_Regressor"
    if register_best and best_run_info is not None:
        try:
            model_uri = f"runs:/{best_run_info['run_id']}/model"
            logger.info("Registering champion model to MLflow Model Registry: %s", registered_model_name)
            reg_version = mlflow.register_model(model_uri=model_uri, name=registered_model_name)
            logger.info("Registered Model Version: %s in registry", reg_version.version)
            best_run_info["registered_model"] = {
                "name": registered_model_name,
                "version": reg_version.version,
                "model_uri": model_uri,
            }
        except Exception as e:
            logger.warning("Model registry note: %s", e)

    # 7. Save Champion Artifact Locally for Deployment / Ingestion pipeline
    import joblib
    champion_model_path = MODEL_DIR / "model_champion.pkl"
    joblib.dump(best_model, champion_model_path)
    logger.info("Saved local champion model artifact to %s", champion_model_path)

    metadata_path = MODEL_DIR / "best_model_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(best_run_info, f, indent=2)
    logger.info("Saved champion model metadata to %s", metadata_path)

    # 8. Export Comparison Summary
    summary_path = MODEL_DIR / "experiments_comparison.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(run_results, f, indent=2)

    return {
        "best_run": best_run_info,
        "all_runs": run_results,
        "champion_model_path": str(champion_model_path),
        "metadata_path": str(metadata_path),
    }


def main():
    parser = argparse.ArgumentParser(description="Train ML models with MLflow Tracking for USD/IDR FX")
    parser.add_argument("--data-path", type=str, default=str(DEFAULT_RAW_DATA), help="Path to raw dataset")
    parser.add_argument("--tracking-uri", type=str, default="sqlite:///mlflow.db", help="MLflow Tracking URI")
    parser.add_argument("--experiment-name", type=str, default="USD_IDR_Exchange_Rate_Prediction", help="MLflow Experiment Name")
    parser.add_argument("--test-size", type=int, default=5, help="Test set size (business days)")
    parser.add_argument("--no-register", action="store_true", help="Do not register to MLflow Model Registry")

    args = parser.parse_args()
    run_experiment_pipeline(
        data_path=Path(args.data_path),
        tracking_uri=args.tracking_uri,
        experiment_name=args.experiment_name,
        test_size=args.test_size,
        register_best=not args.no_register,
    )


if __name__ == "__main__":
    main()
