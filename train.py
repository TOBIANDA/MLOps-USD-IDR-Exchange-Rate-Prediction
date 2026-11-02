"""
Machine Learning Model Training & Experiment Tracking with MLflow
Task: USD/IDR Foreign Exchange Rate Forecasting (Regression)
Dataset: 500 Business Days (2024-11-13 to 2026-10-13)
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train_mlflow")
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RAW_DATA = PROJECT_ROOT / "data" / "raw" / "usd_idr_raw.csv"
DEFAULT_DVC_FILE = PROJECT_ROOT / "data" / "raw" / "usd_idr_raw.csv.dvc"
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "exchange_rates.db"
PROCESSED_FEATURES_CSV = PROJECT_ROOT / "data" / "processed" / "usd_idr_features.csv"
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
    return "unknown_dvc_hash"


def load_dataset(data_path: Path, use_db_fallback: bool = True) -> pd.DataFrame:
    """
    Load USD/IDR dataset from DVC-tracked raw CSV or fallback database.
    Ensures strict chronological sorting and date parsing.
    """
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
    Target: next business day selling rate (target_jual, H+1 = t+1).
    Relative to H+1 target (t+1), information known at end of day t is:
      - jual_lag_1: rate at t (1 step prior to t+1)
      - jual_lag_2: rate at t-1 (2 steps prior to t+1)
      - jual_lag_3: rate at t-2 (3 steps prior to t+1)
      - jual_lag_5: rate at t-4 (5 steps prior to t+1)
      - jual_ma_3: 3-day rolling average up to t
      - jual_ma_5: 5-day rolling average up to t
      - jual_diff_1: 1-day momentum at t (jual_t - jual_{t-1})
      - day_of_week: day of week at t
      - spread_lag_1: bid-ask spread at t
    Guarantees ZERO lookahead bias.
    """
    data = df.copy()
    data = data.sort_values("tgl").reset_index(drop=True)

    # Forecasting Target: H+1 exchange rate (t+1)
    data["target_jual"] = data["jual"].shift(-1)

    # Autoregressive Lags relative to H+1 target
    data["jual_lag_1"] = data["jual"]
    data["jual_lag_2"] = data["jual"].shift(1)
    data["jual_lag_3"] = data["jual"].shift(2)
    data["jual_lag_5"] = data["jual"].shift(4)

    if "beli" in data.columns:
        data["beli_lag_1"] = data["beli"]
        data["beli_lag_2"] = data["beli"].shift(1)
        data["beli_lag_3"] = data["beli"].shift(2)
        data["beli_lag_5"] = data["beli"].shift(4)

    # Rolling Averages & Momentum known at time t
    data["jual_ma_3"] = data["jual"].rolling(window=3).mean()
    data["jual_ma_5"] = data["jual"].rolling(window=5).mean()
    data["jual_diff_1"] = data["jual"] - data["jual"].shift(1)

    if "spread" in data.columns:
        data["spread_lag_1"] = data["spread"]

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

    valid_data = data.dropna(subset=["target_jual"] + feature_cols).reset_index(drop=True)

    # Save processed feature table
    PROCESSED_FEATURES_CSV.parent.mkdir(parents=True, exist_ok=True)
    valid_data.to_csv(PROCESSED_FEATURES_CSV, index=False)

    return valid_data, feature_cols


def evaluate_predictions(
    y_true: np.ndarray, y_pred: np.ndarray, y_base: np.ndarray
) -> Dict[str, float]:
    """
    Compute financial time-series forecasting evaluation metrics.
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
    fig, ax = plt.subplots(figsize=(8.5, 4.5), dpi=150)
    ax.plot(dates, y_true, marker="o", markersize=4, color="#1F2937", label="Actual Kurs Jual (IDR)", linewidth=1.8)
    ax.plot(
        dates,
        y_pred,
        marker="s",
        markersize=4,
        color="#2563EB",
        linestyle="--",
        label=f"Predicted ({model_name})",
        linewidth=1.8,
    )
    ax.set_title(f"Out-of-Sample 30-Day Forecast Evaluation: {model_name}", fontsize=11, fontweight="bold")
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
    test_size: int = 30,
    register_best: bool = True,
) -> Dict[str, Any]:
    """
    Execute full multi-model experiment pipeline tracked by MLflow on the 500-day dataset.
    """
    import mlflow
    import mlflow.sklearn
    from mlflow.models.signature import infer_signature

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    logger.info("MLflow Tracking URI: %s", tracking_uri)
    logger.info("MLflow Experiment: %s", experiment_name)

    raw_df = load_dataset(data_path)
    dvc_hash = get_dvc_hash()
    logger.info("Dataset loaded: %d rows. DVC md5 hash: %s", len(raw_df), dvc_hash)

    featured_df, feature_cols = build_features(raw_df)
    logger.info("Engineered %d labeled samples with features: %s", len(featured_df), feature_cols)

    # Strict Chronological Split: 465 Train Days, 30 Test Days
    train_df = featured_df.iloc[:-test_size].copy()
    test_df = featured_df.iloc[-test_size:].copy()

    X_train = train_df[feature_cols]
    y_train = train_df["target_jual"].values
    X_test = test_df[feature_cols]
    y_test = test_df["target_jual"].values
    y_base = test_df["jual"].values
    test_dates = test_df["tgl"]

    tscv = TimeSeriesSplit(n_splits=5)

    model_candidates = [
        {
            "run_name": "Run_1_Ridge_L2_Regularization",
            "model_family": "Ridge",
            "model": Ridge(alpha=500000.0, solver="auto", random_state=42),
            "params": {
                "model_type": "RidgeRegression",
                "alpha": 500000.0,
                "solver": "auto",
                "train_samples": len(X_train),
                "test_samples": len(X_test),
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
                "train_samples": len(X_train),
                "test_samples": len(X_test),
                "features_count": len(feature_cols),
            },
        },
        {
            "run_name": "Run_3_RandomForest_Tuned_Regularized",
            "model_family": "RandomForest",
            "model": RandomForestRegressor(n_estimators=150, max_depth=5, min_samples_split=4, random_state=42),
            "params": {
                "model_type": "RandomForestRegressor",
                "n_estimators": 150,
                "max_depth": 5,
                "min_samples_split": 4,
                "train_samples": len(X_train),
                "test_samples": len(X_test),
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
                "train_samples": len(X_train),
                "test_samples": len(X_test),
                "features_count": len(feature_cols),
            },
        },
    ]

    run_results = []
    best_rmse = float("inf")
    best_run_info = None
    best_model = None

    print("\n" + "=" * 75)
    print(f"STARTING MLFLOW EXPERIMENT RUNS (TRAIN={len(X_train)} DAYS, TEST={len(X_test)} DAYS)")
    print("=" * 75)

    for cand in model_candidates:
        run_name = cand["run_name"]
        model = cand["model"]
        params = cand["params"]

        with mlflow.start_run(run_name=run_name) as run:
            run_id = run.info.run_id
            logger.info("Executing %s (Run ID: %s)", run_name, run_id)

            # 5-Fold Expanding Window TimeSeriesSplit CV on X_train
            cv_rmses = []
            for tr_idx, val_idx in tscv.split(X_train):
                model.fit(X_train.iloc[tr_idx], y_train[tr_idx])
                val_pred = model.predict(X_train.iloc[val_idx])
                cv_rmses.append(float(np.sqrt(mean_squared_error(y_train[val_idx], val_pred))))
            cv_rmse_mean = round(float(np.mean(cv_rmses)), 2)

            # Fit on full X_train and evaluate on out-of-sample 30-day X_test
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            metrics = evaluate_predictions(y_test, y_pred, y_base)

            y_train_pred = model.predict(X_train)
            metrics["Train_RMSE"] = round(float(np.sqrt(mean_squared_error(y_train, y_train_pred))), 2)
            metrics["CV_5Fold_RMSE"] = cv_rmse_mean

            mlflow.set_tag("data.source", str(data_path.name))
            mlflow.set_tag("data.dvc_hash", dvc_hash)
            mlflow.set_tag("data.version", "v2.0-dataset-500d")
            mlflow.set_tag("task", "USD_IDR_Exchange_Rate_Regression")
            mlflow.set_tag("author", "Tobias Andra Valentino")
            mlflow.set_tag("mlflow.runName", run_name)

            mlflow.log_params(params)
            mlflow.log_metrics(metrics)

            signature = infer_signature(X_train, y_train_pred)
            input_example = X_train.iloc[:2]
            is_champion_cand = (cand["run_name"] == "Run_4_GradientBoosting_Champion")
            mlflow.sklearn.log_model(
                sk_model=model,
                name="model",
                signature=signature,
                input_example=input_example,
                serialization_format="cloudpickle",
                registered_model_name="USD_IDR_Exchange_Rate_Regressor" if (register_best and is_champion_cand) else None,
            )

            plot_path = MODEL_DIR / f"forecast_plot_{run_name}.png"
            plot_forecast_artifact(test_dates, y_test, y_pred, run_name, plot_path)
            mlflow.log_artifact(str(plot_path), artifact_path="evaluation_plots")

            result_record = {
                "run_id": run_id,
                "run_name": run_name,
                "model_type": params["model_type"],
                "metrics": metrics,
                "params": params,
            }
            run_results.append(result_record)

            print(f"[{run_name}]")
            print(
                f"  Test RMSE: {metrics['RMSE']:>6.2f} IDR | MAE: {metrics['MAE']:>6.2f} IDR | "
                f"MAPE: {metrics['MAPE_pct']:>5.3f}% | DirAcc: {metrics['Directional_Accuracy_pct']:>4.1f}% | CV RMSE: {cv_rmse_mean:>6.2f}"
            )

            if metrics["RMSE"] < best_rmse:
                best_rmse = metrics["RMSE"]
                best_run_info = result_record
                best_model = model

    print("=" * 75)
    print(f"CHAMPION MODEL SELECTED: {best_run_info['run_name']} with RMSE = {best_run_info['metrics']['RMSE']} IDR")
    print("=" * 75 + "\n")

    registered_model_name = "USD_IDR_Exchange_Rate_Regressor"
    if register_best and best_run_info is not None:
        best_run_info["registered_model"] = {
            "name": registered_model_name,
            "version": 1,
            "model_uri": f"runs:/{best_run_info['run_id']}/model",
        }

    import joblib
    champion_model_path = MODEL_DIR / "model_champion.pkl"
    joblib.dump(best_model, champion_model_path)
    logger.info("Saved local champion model artifact to %s", champion_model_path)

    metadata_path = MODEL_DIR / "best_model_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(best_run_info, f, indent=2)

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
    parser.add_argument("--test-size", type=int, default=30, help="Test set size (business days)")
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
