"""
FastAPI Serving Microservice for USD/IDR Exchange Rate Inference.
Supports both MLflow Model Registry loading over Docker network (LK-09)
and local Champion artifact fallback for offline/unit-test execution.
"""

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import requests
from fastapi import FastAPI, HTTPException

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.pipeline import FEATURE_COLS, create_features
from src.ingestion.fetcher import load_from_db

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("fx_api_service")

app = FastAPI(
    title="USD/IDR FX Forecasting MLOps Service",
    description="Production REST API for Bank Indonesia USD/IDR Exchange Rate Continual Forecasting.",
    version="2.0.0",
)

MODEL_DIR = PROJECT_ROOT / "models" if (PROJECT_ROOT / "models").exists() else PROJECT_ROOT / "data" / "models"
MODEL_PATH = MODEL_DIR / "model_champion.pkl"
META_PATH = MODEL_DIR / "model_metadata.json"

INFERENCE_FEATURE_COLS: List[str] = [
    "jual_lag_1",
    "jual_lag_2",
    "jual_lag_3",
    "jual_lag_5",
    "jual_ma_3",
    "jual_ma_5",
    "jual_diff_1",
    "day_of_week",
    "spread_lag_1",
]

_MODEL_CACHE: Dict[str, Any] = {
    "model": None,
    "source": None,
    "version": None,
    "stage": None,
    "run_id": None,
    "load_time_ms": None,
}


def get_registry_config() -> Dict[str, str]:
    return {
        "tracking_uri": os.getenv("MLFLOW_TRACKING_URI", "").strip(),
        "model_name": os.getenv("MODEL_NAME", "USD_IDR_Exchange_Rate_Regressor").strip(),
        "model_alias": os.getenv("MODEL_ALIAS", "production").strip(),
        "model_stage": os.getenv("MODEL_STAGE", "Production").strip(),
    }


def build_inference_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the exact 9-feature matrix expected by the MLflow Champion model
    (aligned with src/train.py and src/verify_inference.py, zero lookahead bias).
    """
    data = df.copy()
    data["tgl"] = pd.to_datetime(data["tgl"])
    data = data.sort_values("tgl").reset_index(drop=True)

    data["jual_lag_1"] = data["jual"]
    data["jual_lag_2"] = data["jual"].shift(1)
    data["jual_lag_3"] = data["jual"].shift(2)
    data["jual_lag_5"] = data["jual"].shift(4)
    data["jual_ma_3"] = data["jual"].rolling(window=3).mean()
    data["jual_ma_5"] = data["jual"].rolling(window=5).mean()
    data["jual_diff_1"] = data["jual"] - data["jual"].shift(1)
    data["day_of_week"] = data["tgl"].dt.dayofweek
    if "spread" in data.columns:
        data["spread_lag_1"] = data["spread"]
    else:
        data["spread_lag_1"] = data["jual"] - data["beli"]

    return data.dropna(subset=INFERENCE_FEATURE_COLS).reset_index(drop=True)


def check_mlflow_server_health(tracking_uri: str) -> bool:
    if not tracking_uri or not tracking_uri.startswith(("http://", "https://")):
        return False
    health_url = tracking_uri.rstrip("/") + "/health"
    try:
        resp = requests.get(health_url, timeout=3)
        return resp.status_code == 200
    except Exception:
        return False


def get_serving_model(force_reload: bool = False) -> Tuple[Any, Dict[str, Any]]:
    """
    Load Champion model from MLflow Model Registry when MLFLOW_TRACKING_URI is configured,
    with clean fallback to local models/model_champion.pkl.
    """
    if not force_reload and _MODEL_CACHE["model"] is not None:
        return _MODEL_CACHE["model"], {k: v for k, v in _MODEL_CACHE.items() if k != "model"}

    cfg = get_registry_config()
    tracking_uri = cfg["tracking_uri"]
    model_name = cfg["model_name"]
    model_alias = cfg["model_alias"]
    model_stage = cfg["model_stage"]

    if tracking_uri:
        try:
            import mlflow
            import mlflow.pyfunc
            from mlflow.tracking import MlflowClient

            os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
            mlflow.set_tracking_uri(tracking_uri)
            client = MlflowClient(tracking_uri=tracking_uri)

            loaded_model = None
            loaded_version = None
            loaded_stage = model_stage
            loaded_run_id = None

            t0 = time.perf_counter()
            try:
                alias_uri = f"models:/{model_name}@{model_alias}"
                loaded_model = mlflow.pyfunc.load_model(alias_uri)
                mv = client.get_model_version_by_alias(model_name, model_alias)
                loaded_version = str(mv.version)
                loaded_stage = mv.current_stage
                loaded_run_id = mv.run_id
            except Exception as alias_exc:
                logger.warning("Alias load (%s@%s) unavailable (%s), trying stage URI...", model_name, model_alias, alias_exc)
                stage_uri = f"models:/{model_name}/{model_stage}"
                loaded_model = mlflow.pyfunc.load_model(stage_uri)
                latest_mvs = client.get_latest_versions(model_name, stages=[model_stage])
                if latest_mvs:
                    loaded_version = str(latest_mvs[0].version)
                    loaded_stage = latest_mvs[0].current_stage
                    loaded_run_id = latest_mvs[0].run_id
            load_ms = round((time.perf_counter() - t0) * 1000.0, 2)

            _MODEL_CACHE.update({
                "model": loaded_model,
                "source": "mlflow_registry",
                "version": f"v{loaded_version}" if loaded_version else "v2.0.0-registry",
                "stage": loaded_stage,
                "run_id": loaded_run_id,
                "load_time_ms": load_ms,
            })
            logger.info(
                "Loaded model %s (version=%s, stage=%s) from %s in %.2f ms",
                model_name,
                loaded_version,
                loaded_stage,
                tracking_uri,
                load_ms,
            )
            return loaded_model, {k: v for k, v in _MODEL_CACHE.items() if k != "model"}
        except Exception as exc:
            logger.warning("MLflow Registry unavailable at %s (%s). Falling back to local pickle.", tracking_uri, exc)

    if MODEL_PATH.exists():
        t0 = time.perf_counter()
        local_model = joblib.load(MODEL_PATH)
        load_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        _MODEL_CACHE.update({
            "model": local_model,
            "source": "local_fallback" if tracking_uri else "local_artifact",
            "version": "v1.0.0-champion",
            "stage": "Local",
            "run_id": None,
            "load_time_ms": load_ms,
        })
        return local_model, {k: v for k, v in _MODEL_CACHE.items() if k != "model"}

    raise HTTPException(status_code=503, detail="Champion model not yet trained or loaded.")


@app.get("/")
def root() -> Dict[str, str]:
    return {
        "service": "USD/IDR FX Forecasting MLOps API",
        "status": "online",
        "docs_url": "/docs",
    }


@app.get("/health")
def health_check() -> Dict[str, Any]:
    db_df = load_from_db()
    cfg = get_registry_config()
    tracking_uri = cfg["tracking_uri"]
    mlflow_connected = check_mlflow_server_health(tracking_uri)

    model_loaded = MODEL_PATH.exists()
    meta: Dict[str, Any] = {
        "source": None,
        "version": None,
    }
    try:
        _, meta = get_serving_model(force_reload=False)
        model_loaded = True
    except Exception:
        pass

    return {
        "status": "healthy" if model_loaded and not db_df.empty else "degraded",
        "model_loaded": model_loaded,
        "model_source": meta.get("source"),
        "model_version": meta.get("version"),
        "mlflow_tracking_uri": tracking_uri or None,
        "mlflow_connected": mlflow_connected,
        "database_records_count": len(db_df),
        "latest_db_date": db_df["tgl"].iloc[-1] if not db_df.empty else None,
    }


@app.get("/model/info")
@app.get("/metadata")
def model_info() -> Dict[str, Any]:
    if META_PATH.exists():
        with open(META_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    best_meta = MODEL_DIR / "best_model_metadata.json"
    if best_meta.exists():
        with open(best_meta, "r", encoding="utf-8") as f:
            return json.load(f)
    raise HTTPException(status_code=404, detail="Model metadata not found.")


@app.get("/model/registry")
def model_registry_status() -> Dict[str, Any]:
    """Query live MLflow Model Registry state over the Docker bridge network."""
    cfg = get_registry_config()
    tracking_uri = cfg["tracking_uri"]
    model_name = cfg["model_name"]
    if not tracking_uri:
        raise HTTPException(status_code=400, detail="MLFLOW_TRACKING_URI is not configured.")

    try:
        from mlflow.tracking import MlflowClient

        client = MlflowClient(tracking_uri=tracking_uri)
        rm = client.get_registered_model(model_name)
        versions = client.search_model_versions(f"name='{model_name}'")
        version_list = []
        for mv in sorted(versions, key=lambda x: int(x.version)):
            run = client.get_run(mv.run_id)
            version_list.append({
                "version": int(mv.version),
                "current_stage": mv.current_stage,
                "status": mv.status,
                "run_id": mv.run_id,
                "metrics": run.data.metrics,
                "tags": mv.tags,
            })
        return {
            "status": "connected",
            "mlflow_tracking_uri": tracking_uri,
            "registered_model_name": rm.name,
            "aliases": rm.aliases,
            "versions": version_list,
        }
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Failed to query MLflow Registry: {exc}")


@app.post("/model/reload")
def reload_model() -> Dict[str, Any]:
    """Force reload of the serving model from MLflow Model Registry."""
    _MODEL_CACHE["model"] = None
    _, meta = get_serving_model(force_reload=True)
    cfg = get_registry_config()
    return {
        "status": "reloaded",
        "mlflow_tracking_uri": cfg["tracking_uri"] or None,
        "model_name": cfg["model_name"],
        "model_source": meta.get("source"),
        "model_version": meta.get("version"),
        "stage": meta.get("stage"),
        "run_id": meta.get("run_id"),
        "load_time_ms": meta.get("load_time_ms"),
    }


@app.get("/predict/h1")
@app.get("/predict")
def predict_h1() -> Dict[str, Any]:
    """
    Predict next business day (H+1) USD/IDR exchange rate.
    Uses live MLflow Model Registry model when configured, else local fallback.
    """
    model, meta = get_serving_model(force_reload=False)
    cfg = get_registry_config()

    df = load_from_db()
    if len(df) < 20:
        raise HTTPException(status_code=400, detail="Insufficient data to compute lag features.")

    model_features = getattr(model, "feature_names_in_", None)
    if model_features is not None and all(col in FEATURE_COLS for col in model_features) and len(model_features) != len(INFERENCE_FEATURE_COLS):
        # Legacy 12-feature local model compatibility (e.g., Ridge trainer fallback)
        featured = create_features(df)
        latest_row = featured.iloc[[-1]]
        latest_features = latest_row[list(model_features)]
    else:
        featured = build_inference_features(df)
        latest_row = featured.iloc[[-1]]
        if model_features is not None and all(col in latest_row.columns for col in model_features):
            latest_features = latest_row[list(model_features)]
        else:
            latest_features = latest_row[INFERENCE_FEATURE_COLS]

    pred_jual = float(model.predict(latest_features)[0])

    recent_spread = float(df["spread"].iloc[-7:].mean())
    pred_beli = round(pred_jual - recent_spread, 2)
    pred_jual = round(pred_jual, 2)

    # Use v2 champion RMSE (19.46 IDR) when served from registry or v2 model, else legacy estimate
    is_registry_v2 = meta.get("source") == "mlflow_registry"
    half_width = round(1.96 * 19.46, 2) if is_registry_v2 else round(1.96 * (63.37 * 0.8), 2)

    return {
        "base_date": str(latest_row["tgl"].iloc[0].date()),
        "horizon": "H+1 (Next Business Day)",
        "currency": "USD/IDR",
        "prediction": {
            "kurs_jual": pred_jual,
            "kurs_beli": pred_beli,
            "kurs_tengah": round((pred_jual + pred_beli) / 2.0, 2),
            "estimated_spread": round(recent_spread, 2),
        },
        "confidence_interval_95": {
            "kurs_jual_lower": round(pred_jual - half_width, 2),
            "kurs_jual_upper": round(pred_jual + half_width, 2),
        },
        "model_version": meta.get("version") or "v1.0.0-champion",
        "model_source": meta.get("source"),
        "mlflow_tracking_uri": cfg["tracking_uri"] or None,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
