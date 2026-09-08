"""
FastAPI Serving Microservice for USD/IDR Exchange Rate Inference.
Production-ready REST API with health check, metadata, and H+1 forecasting.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.pipeline import FEATURE_COLS, create_features
from src.ingestion.fetcher import load_from_db

app = FastAPI(
    title="USD/IDR FX Forecasting MLOps Service",
    description="Production REST API for Bank Indonesia USD/IDR Exchange Rate Continual Forecasting.",
    version="1.0.0",
)

MODEL_DIR = PROJECT_ROOT / "models" if (PROJECT_ROOT / "models").exists() else PROJECT_ROOT / "data" / "models"
MODEL_PATH = MODEL_DIR / "model_champion.pkl"
META_PATH = MODEL_DIR / "model_metadata.json"


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
    model_loaded = MODEL_PATH.exists()
    return {
        "status": "healthy" if model_loaded and not db_df.empty else "degraded",
        "model_loaded": model_loaded,
        "database_records_count": len(db_df),
        "latest_db_date": db_df["tgl"].iloc[-1] if not db_df.empty else None,
    }


@app.get("/model/info")
@app.get("/metadata")
def model_info() -> Dict[str, Any]:
    if not META_PATH.exists():
        raise HTTPException(status_code=404, detail="Model metadata not found.")
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return meta


@app.get("/predict/h1")
@app.get("/predict")
def predict_h1() -> Dict[str, Any]:
    """
    Predict next business day (H+1) USD/IDR exchange rate.
    Uses the latest available feature vector from SQLite.
    """
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=503, detail="Champion model not yet trained or loaded.")

    df = load_from_db()
    if len(df) < 20:
        raise HTTPException(status_code=400, detail="Insufficient data to compute lag features.")

    featured = create_features(df)
    latest_row = featured.iloc[[-1]]
    latest_features = latest_row[FEATURE_COLS]

    model = joblib.load(MODEL_PATH)
    pred_jual = float(model.predict(latest_features)[0])

    # Compute spread based on recent 7-day average spread
    recent_spread = float(df["spread"].iloc[-7:].mean())
    pred_beli = round(pred_jual - recent_spread, 2)
    pred_jual = round(pred_jual, 2)

    # Historical MAE used for 95% uncertainty interval (+- 1.96 * MAE approx)
    mae_estimate = 63.37
    half_width = round(1.96 * (mae_estimate * 0.8), 2)

    return {
        "base_date": str(latest_row["tgl"].iloc[0].date()),
        "horizon": "H+1 (Next Business Day)",
        "currency": "USD/IDR",
        "prediction": {
            "kurs_jual": pred_jual,
            "kurs_beli": pred_beli,
            "kurs_tengah": round((pred_jual + pred_beli) / 2.0, 2),
            "estimated_spread": recent_spread,
        },
        "confidence_interval_95": {
            "kurs_jual_lower": round(pred_jual - half_width, 2),
            "kurs_jual_upper": round(pred_jual + half_width, 2),
        },
        "model_version": "v1.0.0-champion",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
