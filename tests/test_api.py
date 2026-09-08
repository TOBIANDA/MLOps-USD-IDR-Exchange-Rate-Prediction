"""
Unit tests for FastAPI serving layer.
"""

from fastapi.testclient import TestClient
from src.api.app import app

client = TestClient(app)

def test_api_root():
    res = client.get("/")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "online"

def test_api_health():
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] in ("healthy", "degraded")
    assert "database_records_count" in data

def test_api_metadata():
    res = client.get("/metadata")
    assert res.status_code == 200
    data = res.json()
    assert "candidate_model" in data or "evaluation_period_days" in data

def test_api_predict():
    res = client.get("/predict")
    assert res.status_code == 200
    data = res.json()
    assert "prediction" in data
    assert data["prediction"]["kurs_jual"] > 10000.0, "Kurs USD should be realistic value (> 10000 IDR)"
