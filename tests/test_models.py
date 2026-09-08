"""
Unit tests for model training and evaluation.
"""

from pathlib import Path
import joblib
from src.models.trainer import train_and_evaluate

def test_train_and_evaluate_output():
    report, model_path = train_and_evaluate(test_days=10)
    assert Path(model_path).exists()
    assert "benchmarks" in report
    assert "candidate_model" in report
    assert "Naive_Random_Walk" in report["benchmarks"]
    assert "MAE" in report["candidate_model"]["metrics"]
    
    # Load and verify model object
    loaded_model = joblib.load(model_path)
    assert hasattr(loaded_model, "predict")
