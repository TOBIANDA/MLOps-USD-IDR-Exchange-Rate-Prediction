"""
Unit tests for feature engineering pipeline.
"""

import pandas as pd
import numpy as np
from src.features.pipeline import create_features, FEATURE_COLS, time_series_split

def test_feature_pipeline_columns_and_shapes():
    # Synthetic test data
    dates = pd.date_range(start="2026-01-01", periods=30, freq="B")
    raw_df = pd.DataFrame({
        "tgl": dates.astype(str),
        "beli": np.linspace(15500, 16000, 30),
        "jual": np.linspace(15600, 16100, 30),
        "tengah": np.linspace(15550, 16050, 30),
        "spread": [100.0] * 30,
        "mata_uang": ["USD"] * 30
    })

    featured = create_features(raw_df)
    assert not featured.empty
    for col in FEATURE_COLS:
        assert col in featured.columns, f"Missing feature column: {col}"
    assert "target_jual" in featured.columns
    assert "target_beli" in featured.columns

def test_zero_lookahead_split():
    # Ensure train and test split preserves strict chronological order
    dates = pd.date_range(start="2026-01-01", periods=40, freq="B")
    raw_df = pd.DataFrame({
        "tgl": dates.astype(str),
        "beli": np.linspace(15500, 16000, 40),
        "jual": np.linspace(15600, 16100, 40),
        "tengah": np.linspace(15550, 16050, 40),
        "spread": [100.0] * 40,
        "mata_uang": ["USD"] * 40
    })
    featured = create_features(raw_df)
    train_df, test_df = time_series_split(featured, test_size=10)
    assert len(test_df) == 10
    assert train_df["tgl"].max() < test_df["tgl"].min(), "Data leakage detected: train date exceeds test date"
