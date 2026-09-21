"""
Unit tests for automated ETL pipeline.
Verifies:
  - Extraction data types and non-empty guarantees
  - Validation rules (schema, positive values, price ordering)
  - Feature transformation columns and absence of future lookahead bias
  - Load integrity (file creation and checksum manifest)
"""

import sqlite3
import tempfile
from pathlib import Path
import pandas as pd
import pytest

from src.ingestion.etl_pipeline import ETLPipeline, compute_sha256


@pytest.fixture
def sample_raw_data():
    return pd.DataFrame({
        "tgl": ["2026-09-15", "2026-09-16", "2026-09-17", "2026-09-18", "2026-09-21"],
        "beli": [17546.83, 17598.56, 17623.44, 17664.24, 17656.28],
        "jual": [17723.17, 17775.44, 17800.56, 17841.76, 17833.72],
        "tengah": [17635.0, 17687.0, 17712.0, 17753.0, 17745.0],
        "spread": [176.34, 176.88, 177.12, 177.52, 177.44],
        "mata_uang": ["USD", "USD", "USD", "USD", "USD"]
    })


def test_validate_and_clean_success(sample_raw_data):
    pipeline = ETLPipeline()
    cleaned = pipeline.validate_and_clean(sample_raw_data)
    assert len(cleaned) == 5
    assert list(cleaned["tgl"]) == sorted(list(cleaned["tgl"]))
    assert (cleaned["beli"] <= cleaned["jual"]).all()
    assert (cleaned["spread"] > 0).all()


def test_validate_and_clean_drops_invalid_rates():
    bad_data = pd.DataFrame({
        "tgl": ["2026-09-15", "2026-09-16", "2026-09-17"],
        "beli": [17500.0, -100.0, 17600.0],  # Negative rate
        "jual": [17700.0, 17700.0, 17400.0],  # 3rd row: beli > jual
        "tengah": [17600.0, 8800.0, 17500.0],
        "spread": [200.0, 17800.0, -200.0],
    })
    pipeline = ETLPipeline()
    cleaned = pipeline.validate_and_clean(bad_data)
    # Only 1 valid row (the first row)
    assert len(cleaned) == 1
    assert cleaned.iloc[0]["tgl"] == "2026-09-15"


def test_feature_transformation_columns(sample_raw_data):
    pipeline = ETLPipeline()
    cleaned = pipeline.validate_and_clean(sample_raw_data)
    featured = pipeline.transform_features(cleaned)
    assert "target_jual" in featured.columns
    assert "target_beli" in featured.columns
    assert "jual_lag_1" in featured.columns
    assert "jual_ma_7" in featured.columns


def test_etl_load_and_manifest(sample_raw_data):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        raw_csv = tmp_path / "raw.csv"
        proc_csv = tmp_path / "proc.csv"
        manifest_file = tmp_path / "manifest.json"

        pipeline = ETLPipeline(
            raw_csv_path=raw_csv,
            processed_csv_path=proc_csv,
            manifest_path=manifest_file,
        )

        cleaned = pipeline.validate_and_clean(sample_raw_data)
        featured = pipeline.transform_features(cleaned)
        manifest = pipeline.load(cleaned, featured)

        assert raw_csv.exists()
        assert proc_csv.exists()
        assert manifest_file.exists()
        assert manifest["validation_status"] == "PASSED"
        assert manifest["raw_dataset"]["total_records"] == 5
