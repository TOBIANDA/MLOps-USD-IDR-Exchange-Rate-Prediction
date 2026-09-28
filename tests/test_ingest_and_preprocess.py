"""
Unit tests for data ingestion (ingest_data.py) and preprocessing (preprocess.py).
Verifies:
  - IngestionWorker functionality and non-destructive timestamped outputs
  - Idempotent SQLite insertion
  - Preprocessor validation, deduplication, missing values handling, and financial sanity
"""

import sqlite3
import tempfile
from pathlib import Path
import pandas as pd
import pytest

from src.ingest_data import IngestionWorker
from src.preprocess import Preprocessor


@pytest.fixture
def sample_raw_exchange_data():
    return pd.DataFrame({
        "tgl": ["2026-09-22", "2026-09-23", "2026-09-24", "2026-09-25", "2026-09-28"],
        "beli": [17723.94, 17793.58, 17713.99, 17808.51, 17827.42],
        "jual": [17902.06, 17972.42, 17892.01, 17987.49, 18006.58],
        "tengah": [17813.0, 17883.0, 17803.0, 17898.0, 17917.0],
        "spread": [178.12, 178.84, 178.02, 178.98, 179.16],
        "mata_uang": ["USD", "USD", "USD", "USD", "USD"]
    })


def test_ingestion_save_raw_dataset_non_destructive(sample_raw_exchange_data):
    with tempfile.TemporaryDirectory() as tmpdir:
        raw_dir = Path(tmpdir) / "raw"
        db_path = Path(tmpdir) / "exchange.db"

        worker = IngestionWorker(raw_dir=raw_dir, db_path=db_path)
        saved_files = worker.save_raw_dataset(
            df=sample_raw_exchange_data,
            base_name="usd_idr_raw",
            export_formats=["csv", "json"],
            use_timestamp=True,
        )

        assert "csv_timestamped" in saved_files
        assert "csv_latest" in saved_files
        assert "csv_canonical" in saved_files
        assert "json_timestamped" in saved_files
        assert saved_files["csv_timestamped"].exists()
        assert saved_files["csv_latest"].exists()
        assert saved_files["json_timestamped"].exists()

        # Read back saved file
        loaded = pd.read_csv(saved_files["csv_timestamped"])
        assert len(loaded) == 5
        assert list(loaded["mata_uang"]) == ["USD"] * 5


def test_ingestion_sqlite_idempotency(sample_raw_exchange_data):
    with tempfile.TemporaryDirectory() as tmpdir:
        raw_dir = Path(tmpdir) / "raw"
        db_path = Path(tmpdir) / "exchange.db"

        worker = IngestionWorker(raw_dir=raw_dir, db_path=db_path)
        # First insertion
        inserted1 = worker.sync_to_sqlite(sample_raw_exchange_data)
        assert inserted1 == 5

        # Second insertion with same data -> should be 0 new rows (idempotent)
        inserted2 = worker.sync_to_sqlite(sample_raw_exchange_data)
        assert inserted2 == 0

        # Query database to confirm count
        cached_df = worker.load_from_db_cache(days=10)
        assert len(cached_df) == 5


def test_preprocessor_clean_and_validate_success(sample_raw_exchange_data):
    with tempfile.TemporaryDirectory() as tmpdir:
        raw_dir = Path(tmpdir) / "raw"
        processed_dir = Path(tmpdir) / "processed"

        preprocessor = Preprocessor(raw_dir=raw_dir, processed_dir=processed_dir)
        clean_df, stats = preprocessor.clean_and_validate(sample_raw_exchange_data)

        assert len(clean_df) == 5
        assert stats["validation_status"] == "PASSED"
        assert stats["dropped_null_rows"] == 0
        assert stats["dropped_duplicate_rows"] == 0
        assert stats["dropped_invalid_financial_rows"] == 0
        assert (clean_df["beli"] <= clean_df["jual"]).all()
        assert (clean_df["spread"] > 0).all()


def test_preprocessor_filters_invalid_and_duplicate_rows():
    dirty_data = pd.DataFrame({
        "tgl": ["2026-09-22", "2026-09-22", "2026-09-23", "2026-09-24", "2026-09-25"],
        "beli": [17700.0, 17700.0, -500.0, 18000.0, 17800.0],  # Negative rate & inverted rate
        "jual": [17900.0, 17900.0, 17900.0, 17500.0, 17950.0],  # row 3: beli > jual
        "tengah": [17800.0, 17800.0, 8700.0, 17750.0, 17875.0],
        "spread": [200.0, 200.0, 18400.0, -500.0, 150.0],
        "mata_uang": ["USD", "USD", "USD", "USD", "USD"],
    })

    with tempfile.TemporaryDirectory() as tmpdir:
        preprocessor = Preprocessor(raw_dir=tmpdir, processed_dir=tmpdir)
        clean_df, stats = preprocessor.clean_and_validate(dirty_data)

        # Row 0 & 1 are duplicates -> deduplicated to 1 row
        # Row 2 has negative beli -> dropped
        # Row 3 has beli > jual -> dropped
        # Row 4 is valid
        # Result should be 2 rows (Row 0/1 and Row 4)
        assert len(clean_df) == 2
        assert list(clean_df["tgl"]) == ["2026-09-22", "2026-09-25"]
        assert stats["dropped_duplicate_rows"] == 1
        assert stats["dropped_invalid_financial_rows"] == 2


def test_preprocessor_save_processed():
    sample_df = pd.DataFrame({
        "tgl": ["2026-09-28"],
        "beli": [17827.42],
        "jual": [18006.58],
        "tengah": [17917.0],
        "spread": [179.16],
        "mata_uang": ["USD"],
    })

    with tempfile.TemporaryDirectory() as tmpdir:
        processed_dir = Path(tmpdir) / "processed"
        preprocessor = Preprocessor(raw_dir=tmpdir, processed_dir=processed_dir)
        clean_df, stats = preprocessor.clean_and_validate(sample_df)
        saved = preprocessor.save_processed(clean_df, stats, output_filename="usd_idr_cleaned.csv")

        assert saved["csv"].exists()
        assert saved["json"].exists()
        assert saved["report"].exists()
