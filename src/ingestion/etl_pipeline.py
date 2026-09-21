"""
Automated ETL Pipeline for USD/IDR Exchange Rate Forecasting.
Covers:
  - Extract: Ingestion from Bank Indonesia Web Service (Kurs Transaksi BI)
  - Transform: Schema validation, data cleaning, sanity check, and temporal feature engineering
  - Load: Storage into raw CSV, SQLite DB, processed featured CSV, and JSON data manifest
"""

import datetime
import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

from src.features.pipeline import create_features
from src.ingestion.fetcher import DEFAULT_DB_PATH, fetch_bi_rates, load_from_db, save_to_db

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ETLPipeline")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_CSV_PATH = PROJECT_ROOT / "data" / "raw" / "usd_idr_raw.csv"
PROCESSED_CSV_PATH = PROJECT_ROOT / "data" / "processed" / "usd_idr_featured.csv"
MANIFEST_PATH = PROJECT_ROOT / "data" / "data_manifest.json"


def compute_sha256(filepath: Path) -> str:
    """Compute SHA256 hash of a file for integrity and DVC tracking."""
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            sha.update(chunk)
    return sha.hexdigest()


class ETLPipeline:
    def __init__(
        self,
        db_path: Path = DEFAULT_DB_PATH,
        raw_csv_path: Path = RAW_CSV_PATH,
        processed_csv_path: Path = PROCESSED_CSV_PATH,
        manifest_path: Path = MANIFEST_PATH,
    ):
        self.db_path = db_path
        self.raw_csv_path = raw_csv_path
        self.processed_csv_path = processed_csv_path
        self.manifest_path = manifest_path

    def extract(self, days: int = 180) -> pd.DataFrame:
        """
        Extract stage: Fetch latest exchange rates from Bank Indonesia Web Service.
        Falls back to local SQLite DB if remote network is unavailable.
        """
        logger.info("=== [ETL - EXTRACT] Fetching data for the past %d days ===", days)
        end_date = datetime.date.today()
        start_date = end_date - datetime.timedelta(days=days)

        try:
            df = fetch_bi_rates(start_date=start_date, end_date=end_date)
            if df.empty:
                logger.warning("BI returned 0 records. Falling back to local SQLite DB.")
                df = load_from_db(self.db_path)
            else:
                # Cache to database
                save_to_db(df, self.db_path)
        except Exception as err:
            logger.error("Error connecting to Bank Indonesia Web Service: %s. Using local DB fallback.", err)
            df = load_from_db(self.db_path)

        logger.info("Extracted %d records.", len(df))
        return df

    def validate_and_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform Stage 1: Schema validation, integrity check, and data cleaning.
        """
        logger.info("=== [ETL - TRANSFORM: CLEANING & VALIDATION] ===")
        if df.empty:
            raise ValueError("Extract returned empty DataFrame. Cannot proceed with transformation.")

        data = df.copy()
        expected_cols = {"tgl", "beli", "jual", "tengah", "spread"}
        missing_cols = expected_cols - set(data.columns)
        if missing_cols:
            raise ValueError(f"Schema error: Missing expected columns {missing_cols}")

        # Ensure correct types
        data["tgl"] = pd.to_datetime(data["tgl"])
        for col in ["beli", "jual", "tengah", "spread"]:
            data[col] = pd.to_numeric(data[col], errors="coerce")

        # Drop any null records in crucial fields
        before_drop = len(data)
        data = data.dropna(subset=["tgl", "beli", "jual", "tengah"]).reset_index(drop=True)
        dropped_nulls = before_drop - len(data)
        if dropped_nulls > 0:
            logger.warning("Dropped %d records containing null values.", dropped_nulls)

        # Sanity validation: Non-negative and correct price order (beli <= tengah <= jual)
        invalid_rates = data[(data["beli"] <= 0) | (data["jual"] <= 0) | (data["beli"] > data["jual"])]
        if not invalid_rates.empty:
            logger.warning("Found %d records violating rate sanity rules. Removing.", len(invalid_rates))
            data = data.drop(invalid_rates.index).reset_index(drop=True)

        # Ensure strict chronological sorting and deduplication
        data = data.drop_duplicates(subset=["tgl"]).sort_values("tgl", ascending=True).reset_index(drop=True)

        # Recalculate tengah & spread to guarantee exact mathematical consistency
        data["tengah"] = ((data["beli"] + data["jual"]) / 2.0).round(2)
        data["spread"] = (data["jual"] - data["beli"]).round(2)
        data["tgl"] = data["tgl"].dt.strftime("%Y-%m-%d")

        logger.info("Validation complete. Cleaned dataset has %d records from %s to %s.", len(data), data["tgl"].min(), data["tgl"].max())
        return data

    def transform_features(self, clean_df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform Stage 2: Time-series feature engineering with zero lookahead bias.
        """
        logger.info("=== [ETL - TRANSFORM: FEATURE ENGINEERING] ===")
        featured_df = create_features(clean_df)
        logger.info("Feature engineering complete. Shape: %s, Columns: %s", featured_df.shape, list(featured_df.columns))
        return featured_df

    def load(self, clean_df: pd.DataFrame, featured_df: pd.DataFrame) -> Dict:
        """
        Load stage: Write datasets to disk and generate metadata manifest.
        """
        logger.info("=== [ETL - LOAD] Saving datasets to disk ===")
        self.raw_csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.processed_csv_path.parent.mkdir(parents=True, exist_ok=True)

        # 1. Save raw clean dataset
        clean_df.to_csv(self.raw_csv_path, index=False)
        raw_hash = compute_sha256(self.raw_csv_path)

        # 2. Save processed feature dataset
        featured_df.to_csv(self.processed_csv_path, index=False)
        processed_hash = compute_sha256(self.processed_csv_path)

        try:
            raw_rel = str(self.raw_csv_path.relative_to(PROJECT_ROOT))
        except ValueError:
            raw_rel = str(self.raw_csv_path)

        try:
            proc_rel = str(self.processed_csv_path.relative_to(PROJECT_ROOT))
        except ValueError:
            proc_rel = str(self.processed_csv_path)

        # 3. Compile metadata manifest for DVC tracking
        manifest = {
            "execution_timestamp": datetime.datetime.now().isoformat(),
            "source": "Bank Indonesia Web Service (getSubKursLokal3)",
            "currency_pair": "USD/IDR",
            "raw_dataset": {
                "file_path": raw_rel,
                "total_records": len(clean_df),
                "date_range": {
                    "start": str(clean_df["tgl"].min()),
                    "end": str(clean_df["tgl"].max()),
                },
                "sha256": raw_hash,
                "file_size_bytes": self.raw_csv_path.stat().st_size,
            },
            "processed_dataset": {
                "file_path": proc_rel,
                "total_records": len(featured_df),
                "total_features": featured_df.shape[1],
                "sha256": processed_hash,
                "file_size_bytes": self.processed_csv_path.stat().st_size,
            },
            "validation_status": "PASSED",
        }

        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        logger.info("Data Manifest written to %s", self.manifest_path.name)
        return manifest

    def run(self, days: int = 180) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        """Execute full ETL pipeline run."""
        raw_df = self.extract(days=days)
        clean_df = self.validate_and_clean(raw_df)
        featured_df = self.transform_features(clean_df)
        manifest = self.load(clean_df, featured_df)
        logger.info("=== ETL PIPELINE COMPLETED SUCCESSFULLY ===")
        return clean_df, featured_df, manifest


if __name__ == "__main__":
    pipeline = ETLPipeline()
    clean_df, featured_df, manifest = pipeline.run(days=180)
    print("\n--- Pipeline Execution Summary ---")
    print(json.dumps(manifest, indent=2))
