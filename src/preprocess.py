"""
Data Preprocessing Module for Bank Indonesia USD/IDR Exchange Rates.

Automates cleaning, sanitization, validation, and preparation of raw ingested
exchange rate data to ensure data readiness for feature extraction and continual learning.
"""

import argparse
import datetime
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("preprocess")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


class Preprocessor:
    """
    Automated preprocessing pipeline for foreign exchange time-series data.
    Enforces financial domain rules, removes duplicates, handles missing values,
    and structures clean data for downstream feature engineering.
    """

    def __init__(
        self,
        raw_dir: Union[str, Path] = DEFAULT_RAW_DIR,
        processed_dir: Union[str, Path] = DEFAULT_PROCESSED_DIR,
    ):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def find_latest_raw_file(self, pattern: str = "usd_idr_raw*.csv") -> Path:
        """
        Locate the most recent raw ingested file in the raw directory.
        Prefers canonical/latest pointer if present, otherwise picks newest by timestamp.
        """
        latest_pointer = self.raw_dir / "usd_idr_raw_latest.csv"
        if latest_pointer.exists():
            return latest_pointer

        canonical_pointer = self.raw_dir / "usd_idr_raw.csv"
        if canonical_pointer.exists():
            return canonical_pointer

        candidates = list(self.raw_dir.glob(pattern))
        if not candidates:
            raise FileNotFoundError(f"No raw files matching pattern '{pattern}' found in {self.raw_dir}")

        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return candidates[0]

    def load_raw_data(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """
        Load raw data from CSV or JSON format.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Input raw file does not exist: {path}")

        logger.info("Loading raw data from: %s", path)
        if path.suffix.lower() == ".csv":
            df = pd.read_csv(path)
        elif path.suffix.lower() == ".json":
            df = pd.read_json(path)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}. Must be .csv or .json")

        logger.info("Loaded raw dataset with %d rows and %d columns.", len(df), df.shape[1])
        return df

    def clean_and_validate(self, raw_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Execute core cleaning and financial sanity validation:
          1. Schema and Column verification
          2. Type coercion & string sanitization
          3. Missing values inspection & dropping
          4. Deduplication
          5. Strict chronological ordering
          6. Domain rule checks:
             - positive values: beli > 0, jual > 0
             - price ordering: beli <= jual
             - consistency: tengah = (beli + jual) / 2, spread = jual - beli > 0
        """
        initial_count = len(raw_df)
        df = raw_df.copy()

        # 1. Required column check
        required_cols = {"tgl", "beli", "jual"}
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns in raw dataset: {missing_cols}")

        # 2. String trimming & data typing
        if "mata_uang" in df.columns:
            df["mata_uang"] = df["mata_uang"].astype(str).str.strip().str.upper()
        else:
            df["mata_uang"] = "USD"

        # Coerce numeric columns
        for col in ["beli", "jual"]:
            if df[col].dtype == object:
                df[col] = df[col].astype(str).str.replace(",", "").str.strip()
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # 3. Missing values check
        null_counts = df[["tgl", "beli", "jual"]].isnull().sum().to_dict()
        df = df.dropna(subset=["tgl", "beli", "jual"]).copy()
        post_dropna_count = len(df)
        dropped_null_rows = initial_count - post_dropna_count

        # 4. Date parsing & strict sorting
        df["tgl"] = pd.to_datetime(df["tgl"], errors="coerce")
        df = df.dropna(subset=["tgl"]).copy()
        df["tgl"] = df["tgl"].dt.strftime("%Y-%m-%d")

        # Deduplication (keep the latest record if duplicates exist)
        pre_dedup_count = len(df)
        df = df.drop_duplicates(subset=["tgl"], keep="last")
        dropped_duplicate_rows = pre_dedup_count - len(df)

        # Sort chronologically
        df = df.sort_values("tgl", ascending=True).reset_index(drop=True)

        # 5. Financial sanity filters
        # Drop rows where beli <= 0 or jual <= 0
        valid_pos = (df["beli"] > 0) & (df["jual"] > 0)
        # Drop inverted bid/ask rates
        valid_order = df["beli"] <= df["jual"]
        valid_mask = valid_pos & valid_order

        dropped_invalid_financial = len(df) - valid_mask.sum()
        df = df[valid_mask].copy().reset_index(drop=True)

        # Re-derive exact arithmetic tengah & spread to eliminate floating point drift
        df["tengah"] = ((df["beli"] + df["jual"]) / 2.0).round(2)
        df["spread"] = (df["jual"] - df["beli"]).round(2)

        # 6. Additional validation: Spread must be strictly positive
        df = df[df["spread"] > 0].reset_index(drop=True)

        # Reorder canonical columns
        canonical_cols = ["tgl", "beli", "jual", "tengah", "spread", "mata_uang"]
        clean_df = df[canonical_cols]

        # Calculate time-series return metrics for continual learning monitoring
        clean_df_calc = clean_df.copy()
        clean_df_calc["daily_return_tengah"] = clean_df_calc["tengah"].pct_change().round(6)

        stats = {
            "initial_rows": initial_count,
            "final_clean_rows": len(clean_df),
            "dropped_null_rows": dropped_null_rows,
            "dropped_duplicate_rows": dropped_duplicate_rows,
            "dropped_invalid_financial_rows": int(dropped_invalid_financial),
            "null_analysis": null_counts,
            "date_range": {
                "start": str(clean_df["tgl"].min()) if not clean_df.empty else None,
                "end": str(clean_df["tgl"].max()) if not clean_df.empty else None,
            },
            "summary_metrics": {
                "min_tengah": float(clean_df["tengah"].min()) if not clean_df.empty else 0.0,
                "max_tengah": float(clean_df["tengah"].max()) if not clean_df.empty else 0.0,
                "mean_tengah": float(clean_df["tengah"].mean()) if not clean_df.empty else 0.0,
                "mean_spread": float(clean_df["spread"].mean()) if not clean_df.empty else 0.0,
                "daily_return_volatility_pct": float(clean_df_calc["daily_return_tengah"].std() * 100.0) if len(clean_df) > 1 else 0.0,
            },
            "validation_status": "PASSED" if not clean_df.empty else "EMPTY",
        }

        logger.info(
            "Preprocessing complete. Cleaned: %d records (Dropped: %d null, %d duplicate, %d invalid).",
            len(clean_df),
            dropped_null_rows,
            dropped_duplicate_rows,
            dropped_invalid_financial,
        )
        return clean_df, stats

    def save_processed(
        self,
        clean_df: pd.DataFrame,
        stats: Dict,
        output_filename: str = "usd_idr_cleaned.csv",
    ) -> Dict[str, Path]:
        """
        Save cleaned dataset and preprocessing report manifest.
        """
        output_path = self.processed_dir / output_filename
        clean_df.to_csv(output_path, index=False)
        logger.info("Saved clean dataset to: %s", output_path)

        # Also save JSON format for flexibility
        json_output_path = self.processed_dir / (Path(output_filename).stem + ".json")
        clean_df.to_json(json_output_path, orient="records", indent=2)

        # Save preprocessing report
        report_path = self.processed_dir / "preprocessing_report.json"
        report_payload = {
            "execution_timestamp": datetime.datetime.now().isoformat(),
            "output_file": str(output_path.relative_to(PROJECT_ROOT)) if output_path.is_relative_to(PROJECT_ROOT) else str(output_path),
            "statistics": stats,
        }
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report_payload, f, indent=2)
        logger.info("Saved preprocessing report to: %s", report_path)

        return {
            "csv": output_path,
            "json": json_output_path,
            "report": report_path,
        }

    def run(
        self,
        input_file: Optional[Union[str, Path]] = None,
        output_filename: str = "usd_idr_cleaned.csv",
    ) -> Tuple[pd.DataFrame, Dict[str, Path]]:
        """
        Execute end-to-end preprocessing workflow.
        """
        if input_file is None:
            input_path = self.find_latest_raw_file()
        else:
            input_path = Path(input_file)

        raw_df = self.load_raw_data(input_path)
        clean_df, stats = self.clean_and_validate(raw_df)
        saved_paths = self.save_processed(clean_df, stats, output_filename=output_filename)
        return clean_df, saved_paths


def preprocess_usd_idr_data(
    input_file: Optional[Union[str, Path]] = None,
    output_filename: str = "usd_idr_cleaned.csv",
    processed_dir: Union[str, Path] = DEFAULT_PROCESSED_DIR,
) -> Tuple[pd.DataFrame, Dict[str, Path]]:
    """Convenience function for direct programmatic usage."""
    preprocessor = Preprocessor(processed_dir=processed_dir)
    return preprocessor.run(input_file=input_file, output_filename=output_filename)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Automated Preprocessing Script for Raw Bank Indonesia USD/IDR Data"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to raw CSV/JSON file (default: automatically locate newest file in data/raw/)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_PROCESSED_DIR),
        help="Directory to save cleaned processed files (default: data/processed/)",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="usd_idr_cleaned.csv",
        help="Filename for the cleaned CSV dataset (default: usd_idr_cleaned.csv)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    preprocessor = Preprocessor(processed_dir=args.output_dir)
    clean_df, paths = preprocessor.run(
        input_file=args.input,
        output_filename=args.output_file,
    )

    print("\n" + "=" * 60)
    print(" [PREPROCESSING SUCCESS] USD/IDR Data Cleaning Summary")
    print("=" * 60)
    print(f"Clean Records Generated : {len(clean_df)}")
    if not clean_df.empty:
        print(f"Date Range Cleaned      : {clean_df['tgl'].min()} to {clean_df['tgl'].max()}")
        print(f"Current Kurs Tengah Mean: IDR {clean_df['tengah'].mean():,.2f}")
        print(f"Average Spread          : IDR {clean_df['spread'].mean():,.2f}")
    print("\nProcessed Output Files:")
    for key, path in paths.items():
        print(f"  • {key.upper()}: {path}")
    print("=" * 60 + "\n")
