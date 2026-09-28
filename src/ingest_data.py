"""
Data Ingestion Module for Bank Indonesia Web Service (Kurs Transaksi BI).
Official endpoint: https://www.bi.go.id/biwebservice/wskursbi.asmx/getSubKursLokal3

Provides automated, periodic, and non-destructive dynamic data ingestion
to support Continual Learning in USD/IDR exchange rate forecasting.
"""

import argparse
import datetime
import json
import logging
import sqlite3
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ingest_data")

BI_API_URL = "https://www.bi.go.id/biwebservice/wskursbi.asmx/getSubKursLokal3"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "exchange_rates.db"

REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


class IngestionWorker:
    """
    Automated Data Ingestion Worker for fetching Bank Indonesia foreign exchange rates.
    Supports non-destructive periodic execution, multi-format export, and SQLite caching.
    """

    def __init__(
        self,
        api_url: str = BI_API_URL,
        raw_dir: Union[str, Path] = DEFAULT_RAW_DIR,
        db_path: Union[str, Path] = DEFAULT_DB_PATH,
        timeout: int = 25,
    ):
        self.api_url = api_url
        self.raw_dir = Path(raw_dir)
        self.db_path = Path(db_path)
        self.timeout = timeout
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def init_database(self) -> None:
        """Initialize SQLite database for idempotent continual storage."""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS exchange_rates (
                    tgl TEXT PRIMARY KEY,
                    beli REAL NOT NULL,
                    jual REAL NOT NULL,
                    tengah REAL NOT NULL,
                    spread REAL NOT NULL,
                    mata_uang TEXT NOT NULL DEFAULT 'USD',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.commit()
        finally:
            conn.close()

    def fetch_dynamic_rates(
        self,
        days: int = 30,
        currency: str = "USD",
        start_date: Optional[datetime.date] = None,
        end_date: Optional[datetime.date] = None,
    ) -> pd.DataFrame:
        """
        Fetch dynamic exchange rate data from Bank Indonesia SOAP/XML endpoint.
        Returns a raw pandas DataFrame.
        """
        if end_date is None:
            end_date = datetime.date.today()
        if start_date is None:
            start_date = end_date - datetime.timedelta(days=days)

        url = f"{self.api_url}?mts={currency}&startdate={start_date}&enddate={end_date}"
        logger.info("Connecting to Bank Indonesia API: %s to %s", start_date, end_date)
        logger.info("Endpoint: %s", url)

        session = requests.Session()
        session.headers.update(REQUEST_HEADERS)

        try:
            resp = session.get(url, timeout=self.timeout)
            resp.raise_for_status()
        except requests.RequestException as exc:
            logger.error("Failed to connect to Bank Indonesia API: %s", exc)
            # Attempt fallback from local database cache if available
            if self.db_path.exists():
                logger.warning("Attempting fallback from local database cache (%s)...", self.db_path)
                return self.load_from_db_cache(days=days)
            raise

        root = ET.fromstring(resp.content)
        records: List[Dict[str, Optional[str]]] = []
        for elem in root.iter():
            if elem.tag.endswith("Table"):
                row = {}
                for child in elem:
                    tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
                    row[tag] = child.text
                records.append(row)

        if not records:
            logger.warning("No records returned from Bank Indonesia for range %s - %s", start_date, end_date)
            return pd.DataFrame()

        df = pd.DataFrame(records)

        # Standardize raw column structure
        df["tgl"] = pd.to_datetime(df["tgl_subkurslokal"]).dt.strftime("%Y-%m-%d")
        df["beli"] = pd.to_numeric(df["beli_subkurslokal"].str.replace(",", ""), errors="coerce")
        df["jual"] = pd.to_numeric(df["jual_subkurslokal"].str.replace(",", ""), errors="coerce")
        df["tengah"] = ((df["beli"] + df["jual"]) / 2.0).round(2)
        df["spread"] = (df["jual"] - df["beli"]).round(2)
        df["mata_uang"] = df.get("mts_subkurslokal", currency)

        raw_df = (
            df[["tgl", "beli", "jual", "tengah", "spread", "mata_uang"]]
            .dropna()
            .drop_duplicates(subset=["tgl"])
            .sort_values("tgl", ascending=True)
            .reset_index(drop=True)
        )

        logger.info("Fetched %d exchange rate records from Bank Indonesia.", len(raw_df))
        return raw_df

    def sync_to_sqlite(self, df: pd.DataFrame) -> int:
        """
        Idempotent database sync using INSERT OR IGNORE.
        Ensures continuous historical collection without duplicates.
        """
        if df.empty:
            return 0

        self.init_database()
        records = df[["tgl", "beli", "jual", "tengah", "spread", "mata_uang"]].to_dict(orient="records")

        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.executemany(
                """
                INSERT OR IGNORE INTO exchange_rates (tgl, beli, jual, tengah, spread, mata_uang)
                VALUES (:tgl, :beli, :jual, :tengah, :spread, :mata_uang)
                """,
                records,
            )
            inserted = cursor.rowcount
            conn.commit()
        finally:
            conn.close()

        logger.info("Synchronized to SQLite: %d new rows inserted into %s.", inserted, self.db_path.name)
        return inserted

    def load_from_db_cache(self, days: int = 30) -> pd.DataFrame:
        """Fallback query from SQLite cache."""
        self.init_database()
        conn = sqlite3.connect(self.db_path)
        try:
            df = pd.read_sql_query(
                "SELECT tgl, beli, jual, tengah, spread, mata_uang FROM exchange_rates ORDER BY tgl ASC",
                conn,
            )
        finally:
            conn.close()
        if not df.empty and days:
            df = df.tail(days).reset_index(drop=True)
        return df

    def save_raw_dataset(
        self,
        df: pd.DataFrame,
        base_name: str = "usd_idr_raw",
        export_formats: Optional[List[str]] = None,
        use_timestamp: bool = True,
    ) -> Dict[str, Path]:
        """
        Save raw dataset non-destructively with timestamped filename.
        Also creates/updates the latest pointer (e.g. usd_idr_raw_latest.csv and usd_idr_raw.csv).
        """
        if export_formats is None:
            export_formats = ["csv"]

        now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_paths: Dict[str, Path] = {}

        for fmt in export_formats:
            fmt_lower = fmt.lower()
            if use_timestamp:
                timestamped_filename = f"{base_name}_{now_str}.{fmt_lower}"
                timestamped_path = self.raw_dir / timestamped_filename

                if fmt_lower == "csv":
                    df.to_csv(timestamped_path, index=False)
                elif fmt_lower == "json":
                    df.to_json(timestamped_path, orient="records", indent=2)
                elif fmt_lower == "parquet":
                    df.to_parquet(timestamped_path, index=False)
                else:
                    raise ValueError(f"Unsupported export format: {fmt_lower}")

                saved_paths[f"{fmt_lower}_timestamped"] = timestamped_path
                logger.info("Saved non-destructive raw snapshot: %s", timestamped_path)

            # Master / Latest copy for seamless downstream pipeline consumption
            latest_filename = f"{base_name}_latest.{fmt_lower}"
            latest_path = self.raw_dir / latest_filename
            canonical_filename = f"{base_name}.{fmt_lower}"
            canonical_path = self.raw_dir / canonical_filename

            if fmt_lower == "csv":
                df.to_csv(latest_path, index=False)
                df.to_csv(canonical_path, index=False)
            elif fmt_lower == "json":
                df.to_json(latest_path, orient="records", indent=2)
                df.to_json(canonical_path, orient="records", indent=2)
            elif fmt_lower == "parquet":
                df.to_parquet(latest_path, index=False)
                df.to_parquet(canonical_path, index=False)

            saved_paths[f"{fmt_lower}_latest"] = latest_path
            saved_paths[f"{fmt_lower}_canonical"] = canonical_path

        # Save metadata run manifest
        manifest = {
            "ingestion_timestamp": datetime.datetime.now().isoformat(),
            "source_endpoint": self.api_url,
            "currency": "USD",
            "total_records": len(df),
            "date_range": {
                "start": str(df["tgl"].min()) if not df.empty else None,
                "end": str(df["tgl"].max()) if not df.empty else None,
            },
            "saved_files": [str(p.relative_to(PROJECT_ROOT)) if p.is_relative_to(PROJECT_ROOT) else str(p) for p in saved_paths.values()],
        }

        manifest_path = self.raw_dir / f"ingestion_manifest_{now_str}.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        saved_paths["manifest"] = manifest_path

        return saved_paths

    def run(
        self,
        days: int = 30,
        currency: str = "USD",
        export_formats: Optional[List[str]] = None,
        sync_db: bool = True,
        use_timestamp: bool = True,
    ) -> Tuple[pd.DataFrame, Dict[str, Path]]:
        """
        Execute full end-to-end ingestion workflow.
        """
        raw_df = self.fetch_dynamic_rates(days=days, currency=currency)
        if raw_df.empty:
            logger.warning("Empty dataset received from ingestion. No file written.")
            return raw_df, {}

        if sync_db:
            self.sync_to_sqlite(raw_df)

        saved_files = self.save_raw_dataset(
            df=raw_df,
            base_name="usd_idr_raw",
            export_formats=export_formats or ["csv"],
            use_timestamp=use_timestamp,
        )
        logger.info("Ingestion workflow finished successfully. Records: %d", len(raw_df))
        return raw_df, saved_files


def ingest_usd_idr_rates(
    days: int = 30,
    output_dir: Union[str, Path] = DEFAULT_RAW_DIR,
    export_formats: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Path]]:
    """Convenience function for direct programmatic usage."""
    worker = IngestionWorker(raw_dir=output_dir)
    return worker.run(days=days, export_formats=export_formats)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Automated Data Ingestion Script for Bank Indonesia USD/IDR Exchange Rates"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Number of historical business days to fetch (default: 30)",
    )
    parser.add_argument(
        "--currency",
        type=str,
        default="USD",
        help="Target currency code (default: USD)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_RAW_DIR),
        help="Directory to save raw ingested files (default: data/raw)",
    )
    parser.add_argument(
        "--format",
        choices=["csv", "json", "all"],
        default="csv",
        help="Output file format (csv, json, or all; default: csv)",
    )
    parser.add_argument(
        "--no-timestamp",
        action="store_true",
        help="Disable timestamped filenames (only write canonical usd_idr_raw.csv)",
    )
    parser.add_argument(
        "--no-db-sync",
        action="store_true",
        help="Skip syncing ingested records to local SQLite database",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    formats = ["csv", "json"] if args.format == "all" else [args.format]

    worker = IngestionWorker(raw_dir=args.output_dir)
    df, paths = worker.run(
        days=args.days,
        currency=args.currency,
        export_formats=formats,
        sync_db=not args.no_db_sync,
        use_timestamp=not args.no_timestamp,
    )

    print("\n" + "=" * 60)
    print(" [INGESTION SUCCESS] Bank Indonesia Data Ingestion Summary")
    print("=" * 60)
    print(f"Total Records Ingested : {len(df)}")
    if not df.empty:
        print(f"Date Coverage Range    : {df['tgl'].min()} to {df['tgl'].max()}")
        print(f"Latest Rate (Beli/Jual): IDR {df.iloc[-1]['beli']:,.2f} / IDR {df.iloc[-1]['jual']:,.2f}")
    print("\nSaved Output Files:")
    for key, path in paths.items():
        print(f"  • {key.upper()}: {path}")
    print("=" * 60 + "\n")
