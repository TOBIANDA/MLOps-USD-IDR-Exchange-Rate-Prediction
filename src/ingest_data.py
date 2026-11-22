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
import time
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

    def reset_database(self) -> None:
        """Clear existing records from SQLite table prior to a fresh historical load."""
        self.init_database()
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("DELETE FROM exchange_rates")
            conn.commit()
        finally:
            conn.close()
        logger.info("Cleared existing rows in %s.", self.db_path.name)

    def _parse_bi_xml(self, content: bytes, currency: str) -> pd.DataFrame:
        """Parse Bank Indonesia XML DataSet payload into standardized rate records."""
        root = ET.fromstring(content)
        records: List[Dict[str, Optional[str]]] = []
        for elem in root.iter():
            if elem.tag.endswith("Table"):
                row: Dict[str, Optional[str]] = {}
                for child in elem:
                    tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
                    row[tag] = child.text
                records.append(row)

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records)
        if "tgl_subkurslokal" not in df.columns or "beli_subkurslokal" not in df.columns or "jual_subkurslokal" not in df.columns:
            return pd.DataFrame()

        df["tgl"] = pd.to_datetime(df["tgl_subkurslokal"], errors="coerce").dt.strftime("%Y-%m-%d")
        df["beli"] = pd.to_numeric(df["beli_subkurslokal"].astype(str).str.replace(",", ""), errors="coerce")
        df["jual"] = pd.to_numeric(df["jual_subkurslokal"].astype(str).str.replace(",", ""), errors="coerce")
        df = df.dropna(subset=["tgl", "beli", "jual"]).copy()
        df = df[(df["beli"] > 0) & (df["jual"] > 0) & (df["beli"] <= df["jual"])].copy()

        df["tengah"] = ((df["beli"] + df["jual"]) / 2.0).round(2)
        df["spread"] = (df["jual"] - df["beli"]).round(2)
        if "mts_subkurslokal" in df.columns:
            df["mata_uang"] = df["mts_subkurslokal"].astype(str).str.strip()
        else:
            df["mata_uang"] = currency

        return (
            df[["tgl", "beli", "jual", "tengah", "spread", "mata_uang"]]
            .dropna()
            .drop_duplicates(subset=["tgl"])
            .sort_values("tgl", ascending=True)
            .reset_index(drop=True)
        )

    def fetch_dynamic_rates(
        self,
        days: int = 730,
        currency: str = "USD",
        start_date: Optional[datetime.date] = None,
        end_date: Optional[datetime.date] = None,
        chunk_days: int = 120,
        allow_db_fallback: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch dynamic exchange rate data from Bank Indonesia SOAP/XML endpoint.
        Automatically chunks multi-month/multi-year windows into `chunk_days` spans.
        Returns a raw chronologically sorted pandas DataFrame.
        """
        if end_date is None:
            end_date = datetime.date.today()
        if start_date is None:
            start_date = end_date - datetime.timedelta(days=days)

        chunks: List[Tuple[datetime.date, datetime.date]] = []
        cur_start = start_date
        while cur_start <= end_date:
            cur_end = min(cur_start + datetime.timedelta(days=chunk_days - 1), end_date)
            chunks.append((cur_start, cur_end))
            cur_start = cur_end + datetime.timedelta(days=1)

        session = requests.Session()
        session.headers.update(REQUEST_HEADERS)

        chunk_frames: List[pd.DataFrame] = []
        try:
            for idx, (c_start, c_end) in enumerate(chunks):
                if idx > 0:
                    time.sleep(0.8)
                url = f"{self.api_url}?mts={currency}&startdate={c_start}&enddate={c_end}"
                logger.info(
                    "Fetching BI chunk %d/%d: %s to %s (%s)",
                    idx + 1,
                    len(chunks),
                    c_start,
                    c_end,
                    url,
                )
                resp = session.get(url, timeout=self.timeout)
                logger.info("Chunk %d/%d HTTP status: %s (%d bytes)", idx + 1, len(chunks), resp.status_code, len(resp.content))
                resp.raise_for_status()

                chunk_df = self._parse_bi_xml(resp.content, currency=currency)
                logger.info("Chunk %d/%d parsed %d valid business-day rows.", idx + 1, len(chunks), len(chunk_df))
                if not chunk_df.empty:
                    chunk_frames.append(chunk_df)
        except (requests.RequestException, ET.ParseError) as exc:
            logger.error("Failed to fetch from Bank Indonesia API: %s", exc)
            if allow_db_fallback and self.db_path.exists():
                logger.warning("Attempting fallback from local database cache (%s)...", self.db_path)
                return self.load_from_db_cache(days=days)
            raise

        if not chunk_frames:
            logger.warning("No records returned from Bank Indonesia for range %s - %s", start_date, end_date)
            return pd.DataFrame()

        raw_df = (
            pd.concat(chunk_frames, ignore_index=True)
            .drop_duplicates(subset=["tgl"], keep="last")
            .sort_values("tgl", ascending=True)
            .reset_index(drop=True)
        )

        logger.info("Fetched %d total unique exchange rate records from Bank Indonesia.", len(raw_df))
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

    def load_from_db_cache(self, days: Optional[int] = None) -> pd.DataFrame:
        """Query records from SQLite cache (returns all rows when days is None)."""
        self.init_database()
        conn = sqlite3.connect(self.db_path)
        try:
            df = pd.read_sql_query(
                "SELECT tgl, beli, jual, tengah, spread, mata_uang FROM exchange_rates ORDER BY tgl ASC",
                conn,
            )
        finally:
            conn.close()
        if not df.empty and days is not None and days > 0:
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
        days: int = 730,
        currency: str = "USD",
        export_formats: Optional[List[str]] = None,
        sync_db: bool = True,
        use_timestamp: bool = True,
        reset_db: bool = False,
    ) -> Tuple[pd.DataFrame, Dict[str, Path]]:
        """
        Execute full end-to-end ingestion workflow.
        """
        raw_df = self.fetch_dynamic_rates(
            days=days,
            currency=currency,
            allow_db_fallback=not reset_db,
        )
        if raw_df.empty:
            logger.warning("Empty dataset received from ingestion. No file written.")
            return raw_df, {}

        if sync_db:
            if reset_db:
                self.reset_database()
            self.sync_to_sqlite(raw_df)
            raw_df = self.load_from_db_cache()

        saved_files = self.save_raw_dataset(
            df=raw_df,
            base_name="usd_idr_raw",
            export_formats=export_formats or ["csv"],
            use_timestamp=use_timestamp,
        )
        logger.info("Ingestion workflow finished successfully. Records: %d", len(raw_df))
        return raw_df, saved_files


def ingest_usd_idr_rates(
    days: int = 730,
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
        default=730,
        help="Calendar days lookback window to fetch in chunks (default: 730)",
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
    parser.add_argument(
        "--reset-db",
        action="store_true",
        help="Clear existing SQLite exchange_rates rows before inserting freshly fetched data",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate visual analysis dashboard after ingestion",
    )
    parser.add_argument(
        "--table",
        action="store_true",
        help="Display formatted tabular visualization of ingested rates in terminal and export HTML/Markdown table",
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
        reset_db=args.reset_db,
    )

    plot_path = None
    if args.plot and not df.empty:
        try:
            if str(PROJECT_ROOT) not in sys.path:
                sys.path.insert(0, str(PROJECT_ROOT))
            from src.visualize import generate_exchange_rate_dashboard
            plot_file = PROJECT_ROOT / "docs" / "visualisasi_kurs_usd_idr.png"
            plot_path = generate_exchange_rate_dashboard(df, plot_file)
            paths["visualization"] = plot_path
        except Exception as err:
            logger.warning(f"Could not generate plot: {err}")

    if args.table and not df.empty:
        try:
            if str(PROJECT_ROOT) not in sys.path:
                sys.path.insert(0, str(PROJECT_ROOT))
            from src.visualize import (
                format_exchange_rate_table,
                format_summary_metrics_table,
                export_markdown_table,
                export_html_table,
            )
            print("\n" + "=" * 80)
            print(" [VISUALISASI TABEL KURS USD/IDR BANK INDONESIA]")
            print("=" * 80)
            print(format_exchange_rate_table(df, limit=args.days))
            print("\n>> RINGKASAN METRIK OBSERVASI:")
            print(format_summary_metrics_table(df))

            md_path = export_markdown_table(df, PROJECT_ROOT / "docs" / "tabel_kurs_usd_idr.md", limit=args.days)
            html_path = export_html_table(df, PROJECT_ROOT / "docs" / "tabel_kurs_usd_idr.html", limit=args.days)
            paths["table_markdown"] = md_path
            paths["table_html"] = html_path
        except Exception as err:
            logger.warning(f"Could not render table: {err}")

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
