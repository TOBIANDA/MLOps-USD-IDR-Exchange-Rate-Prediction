"""
Data ingestion module for Bank Indonesia Web Service (Kurs Transaksi BI).
Official endpoint: https://www.bi.go.id/biwebservice/wskursbi.asmx/getSubKursLokal3
"""

import datetime
import logging
import sqlite3
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BI_API_URL = "https://www.bi.go.id/biwebservice/wskursbi.asmx/getSubKursLokal3"
DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "exchange_rates.db"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/xml,text/xml,*/*;q=0.9",
}


def init_db(db_path: Path = DEFAULT_DB_PATH) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
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


def fetch_bi_rates(
    start_date: Optional[datetime.date] = None,
    end_date: Optional[datetime.date] = None,
    currency: str = "USD",
    timeout: int = 25,
) -> pd.DataFrame:
    """
    Fetch exchange rates from Bank Indonesia XML Web Service.
    """
    if end_date is None:
        end_date = datetime.date.today()
    if start_date is None:
        start_date = end_date - datetime.timedelta(days=180)

    url = f"{BI_API_URL}?mts={currency}&startdate={start_date}&enddate={end_date}"
    logger.info("Fetching BI rates: %s -> %s (URL: %s)", start_date, end_date, url)

    session = requests.Session()
    session.headers.update(HEADERS)
    resp = session.get(url, timeout=timeout)
    resp.raise_for_status()

    root = ET.fromstring(resp.content)
    rows = []
    for elem in root.iter():
        if elem.tag.endswith("Table"):
            row = {}
            for child in elem:
                tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
                row[tag] = child.text
            rows.append(row)

    if not rows:
        logger.warning("No records returned from BI webservice for range %s to %s", start_date, end_date)
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    # Standardize column types and names
    # XML fields: tgl_subkurslokal, beli_subkurslokal, jual_subkurslokal, mts_subkurslokal
    df["tgl"] = pd.to_datetime(df["tgl_subkurslokal"]).dt.strftime("%Y-%m-%d")
    df["beli"] = pd.to_numeric(df["beli_subkurslokal"].str.replace(",", ""), errors="coerce")
    df["jual"] = pd.to_numeric(df["jual_subkurslokal"].str.replace(",", ""), errors="coerce")
    df["tengah"] = ((df["beli"] + df["jual"]) / 2.0).round(2)
    df["spread"] = (df["jual"] - df["beli"]).round(2)
    df["mata_uang"] = df.get("mts_subkurslokal", currency)

    clean_df = (
        df[["tgl", "beli", "jual", "tengah", "spread", "mata_uang"]]
        .dropna()
        .drop_duplicates(subset=["tgl"])
        .sort_values("tgl", ascending=True)
        .reset_index(drop=True)
    )
    logger.info("Successfully parsed %d records from BI XML.", len(clean_df))
    return clean_df


def save_to_db(df: pd.DataFrame, db_path: Path = DEFAULT_DB_PATH) -> int:
    """
    Insert records with idempotency check (INSERT OR IGNORE).
    Returns number of newly inserted rows.
    """
    if df.empty:
        return 0

    init_db(db_path)
    records = df[["tgl", "beli", "jual", "tengah", "spread", "mata_uang"]].to_dict(orient="records")

    with sqlite3.connect(db_path) as conn:
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

    logger.info("Inserted %d new rows into database (%s).", inserted, db_path.name)
    return inserted


def load_from_db(db_path: Path = DEFAULT_DB_PATH) -> pd.DataFrame:
    init_db(db_path)
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(
            "SELECT tgl, beli, jual, tengah, spread, mata_uang FROM exchange_rates ORDER BY tgl ASC",
            conn,
        )
    return df


def sync_rates(days: int = 180, db_path: Path = DEFAULT_DB_PATH) -> Tuple[pd.DataFrame, int]:
    """
    Fetch rates from BI for the last `days` days and sync into SQLite DB.
    Returns (loaded_df, inserted_count).
    """
    init_db(db_path)
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=days)
    fetched_df = fetch_bi_rates(start_date=start_date, end_date=end_date)
    inserted = save_to_db(fetched_df, db_path=db_path)
    all_df = load_from_db(db_path=db_path)
    return all_df, inserted



if __name__ == "__main__":
    init_db()
    # Fetch past 180 days by default
    df = fetch_bi_rates()
    save_to_db(df)
    loaded = load_from_db()
    print("Database summary:")
    print(f"Total rows in DB: {len(loaded)}")
    print(loaded.tail(5))
