"""
Unit tests for data ingestion module.
"""

from pathlib import Path
import pandas as pd
from src.ingestion.fetcher import load_from_db

def test_load_from_db():
    df = load_from_db()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    expected_cols = {"tgl", "beli", "jual", "tengah", "spread", "mata_uang"}
    assert expected_cols.issubset(set(df.columns))
    assert (df["jual"] >= df["beli"]).all(), "Kurs Jual should be greater than or equal to Kurs Beli"
