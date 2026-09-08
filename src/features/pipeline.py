"""
Feature engineering pipeline for time-series exchange rate forecasting.
Ensures zero data leakage by computing rolling windows chronologically.
"""

from typing import Tuple
import pandas as pd
import numpy as np


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct time-series lag and rolling features.
    Target: next business day rate (H+1).
    """
    data = df.copy()
    data["tgl"] = pd.to_datetime(data["tgl"])
    data = data.sort_values("tgl").reset_index(drop=True)

    # Core price targets (H+1)
    data["target_jual"] = data["jual"].shift(-1)
    data["target_beli"] = data["beli"].shift(-1)

    # Autoregressive lag features
    for lag in [1, 2, 3, 5, 7, 14]:
        data[f"jual_lag_{lag}"] = data["jual"].shift(lag)
        data[f"beli_lag_{lag}"] = data["beli"].shift(lag)

    # Rolling statistics (computed strictly on past values)
    data["jual_ma_7"] = data["jual"].shift(1).rolling(window=7).mean()
    data["jual_ma_14"] = data["jual"].shift(1).rolling(window=14).mean()
    data["jual_std_7"] = data["jual"].shift(1).rolling(window=7).std()

    # Momentum / daily delta
    data["jual_diff_1"] = data["jual"].shift(1) - data["jual"].shift(2)
    data["spread_lag_1"] = data["spread"].shift(1)

    # Day of week (cyclical/seasonal effect in FX trading)
    data["day_of_week"] = data["tgl"].dt.dayofweek

    # Drop NaNs produced by lag/rolling operations
    return data


def time_series_split(
    data: pd.DataFrame, test_size: int = 14
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Time-aware holdout split (no random shuffling).
    The last `test_size` business days are used as the test/validation set.
    """
    # Filter rows where target is known (exclude the very last row whose target is in the future)
    labeled = data.dropna(subset=["target_jual", "jual_lag_14"]).reset_index(drop=True)
    if len(labeled) <= test_size:
        raise ValueError(f"Not enough data ({len(labeled)} rows) for test_size={test_size}")

    train = labeled.iloc[:-test_size].copy()
    test = labeled.iloc[-test_size:].copy()
    return train, test


FEATURE_COLS = [
    "jual_lag_1",
    "jual_lag_2",
    "jual_lag_3",
    "jual_lag_5",
    "jual_lag_7",
    "jual_lag_14",
    "jual_ma_7",
    "jual_ma_14",
    "jual_std_7",
    "jual_diff_1",
    "spread_lag_1",
    "day_of_week",
]
