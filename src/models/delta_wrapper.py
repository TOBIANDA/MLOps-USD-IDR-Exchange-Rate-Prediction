"""
Stationary Delta Forecasting Wrapper for Foreign Exchange Time Series.

Transforms non-stationary exchange rate level features (9 canonical input columns)
into strictly stationary, zero-lookahead differences/momentum features, fits a base
scikit-learn regressor on daily price changes (y_{t+1} - y_t), and reconstructs
predicted IDR levels (y_t + delta_hat) during inference.
"""

from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

INPUT_FEATURE_COLS: List[str] = [
    "jual_lag_1",
    "jual_lag_2",
    "jual_lag_3",
    "jual_lag_5",
    "jual_ma_3",
    "jual_ma_5",
    "jual_diff_1",
    "day_of_week",
    "spread_lag_1",
]


class DeltaForecastRegressor(BaseEstimator, RegressorMixin):
    """Scikit-learn compatible wrapper that forecasts daily USD/IDR changes."""

    def __init__(self, estimator: Optional[BaseEstimator] = None, use_scaler: bool = False):
        self.estimator = estimator
        self.use_scaler = use_scaler

    def _to_frame(self, X) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X
        arr = np.asarray(X, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        cols = list(getattr(self, "feature_names_in_", INPUT_FEATURE_COLS))
        return pd.DataFrame(arr, columns=cols)

    def _extract_stationary_features(self, X) -> tuple[np.ndarray, np.ndarray]:
        df = self._to_frame(X)
        lag1 = df["jual_lag_1"].to_numpy(dtype=float)
        lag2 = df["jual_lag_2"].to_numpy(dtype=float)
        lag3 = df["jual_lag_3"].to_numpy(dtype=float)
        lag5 = df["jual_lag_5"].to_numpy(dtype=float)
        ma3 = df["jual_ma_3"].to_numpy(dtype=float)
        ma5 = df["jual_ma_5"].to_numpy(dtype=float)
        diff1 = df["jual_diff_1"].to_numpy(dtype=float)
        dow = df["day_of_week"].to_numpy(dtype=float)
        spread = df["spread_lag_1"].to_numpy(dtype=float)

        diff2 = lag2 - lag3
        mom_2d = lag1 - lag3
        mom_4d = lag1 - lag5
        gap_ma3 = lag1 - ma3
        gap_ma5 = lag1 - ma5
        ma3_minus_ma5 = ma3 - ma5
        safe_lag1 = np.maximum(lag1, 1.0)
        ret_1d_pct = (diff1 / safe_lag1) * 100.0
        spread_pct = (spread / safe_lag1) * 100.0
        sin_dow = np.sin(2.0 * np.pi * dow / 5.0)
        cos_dow = np.cos(2.0 * np.pi * dow / 5.0)

        X_stat = np.column_stack(
            [
                diff1,
                diff2,
                mom_2d,
                mom_4d,
                gap_ma3,
                gap_ma5,
                ma3_minus_ma5,
                ret_1d_pct,
                spread_pct,
                dow,
                sin_dow,
                cos_dow,
            ]
        )
        return X_stat, lag1

    def fit(self, X, y):
        df = self._to_frame(X)
        self.feature_names_in_ = np.array(list(df.columns), dtype=object)
        self.n_features_in_ = len(self.feature_names_in_)

        X_stat, anchor = self._extract_stationary_features(df)
        y_delta = np.asarray(y, dtype=float) - anchor

        base_est = clone(self.estimator) if self.estimator is not None else Ridge(alpha=10.0)
        if self.use_scaler:
            self.estimator_ = Pipeline([("scaler", StandardScaler()), ("model", base_est)])
        else:
            self.estimator_ = base_est

        self.estimator_.fit(X_stat, y_delta)
        return self

    def predict(self, X) -> np.ndarray:
        X_stat, anchor = self._extract_stationary_features(X)
        pred_delta = self.estimator_.predict(X_stat)
        return anchor + np.asarray(pred_delta, dtype=float)
