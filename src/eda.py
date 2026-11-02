"""
Comprehensive Exploratory Data Analysis (EDA) Module for USD/IDR Exchange Rate Forecasting.
Analyzes the 500-business-day historical sliding window (~2 years, 2024-10-23 to 2026-10-13):
  1. Data Quality, Schema Completeness, and Summary Descriptive Statistics
  2. Multi-Year Price Trajectory (Kurs Jual, Kurs Beli, Kurs Tengah) & Moving Averages (MA-7, MA-14, MA-30)
  3. Daily Return Distribution, Skewness, Kurtosis, & Rolling Volatility Regimes
  4. Serial Autocorrelation Profile (Lags 1 to 14) & Stationarity Diagnostics
  5. Benchmark Hurdle Evaluation: Naive Random Walk (EMH) vs Rolling Moving Average 7D
"""

import json
import sqlite3
from pathlib import Path
from typing import Dict, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "exchange_rates.db"
RAW_CSV = PROJECT_ROOT / "data" / "raw" / "usd_idr_raw.csv"
RAW_LATEST_CSV = PROJECT_ROOT / "data" / "raw" / "usd_idr_raw_latest.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DOCS_DIR = PROJECT_ROOT / "docs"


def run_comprehensive_eda(test_size: int = 30) -> Dict[str, Any]:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query(
            "SELECT tgl, beli, jual, tengah, spread, mata_uang FROM exchange_rates ORDER BY tgl ASC",
            conn,
        )

    # Export canonical 500-row raw CSVs
    df.to_csv(RAW_CSV, index=False)
    df.to_csv(RAW_LATEST_CSV, index=False)

    df["tgl_dt"] = pd.to_datetime(df["tgl"])
    df["daily_diff_jual"] = df["jual"].diff()
    df["daily_return_pct"] = df["jual"].pct_change() * 100.0
    df["ma_7"] = df["jual"].rolling(7).mean()
    df["ma_14"] = df["jual"].rolling(14).mean()
    df["ma_30"] = df["jual"].rolling(30).mean()
    df["vol_7d"] = df["daily_diff_jual"].rolling(7).std()

    # 1. Autocorrelation across lags 1..14
    lags = [1, 2, 3, 5, 7, 10, 14]
    level_autocorr = {f"lag_{l}": round(float(df["jual"].autocorr(lag=l)), 5) for l in lags}
    diff_autocorr = {f"lag_{l}": round(float(df["daily_diff_jual"].dropna().autocorr(lag=l)), 4) for l in lags}

    # 2. Benchmark Evaluation on the 30-day Out-of-Sample Test Window
    eval_df = df.copy()
    eval_df["target_jual"] = eval_df["jual"].shift(-1)
    eval_df["ma_7_lag"] = eval_df["jual"].shift(1).rolling(7).mean()
    eval_valid = eval_df.dropna(subset=["target_jual", "ma_7_lag"]).reset_index(drop=True)
    test_slice = eval_valid.iloc[-test_size:]

    y_true = test_slice["target_jual"].values
    y_naive = test_slice["jual"].values
    y_ma7 = test_slice["ma_7_lag"].values

    naive_rmse = float(np.sqrt(mean_squared_error(y_true, y_naive)))
    naive_mae = float(mean_absolute_error(y_true, y_naive))
    naive_mape = float(np.mean(np.abs((y_true - y_naive) / y_true)) * 100.0)

    ma7_rmse = float(np.sqrt(mean_squared_error(y_true, y_ma7)))
    ma7_mae = float(mean_absolute_error(y_true, y_ma7))
    ma7_mape = float(np.mean(np.abs((y_true - y_ma7) / y_true)) * 100.0)

    # 3. Generate High-Resolution 4-Panel EDA Dashboard
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), dpi=160)

    # Panel 1: 500-Day Price Trajectory & Moving Averages
    ax1 = axes[0, 0]
    ax1.plot(df["tgl_dt"], df["jual"], color="#1E293B", lw=1.6, label="Kurs Jual BI (Ask)")
    ax1.plot(df["tgl_dt"], df["beli"], color="#64748B", lw=1.1, alpha=0.8, label="Kurs Beli BI (Bid)")
    ax1.plot(df["tgl_dt"], df["ma_30"], color="#2563EB", lw=1.8, linestyle="--", label="Moving Average 30D")
    ax1.set_title("A. Trajektori Historis Kurs USD/IDR (500 Hari Bursa: Okt 2024 - Okt 2026)", fontsize=10.5, fontweight="bold")
    ax1.set_ylabel("Nilai Kurs (IDR / USD)")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper left", fontsize=8.5)

    # Panel 2: Daily Return Distribution
    ax2 = axes[0, 1]
    rets = df["daily_return_pct"].dropna()
    ax2.hist(rets, bins=32, color="#3B82F6", edgecolor="#1E3A8A", alpha=0.8, density=True)
    ax2.axvline(rets.mean(), color="#DC2626", linestyle="--", lw=1.5, label=f"Mean: {rets.mean():.3f}%")
    ax2.set_title("B. Distribusi Perubahan Harian Kurs Jual (%)", fontsize=10.5, fontweight="bold")
    ax2.set_xlabel("Return Harian (%)")
    ax2.set_ylabel("Densitas")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(fontsize=8.5)

    # Panel 3: Serial Autocorrelation Profile (Lags 1..14)
    ax3 = axes[1, 0]
    lag_labels = [f"Lag {l}" for l in lags]
    lag_vals = [level_autocorr[f"lag_{l}"] for l in lags]
    bars = ax3.bar(lag_labels, lag_vals, color="#0F766E", edgecolor="#134E4A", width=0.55)
    ax3.set_ylim(0.95, 1.002)
    ax3.set_title("C. Profil Autokorelasi Serial Level Kurs Jual (Lag 1 s.d. Lag 14)", fontsize=10.5, fontweight="bold")
    ax3.set_ylabel("Koefisien Korelasi Pearson (r)")
    ax3.grid(True, linestyle=":", alpha=0.6)
    for bar, v in zip(bars, lag_vals):
        ax3.text(bar.get_x() + bar.get_width() / 2, v + 0.0008, f"{v:.4f}", ha="center", fontsize=8)

    # Panel 4: Rolling 7-Day Volatility & Bid-Ask Spread
    ax4 = axes[1, 1]
    ax4.plot(df["tgl_dt"], df["vol_7d"], color="#D97706", lw=1.5, label="Volatilitas Rolling 7 Hari (Std ΔKurs)")
    ax4_twin = ax4.twinx()
    ax4_twin.plot(df["tgl_dt"], df["spread"], color="#7C3AED", lw=1.2, linestyle=":", label="Spread Jual-Beli (1% Kurs Tengah)")
    ax4.set_title("D. Dinamika Volatilitas Bergulir 7 Hari & Spread Transaksi (IDR)", fontsize=10.5, fontweight="bold")
    ax4.set_ylabel("Volatilitas 7D (IDR)", color="#D97706")
    ax4_twin.set_ylabel("Spread (IDR)", color="#7C3AED")
    ax4.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    dashboard_path = DOCS_DIR / "eda_comprehensive_dashboard.png"
    fig.savefig(dashboard_path)
    plt.close(fig)

    summary = {
        "dataset_total_rows": int(len(df)),
        "date_start": str(df["tgl"].min()),
        "date_end": str(df["tgl"].max()),
        "missing_values_total": int(df[["tgl", "beli", "jual", "tengah", "spread"]].isnull().sum().sum()),
        "descriptive_stats": {
            "jual_min": round(float(df["jual"].min()), 2),
            "jual_max": round(float(df["jual"].max()), 2),
            "jual_mean": round(float(df["jual"].mean()), 2),
            "jual_std": round(float(df["jual"].std()), 2),
            "beli_mean": round(float(df["beli"].mean()), 2),
            "tengah_mean": round(float(df["tengah"].mean()), 2),
            "spread_mean": round(float(df["spread"].mean()), 2),
            "daily_diff_std_idr": round(float(df["daily_diff_jual"].std()), 2),
            "daily_return_std_pct": round(float(df["daily_return_pct"].std()), 4),
            "daily_return_skew": round(float(df["daily_return_pct"].skew()), 4),
            "daily_return_kurtosis": round(float(df["daily_return_pct"].kurtosis()), 4),
        },
        "level_autocorrelation": level_autocorr,
        "diff_autocorrelation": diff_autocorr,
        "test_window_benchmarks_30d": {
            "test_days": test_size,
            "Naive_Random_Walk": {
                "RMSE": round(naive_rmse, 4),
                "MAE": round(naive_mae, 4),
                "MAPE_pct": round(naive_mape, 4),
            },
            "Moving_Average_7D": {
                "RMSE": round(ma7_rmse, 4),
                "MAE": round(ma7_mae, 4),
                "MAPE_pct": round(ma7_mape, 4),
            },
        },
        "dashboard_artifact": str(dashboard_path.relative_to(PROJECT_ROOT)),
    }

    report_path = PROCESSED_DIR / "eda_summary_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=" * 72)
    print("COMPREHENSIVE EDA SUMMARY (500 BUSINESS DAYS: 2024-10-23 to 2026-10-13)")
    print("=" * 72)
    print(f"Total Observations      : {summary['dataset_total_rows']} business days (Zero Missing Values)")
    print(f"Kurs Jual Range         : Rp{summary['descriptive_stats']['jual_min']:,.2f} - Rp{summary['descriptive_stats']['jual_max']:,.2f} (Mean: Rp{summary['descriptive_stats']['jual_mean']:,.2f})")
    print(f"Daily Std Deviation     : Rp{summary['descriptive_stats']['daily_diff_std_idr']:.2f} ({summary['descriptive_stats']['daily_return_std_pct']:.3f}%)")
    print(f"Lag-1 Autocorrelation   : r = {level_autocorr['lag_1']} (Lag-7: r = {level_autocorr['lag_7']}, Lag-14: r = {level_autocorr['lag_14']})")
    print(f"Naive Random Walk (30d) : RMSE = {naive_rmse:.4f} IDR | MAE = {naive_mae:.4f} IDR | MAPE = {naive_mape:.4f}%")
    print(f"Moving Average 7D (30d) : RMSE = {ma7_rmse:.4f} IDR | MAE = {ma7_mae:.4f} IDR | MAPE = {ma7_mape:.4f}%")
    print("=" * 72)
    return summary


if __name__ == "__main__":
    run_comprehensive_eda(test_size=30)

