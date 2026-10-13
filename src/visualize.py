"""
Visualisasi Data Kurs USD/IDR Bank Indonesia
Mendukung analisis visual data mentah, data bersih prapemrosesan, dan data historis.
Menghasilkan:
1. Visualisasi Tabel Terminal (ASCII / Unicode) dengan kalkulasi spread & delta harian
2. Tabel Markdown (.md) dan Web Table responsif berstandar modern (.html)
3. Dashboard Grafis berstandar publikasi 300 DPI (.png)
"""

import argparse
import html
import sqlite3
import sys
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROCESSED_CSV = PROJECT_ROOT / "data" / "processed" / "usd_idr_cleaned.csv"
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "exchange_rates.db"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "docs"


def set_visual_style():
    """Mengatur konfigurasi visual grafik modern, bersih, dan kontras terarah."""
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Segoe UI", "DejaVu Sans", "Arial", "Helvetica"]
    plt.rcParams["axes.edgecolor"] = "#334155"
    plt.rcParams["axes.linewidth"] = 1.0
    plt.rcParams["grid.color"] = "#e2e8f0"
    plt.rcParams["grid.linestyle"] = "--"
    plt.rcParams["grid.alpha"] = 0.8


def _prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    """Memastikan kolom tgl berformat datetime serta kolom spread, tengah, dan perubahan ada."""
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df["tgl"]):
        df["tgl"] = pd.to_datetime(df["tgl"])
    df = df.sort_values("tgl").reset_index(drop=True)

    if "spread" not in df.columns:
        df["spread"] = df["jual"] - df["beli"]
    if "tengah" not in df.columns:
        df["tengah"] = (df["beli"] + df["jual"]) / 2.0

    df["perubahan_rp"] = df["tengah"].diff()
    df["pct_change"] = df["tengah"].pct_change() * 100.0
    return df


def load_dataset(source: str = "processed", db_path: Path = DEFAULT_DB_PATH, csv_path: Path = DEFAULT_PROCESSED_CSV) -> pd.DataFrame:
    """Memuat data kurs dari database historis atau file CSV prapemrosesan."""
    if source == "db" and db_path.exists():
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query("SELECT * FROM exchange_rates ORDER BY tgl ASC", conn)
        conn.close()
    elif csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        raise FileNotFoundError(f"Sumber data tidak ditemukan: {csv_path} atau {db_path}")

    return _prepare_df(df)


def format_exchange_rate_table(df: pd.DataFrame, limit: Optional[int] = 14) -> str:
    """
    Menghasilkan representasi tabel teks/terminal visual yang rapi
    lengkap dengan harga beli, tengah, jual, spread, dan dinamika harian.
    """
    df = _prepare_df(df)
    view_df = df.tail(limit).copy() if limit else df.copy()
    view_df["perubahan_rp"] = view_df["tengah"].diff()
    view_df["pct_change"] = view_df["tengah"].pct_change() * 100.0

    lines = []
    divider = "+-----+------------+---------------+---------------+---------------+-----------+-----------------------+----------+"
    header  = "| No  |  Tanggal   |   Kurs Beli   |  Kurs Tengah  |   Kurs Jual   |  Spread   |    Perubahan Harian   |  Status  |"
    
    lines.append(divider)
    lines.append(header)
    lines.append(divider)

    for idx, (_, row) in enumerate(view_df.iterrows(), start=1):
        tgl_str = row["tgl"].strftime("%Y-%m-%d")
        beli_str = f"Rp {row['beli']:>9,.2f}"
        tengah_str = f"Rp {row['tengah']:>9,.2f}"
        jual_str = f"Rp {row['jual']:>9,.2f}"
        spread_str = f"Rp {row['spread']:>6,.2f}"

        chg_rp = row["perubahan_rp"]
        chg_pct = row["pct_change"]

        if pd.isna(chg_rp):
            chg_str = "       -       "
            status_str = " Baseline "
        else:
            sign = "+" if chg_rp > 0 else ""
            chg_str = f"{sign}{chg_rp:>6.2f} ({sign}{chg_pct:>5.2f}%)"
            if chg_rp > 0:
                status_str = " Naik  [+]"
            elif chg_rp < 0:
                status_str = " Turun [-]"
            else:
                status_str = " Tetap [=]"

        line = (
            f"| {idx:>3} | {tgl_str} | {beli_str} | {tengah_str} | {jual_str} | "
            f"{spread_str} | {chg_str:>21} | {status_str} |"
        )
        lines.append(line)

    lines.append(divider)
    return "\n".join(lines)


def format_summary_metrics_table(df: pd.DataFrame) -> str:
    """Menghasilkan tabel ringkasan metrik deskriptif dan statistik data kurs."""
    df = _prepare_df(df)
    tengah = df["tengah"]
    spread = df["spread"]
    pct = df["pct_change"].dropna()

    metrics = [
        ("Total Hari Observasi", f"{len(df)} hari bursa", "Jumlah sampel kalender transaksi aktif"),
        ("Rentang Waktu", f"{df['tgl'].min().strftime('%Y-%m-%d')} s.d. {df['tgl'].max().strftime('%Y-%m-%d')}", "Cakupan observasi data"),
        ("Kurs Terendah (Min)", f"Rp {tengah.min():,.2f}", f"Dicatat pada {df.loc[tengah.idxmin(), 'tgl'].strftime('%Y-%m-%d')}"),
        ("Kurs Tertinggi (Max)", f"Rp {tengah.max():,.2f}", f"Dicatat pada {df.loc[tengah.idxmax(), 'tgl'].strftime('%Y-%m-%d')}"),
        ("Kurs Rata-Rata (Mean)", f"Rp {tengah.mean():,.2f}", "Tendensi sentral nilai kurs tengah"),
        ("Kurs Median", f"Rp {tengah.median():,.2f}", "Nilai tengah sebaran distribusi kurs"),
        ("Kurs Terakhir (Latest)", f"Rp {tengah.iloc[-1]:,.2f}", f"Posisi per {df['tgl'].iloc[-1].strftime('%Y-%m-%d')}"),
        ("Rata-Rata Bid-Ask Spread", f"Rp {spread.mean():,.2f}", "Rerata selisih beli-jual (indikator friksi pasar)"),
        ("Volatilitas Harian (Std Dev)", f"Rp {tengah.std():,.2f}", "Standar deviasi nilai tukar"),
        ("Rerata Perubahan Harian", f"{pct.mean():+.2f}% (Std: {pct.std():.2f}%)", "Dinamika rata-rata return persentase harian"),
    ]

    div = "+-----------------------------+------------------------------------+-------------------------------------------+"
    head = "| Parameter Metrik            | Nilai Observasi                    | Keterangan Analitik                       |"
    out = [div, head, div]
    for label, val, desc in metrics:
        out.append(f"| {label:<27} | {val:<34} | {desc:<41} |")
    out.append(div)
    return "\n".join(out)


def export_markdown_table(df: pd.DataFrame, output_path: Path, limit: Optional[int] = 14) -> Path:
    """Mengekspor visualisasi tabel ke berkas Markdown (.md)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = _prepare_df(df)
    view_df = df.tail(limit).copy() if limit else df.copy()
    view_df["perubahan_rp"] = view_df["tengah"].diff()
    view_df["pct_change"] = view_df["tengah"].pct_change() * 100.0

    lines = [
        f"# Visualisasi Tabel Kurs Valuta Asing USD/IDR Bank Indonesia",
        f"\n**Periode Observasi**: {df['tgl'].min().strftime('%d %B %Y')} s.d. {df['tgl'].max().strftime('%d %B %Y')} | **Total Data**: {len(df)} hari\n",
        "## 1. Tabel Dinamika Kurs Harian",
        "\n| No | Tanggal | Kurs Beli (Bid) | Kurs Tengah (Mid) | Kurs Jual (Ask) | Spread (Rp) | Perubahan Harian (Rp & %) | Status Tren |",
        "|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]

    for idx, (_, row) in enumerate(view_df.iterrows(), start=1):
        tgl_str = row["tgl"].strftime("%Y-%m-%d")
        beli_str = f"Rp {row['beli']:,.2f}"
        tengah_str = f"**Rp {row['tengah']:,.2f}**"
        jual_str = f"Rp {row['jual']:,.2f}"
        spread_str = f"Rp {row['spread']:,.2f}"

        chg_rp = row["perubahan_rp"]
        chg_pct = row["pct_change"]

        if pd.isna(chg_rp):
            chg_str = "-"
            status_badge = "⚪ `Baseline`"
        else:
            sign = "+" if chg_rp > 0 else ""
            chg_str = f"{sign}{chg_rp:,.2f} ({sign}{chg_pct:.2f}%)"
            if chg_rp > 0:
                status_badge = "🟢 `▲ Naik`"
            elif chg_rp < 0:
                status_badge = "🔴 `▼ Turun`"
            else:
                status_badge = "⚪ `■ Tetap`"

        lines.append(f"| {idx} | {tgl_str} | {beli_str} | {tengah_str} | {jual_str} | {spread_str} | {chg_str} | {status_badge} |")

    lines.append("\n## 2. Ringkasan Metrik Statistik Kurs")
    lines.append("\n| Parameter Metrik | Nilai Observasi | Keterangan |")
    lines.append("|:---|:---|:---|")
    tengah = df["tengah"]
    lines.append(f"| **Kurs Tertinggi (Max)** | Rp {tengah.max():,.2f} | Puncak tertinggi periode observasi |")
    lines.append(f"| **Kurs Terendah (Min)** | Rp {tengah.min():,.2f} | Titik terendah periode observasi |")
    lines.append(f"| **Kurs Rata-Rata (Mean)** | Rp {tengah.mean():,.2f} | Nilai tengah mean aritmetika |")
    lines.append(f"| **Kurs Terakhir (Latest)** | Rp {tengah.iloc[-1]:,.2f} | Kurs posisi terkini |")
    lines.append(f"| **Rata-Rata Bid-Ask Spread** | Rp {df['spread'].mean():,.2f} | Stabilitas selisih transaksi jual-beli |")
    lines.append(f"| **Volatilitas Standar Deviasi** | Rp {tengah.std():,.2f} | Dispersi simpangan baku harga |")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def export_html_table(
    df: pd.DataFrame,
    output_path: Path,
    title_suffix: str = "Bank Indonesia (USD/IDR)",
    limit: Optional[int] = 14,
) -> Path:
    """Mengekspor visualisasi tabel ke web HTML modern dengan styling premium."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = _prepare_df(df)
    view_df = df.tail(limit).copy() if limit else df.copy()
    view_df["perubahan_rp"] = view_df["tengah"].diff()
    view_df["pct_change"] = view_df["tengah"].pct_change() * 100.0

    tengah = df["tengah"]
    spread = df["spread"]
    latest_val = tengah.iloc[-1]
    mean_val = tengah.mean()
    spread_mean = spread.mean()
    min_val = tengah.min()
    max_val = tengah.max()

    rows_html = []
    for idx, (_, row) in enumerate(view_df.iterrows(), start=1):
        tgl_str = row["tgl"].strftime("%d %b %Y")
        beli_str = f"Rp {row['beli']:,.2f}"
        tengah_str = f"Rp {row['tengah']:,.2f}"
        jual_str = f"Rp {row['jual']:,.2f}"
        spread_str = f"Rp {row['spread']:,.2f}"

        chg_rp = row["perubahan_rp"]
        chg_pct = row["pct_change"]

        if pd.isna(chg_rp):
            chg_html = '<span style="color:#64748b;">-</span>'
            status_badge = '<span class="badge badge-neutral">Baseline</span>'
        else:
            sign = "+" if chg_rp > 0 else ""
            if chg_rp > 0:
                chg_html = f'<span style="color:#10b981;font-weight:600;">+{chg_rp:,.2f} (+{chg_pct:.2f}%)</span>'
                status_badge = '<span class="badge badge-success">▲ Naik</span>'
            elif chg_rp < 0:
                chg_html = f'<span style="color:#ef4444;font-weight:600;">{chg_rp:,.2f} ({chg_pct:.2f}%)</span>'
                status_badge = '<span class="badge badge-danger">▼ Turun</span>'
            else:
                chg_html = f'<span style="color:#64748b;">0.00 (0.00%)</span>'
                status_badge = '<span class="badge badge-neutral">■ Tetap</span>'

        rows_html.append(f"""
        <tr>
            <td style="text-align:center;color:#64748b;font-weight:500;">{idx}</td>
            <td style="font-weight:600;color:#0f172a;">{tgl_str}</td>
            <td style="text-align:right;color:#047857;font-family:monospace;font-size:0.95rem;">{beli_str}</td>
            <td style="text-align:right;font-weight:700;color:#0284c7;font-family:monospace;font-size:1.0rem;background:#f0f9ff;">{tengah_str}</td>
            <td style="text-align:right;color:#b91c1c;font-family:monospace;font-size:0.95rem;">{jual_str}</td>
            <td style="text-align:right;color:#6d28d9;font-family:monospace;font-size:0.95rem;">{spread_str}</td>
            <td style="text-align:right;font-family:monospace;">{chg_html}</td>
            <td style="text-align:center;">{status_badge}</td>
        </tr>
        """)

    html_content = f"""<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Visualisasi Tabel Kurs {title_suffix}</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg: #f8fafc;
            --surface: #ffffff;
            --text-primary: #0f172a;
            --text-secondary: #475569;
            --border: #e2e8f0;
            --primary: #0284c7;
            --primary-light: #e0f2fe;
            --success: #10b981;
            --danger: #ef4444;
            --shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.05);
            --shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
            --shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1);
        }}
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
            background-color: var(--bg);
            color: var(--text-primary);
            padding: 32px 24px;
            line-height: 1.5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        .header {{
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            color: white;
            padding: 32px;
            border-radius: 16px;
            margin-bottom: 24px;
            box-shadow: var(--shadow-lg);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }}
        .header h1 {{
            font-size: 1.85rem;
            font-weight: 800;
            letter-spacing: -0.02em;
            margin-bottom: 8px;
        }}
        .header p {{
            color: #94a3b8;
            font-size: 0.95rem;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }}
        .metric-card {{
            background: var(--surface);
            padding: 20px;
            border-radius: 12px;
            border: 1px solid var(--border);
            box-shadow: var(--shadow-sm);
        }}
        .metric-label {{
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-secondary);
            font-weight: 600;
            margin-bottom: 6px;
        }}
        .metric-value {{
            font-size: 1.5rem;
            font-weight: 800;
            color: var(--text-primary);
            font-family: 'JetBrains Mono', monospace;
        }}
        .metric-desc {{
            font-size: 0.8rem;
            color: #64748b;
            margin-top: 4px;
        }}
        .table-card {{
            background: var(--surface);
            border-radius: 16px;
            border: 1px solid var(--border);
            box-shadow: var(--shadow-md);
            overflow: hidden;
        }}
        .table-header-title {{
            padding: 20px 24px;
            border-bottom: 1px solid var(--border);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .table-header-title h2 {{
            font-size: 1.2rem;
            font-weight: 700;
            color: var(--text-primary);
        }}
        .table-responsive {{
            overflow-x: auto;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.92rem;
        }}
        th {{
            background: #f1f5f9;
            color: #334155;
            font-weight: 700;
            padding: 14px 16px;
            text-align: left;
            border-bottom: 2px solid var(--border);
            white-space: nowrap;
        }}
        td {{
            padding: 13px 16px;
            border-bottom: 1px solid var(--border);
            white-space: nowrap;
        }}
        tr:hover {{
            background-color: #f8fafc;
        }}
        .badge {{
            display: inline-block;
            padding: 4px 10px;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 700;
            letter-spacing: 0.02em;
        }}
        .badge-success {{
            background-color: #d1fae5;
            color: #065f46;
        }}
        .badge-danger {{
            background-color: #fee2e2;
            color: #991b1b;
        }}
        .badge-neutral {{
            background-color: #f1f5f9;
            color: #475569;
        }}
        .footer {{
            margin-top: 24px;
            text-align: center;
            font-size: 0.85rem;
            color: #64748b;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header class="header">
            <h1>Visualisasi Tabel Data Kurs Valuta Asing</h1>
            <p>Sumber: {title_suffix} | Pipeline: Automated Data Ingestion & Preprocessing MLOps</p>
        </header>

        <section class="metrics-grid">
            <div class="metric-card">
                <div class="metric-label">Kurs Terkini (Mid)</div>
                <div class="metric-value" style="color:#0284c7;">Rp {latest_val:,.0f}</div>
                <div class="metric-desc">Posisi data per {df['tgl'].iloc[-1].strftime('%d %B %Y')}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Rata-Rata Kurs</div>
                <div class="metric-value">Rp {mean_val:,.0f}</div>
                <div class="metric-desc">Mean selama rentang {len(df)} hari aktif</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Rata-Rata Spread</div>
                <div class="metric-value" style="color:#8b5cf6;">Rp {spread_mean:,.2f}</div>
                <div class="metric-desc">Selisih Kurs Jual vs Kurs Beli</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Rentang Nilai (Min - Max)</div>
                <div class="metric-value" style="font-size:1.15rem;padding-top:4px;">
                    {min_val:,.0f} - {max_val:,.0f}
                </div>
                <div class="metric-desc">IDR per 1 USD (Volatilitas: Rp {tengah.std():,.0f})</div>
            </div>
        </section>

        <main class="table-card">
            <div class="table-header-title">
                <h2>Rincian Observasi Harian Kurs USD / IDR ({len(view_df)} Titik Terakhir)</h2>
                <span style="font-size:0.85rem;color:#64748b;background:#f1f5f9;padding:6px 12px;border-radius:8px;font-weight:600;">
                    Status: Valid & Bersih
                </span>
            </div>
            <div class="table-responsive">
                <table>
                    <thead>
                        <tr>
                            <th style="text-align:center;">No</th>
                            <th>Tanggal</th>
                            <th style="text-align:right;">Kurs Beli (Bid)</th>
                            <th style="text-align:right;">Kurs Tengah (Mid)</th>
                            <th style="text-align:right;">Kurs Jual (Ask)</th>
                            <th style="text-align:right;">Bid-Ask Spread</th>
                            <th style="text-align:right;">Perubahan Harian (Rp & %)</th>
                            <th style="text-align:center;">Status Tren</th>
                        </tr>
                    </thead>
                    <tbody>
                        {"".join(rows_html)}
                    </tbody>
                </table>
            </div>
        </main>

        <footer class="footer">
            <p>MLOps Kurs USD/IDR &copy; 2026. Data diverifikasi otomatis dari Bank Indonesia.</p>
        </footer>
    </div>
</body>
</html>
"""
    output_path.write_text(html_content, encoding="utf-8")
    return output_path


def generate_exchange_rate_dashboard(
    df: pd.DataFrame,
    output_path: Path,
    title_suffix: str = "Bank Indonesia (USD/IDR)",
) -> Path:
    """
    Membuat 4-panel dashboard analitik kurs USD/IDR beresolusi 300 DPI:
    1. Tren Runtun Waktu Kurs Beli, Jual, Tengah & Pita Spread
    2. Dinamika Bid-Ask Spread (Likuiditas & Friksi Transaksi)
    3. Fluktuasi Return Harian (%)
    4. Distribusi & Densitas Kurs Tengah
    """
    set_visual_style()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = df.copy()
    df["tgl"] = pd.to_datetime(df["tgl"])
    df = df.sort_values("tgl").reset_index(drop=True)
    if "spread" not in df.columns:
        df["spread"] = df["jual"] - df["beli"]
    if "tengah" not in df.columns:
        df["tengah"] = (df["beli"] + df["jual"]) / 2.0
    if "pct_change" not in df.columns:
        df["pct_change"] = df["tengah"].pct_change() * 100.0

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), dpi=300)
    fig.patch.set_facecolor("#ffffff")
    fig.suptitle(
        f"Dashboard Analisis Data Kurs Valuta Asing: {title_suffix}\n"
        f"Periode Observasi: {df['tgl'].min().strftime('%d %b %Y')} s.d. {df['tgl'].max().strftime('%d %b %Y')} ({len(df)} Titik Observasi)",
        fontsize=15,
        fontweight="bold",
        color="#0f172a",
        y=0.98,
    )

    # Panel 1: Time Series Kurs Beli, Tengah, Jual + Spread Band
    ax1 = axes[0, 0]
    ax1.fill_between(
        df["tgl"],
        df["beli"],
        df["jual"],
        color="#38bdf8",
        alpha=0.25,
        label="Bid-Ask Spread Band",
    )
    ax1.plot(df["tgl"], df["jual"], label="Kurs Jual (Ask)", color="#ef4444", linewidth=1.8, linestyle="--", alpha=0.85)
    ax1.plot(df["tgl"], df["tengah"], label="Kurs Tengah (Mid)", color="#0284c7", linewidth=2.4)
    ax1.plot(df["tgl"], df["beli"], label="Kurs Beli (Bid)", color="#10b981", linewidth=1.8, linestyle="--", alpha=0.85)

    latest_row = df.iloc[-1]
    ax1.scatter([latest_row["tgl"]], [latest_row["tengah"]], color="#0f172a", s=60, zorder=5)
    ax1.annotate(
        f"Terbaru: Rp {latest_row['tengah']:,.0f}",
        xy=(latest_row["tgl"], latest_row["tengah"]),
        xytext=(-80, 15),
        textcoords="offset points",
        bbox=dict(boxstyle="round,pad=0.4", fc="#f8fafc", ec="#0284c7", lw=1.2),
        fontsize=9,
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0", color="#0284c7"),
    )

    ax1.set_title("1. Dinamika Runtun Waktu Kurs (Beli, Tengah, Jual)", fontsize=12, fontweight="bold", color="#1e293b", pad=8)
    ax1.set_ylabel("Nilai Tukar (IDR)", fontsize=10, fontweight="bold", color="#334155")
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: f"Rp {int(x):,}"))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b"))
    ax1.grid(True)
    ax1.legend(loc="upper left", frameon=True, facecolor="#ffffff", framealpha=0.9, fontsize=8.5)

    # Panel 2: Bid-Ask Spread Dynamics
    ax2 = axes[0, 1]
    spread_mean = df["spread"].mean()
    ax2.plot(df["tgl"], df["spread"], color="#8b5cf6", linewidth=2.0, marker="o", markersize=4, label="Spread Aktual")
    ax2.axhline(spread_mean, color="#f59e0b", linestyle=":", linewidth=1.8, label=f"Rata-rata: Rp {spread_mean:.1f}")

    ax2.set_title("2. Stabilitas Bid-Ask Spread (Spread = Jual - Beli)", fontsize=12, fontweight="bold", color="#1e293b", pad=8)
    ax2.set_ylabel("Selisih Spread (IDR)", fontsize=10, fontweight="bold", color="#334155")
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: f"Rp {int(x):,}"))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b"))
    ax2.grid(True)
    ax2.legend(loc="upper right", frameon=True, facecolor="#ffffff", framealpha=0.9, fontsize=8.5)

    # Panel 3: Daily % Return
    ax3 = axes[1, 0]
    valid_changes = df.dropna(subset=["pct_change"])
    colors = ["#10b981" if val >= 0 else "#ef4444" for val in valid_changes["pct_change"]]
    ax3.bar(valid_changes["tgl"], valid_changes["pct_change"], color=colors, width=0.7, edgecolor="#1e293b", linewidth=0.5)
    ax3.axhline(0, color="#64748b", linestyle="-", linewidth=1.0)

    ax3.set_title("3. Volatilitas & Perubahan Harian (%) Kurs Tengah", fontsize=12, fontweight="bold", color="#1e293b", pad=8)
    ax3.set_ylabel("Perubahan (%)", fontsize=10, fontweight="bold", color="#334155")
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b"))
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: f"{x:+.2f}%"))
    ax3.grid(True)

    # Panel 4: Distribution & Density
    ax4 = axes[1, 1]
    sns.histplot(
        df["tengah"],
        kde=True,
        ax=ax4,
        color="#0284c7",
        bins=min(15, max(5, len(df) // 2)),
        stat="density",
        edgecolor="#0f172a",
        linewidth=0.8,
        alpha=0.55,
    )
    mean_val = df["tengah"].mean()
    median_val = df["tengah"].median()
    ax4.axvline(mean_val, color="#ef4444", linestyle="--", linewidth=1.8, label=f"Mean: Rp {mean_val:,.0f}")
    ax4.axvline(median_val, color="#10b981", linestyle="-.", linewidth=1.8, label=f"Median: Rp {median_val:,.0f}")

    ax4.set_title("4. Estimasi Densitas & Distribusi Probabilitas Kurs Tengah", fontsize=12, fontweight="bold", color="#1e293b", pad=8)
    ax4.set_xlabel("Nilai Tukar Tengah (IDR)", fontsize=10, fontweight="bold", color="#334155")
    ax4.set_ylabel("Densitas", fontsize=10, fontweight="bold", color="#334155")
    ax4.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: f"Rp {int(x):,}"))
    ax4.grid(True)
    ax4.legend(loc="upper right", frameon=True, facecolor="#ffffff", framealpha=0.9, fontsize=8.5)

    plt.tight_layout(rect=[0, 0.03, 1, 0.94])
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Visualisasi Data Kurs USD/IDR Bank Indonesia")
    parser.add_argument(
        "--source",
        choices=["processed", "db"],
        default="processed",
        help="Sumber data: 'processed' (data cleaned CSV terbaru) atau 'db' (database historis SQLite)",
    )
    parser.add_argument(
        "--mode",
        choices=["table", "plot", "all"],
        default="all",
        help="Mode output visualisasi: 'table' (tabel terminal, markdown & HTML), 'plot' (grafik PNG 300 DPI), atau 'all' (keduanya)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=14,
        help="Jumlah hari observasi yang ditampilkan pada tabel (default: 14 hari terakhir)",
    )
    parser.add_argument(
        "--output-plot",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR / "visualisasi_kurs_usd_idr.png"),
        help="Path file output PNG grafik",
    )
    parser.add_argument(
        "--output-html",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR / "tabel_kurs_usd_idr.html"),
        help="Path file output tabel HTML",
    )
    parser.add_argument(
        "--output-md",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR / "tabel_kurs_usd_idr.md"),
        help="Path file output tabel Markdown",
    )
    args = parser.parse_args()

    df = load_dataset(source=args.source)
    source_label = "Hasil Prapemrosesan Terbaru" if args.source == "processed" else "Arsip Historis SQLite"

    print("\n" + "=" * 80)
    print(f" VISUALISASI DATA KURS USD/IDR BANK INDONESIA ({source_label})")
    print(f" Rentang Data: {df['tgl'].min().strftime('%Y-%m-%d')} s.d. {df['tgl'].max().strftime('%Y-%m-%d')} | Total: {len(df)} Titik")
    print("=" * 80 + "\n")

    if args.mode in ("table", "all"):
        print(">> [TABEL 1] OBSERVASI HARIAN KURS USD/IDR & PERGERAKAN HARIAN:")
        print(format_exchange_rate_table(df, limit=args.limit))
        print("\n>> [TABEL 2] RINGKASAN METRIK DESKRIPTIF & VOLATILITAS:")
        print(format_summary_metrics_table(df))
        print()

        # Ekspor berkas tabel
        md_path = export_markdown_table(df, Path(args.output_md), limit=args.limit)
        html_path = export_html_table(df, Path(args.output_html), title_suffix=f"Bank Indonesia ({source_label})", limit=args.limit)
        print(f"[OK] Tabel Markdown disimpan di : {md_path.resolve()}")
        print(f"[OK] Tabel HTML Interaktif di   : {html_path.resolve()}")

    if args.mode in ("plot", "all"):
        out_plot_path = Path(args.output_plot)
        generated_plot = generate_exchange_rate_dashboard(df, out_plot_path, title_suffix=f"Bank Indonesia ({source_label})")
        print(f"[OK] Dashboard PNG 300 DPI di   : {generated_plot.resolve()}")

    print("\n[SELESAI] Visualisasi data kurs berhasil dieksekusi dengan sempurna.\n")


if __name__ == "__main__":
    main()
