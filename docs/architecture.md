# Arsitektur Teknis MLOps — Prediksi Kurs USD/IDR

Dokumen ini mendokumentasikan fondasi teknis, topologi pipeline, dan arsitektur reproducibility untuk sistem prediksi nilai tukar mata uang asing (USD/IDR) berbasis Bank Indonesia Web Service.

```mermaid
graph TD
    subgraph Data Layer
        BI_API[Bank Indonesia Web Service API] --> Ingestion[src/ingestion/fetcher.py]
        Ingestion --> SQLite[(data/exchange_rates.db)]
        SQLite --> RawCSV[data/raw/usd_idr_raw.csv]
    end

    subgraph Feature & Experiment Layer
        RawCSV --> Features[src/features/pipeline.py]
        Features --> ProcCSV[data/processed/usd_idr_featured.csv]
        ProcCSV --> Notebooks[notebooks/01_initial_eda.ipynb]
        ProcCSV --> Trainer[src/models/trainer.py]
    end

    subgraph Model & Registry Layer
        Trainer --> Hurdle{Kalahkan Naive Random Walk?}
        Hurdle -- Yes --> Champion[models/model_champion.pkl]
        Hurdle -- No --> LogReport[models/model_metadata.json]
    end

    subgraph Serving & Reproducibility Layer
        Champion --> API[src/api/app.py - FastAPI]
        Codespaces[.devcontainer/devcontainer.json] --> Container[Docker Python 3.11 Environment]
        Container --> API
    end
```

## 1. Prinsip Reproducibility
- **Lingkungan Standar**: Menggunakan GitHub Codespaces (`.devcontainer/devcontainer.json`) dengan base image Python 3.11 dan instalasi otomatis seluruh dependensi `requirements.txt`.
- **Zero Lookahead Bias**: Seluruh feature engineering (lag dan rolling window) dikomputasi strictly terhadap observasi masa lalu ($t-1$), mencegah terjadinya data leakage pada time-series cross-validation.
- **Konvensi Cookiecutter Data Science**: Pemisahan jelas antara konfigurasi (`config/`), data (`data/`), model terlatih (`models/`), notebook eksperimen (`notebooks/`), kode modular produksi (`src/`), dan pengujian otomatis (`tests/`).
