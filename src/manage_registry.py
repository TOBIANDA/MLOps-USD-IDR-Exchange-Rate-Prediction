"""
Model Registry Lifecycle Management & Stage Transition Script for LK-07.
Implements industrial lifecycle transitions:
- Version 1: None -> Staging -> Archived
- Version 2: None -> Staging -> Production (Champion)
Sets aliases (@production, @champion) and metadata tags.
"""

import datetime
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict

import mlflow
from mlflow.tracking import MlflowClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("model_registry_lifecycle")
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"

TRACKING_URI = "sqlite:///mlflow.db"
MODEL_NAME = "USD_IDR_Exchange_Rate_Regressor"


def manage_stage_transitions() -> Dict[str, Any]:
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient(tracking_uri=TRACKING_URI)

    # 1. Update Registered Model Description
    logger.info("Updating Registered Model overarching description...")
    client.update_registered_model(
        name=MODEL_NAME,
        description=(
            "Model Registry Resmi Sistem Peramalan Kurs USD/IDR (Bank Indonesia Foreign Exchange Rate Forecasting). "
            "Mengelola siklus hidup model regresi time-series dari tahap eksperimentasi, staging validasi, "
            "hingga deployment produksi berkelanjutan sesuai tata kelola MLOps."
        ),
    )
    client.set_registered_model_tag(MODEL_NAME, "framework", "scikit-learn")
    client.set_registered_model_tag(MODEL_NAME, "domain", "financial-forecasting")
    client.set_registered_model_tag(MODEL_NAME, "institution", "FILKOM Universitas Brawijaya")

    # 2. Simulate Version 1 Lifecycle
    # Version 1 was initially in Staging, then archived when Version 2 proved superior
    logger.info("Transitioning Version 1: Staging -> Archived...")
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version="1",
        stage="Staging",
        archive_existing_versions=False,
    )
    client.update_model_version(
        name=MODEL_NAME,
        version="1",
        description="Version 1: Baseline GradientBoosting (depth=3, n_est=100, lr=0.05). RMSE: 27.34 IDR. Status: Superseded by v2.",
    )
    client.set_model_version_tag(MODEL_NAME, "1", "lifecycle_phase", "Archived_Superseded")
    client.set_model_version_tag(MODEL_NAME, "1", "benchmark_rmse", "27.34")

    # 3. Simulate Version 2 Lifecycle
    # Step A: Promote Version 2 to Staging for validation gate
    logger.info("Step A: Transitioning Version 2: None -> Staging...")
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version="2",
        stage="Staging",
        archive_existing_versions=False,
    )
    client.set_model_version_tag(MODEL_NAME, "2", "staging_validation_date", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    client.set_model_version_tag(MODEL_NAME, "2", "staging_status", "VALIDATED_TEST_PASS")

    # Step B: Promote Version 2 to Production and automatically archive existing versions (v1)
    logger.info("Step B: Promoting Version 2: Staging -> Production (with archive_existing_versions=True)...")
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version="2",
        stage="Production",
        archive_existing_versions=True,
    )

    # Set Model Aliases (MLflow modern standard)
    logger.info("Setting Model Aliases: @production and @champion on Version 2...")
    client.set_registered_model_alias(MODEL_NAME, "production", 2)
    client.set_registered_model_alias(MODEL_NAME, "champion", 2)
    client.set_registered_model_alias(MODEL_NAME, "archived", 1)

    # Set tags on Version 2 Production
    client.set_model_version_tag(MODEL_NAME, "2", "deployment_status", "ACTIVE_PRODUCTION")
    client.set_model_version_tag(MODEL_NAME, "2", "promoted_by", "Tobias Andra Valentino")
    client.set_model_version_tag(MODEL_NAME, "2", "promoted_at", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    client.set_model_version_tag(MODEL_NAME, "2", "performance_rmse", "14.90_IDR")
    client.set_model_version_tag(MODEL_NAME, "2", "performance_mae", "13.80_IDR")
    client.set_model_version_tag(MODEL_NAME, "2", "directional_accuracy", "60.0%")
    client.set_model_version_tag(MODEL_NAME, "2", "inference_sla_ms", "<10ms")

    # 4. Readback and Print Registry Status
    model_details = client.get_registered_model(MODEL_NAME)
    v1_details = client.get_model_version(MODEL_NAME, "1")
    v2_details = client.get_model_version(MODEL_NAME, "2")

    summary = {
        "registered_model": model_details.name,
        "aliases": model_details.aliases,
        "version_1": {
            "version": v1_details.version,
            "stage": v1_details.current_stage,
            "run_id": v1_details.run_id,
            "status": v1_details.status,
        },
        "version_2": {
            "version": v2_details.version,
            "stage": v2_details.current_stage,
            "run_id": v2_details.run_id,
            "status": v2_details.status,
            "aliases": ["production", "champion"],
        },
    }
    logger.info("Registry lifecycle transitions completed successfully!")
    return summary


if __name__ == "__main__":
    result = manage_stage_transitions()
    print("\n--- MODEL REGISTRY STATUS ---")
    print(json.dumps(result, indent=2))
