"""
Automated MLflow Model Registry Staging Registration Script (LK-08 Stage 4).

Reads evaluation_gate_report.json produced by Stage 3. If validation passed,
registers the winning MLflow run into the Model Registry and transitions the
resulting version to stage 'Staging' with alias '@staging' and CI provenance tags.
"""

import argparse
import datetime
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict

import mlflow
from mlflow.tracking import MlflowClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("register_staging")
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"

DEFAULT_REPORT_PATH = PROJECT_ROOT / "models" / "evaluation_gate_report.json"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "models" / "staging_registration_report.json"
DEFAULT_MODEL_NAME = "USD_IDR_Exchange_Rate_Regressor"


def register_validated_model_to_staging(
    tracking_uri: str = "sqlite:///mlflow.db",
    model_name: str = DEFAULT_MODEL_NAME,
    report_path: Path = DEFAULT_REPORT_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> Dict[str, Any]:
    """Register a gate-validated MLflow run and promote its version to Staging."""
    report_path = Path(report_path)
    output_path = Path(output_path)

    if not report_path.exists():
        raise FileNotFoundError(f"Evaluation report not found: {report_path}")

    with open(report_path, "r", encoding="utf-8") as f:
        eval_report = json.load(f)

    if eval_report.get("gate_status") != "PASSED":
        raise RuntimeError(
            f"Cannot register model to Staging: gate_status is {eval_report.get('gate_status')!r}."
        )

    candidate = eval_report["candidate"]
    run_id = candidate["run_id"]
    run_name = candidate["run_name"]
    dvc_hash = candidate.get("dvc_hash", "unknown_dvc_hash")
    metrics = candidate.get("metrics", {})
    params = candidate.get("params", {})

    if not run_id:
        raise ValueError("Candidate run_id missing from evaluation gate report.")

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    model_uri = f"runs:/{run_id}/model"
    logger.info("Registering %s to MLflow Model Registry '%s'...", model_uri, model_name)
    mv = mlflow.register_model(model_uri=model_uri, name=model_name)
    version_str = str(mv.version)

    ci_event = os.getenv("GITHUB_EVENT_NAME", "local")
    git_sha = os.getenv("GITHUB_SHA", "local")

    client.update_model_version(
        name=model_name,
        version=version_str,
        description=(
            f"CI-Automated Staging Candidate ({run_name}): "
            f"MAE={metrics.get('MAE')} IDR, RMSE={metrics.get('RMSE')} IDR, "
            f"MAPE={metrics.get('MAPE_pct')}%, DirAcc={metrics.get('Directional_Accuracy_pct')}%. "
            f"Params: n_est={params.get('n_estimators')}, lr={params.get('learning_rate')}, "
            f"depth={params.get('max_depth')}, subsample={params.get('subsample')}. "
            f"Commit={git_sha[:8]}."
        ),
    )

    logger.info("Transitioning %s Version %s -> Staging...", model_name, version_str)
    client.transition_model_version_stage(
        name=model_name,
        version=version_str,
        stage="Staging",
        archive_existing_versions=False,
    )
    client.set_registered_model_alias(model_name, "staging", int(mv.version))

    tags = {
        "validation_gate": "PASSED_LK01_THRESHOLDS",
        "ci_event": ci_event,
        "git_sha": git_sha,
        "dvc_data_hash": dvc_hash,
        "mae_idr": str(metrics.get("MAE", "")),
        "rmse_idr": str(metrics.get("RMSE", "")),
        "mape_pct": str(metrics.get("MAPE_pct", "")),
        "directional_accuracy_pct": str(metrics.get("Directional_Accuracy_pct", "")),
    }
    for k, v in tags.items():
        client.set_model_version_tag(model_name, version_str, k, v)

    verified_mv = client.get_model_version(model_name, version_str)
    result = {
        "status": "REGISTERED_STAGING",
        "model_name": model_name,
        "version": int(verified_mv.version),
        "stage": verified_mv.current_stage,
        "alias": "staging",
        "run_id": run_id,
        "run_name": run_name,
        "model_uri": model_uri,
        "registry_stage_uri": f"models:/{model_name}/Staging",
        "registry_alias_uri": f"models:/{model_name}@staging",
        "ci_event": ci_event,
        "git_sha": git_sha,
        "dvc_data_hash": dvc_hash,
        "metrics": metrics,
        "registered_at_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("=" * 78)
    print("STAGE 4: AUTO-REGISTRY UPDATE TO STAGING COMPLETED")
    print("=" * 78)
    print(f"Registered Model : {model_name}")
    print(f"Version          : {verified_mv.version}")
    print(f"Current Stage    : {verified_mv.current_stage}")
    print(f"Assigned Alias   : @staging")
    print(f"Source Run ID    : {run_id}")
    print(f"CI Event / SHA   : {ci_event} / {git_sha}")
    print(f"Report saved to  : {output_path}")
    print("=" * 78)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Register validated model to MLflow Model Registry Staging.")
    parser.add_argument("--tracking-uri", type=str, default="sqlite:///mlflow.db")
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--report-path", type=str, default=str(DEFAULT_REPORT_PATH))
    parser.add_argument("--output-path", type=str, default=str(DEFAULT_OUTPUT_PATH))
    args = parser.parse_args()

    try:
        register_validated_model_to_staging(
            tracking_uri=args.tracking_uri,
            model_name=args.model_name,
            report_path=Path(args.report_path),
            output_path=Path(args.output_path),
        )
    except Exception as exc:
        logger.error("Staging registration aborted: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
