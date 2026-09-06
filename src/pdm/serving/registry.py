"""
MLflow Model Registry helper.

Why a registry module?
- Centralizes all model management logic in one place
- The API (Step 9) calls load_production_model() to serve predictions
- The retraining pipeline (Step 12) calls promote_model() after
  champion/challenger comparison
- Avoids scattering MLflow API calls across the codebase

MLflow model stages:
- "None"       → just logged, not deployed
- "Staging"    → candidate for production, under evaluation
- "Production" → currently serving predictions
- "Archived"   → previous production model, kept for audit trail
"""
import logging

import mlflow
import torch
from mlflow.tracking import MlflowClient

from pdm.config import DEVICE, FEATURE_SENSORS, PROC
from pdm.models.lstm import RULLSTM
from pdm.models.tcn import RULTCN

logger = logging.getLogger(__name__)

EXPERIMENT_NAME = "rul-cmapss"


def get_client() -> MlflowClient:
    """Return an MLflow client with the experiment set."""
    mlflow.set_experiment(EXPERIMENT_NAME)
    return MlflowClient()


def get_all_model_metrics() -> list[dict]:
    """
    Retrieve metrics for all runs in the experiment.
    Returns a list of dicts sorted by RMSE (best first).

    Use this to build comparison tables for the README,
    dashboard, and model card.
    """
    client = get_client()
    exp = client.get_experiment_by_name(EXPERIMENT_NAME)
    if exp is None:
        return []

    runs = client.search_runs(
        exp.experiment_id,
        filter_string="",
        order_by=["metrics.rmse ASC"],
    )

    results = []
    for r in runs:
        # Skip non-model runs (like conformal calibration or retrain decisions)
        if "rmse" not in r.data.metrics:
            continue
        results.append({
            "run_id": r.info.run_id,
            "run_name": r.info.run_name,
            "model": r.data.params.get("model", r.info.run_name),
            "rmse": r.data.metrics.get("rmse"),
            "mae": r.data.metrics.get("mae"),
            "phm_score": r.data.metrics.get("phm_score"),
            "start_time": r.info.start_time,
        })

    return results


def get_registered_models() -> list[dict]:
    """
    List all registered models and their versions/stages.
    Shows what's in Production, Staging, and Archived.
    """
    client = get_client()
    results = []

    for rm in client.search_registered_models():
        for v in client.search_model_versions(f"name='{rm.name}'"):
            results.append({
                "name": rm.name,
                "version": v.version,
                "stage": v.current_stage,
                "run_id": v.run_id,
                "status": v.status,
            })

    return results


def promote_model(model_name: str, version: str = None,
                  stage: str = "Production") -> dict:
    """
    Promote a model version to a stage (Production, Staging, Archived).

    If version is None, promotes the latest version.
    Archives any existing model in the target stage.

    Args:
        model_name: registered model name (e.g., "rul-lstm")
        version: specific version number, or None for latest
        stage: target stage ("Production", "Staging", "Archived")

    Returns:
        dict with promotion details
    """
    client = get_client()

    if version is None:
        versions = client.search_model_versions(f"name='{model_name}'")
        if not versions:
            raise ValueError(f"No versions found for model '{model_name}'")
        version = max(v.version for v in versions)

    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage=stage,
        archive_existing_versions=True,
    )

    logger.info(f"Promoted {model_name} v{version} to {stage}")

    return {
        "model_name": model_name,
        "version": version,
        "stage": stage,
    }


def load_production_model(model_name: str = "rul-lstm"):
    """
    Load the current Production model from the registry.

    This is what the FastAPI server calls on startup.
    Falls back to loading from the saved .pt file if no
    Production model is registered.

    Returns:
        tuple: (model, model_info_dict)
    """
    client = get_client()

    try:
        versions = client.get_latest_versions(model_name, stages=["Production"])
        if versions:
            v = versions[0]
            logger.info(f"Loading {model_name} v{v.version} from registry (Production)")

            # Load the model via MLflow
            model_uri = f"models:/{model_name}/{v.version}"
            model = mlflow.pytorch.load_model(model_uri, map_location=DEVICE)
            model = model.to(DEVICE)
            model.eval()

            return model, {
                "name": model_name,
                "version": v.version,
                "stage": "Production",
                "run_id": v.run_id,
            }
    except Exception as e:
        logger.warning(f"Could not load from registry: {e}")

    # Fallback: load from saved checkpoint
    logger.info("Falling back to saved checkpoint")
    n_features = len(FEATURE_SENSORS)

    if model_name == "rul-tcn":
        model = RULTCN(n_features=n_features)
        state_path = PROC / "tcn_best.pt"
    else:
        model = RULLSTM(n_features=n_features)
        state_path = PROC / "lstm.pt"

    model.load_state_dict(torch.load(state_path, weights_only=True, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()

    return model, {
        "name": model_name,
        "version": "local",
        "stage": "fallback",
        "run_id": None,
    }
