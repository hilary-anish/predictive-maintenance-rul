"""
Automated retraining pipeline.

Triggered when Evidently detects data drift or model performance degradation.
Retrains the model, compares against the current champion, and promotes
only if the challenger is statistically better.

This is the "continuous learning" story that DNV, Sonova, and SafeAD want.

Full flow:
    1. Check for drift (Evidently)
    2. If drift detected (or force=True), retrain the LSTM
    3. Compare challenger vs champion metrics
    4. Promote only if challenger beats champion by >= RMSE_IMPROVEMENT_MIN
    5. Log everything for audit trail

Why champion/challenger instead of just deploying the new model?
    - Retraining on drifted data might produce a WORSE model
    - The improvement threshold prevents noise-level "improvements"
    - The audit log shows every decision for governance/compliance
    - This is how production ML teams actually work
"""
import json
import logging
from datetime import datetime

import mlflow
import numpy as np
from mlflow.tracking import MlflowClient

from pdm.config import PROC
from pdm.models.train_lstm import main as train_lstm
from pdm.monitoring.drift import run_drift_check, simulate_drift

logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────
# Challenger must beat champion by at least this much RMSE.
# Why 0.5? Smaller improvements could be noise. This threshold
# prevents unnecessary model swaps that add deployment risk
# without meaningful performance gain.
RMSE_IMPROVEMENT_MIN = 0.5

EXPERIMENT_NAME = "rul-cmapss"


def get_champion_metrics() -> dict | None:
    """
    Retrieve the current production model's metrics from MLflow.

    Returns None if no model is in Production stage.
    This is the "champion" — the model currently serving predictions.
    """
    client = MlflowClient()
    try:
        latest = client.get_latest_versions("rul-lstm", stages=["Production"])
        if not latest:
            logger.info("No Production model found in registry")
            return None

        run = client.get_run(latest[0].run_id)
        return {
            "run_id": run.info.run_id,
            "rmse": run.data.metrics.get("rmse"),
            "mae": run.data.metrics.get("mae"),
            "phm_score": run.data.metrics.get("phm_score"),
            "model_version": latest[0].version,
        }
    except Exception as e:
        logger.warning(f"Could not retrieve champion metrics: {e}")
        return None


def get_challenger_metrics() -> dict:
    """
    Get metrics from the most recent training run.

    After retraining, this retrieves the challenger's metrics
    for comparison against the champion.
    """
    client = MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    runs = client.search_runs(
        experiment.experiment_id,
        filter_string="",  # only LSTM runs
        order_by=["start_time DESC"],
        max_results=1,
    )

    if not runs:
        raise RuntimeError("No training runs found after retraining")

    run = runs[0]
    return {
        "run_id": run.info.run_id,
        "rmse": run.data.metrics.get("rmse"),
        "mae": run.data.metrics.get("mae"),
        "phm_score": run.data.metrics.get("phm_score"),
    }


def compare_and_promote(champion: dict | None, challenger: dict) -> dict:
    """
    Compare challenger against champion and promote if better.

    Decision logic:
    - No champion exists → promote (first model)
    - Challenger RMSE < champion RMSE by >= RMSE_IMPROVEMENT_MIN → promote
    - Otherwise → keep champion

    Returns a decision dict with full reasoning for the audit trail.
    """
    client = MlflowClient()

    if champion is None:
        decision = "promote"
        reason = "No existing champion. Promoting first model."
    else:
        rmse_improvement = champion["rmse"] - challenger["rmse"]

        if rmse_improvement >= RMSE_IMPROVEMENT_MIN:
            decision = "promote"
            reason = (
                f"Challenger RMSE {challenger['rmse']:.2f} beats "
                f"champion {champion['rmse']:.2f} by {rmse_improvement:.2f} "
                f"(threshold: {RMSE_IMPROVEMENT_MIN})"
            )
        else:
            decision = "keep_champion"
            reason = (
                f"Challenger RMSE {challenger['rmse']:.2f} does not beat "
                f"champion {champion['rmse']:.2f} by required "
                f"{RMSE_IMPROVEMENT_MIN} (improvement: {rmse_improvement:.2f})"
            )

    # Promote if warranted
    if decision == "promote":
        try:
            latest_versions = client.get_latest_versions("rul-lstm")
            if latest_versions:
                version = max(v.version for v in latest_versions)
                client.transition_model_version_stage(
                    name="rul-lstm",
                    version=version,
                    stage="Production",
                    archive_existing_versions=True,
                )
                logger.info(f"Promoted rul-lstm v{version} to Production")
        except Exception as e:
            logger.error(f"Failed to promote model: {e}")
            decision = "promote_failed"
            reason += f" (promotion error: {e})"

    # Log the decision to MLflow for audit
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name="retrain-decision"):
        mlflow.log_params({"decision": decision, "reason": reason[:250]})
        if champion:
            mlflow.log_metrics({
                "champion_rmse": champion["rmse"],
                "challenger_rmse": challenger["rmse"],
            })

    result = {
        "timestamp": datetime.utcnow().isoformat(),
        "champion": champion,
        "challenger": challenger,
        "decision": decision,
        "reason": reason,
    }

    logger.info(f"Retrain decision: {decision} — {reason}")
    return result


def run_pipeline(recent_data_path: str = None, force: bool = False) -> dict:
    """
    Full automated retraining pipeline:
        1. Check for drift
        2. If drift detected (or force=True), retrain
        3. Compare with champion
        4. Promote or keep

    This is what the Kubernetes CronJob calls nightly.

    Args:
        recent_data_path: path to recent production data (.npz)
                          If None, uses test data as current data.
        force: skip drift check and retrain regardless

    Returns:
        dict with full pipeline result for audit logging
    """
    result = {"stage": "drift_check", "timestamp": datetime.utcnow().isoformat()}

    # ── Step 1: Check for drift ────────────────────────────────────
    if not force:
        logger.info("Running drift check...")

        d = np.load(PROC / "FD001.npz")
        reference_data = d["Xtr"]

        if recent_data_path:
            current = np.load(recent_data_path)
            current_data = current["Xte"] if "Xte" in current else current["Xtr"]
        else:
            current_data = d["Xte"]

        drift = run_drift_check(
            reference_data=reference_data,
            current_data=current_data,
            save_html=True,
        )
        result["drift"] = {
            "dataset_drift": drift["dataset_drift"],
            "n_drifted_features": drift["n_drifted_features"],
            "total_features": drift["total_features"],
            "drift_ratio": drift["drift_ratio"],
        }

        if not drift["dataset_drift"]:
            result["action"] = "no_retrain"
            result["reason"] = "No drift detected"
            logger.info("No drift detected. Skipping retraining.")

            # Still log to audit trail
            _save_audit_log(result)
            return result

    # ── Step 2: Get champion metrics ───────────────────────────────
    result["stage"] = "retrain"
    logger.info("Drift detected (or forced). Starting retraining...")

    champion = get_champion_metrics()
    if champion:
        logger.info(f"Current champion RMSE: {champion['rmse']:.2f}")
    else:
        logger.info("No champion found. Training initial model.")

    # ── Step 3: Retrain ────────────────────────────────────────────
    logger.info("Retraining LSTM...")
    train_lstm(subset="FD001")

    # ── Step 4: Compare and promote ────────────────────────────────
    challenger = get_challenger_metrics()
    logger.info(f"Challenger RMSE: {challenger['rmse']:.2f}")

    decision = compare_and_promote(champion, challenger)
    result.update(decision)

    # ── Step 5: Save audit log ─────────────────────────────────────
    _save_audit_log(result)

    return result


def _save_audit_log(result: dict):
    """
    Append result to JSONL audit log.

    Why JSONL (one JSON object per line)?
    - Easy to append (just add a line)
    - Easy to parse (read line by line)
    - Easy to load into pandas for analysis
    - Standard format for log streams

    This audit trail is what governance teams want to see:
    every retraining decision with full reasoning.
    """
    log_path = PROC / "retrain_log.jsonl"
    with open(log_path, "a") as f:
        f.write(json.dumps(result, default=str) + "\n")
    logger.info(f"Audit log saved to {log_path}")


if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    force = "--force" in sys.argv
    simulate = "--simulate-drift" in sys.argv

    if simulate:
        # Create simulated drifted data for testing
        d = np.load(PROC / "FD001.npz")
        drifted = simulate_drift(d["Xte"], drift_fraction=0.5, shift_magnitude=3.0)
        drift_path = PROC / "simulated_drift.npz"
        np.savez_compressed(drift_path, Xte=drifted)
        result = run_pipeline(recent_data_path=str(drift_path))
    else:
        result = run_pipeline(force=force)

    print("\n" + "=" * 60)
    print("PIPELINE RESULT")
    print("=" * 60)
    print(json.dumps(result, indent=2, default=str))
