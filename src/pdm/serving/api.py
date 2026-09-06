"""
FastAPI prediction server for RUL estimation.

Why FastAPI for model serving?
- Auto-generates interactive docs at /docs (recruiters love this)
- Pydantic validates input before it hits the model (no silent garbage-in)
- Async-capable for high concurrency (though we use sync for model inference)
- Startup/shutdown lifecycle hooks for clean model loading

Architecture:
    Client → HTTP POST/GET → FastAPI → loads model from registry → prediction → JSON response

The model is loaded ONCE at startup (not per-request).
This is critical for performance — loading a PyTorch model takes seconds,
but inference takes milliseconds.
"""
import logging

import joblib
import numpy as np
import torch
from fastapi import FastAPI, HTTPException

from pdm.config import DEVICE, FEATURE_SENSORS, PROC
from pdm.serving.schemas import (
    Alert,
    AlertResponse,
    FleetKPIs,
    FleetResponse,
    HealthResponse,
    PredictRequest,
    PredictResponse,
)

logger = logging.getLogger(__name__)

# ── Application state ──────────────────────────────────────────────
# We store the model and conformal predictor here so they're loaded
# once at startup and reused for every request. This dict acts as
# a simple service container.
STATE: dict = {}

app = FastAPI(
    title="Predictive Maintenance API",
    description=(
        "RUL prediction for turbofan engines. "
        "LSTM model with calibrated conformal prediction intervals."
    ),
    version="1.0.0",
)


# ── Startup: load model once ───────────────────────────────────────
@app.on_event("startup")
def load_model():
    """
    Load the production model and conformal predictor on server start.

    Why on_event("startup") instead of loading at module level?
    - Cleaner error handling (FastAPI catches and reports startup failures)
    - Model isn't loaded during import (faster tests, cleaner imports)
    - Can be swapped to lifespan context manager in newer FastAPI versions
    """
    from pdm.serving.registry import load_production_model

    model, info = load_production_model("rul-lstm")
    STATE["model"] = model
    STATE["model_info"] = info

    # Load conformal predictor for uncertainty intervals
    conformal_path = PROC / "conformal.joblib"
    if conformal_path.exists():
        STATE["conformal"] = joblib.load(conformal_path)
        logger.info("Loaded conformal predictor")
    else:
        logger.warning("No conformal predictor found — intervals unavailable")
        STATE["conformal"] = None

    logger.info(f"Model loaded: {info}")


# ── Health check ───────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse)
def health():
    """
    Health check endpoint.

    Every production API needs this. Kubernetes uses it for liveness
    and readiness probes — if this returns non-200, k8s restarts the pod.
    Load balancers use it to route traffic only to healthy instances.
    """
    info = STATE.get("model_info", {})
    return HealthResponse(
        status="healthy",
        model_name=info.get("name", "unknown"),
        model_version=str(info.get("version", "unknown")),
        model_stage=info.get("stage", "unknown"),
    )


# ── Single-unit prediction ────────────────────────────────────────
@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """
    Predict RUL for a single engine unit.

    Input: 30 timesteps × 14 sensor values (already scaled).
    Output: RUL point estimate + 90% conformal prediction interval.

    This is the endpoint an engineer would call to check a specific
    engine. The dashboard and alerting system use /predict/fleet instead.
    """
    # Validate input shape
    seq = np.array(req.sequence, dtype=np.float32)
    if seq.shape != (30, len(FEATURE_SENSORS)):
        raise HTTPException(
            status_code=422,
            detail=f"Expected shape (30, {len(FEATURE_SENSORS)}), "
                   f"got {seq.shape}",
        )

    # Run inference
    model = STATE["model"]
    x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)  # (1, 30, 14)

    model.eval()
    with torch.no_grad():
        rul = model(x).cpu().item()

    # Conformal intervals
    if STATE.get("conformal"):
        intervals = STATE["conformal"].predict(np.array([rul]))
        lower = float(intervals["lower"][0])
        upper = float(intervals["upper"][0])
    else:
        lower = max(rul - 20, 0)  # fallback: rough ±20
        upper = rul + 20

    # Status classification
    if rul < 30:
        status = "critical"
    elif rul < 80:
        status = "warning"
    else:
        status = "healthy"

    return PredictResponse(rul=round(rul, 1), lower=round(lower, 1),
                           upper=round(upper, 1), unit_status=status)


# ── Fleet prediction ──────────────────────────────────────────────
@app.get("/predict/fleet", response_model=FleetResponse)
def predict_fleet():
    """
    Predict RUL for ALL units in the test set.

    Returns per-unit predictions + fleet-level KPIs.
    This is what the Streamlit dashboard (Step 13) calls to populate
    the Fleet Health tab. It's also what a fleet manager would hit
    to get a full picture of their equipment.

    In production, this would read from a database of live sensor
    data instead of the static test set.
    """
    # Load test data
    data_path = PROC / "FD001.npz"
    if not data_path.exists():
        raise HTTPException(status_code=503, detail="Test data not found")

    d = np.load(data_path)
    Xte = torch.tensor(d["Xte"], dtype=torch.float32).to(DEVICE)

    # Batch inference
    model = STATE["model"]
    model.eval()
    with torch.no_grad():
        preds = model(Xte).cpu().numpy()

    # Conformal intervals for all units
    if STATE.get("conformal"):
        intervals = STATE["conformal"].predict(preds)
        lower = intervals["lower"].tolist()
        upper = intervals["upper"].tolist()
    else:
        lower = np.maximum(preds - 20, 0).tolist()
        upper = (preds + 20).tolist()

    preds_list = preds.tolist()
    n_units = len(preds_list)

    # Fleet KPIs
    preds_arr = np.array(preds_list)
    critical = int((preds_arr < 30).sum())
    warning = int(((preds_arr >= 30) & (preds_arr < 80)).sum())
    healthy = int((preds_arr >= 80).sum())

    return FleetResponse(
        units=list(range(1, n_units + 1)),
        rul_predictions=[round(p, 1) for p in preds_list],
        confidence_lower=[round(v, 1) for v in lower],
        confidence_upper=[round(v, 1) for v in upper],
        kpis=FleetKPIs(
            total_units=n_units,
            critical=critical,
            warning=warning,
            healthy=healthy,
            avg_rul=round(float(preds_arr.mean()), 1),
            min_rul=round(float(preds_arr.min()), 1),
        ),
    )


# ── Alerting ──────────────────────────────────────────────────────
@app.get("/alerts", response_model=AlertResponse)
def get_alerts(rul_threshold: int = 30):
    """
    Return units below the RUL threshold that need attention.

    This is the "alerting system" that FLEXOO and DNV JDs want.
    In production, this would integrate with PagerDuty, Slack, or email.

    Args:
        rul_threshold: units with predicted RUL below this get flagged.
                       Default 30 cycles (about 1 month of operation).
    """
    fleet = predict_fleet()

    alerts = []
    for i, rul in enumerate(fleet.rul_predictions):
        if rul < rul_threshold:
            alerts.append(Alert(
                unit_id=i + 1,
                predicted_rul=round(rul, 1),
                confidence_lower=round(fleet.confidence_lower[i], 1),
                confidence_upper=round(fleet.confidence_upper[i], 1),
                severity="critical" if rul < 15 else "warning",
                action=(
                    "Schedule immediate maintenance"
                    if rul < 15
                    else "Plan maintenance within next cycle window"
                ),
            ))

    # Sort by most urgent first
    alerts.sort(key=lambda a: a.predicted_rul)

    return AlertResponse(
        threshold=rul_threshold,
        total_alerts=len(alerts),
        alerts=alerts,
    )
