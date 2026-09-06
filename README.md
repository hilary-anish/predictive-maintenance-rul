# Predictive Maintenance — Remaining Useful Life (RUL) Estimation

End-to-end predictive maintenance system that estimates turbofan engine Remaining Useful Life from multivariate sensor time-series. Benchmarks LSTM and Temporal CNN against an XGBoost baseline, with calibrated conformal prediction intervals, automated drift-triggered retraining, and a production deployment pipeline.

[![CI](https://github.com/hilary-anish/predictive-maintenance-rul/actions/workflows/ci.yml/badge.svg)](https://github.com/hilary-anish/predictive-maintenance-rul/actions/workflows/ci.yml)

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [System Architecture](#system-architecture)
- [Data: NASA C-MAPSS](#data-nasa-c-mapss)
- [Data Pipeline](#data-pipeline)
- [Model Architectures](#model-architectures)
- [Uncertainty Quantification](#uncertainty-quantification)
- [Experiment Tracking and Model Registry](#experiment-tracking-and-model-registry)
- [API Serving](#api-serving)
- [Drift Monitoring](#drift-monitoring)
- [Automated Retraining Pipeline](#automated-retraining-pipeline)
- [Dashboard](#dashboard)
- [Testing and CI/CD](#testing-and-cicd)
- [Containerization and Orchestration](#containerization-and-orchestration)
- [Model Card](#model-card)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Tech Stack](#tech-stack)

---

## Problem Statement

Unplanned equipment failure costs industrial operations thousands of dollars per hour in downtime, emergency repairs, and cascading production delays. Traditional maintenance strategies fall short: reactive maintenance waits for failure (too late), while scheduled maintenance replaces parts on fixed intervals regardless of actual condition (wasteful and still misses some failures).

Predictive maintenance uses sensor data to estimate when a machine will actually fail, allowing maintenance teams to intervene at the right time — not too early (wasting part life), not too late (risking failure). The core prediction task is Remaining Useful Life (RUL): given a sequence of sensor readings, how many operational cycles does this engine have left before it needs repair?

This project builds a complete production-grade RUL prediction system — from raw sensor data to a deployed, monitored, auto-retraining API with a stakeholder-facing dashboard.

---

## System Architecture

```
Sensor Data → Preprocessing → Windowing → Model (LSTM / TCN / XGBoost)
               (MinMaxScaler)  (30-step)           ↓
                                          Conformal Prediction
                                          (calibrated 90% intervals)
                                                   ↓
                                 FastAPI ←── MLflow Model Registry
                                   ↓                    ↑
                  ┌────────────────┼────────────┐       │
                  ↓                ↓             ↓       │
             /predict         /predict/fleet   /alerts   │
                  ↓                ↓             ↓       │
              Streamlit KPI Dashboard                    │
              (Fleet Health, Drift,                      │
               Model Comparison,                         │
               Retrain History)                          │
                                                         │
              Evidently Drift Monitor ──→ Automated  ────┘
              (Wasserstein distance)      Retraining
                                          Pipeline
                                          (Champion/Challenger)
                                                ↓
                                    Kubernetes CronJob (nightly)
```

---

## Data: NASA C-MAPSS

**Dataset:** NASA Commercial Modular Aero-Propulsion System Simulation (C-MAPSS), subset FD001. This is the standard benchmark in the predictive maintenance research community.

**Why C-MAPSS?** It provides run-to-failure trajectories with ground truth RUL labels — something that is extremely rare in real industrial data (most equipment gets maintained before failure, so you never observe the actual failure point). FD001 contains a single fault mode (High Pressure Compressor degradation) under one operating condition, making it ideal for demonstrating the pipeline without conflating multiple failure mechanisms.

**Structure:**
- 100 training engines run to failure (variable length sequences, 128–362 cycles each)
- 100 test engines with partial trajectories + ground truth RUL for the last cycle
- 21 sensor channels + 3 operational settings per cycle
- After removing constant/near-constant sensors, 14 informative features remain

**Preprocessing decisions:**
- **MinMaxScaler** (not StandardScaler): sensor readings have physical bounds and skewed distributions. MinMaxScaler preserves the bounded nature and works better with the sigmoid-like activations in the LSTM.
- **RUL clipping at 125 cycles**: early in an engine's life, the exact RUL (e.g., 300 vs 280 cycles) is irrelevant for maintenance decisions. Clipping at 125 focuses the model on the critical degradation phase where predictions matter. This is standard in the C-MAPSS literature.
- **Sliding windows of 30 timesteps**: the LSTM and TCN need fixed-length input sequences. 30 cycles captures enough degradation trajectory to be informative without exceeding the shortest engine life in the dataset.

---

## Data Pipeline

The data pipeline is split into separate, testable modules rather than a single monolithic notebook. Each module has a single responsibility:

```
load.py        → reads raw .txt files into DataFrames
validate.py    → Pandera schema validation (types, ranges, nulls)
labels.py      → generates piecewise-linear RUL labels (clipped at 125)
preprocess.py  → fits MinMaxScaler on training data, transforms both sets
windows.py     → creates sliding windows (30 × 14) for sequence models
build_dataset.py → orchestrates the full pipeline, saves .npz
```

**Why Pandera for validation?** Data validation catches silent corruption before it poisons the model. Pandera provides declarative schema definitions (column types, value ranges, null checks) that run automatically. In production, incoming sensor data is validated against this schema before being fed to the model.

**Why .npz output?** NumPy's compressed format is fast to load, small on disk, and has zero dependencies. It stores the exact arrays the models consume: `Xtr (N, 30, 14)`, `ytr (N,)`, `Xte (100, 30, 14)`, `yte (100,)`.

---

## Model Architectures

Three architectures are benchmarked on the same data split with the same evaluation metrics, all tracked in the same MLflow experiment for direct comparison.

### XGBoost Baseline

**Why start with XGBoost?** Every deep learning project needs a non-neural baseline to prove the added complexity of DL is justified. If XGBoost matches or beats the LSTM, you don't need the LSTM. XGBoost also serves as a sanity check — if it gets terrible results, the problem is likely in the data pipeline, not the model.

**Feature engineering for XGBoost:** Since XGBoost expects tabular input (not sequences), each 30-step window is summarized into four statistics per sensor: mean, standard deviation, last value, and delta (last minus first). This produces a 56-dimensional feature vector (14 sensors × 4 statistics) that captures both the level and trend of each sensor.

### LSTM (Long Short-Term Memory)

**Why LSTM?** RUL prediction is inherently sequential — the degradation pattern over time matters, not just the current sensor values. LSTMs process sequences step-by-step, maintaining a hidden state that captures long-range temporal dependencies. They are the most widely used architecture for time-series prediction in the predictive maintenance literature.

**Architecture:** 2-layer LSTM (hidden size 64) with dropout (0.3) between layers, followed by a linear head that maps the last hidden state to a single RUL prediction. MC-Dropout (keeping dropout active during inference) enables uncertainty estimation via multiple forward passes.

### Temporal CNN (TCN)

**Why TCN alongside LSTM?** Multiple job descriptions mentioned "Temporal CNN" alongside LSTM as a desired skill. Having both demonstrates understanding of the tradeoffs between recurrent and convolutional sequence modeling:

| Aspect | LSTM | TCN |
|--------|------|-----|
| Sequence processing | Sequential (one step at a time) | Parallel (all steps at once) |
| Long-range memory | Can forget via vanishing gradients | Fixed receptive field via dilated convolutions |
| Training speed | Slower (cannot parallelize across time) | Faster (convolutions are GPU-friendly) |
| When to prefer | Strong sequential dependencies | Speed + long-range patterns |

**Architecture:** Three temporal blocks with exponentially increasing dilation (1, 2, 4), each containing two causal convolutions with residual connections. The causal design ensures the model never looks at future timesteps — critical for time-series prediction. Channel sizes of [64, 64, 32] progressively compress the representation before a linear head produces the RUL prediction.

### Results

| Model | RMSE ↓ | MAE ↓ | PHM Score ↓ | Parameters |
|-------|--------|-------|-------------|------------|
| **LSTM** | **13.04** | **9.53** | **270.81** | 55,873 |
| XGBoost | 15.17 | 11.15 | 378.67 | — |
| TCN | 15.35 | 11.74 | 362.21 | 52,161 |

LSTM outperforms both alternatives on all three metrics. The XGBoost baseline justifies the deep learning approach (2+ RMSE improvement). TCN achieves comparable performance with 7% fewer parameters and faster training, but does not surpass LSTM on this dataset — consistent with published results on C-MAPSS.

**PHM Score:** The PHM08 asymmetric scoring function penalizes late predictions (predicting longer RUL than actual — the engine fails before expected) more heavily than early predictions (predicting shorter RUL — unnecessary early maintenance). This reflects real-world safety priorities where missed failures are far more costly than conservative maintenance.

---

## Uncertainty Quantification

Most ML projects output a single number: "RUL = 65 cycles." This is useless for a maintenance planner. They need: "RUL is between 42 and 88 cycles with 90% confidence." That interval drives the scheduling decision.

### Why Conformal Prediction?

| Method | Coverage Guarantee | Complexity |
|--------|--------------------|------------|
| MC-Dropout | No formal guarantee | Moderate |
| Ensemble variance | No formal guarantee | High (train N models) |
| Bayesian Neural Network | Approximate only | Very high |
| **Conformal Prediction** | **Mathematically guaranteed** | **Low** |

Conformal prediction provides distribution-free, model-agnostic prediction intervals with finite-sample coverage guarantees. If you ask for 90% intervals, the true value will fall inside the interval at least 90% of the time — provably, under the exchangeability assumption.

### How It Works

1. Hold out 50 calibration samples the model has never seen during training
2. Compute absolute residuals: `|y_true - y_pred|` for each calibration sample
3. Sort residuals and take the 90th percentile: `q̂ = 23.03 cycles`
4. At prediction time: interval = `[prediction - q̂, prediction + q̂]`

The finite-sample correction `ceil((n+1)(1-α))/n` from Vovk et al. (2005) ensures exact coverage even with small calibration sets.

### Calibration Results

| Metric | Value |
|--------|-------|
| Target Coverage | 90% |
| Empirical Coverage | 94% |
| Coverage Gap | +4% (conservative — preferred for safety) |
| Conformal Quantile (q̂) | 23.03 cycles |
| Mean Interval Width | 44.5 cycles |

The +4% overcoverage means the intervals are slightly conservative. For maintenance planning this is preferable — better to schedule maintenance slightly early than to miss a failure.

---

## Experiment Tracking and Model Registry

### MLflow Experiment Tracking

All training runs log to the same MLflow experiment (`rul-cmapss`), enabling direct comparison across architectures. Each run records hyperparameters, training metrics, evaluation metrics, and the serialized model artifact.

**Why MLflow?** It's the industry standard for experiment tracking. It replaces the "which notebook had the best results?" problem with a structured, queryable database of every experiment.

### MLflow Model Registry

The registry adds lifecycle management on top of experiment tracking:

- **Production** — the model currently serving predictions via the API
- **Staging** — candidate model under evaluation
- **Archived** — previous production models kept for audit trail

The API calls `load_production_model()` on startup, which queries the registry for whichever model is currently marked as Production. The automated retraining pipeline (see below) updates this programmatically when a better model is found.

---

## API Serving

FastAPI serves predictions over HTTP with three endpoints:

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Liveness/readiness check (used by Kubernetes probes) |
| `/predict` | POST | Single-unit RUL prediction with conformal intervals |
| `/predict/fleet` | GET | All units with fleet-level KPIs (critical/warning/healthy counts) |
| `/alerts` | GET | Units below a configurable RUL threshold, sorted by urgency |

**Why FastAPI over Flask?** FastAPI generates interactive OpenAPI documentation automatically at `/docs`, validates request/response schemas via Pydantic, supports async request handling, and has better performance benchmarks. The auto-generated docs mean reviewers can test the API directly from their browser.

**Design decisions:**
- Model is loaded once at startup (not per-request) — inference takes milliseconds, loading takes seconds
- Conformal intervals are attached to every prediction — the API never returns a point estimate without uncertainty
- The `/alerts` endpoint mimics what a PagerDuty/Slack integration would consume in production

---

## Drift Monitoring

In production, data distributions shift over time: sensors degrade differently, operating conditions change, maintenance patterns evolve. If the model's input distribution changes, its predictions become unreliable. This is data drift.

### Evidently Implementation

The drift monitor compares a reference dataset (training data) against current production data using the Wasserstein distance (for large sample sizes) or the Kolmogorov-Smirnov test (for small sample sizes). Evidently auto-selects the appropriate test.

Each 3D input window (30 timesteps × 14 sensors) is summarized into 42 features: mean, standard deviation, and last value per sensor. Drift is evaluated per feature. Dataset-level drift is flagged when >50% of features show statistically significant distribution shifts.

**Why Evidently?** It's the leading open-source ML monitoring library, generates visual HTML reports for non-technical stakeholders, and provides structured JSON output for programmatic consumption by the retraining pipeline.

The drift monitor runs as a standalone script and also integrates with the automated retraining pipeline. Drift results and HTML reports are saved for audit and stakeholder review.

---

## Automated Retraining Pipeline

The retraining pipeline closes the loop between drift detection and model updates. It implements the champion/challenger pattern:

```
Nightly CronJob triggers
    → Check for drift (Evidently)
    → If drift detected: retrain LSTM
    → Compare new model (challenger) vs current Production model (champion)
    → If challenger RMSE < champion RMSE by ≥ 0.5: promote challenger
    → If not: keep champion (prevents noise-level model swaps)
    → Log decision to JSONL audit trail
```

**Why champion/challenger instead of always deploying the new model?**
- Retraining on drifted data might produce a worse model
- The 0.5 RMSE improvement threshold prevents unnecessary model swaps that add deployment risk without meaningful gain
- The audit log records every decision with full reasoning — this is what governance teams need for compliance

**Why JSONL for the audit log?** One JSON object per line is easy to append (no file corruption risk), easy to parse (line by line), and easy to load into pandas or a dashboard for analysis.

The pipeline runs via a Kubernetes CronJob (nightly at 02:00 UTC), but can also be triggered manually with `--force` or `--simulate-drift` flags for testing.

---

## Dashboard

A Streamlit KPI dashboard designed for non-technical stakeholders (maintenance managers, operations leads, fleet planners) with four tabs:

**Fleet Health** — KPI cards (critical/warning/healthy unit counts, average fleet RUL), RUL distribution histogram with threshold lines, maintenance priority queue (top 10 most urgent units), and a scatter plot of all unit predictions with 90% confidence bands.

**Model Performance** — Architecture comparison table pulled from MLflow, RMSE bar chart, and conformal calibration metrics.

**Drift Monitor** — Latest drift detection results with feature-level breakdown, and the Evidently HTML report embedded for visual inspection.

**Retrain History** — Timeline of all automated retraining decisions with expandable details showing champion vs challenger metrics, drift ratios, and promotion reasoning.

The dashboard works in two modes: local (reads model files directly, no API needed) and API mode (calls FastAPI endpoints). Local mode enables deployment on Hugging Face Spaces for a live recruiter demo without infrastructure.

---

## Testing and CI/CD

### Test Suite (22 tests)

| Test File | What It Validates |
|-----------|-------------------|
| `test_labels.py` | RUL label generation, clipping, one-row-per-unit for test set |
| `test_windows.py` | Sliding window shapes, count, label alignment, short-unit padding |
| `test_metrics.py` | RMSE, MAE correctness, PHM score asymmetry property |
| `test_api.py` | Health endpoint, valid prediction response, input shape rejection |
| `test_retrain.py` | Promote on no champion, promote on improvement, keep on insufficient improvement |

API tests use FastAPI's `TestClient` with mocked models — no GPU, no data files, runs in milliseconds. Pipeline tests verify decision logic in isolation using `unittest.mock`.

### GitHub Actions CI

Every push to `main` and every pull request triggers:
1. Code checkout
2. Python 3.11 environment setup with `uv`
3. Dependency installation
4. `ruff check .` — linting for style, import ordering, line length
5. `pytest tests/ -v` — full test suite

The CI badge at the top of this README reflects the latest pipeline status.

---

## Containerization and Orchestration

### Docker

Multi-stage Dockerfile: the build stage installs compilers and dependencies, the runtime stage copies only the installed packages. This cuts the image size roughly in half. The container runs as a non-root user (`appuser`) to limit the blast radius of any security exploit.

Docker Compose orchestrates the API service with volume mounts for data and MLflow artifacts, so model updates don't require image rebuilds.

### Kubernetes

Kubernetes manifests for production-grade deployment:

| Resource | Purpose |
|----------|---------|
| **Deployment** | Runs 2 API replicas with rolling update strategy |
| **Service** | Stable internal endpoint, load-balances across replicas |
| **Ingress** | Routes external HTTP traffic to the Service |
| **HPA** | Auto-scales replicas (2–5) when CPU exceeds 70% |
| **ConfigMap** | Environment variables decoupled from code |
| **CronJob** | Nightly retraining pipeline execution |

Liveness and readiness probes hit `/health` to ensure Kubernetes only routes traffic to pods that have successfully loaded the model.

**Why Kubernetes for this project?** This workload is genuinely multi-service: an API Deployment (scalable), a CronJob for retraining (scheduled batch), a ConfigMap for settings, and an HPA for autoscaling. That justifies the orchestration layer — unlike a single-container demo where Docker Compose would suffice.

Local development uses k3d, which runs a lightweight Kubernetes cluster inside Docker containers.

---

## Model Card

### Model Details
- **Model type:** LSTM sequence-to-value regressor (2-layer, hidden=64, dropout=0.3)
- **Framework:** PyTorch
- **Version:** rul-lstm v1 (MLflow model registry)
- **Maintainer:** Anish Hilary Ignatius

### Intended Use
- **Primary use:** Predict Remaining Useful Life of turbofan engines from multivariate operational sensor time-series
- **Users:** Maintenance planners, reliability engineers, fleet managers
- **Out of scope:** Real-time safety-critical control decisions. This model provides advisory predictions, not control signals. All maintenance decisions should involve human judgment.

### Limitations
- Trained on simulated data only (C-MAPSS), not real engine telemetry
- Single fault mode (HPC degradation in FD001) — does not generalize to multi-condition datasets (FD002–004) without retraining
- Conformal prediction intervals assume exchangeability between calibration data and future production data
- No GPU required for inference, but training benefits significantly from CUDA

### Ethical Considerations
- False negatives (predicting long RUL for a failing engine) carry direct safety risk. The PHM asymmetric scoring function reflects this by penalizing late predictions more heavily.
- Model predictions should augment, not replace, human expertise. Maintenance decisions involve contextual factors (parts availability, crew scheduling, flight routing) that the model does not capture.
- The automated retraining pipeline includes a champion/challenger safeguard — new models are never deployed without outperforming the current production model by a meaningful margin.

### Monitoring and Governance
- Evidently drift detection evaluates 42 engineered features for distribution shifts
- Automated retraining triggers on drift with full champion/challenger comparison
- Every retraining decision is logged to a JSONL audit trail with timestamps, metrics, and reasoning
- All experiments are tracked in MLflow with full parameter and metric reproducibility

---

## Project Structure

```
predictive-maintenance-rul/
├── README.md                        # This file
├── MODEL_CARD.md                    # Standalone model card (subset of this README)
├── pyproject.toml                   # Dependencies, build config, tool settings
├── Dockerfile                       # Multi-stage build for the API container
├── docker-compose.yml               # Service orchestration with volume mounts
├── .github/workflows/ci.yml         # GitHub Actions CI pipeline
│
├── data/
│   ├── raw/                         # C-MAPSS .txt files (gitignored)
│   └── processed/                   # .npz arrays, model checkpoints (gitignored)
│
├── notebooks/
│   └── 01_eda.ipynb                 # Exploratory data analysis
│
├── src/pdm/
│   ├── config.py                    # Paths, constants, device selection
│   ├── data/
│   │   ├── load.py                  # Raw file reading
│   │   ├── validate.py              # Pandera schema validation
│   │   ├── labels.py                # RUL label generation (piecewise-linear, clipped)
│   │   ├── preprocess.py            # MinMaxScaler fitting and transformation
│   │   ├── windows.py              # Sliding window generation for sequence models
│   │   └── build_dataset.py         # End-to-end data pipeline orchestration
│   ├── features/
│   │   └── build.py                 # Feature engineering utilities
│   ├── models/
│   │   ├── lstm.py                  # RULLSTM architecture
│   │   ├── tcn.py                   # RULTCN architecture (causal dilated convolutions)
│   │   ├── train_lstm.py            # LSTM training with MLflow logging
│   │   ├── train_tcn.py             # TCN training with MLflow logging
│   │   ├── train_xgb.py             # XGBoost baseline with MLflow logging
│   │   ├── conformal.py             # ConformalPredictor class
│   │   └── calibrate.py             # Calibration script (split, calibrate, evaluate)
│   ├── evaluate/
│   │   └── metrics.py               # RMSE, MAE, PHM asymmetric score
│   ├── monitoring/
│   │   └── drift.py                 # Evidently drift detection + simulation
│   ├── pipeline/
│   │   └── retrain.py               # Automated retraining (drift → retrain → compare → promote)
│   └── serving/
│       ├── api.py                   # FastAPI server (predict, fleet, alerts, health)
│       ├── schemas.py               # Pydantic request/response models
│       └── registry.py              # MLflow model registry helpers
│
├── app/
│   └── streamlit_app.py             # KPI dashboard (fleet health, drift, retrain history)
│
├── tests/                           # 22 pytest tests
│   ├── test_labels.py
│   ├── test_windows.py
│   ├── test_metrics.py
│   ├── test_api.py
│   └── test_retrain.py
│
└── k8s/                             # Kubernetes manifests
    ├── configmap.yaml
    ├── deployment.yaml
    ├── service.yaml
    ├── ingress.yaml
    ├── hpa.yaml
    └── cronjob-retrain.yaml
```

---

## Quick Start

```bash
# Clone and install
git clone https://github.com/hilary-anish/predictive-maintenance-rul.git
cd predictive-maintenance-rul
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[dev]"

# Download NASA C-MAPSS FD001 data into data/raw/
# (see NASA Prognostics Data Repository)

# Build the dataset
python -m pdm.data.build_dataset

# Train all three models
python -m pdm.models.train_xgb
python -m pdm.models.train_lstm
python -m pdm.models.train_tcn

# Calibrate conformal prediction intervals
python -m pdm.models.calibrate

# Run drift detection
python -m pdm.monitoring.drift
python -m pdm.monitoring.drift --simulate    # with artificial drift

# Test the retraining pipeline
python -m pdm.pipeline.retrain --force

# Start the API
uvicorn pdm.serving.api:app --host 0.0.0.0 --port 8000
# Interactive docs at http://localhost:8000/docs

# Launch the dashboard
streamlit run app/streamlit_app.py

# Run tests
pytest tests/ -v

# Docker
docker compose build
docker compose up

# Kubernetes (local)
k3d cluster create rul-cluster --port "8080:80@loadbalancer"
kubectl apply -f k8s/
```

---

## Tech Stack

| Category | Tool | Why This Choice |
|----------|------|-----------------|
| ML / DL | PyTorch | Industry standard for research + production; dynamic graphs for flexible architectures |
| Baseline | XGBoost | Strongest tabular baseline; proves DL earns its complexity |
| Uncertainty | Conformal Prediction | Only method with finite-sample coverage guarantees; model-agnostic |
| Experiment Tracking | MLflow | Industry standard; tracks params, metrics, artifacts, model registry |
| Data Validation | Pandera | Declarative schema validation; catches silent data corruption |
| Drift Monitoring | Evidently | Leading open-source ML monitoring; visual reports + JSON for automation |
| API Framework | FastAPI | Auto-generated docs, Pydantic validation, async-capable, production-grade |
| Dashboard | Streamlit + Plotly | Python-native, interactive, deployable on Hugging Face Spaces (free) |
| Containerization | Docker | Reproducible environments; required for Kubernetes deployment |
| Orchestration | Kubernetes (k3d) | Multi-service workload (API + CronJob + HPA) justifies orchestration |
| CI/CD | GitHub Actions | Free for public repos; runs lint + tests on every push |
| Linting | ruff | 10–100× faster than flake8/black; single tool for lint + format |
| Package Management | uv | 10–100× faster than pip; modern Python packaging |

---

## License

MIT
