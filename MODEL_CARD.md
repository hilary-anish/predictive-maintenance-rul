# Model Card: RUL Prediction (LSTM + Conformal)

## Model Details
- **Model type:** LSTM sequence-to-value regressor with MC-Dropout uncertainty
- **Framework:** PyTorch
- **Version:** rul-lstm v1 (see MLflow registry)
- **Maintainer:** Anish Hilary Ignatius

## Intended Use
- **Primary use:** Predict Remaining Useful Life (RUL) of turbofan engines
  from multivariate operational sensor time-series
- **Users:** Maintenance planners, reliability engineers, fleet managers
- **Out of scope:** Real-time safety-critical control decisions (model
  provides advisory predictions, not control signals)

## Training Data
- **Dataset:** NASA C-MAPSS FD001 (run-to-failure simulation)
- **Size:** 100 training units, ~17,700 time windows (after windowing)
- **Features:** 14 operational sensors (constant/near-constant sensors removed)
- **Target:** Remaining Useful Life (cycles), capped at 125

## Performance Metrics

| Model | RMSE | MAE | PHM Score | Parameters |
|-------|------|-----|-----------|------------|
| **LSTM** | **13.04** | **9.53** | **270.81** | 55,873 |
| XGBoost | 15.17 | 11.15 | 378.67 | — |
| TCN | 15.35 | 11.74 | 362.21 | 52,161 |

### Uncertainty Calibration

| Metric | Value |
|--------|-------|
| Target Coverage | 90% |
| Empirical Coverage | 94% |
| Mean Interval Width | 44.5 cycles |
| Conformal Quantile (q̂) | 23.03 cycles |

## Limitations
- Trained on simulated data only (C-MAPSS), not real engine data
- Single fault mode (HPC degradation in FD001)
- Does not generalize to multi-condition datasets (FD002-004) without retraining
- Uncertainty intervals assume exchangeability of calibration data

## Ethical Considerations
- Model predictions should support, not replace, human maintenance decisions
- False negatives (predicting long RUL for failing equipment) carry safety risk
- The PHM08 asymmetric score penalizes late predictions more heavily, reflecting
  this safety priority

## Monitoring
- Evidently drift detection runs via Kubernetes CronJob
- Automated retraining triggers when dataset drift is detected
- Champion/challenger comparison prevents regression on deployment
- All experiments tracked in MLflow with full reproducibility
