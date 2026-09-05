"""
Calibrate conformal prediction intervals on the LSTM model.

Workflow:
    1. Load trained LSTM and test data
    2. Split test into calibration (50%) and evaluation (50%)
    3. Calibrate conformal predictor on calibration set
    4. Evaluate coverage on evaluation set
    5. Log everything to MLflow

Why 50/50 split?
- More calibration data = tighter intervals (better precision)
- More evaluation data = more reliable coverage estimate
- 50/50 is the standard split in conformal prediction literature
- With 100 test units in FD001, that's 50 cal + 50 eval
"""
import numpy as np
import torch
import mlflow
import joblib
from pdm.config import PROC, FEATURE_SENSORS, RANDOM_STATE, DEVICE
from pdm.models.lstm import RULLSTM
from pdm.models.conformal import ConformalPredictor


def main(subset: str = "FD001", alpha: float = 0.10):
    np.random.seed(RANDOM_STATE)

    # --- Load data and trained model ---
    d = np.load(PROC / f"{subset}.npz")
    Xte = torch.tensor(d["Xte"], dtype=torch.float32).to(DEVICE)
    yte = d["yte"]

    model = RULLSTM(len(FEATURE_SENSORS)).to(DEVICE)
    model.load_state_dict(torch.load(PROC / "lstm.pt", weights_only=True))
    model.eval()

    # --- Get predictions ---
    with torch.no_grad():
        preds = model(Xte).cpu().numpy()

    # --- Split into calibration and evaluation ---
    n = len(yte)
    idx = np.random.permutation(n)
    cal_idx = idx[:n // 2]
    eval_idx = idx[n // 2:]

    y_cal_true = yte[cal_idx]
    y_cal_pred = preds[cal_idx]
    y_eval_true = yte[eval_idx]
    y_eval_pred = preds[eval_idx]

    print(f"Calibration set: {len(cal_idx)} units")
    print(f"Evaluation set:  {len(eval_idx)} units")

    # --- Calibrate ---
    cp = ConformalPredictor(alpha=alpha)
    q_hat = cp.calibrate(y_cal_true, y_cal_pred)
    print(f"\nConformal quantile (q_hat): {q_hat:.2f} cycles")
    print(f"This means intervals are +/- {q_hat:.2f} cycles around predictions")

    # --- Evaluate on held-out evaluation set ---
    eval_result = cp.evaluate_coverage(y_eval_true, y_eval_pred)

    print(f"\n--- Calibration Results (alpha={alpha}) ---")
    print(f"Target coverage:    {eval_result['target_coverage']:.0%}")
    print(f"Empirical coverage: {eval_result['empirical_coverage']:.0%}")
    print(f"Coverage gap:       {eval_result['coverage_gap']:+.1%}")
    print(f"Mean interval width: {eval_result['mean_interval_width']:.1f} cycles")

    # --- Also evaluate on FULL test set (what goes in README) ---
    # Re-calibrate on first half, predict for all
    full_intervals = cp.predict(preds)
    full_covered = (yte >= full_intervals["lower"]) & (yte <= full_intervals["upper"])
    full_coverage = float(full_covered.mean())

    print(f"\n--- Full Test Set ---")
    print(f"Coverage (all {n} units): {full_coverage:.0%}")
    print(f"Mean interval width:      {full_intervals['width'].mean():.1f} cycles")

    # --- Save conformal predictor for use in the API ---
    joblib.dump(cp, PROC / "conformal.joblib")
    print(f"\nSaved conformal predictor to {PROC / 'conformal.joblib'}")

    # --- Log to MLflow ---
    mlflow.set_experiment("rul-cmapss")
    with mlflow.start_run(run_name="conformal-calibration"):
        mlflow.log_params({
            "alpha": alpha,
            "target_coverage": 1 - alpha,
            "n_calibration": len(cal_idx),
            "n_evaluation": len(eval_idx),
        })
        mlflow.log_metrics({
            "q_hat": q_hat,
            "empirical_coverage": eval_result["empirical_coverage"],
            "coverage_gap": eval_result["coverage_gap"],
            "mean_interval_width": eval_result["mean_interval_width"],
            "full_test_coverage": full_coverage,
        })
        mlflow.log_artifact(str(PROC / "conformal.joblib"))

    # --- Print example predictions ---
    print(f"\n--- Example Predictions (first 5 units) ---")
    print(f"{'Unit':>6s} {'True':>8s} {'Pred':>8s} {'Lower':>8s} {'Upper':>8s} {'Covered':>8s}")
    print("-" * 52)
    for i in range(min(5, n)):
        covered = "✓" if full_covered[i] else "✗"
        print(f"{i+1:6d} {yte[i]:8.1f} {preds[i]:8.1f} "
              f"{full_intervals['lower'][i]:8.1f} "
              f"{full_intervals['upper'][i]:8.1f} "
              f"{covered:>8s}")


if __name__ == "__main__":
    main()
