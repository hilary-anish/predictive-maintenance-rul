"""
Data drift detection using Evidently.

Why Evidently?
- Industry standard for ML monitoring (used at major companies)
- Statistical tests per feature (KS test, Wasserstein distance)
- Generates visual HTML reports for stakeholders
- Integrates with our automated retraining pipeline (Step 12)

How drift detection works:
    1. Keep a REFERENCE dataset (what the model was trained on)
    2. Collect CURRENT data (recent production predictions)
    3. Run statistical tests comparing distributions
    4. If enough features have drifted -> trigger retraining

Evidently auto-selects the statistical test based on sample sizes:
    - Small datasets (<1000): Kolmogorov-Smirnov test (p-value < 0.05 = drift)
    - Large datasets (>=1000): Wasserstein distance (distance > 0.1 = drift)
"""
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

from pdm.config import FEATURE_SENSORS, PROC

logger = logging.getLogger(__name__)


def _arrays_to_dataframe(data: np.ndarray) -> pd.DataFrame:
    """
    Convert 3D array (N, seq_len, features) to 2D DataFrame.

    For drift detection, we summarize each window into statistics:
    - mean of each feature across the window
    - std of each feature across the window
    - last value of each feature (most recent reading)
    """
    means = pd.DataFrame(
        data.mean(axis=1),
        columns=[f"{s}_mean" for s in FEATURE_SENSORS]
    )
    stds = pd.DataFrame(
        data.std(axis=1),
        columns=[f"{s}_std" for s in FEATURE_SENSORS]
    )
    lasts = pd.DataFrame(
        data[:, -1, :],
        columns=[f"{s}_last" for s in FEATURE_SENSORS]
    )

    return pd.concat([means, stds, lasts], axis=1)


def _is_drifted(method: str, value: float, threshold: float) -> bool:
    """
    Determine if a feature has drifted based on the test method.

    Evidently auto-selects the test:
    - K-S p_value: LOWER value = MORE drift (p < threshold = drifted)
    - Wasserstein distance: HIGHER value = MORE drift (dist > threshold = drifted)

    We read the method from Evidently's own output so we always
    compare in the right direction.
    """
    if "p_value" in method.lower():
        return value < threshold  # low p-value = drift
    else:
        return value > threshold  # high distance = drift


def run_drift_check(
    reference_path: str | Path = None,
    current_path: str | Path = None,
    reference_data: np.ndarray = None,
    current_data: np.ndarray = None,
    save_html: bool = True,
) -> dict:
    """
    Run Evidently drift detection comparing reference vs current data.

    Args:
        reference_path: path to reference .npz file (training data)
        current_path: path to current .npz file (recent data)
        reference_data: 3D numpy array (N, seq_len, features)
        current_data: 3D numpy array (N, seq_len, features)
        save_html: whether to save visual HTML report

    Returns:
        dict with drift results per feature and dataset-level summary
    """
    # Load data
    if reference_data is None:
        if reference_path is None:
            reference_path = PROC / "FD001.npz"
        d = np.load(reference_path)
        reference_data = d["Xtr"]

    if current_data is None:
        if current_path is None:
            current_path = PROC / "FD001.npz"
        d = np.load(current_path)
        current_data = d["Xte"]

    # Convert to DataFrames with summary statistics
    ref_df = _arrays_to_dataframe(reference_data)
    cur_df = _arrays_to_dataframe(current_data)

    logger.info(f"Reference shape: {ref_df.shape}, Current shape: {cur_df.shape}")

    # Run Evidently drift report
    report = Report(metrics=[DataDriftPreset()])
    snapshot = report.run(reference_data=ref_df, current_data=cur_df)

    # Save HTML report for visual inspection
    if save_html:
        report_path = PROC / "drift_report.html"
        snapshot.save_html(str(report_path))
        logger.info(f"Drift report saved to {report_path}")

    # Extract results from the snapshot
    report_dict = snapshot.dump_dict()

    # Parse per-feature drift results
    features = []
    n_drifted = 0
    total = 0

    for key, val in report_dict["metric_results"].items():
        display_name = val.get("display_name", "")

        if display_name.startswith("Value drift for "):
            col_name = display_name.replace("Value drift for ", "")
            value = val.get("value", 0.0)

            # Read method and threshold from Evidently's own params
            params = val["metric_value_location"]["metric"]["params"]
            method = params.get("method", "unknown")
            threshold = params.get("threshold", 0.05)

            drifted = _is_drifted(method, value, threshold)

            features.append({
                "feature": col_name,
                "drift_detected": drifted,
                "drift_score": value,
                "method": method,
                "threshold": threshold,
            })

            total += 1
            if drifted:
                n_drifted += 1

    # Dataset-level drift: >50% of features drifted
    dataset_drift = (n_drifted / total) > 0.5 if total > 0 else False

    result = {
        "timestamp": datetime.utcnow().isoformat(),
        "dataset_drift": dataset_drift,
        "n_drifted_features": n_drifted,
        "total_features": total,
        "drift_ratio": round(n_drifted / total, 3) if total > 0 else 0.0,
        "features": features,
    }

    logger.info(
        f"Drift check: {n_drifted}/{total} features drifted, "
        f"dataset_drift={dataset_drift}"
    )

    return result


def simulate_drift(data: np.ndarray, drift_fraction: float = 0.3,
                   shift_magnitude: float = 2.0) -> np.ndarray:
    """
    Simulate data drift for testing the pipeline.

    Artificially shifts some features to trigger drift detection,
    proving the monitoring pipeline works end-to-end.

    Args:
        data: original data array (N, seq_len, features)
        drift_fraction: fraction of features to shift (0.3 = 30%)
        shift_magnitude: how many standard deviations to shift by

    Returns:
        drifted copy of the data
    """
    drifted = data.copy()
    n_features = data.shape[2]
    n_drift = max(1, int(n_features * drift_fraction))

    rng = np.random.RandomState(42)
    drift_indices = rng.choice(n_features, size=n_drift, replace=False)

    for idx in drift_indices:
        feature_std = data[:, :, idx].std()
        drifted[:, :, idx] += shift_magnitude * feature_std

    logger.info(
        f"Simulated drift: shifted {n_drift}/{n_features} features "
        f"by {shift_magnitude} std devs"
    )

    return drifted


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    d = np.load(PROC / "FD001.npz")

    if "--simulate" in sys.argv:
        print("=== Running with SIMULATED drift ===\n")
        current = simulate_drift(d["Xte"])
        result = run_drift_check(reference_data=d["Xtr"], current_data=current)
    else:
        print("=== Running with original test data (no artificial drift) ===\n")
        result = run_drift_check(reference_data=d["Xtr"], current_data=d["Xte"])

    print(f"Dataset drift detected: {result['dataset_drift']}")
    print(f"Features drifted: {result['n_drifted_features']}/{result['total_features']}")
    print(f"Drift ratio: {result['drift_ratio']:.1%}")

    print("\nPer-feature results:")
    print(f"{'Feature':25s} {'Drifted':>8s} {'Score':>10s} {'Method'}")
    print("-" * 75)
    for f in result["features"]:
        marker = "⚠ YES" if f["drift_detected"] else "  no"
        print(f"{f['feature']:25s} {marker:>8s} {f['drift_score']:10.4f}   "
              f"{f['method']} (threshold={f['threshold']})")
