"""
Conformal prediction for calibrated uncertainty intervals.

Why conformal prediction for RUL:
- Maintenance planners need intervals, not point estimates
- Conformal gives a mathematical GUARANTEE on coverage (e.g., 90%)
- Distribution-free: no assumptions about error distribution shape
- Model-agnostic: works with LSTM, TCN, XGBoost, anything
- Simple to implement but powerful story for interviews

Theory (simplified):
    1. Compute absolute residuals on a held-out calibration set:
       scores[i] = |y_true[i] - y_pred[i]|
    2. Sort scores and take the (1-alpha) quantile with finite-sample
       correction: q = quantile(scores, ceil((n+1)*(1-alpha))/n)
    3. For a new prediction y_hat, the interval is:
       [y_hat - q, y_hat + q]

    The finite-sample correction (n+1)/n comes from Vovk et al. (2005)
    and ensures exact coverage even with small calibration sets.

Reference:
    Vovk, Gammerman, Shafer. "Algorithmic Learning in a Random World" (2005)
"""
import numpy as np


class ConformalPredictor:
    """
    Split conformal predictor for regression.

    Usage:
        cp = ConformalPredictor(alpha=0.10)  # 90% coverage
        cp.calibrate(y_cal_true, y_cal_pred)
        intervals = cp.predict(y_new_pred)
        # intervals["lower"], intervals["upper"], intervals["width"]
    """

    def __init__(self, alpha: float = 0.10):
        """
        Args:
            alpha: miscoverage rate. 0.10 = 90% prediction intervals.
                   0.05 = 95% intervals (wider but more conservative).
        """
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        self.alpha = alpha
        self.q_hat = None  # will be set during calibration
        self.scores = None  # stored for diagnostics

    def calibrate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calibrate on held-out data.

        Args:
            y_true: ground truth RUL values (calibration set)
            y_pred: model predictions on the same calibration set

        Returns:
            q_hat: the conformal quantile (half-width of intervals)

        The key insight: we're measuring "how wrong is our model?"
        on data it hasn't seen. The distribution of these errors
        tells us how wide our intervals need to be.
        """
        y_true = np.asarray(y_true, dtype=np.float64)
        y_pred = np.asarray(y_pred, dtype=np.float64)

        if len(y_true) != len(y_pred):
            raise ValueError(
                f"Length mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}"
            )

        # Nonconformity scores = absolute residuals
        # Why absolute? We want symmetric intervals.
        # You could use signed residuals for asymmetric intervals,
        # but symmetric is standard and simpler to explain.
        self.scores = np.abs(y_true - y_pred)

        # Finite-sample corrected quantile level
        # Standard quantile at (1-alpha) would give slightly less than
        # (1-alpha) coverage. The (n+1)/n correction fixes this.
        n = len(self.scores)
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        q_level = min(q_level, 1.0)  # cap at 1.0

        self.q_hat = float(np.quantile(self.scores, q_level))
        return self.q_hat

    def predict(self, y_pred: np.ndarray) -> dict:
        """
        Generate prediction intervals.

        Args:
            y_pred: point predictions from the model

        Returns:
            dict with "lower", "upper", "point", "width" arrays
            Lower bounds are clipped at 0 (RUL can't be negative).
        """
        if self.q_hat is None:
            raise RuntimeError("Call calibrate() before predict()")

        y_pred = np.asarray(y_pred, dtype=np.float64)

        lower = np.maximum(y_pred - self.q_hat, 0.0)  # RUL >= 0
        upper = y_pred + self.q_hat

        return {
            "point": y_pred,
            "lower": lower,
            "upper": upper,
            "width": upper - lower,
            "q_hat": self.q_hat,
        }

    def evaluate_coverage(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """
        Evaluate calibration quality on test data.

        Returns coverage (should be >= 1-alpha) and mean interval width.
        This is what you show in the README and model card.

        Good calibration = coverage close to (1-alpha), not much higher.
        If coverage >> (1-alpha), intervals are too wide (conservative).
        If coverage < (1-alpha), something went wrong.
        """
        if self.q_hat is None:
            raise RuntimeError("Call calibrate() before evaluate_coverage()")

        y_true = np.asarray(y_true, dtype=np.float64)
        intervals = self.predict(y_pred)

        covered = (y_true >= intervals["lower"]) & (y_true <= intervals["upper"])
        coverage = float(covered.mean())
        mean_width = float(intervals["width"].mean())

        return {
            "target_coverage": 1 - self.alpha,
            "empirical_coverage": coverage,
            "coverage_gap": coverage - (1 - self.alpha),
            "mean_interval_width": mean_width,
            "median_interval_width": float(np.median(intervals["width"])),
            "q_hat": self.q_hat,
            "n_test": len(y_true),
        }
