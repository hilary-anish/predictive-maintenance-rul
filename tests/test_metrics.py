"""
Test evaluation metrics.

Why test metrics? If your RMSE calculation is wrong, your entire
model comparison is invalid. These are the numbers that go in your
README and CV — they must be correct.
"""
import numpy as np
from pdm.evaluate.metrics import rmse, mae, phm_score


def test_rmse_perfect():
    """Perfect predictions should give RMSE = 0."""
    y = np.array([10.0, 20.0, 30.0])
    assert rmse(y, y) == 0.0


def test_rmse_known_value():
    """RMSE of [0,0] vs [3,4] = sqrt((9+16)/2) = sqrt(12.5)."""
    y = np.array([0.0, 0.0])
    yhat = np.array([3.0, 4.0])
    expected = np.sqrt(12.5)
    assert abs(rmse(y, yhat) - expected) < 1e-6


def test_mae_perfect():
    """Perfect predictions should give MAE = 0."""
    y = np.array([10.0, 20.0, 30.0])
    assert mae(y, y) == 0.0


def test_mae_known_value():
    """MAE of [0,0] vs [3,4] = (3+4)/2 = 3.5."""
    y = np.array([0.0, 0.0])
    yhat = np.array([3.0, 4.0])
    assert abs(mae(y, yhat) - 3.5) < 1e-6


def test_phm_score_perfect():
    """Perfect predictions should give PHM score = 0."""
    y = np.array([10.0, 20.0])
    assert phm_score(y, y) == 0.0


def test_phm_score_asymmetry():
    """
    PHM score should penalize late predictions (positive error)
    more heavily than early predictions (negative error).

    Late = predicted RUL > actual RUL = engine fails before expected
    Early = predicted RUL < actual RUL = unnecessary early maintenance

    Late is worse because it means missed failures.
    """
    y = np.array([50.0])
    early = np.array([40.0])   # predicted 40, actual 50 → d = -10
    late = np.array([60.0])    # predicted 60, actual 50 → d = +10

    score_early = phm_score(y, early)
    score_late = phm_score(y, late)

    # Late prediction should have higher (worse) score
    assert score_late > score_early
