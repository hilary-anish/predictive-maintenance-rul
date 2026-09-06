"""
Test sliding window generation.

Why test windows? Wrong window shapes cause silent model errors —
the model trains on garbage and produces garbage predictions.
Shape mismatches are the #1 bug in sequence model pipelines.
"""
import numpy as np
import pandas as pd

from pdm.config import FEATURE_SENSORS
from pdm.data.windows import last_window_per_unit, make_windows


def _make_dummy_df(n_units=2, cycles_per_unit=50):
    """Create a minimal DataFrame mimicking preprocessed C-MAPSS data."""
    rows = []
    for u in range(1, n_units + 1):
        for c in range(1, cycles_per_unit + 1):
            row = {"unit": u, "cycle": c, "RUL": cycles_per_unit - c}
            for s in FEATURE_SENSORS:
                row[s] = np.random.rand()
            rows.append(row)
    return pd.DataFrame(rows)


def test_make_windows_shape():
    """Output shape should be (N, seq_len, n_features)."""
    df = _make_dummy_df(n_units=2, cycles_per_unit=50)
    X, y = make_windows(df, seq_len=30)
    assert X.ndim == 3
    assert X.shape[1] == 30            # seq_len
    assert X.shape[2] == len(FEATURE_SENSORS)  # 14 features
    assert len(X) == len(y)            # one label per window


def test_make_windows_count():
    """Number of windows = sum of (cycles - seq_len + 1) per unit."""
    df = _make_dummy_df(n_units=2, cycles_per_unit=50)
    X, y = make_windows(df, seq_len=30)
    # Each unit with 50 cycles produces 50-30+1 = 21 windows
    assert len(X) == 2 * 21


def test_make_windows_label_alignment():
    """The label for each window should be the RUL at the last timestep."""
    df = _make_dummy_df(n_units=1, cycles_per_unit=40)
    X, y = make_windows(df, seq_len=30)
    # First window covers cycles 1-30, label = RUL at cycle 30
    assert y[0] == 40 - 30  # = 10


def test_last_window_per_unit_shape():
    """last_window_per_unit should return one window per unit."""
    df = _make_dummy_df(n_units=3, cycles_per_unit=50)
    X, units = last_window_per_unit(df, seq_len=30)
    assert X.shape == (3, 30, len(FEATURE_SENSORS))
    assert len(units) == 3


def test_last_window_short_unit_padding():
    """Units with fewer cycles than seq_len should be padded."""
    df = _make_dummy_df(n_units=1, cycles_per_unit=10)
    X, units = last_window_per_unit(df, seq_len=30)
    assert X.shape == (1, 30, len(FEATURE_SENSORS))
    # First 20 rows should be padding (repeated first row)
    assert np.allclose(X[0, 0, :], X[0, 19, :])
