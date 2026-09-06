import numpy as np
import pandas as pd

from pdm.config import FEATURE_SENSORS, SEQ_LEN


def make_windows(df: pd.DataFrame, seq_len: int = SEQ_LEN):
    X, y = [], []
    for _, g in df.groupby("unit"):
        g = g.sort_values("cycle")
        feats = g[FEATURE_SENSORS].to_numpy(dtype="float32")
        rul = g["RUL"].to_numpy(dtype="float32")
        for i in range(len(g) - seq_len + 1):
            X.append(feats[i:i + seq_len])
            y.append(rul[i + seq_len - 1])
    return np.asarray(X), np.asarray(y)


def last_window_per_unit(df: pd.DataFrame, seq_len: int = SEQ_LEN):
    """For test: take the final `seq_len` cycles of each unit (pad short units)."""
    X, units = [], []
    for u, g in df.groupby("unit"):
        g = g.sort_values("cycle")
        feats = g[FEATURE_SENSORS].to_numpy(dtype="float32")
        if len(feats) < seq_len:
            pad = np.repeat(feats[:1], seq_len - len(feats), axis=0)
            feats = np.vstack([pad, feats])
        X.append(feats[-seq_len:])
        units.append(u)
    return np.asarray(X), np.asarray(units)
