import pandas as pd
from pdm.config import RUL_CLIP


def add_rul_train(df: pd.DataFrame, clip: int = RUL_CLIP) -> pd.DataFrame:
    df = df.copy()
    max_cycle = df.groupby("unit")["cycle"].transform("max")
    df["RUL"] = (max_cycle - df["cycle"]).clip(upper=clip)
    return df


def make_test_targets(test: pd.DataFrame, rul_true: pd.Series, clip: int = RUL_CLIP) -> pd.DataFrame:
    """RUL at each test unit's LAST cycle = provided RUL (clipped)."""
    last = test.groupby("unit")["cycle"].transform("max") == test["cycle"]
    out = test[last].copy().reset_index(drop=True)
    out["RUL"] = rul_true.clip(upper=clip).values
    return out
