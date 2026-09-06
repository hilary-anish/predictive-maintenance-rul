import pandas as pd

from pdm.config import ALL_COLS, RAW


def load_raw(subset: str = "FD001") -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    train = pd.read_csv(RAW / f"train_{subset}.txt", sep=r"\s+", header=None).iloc[:, :26]
    test  = pd.read_csv(RAW / f"test_{subset}.txt",  sep=r"\s+", header=None).iloc[:, :26]
    rul   = pd.read_csv(RAW / f"RUL_{subset}.txt",   sep=r"\s+", header=None)[0]
    train.columns = test.columns = ALL_COLS
    return train, test, rul
