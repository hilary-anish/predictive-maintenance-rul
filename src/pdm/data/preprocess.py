import joblib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from pdm.config import FEATURE_SENSORS, PROC


def fit_scaler(train: pd.DataFrame) -> MinMaxScaler:
    scaler = MinMaxScaler().fit(train[FEATURE_SENSORS])
    joblib.dump(scaler, PROC / "scaler.joblib")
    return scaler


def apply_scaler(df: pd.DataFrame, scaler: MinMaxScaler) -> pd.DataFrame:
    df = df.copy()
    df[FEATURE_SENSORS] = scaler.transform(df[FEATURE_SENSORS])
    return df
