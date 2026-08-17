import numpy as np


def rmse(y, yhat): return float(np.sqrt(np.mean((y - yhat) ** 2)))


def mae(y, yhat): return float(np.mean(np.abs(y - yhat)))


def phm_score(y, yhat):
    d = yhat - y
    return float(np.sum(np.where(d < 0, np.exp(-d/13) - 1, np.exp(d/10) - 1)))
