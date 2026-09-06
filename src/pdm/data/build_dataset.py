# src/pdm/data/build_dataset.py
import numpy as np

from pdm.config import PROC
from pdm.data.labels import add_rul_train, make_test_targets
from pdm.data.load import load_raw
from pdm.data.preprocess import apply_scaler, fit_scaler
from pdm.data.validate import validate_raw
from pdm.data.windows import last_window_per_unit, make_windows


def main(subset="FD001"):
    train, test, rul = load_raw(subset)
    validate_raw(train)
    validate_raw(test)
    train = add_rul_train(train)
    scaler = fit_scaler(train)
    train_s = apply_scaler(train, scaler)
    test_s  = apply_scaler(test,  scaler)
    Xtr, ytr = make_windows(train_s)
    Xte, units = last_window_per_unit(test_s)
    yte = make_test_targets(test, rul)["RUL"].to_numpy(dtype="float32")
    np.savez_compressed(PROC / f"{subset}.npz", Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte)
    print("saved", PROC / f"{subset}.npz", Xtr.shape, Xte.shape)


if __name__ == "__main__":
    main()
