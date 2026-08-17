import numpy as np
import mlflow
import mlflow.xgboost
from xgboost import XGBRegressor
from pdm.config import PROC, RANDOM_STATE
from pdm.evaluate.metrics import rmse, mae, phm_score


def window_features(X):                            # X: (N, seq_len, F) -> (N, 4F)
    return np.concatenate([X.mean(1), X.std(1), X[:, -1, :], X[:, -1, :] - X[:, 0, :]], axis=1)


def main(subset="FD001"):
    d = np.load(PROC / f"{subset}.npz")
    Xtr, ytr, Xte, yte = d["Xtr"], d["ytr"], d["Xte"], d["yte"]
    Ftr, Fte = window_features(Xtr), window_features(Xte)

    mlflow.set_experiment("rul-cmapss")

    with mlflow.start_run(run_name="xgb-baseline"):
        params = dict(n_estimators=400, max_depth=5, learning_rate=0.05,
                      subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_STATE)
        mlflow.log_params(params)
        model = XGBRegressor(**params).fit(Ftr, ytr)
        pred = model.predict(Fte)
        mlflow.log_metrics({"rmse": rmse(yte, pred), "mae": mae(yte, pred),
                            "phm_score": phm_score(yte, pred)})
        mlflow.xgboost.log_model(model, name="model",
                                 registered_model_name="rul-xgb")
        print("XGB RMSE", rmse(yte, pred))


if __name__ == "__main__":
    main()
