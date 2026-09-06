import numpy as np
import torch
import mlflow
import mlflow.pytorch
from torch.utils.data import TensorDataset, DataLoader
from pdm.config import PROC, FEATURE_SENSORS, RANDOM_STATE
from pdm.models.lstm import RULLSTM
from pdm.evaluate.metrics import rmse, mae, phm_score


def main(subset="FD001", epochs=40, bs=256, lr=1e-3):
    torch.manual_seed(RANDOM_STATE)
    d = np.load(PROC / f"{subset}.npz")
    Xtr, ytr, Xte, yte = (torch.tensor(d[k], dtype=torch.float32) for k in ("Xtr", "ytr", "Xte", "yte"))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dl = DataLoader(TensorDataset(Xtr, ytr), batch_size=bs, shuffle=True)
    model = RULLSTM(len(FEATURE_SENSORS)).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    mlflow.set_experiment("rul-cmapss")

    with mlflow.start_run(run_name="lstm"):
        mlflow.log_params(dict(model="LSTM", epochs=epochs, bs=bs, lr=lr, seq_len=Xtr.shape[1]))
        for ep in range(epochs):
            model.train()
            for xb, yb in dl:
                xb, yb = xb.to(dev), yb.to(dev)
                opt.zero_grad(); loss = loss_fn(model(xb), yb); loss.backward()
                opt.step()
        model.eval()
        with torch.no_grad():
            pred = model(Xte.to(dev)).cpu().numpy()
        mlflow.log_metrics({"rmse": rmse(yte.numpy(), pred), "mae": mae(yte.numpy(), pred),
                            "phm_score": phm_score(yte.numpy(), pred)})
        torch.save(model.state_dict(), PROC / "lstm.pt")
        mlflow.log_artifact(str(PROC / "lstm.pt"))
        model.cpu()
        mlflow.pytorch.log_model(model, name="model", registered_model_name="rul-lstm",
                                 input_example=Xte[:1].numpy(),
                                 serialization_format="pickle")
        print("LSTM RMSE", rmse(yte.numpy(), pred))


if __name__ == "__main__":
    main()
