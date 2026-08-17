import torch
import torch.nn as nn


class RULLSTM(nn.Module):
    def __init__(self, n_features, hidden=64, layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(),
                                  self.drop, nn.Linear(32, 1))

    def forward(self, x):
        out, _ = self.lstm(x)                       # (B, T, H)
        return self.head(self.drop(out[:, -1, :])).squeeze(-1)
