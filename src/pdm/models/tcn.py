"""
Temporal Convolutional Network (TCN) for RUL prediction.

Why TCN alongside LSTM:
- Parallelizable (faster training than recurrent models)
- Fixed receptive field (no vanishing gradient over long sequences)
- Comparable or better performance on many time-series tasks
- Shows architectural breadth in your portfolio

Architecture overview:
    Input (B, T, F) -> transpose to (B, F, T) for Conv1d
    -> TemporalBlock (dilation=1)
    -> TemporalBlock (dilation=2)
    -> TemporalBlock (dilation=4)
    -> Take last timestep -> Linear -> RUL prediction

Each TemporalBlock = two causal convolutions + residual connection + dropout.
Dilations grow exponentially so the receptive field covers the full sequence
without needing a huge kernel.
"""
import torch
import torch.nn as nn


class CausalConv1d(nn.Module):
    """
    Causal convolution: output at time t depends only on inputs <= t.

    Standard Conv1d with 'same' padding sees future timesteps — that's
    data leakage for time-series. Causal padding adds zeros only on the
    left side, then trims the right to enforce causality.

    Why not just use padding='same'?
    Because 'same' centers the padding, letting the convolution peek
    at future values — a subtle but critical bug in time-series models.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, dilation: int = 1):
        super().__init__()
        # Left-pad by (kernel_size - 1) * dilation to keep output length = input length
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=self.padding, dilation=dilation
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        if self.padding > 0:
            out = out[:, :, :-self.padding]  # chop off right (future) padding
        return out


class TemporalBlock(nn.Module):
    """
    Two causal convolutions + residual connection + dropout.

    The residual (skip) connection is critical:
    - Allows gradients to flow directly through the network
    - Lets the block learn "corrections" to the identity mapping
    - Same idea as ResNet — proven to help deeper networks train

    If input channels != output channels, a 1x1 convolution adapts
    the residual path to match dimensions.
    """

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int,
                 dilation: int, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            CausalConv1d(in_ch, out_ch, kernel_size, dilation),
            nn.ReLU(),
            nn.Dropout(dropout),
            CausalConv1d(out_ch, out_ch, kernel_size, dilation),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        # 1x1 conv to match channel dimensions for the skip connection
        self.downsample = (
            nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        )
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.net(x) + self.downsample(x))


class RULTCN(nn.Module):
    """
    TCN for Remaining Useful Life prediction.

    Architecture decisions explained:
    - n_channels=[64, 64, 32]: 3 blocks. First two at 64 capture complex
      patterns, last at 32 compresses before the linear head. Fewer params
      than LSTM with comparable performance.
    - kernel_size=3: small kernel, but dilations (1, 2, 4) give a receptive
      field of 3 + 6 + 12 = 21 timesteps. With seq_len=30 this covers most
      of the input window.
    - dropout=0.3: same as LSTM for fair comparison. Also enables MC-Dropout
      uncertainty estimation (same technique as LSTM — keep dropout on at
      inference, run N forward passes, measure variance).

    Args:
        n_features: number of sensor channels (14 for C-MAPSS FD001)
        n_channels: list of channel sizes per temporal block
        kernel_size: convolution kernel size
        dropout: dropout rate (also used for MC-Dropout uncertainty)
    """

    def __init__(self, n_features: int, n_channels: list[int] | None = None,
                 kernel_size: int = 3, dropout: float = 0.3):
        super().__init__()
        if n_channels is None:
            n_channels = [64, 64, 32]

        layers = []
        for i, out_ch in enumerate(n_channels):
            in_ch = n_features if i == 0 else n_channels[i - 1]
            dilation = 2 ** i  # exponential dilation: 1, 2, 4, ...
            layers.append(TemporalBlock(in_ch, out_ch, kernel_size,
                                        dilation, dropout))

        self.network = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(n_channels[-1], 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (Batch, Time, Features) — standard for time-series
        # Conv1d expects: (Batch, Channels, Length) — so we transpose
        out = self.network(x.transpose(1, 2))
        # Take the LAST timestep's output (the most recent prediction)
        return self.head(out[:, :, -1]).squeeze(-1)
