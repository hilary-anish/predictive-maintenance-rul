from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW = PROJECT_ROOT / "data" / "raw"
PROC = PROJECT_ROOT / "data" / "processed"
PROC.mkdir(parents=True, exist_ok=True)

INDEX_COLS = ["unit", "cycle"]
OP_COLS = [f"op_{i}" for i in range(1, 4)]
SENSOR_COLS = [f"s_{i}" for i in range(1, 22)]
ALL_COLS = INDEX_COLS + OP_COLS + SENSOR_COLS

# Sensors that are constant/near-constant on FD001 and add no signal:
DROP_SENSORS = ["s_1", "s_5", "s_6", "s_10", "s_16", "s_18", "s_19"]
FEATURE_SENSORS = [s for s in SENSOR_COLS if s not in DROP_SENSORS]

RUL_CLIP = 125          # piecewise-linear RUL cap (standard for C-MAPSS)
MAX_RUL = RUL_CLIP      # alias used by newer phases
SEQ_LEN = 30            # sliding-window length for the LSTM
WINDOW_SIZE = SEQ_LEN   # alias used by newer phases
RANDOM_STATE = 42

# PyTorch device: use GPU if available, else CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
