from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW = PROJECT_ROOT / "data" / "raw"
PROC = PROJECT_ROOT / "data" / "processed"; PROC.mkdir(parents=True, exist_ok=True)

INDEX_COLS = ["unit", "cycle"]
OP_COLS = [f"op_{i}" for i in range(1, 4)]
SENSOR_COLS = [f"s_{i}" for i in range(1, 22)]
ALL_COLS = INDEX_COLS + OP_COLS + SENSOR_COLS

# Sensors that are constant/near-constant on FD001 and add no signal:
DROP_SENSORS = ["s_1", "s_5", "s_6", "s_10", "s_16", "s_18", "s_19"]
FEATURE_SENSORS = [s for s in SENSOR_COLS if s not in DROP_SENSORS]

RUL_CLIP = 125          # piecewise-linear RUL cap (standard for C-MAPSS)
SEQ_LEN = 30            # sliding-window length for the LSTM
RANDOM_STATE = 42