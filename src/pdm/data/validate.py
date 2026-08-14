import pandera as pa
from pandera import Column, Check
from pdm.config import SENSOR_COLS


raw_schema = pa.DataFrameSchema(
    {
        "unit":  Column(int, Check.ge(1)),
        "cycle": Column(int, Check.ge(1)),
        **{s: Column(float, nullable=False) for s in SENSOR_COLS},
    },
    strict=False,
    coerce=True,
)


def validate_raw(df):
    return raw_schema.validate(df)
