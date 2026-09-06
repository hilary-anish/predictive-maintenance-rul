"""
Test RUL label generation.

Why test labels? If RUL labels are wrong, every model trained on them
is wrong. This is the foundation — get it right and verify it.
"""
import pandas as pd
from pdm.data.labels import add_rul_train, make_test_targets


def test_add_rul_train_basic():
    """RUL at the last cycle of a unit should be 0."""
    df = pd.DataFrame({
        "unit": [1, 1, 1, 2, 2],
        "cycle": [1, 2, 3, 1, 2],
    })
    result = add_rul_train(df)
    # Unit 1 has 3 cycles: RUL = [2, 1, 0]
    assert result[result["unit"] == 1]["RUL"].tolist() == [2, 1, 0]
    # Unit 2 has 2 cycles: RUL = [1, 0]
    assert result[result["unit"] == 2]["RUL"].tolist() == [1, 0]


def test_add_rul_train_clipping():
    """RUL should be clipped at the clip value."""
    df = pd.DataFrame({
        "unit": [1] * 200,
        "cycle": list(range(1, 201)),
    })
    result = add_rul_train(df, clip=125)
    # First cycle: raw RUL = 199, clipped to 125
    assert result.iloc[0]["RUL"] == 125
    # Last cycle: RUL = 0 (never clipped)
    assert result.iloc[-1]["RUL"] == 0


def test_test_targets():
    """test_targets should return one row per unit with correct RUL."""
    test_df = pd.DataFrame({
        "unit": [1, 1, 1, 2, 2],
        "cycle": [1, 2, 3, 1, 2],
    })
    rul_true = pd.Series([50, 80])
    result = make_test_targets(test_df, rul_true, clip=125)
    assert len(result) == 2  # one row per unit
    assert result.iloc[0]["RUL"] == 50
    assert result.iloc[1]["RUL"] == 80


def test_test_targets_clipping():
    """RUL values above clip should be capped."""
    test_df = pd.DataFrame({
        "unit": [1, 1],
        "cycle": [1, 2],
    })
    rul_true = pd.Series([200])  # above clip
    result = make_test_targets(test_df, rul_true, clip=125)
    assert result.iloc[0]["RUL"] == 125
