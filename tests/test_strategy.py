from __future__ import annotations

import pandas as pd
import pytest

from argos.strategy import BuyAndHold, SmaCrossover


def _prices(values: list[float]) -> pd.DataFrame:
    index = pd.date_range("2020-01-01", periods=len(values), freq="D")
    return pd.DataFrame({"Close": values}, index=index)


def test_crossover_requires_fast_below_slow() -> None:
    with pytest.raises(ValueError):
        SmaCrossover(fast=50, slow=20)
    with pytest.raises(ValueError):
        SmaCrossover(fast=20, slow=20)


def test_no_signal_before_slow_window_fills() -> None:
    df = _prices(list(range(1, 61)))
    out = SmaCrossover(fast=5, slow=20).generate(df)
    assert out.iloc[:19].isna().all()
    assert out.notna().sum() == 41


def test_signal_is_one_when_uptrend_and_zero_when_downtrend() -> None:
    up = _prices(list(range(1, 61)))
    assert SmaCrossover(fast=5, slow=20).generate(up).iloc[-1] == 1.0

    down = _prices(list(range(60, 0, -1)))
    assert SmaCrossover(fast=5, slow=20).generate(down).iloc[-1] == 0.0


def test_buy_and_hold_is_long_from_first_valid_bar() -> None:
    df = _prices([1.0, 2.0, 3.0])
    out = BuyAndHold().generate(df)
    assert list(out) == [1.0, 1.0, 1.0]