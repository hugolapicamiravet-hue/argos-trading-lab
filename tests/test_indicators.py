from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from argos.indicators import cross_up, crossover, ema, sma


def test_sma_matches_manual_mean() -> None:
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    out = sma(s, 3)
    assert out.iloc[:2].isna().all()
    assert out.iloc[2] == pytest.approx(2.0)
    assert out.iloc[4] == pytest.approx(4.0)


def test_sma_rejects_zero_window() -> None:
    with pytest.raises(ValueError):
        sma(pd.Series([1.0]), 0)


def test_ema_is_causal_and_bounded() -> None:
    s = pd.Series(range(1, 50), dtype="float64")
    out = ema(s, 10)
    assert out.iloc[:9].isna().all()
    assert out.iloc[9] <= s.iloc[9]
    assert out.iloc[-1] < s.iloc[-1]


def test_crossover_is_one_when_fast_above_slow() -> None:
    out = crossover(pd.Series([1.0, 3.0, 5.0]), pd.Series([2.0, 2.0, 2.0]))
    assert list(out.astype(bool)) == [False, True, True]


def test_crossover_is_nan_during_warmup() -> None:
    out = crossover(pd.Series([np.nan, 1.0]), pd.Series([1.0, 1.0]))
    assert pd.isna(out.iloc[0])


def test_cross_up_marks_only_the_transition() -> None:
    fast = pd.Series([1.0, 3.0, 4.0, 1.0, 5.0])
    slow = pd.Series([2.0] * 5)
    up = cross_up(fast, slow).dropna()
    assert list(up.index) == [1, 4]
    assert list(up) == [1.0, 1.0]