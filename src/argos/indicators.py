"""Pure causal indicators. No IO, no state, no future data."""

from __future__ import annotations

import pandas as pd


def sma(series: pd.Series, window: int) -> pd.Series:
    """Simple moving average over a trailing window (never centred)."""
    if window < 1:
        raise ValueError("window must be >= 1")
    return series.rolling(window=window, min_periods=window).mean()


def ema(series: pd.Series, window: int) -> pd.Series:
    """Exponential moving average."""
    if window < 1:
        raise ValueError("window must be >= 1")
    return series.ewm(span=window, adjust=False, min_periods=window).mean()


def crossover(fast: pd.Series, slow: pd.Series) -> pd.Series:
    """1.0 while fast is above slow, 0.0 otherwise.

    Values are NaN wherever the inputs are NaN, so warm-up periods produce no
    signal instead of a fabricated one.
    """
    above = (fast > slow).where(fast.notna() & slow.notna())
    return above.astype("float64")


def cross_up(fast: pd.Series, slow: pd.Series) -> pd.Series:
    above = crossover(fast, slow)
    prev = above.shift(1)
    is_up = (above == 1.0) & (prev == 0.0) & above.notna() & prev.notna()
    return is_up.astype("float64").where(is_up)