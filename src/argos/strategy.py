"""Strategies: turn bars into target positions (1 = long, 0 = flat).

A strategy only decides *what* to hold. Sizing, costs and execution live
elsewhere.

Temporal semantics
------------------
Every strategy declares how its decision relates to the bar it is computed on,
via the `timing` class attribute:

    SignalTiming.DERIVED
        The decision consumes market data observed up to and including bar t.
        It therefore cannot be executed before the open of bar t+1. The engine
        shifts the series by one bar. This is the normal case.

    SignalTiming.PREDEFINED
        The decision exists *before* the first bar. It consumes no bar data, so
        it may be executed at the open of bar 0. Buy-and-hold is the canonical
        example: "be fully invested" is a constant allocation, not a function of
        prices.

PREDEFINED is not an unchecked promise. The engine verifies that the signal is
a single constant value with no NaN, which makes the class structurally
incapable of leaking future information: a constant cannot depend on data that
does not exist yet.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pandas as pd

from argos.errors import ConfigurationError
from argos.indicators import crossover, sma


class SignalTiming(Enum):
    """How a strategy's decision relates to the bar it is computed on."""

    DERIVED = "derived"
    PREDEFINED = "predefined"


@dataclass(frozen=True)
class SmaCrossover:
    """Long when the fast SMA is above the slow SMA, flat otherwise.

    Parameters are fixed by experimental design and are not tuned.
    """

    fast: int = 20
    slow: int = 50
    timing = SignalTiming.DERIVED

    def __post_init__(self) -> None:
        if self.fast >= self.slow:
            raise ValueError("fast must be strictly less than slow")

    def generate(self, df: pd.DataFrame) -> pd.Series:
        close = df["Close"]
        return crossover(sma(close, self.fast), sma(close, self.slow))


@dataclass(frozen=True)
class Momentum:
    """Long when the close is above the close `lookback` bars ago, flat otherwise.

    Deliberately a pure directional filter: the magnitude of the move is
    ignored and there is no volatility normalisation. See the README for the
    conceptual consequences. `lookback` is fixed at 126 by experimental design.
    """

    lookback: int = 126
    timing = SignalTiming.DERIVED

    def __post_init__(self) -> None:
        if self.lookback < 1:
            raise ValueError("lookback must be >= 1")

    def generate(self, df: pd.DataFrame) -> pd.Series:
        close = df["Close"]
        past = close.shift(self.lookback)
        return (close > past).astype("float64").where(past.notna())


@dataclass(frozen=True)
class BuyAndHold:
    """Fully invested from the first available open.

    PREDEFINED: the allocation is chosen before the data starts, so there is no
    bar to observe and no warm-up. The engine therefore executes it at the open
    of bar 0 rather than one bar later.
    """

    timing = SignalTiming.PREDEFINED

    def generate(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series(1.0, index=df.index, dtype="float64")


@dataclass(frozen=True)
class NeverTrade:
    """Always flat. Useful as a control in tests and experiments."""

    timing = SignalTiming.PREDEFINED

    def generate(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series(0.0, index=df.index, dtype="float64")


@dataclass(frozen=True)
class StrategySpec:
    """A named, instantiable strategy with its display label."""

    key: str
    label: str
    factory: type
    params: dict


def default_specs() -> list[StrategySpec]:
    """The experiment set, with parameters fixed before any run."""
    return [
        StrategySpec("sma", "SMA 20/50", SmaCrossover, {"fast": 20, "slow": 50}),
        StrategySpec("momentum", "Momentum 126", Momentum, {"lookback": 126}),
        StrategySpec("buyhold", "Buy & Hold", BuyAndHold, {}),
    ]


def specs_by_key(keys: list[str] | None = None) -> list[StrategySpec]:
    specs = default_specs()
    if keys is None:
        return specs
    wanted = {k.strip().lower() for k in keys if k.strip()}
    unknown = wanted - {s.key for s in specs}
    if unknown:
        raise ConfigurationError(
            f"unknown strategy key(s): {sorted(unknown)}. "
            f"Available: {', '.join(s.key for s in specs)}"
        )
    if not wanted:
        raise ConfigurationError(
            "no strategies selected; pass at least one of sma, momentum, buyhold"
        )
    return [s for s in specs if s.key in wanted]