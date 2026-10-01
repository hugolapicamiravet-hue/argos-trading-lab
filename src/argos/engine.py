"""Bar-by-bar backtest engine.

The loop is explicit on purpose: positions, costs and per-execution bookkeeping
are stateful and cannot be expressed as vector arithmetic without losing them.

Look-ahead protection
---------------------
Protection is derived from each strategy's declared temporal semantics rather
than applied blindly to every strategy:

* `DERIVED` strategies compute their signal from data observed up to and
  including bar t, so the engine shifts the series by one bar and executes at
  the open of t+1.
* `PREDEFINED` strategies decide before the data starts, so there is nothing to
  wait for and the engine executes at the open of bar 0.

    Either way, no strategy can execute using information that was unavailable when
    the decision was made.

    The result also carries the per-bar cash curve. It is redundant with the
    portfolio and the equity curve, which is exactly why it is useful: it makes
    the accounting identity `equity == cash + units * close` checkable on every
    single bar rather than only on the last one.
    """

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from argos.errors import BacktestError, InvalidSignals
from argos.portfolio import Portfolio
from argos.strategy import SignalTiming


@dataclass(frozen=True)
class BacktestResult:
    """Everything one simulation produced."""

    equity_curve: pd.Series
    portfolio: Portfolio
    signals: pd.Series
    open_positions: pd.Series
    timing: SignalTiming
    strategy_name: str
    first_actionable_bar: int | None
    commission_pct: float
    slippage_pct: float
    cash_curve: pd.Series | None = None

    @property
    def is_actionable(self) -> bool:
        """False when the dataset was too short for the strategy to ever trade."""
        return self.first_actionable_bar is not None


def _timing_of(strategy) -> SignalTiming:
    timing = getattr(strategy, "timing", None)
    if not isinstance(timing, SignalTiming):
        raise InvalidSignals(
            f"{type(strategy).__name__} must declare a SignalTiming as its "
            f"'timing' attribute; got {timing!r}"
        )
    return timing


def _check_signals(signals, df: pd.DataFrame, strategy) -> pd.Series:
    """Validate and normalise a strategy's output before it can drive trades."""
    name = type(strategy).__name__

    if not isinstance(signals, pd.Series):
        raise InvalidSignals(
            f"{name}.generate must return a pandas Series, got {type(signals).__name__}"
        )
    if len(signals) != len(df):
        raise InvalidSignals(
            f"{name}.generate returned {len(signals)} signals for {len(df)} bars; "
            "signals must have exactly one value per bar"
        )
    if not signals.index.equals(df.index):
        raise InvalidSignals(
            f"{name}.generate returned signals on a different index than the price data. "
            "Signal timestamps must match bar timestamps exactly."
        )
    try:
        return signals.astype("float64")
    except (TypeError, ValueError) as exc:
        raise InvalidSignals(f"{name}.generate produced non-numeric signals: {exc}") from exc


def _validate_predefined(signals: pd.Series, strategy) -> None:
    """A PREDEFINED strategy must emit one constant value and no NaN.

    This is what makes the declaration safe: a constant signal cannot depend on
    information that did not exist when the decision was taken.
    """
    name = type(strategy).__name__
    if signals.isna().any():
        raise InvalidSignals(
            f"{name} declares SignalTiming.PREDEFINED but produced {int(signals.isna().sum())} "
            "NaN signals. A predefined decision is a constant and cannot have missing values."
        )
    unique = signals.unique()
    if len(unique) != 1:
        raise InvalidSignals(
            f"{name} declares SignalTiming.PREDEFINED but its signal changes over time "
            f"(values: {sorted(unique)[:5]}). A decision taken before the data starts cannot "
            "depend on the data; declare SignalTiming.DERIVED instead."
        )


def _first_valid_position(signals: pd.Series) -> int | None:
    valid = signals.notna().to_numpy()
    if not valid.any():
        return None
    return int(valid.argmax())


def run_backtest(
    df: pd.DataFrame,
    strategy,
    initial_capital: float = 10_000.0,
    commission_pct: float = 0.0,
    slippage_pct: float = 0.0,
    strategy_name: str | None = None,
) -> BacktestResult:
    """Simulate `strategy` over `df` with a single long position at a time.

    Position sizing is fixed at 100% of available cash. There is no minimum hold
    period, so a position may be closed and reopened on consecutive bars. The
    last position is never force-closed: final equity is marked to market at the
    last close.
    """
    if df.empty:
        raise BacktestError("cannot run a backtest on an empty dataset")
    if initial_capital <= 0:
        raise BacktestError("initial_capital must be positive")

    timing = _timing_of(strategy)
    raw_signals = _check_signals(strategy.generate(df), df, strategy)
    name = strategy_name or type(strategy).__name__

    if timing is SignalTiming.PREDEFINED:
        _validate_predefined(raw_signals, strategy)
        executed = raw_signals
        first_actionable = 0 if len(df) else None
    else:
        # The look-ahead barrier: bar t's signal acts on bar t+1's open.
        executed = raw_signals.shift(1).fillna(0.0)
        first_defined = _first_valid_position(raw_signals)
        # The signal at `first_defined` is executed on the following bar.
        first_actionable = None if first_defined is None else first_defined + 1

    portfolio = Portfolio(
        initial_capital=initial_capital,
        commission_pct=commission_pct,
        slippage_pct=slippage_pct,
    )

    opens = df["Open"].to_numpy(dtype="float64")
    closes = df["Close"].to_numpy(dtype="float64")
    targets = executed.to_numpy(dtype="float64")

    equity: list[float] = []
    held: list[float] = []
    cash: list[float] = []

    for i in range(len(df)):
        target = targets[i]
        if target > 0 and not portfolio.in_position:
            portfolio.buy(float(opens[i]))
        elif target <= 0 and portfolio.in_position:
            portfolio.sell(float(opens[i]))

        equity.append(portfolio.equity_at(float(closes[i])))
        held.append(portfolio.units)
        cash.append(portfolio.cash)

    return BacktestResult(
        equity_curve=pd.Series(equity, index=df.index, name="equity", dtype="float64"),
        portfolio=portfolio,
        signals=raw_signals,
        open_positions=pd.Series(held, index=df.index, name="units", dtype="float64"),
        timing=timing,
        strategy_name=name,
        first_actionable_bar=first_actionable,
        commission_pct=commission_pct,
        slippage_pct=slippage_pct,
        cash_curve=pd.Series(cash, index=df.index, name="cash", dtype="float64"),
    )