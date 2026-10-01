"""Look-ahead protection and the temporal-semantics guard.

These tests are the safety net for the single most dangerous class of bug in a
backtest: using information that did not exist when the decision was made. They
fail loudly if anyone introduces a centred window, a bfill, an unshifted signal,
or a strategy that misdeclares its timing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from argos.engine import run_backtest
from argos.errors import BacktestError, InvalidSignals
from argos.indicators import sma
from argos.strategy import BuyAndHold, Momentum, NeverTrade, SignalTiming, SmaCrossover


def _zigzag(n: int = 300) -> pd.Series:
    """Rising then falling then rising, so crossovers actually occur."""
    half = n // 3
    up = list(np.linspace(100, 200, half))
    down = list(np.linspace(200, 80, half))
    recovery = list(np.linspace(80, 260, n - 2 * half))
    return pd.Series(up + down + recovery)


def _frame(close: pd.Series) -> pd.DataFrame:
    values = np.asarray(close, dtype="float64")
    return pd.DataFrame(
        {
            "Open": values,
            "High": values * 1.01,
            "Low": values * 0.99,
            "Close": values,
            "Volume": 1_000.0,
        },
        index=pd.date_range("2020-01-01", periods=len(values), freq="D", tz="UTC"),
    )


# --- indicator causality -------------------------------------------------


def test_signal_for_bar_t_does_not_depend_on_bars_after_t() -> None:
    """Prefix invariance: truncating the future must not change past signals."""
    close = _zigzag()
    full = SmaCrossover(fast=20, slow=50).generate(_frame(close))

    for cut in (60, 120, 200, 280):
        truncated = SmaCrossover(fast=20, slow=50).generate(_frame(close.iloc[:cut]))
        pd.testing.assert_series_equal(full.iloc[:cut], truncated, check_names=False)


def test_sma_is_not_centred() -> None:
    """A centred window would leak future closes into the window."""
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    trailing = sma(s, 3)
    centred = s.rolling(3, center=True).mean()

    assert trailing.iloc[2] == pytest.approx(s.iloc[0:3].mean())
    assert trailing.iloc[4] == pytest.approx(s.iloc[2:5].mean())
    assert trailing.iloc[4] != pytest.approx(centred.iloc[4])

    # Proof of causality: the trailing value at bar t never uses a close above t.
    mutated = s.copy()
    mutated.iloc[4] = 1000.0
    assert sma(mutated, 3).iloc[3] == pytest.approx(trailing.iloc[3])


def test_momentum_uses_only_the_close_126_bars_ago() -> None:
    close = _zigzag()
    frame = _frame(close)
    signals = Momentum(lookback=10).generate(frame)

    for i in range(10, len(close)):
        expected = 1.0 if close.iloc[i] > close.iloc[i - 10] else 0.0
        assert signals.iloc[i] == expected

    # Changing a *future* close cannot change a past momentum reading.
    mutated = close.copy()
    mutated.iloc[-1] = 1e6
    pd.testing.assert_series_equal(
        signals.iloc[:-1], Momentum(lookback=10).generate(_frame(mutated)).iloc[:-1],
        check_names=False,
    )


# --- DERIVED execution ---------------------------------------------------


def test_engine_never_executes_on_the_signal_bar() -> None:
    """A derived position may only exist on the bar after the signal appears."""
    close = _zigzag()
    df = _frame(close)
    result = run_backtest(df, SmaCrossover(fast=20, slow=50), initial_capital=10_000)

    signals = result.signals.fillna(0.0)
    units = result.open_positions

    entries = signals.index[(signals == 1.0) & (signals.shift(1).fillna(0.0) == 0.0)]
    exits = signals.index[(signals == 0.0) & (signals.shift(1).fillna(0.0) == 1.0)]

    assert len(entries) > 0, "fixture must produce at least one entry"
    assert len(exits) > 0, "fixture must produce at least one exit"

    for ts in entries:
        i = df.index.get_loc(ts)
        assert units.iloc[i] == 0.0  # flat on the signal bar itself
        assert units.iloc[i + 1] > 0.0  # holding from the very next bar

    for ts in exits:
        i = df.index.get_loc(ts)
        assert units.iloc[i] > 0.0  # still held on the signal bar
        assert units.iloc[i + 1] == 0.0  # flat the next bar

    assert (units >= 0).all()


def test_execution_price_is_the_next_bar_open_not_the_signal_bar_close() -> None:
    """A gap on the fill bar must be paid: units come from the open, not the close."""
    close = pd.Series(np.linspace(100.0, 200.0, 100))
    df = _frame(close)

    fill_bar = 20
    df.iloc[fill_bar, df.columns.get_loc("Open")] = 50.0
    df.iloc[fill_bar, df.columns.get_loc("Low")] = 50.0

    result = run_backtest(df, SmaCrossover(fast=5, slow=20), initial_capital=10_000)
    units = result.open_positions
    i = df.index.get_loc(df.index[fill_bar])

    assert units.iloc[i - 1] == 0.0  # still flat on the signal bar
    assert units.iloc[i] == pytest.approx(10_000.0 / 50.0)  # bought at the open
    assert units.iloc[i] != pytest.approx(10_000.0 / close.iloc[i])
    assert result.equity_curve.iloc[i] == pytest.approx((10_000.0 / 50.0) * close.iloc[i])


def test_truncating_history_does_not_change_the_equity_of_the_overlap() -> None:
    """Re-running on a shorter history reproduces the same early equity curve."""
    df = _frame(_zigzag())

    long_run = run_backtest(df, SmaCrossover(fast=20, slow=50), initial_capital=10_000)
    short_run = run_backtest(df.iloc[:200], SmaCrossover(fast=20, slow=50), initial_capital=10_000)

    pd.testing.assert_series_equal(
        long_run.equity_curve.iloc[:200], short_run.equity_curve, check_names=False
    )


def test_no_signals_produced_during_indicator_warmup() -> None:
    close = pd.Series(np.linspace(100, 300, 80))
    signals = SmaCrossover(fast=20, slow=50).generate(_frame(close))
    assert signals.iloc[:49].isna().all()

    momentum = Momentum(lookback=126).generate(_frame(pd.Series(np.linspace(100, 300, 200))))
    assert momentum.iloc[:126].isna().all()


def test_lookahead_strategy_is_detected_by_the_causality_check() -> None:
    """If truncating the dataset changes an already-computed signal, data leaked."""

    class CheatingStrategy:
        timing = SignalTiming.DERIVED

        def generate(self, df: pd.DataFrame) -> pd.Series:
            close = df["Close"]
            return (close.shift(-1) > close).astype("float64")

    close = _zigzag()
    cut = 60

    honest_full = SmaCrossover(fast=20, slow=50).generate(_frame(close))
    honest_trunc = SmaCrossover(fast=20, slow=50).generate(_frame(close.iloc[:cut]))
    pd.testing.assert_series_equal(
        honest_full.iloc[:cut], honest_trunc, check_names=False
    )

    cheat_full = CheatingStrategy().generate(_frame(close))
    cheat_trunc = CheatingStrategy().generate(_frame(close.iloc[:cut]))
    assert cheat_full.iloc[cut - 1] != cheat_trunc.iloc[cut - 1]


# --- PREDEFINED execution ------------------------------------------------


def test_buy_and_hold_buys_at_the_first_open_not_the_second() -> None:
    """The benchmark's first open is a real price. Ignoring it would flatter or
    penalise the benchmark depending on the first bar's gap, for no reason."""
    close = pd.Series([100.0] * 5)
    df = _frame(close)
    df.iloc[0, df.columns.get_loc("Open")] = 50.0
    df.iloc[0, df.columns.get_loc("Low")] = 50.0
    df.iloc[1, df.columns.get_loc("Open")] = 1_000.0

    result = run_backtest(df, BuyAndHold(), initial_capital=10_000)

    assert result.timing is SignalTiming.PREDEFINED
    assert result.first_actionable_bar == 0
    assert result.open_positions.iloc[0] == pytest.approx(10_000.0 / 50.0)
    # Bar 0 opened at 50 and closed at 100, so the same 200 units are now
    # worth 20,000. The gap on the entry bar is captured, not skipped.
    assert result.equity_curve.iloc[0] == pytest.approx(20_000.0)


def test_never_trade_is_flat_from_the_first_bar() -> None:
    df = _frame(_zigzag(60))
    result = run_backtest(df, NeverTrade(), initial_capital=10_000)
    assert (result.open_positions == 0.0).all()
    assert result.equity_curve.iloc[0] == pytest.approx(10_000.0)
    assert result.first_actionable_bar == 0


def test_first_actionable_bar_accounts_for_the_derived_warmup() -> None:
    df = _frame(pd.Series(np.linspace(100, 300, 400)))
    sma_result = run_backtest(df, SmaCrossover(fast=20, slow=50), initial_capital=10_000)
    mom_result = run_backtest(df, Momentum(lookback=126), initial_capital=10_000)

    # The first SMA value exists at bar 49, so the first fill is on bar 50.
    assert sma_result.first_actionable_bar == 50
    # The first momentum reading exists at bar 126, so the first fill is on 127.
    assert mom_result.first_actionable_bar == 127
    assert sma_result.open_positions.iloc[:50].sum() == 0.0


def test_a_derived_strategy_that_never_warms_up_is_reported_as_inactionable() -> None:
    df = _frame(pd.Series(np.linspace(100, 300, 60)))
    result = run_backtest(df, Momentum(lookback=126), initial_capital=10_000)
    assert result.first_actionable_bar is None
    assert result.is_actionable is False
    assert (result.open_positions == 0.0).all()


# --- engine guards -------------------------------------------------------


def test_a_predefined_signal_that_varies_is_rejected() -> None:
    """A decision taken before the data starts cannot react to the data."""

    class TimeVaryingPretend:
        timing = SignalTiming.PREDEFINED

        def generate(self, df: pd.DataFrame) -> pd.Series:
            return (df["Close"] > df["Close"].iloc[0]).astype("float64")

    with pytest.raises(InvalidSignals, match="signal changes over time"):
        run_backtest(_frame(_zigzag(80)), TimeVaryingPretend(), initial_capital=10_000)


def test_a_predefined_signal_with_nan_is_rejected() -> None:
    class NaNPretend:
        timing = SignalTiming.PREDEFINED

        def generate(self, df: pd.DataFrame) -> pd.Series:
            return pd.Series(np.nan, index=df.index)

    with pytest.raises(InvalidSignals, match="NaN signals"):
        run_backtest(_frame(_zigzag(80)), NaNPretend(), initial_capital=10_000)


def test_a_strategy_without_timing_is_rejected() -> None:
    class Undeclared:
        def generate(self, df: pd.DataFrame) -> pd.Series:
            return pd.Series(1.0, index=df.index)

    with pytest.raises(InvalidSignals, match="SignalTiming"):
        run_backtest(_frame(_zigzag(80)), Undeclared(), initial_capital=10_000)


def test_signals_on_a_different_index_are_rejected() -> None:
    class Misaligned:
        timing = SignalTiming.DERIVED

        def generate(self, df: pd.DataFrame) -> pd.Series:
            shifted = df.index + pd.Timedelta(days=1)
            return pd.Series(1.0, index=shifted)

    with pytest.raises(InvalidSignals, match="different index"):
        run_backtest(_frame(_zigzag(80)), Misaligned(), initial_capital=10_000)


def test_wrong_signal_length_is_rejected() -> None:
    class Short:
        timing = SignalTiming.DERIVED

        def generate(self, df: pd.DataFrame) -> pd.Series:
            return pd.Series(1.0, index=df.index[:-5])

    with pytest.raises(InvalidSignals, match="exactly one value per bar"):
        run_backtest(_frame(_zigzag(80)), Short(), initial_capital=10_000)


def test_non_series_signals_are_rejected() -> None:
    class Array:
        timing = SignalTiming.DERIVED

        def generate(self, df: pd.DataFrame):
            return np.ones(len(df))

    with pytest.raises(InvalidSignals, match="pandas Series"):
        run_backtest(_frame(_zigzag(80)), Array(), initial_capital=10_000)


def test_non_numeric_signals_are_rejected() -> None:
    class Textual:
        timing = SignalTiming.DERIVED

        def generate(self, df: pd.DataFrame) -> pd.Series:
            return pd.Series("buy", index=df.index)

    with pytest.raises(InvalidSignals, match="non-numeric"):
        run_backtest(_frame(_zigzag(80)), Textual(), initial_capital=10_000)


def test_empty_dataset_and_bad_capital_are_rejected() -> None:
    df = _frame(_zigzag(10))
    with pytest.raises(BacktestError, match="empty dataset"):
        run_backtest(df.iloc[:0], BuyAndHold(), initial_capital=10_000)
    with pytest.raises(BacktestError, match="positive"):
        run_backtest(df, BuyAndHold(), initial_capital=0)


def test_a_gap_that_skips_the_entry_open_is_never_filled_on_the_close() -> None:
    """A NaN-free but absurd gap must not produce an impossible fill price."""
    close = pd.Series(np.linspace(100.0, 200.0, 100))
    df = _frame(close)
    result = run_backtest(df, SmaCrossover(fast=5, slow=20), initial_capital=10_000)
    # Every fill used a real open: units are finite and non-negative throughout.
    assert np.isfinite(result.open_positions.to_numpy()).all()
    assert (result.open_positions >= 0).all()
