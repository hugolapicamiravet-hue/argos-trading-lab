"""Comparison mode: the common window, and what rebasing does and does not change.

Comparison is only meaningful if every strategy is measured over the same
stretch of market. Momentum needs 126 bars of history before it can act, so a
naive side-by-side table would quietly give buy-and-hold a head start.

The other half of the contract is that rebasing is a *scale* change only. If
rebasing moved a drawdown or a Sharpe, the comparison would be inventing
differences that do not exist in the simulation.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from argos.comparison import Comparison, common_window, rebase, run_comparison
from argos.engine import run_backtest
from argos.errors import ConfigurationError
from argos.metrics import compute_metrics
from argos.report import format_comparison
from argos.strategy import BuyAndHold, Momentum, SmaCrossover, default_specs


def _frame(n: int = 500, seed: int = 3) -> pd.DataFrame:
    """A deterministic zigzag long enough for every strategy to warm up."""
    rng = np.random.default_rng(seed)
    steps = rng.normal(0.0004, 0.011, n)
    close = 100.0 * np.cumprod(1.0 + steps)
    index = pd.date_range("2015-01-01", periods=n, freq="B", tz="UTC")
    return pd.DataFrame(
        {
            "Open": close * 0.999,
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Volume": 1_000_000.0,
        },
        index=index,
    )


def _comparison(df: pd.DataFrame, **kwargs) -> Comparison:
    return run_comparison(df, default_specs(), **kwargs)


# --- the common window ---------------------------------------------------


def test_common_window_starts_when_the_slowest_strategy_can_act() -> None:
    df = _frame(500)
    comparison = _comparison(df)

    # Momentum(126) first reads a signal on bar 126, so its first fill is bar 127.
    assert comparison.by_key("momentum").result.first_actionable_bar == 127
    assert comparison.common_start == df.index[127]
    # The benchmark is actionable from bar 0; the comparison follows momentum.
    assert comparison.by_key("buyhold").result.first_actionable_bar == 0


def test_every_strategy_is_measured_over_the_same_bars() -> None:
    df = _frame(500)
    comparison = _comparison(df)

    for run in comparison.runs:
        assert run.metrics.evaluations == len(df.index) - 127
        assert run.metrics.initial_capital == pytest.approx(10_000.0)
        # All curves start from the same capital: the table is like for like.
        assert run.result.equity_curve.iloc[127] > 0


def test_individual_backtests_keep_their_own_period() -> None:
    """Comparison is the only mode that trims. A single backtest does not."""
    df = _frame(500)
    individual = run_backtest(df, BuyAndHold(), initial_capital=10_000)
    assert len(individual.equity_curve) == len(df.index)

    sma_only = run_comparison(df, [s for s in default_specs() if s.key == "sma"])
    # SMA needs 50 bars, so trimming happens even without momentum.
    assert sma_only.common_start == df.index[50]


def test_common_window_equals_the_whole_dataset_for_buy_hold_only() -> None:
    df = _frame(300)
    comparison = run_comparison(df, [s for s in default_specs() if s.key == "buyhold"])
    assert comparison.common_start == df.index[0]
    assert comparison.common_end == df.index[-1]


def test_both_periods_are_reported_explicitly() -> None:
    df = _frame(500)
    comparison = _comparison(df)
    assert comparison.requested_period == f"{df.index[0]:%Y-%m-%d} -> {df.index[-1]:%Y-%m-%d}"
    assert comparison.common_period == f"{df.index[127]:%Y-%m-%d} -> {df.index[-1]:%Y-%m-%d}"
    assert comparison.requested_period != comparison.common_period


# --- rebasing invariants -------------------------------------------------


def test_rebase_starts_the_curve_at_the_requested_capital() -> None:
    curve = pd.Series([4_000.0, 5_000.0, 6_000.0])
    out = rebase(curve, 10_000.0)
    assert out.iloc[0] == pytest.approx(10_000.0)
    assert out.iloc[-1] == pytest.approx(15_000.0)


def test_rebase_preserves_every_ratio_metric() -> None:
    """The core promise: scaling money away cannot change risk or reward ratios.

    Invariance is only meaningful when both sides measure the same window from
    the same baseline, so every arm here is rebased to its own capital. Comparing
    a rebased curve against an unrebased one would measure capital-scaled
    metrics, not ratios.
    """
    df = _frame(500)
    result = run_backtest(df, SmaCrossover(fast=20, slow=50), initial_capital=10_000)
    window = result.equity_curve.iloc[127:]
    units = result.open_positions.iloc[127:]

    scaled = {
        capital: compute_metrics(rebase(window, capital), units, initial_capital=capital)
        for capital in (1.0, 10_000.0, 7_777.0)
    }
    raw = scaled[10_000.0]

    for capital, metrics in scaled.items():
        assert metrics.annualized_volatility == pytest.approx(raw.annualized_volatility)
        assert metrics.sharpe_ratio == pytest.approx(raw.sharpe_ratio)
        assert metrics.sortino_ratio == pytest.approx(raw.sortino_ratio)
        assert metrics.max_drawdown == pytest.approx(raw.max_drawdown)
        assert metrics.calmar_ratio == pytest.approx(raw.calmar_ratio)
        assert metrics.exposure == pytest.approx(raw.exposure)
        assert metrics.evaluations == raw.evaluations
        # total_return, cagr and final_equity are *not* invariant: rescaling the
        # money is the point. Each must instead follow from its own capital.
        assert metrics.total_return == pytest.approx(metrics.final_equity / capital - 1.0)
        assert metrics.final_equity == pytest.approx(capital * (1.0 + metrics.total_return))

    # The rebased level really did move, otherwise nothing above is a real test.
    assert scaled[1.0].final_equity != pytest.approx(scaled[10_000.0].final_equity)


def test_rebased_final_equity_follows_the_same_return() -> None:
    df = _frame(500)
    comparison = _comparison(df)
    for run in comparison.runs:
        m = run.metrics
        assert m.final_equity == pytest.approx(10_000.0 * (1.0 + m.total_return))


def test_rebase_rejects_a_curve_starting_at_zero() -> None:
    with pytest.raises(ValueError):
        rebase(pd.Series([0.0, 1.0]), 10_000.0)
    with pytest.raises(ValueError):
        rebase(pd.Series([], dtype="float64"), 10_000.0)


# --- the benchmark relationship -----------------------------------------


def test_benchmark_is_buy_and_hold_by_default() -> None:
    comparison = _comparison(_frame(500))
    assert comparison.benchmark is not None
    assert comparison.benchmark.key == "buyhold"


def test_excess_return_is_measured_against_the_benchmark() -> None:
    comparison = _comparison(_frame(500))
    bench = comparison.benchmark
    for run in comparison.runs:
        expected = run.metrics.total_return - bench.metrics.total_return
        assert comparison.excess_return(run.key) == pytest.approx(expected)
    assert comparison.excess_return("buyhold") == pytest.approx(0.0)


def test_benchmark_lookup_returns_none_when_absent() -> None:
    df = _frame(500)
    comparison = run_comparison(df, [s for s in default_specs() if s.key == "sma"])
    assert comparison.benchmark is None
    with pytest.raises(KeyError):
        comparison.by_key("nope")


# --- failure modes -------------------------------------------------------


def test_a_strategy_that_can_never_act_makes_comparison_impossible() -> None:
    """Reported, not silently dropped: a missing strategy would bias the table."""
    from argos.strategy import StrategySpec

    specs = default_specs() + [StrategySpec("huge", "Lookback 9999", Momentum, {"lookback": 9999})]
    with pytest.raises(ConfigurationError, match="never became actionable"):
        run_comparison(_frame(300), specs)


def test_an_empty_common_window_is_reported() -> None:
    df = _frame(260)
    specs = default_specs() + [
        # Actionable at the very last bar, leaving no window to evaluate.
        type(default_specs()[0])("edge", "Edge", Momentum, {"lookback": 258})
    ]
    with pytest.raises(ConfigurationError, match="common evaluation window is empty"):
        run_comparison(df, specs)


def test_common_window_requires_at_least_one_result() -> None:
    with pytest.raises(ConfigurationError, match="at least one strategy"):
        common_window({}, pd.Index([]))


# --- documented counter semantics ---------------------------------------


def test_return_metrics_cover_the_window_while_counters_cover_the_full_run() -> None:
    """A pinned asymmetry.

    The engine cannot attribute an execution to the window without per-bar
    trade records, and trimming the input would re-warm the indicators and
    change the signals themselves. So the activity counters describe everything
    the engine did, and only the return and risk metrics describe the window.
    The report says so explicitly.
    """
    df = _frame(500)
    comparison = _comparison(df)
    for run in comparison.runs:
        assert run.metrics.evaluations == len(df.index) - 127
        assert run.metrics.executions == run.result.portfolio.executions
        assert run.metrics.round_trips == run.result.portfolio.round_trips
        assert len(run.result.equity_curve) == len(df.index)


def test_the_report_states_which_measurements_share_a_window() -> None:
    from argos.report import format_comparison

    text = format_comparison(_comparison(_frame(500)))
    flat = " ".join(text.split())  # footnotes are wrapped to the table width
    assert "requested period" in text
    assert "common evaluation period" in text
    assert "counted over the full requested period" in flat
    assert "made no trade inside the window" in flat
    assert "n/a" not in text


def test_report_footnotes_stay_within_the_table_width() -> None:
    """A footnote wider than the table turns into an unreadable wall of text."""
    from argos.report import format_comparison

    text = format_comparison(_comparison(_frame(500)))
    lines = text.splitlines()
    rule = next(line for line in lines if line.startswith("="))
    assert all(len(line) <= len(rule) + 2 for line in lines), [
        line for line in lines if len(line) > len(rule) + 2
    ]


# --- costs under comparison ---------------------------------------------


def test_every_strategy_loses_money_under_costs() -> None:
    df = _frame(500)
    free = _comparison(df, commission_pct=0.0, slippage_pct=0.0)
    costly = _comparison(df, commission_pct=0.001, slippage_pct=0.001)

    for run in costly.runs:
        assert run.metrics.total_commission > 0
        assert run.metrics.executions == free.by_key(run.key).metrics.executions
        if run.trades_inside_window:
            assert run.metrics.final_equity < free.by_key(run.key).metrics.final_equity
        else:
            # Pinned limitation: no fill inside the window means costs cannot
            # show up in the window's return. The report says so.
            assert run.metrics.final_equity == pytest.approx(
                free.by_key(run.key).metrics.final_equity, rel=1e-9
            )


def test_buy_and_hold_entry_cost_is_reported_not_hidden() -> None:
    """The benchmark's single fill precedes the common window.

    The honest consequence is that costs cannot reduce Buy & Hold's window
    return. Rather than let a cost table imply otherwise, the report names it.
    """
    df = _frame(500)
    free = _comparison(df)
    costly = _comparison(df, commission_pct=0.001, slippage_pct=0.001)

    bench_free = free.by_key("buyhold")
    bench_costly = costly.by_key("buyhold")
    assert bench_free.carries_position_from_before is True
    assert bench_costly.metrics.total_commission > 0
    assert bench_free.metrics.total_commission == 0.0

    text = format_comparison(costly)
    assert "Buy & Hold made no trade inside the window" in text
    # A derived strategy that trades inside the window does pay costs in it.
    assert costly.by_key("sma").trades_inside_window is True


def test_a_strategy_trading_inside_the_window_pays_costs_in_the_window() -> None:
    """SMA does not hold a position at the window start, so its cost is visible."""
    df = _frame(500)
    free = _comparison(df)
    costly = _comparison(df, commission_pct=0.001, slippage_pct=0.001)

    sma_costly = costly.by_key("sma")
    if not sma_costly.carries_position_from_before:
        assert sma_costly.metrics.final_equity < free.by_key("sma").metrics.final_equity


def test_comparison_rejects_non_positive_capital() -> None:
    from argos.errors import BacktestError

    with pytest.raises(BacktestError, match="positive"):
        _comparison(_frame(500), initial_capital=0)


def test_metrics_stay_finite_across_many_windows() -> None:
    """No inf may ever reach the comparison table."""
    for seed in range(6):
        df = _frame(500, seed=seed)
        for spec in default_specs():
            result = run_backtest(df, spec.factory(**spec.params), initial_capital=10_000)
            m = compute_metrics(
                result.equity_curve, result.open_positions, initial_capital=10_000
            )
            for value in vars(m).values():
                if isinstance(value, float) and not math.isnan(value):
                    assert math.isfinite(value), (seed, spec.key, value)
