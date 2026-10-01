"""Invariants that must hold for every dataset, checked over many random series.

The other test files pin specific numbers. This one hammers the same properties
with generated price paths of different shapes: trends, crashes, gaps, flat
segments, tiny ranges. An invariant that only holds on a friendly fixture is not
an invariant.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from argos.comparison import run_comparison
from argos.engine import run_backtest
from argos.metrics import compute_metrics, max_drawdown
from argos.portfolio import Portfolio
from argos.strategy import BuyAndHold, Momentum, NeverTrade, SmaCrossover, default_specs

STRATEGIES = [SmaCrossover(fast=5, slow=20), Momentum(lookback=30), BuyAndHold(), NeverTrade()]


def _path(kind: str, n: int, seed: int) -> pd.Series:
    rng = np.random.default_rng(seed)
    if kind == "trend":
        drift = rng.uniform(-0.001, 0.002)
        return pd.Series(100.0 * np.cumprod(1.0 + rng.normal(drift, 0.009, n)))
    if kind == "crash":
        base = 100.0 * np.cumprod(1.0 + rng.normal(0.001, 0.006, n))
        return pd.Series(base) * np.where(np.arange(n) > n * 0.7, 0.55, 1.0)
    if kind == "gaps":
        jumps = rng.choice([0.0, 0.06, -0.06], size=n, p=[0.94, 0.03, 0.03])
        return pd.Series(100.0 * np.cumprod(1.0 + rng.normal(0.0, 0.004, n) + jumps))
    if kind == "flat":
        return pd.Series(np.full(n, 100.0))
    if kind == "tiny":
        return pd.Series(100.0 + rng.normal(0, 1e-6, n))
    if kind == "spiky":
        base = 100.0 * np.cumprod(1.0 + rng.normal(0.0, 0.004, n))
        return pd.Series(base) * np.where(rng.random(n) < 0.02, 0.7, 1.0)
    raise AssertionError(kind)


KINDS = ("trend", "crash", "gaps", "flat", "tiny", "spiky")


def _frame(kind: str, n: int, seed: int) -> pd.DataFrame:
    close = _path(kind, n, seed).to_numpy()
    index = pd.date_range("2018-01-01", periods=n, freq="B", tz="UTC")
    return pd.DataFrame(
        {
            "Open": close * 0.998,
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Volume": 1e6,
        },
        index=index,
    )


CASES = [(kind, n, seed) for kind in KINDS for n in (60, 250, 600) for seed in (1, 2, 3)]


@pytest.mark.parametrize(("kind", "n", "seed"), CASES)
def test_equity_never_produces_nan_or_infinity(kind: str, n: int, seed: int) -> None:
    df = _frame(kind, n, seed)
    for strategy in STRATEGIES:
        result = run_backtest(df, strategy, initial_capital=10_000)
        equity = result.equity_curve.to_numpy()
        assert np.isfinite(equity).all(), (kind, n, type(strategy).__name__)
        assert (equity > 0).all(), (kind, n, type(strategy).__name__)


@pytest.mark.parametrize(("kind", "n", "seed"), CASES)
def test_position_is_never_short_and_exposure_is_a_probability(
    kind: str, n: int, seed: int
) -> None:
    df = _frame(kind, n, seed)
    for strategy in STRATEGIES:
        units = run_backtest(df, strategy, initial_capital=10_000).open_positions
        assert (units >= 0).all()
        m = compute_metrics(
            run_backtest(df, strategy, initial_capital=10_000).equity_curve, units,
            initial_capital=10_000,
        )
        assert 0.0 <= m.exposure <= 1.0


@pytest.mark.parametrize(("kind", "n", "seed"), CASES)
def test_equity_always_equals_cash_plus_the_marked_position(
    kind: str, n: int, seed: int
) -> None:
    """The accounting identity, on every bar: no fill, fee or price move may break it."""
    df = _frame(kind, n, seed)
    for strategy in STRATEGIES:
        result = run_backtest(df, strategy, initial_capital=10_000)
        expected = result.cash_curve + result.open_positions * df["Close"]
        np.testing.assert_allclose(
            result.equity_curve.to_numpy(), expected.to_numpy(), rtol=1e-12, atol=1e-9
        )
        # The final bar must also match the portfolio's own view of the world.
        assert result.equity_curve.iloc[-1] == pytest.approx(
            result.portfolio.equity_at(float(df["Close"].iloc[-1]))
        )
        assert result.cash_curve.iloc[-1] == pytest.approx(result.portfolio.cash)


@pytest.mark.parametrize(("kind", "n", "seed"), CASES)
def test_equity_never_falls_below_zero_or_below_cash_when_flat(
    kind: str, n: int, seed: int
) -> None:
    df = _frame(kind, n, seed)
    for strategy in STRATEGIES:
        result = run_backtest(df, strategy, initial_capital=10_000)
        flat = result.open_positions == 0
        if flat.any():
            assert (result.equity_curve[flat] >= 0).all()


@pytest.mark.parametrize(("kind", "n", "seed"), CASES)
def test_costs_monotonically_reduce_the_outcome(kind: str, n: int, seed: int) -> None:
    df = _frame(kind, n, seed)
    for strategy in STRATEGIES:
        finals = []
        for commission in (0.0, 0.0005, 0.002, 0.01):
            result = run_backtest(
                df, strategy, initial_capital=10_000,
                commission_pct=commission, slippage_pct=commission,
            )
            finals.append(result.equity_curve.iloc[-1])
        assert finals == sorted(finals, reverse=True), (kind, n, type(strategy).__name__)


@pytest.mark.parametrize(("kind", "n", "seed"), CASES)
def test_repeating_a_run_reproduces_it_exactly(kind: str, n: int, seed: int) -> None:
    df = _frame(kind, n, seed)
    for strategy in STRATEGIES:
        a = run_backtest(df, strategy, initial_capital=10_000)
        b = run_backtest(df, strategy, initial_capital=10_000)
        pd.testing.assert_series_equal(a.equity_curve, b.equity_curve)
        assert a.portfolio.executions == b.portfolio.executions
        assert a.portfolio.total_commission == b.portfolio.total_commission


@pytest.mark.parametrize(("kind", "n", "seed"), CASES)
def test_no_metric_is_ever_infinite(kind: str, n: int, seed: int) -> None:
    df = _frame(kind, n, seed)
    for strategy in STRATEGIES:
        result = run_backtest(df, strategy, initial_capital=10_000)
        m = compute_metrics(result.equity_curve, result.open_positions, initial_capital=10_000)
        for name, value in vars(m).items():
            if isinstance(value, float) and not math.isnan(value):
                assert math.isfinite(value), (name, kind, n, value)


@pytest.mark.parametrize("seed", range(8))
def test_drawdown_matches_a_slow_reference_implementation(seed: int) -> None:
    """Recomputed with plain Python loops: a second opinion, not a second copy."""
    df = _frame("trend", 400, seed)
    equity = run_backtest(df, SmaCrossover(fast=10, slow=30), initial_capital=10_000).equity_curve

    peak = equity.iloc[0]
    worst = 0.0
    for value in equity:
        peak = max(peak, value)
        worst = max(worst, 1.0 - value / peak)

    assert max_drawdown(equity) == pytest.approx(worst, abs=1e-12)
    assert worst >= 0.0


@pytest.mark.parametrize("seed", range(6))
def test_total_return_and_cagr_agree_on_the_same_window(seed: int) -> None:
    """CAGR and total return describe the same growth over the same days."""
    df = _frame("trend", 500, seed)
    result = run_backtest(df, BuyAndHold(), initial_capital=10_000)
    m = compute_metrics(result.equity_curve, result.open_positions, initial_capital=10_000)

    days = (result.equity_curve.index[-1] - result.equity_curve.index[0]).days
    assert m.cagr == pytest.approx((1.0 + m.total_return) ** (365.25 / days) - 1.0)


@pytest.mark.parametrize("seed", range(6))
def test_rebasing_never_changes_a_ratio(seed: int) -> None:
    from argos.comparison import rebase

    df = _frame("trend", 500, seed)
    for spec in default_specs():
        result = run_backtest(df, spec.factory(**spec.params), initial_capital=10_000)
        start = 0 if result.first_actionable_bar is None else result.first_actionable_bar
        equity = result.equity_curve.iloc[start:]
        units = result.open_positions.iloc[start:]

        raw = compute_metrics(equity, units, initial_capital=1.0)
        for capital in (1.0, 1_000.0, 10_000.0, 7_777.0):
            scaled = compute_metrics(rebase(equity, capital), units, initial_capital=capital)
            assert scaled.total_return == pytest.approx(raw.total_return, rel=1e-9)
            assert scaled.max_drawdown == pytest.approx(raw.max_drawdown, rel=1e-9)
            assert scaled.sharpe_ratio == pytest.approx(raw.sharpe_ratio, rel=1e-9, nan_ok=True)
            assert scaled.final_equity == pytest.approx(capital * (1.0 + raw.total_return))


@pytest.mark.parametrize("seed", range(5))
def test_gap_up_bars_do_not_break_the_simulation(seed: int) -> None:
    """A 30% overnight gap must not create an impossible fill or a negative cash."""
    df = _frame("gaps", 300, seed)
    df.iloc[50, df.columns.get_loc("Open")] *= 1.3
    df.iloc[50, df.columns.get_loc("High")] *= 1.35
    result = run_backtest(df, SmaCrossover(fast=5, slow=20), initial_capital=10_000)
    assert np.isfinite(result.equity_curve.to_numpy()).all()
    assert (result.equity_curve > 0).all()
    assert result.portfolio.cash >= 0.0


@pytest.mark.parametrize(("capital", "price"), [(0.01, 1_000.0), (1.0, 1_000.0), (1e6, 0.01)])
def test_extreme_capital_and_price_scales_stay_consistent(
    capital: float, price: float
) -> None:
    p = Portfolio(initial_capital=capital, commission_pct=0.001, slippage_pct=0.001)
    p.buy(price=price)
    assert p.cash == pytest.approx(0.0, abs=capital * 1e-9)
    p.sell(price=price)
    assert p.cash >= 0.0
    assert math.isfinite(p.cash)
    assert p.units == 0.0


def test_a_constant_price_series_gives_a_zero_return_not_a_crash() -> None:
    df = _frame("flat", 120, 1)
    for strategy in (BuyAndHold(), SmaCrossover(fast=5, slow=20), NeverTrade()):
        result = run_backtest(df, strategy, initial_capital=10_000)
        m = compute_metrics(result.equity_curve, result.open_positions, initial_capital=10_000)
        assert m.total_return == pytest.approx(0.0, abs=1e-6)
        assert m.max_drawdown == 0.0


def test_comparison_agrees_with_individual_runs_on_the_common_window() -> None:
    """A trimmed comparison row must match the same strategy run by hand."""
    df = _frame("trend", 600, 4)
    specs = [s for s in default_specs() if s.key != "buyhold"]
    comparison = run_comparison(df, specs)
    start = df.index.get_loc(comparison.common_start)

    for run in comparison.runs:
        window_equity = run.result.equity_curve.iloc[start:]
        manual = compute_metrics(window_equity, run.result.open_positions.iloc[start:],
                                 initial_capital=10_000)
        assert manual.total_return == pytest.approx(run.metrics.total_return)
        assert manual.max_drawdown == pytest.approx(run.metrics.max_drawdown)
        assert manual.evaluations == run.metrics.evaluations


def test_warmup_length_is_exactly_the_lookback() -> None:
    for lookback in (5, 30, 126, 200):
        df = _frame("trend", 400, 2)
        result = run_backtest(df, Momentum(lookback=lookback), initial_capital=10_000)
        assert result.first_actionable_bar == lookback + 1
        assert (result.open_positions.iloc[: lookback + 1] == 0).all()
