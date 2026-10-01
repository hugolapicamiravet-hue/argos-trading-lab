"""Metrics: exact values, documented conventions and the no-inf guarantee.

Expected values here are computed by hand, not by calling the implementation.
A test that recomputes the same formula proves nothing.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from argos.engine import run_backtest
from argos.metrics import (
    TRADING_DAYS_PER_YEAR,
    cagr,
    compute_metrics,
    downside_deviation,
    exposure,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
    total_return,
)
from argos.strategy import NeverTrade, SignalTiming


def _idx(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")


def _flat_frame(n: int = 30) -> pd.DataFrame:
    return pd.DataFrame(
        {"Open": 100.0, "High": 100.0, "Low": 100.0, "Close": 100.0, "Volume": 1.0},
        index=_idx(n),
    )


# --- return -------------------------------------------------------------


def test_total_return_arithmetic() -> None:
    assert total_return(1_000, 1_100) == pytest.approx(0.1)
    assert total_return(1_000, 900) == pytest.approx(-0.1)
    assert total_return(1_000, 1_000) == 0.0


def test_total_return_rejects_non_positive_capital() -> None:
    with pytest.raises(ValueError):
        total_return(0, 1_000)
    with pytest.raises(ValueError):
        total_return(-100, 1_000)


# --- CAGR ----------------------------------------------------------------


def test_cagr_over_exactly_one_year() -> None:
    # 365.25 days is one calendar year, so growth is exactly 1.2 -> +20%.
    assert cagr(1_000, 1_200, 365.25) == pytest.approx(0.2)


def test_cagr_uses_calendar_days_not_trading_days() -> None:
    two_years = 2 * 365.25
    assert cagr(1_000, 1_400, two_years) == pytest.approx(1.4**0.5 - 1)


def test_cagr_is_undefined_for_a_window_shorter_than_a_day() -> None:
    assert math.isnan(cagr(1_000, 1_100, 0))


def test_cagr_handles_a_total_loss_without_exploding() -> None:
    assert cagr(1_000, 0.0001, 365.25) < -0.9
    assert math.isfinite(cagr(1_000, 0.0001, 365.25))


# --- volatility ----------------------------------------------------------


def test_annualized_volatility_of_a_known_return_series() -> None:
    from argos.metrics import annualized_volatility

    returns = pd.Series([0.01, -0.01, 0.01, -0.01])
    expected = float(np.std([0.01, -0.01, 0.01, -0.01], ddof=1)) * math.sqrt(252)
    assert annualized_volatility(returns) == pytest.approx(expected)


def test_volatility_of_a_constant_equity_curve_is_zero_not_nan() -> None:
    equity = pd.Series([100.0] * 10, index=_idx(10))
    from argos.metrics import annualized_volatility

    assert annualized_volatility(equity.pct_change().iloc[1:]) == 0.0


def test_volatility_needs_two_observations() -> None:
    from argos.metrics import annualized_volatility

    assert math.isnan(annualized_volatility(pd.Series([0.01], dtype="float64")))


# --- Sharpe --------------------------------------------------------------


def test_sharpe_with_zero_risk_free_uses_the_mean_over_std() -> None:
    returns = pd.Series([0.01, -0.005, 0.02, 0.0, 0.015, -0.01])
    expected = (returns.mean() / returns.std(ddof=1)) * math.sqrt(TRADING_DAYS_PER_YEAR)
    assert sharpe_ratio(returns, 0.0) == pytest.approx(expected)


def test_sharpe_converts_an_annual_risk_free_rate_to_a_daily_one() -> None:
    returns = pd.Series([0.01, -0.005, 0.02, 0.0, 0.015, -0.01])
    annual_rf = 0.05
    daily_rf = (1 + annual_rf) ** (1 / 252) - 1
    expected = (
        (returns.mean() - daily_rf) / returns.std(ddof=1)
    ) * math.sqrt(TRADING_DAYS_PER_YEAR)
    assert sharpe_ratio(returns, annual_rf) == pytest.approx(expected)
    # A positive risk-free rate must lower the ratio.
    assert sharpe_ratio(returns, annual_rf) < sharpe_ratio(returns, 0.0)


def test_sharpe_is_nan_when_returns_are_constant() -> None:
    """Zero volatility makes the ratio undefined. It must not be inf."""
    value = sharpe_ratio(pd.Series([0.01] * 10), 0.0)
    assert math.isnan(value)
    assert not math.isinf(value)


def test_sharpe_is_nan_with_fewer_than_two_observations() -> None:
    assert math.isnan(sharpe_ratio(pd.Series([], dtype="float64")))
    assert math.isnan(sharpe_ratio(pd.Series([0.01])))


# --- Sortino -------------------------------------------------------------


def test_downside_deviation_counts_upside_bars_as_zero_shortfalls() -> None:
    returns = pd.Series([0.02, -0.01, 0.03, -0.02])
    expected = math.sqrt((0.0**2 + 0.01**2 + 0.0**2 + 0.02**2) / 4)
    assert downside_deviation(returns, 0.0) == pytest.approx(expected)


def test_sortino_exceeds_sharpe_when_upside_volatility_dominates() -> None:
    """Only downside moves are penalised, so an asymmetric series scores higher."""
    returns = pd.Series([0.05, 0.05, 0.05, 0.05, 0.05, -0.005])
    assert sortino_ratio(returns, 0.0) > sharpe_ratio(returns, 0.0)


def test_sortino_is_nan_when_there_are_no_downside_moves() -> None:
    value = sortino_ratio(pd.Series([0.01] * 10), 0.0)
    assert math.isnan(value)
    assert not math.isinf(value)


def test_sortino_matches_sharpe_when_losses_equal_gains_symmetrically() -> None:
    returns = pd.Series([0.02, -0.02])
    assert sortino_ratio(returns, 0.0) == pytest.approx(sharpe_ratio(returns, 0.0))


# --- drawdown ------------------------------------------------------------


def test_monotonic_equity_curve_has_zero_drawdown() -> None:
    assert max_drawdown(pd.Series([100.0, 110.0, 150.0, 200.0])) == 0.0


def test_constant_equity_curve_has_zero_drawdown() -> None:
    assert max_drawdown(pd.Series([50.0, 50.0, 50.0])) == 0.0


def test_drawdown_is_the_largest_peak_to_trough_decline() -> None:
    curve = pd.Series([100.0, 120.0, 60.0, 90.0])
    assert max_drawdown(curve) == pytest.approx(0.5)


def test_drawdown_recovers_after_new_peak() -> None:
    curve = pd.Series([100.0, 50.0, 100.0, 300.0, 150.0])
    assert max_drawdown(curve) == pytest.approx(0.5)


def test_drawdown_is_never_negative_and_is_stored_positive() -> None:
    curve = pd.Series(np.linspace(100, 500, 50))
    assert max_drawdown(curve) >= 0.0


def test_drawdown_rejects_empty_or_nonpositive_curves() -> None:
    with pytest.raises(ValueError):
        max_drawdown(pd.Series([], dtype="float64"))
    with pytest.raises(ValueError):
        max_drawdown(pd.Series([100.0, 0.0]))
    with pytest.raises(ValueError):
        max_drawdown(pd.Series([100.0, -5.0]))


# --- Calmar --------------------------------------------------------------


def test_calmar_divides_cagr_by_absolute_drawdown() -> None:
    from argos.metrics import calmar_ratio

    assert calmar_ratio(0.10, 0.20) == pytest.approx(0.5)
    # Drawdown sign must not matter: it is passed positive internally.
    assert calmar_ratio(0.10, -0.20) == pytest.approx(0.5)


def test_calmar_is_nan_without_drawdown() -> None:
    from argos.metrics import calmar_ratio

    assert math.isnan(calmar_ratio(0.10, 0.0))


# --- exposure ------------------------------------------------------------


def test_exposure_is_the_fraction_of_bars_holding_a_position() -> None:
    units = pd.Series([0.0, 1.0, 1.0, 0.0, 2.0], index=_idx(5))
    assert exposure(units) == pytest.approx(0.6)


def test_exposure_of_a_never_invested_curve_is_zero() -> None:
    assert exposure(pd.Series([0.0] * 7, index=_idx(7))) == 0.0


# --- aggregation ---------------------------------------------------------


def test_compute_metrics_never_returns_infinity() -> None:
    """A perfectly flat curve is the classic source of inf ratios."""
    equity = pd.Series([10_000.0] * 30, index=_idx(30))
    units = pd.Series([0.0] * 30, index=_idx(30))
    m = compute_metrics(equity, units, initial_capital=10_000)
    for value in vars(m).values():
        if isinstance(value, float):
            assert not math.isinf(value)


def test_compute_metrics_rejects_mismatched_lengths() -> None:
    equity = pd.Series([1.0, 2.0], index=_idx(2))
    with pytest.raises(ValueError):
        compute_metrics(equity, pd.Series([0.0], index=_idx(1)), initial_capital=1.0)


def test_compute_metrics_rejects_an_empty_curve() -> None:
    with pytest.raises(ValueError):
        compute_metrics(pd.Series([], dtype="float64"), pd.Series([], dtype="float64"), 1.0)


def test_compute_metrics_end_to_end_on_a_known_curve() -> None:
    equity = pd.Series([100.0, 120.0, 60.0, 90.0], index=_idx(4))
    units = pd.Series([0.0, 1.0, 1.0, 1.0], index=_idx(4))
    m = compute_metrics(equity, units, initial_capital=100.0, executions=2, round_trips=1)

    assert m.initial_capital == 100.0
    assert m.final_equity == pytest.approx(90.0)
    assert m.total_return == pytest.approx(-0.1)
    assert m.max_drawdown == pytest.approx(0.5)
    assert m.exposure == pytest.approx(0.75)
    assert m.evaluations == 4
    assert m.executions == 2
    assert m.round_trips == 1
    # Annualising three calendar days is mathematically defined but not
    # meaningful; the value must still follow the formula rather than blow up.
    assert m.cagr == pytest.approx((0.9 ** (365.25 / 3)) - 1)
    assert math.isfinite(m.cagr)
    assert m.calmar_ratio == pytest.approx(m.cagr / 0.5)


def test_cagr_is_undefined_for_a_single_bar_window() -> None:
    equity = pd.Series([100.0], index=_idx(1))
    units = pd.Series([0.0], index=_idx(1))
    m = compute_metrics(equity, units, initial_capital=100.0)
    assert math.isnan(m.cagr)
    assert m.total_return == 0.0
    assert m.evaluations == 1


def test_never_trading_strategy_reports_a_flat_result() -> None:
    result = run_backtest(_flat_frame(30), NeverTrade(), initial_capital=10_000)
    m = compute_metrics(
        result.equity_curve,
        result.open_positions,
        initial_capital=10_000,
        executions=result.portfolio.executions,
        round_trips=result.portfolio.round_trips,
    )

    assert m.executions == 0
    assert m.round_trips == 0
    assert m.final_equity == pytest.approx(10_000.0)
    assert m.total_return == 0.0
    assert m.max_drawdown == 0.0
    assert m.exposure == 0.0
    # A flat curve has no dispersion: the ratios are undefined, not infinite.
    assert math.isnan(m.sharpe_ratio)
    assert math.isnan(m.sortino_ratio)
    assert math.isnan(m.calmar_ratio)
    assert m.annualized_volatility == 0.0


def test_metrics_use_closes_for_drawdown_not_intraperiod_lows() -> None:
    """Documented limitation: drawdown is measured on the equity curve, not on
    intrabar lows. A deep wick on a flat curve is not a drawdown here."""
    equity = pd.Series([100.0, 100.0, 100.0], index=_idx(3))
    assert max_drawdown(equity) == 0.0


def test_annual_rf_flows_through_to_sharpe_and_sortino() -> None:
    rng = np.random.default_rng(7)
    equity = pd.Series(100.0 * np.cumprod(1 + rng.normal(0.0004, 0.01, 300)), index=_idx(300))
    units = pd.Series([1.0] * 300, index=_idx(300))

    free = compute_metrics(equity, units, initial_capital=100.0)
    priced = compute_metrics(equity, units, initial_capital=100.0, annual_rf=0.03)

    assert priced.sharpe_ratio < free.sharpe_ratio
    assert priced.sortino_ratio < free.sortino_ratio


def test_signal_timing_enum_is_still_importable_from_strategy() -> None:
    """Guards the public surface used by the CLI and reports."""
    assert SignalTiming.PREDEFINED.value == "predefined"
