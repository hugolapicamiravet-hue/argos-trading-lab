"""Portfolio invariants: money is conserved, costs only ever subtract."""

from __future__ import annotations

import math

import pytest

from argos.portfolio import Portfolio


def test_invalid_construction_is_rejected() -> None:
    with pytest.raises(ValueError):
        Portfolio(initial_capital=0)
    with pytest.raises(ValueError):
        Portfolio(initial_capital=100, commission_pct=-0.001)
    with pytest.raises(ValueError):
        Portfolio(initial_capital=100, slippage_pct=-0.001)


def test_buy_does_not_create_money() -> None:
    p = Portfolio(initial_capital=10_000)
    p.buy(price=50.0)
    assert p.cash + p.units * 50.0 == pytest.approx(10_000.0)
    assert p.cash <= 0


def test_round_trip_without_costs_conserves_capital() -> None:
    p = Portfolio(initial_capital=10_000)
    p.buy(price=50.0)
    p.sell(price=50.0)
    assert p.cash == pytest.approx(10_000.0)
    assert p.units == 0.0
    assert p.round_trips == 1


def test_round_trip_with_commission_loses_exactly_the_fee() -> None:
    p = Portfolio(initial_capital=10_000, commission_pct=0.001)
    p.buy(price=50.0)
    units = p.units
    entry_fee = units * 50.0 * 0.001
    p.sell(price=50.0)
    exit_fee = units * 50.0 * 0.001
    assert p.cash == pytest.approx(10_000.0 - entry_fee - exit_fee)
    assert p.total_commission == pytest.approx(entry_fee + exit_fee)
    assert p.total_commission == pytest.approx(10_000.0 - p.cash)


def test_slippage_always_costs_money_in_both_directions() -> None:
    up = Portfolio(initial_capital=10_000, slippage_pct=0.001)
    up.buy(price=50.0)
    up.sell(price=50.0)
    assert up.cash < 10_000.0

    down = Portfolio(initial_capital=10_000, slippage_pct=0.001)
    down.buy(price=50.0)
    down.sell(price=50.0)
    assert down.cash < 10_000.0


def test_sell_without_position_is_a_noop() -> None:
    p = Portfolio(initial_capital=1_000)
    assert p.sell(price=10.0) == 0.0
    assert p.executions == 0


def test_buying_twice_is_rejected() -> None:
    p = Portfolio(initial_capital=1_000)
    p.buy(price=10.0)
    with pytest.raises(ValueError):
        p.buy(price=10.0)


def test_fractional_sizing_leaves_residual_cash() -> None:
    p = Portfolio(initial_capital=10_000)
    p.buy(price=100.0, fraction=0.5)
    assert p.cash == pytest.approx(5_000.0)
    assert p.units == pytest.approx(50.0)


def test_equity_at_price_reflects_position() -> None:
    p = Portfolio(initial_capital=1_000)
    p.buy(price=10.0)
    assert p.equity_at(12.0) > 1_000.0
    assert p.equity_at(8.0) < 1_000.0


def test_invalid_order_parameters_are_rejected() -> None:
    p = Portfolio(initial_capital=1_000)
    with pytest.raises(ValueError):
        p.buy(price=0)
    with pytest.raises(ValueError):
        p.buy(price=-10.0)
    with pytest.raises(ValueError):
        p.sell(price=0)
    with pytest.raises(ValueError):
        p.buy(price=10.0, fraction=0)
    with pytest.raises(ValueError):
        p.buy(price=10.0, fraction=1.5)


@pytest.mark.parametrize("price", [0.01, 0.5, 1.0, 137.42, 1_000.0, 250_000.0])
def test_fills_never_lose_or_invent_money_at_any_price_scale(price: float) -> None:
    p = Portfolio(initial_capital=10_000, commission_pct=0.0005, slippage_pct=0.0005)
    p.buy(price=price)
    assert p.cash == pytest.approx(0.0, abs=1e-6)
    p.sell(price=price)
    assert p.units == 0.0
    # Only costs were lost, and the loss is bounded by the cost model.
    assert p.cash < 10_000.0
    assert p.cash > 9_900.0


@pytest.mark.parametrize("capital", [0.01, 1.0, 100.0, 10_000.0, 1e9])
def test_cash_and_units_stay_finite_for_any_initial_capital(capital: float) -> None:
    p = Portfolio(initial_capital=capital, commission_pct=0.001, slippage_pct=0.001)
    p.buy(price=123.45)
    p.sell(price=123.45)
    assert math.isfinite(p.cash)
    assert math.isfinite(p.units)
    assert p.cash >= 0.0
    assert p.units >= 0.0


def test_final_cash_is_monotonically_non_increasing_in_costs() -> None:
    """More friction can never produce a better outcome for the same trades."""
    results = []
    for pct in (0.0, 0.0001, 0.0005, 0.001, 0.01, 0.05, 0.5):
        p = Portfolio(initial_capital=10_000, commission_pct=pct, slippage_pct=pct)
        p.buy(price=100.0)
        p.sell(price=100.0)
        results.append(p.cash)
    assert results == sorted(results, reverse=True)
    assert results[0] == pytest.approx(10_000.0)


def test_extreme_cost_model_does_not_produce_negative_cash_on_exit() -> None:
    p = Portfolio(initial_capital=10_000, commission_pct=0.9, slippage_pct=0.5)
    p.buy(price=100.0)
    p.sell(price=100.0)
    assert p.cash >= 0.0
    assert math.isfinite(p.cash)


def test_fully_spent_capital_stays_finite_and_non_negative_after_a_round_trip() -> None:
    """Sizing funds the commission out of the same cash, so the balance hits
    exactly zero on entry and can never go negative afterwards."""
    p = Portfolio(initial_capital=1_000, commission_pct=0.001)
    p.buy(price=1_000.0)
    assert p.cash == pytest.approx(0.0, abs=1e-6)
    p.sell(price=1_000.0)
    assert p.cash >= 0.0
    assert not math.isnan(p.cash)
    assert p.cash == pytest.approx(1_000.0 - p.total_commission)
    assert p.cash == pytest.approx(998.002, abs=0.01)


def test_round_trip_counter_only_counts_full_exits() -> None:
    """Partial exits are outside the model, so they must not inflate the count."""
    p = Portfolio(initial_capital=10_000)
    p.buy(price=100.0)
    p.sell(price=100.0, fraction=0.5)
    assert p.round_trips == 0
    p.sell(price=100.0)  # full exit of what remains
    assert p.round_trips == 1
    assert p.units == 0.0


def test_executions_count_each_fill_separately() -> None:
    p = Portfolio(initial_capital=10_000)
    p.buy(price=100.0)
    assert p.executions == 1
    p.sell(price=100.0)
    assert p.executions == 2
