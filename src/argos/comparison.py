"""Comparison mode: evaluate several strategies on a common window.

An *individual* backtest respects each strategy's own semantics and warm-up.
That is the right thing when you want to know how one strategy behaves.

It is the wrong thing when you want to compare strategies in a table, because
they would then be measured over different stretches of the market: momentum
needs 126 bars of history before it can act at all, so it would sit in cash
while buy-and-hold was already invested.

Comparison mode therefore computes the first bar at which *every* strategy
could act, trims every equity curve to that window, and rescales each curve
multiplicatively to the same starting capital. Rebasing is a pure scale change:
returns, volatility, Sharpe, Sortino, drawdown and exposure are all invariant
under it, which the tests verify.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from argos.engine import BacktestResult, run_backtest
from argos.errors import ConfigurationError
from argos.metrics import Metrics, compute_metrics
from argos.strategy import StrategySpec


@dataclass(frozen=True)
class CostScenario:
    """A synthetic cost setting. Not a model of any real broker."""

    name: str
    commission_pct: float
    slippage_pct: float


SCENARIOS: tuple[CostScenario, ...] = (
    CostScenario("A: frictionless", 0.0, 0.0),
    CostScenario("B: 5bp/side", 0.0005, 0.0005),
    CostScenario("C: 10bp/side", 0.001, 0.001),
)


@dataclass(frozen=True)
class ComparisonRun:
    """One strategy, measured over the common window.

    `carries_position_from_before` marks a strategy that already held a position
    on the first bar of the window. For Buy & Hold that is always true, and it
    has a consequence worth stating plainly: the entry fill, and therefore its
    commission, happened before the window. Rebasing the curve to 10,000 at the
    window start makes that cost invisible in the window's return even though
    `total_commission` still reports it. A cost table that hid this would look
    like evidence that costs do not matter, so it is reported instead.

    `trades_inside_window` records whether any fill happened during the window.
    Only then can costs reduce the window's own return.
    """

    key: str
    label: str
    result: BacktestResult
    metrics: Metrics
    carries_position_from_before: bool = False
    trades_inside_window: bool = False


@dataclass(frozen=True)
class Comparison:
    """A full comparison experiment."""

    ticker: str
    requested_start: pd.Timestamp
    requested_end: pd.Timestamp
    common_start: pd.Timestamp
    common_end: pd.Timestamp
    initial_capital: float
    runs: list[ComparisonRun]
    benchmark_key: str = "buyhold"

    @property
    def requested_period(self) -> str:
        return f"{self.requested_start:%Y-%m-%d} -> {self.requested_end:%Y-%m-%d}"

    @property
    def common_period(self) -> str:
        return f"{self.common_start:%Y-%m-%d} -> {self.common_end:%Y-%m-%d}"

    def by_key(self, key: str) -> ComparisonRun:
        for run in self.runs:
            if run.key == key:
                return run
        raise KeyError(key)

    @property
    def benchmark(self) -> ComparisonRun | None:
        try:
            return self.by_key(self.benchmark_key)
        except KeyError:
            return None

    def excess_return(self, key: str) -> float:
        bench = self.benchmark
        if bench is None:
            return float("nan")
        return self.by_key(key).metrics.total_return - bench.metrics.total_return


def common_window(results: dict[str, BacktestResult], index: pd.Index) -> int:
    """First positional bar at which every strategy could act.

    A strategy that never becomes actionable over the dataset makes the
    comparison impossible, which is reported rather than silently dropped.
    """
    if not results:
        raise ConfigurationError("comparison requires at least one strategy")

    actionable = []
    for key, result in results.items():
        if result.first_actionable_bar is None:
            raise ConfigurationError(
                f"strategy '{key}' never became actionable on this dataset: its warm-up "
                "is longer than the requested period"
            )
        actionable.append(result.first_actionable_bar)

    return int(max(actionable))


def rebase(equity_curve: pd.Series, initial_capital: float) -> pd.Series:
    """Rescale an equity curve to start at `initial_capital`.

    A pure multiplicative change, so every ratio metric is preserved.
    """
    if equity_curve.empty:
        raise ValueError("cannot rebase an empty equity curve")
    first = float(equity_curve.iloc[0])
    if first <= 0:
        raise ValueError("cannot rebase an equity curve that starts at or below zero")
    return equity_curve * (initial_capital / first)


def _trim(result: BacktestResult, start: int, end: int) -> tuple[pd.Series, pd.Series]:
    return (
        result.equity_curve.iloc[start : end + 1],
        result.open_positions.iloc[start : end + 1],
    )


def _has_fills_inside(units: pd.Series) -> bool:
    """True when the position size changed anywhere inside the window.

    A change in units can only come from a fill, so this answers "did any
    execution land in this window" without the engine having to keep per-bar
    trade records.
    """
    if len(units) < 2:
        return False
    values = units.to_numpy(dtype="float64")
    return bool((values[1:] != values[:-1]).any())


def run_comparison(
    df: pd.DataFrame,
    specs: list[StrategySpec],
    initial_capital: float = 10_000.0,
    commission_pct: float = 0.0,
    slippage_pct: float = 0.0,
    annual_rf: float = 0.0,
    benchmark_key: str = "buyhold",
) -> Comparison:
    """Run every spec over the full dataset, then compare on the common window."""
    results = {
        spec.key: run_backtest(
            df,
            spec.factory(**spec.params),
            initial_capital=initial_capital,
            commission_pct=commission_pct,
            slippage_pct=slippage_pct,
            strategy_name=spec.label,
        )
        for spec in specs
    }

    start = common_window(results, df.index)
    end = len(df.index) - 1

    if end <= start:
        raise ConfigurationError(
            f"the common evaluation window is empty: all strategies become actionable at "
            f"bar {start} but the dataset ends at bar {end}"
        )

    runs: list[ComparisonRun] = []
    for spec in specs:
        result = results[spec.key]
        equity, units = _trim(result, start, end)
        # Rebasing exists to inject fresh capital at a window that begins part
        # way through a run, where the portfolio arrives already holding a
        # position. When the window is not trimmed, the curve already starts from
        # the capital the run was given, and rescaling it would discard whatever
        # the first bar did -- which would make this mode disagree with an
        # individual run covering exactly the same bars.
        rebased = rebase(equity, initial_capital) if start > 0 else equity
        runs.append(
            ComparisonRun(
                key=spec.key,
                label=spec.label,
                result=result,
                carries_position_from_before=bool(units.iloc[0] > 0),
                trades_inside_window=_has_fills_inside(units),
                metrics=compute_metrics(
                    rebased,
                    units,
                    initial_capital=initial_capital,
                    executions=result.portfolio.executions,
                    round_trips=result.portfolio.round_trips,
                    total_commission=result.portfolio.total_commission,
                    annual_rf=annual_rf,
                ),
            )
        )

    return Comparison(
        ticker="",
        requested_start=df.index[0],
        requested_end=df.index[-1],
        common_start=df.index[start],
        common_end=df.index[end],
        initial_capital=initial_capital,
        runs=runs,
        benchmark_key=benchmark_key,
    )


@dataclass
class ExperimentRow:
    """One row of the experiment matrix."""

    ticker: str
    requested_start: str
    requested_end: str
    common_start: str
    common_end: str
    strategy_key: str
    strategy_label: str
    initial_capital: float
    final_equity: float
    total_return: float
    cagr: float
    annualized_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    exposure: float
    round_trips: int
    executions: int
    total_commission: float
    commission_pct: float
    slippage_pct: float
    scenario: str
    trades_inside_window: bool = False
    status: str = "ok"
    note: str = ""
    extras: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if k != "extras"}


MATRIX_COLUMNS = [
    "ticker",
    "requested_start",
    "requested_end",
    "common_start",
    "common_end",
    "scenario",
    "commission_pct",
    "slippage_pct",
    "strategy_key",
    "strategy_label",
    "initial_capital",
    "final_equity",
    "total_return",
    "cagr",
    "annualized_volatility",
    "sharpe_ratio",
    "sortino_ratio",
    "max_drawdown",
    "calmar_ratio",
    "exposure",
    "round_trips",
    "executions",
    "total_commission",
    "trades_inside_window",
    "status",
    "note",
]