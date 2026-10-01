"""Experiment matrix: many tickers x many windows x many strategies.

Every combination is an independent experiment that starts from fresh capital.
No equity is carried between windows and no asset is dropped because it looks
bad: a window where the provider has no data is reported as such rather than
silently replaced.

Parameters are fixed before the matrix runs. There is no search, no tuning and
no selection of the best cell.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from argos.comparison import Comparison, ExperimentRow, run_comparison
from argos.datafeed import load_prices
from argos.errors import ArgosError
from argos.strategy import StrategySpec, default_specs

# Fixed experiment universe. Chosen for liquidity and long histories, not
# because any strategy is known to work on them.
DEFAULT_TICKERS = ("SPY", "QQQ", "AAPL", "MSFT")

# Fixed evaluation windows. Each is an independent experiment.
DEFAULT_WINDOWS = (
    (dt.date(2010, 1, 1), dt.date(2014, 12, 31)),
    (dt.date(2015, 1, 1), dt.date(2019, 12, 31)),
    (dt.date(2020, 1, 1), dt.date(2025, 12, 31)),
)


@dataclass
class MatrixResult:
    rows: list[ExperimentRow]
    skipped: list[str]

    def by_strategy(self) -> dict[str, list[ExperimentRow]]:
        grouped: dict[str, list[ExperimentRow]] = {}
        for row in self.rows:
            grouped.setdefault(row.strategy_key, []).append(row)
        return grouped


def _rows_from_comparison(
    comparison: Comparison, ticker: str, scenario_name: str
) -> list[ExperimentRow]:
    rows = []
    for run in comparison.runs:
        m = run.metrics
        rows.append(
            ExperimentRow(
                ticker=ticker,
                requested_start=comparison.requested_period.split(" -> ")[0],
                requested_end=comparison.requested_period.split(" -> ")[1],
                common_start=f"{comparison.common_start:%Y-%m-%d}",
                common_end=f"{comparison.common_end:%Y-%m-%d}",
                strategy_key=run.key,
                strategy_label=run.label,
                initial_capital=comparison.initial_capital,
                final_equity=m.final_equity,
                total_return=m.total_return,
                cagr=m.cagr,
                annualized_volatility=m.annualized_volatility,
                sharpe_ratio=m.sharpe_ratio,
                sortino_ratio=m.sortino_ratio,
                max_drawdown=m.max_drawdown,
                calmar_ratio=m.calmar_ratio,
                exposure=m.exposure,
                round_trips=m.round_trips,
                executions=m.executions,
                total_commission=m.total_commission,
                commission_pct=run.result.commission_pct,
                slippage_pct=run.result.slippage_pct,
                scenario=scenario_name,
                trades_inside_window=run.trades_inside_window,
            )
        )
    return rows


def run_matrix(
    tickers: tuple[str, ...] = DEFAULT_TICKERS,
    windows: tuple[tuple[dt.date, dt.date], ...] = DEFAULT_WINDOWS,
    specs: list[StrategySpec] | None = None,
    initial_capital: float = 10_000.0,
    scenarios: tuple = (),
    cache_dir=None,
    offline: bool = False,
    annual_rf: float = 0.0,
) -> MatrixResult:
    """Run the full grid. Missing data is reported in `skipped`, never replaced."""
    specs = specs or default_specs()
    if not scenarios:
        from argos.comparison import SCENARIOS

        scenarios = (SCENARIOS[0],)

    rows: list[ExperimentRow] = []
    skipped: list[str] = []

    for ticker in tickers:
        for start, end in windows:
            for scenario in scenarios:
                label = f"{ticker} {start:%Y-%m-%d}..{end:%Y-%m-%d} [{scenario.name}]"
                try:
                    prices = load_prices(
                        ticker=ticker,
                        start=start,
                        end=end,
                        cache_dir=cache_dir,
                        offline=offline,
                    )
                    comparison = run_comparison(
                        prices,
                        specs,
                        initial_capital=initial_capital,
                        commission_pct=scenario.commission_pct,
                        slippage_pct=scenario.slippage_pct,
                        annual_rf=annual_rf,
                    )
                    comparison = Comparison(
                        ticker=ticker,
                        requested_start=comparison.requested_start,
                        requested_end=comparison.requested_end,
                        common_start=comparison.common_start,
                        common_end=comparison.common_end,
                        initial_capital=comparison.initial_capital,
                        runs=comparison.runs,
                        benchmark_key=comparison.benchmark_key,
                    )
                except ArgosError as exc:
                    skipped.append(f"{label}: {exc}")
                    continue

                rows.extend(_rows_from_comparison(comparison, ticker, scenario.name))

    return MatrixResult(rows=rows, skipped=skipped)


def to_csv(matrix: MatrixResult, path) -> int:
    """Write the matrix to CSV. No pandas index column is ever emitted."""
    from argos.comparison import MATRIX_COLUMNS
    from argos.errors import ConfigurationError

    frame = pd.DataFrame([row.as_dict() for row in matrix.rows])
    if frame.empty:
        frame = pd.DataFrame(columns=MATRIX_COLUMNS)
    frame = frame.reindex(columns=[c for c in MATRIX_COLUMNS if c in frame.columns])

    path = Path(path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
    except OSError as exc:
        raise ConfigurationError(f"could not write the CSV export to {path}: {exc}") from exc
    return len(frame)


@dataclass
class StrategySummary:
    key: str
    label: str
    experiments: int
    median_cagr: float
    median_sharpe: float
    median_max_drawdown: float
    positive_return_experiments: int
    outperforming_benchmark: int
    worst: str
    best: str


def _extreme(rows: list[ExperimentRow], maximise: bool) -> str:
    if not rows:
        return "n/a"
    pick = max if maximise else min
    row = pick(rows, key=lambda r: r.total_return)
    span = f"{row.requested_start[:4]}..{row.requested_end[:4]}"
    return f"{row.ticker} {span} ({row.total_return:+.2%})"


def summarise(matrix: MatrixResult, benchmark_key: str = "buyhold") -> list[StrategySummary]:
    """Descriptive aggregation only.

    Deliberately no composite score and no ranking: combining distinct metrics
    into a single number would imply a comparison the data does not support.
    """
    by_key = matrix.by_strategy()
    benchmark_by_cell = {
        (r.ticker, r.requested_start, r.requested_end, r.scenario): r
        for r in by_key.get(benchmark_key, [])
    }

    summaries = []
    for key, rows in by_key.items():
        beats = 0
        for r in rows:
            cell = benchmark_by_cell.get(
                (r.ticker, r.requested_start, r.requested_end, r.scenario)
            )
            if cell is not None and r.total_return > cell.total_return:
                beats += 1
        summaries.append(
            StrategySummary(
                key=key,
                label=rows[0].strategy_label,
                experiments=len(rows),
                median_cagr=float(pd.Series([r.cagr for r in rows]).median()),
                median_sharpe=float(pd.Series([r.sharpe_ratio for r in rows]).median()),
                median_max_drawdown=float(pd.Series([r.max_drawdown for r in rows]).median()),
                positive_return_experiments=sum(1 for r in rows if r.total_return > 0),
                outperforming_benchmark=beats,
                worst=_extreme(rows, maximise=False),
                best=_extreme(rows, maximise=True),
            )
        )
    return summaries