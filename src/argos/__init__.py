"""ARGOS · educational quantitative backtesting laboratory.

Long-only simulation of trading strategies on historical daily data. No broker
connectivity, no order routing, no live trading. Everything is virtual money.
"""

from __future__ import annotations

__version__ = "0.2.0"

from argos.comparison import (
    SCENARIOS,
    Comparison,
    ComparisonRun,
    CostScenario,
    ExperimentRow,
    common_window,
    rebase,
    run_comparison,
)
from argos.datafeed import load_prices
from argos.engine import BacktestResult, run_backtest
from argos.errors import ArgosError
from argos.experiment import MatrixResult, run_matrix, summarise, to_csv
from argos.metrics import Metrics, cagr, compute_metrics, max_drawdown, total_return
from argos.portfolio import Portfolio
from argos.report import format_comparison, format_individual
from argos.strategy import (
    BuyAndHold,
    Momentum,
    NeverTrade,
    SignalTiming,
    SmaCrossover,
    default_specs,
)

__all__ = [
    "ArgosError",
    "BacktestResult",
    "BuyAndHold",
    "Comparison",
    "ComparisonRun",
    "CostScenario",
    "ExperimentRow",
    "MatrixResult",
    "Metrics",
    "Momentum",
    "NeverTrade",
    "Portfolio",
    "SCENARIOS",
    "SignalTiming",
    "SmaCrossover",
    "__version__",
    "cagr",
    "common_window",
    "compute_metrics",
    "default_specs",
    "format_comparison",
    "format_individual",
    "load_prices",
    "max_drawdown",
    "rebase",
    "run_backtest",
    "run_comparison",
    "run_matrix",
    "summarise",
    "to_csv",
    "total_return",
]