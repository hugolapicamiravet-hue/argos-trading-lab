#!/usr/bin/env python3
"""ARGOS command line interface.

Subcommands:
    backtest   one strategy over the full requested period (its own semantics)
    compare    all strategies on a common evaluation window
    costs      the cost-sensitivity grid
    matrix     the multi-asset, multi-window experiment grid

Expected failures print one clear line and exit non-zero. `--debug` adds the
traceback.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from argos.comparison import SCENARIOS, run_comparison  # noqa: E402
from argos.datafeed import load_prices  # noqa: E402
from argos.engine import run_backtest  # noqa: E402
from argos.errors import ArgosError, ConfigurationError  # noqa: E402
from argos.experiment import (  # noqa: E402
    DEFAULT_TICKERS,
    DEFAULT_WINDOWS,
    run_matrix,
    summarise,
    to_csv,
)
from argos.metrics import compute_metrics  # noqa: E402
from argos.report import (  # noqa: E402
    format_comparison,
    format_individual,
    format_matrix_summary,
)
from argos.strategy import specs_by_key  # noqa: E402

EXIT_OK = 0
EXIT_ERROR = 1
# argparse itself exits with 2 for a usage error, so that code is never raised here.


def _parse_date(value: str) -> dt.date:
    try:
        return dt.date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"not a valid ISO date: {value!r}") from exc


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--ticker", default="SPY")
    p.add_argument("--start", type=_parse_date, default=dt.date(2020, 1, 1))
    p.add_argument("--end", type=_parse_date, default=dt.date(2025, 12, 31))
    p.add_argument("--initial-capital", type=float, default=10_000.0)
    p.add_argument(
        "--commission-pct", type=float, default=0.0,
        help="per-side commission fraction",
    )
    p.add_argument("--slippage-pct", type=float, default=0.0, help="per-side slippage fraction")
    p.add_argument(
        "--annual-rf", type=float, default=0.0,
        help="annual risk-free rate for Sharpe/Sortino",
    )
    p.add_argument("--cache-dir", default="data/raw")
    p.add_argument("--offline", action="store_true", help="use only the local cache")
    p.add_argument("--debug", action="store_true", help="show the full traceback on error")


def _add_strategies(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--strategies",
        default="sma,momentum,buyhold",
        help="comma-separated subset of: sma, momentum, buyhold",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="argos", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p_backtest = sub.add_parser("backtest", help="run one strategy over the full requested period")
    _add_common(p_backtest)
    _add_strategies(p_backtest)

    p_compare = sub.add_parser("compare", help="run every strategy on a common window")
    _add_common(p_compare)
    _add_strategies(p_compare)

    p_costs = sub.add_parser("costs", help="run the cost-sensitivity grid")
    _add_common(p_costs)
    _add_strategies(p_costs)

    p_matrix = sub.add_parser("matrix", help="run the multi-asset multi-window grid")
    _add_common(p_matrix)
    _add_strategies(p_matrix)
    p_matrix.add_argument("--tickers", default=",".join(DEFAULT_TICKERS))
    p_matrix.add_argument(
        "--windows",
        default=";".join(f"{s:%Y-%m-%d}:{e:%Y-%m-%d}" for s, e in DEFAULT_WINDOWS),
        help="semicolon-separated start:end pairs",
    )
    p_matrix.add_argument("--scenarios", default="A,B,C", help="comma-separated scenario names")
    p_matrix.add_argument("--export-csv", default=None)
    p_matrix.add_argument("--summary-only", action="store_true")

    return parser


def _selected_specs(args):
    """Resolve the requested strategies before any data is touched.

    A typo in a strategy name is a user error and must be reported before the
    program spends time downloading or reading a dataset.
    """
    return specs_by_key(args.strategies.split(","))


def _validate_window(args) -> None:
    if args.end <= args.start:
        raise ConfigurationError(f"--end ({args.end}) must be after --start ({args.start})")
    if args.initial_capital <= 0:
        raise ConfigurationError("--initial-capital must be positive")
    if args.commission_pct < 0 or args.slippage_pct < 0:
        raise ConfigurationError("--commission-pct and --slippage-pct must be non-negative")


def _load(args):
    return load_prices(
        ticker=args.ticker,
        start=args.start,
        end=args.end,
        cache_dir=Path(args.cache_dir),
        offline=args.offline,
    )


def _parse_windows(spec: str) -> list[tuple[dt.date, dt.date]]:
    windows = []
    for chunk in spec.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            raise ConfigurationError(f"malformed window {chunk!r}, expected start:end")
        raw_start, raw_end = chunk.split(":", 1)
        windows.append((_parse_date(raw_start), _parse_date(raw_end)))
    if not windows:
        raise ConfigurationError("no evaluation windows given")
    return windows


def cmd_backtest(args) -> int:
    _validate_window(args)
    specs = _selected_specs(args)
    prices = _load(args)

    for spec in specs:
        result = run_backtest(
            prices,
            spec.factory(**spec.params),
            initial_capital=args.initial_capital,
            commission_pct=args.commission_pct,
            slippage_pct=args.slippage_pct,
            strategy_name=spec.label,
        )
        equity = result.equity_curve
        units = result.open_positions
        metrics = compute_metrics(
            equity,
            units,
            initial_capital=args.initial_capital,
            executions=result.portfolio.executions,
            round_trips=result.portfolio.round_trips,
            total_commission=result.portfolio.total_commission,
            annual_rf=args.annual_rf,
        )
        actionable = (
            f"{prices.index[result.first_actionable_bar]:%Y-%m-%d}"
            if result.first_actionable_bar is not None
            else "never (warm-up longer than dataset)"
        )
        print(
            format_individual(
                label=spec.label,
                metrics=metrics,
                timing=result.timing.value,
                first_actionable=actionable,
                bars=len(equity),
                capital=args.initial_capital,
                commission_pct=args.commission_pct,
                slippage_pct=args.slippage_pct,
            )
        )
        print()
    return EXIT_OK


def cmd_compare(args) -> int:
    _validate_window(args)
    specs = _selected_specs(args)
    prices = _load(args)
    comparison = run_comparison(
        prices,
        specs,
        initial_capital=args.initial_capital,
        commission_pct=args.commission_pct,
        slippage_pct=args.slippage_pct,
        annual_rf=args.annual_rf,
    )
    comparison = _with_ticker(comparison, args.ticker)
    print(format_comparison(comparison))
    return EXIT_OK


def cmd_costs(args) -> int:
    _validate_window(args)
    specs = _selected_specs(args)
    prices = _load(args)

    for scenario in SCENARIOS:
        comparison = run_comparison(
            prices,
            specs,
            initial_capital=args.initial_capital,
            commission_pct=scenario.commission_pct,
            slippage_pct=scenario.slippage_pct,
            annual_rf=args.annual_rf,
        )
        comparison = _with_ticker(comparison, args.ticker)
        print(
            f"Scenario {scenario.name}   "
            f"commission={scenario.commission_pct}  slippage={scenario.slippage_pct}"
        )
        print(format_comparison(comparison))
        print()

    print("Cost scenarios are synthetic sensitivity settings, not models of any broker.")
    return EXIT_OK


def cmd_matrix(args) -> int:
    _validate_window(args)
    specs = _selected_specs(args)
    tickers = tuple(t.strip().upper() for t in args.tickers.split(",") if t.strip())
    windows = _parse_windows(args.windows)
    wanted = {s.strip().upper() for s in args.scenarios.split(",") if s.strip()}
    scenarios = tuple(s for s in SCENARIOS if s.name.split(":")[0] in wanted)
    if not scenarios:
        raise ConfigurationError(
            f"unknown scenario(s) {sorted(wanted)}; available: "
            + ", ".join(s.name.split(':')[0] for s in SCENARIOS)
        )

    matrix = run_matrix(
        tickers=tickers,
        windows=tuple(windows),
        specs=specs,
        initial_capital=args.initial_capital,
        scenarios=scenarios,
        cache_dir=Path(args.cache_dir),
        offline=args.offline,
        annual_rf=args.annual_rf,
    )

    if not args.summary_only:
        frame = pd.DataFrame([r.as_dict() for r in matrix.rows])
        if not frame.empty:
            with pd.option_context("display.width", 250, "display.max_columns", 50):
                print(frame.to_string(index=False))
            print()

    print(format_matrix_summary(matrix, summarise(matrix)))

    if args.export_csv:
        n = to_csv(matrix, args.export_csv)
        print(f"\nwrote {n} rows to {args.export_csv}")
    return EXIT_OK


def _with_ticker(comparison, ticker: str):
    from argos.comparison import Comparison

    return Comparison(
        ticker=ticker,
        requested_start=comparison.requested_start,
        requested_end=comparison.requested_end,
        common_start=comparison.common_start,
        common_end=comparison.common_end,
        initial_capital=comparison.initial_capital,
        runs=comparison.runs,
        benchmark_key=comparison.benchmark_key,
    )


HANDLERS = {
    "backtest": cmd_backtest,
    "compare": cmd_compare,
    "costs": cmd_costs,
    "matrix": cmd_matrix,
}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return HANDLERS[args.command](args)
    except ArgosError as exc:
        if getattr(args, "debug", False):
            raise
        print(f"error [{type(exc).__name__}]: {exc}", file=sys.stderr)
        return EXIT_ERROR
    except OSError as exc:
        # Disk full, read-only path, bad permissions: still a one-line message.
        if getattr(args, "debug", False):
            raise
        print(f"error [FileSystemError]: {exc}", file=sys.stderr)
        return EXIT_ERROR
    except (KeyboardInterrupt, BrokenPipeError):  # pragma: no cover
        return EXIT_ERROR


if __name__ == "__main__":
    raise SystemExit(main())