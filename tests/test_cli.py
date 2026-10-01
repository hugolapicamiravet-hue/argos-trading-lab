"""CLI behaviour: exit codes, one-line errors and reproducible output.

The CLI is the contract with the user. A traceback on a bad ticker is a bug,
not a style issue, so every expected failure is asserted to produce one clear
line and a non-zero exit code.
"""

from __future__ import annotations

import datetime as dt
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from argos.datafeed import _cache_path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_backtest.py"
START, END = dt.date(2015, 1, 1), dt.date(2016, 12, 31)


@pytest.fixture
def cache(tmp_path: Path) -> Path:
    """A 500-bar synthetic series, long enough for every strategy to warm up."""
    rng = np.random.default_rng(5)
    close = 100.0 * np.cumprod(1.0 + rng.normal(0.0005, 0.011, 520))
    index = pd.date_range(START, periods=520, freq="B", tz="UTC")
    df = pd.DataFrame(
        {
            "Open": close * 0.999,
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Volume": 1e6,
        },
        index=index,
    )
    df.index.name = "Date"
    path = _cache_path(tmp_path, "SPY", START, END)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)
    return tmp_path


def _args(cache_dir: Path, *extra: str) -> list[str]:
    return [
        "--ticker", "SPY",
        "--start", START.isoformat(),
        "--end", END.isoformat(),
        "--cache-dir", str(cache_dir),
        "--offline",
        *extra,
    ]


def run(*argv: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *argv], capture_output=True, text=True, timeout=180
    )


# --- success paths -------------------------------------------------------


def test_backtest_prints_a_report_for_every_strategy(cache: Path) -> None:
    result = run("backtest", *_args(cache))
    assert result.returncode == 0, result.stderr
    for label in ("SMA 20/50", "Momentum 126", "Buy & Hold"):
        assert label in result.stdout
    assert "total return" in result.stdout
    assert "CAGR" in result.stdout
    assert "maximum drawdown" in result.stdout
    assert "signal timing" in result.stdout
    assert "derived" in result.stdout and "predefined" in result.stdout


def test_backtest_can_select_a_single_strategy(cache: Path) -> None:
    result = run("backtest", *_args(cache, "--strategies", "buyhold"))
    assert result.returncode == 0, result.stderr
    assert "Buy & Hold" in result.stdout
    assert "SMA 20/50" not in result.stdout


def test_compare_shows_both_periods_and_every_strategy(cache: Path) -> None:
    result = run("compare", *_args(cache))
    assert result.returncode == 0, result.stderr
    flat = " ".join(result.stdout.split())  # footnotes wrap to the table width
    assert "requested period" in result.stdout
    assert "common evaluation period" in result.stdout
    assert "excess return" in result.stdout
    assert "Return and risk metrics cover the common evaluation period only" in flat
    assert result.stdout.index("SMA 20/50") < result.stdout.index("Buy & Hold")


def test_costs_runs_all_three_scenarios(cache: Path) -> None:
    result = run("costs", *_args(cache))
    assert result.returncode == 0, result.stderr
    for letter in ("Scenario A", "Scenario B", "Scenario C"):
        assert letter in result.stdout
    assert "synthetic sensitivity settings" in result.stdout


def test_matrix_writes_a_csv_without_an_index_column(cache: Path, tmp_path: Path) -> None:
    out = tmp_path / "out.csv"
    result = run(
        "matrix",
        *_args(cache, "--tickers", "SPY", "--windows", "2015-01-01:2016-12-31",
               "--scenarios", "A", "--summary-only", "--export-csv", str(out)),
    )
    assert result.returncode == 0, result.stderr
    assert "experiment matrix summary" in result.stdout
    assert "descriptive" in result.stdout

    frame = pd.read_csv(out)
    assert len(frame) == 3  # one strategy per row
    assert "Unnamed: 0" not in frame.columns
    assert "status" in frame.columns and "note" in frame.columns


def test_matrix_reports_windows_it_could_not_run(cache: Path) -> None:
    result = run(
        "matrix",
        *_args(cache, "--tickers", "SPY,MSFT", "--windows", "2015-01-01:2016-12-31",
               "--scenarios", "A", "--summary-only"),
    )
    assert result.returncode == 0, result.stderr
    assert "skipped" in result.stdout
    assert "MSFT" in result.stdout


def test_output_is_byte_identical_across_runs(cache: Path) -> None:
    first = run("compare", *_args(cache))
    second = run("compare", *_args(cache))
    assert first.stdout == second.stdout


def test_costs_change_the_reported_commission(cache: Path) -> None:
    free = run("compare", *_args(cache, "--commission-pct", "0"))
    costly = run("compare", *_args(cache, "--commission-pct", "0.001", "--slippage-pct", "0.001"))
    assert free.returncode == 0 and costly.returncode == 0
    # A cost table is meaningless without the cost assumption on it.
    assert "commission per side" in free.stdout
    assert "slippage per side" in free.stdout
    assert "0.00%" in free.stdout
    assert "0.10%" in costly.stdout


def test_individual_report_names_its_cost_parameters(cache: Path) -> None:
    result = run("backtest", *_args(cache, "--commission-pct", "0.0005",
                                    "--slippage-pct", "0.0005"))
    assert result.returncode == 0, result.stderr
    assert "commission_pct" in result.stdout
    assert "slippage_pct" in result.stdout
    assert "0.05%" in result.stdout


# --- failure paths -------------------------------------------------------


def test_missing_cache_in_offline_mode_is_one_clear_line() -> None:
    result = run("backtest", "--ticker", "SPY", "--cache-dir", "/nonexistent/argos", "--offline")
    assert result.returncode == 1
    assert "error [OfflineUnavailable]" in result.stderr
    assert "Traceback" not in result.stderr
    assert len(result.stderr.strip().splitlines()) == 1


def test_unknown_strategy_is_reported_not_traced() -> None:
    result = run("compare", "--strategies", "sma,crypto", "--cache-dir", "/nonexistent/argos",
                 "--offline")
    # The strategy is validated before any data is touched.
    assert result.returncode == 1
    assert "unknown strategy key" in result.stderr
    assert "Traceback" not in result.stderr


def test_unknown_scenario_is_reported() -> None:
    result = run("matrix", "--scenarios", "Z", "--cache-dir", "/nonexistent/argos", "--offline")
    assert result.returncode == 1
    assert "unknown scenario" in result.stderr
    assert "Traceback" not in result.stderr


def test_end_before_start_is_reported() -> None:
    result = run("backtest", "--start", "2020-01-01", "--end", "2019-01-01",
                 "--cache-dir", "/nonexistent/argos", "--offline")
    assert result.returncode == 1
    assert "must be after" in result.stderr


def test_negative_capital_is_reported() -> None:
    result = run("backtest", "--initial-capital", "-5", "--cache-dir", "/nonexistent/argos",
                 "--offline")
    assert result.returncode == 1
    assert "must be positive" in result.stderr


def test_malformed_date_is_a_usage_error() -> None:
    result = run("backtest", "--start", "01-01-2020")
    assert result.returncode == 2
    assert "not a valid ISO date" in result.stderr


def test_malformed_window_is_reported() -> None:
    result = run("matrix", "--windows", "2020", "--cache-dir", "/nonexistent/argos", "--offline")
    assert result.returncode == 1
    assert "expected start:end" in result.stderr


def test_no_subcommand_is_a_usage_error() -> None:
    result = run()
    assert result.returncode == 2
    assert "usage" in result.stderr.lower()


def test_help_lists_every_subcommand() -> None:
    result = run("--help")
    assert result.returncode == 0
    for command in ("backtest", "compare", "costs", "matrix"):
        assert command in result.stdout
