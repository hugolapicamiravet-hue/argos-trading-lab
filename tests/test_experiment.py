"""Experiment matrix: completeness, reproducibility and honest reporting.

The matrix exists to make a small claim about many situations instead of a big
claim about one. That only works if every cell is reported, including the ones
that failed, and if the same inputs always produce the same numbers.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from argos.comparison import MATRIX_COLUMNS, SCENARIOS, run_comparison
from argos.errors import DownloadError, InvalidMarketData
from argos.experiment import (
    DEFAULT_TICKERS,
    DEFAULT_WINDOWS,
    StrategySummary,
    run_matrix,
    summarise,
    to_csv,
)
from argos.strategy import default_specs


def _prices(seed: int, n: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100.0 * np.cumprod(1.0 + rng.normal(0.0004, 0.01, n))
    return pd.DataFrame(
        {
            "Open": close * 0.999,
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Volume": 1e6,
        },
        index=pd.date_range("2015-01-01", periods=n, freq="B", tz="UTC"),
    )


@pytest.fixture
def fake_feed(monkeypatch):
    """Serve synthetic data, and fail loudly for any ticker we did not prepare."""
    available = {"SPY": 0, "QQQ": 1, "AAPL": 2, "MSFT": 3}

    def fake_load_prices(ticker, start, end, cache_dir=None, offline=False):
        if ticker not in available:
            raise DownloadError(f"no data for {ticker}")
        return _prices(available[ticker])

    monkeypatch.setattr("argos.experiment.load_prices", fake_load_prices)
    return fake_load_prices


WINDOWS = (
    (dt.date(2015, 1, 1), dt.date(2016, 6, 30)),
    (dt.date(2016, 7, 1), dt.date(2018, 6, 30)),
)


def test_matrix_produces_one_row_per_cell(fake_feed) -> None:
    matrix = run_matrix(
        tickers=("SPY", "QQQ"),
        windows=WINDOWS,
        scenarios=(SCENARIOS[0],),
    )
    # 2 tickers x 2 windows x 1 scenario x 3 strategies
    assert len(matrix.rows) == 12
    assert matrix.skipped == []


def test_every_row_starts_from_the_same_fresh_capital(fake_feed) -> None:
    matrix = run_matrix(tickers=("SPY",), windows=WINDOWS, scenarios=(SCENARIOS[0],))
    assert all(row.initial_capital == 10_000.0 for row in matrix.rows)
    assert all(row.requested_start != row.common_start for row in matrix.rows)


def test_matrix_is_reproducible(fake_feed) -> None:
    first = run_matrix(tickers=("SPY",), windows=WINDOWS, scenarios=(SCENARIOS[0],))
    second = run_matrix(tickers=("SPY",), windows=WINDOWS, scenarios=(SCENARIOS[0],))
    a = pd.DataFrame([r.as_dict() for r in first.rows])
    b = pd.DataFrame([r.as_dict() for r in second.rows])
    pd.testing.assert_frame_equal(a, b)


def test_a_missing_window_is_reported_and_never_substituted(fake_feed) -> None:
    """Dropping an asset that performs badly would bias the whole summary."""
    matrix = run_matrix(
        tickers=("SPY", "NOPE"),
        windows=WINDOWS,
        scenarios=(SCENARIOS[0],),
    )
    assert len(matrix.rows) == 6  # only SPY
    assert len(matrix.skipped) == 2  # both windows of the unknown ticker
    assert all("NOPE" in note for note in matrix.skipped)
    assert all("no data" in note for note in matrix.skipped)


def test_a_window_too_short_for_warmup_is_reported_not_dropped(fake_feed, monkeypatch) -> None:
    monkeypatch.setattr("argos.experiment.load_prices", lambda *a, **k: _prices(0, n=60))
    matrix = run_matrix(tickers=("SPY",), windows=WINDOWS, scenarios=(SCENARIOS[0],))
    assert matrix.rows == []
    assert len(matrix.skipped) == 2
    assert all("never became actionable" in note for note in matrix.skipped)


def test_scenarios_change_costs_but_not_behaviour(fake_feed) -> None:
    free = run_matrix(tickers=("SPY",), windows=WINDOWS, scenarios=(SCENARIOS[0],))
    costly = run_matrix(tickers=("SPY",), windows=WINDOWS, scenarios=(SCENARIOS[1],))

    free_by_cell = {(r.ticker, r.requested_start, r.strategy_key): r for r in free.rows}
    for row in costly.rows:
        other = free_by_cell[(row.ticker, row.requested_start, row.strategy_key)]
        assert row.executions == other.executions
        assert row.round_trips == other.round_trips
        assert row.total_commission >= other.total_commission
        if row.trades_inside_window:
            assert row.final_equity <= other.final_equity + 1e-9


def test_csv_export_has_no_index_column(fake_feed, tmp_path) -> None:
    matrix = run_matrix(tickers=("SPY",), windows=WINDOWS, scenarios=(SCENARIOS[0],))
    path = tmp_path / "matrix.csv"
    written = to_csv(matrix, path)

    frame = pd.read_csv(path)
    assert written == len(frame) == len(matrix.rows)
    assert "Unnamed: 0" not in frame.columns
    assert list(frame.columns) == MATRIX_COLUMNS
    assert "date" not in [c.lower() for c in frame.columns]
    # The period columns are plain strings, not a stray index.
    assert frame["common_start"].str.match(r"^\d{4}-\d{2}-\d{2}$").all()


def test_csv_export_creates_missing_directories(fake_feed, tmp_path) -> None:
    """A user must not have to mkdir before exporting."""
    matrix = run_matrix(tickers=("SPY",), windows=WINDOWS, scenarios=(SCENARIOS[0],))
    out = tmp_path / "nested" / "deeper" / "matrix.csv"
    assert to_csv(matrix, out) == len(matrix.rows)
    assert out.exists()


def test_csv_export_failure_is_reported_as_configuration(fake_feed, tmp_path) -> None:
    from argos.errors import ConfigurationError

    matrix = run_matrix(tickers=("SPY",), windows=WINDOWS, scenarios=(SCENARIOS[0],))
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory")
    with pytest.raises(ConfigurationError, match="could not write the CSV export"):
        to_csv(matrix, blocker / "matrix.csv")


def test_csv_export_of_an_empty_matrix_still_has_headers(tmp_path) -> None:
    from argos.experiment import MatrixResult

    path = tmp_path / "empty.csv"
    assert to_csv(MatrixResult(rows=[], skipped=["nothing to do"]), path) == 0
    frame = pd.read_csv(path)
    assert list(frame.columns) == MATRIX_COLUMNS
    assert len(frame) == 0


def test_summary_is_descriptive_and_has_no_composite_score(fake_feed) -> None:
    matrix = run_matrix(tickers=("SPY", "QQQ"), windows=WINDOWS, scenarios=(SCENARIOS[0],))
    summaries = summarise(matrix)

    assert {s.key for s in summaries} == {"sma", "momentum", "buyhold"}
    for s in summaries:
        assert isinstance(s, StrategySummary)
        assert s.experiments == 4
        assert 0 <= s.positive_return_experiments <= 4
        assert 0 <= s.outperforming_benchmark <= 4
        assert "n/a" not in s.worst and "n/a" not in s.best
        # The best case beats the worst case by construction.
        assert s.best != s.worst or s.experiments == 1


def test_summary_counts_beats_against_the_benchmark(fake_feed) -> None:
    matrix = run_matrix(tickers=("SPY",), windows=WINDOWS, scenarios=(SCENARIOS[0],))
    summaries = {s.key: s for s in summarise(matrix)}
    # Buy & Hold can never beat itself.
    assert summaries["buyhold"].outperforming_benchmark == 0


def test_default_universe_is_fixed_and_documented() -> None:
    assert DEFAULT_TICKERS == ("SPY", "QQQ", "AAPL", "MSFT")
    assert len(DEFAULT_WINDOWS) == 3
    assert DEFAULT_WINDOWS[0] == (dt.date(2010, 1, 1), dt.date(2014, 12, 31))
    assert DEFAULT_WINDOWS[-1] == (dt.date(2020, 1, 1), dt.date(2025, 12, 31))
    for start, end in DEFAULT_WINDOWS:
        assert start < end


def test_matrix_defaults_to_the_frictionless_scenario(fake_feed) -> None:
    matrix = run_matrix(tickers=("SPY",), windows=WINDOWS)
    assert all(row.scenario == SCENARIOS[0].name for row in matrix.rows)
    assert all(row.commission_pct == 0.0 for row in matrix.rows)


def test_every_cell_matches_a_direct_comparison_run(fake_feed) -> None:
    """The matrix must not quietly apply different maths than the single run."""
    matrix = run_matrix(tickers=("SPY",), windows=WINDOWS[:1], scenarios=(SCENARIOS[0],))
    direct = run_comparison(_prices(0), default_specs())

    for row in matrix.rows:
        run = direct.by_key(row.strategy_key)
        assert row.total_return == pytest.approx(run.metrics.total_return)
        assert row.max_drawdown == pytest.approx(run.metrics.max_drawdown)
        assert row.exposure == pytest.approx(run.metrics.exposure)


def test_scenarios_expose_their_settings() -> None:
    assert [s.name.split(":")[0] for s in SCENARIOS] == ["A", "B", "C"]
    assert SCENARIOS[0].commission_pct == 0.0
    assert SCENARIOS[1].commission_pct == 0.0005
    assert SCENARIOS[2].commission_pct == 0.001
    for scenario in SCENARIOS:
        assert scenario.commission_pct == scenario.slippage_pct
        assert scenario.commission_pct >= 0


def test_invalid_market_data_is_caught_by_the_matrix(fake_feed, monkeypatch) -> None:
    def bad(*_a, **_k):
        raise InvalidMarketData("Close contains a non-positive price")

    monkeypatch.setattr("argos.experiment.load_prices", bad)
    matrix = run_matrix(tickers=("SPY",), windows=WINDOWS, scenarios=(SCENARIOS[0],))
    assert matrix.rows == []
    assert all("non-positive price" in note for note in matrix.skipped)
