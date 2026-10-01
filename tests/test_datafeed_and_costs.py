"""Data feed: validation, caching and the cost model.

The critical guarantee here is that malformed data never reaches disk. A cache
poisoned by a bad download would silently corrupt every later run.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from argos.datafeed import _cache_path, _validate, load_prices
from argos.engine import run_backtest
from argos.errors import CorruptCache, InvalidMarketData, OfflineUnavailable
from argos.strategy import BuyAndHold, Momentum, SignalTiming, SmaCrossover


def _frame(close: list[float], freq: str = "B") -> pd.DataFrame:
    index = pd.date_range("2020-01-01", periods=len(close), freq=freq, tz="UTC")
    return pd.DataFrame(
        {
            "Open": close,
            "High": [c * 1.01 for c in close],
            "Low": [c * 0.99 for c in close],
            "Close": close,
            "Volume": [1_000.0] * len(close),
        },
        index=index,
    )


def _write_cache(cache_dir: Path, close: list[float], name: str = "TEST") -> Path:
    df = _frame(close)
    df.index.name = "Date"
    path = _cache_path(cache_dir, name, dt.date(2020, 1, 1), dt.date(2020, 12, 31))
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)
    return path


def _load(tmp_path: Path, offline: bool = True, end: dt.date = dt.date(2020, 12, 31)):
    return load_prices("TEST", dt.date(2020, 1, 1), end, cache_dir=tmp_path, offline=offline)


# --- offline and cache behaviour ----------------------------------------


def test_offline_mode_fails_with_a_clear_message(tmp_path: Path) -> None:
    with pytest.raises(OfflineUnavailable, match="offline"):
        _load(tmp_path, offline=True)


def test_cached_data_is_returned_and_typed(tmp_path: Path) -> None:
    _write_cache(tmp_path, [100.0, 101.0, 102.0])
    df = _load(tmp_path)
    assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert df.index.tz is not None
    assert df["Close"].iloc[-1] == 102.0
    assert df.index.is_monotonic_increasing
    assert df.index.is_unique


def test_valid_cache_is_reused_without_touching_the_network(tmp_path: Path, monkeypatch) -> None:
    _write_cache(tmp_path, [100.0, 101.0, 102.0])

    def explode(*_a, **_k):
        raise AssertionError("the network must not be used when a valid cache exists")

    monkeypatch.setattr("argos.datafeed._download", explode)
    assert len(_load(tmp_path, offline=False)) == 3


def test_corrupt_cache_raises_corrupt_cache_not_a_parse_crash(tmp_path: Path) -> None:
    path = _write_cache(tmp_path, [100.0, 101.0])
    path.write_text("this is not,,,a\ncsv file\n\"unterminated\n")
    with pytest.raises(CorruptCache, match="could not be parsed"):
        _load(tmp_path)


def test_empty_cache_file_raises_corrupt_cache(tmp_path: Path) -> None:
    path = _cache_path(tmp_path, "TEST", dt.date(2020, 1, 1), dt.date(2020, 12, 31))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("")
    with pytest.raises(CorruptCache):
        _load(tmp_path)


# --- validation ---------------------------------------------------------


def test_missing_column_fails_validation(tmp_path: Path) -> None:
    path = _write_cache(tmp_path, [100.0, 101.0])
    raw = pd.read_csv(path, index_col=0)
    raw.drop(columns=["Volume"]).to_csv(path)
    with pytest.raises(InvalidMarketData, match="missing expected columns"):
        _load(tmp_path)


@pytest.mark.parametrize(
    ("column", "value", "pattern"),
    [
        ("Open", 0.0, "non-positive prices"),
        ("Close", -5.0, "non-positive prices"),
        ("Low", 0.0, "non-positive prices"),
        ("High", -1.0, "non-positive prices"),
        ("Volume", -1.0, "negative volume"),
    ],
)
def test_invalid_values_fail_validation(
    tmp_path: Path, column: str, value: float, pattern: str
) -> None:
    path = _write_cache(tmp_path, [100.0, 101.0, 102.0])
    raw = pd.read_csv(path, index_col=0)
    raw.loc[raw.index[1], column] = value
    raw.to_csv(path)
    with pytest.raises(InvalidMarketData, match=pattern):
        _load(tmp_path)


def test_low_above_high_fails_validation(tmp_path: Path) -> None:
    path = _write_cache(tmp_path, [100.0, 101.0])
    raw = pd.read_csv(path, index_col=0)
    raw.loc[raw.index[1], "Low"] = 500.0
    raw.to_csv(path)
    with pytest.raises(InvalidMarketData, match="Low > High"):
        _load(tmp_path)


def test_open_outside_the_bar_range_fails_validation(tmp_path: Path) -> None:
    path = _write_cache(tmp_path, [100.0, 101.0])
    raw = pd.read_csv(path, index_col=0)
    raw.loc[raw.index[1], "Open"] = 9_999.0
    raw.to_csv(path)
    with pytest.raises(InvalidMarketData, match="Open falls outside"):
        _load(tmp_path)


def test_close_outside_the_bar_range_fails_validation(tmp_path: Path) -> None:
    path = _write_cache(tmp_path, [100.0, 101.0])
    raw = pd.read_csv(path, index_col=0)
    raw.loc[raw.index[1], "Close"] = 9_999.0
    raw.to_csv(path)
    with pytest.raises(InvalidMarketData, match="Close falls outside"):
        _load(tmp_path)


def test_duplicated_timestamps_fail_validation(tmp_path: Path) -> None:
    path = _write_cache(tmp_path, [100.0, 101.0])
    raw = pd.read_csv(path, index_col=0)
    duplicated = pd.concat([raw, raw.iloc[[0]]]).sort_index()
    duplicated.to_csv(path)
    with pytest.raises(InvalidMarketData, match="duplicated timestamps"):
        _load(tmp_path)


def test_unsorted_index_fails_validation() -> None:
    df = _frame([100.0, 101.0, 102.0])
    with pytest.raises(InvalidMarketData, match="not sorted"):
        _validate(df.iloc[::-1], "TEST")


def test_missing_price_is_an_error_and_is_never_filled() -> None:
    """A hole in the data is a fact about the provider, not something to repair."""
    df = _frame([100.0, 101.0, 102.0])
    df.iloc[1, df.columns.get_loc("Close")] = np.nan
    with pytest.raises(InvalidMarketData, match="incomplete OHLCV"):
        _validate(df, "TEST")


def test_empty_period_fails_with_clear_error(tmp_path: Path) -> None:
    """A cached file whose bars fall outside the requested window must fail loudly.

    The filename encodes the requested window, so it has to be written for that
    window while holding bars from a different one.
    """
    path = _cache_path(tmp_path, "TEST", dt.date(2021, 1, 1), dt.date(2021, 12, 31))
    path.parent.mkdir(parents=True, exist_ok=True)
    df = _frame([100.0, 101.0])
    df.index.name = "Date"
    df.to_csv(path)
    with pytest.raises(InvalidMarketData, match="no bars"):
        load_prices("TEST", dt.date(2021, 1, 1), dt.date(2021, 12, 31),
                    cache_dir=tmp_path, offline=True)


def test_a_single_bar_dataset_is_valid(tmp_path: Path) -> None:
    """One bar is thin but not malformed: the engine, not the feed, judges that."""
    _write_cache(tmp_path, [100.0])
    assert len(_load(tmp_path)) == 1


def test_bad_download_is_never_written_to_the_cache(tmp_path: Path, monkeypatch) -> None:
    """The guarantee that matters: validation runs before persistence."""
    monkeypatch.setattr(
        "argos.datafeed._download",
        lambda *a, **k: _frame([100.0, 101.0, 102.0]).assign(Close=[100.0, -1.0, 102.0]),
    )
    path = _cache_path(tmp_path, "TEST", dt.date(2020, 1, 1), dt.date(2020, 12, 31))
    with pytest.raises(InvalidMarketData, match="non-positive prices"):
        load_prices("TEST", dt.date(2020, 1, 1), dt.date(2020, 12, 31), cache_dir=tmp_path)
    assert not path.exists(), "invalid data must never reach the cache"


def test_good_download_is_written_to_the_cache(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("argos.datafeed._download", lambda *a, **k: _frame([100.0, 101.0, 102.0]))
    path = _cache_path(tmp_path, "TEST", dt.date(2020, 1, 1), dt.date(2020, 12, 31))
    df = load_prices("TEST", dt.date(2020, 1, 1), dt.date(2020, 12, 31), cache_dir=tmp_path)
    assert path.exists()
    assert len(df) == 3
    # The persisted file must itself be valid, so a reload succeeds offline.
    assert len(load_prices("TEST", dt.date(2020, 1, 1), dt.date(2020, 12, 31),
                           cache_dir=tmp_path, offline=True)) == 3


def test_download_failure_is_reported_as_download_error(tmp_path: Path, monkeypatch) -> None:
    from argos.errors import DownloadError

    def boom(*_a, **_k):
        raise RuntimeError("connection reset")

    monkeypatch.setattr("argos.datafeed._download", boom)
    with pytest.raises(DownloadError, match="connection reset"):
        load_prices("TEST", dt.date(2020, 1, 1), dt.date(2020, 12, 31), cache_dir=tmp_path)


def test_provider_returning_nothing_is_reported_clearly(tmp_path: Path, monkeypatch) -> None:
    from argos.errors import DownloadError

    monkeypatch.setattr("argos.datafeed._download", lambda *a, **k: pd.DataFrame())
    with pytest.raises(DownloadError, match="no rows"):
        load_prices("TEST", dt.date(2020, 1, 1), dt.date(2020, 12, 31), cache_dir=tmp_path)


def test_lowercase_provider_columns_are_normalised() -> None:
    df = _frame([100.0, 101.0])
    out = _validate(df.rename(columns=str.lower), "TEST")
    assert list(out.columns) == ["Open", "High", "Low", "Close", "Volume"]


def test_multiindex_columns_from_the_provider_are_flattened(tmp_path: Path, monkeypatch) -> None:
    df = _frame([100.0, 101.0, 102.0])
    flat = df.set_axis(pd.MultiIndex.from_product([df.columns, ["TEST"]]), axis=1)

    def fake_download(ticker, start, end):
        return flat

    monkeypatch.setattr("argos.datafeed._download", fake_download)
    out = load_prices("TEST", dt.date(2020, 1, 1), dt.date(2020, 12, 31), cache_dir=tmp_path)
    assert list(out.columns) == ["Open", "High", "Low", "Close", "Volume"]


# --- cost model ---------------------------------------------------------


def test_buy_and_hold_uses_the_same_cost_model(tmp_path: Path) -> None:
    _write_cache(tmp_path, list(np.linspace(100, 200, 60)))
    df = _load(tmp_path)

    free = run_backtest(df, BuyAndHold(), initial_capital=10_000)
    costly = run_backtest(df, BuyAndHold(), initial_capital=10_000,
                          commission_pct=0.001, slippage_pct=0.001)

    assert free.portfolio.executions == 1
    assert free.portfolio.round_trips == 0  # still open at the end
    assert costly.portfolio.total_commission > 0
    assert costly.equity_curve.iloc[-1] < free.equity_curve.iloc[-1]


def test_costs_reduce_final_equity_for_the_crossover_strategy(tmp_path: Path) -> None:
    _write_cache(tmp_path, list(np.linspace(100, 300, 80)) + list(np.linspace(300, 120, 60)))
    df = _load(tmp_path)

    free = run_backtest(df, SmaCrossover(fast=5, slow=20), initial_capital=10_000)
    costly = run_backtest(df, SmaCrossover(fast=5, slow=20), initial_capital=10_000,
                          commission_pct=0.002, slippage_pct=0.001)

    assert free.portfolio.executions == costly.portfolio.executions
    assert costly.equity_curve.iloc[-1] < free.equity_curve.iloc[-1]
    assert costly.portfolio.total_commission > 0


def test_costs_never_change_the_number_of_trades(tmp_path: Path) -> None:
    """Costs must change P&L, not behaviour."""
    _write_cache(tmp_path, list(np.linspace(100, 300, 80)) + list(np.linspace(300, 120, 60)))
    df = _load(tmp_path)

    for strategy in (SmaCrossover(fast=5, slow=20), Momentum(lookback=10), BuyAndHold()):
        free = run_backtest(df, strategy, initial_capital=10_000)
        costly = run_backtest(df, strategy, initial_capital=10_000,
                              commission_pct=0.005, slippage_pct=0.005)
        assert free.portfolio.executions == costly.portfolio.executions, strategy
        assert free.portfolio.round_trips == costly.portfolio.round_trips, strategy
        assert costly.equity_curve.iloc[-1] <= free.equity_curve.iloc[-1] + 1e-9, strategy


def test_timing_declaration_is_public_and_stable() -> None:
    assert SmaCrossover().timing is SignalTiming.DERIVED
    assert Momentum().timing is SignalTiming.DERIVED
    assert BuyAndHold().timing is SignalTiming.PREDEFINED
