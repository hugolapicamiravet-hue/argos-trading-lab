"""Historical market data acquisition, validation and local caching.

Only module in ARGOS that touches the network. Everything downstream assumes a
clean, validated frame with daily bars on a tz-aware, sorted, unique index.

Validation is strict by design: malformed data raises. Prices are never
fabricated, forward-filled or interpolated. Non-trading days are simply absent
from the frame.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd

from argos.errors import (
    ArgosError,
    CorruptCache,
    DownloadError,
    InvalidMarketData,
    OfflineUnavailable,
)

REQUIRED_COLUMNS = ("Open", "High", "Low", "Close", "Volume")

# Tolerances for float comparisons on provider data.
_EPS = 1e-9


class DataValidationError(InvalidMarketData):
    """Backwards-compatible alias."""


def _cache_path(cache_dir: Path, ticker: str, start: dt.date, end: dt.date) -> Path:
    return cache_dir / f"{ticker}_{start:%Y%m%d}_{end:%Y%m%d}.csv"


def _canonical_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise provider/cached column names to the canonical set.

    Some providers return one column per ticker under a second header level
    (`df.columns` is then a MultiIndex); yfinance returns plain `Close`. Both
    shapes are flattened here, at the validation boundary, so the rest of the
    code only ever sees the five canonical names.
    """
    if isinstance(df.columns, pd.MultiIndex):
        df = df.droplevel(-1, axis=1)
    lookup = {str(c).lower(): c for c in df.columns}
    rename = {lookup[c.lower()]: c for c in REQUIRED_COLUMNS if c.lower() in lookup}
    return df.rename(columns=rename)


def _validate(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    df = _canonical_columns(df)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise InvalidMarketData(f"{ticker}: missing expected columns {missing}")

    out = df.loc[:, list(REQUIRED_COLUMNS)].copy()

    # No ffill: a bar with incomplete OHLCV is an error, never repaired.
    if out.isna().any().any():
        counts = out.isna().sum()
        detail = ", ".join(f"{col}={int(n)}" for col, n in counts.items() if n)
        raise InvalidMarketData(
            f"{ticker}: found bars with incomplete OHLCV ({detail}). "
            "Refusing to fabricate prices; days without data must be absent."
        )

    out = out.astype("float64")

    if not out.index.is_monotonic_increasing:
        raise InvalidMarketData(f"{ticker}: index is not sorted ascending")
    if not out.index.is_unique:
        duplicates = out.index[out.index.duplicated()].unique()
        raise InvalidMarketData(
            f"{ticker}: duplicated timestamps in index, e.g. {list(duplicates[:3])}"
        )

    prices = out[["Open", "High", "Low", "Close"]]
    if (prices <= 0).any().any():
        counts = (prices <= 0).sum()
        detail = ", ".join(f"{col}={int(n)}" for col, n in counts.items() if n)
        raise InvalidMarketData(f"{ticker}: non-positive prices found ({detail})")

    if (out["Volume"] < 0).any():
        raise InvalidMarketData(f"{ticker}: negative volume found")

    if (out["Low"] > out["High"] + _EPS).any():
        raise InvalidMarketData(f"{ticker}: found bars where Low > High")

    if (out["Open"] > out["High"] + _EPS).any() or (out["Open"] < out["Low"] - _EPS).any():
        raise InvalidMarketData(f"{ticker}: Open falls outside the High/Low range")

    if (out["Close"] > out["High"] + _EPS).any() or (out["Close"] < out["Low"] - _EPS).any():
        raise InvalidMarketData(f"{ticker}: Close falls outside the High/Low range")

    return out


def load_prices(
    ticker: str,
    start: dt.date,
    end: dt.date,
    cache_dir: Path | None = None,
    offline: bool = False,
) -> pd.DataFrame:
    """Return validated daily OHLCV bars for `ticker` between `start` and `end`.

    Adjusted prices (splits and dividends folded in) are used, so the equity
    curve has no artificial gaps. Bars are validated *before* anything is
    written to disk, so a bad download can never poison the cache.
    """
    cache_dir = cache_dir if cache_dir is not None else Path("data/raw")
    path = _cache_path(cache_dir, ticker, start, end)

    raw = _read_cache(path, ticker, start, end, offline)
    if raw is None:
        try:
            raw = _download(ticker, start, end)
        except ArgosError:
            raise
        except Exception as exc:
            # Anything unexpected from the network layer is still a download
            # failure. The CLI must report one clear line, not a traceback.
            raise DownloadError(
                f"failed to download {ticker}: {type(exc).__name__}: {exc}"
            ) from exc
        if raw is None or raw.empty:
            raise DownloadError(
                f"no rows returned for {ticker} between {start} and {end}. "
                "Check the ticker symbol and the requested period."
            )
        raw = _window(raw, start, end)
        # Validate before persisting: a corrupt dataset must not reach the disk.
        validated = _validate(raw, ticker)
        _write_cache(path, validated)
        return validated

    return _validate(_window(raw, start, end), ticker)


def _read_cache(path: Path, ticker: str, start: dt.date, end: dt.date, offline: bool):
    if not path.exists():
        if offline:
            raise OfflineUnavailable(
                f"offline mode requested but no cached file at {path}. "
                "Run once without --offline to populate the cache."
            )
        return None
    try:
        return pd.read_csv(path, index_col=0, parse_dates=True)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, OSError, ValueError) as exc:
        raise CorruptCache(
            f"cached file {path} could not be parsed ({exc}). "
            "Delete it and re-run without --offline."
        ) from exc


def _write_cache(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)


def _window(raw: pd.DataFrame, start: dt.date, end: dt.date) -> pd.DataFrame:
    idx = pd.DatetimeIndex(raw.index)
    idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    raw = raw.set_axis(idx)

    window_start = pd.Timestamp(start, tz="UTC")
    window_end = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
    windowed = raw.loc[(raw.index >= window_start) & (raw.index < window_end)]

    if windowed.empty:
        raise InvalidMarketData("no bars returned for the requested period")
    return windowed


def _download(ticker: str, start: dt.date, end: dt.date) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise DownloadError("yfinance is not installed") from exc

    try:
        raw = yf.download(
            ticker,
            start=pd.Timestamp(start),
            end=pd.Timestamp(end) + pd.Timedelta(days=1),
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False,
        )
    except Exception as exc:
        raise DownloadError(
            f"failed to download {ticker} from yfinance: {type(exc).__name__}: {exc}"
        ) from exc

    if raw is None or raw.empty:
        raise DownloadError(
            f"yfinance returned no rows for {ticker} between {start} and {end}. "
            "Check the ticker symbol and the requested period."
        )

    return raw