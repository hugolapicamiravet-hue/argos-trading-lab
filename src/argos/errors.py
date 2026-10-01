"""Exception types for ARGOS.

One root, a small number of specific leaves. The goal is that a user can tell
*which kind* of failure happened without reading a stack trace:

    DownloadError      the provider could not be reached or returned nothing
    InvalidMarketData  the data is malformed (bad OHLCV, duplicates, NaN)
    CorruptCache       the cached file exists but cannot be parsed
    OfflineUnavailable the cache is required but absent
    InvalidSignals     a strategy produced signals the engine cannot trust
    ConfigurationError the user asked for something incoherent
    BacktestError      the simulation itself could not complete

Every one of these is expected during normal use and is reported as a single
clear CLI line with a non-zero exit code.
"""

from __future__ import annotations


class ArgosError(Exception):
    """Base class for every expected ARGOS failure."""


class DownloadError(ArgosError):
    """Market data could not be downloaded from the provider."""


class InvalidMarketData(ArgosError):
    """Market data failed validation."""


class CorruptCache(ArgosError):
    """A cached dataset exists but could not be read."""


class OfflineUnavailable(ArgosError):
    """Offline mode was requested but the cache holds no data for this run."""


class InvalidSignals(ArgosError):
    """A strategy returned signals the engine cannot use."""


class ConfigurationError(ArgosError):
    """The requested configuration is internally inconsistent."""


class BacktestError(ArgosError):
    """The backtest could not be completed."""


# Backwards-compatible aliases kept from the first MVP.
DataValidationError = InvalidMarketData
OfflineDataError = OfflineUnavailable

__all__ = [
    "ArgosError",
    "BacktestError",
    "ConfigurationError",
    "CorruptCache",
    "DataValidationError",
    "DownloadError",
    "InvalidMarketData",
    "InvalidSignals",
    "OfflineDataError",
    "OfflineUnavailable",
]