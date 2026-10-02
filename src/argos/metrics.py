"""Performance metrics computed from an equity curve.

Conventions
-----------
* Daily **simple** returns: ``r_t = equity_t / equity_(t-1) - 1``.
* Drawdown is stored **positive** internally (0.2352 means a 23.52% decline).
  Presentation negates it. One convention, enforced by tests.
* A metric that is mathematically undefined (zero volatility, zero drawdown,
  a window shorter than two bars, elapsed time under one day) returns ``nan``.
  A metric is never ``inf``: an infinite ratio would be a presentation bug, and
  ``nan`` makes the cause visible instead of hiding it behind a huge number.
* ``equity <= 0`` is always an error, never a metric value.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252
DAYS_PER_YEAR = 365.25


@dataclass(frozen=True)
class Metrics:
    """Performance summary of one strategy over one evaluation window."""

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
    evaluations: int
    executions: int
    round_trips: int
    total_commission: float

    @property
    def positive_return(self) -> bool:
        return self.total_return > 0


# --- primitives -------------------------------------------------------------


def _positive(values) -> np.ndarray:
    arr = np.asarray(values, dtype="float64")
    if (arr <= 0).any():
        raise ValueError("equity curve must be strictly positive to compute drawdown-style metrics")
    return arr


# Dispersion below this is treated as zero. A constant series does not always
# produce a std of exactly 0.0 (floating point leaves ~1e-18 of noise), and
# dividing by that noise would report a Sharpe of 1e16 instead of admitting the
# ratio is undefined.
_MIN_DISPERSION = 1e-12


def _safe_div(numerator: float, denominator: float) -> float:
    if not math.isfinite(denominator) or abs(denominator) < _MIN_DISPERSION:
        return math.nan
    result = numerator / denominator
    return result if math.isfinite(result) else math.nan


def daily_returns(equity_curve: pd.Series) -> pd.Series:
    """Daily simple returns of an equity curve. The first bar has no return."""
    if equity_curve.empty:
        return pd.Series(dtype="float64")
    _positive(equity_curve.to_numpy())
    return equity_curve.pct_change().iloc[1:]


def total_return(initial: float, final: float) -> float:
    if initial <= 0:
        raise ValueError("initial capital must be positive")
    return final / initial - 1.0


def cagr(initial: float, final: float, calendar_days: float) -> float:
    """Compound annual growth rate over *real* elapsed calendar days.

    Uses the actual duration between the first and last evaluated bar, not
    `252 * n_observations`, so a window that starts mid-week is annualised
    correctly.
    """
    if initial <= 0 or final <= 0:
        raise ValueError("capital must be positive")
    if calendar_days < 1:
        return math.nan
    years = calendar_days / DAYS_PER_YEAR
    return (final / initial) ** (1.0 / years) - 1.0


def annualized_volatility(returns: pd.Series) -> float:
    """``std(daily simple returns, ddof=1) * sqrt(252)``."""
    values = returns.dropna().to_numpy(dtype="float64")
    if values.size < 2:
        return math.nan
    return float(values.std(ddof=1) * math.sqrt(TRADING_DAYS_PER_YEAR))


def _daily_risk_free(annual_rf: float) -> float:
    if annual_rf <= -1:
        raise ValueError("annual risk-free rate must be greater than -1")
    return (1.0 + annual_rf) ** (1.0 / TRADING_DAYS_PER_YEAR) - 1.0


def sharpe_ratio(returns: pd.Series, annual_rf: float = 0.0) -> float:
    """``(mean(r) - daily_rf) / std(r, ddof=1) * sqrt(252)``."""
    values = returns.dropna().to_numpy(dtype="float64")
    if values.size < 2:
        return math.nan
    daily_rf = _daily_risk_free(annual_rf)
    return _safe_div(
        (values.mean() - daily_rf) * math.sqrt(TRADING_DAYS_PER_YEAR), values.std(ddof=1)
    )


def downside_deviation(returns: pd.Series, annual_rf: float = 0.0) -> float:
    """``sqrt(mean(min(r - daily_rf, 0)^2))`` over *every* observation.

    Including the bars above the minimum acceptable return as zeros is
    deliberate: dropping them shrinks the denominator and inflates the ratio
    relative to Sharpe.
    """
    values = returns.dropna().to_numpy(dtype="float64")
    if values.size == 0:
        return math.nan
    daily_rf = _daily_risk_free(annual_rf)
    shortfall = np.minimum(values - daily_rf, 0.0)
    return float(np.sqrt((shortfall**2).mean()))


def sortino_ratio(returns: pd.Series, annual_rf: float = 0.0) -> float:
    """``(mean(r) - daily_rf) / downside_deviation * sqrt(252)``."""
    values = returns.dropna().to_numpy(dtype="float64")
    if values.size == 0:
        return math.nan
    daily_rf = _daily_risk_free(annual_rf)
    downside = downside_deviation(returns, annual_rf)
    return _safe_div((values.mean() - daily_rf) * math.sqrt(TRADING_DAYS_PER_YEAR), downside)


def max_drawdown(equity_curve: pd.Series) -> float:
    """Largest peak-to-trough decline, returned as a **positive** fraction.

    Measured on the mark-to-market equity curve. 0.0 for a monotonically
    non-decreasing curve.
    """
    if equity_curve.empty:
        raise ValueError("equity curve must not be empty")
    values = _positive(equity_curve.to_numpy(dtype="float64"))
    running_peak = np.maximum.accumulate(values)
    return float((1.0 - values / running_peak).max())


def calmar_ratio(cagr_value: float, max_dd: float) -> float:
    """``CAGR / abs(max_drawdown)``. ``nan`` when there is no drawdown."""
    if not math.isfinite(cagr_value):
        return math.nan
    return _safe_div(cagr_value, abs(max_dd))


def exposure(units: pd.Series) -> float:
    """Fraction of evaluated bars carrying a long position.

    Measured on bar closes, over the evaluated window only.
    """
    if units.empty:
        return math.nan
    return float((units > 0).sum() / len(units))


# --- aggregation ------------------------------------------------------------


def compute_metrics(
    equity_curve: pd.Series,
    units: pd.Series,
    initial_capital: float,
    executions: int = 0,
    round_trips: int = 0,
    total_commission: float = 0.0,
    annual_rf: float = 0.0,
) -> Metrics:
    """Compute the full metric set for one strategy over one window.

    `initial_capital` is the capital the window is measured against. It is the
    baseline for `total_return` and `cagr`, and it is deliberately not taken from
    the first recorded point of the curve: a predefined-timing strategy enters
    at the first bar's open, so its first recorded equity -- that bar's close --
    already contains one bar's move. Basing the return on it discarded the first
    bar and made `total_return` inconsistent with the `initial_capital` and
    `final_equity` printed beside it. Ratio metrics are unaffected either way,
    since they are invariant to the choice of baseline.
    """
    if equity_curve.empty:
        raise ValueError("equity curve must not be empty")
    if len(units) != len(equity_curve):
        raise ValueError("units and equity curve must have the same length")

    _positive(equity_curve.to_numpy())

    initial = float(initial_capital)
    final = float(equity_curve.iloc[-1])
    returns = daily_returns(equity_curve)
    calendar_days = float((equity_curve.index[-1] - equity_curve.index[0]).days)
    dd = max_drawdown(equity_curve)
    growth = cagr(initial, final, calendar_days)

    return Metrics(
        initial_capital=float(initial_capital),
        final_equity=final,
        total_return=total_return(initial, final),
        cagr=growth,
        annualized_volatility=annualized_volatility(returns),
        sharpe_ratio=sharpe_ratio(returns, annual_rf),
        sortino_ratio=sortino_ratio(returns, annual_rf),
        max_drawdown=dd,
        calmar_ratio=calmar_ratio(growth, dd),
        exposure=exposure(units),
        evaluations=len(equity_curve),
        executions=int(executions),
        round_trips=int(round_trips),
        total_commission=float(total_commission),
    )