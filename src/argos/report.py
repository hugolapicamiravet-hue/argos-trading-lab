"""Formatting. No trading logic and no metric computation live here."""

from __future__ import annotations

import math
import textwrap

from argos.comparison import Comparison
from argos.experiment import MatrixResult, StrategySummary
from argos.metrics import Metrics


def money(value: float) -> str:
    return "n/a" if not math.isfinite(value) else f"{value:,.2f}"


def pct(value: float, signed: bool = True) -> str:
    if not math.isfinite(value):
        return "n/a"
    return f"{value * 100:+.2f}%" if signed else f"{value * 100:.2f}%"


def num(value: float, digits: int = 3) -> str:
    return "n/a" if not math.isfinite(value) else f"{value:.{digits}f}"


def _wrap(text: str, width: int) -> list[str]:
    """Wrap a footnote to the table width so the block stays readable."""
    return textwrap.wrap(text, width=max(width, 40))


def format_individual(
    label: str,
    metrics: Metrics,
    timing: str,
    first_actionable: str,
    bars: int,
    capital: float,
    commission_pct: float,
    slippage_pct: float,
) -> str:
    rows = [
        ("strategy", label),
        ("signal timing", timing),
        ("first actionable bar", first_actionable),
        ("bars evaluated", str(bars)),
        ("initial capital", money(capital)),
        ("final equity", money(metrics.final_equity)),
        ("total return", pct(metrics.total_return)),
        ("CAGR", pct(metrics.cagr)),
        ("annualized volatility", pct(metrics.annualized_volatility, signed=False)),
        ("Sharpe ratio", num(metrics.sharpe_ratio)),
        ("Sortino ratio", num(metrics.sortino_ratio)),
        ("maximum drawdown", pct(-metrics.max_drawdown)),
        ("Calmar ratio", num(metrics.calmar_ratio)),
        ("exposure", pct(metrics.exposure, signed=False)),
        ("round trips", str(metrics.round_trips)),
        ("executions", str(metrics.executions)),
        ("total commission", money(metrics.total_commission)),
        ("commission_pct", pct(commission_pct, signed=False)),
        ("slippage_pct", pct(slippage_pct, signed=False)),
    ]
    # Two spaces of separation, then the rule is as wide as the widest row.
    key_w = max(len(k) for k, _ in rows) + 2
    value_w = max(len(v) for _, v in rows)
    rule = "=" * (key_w + value_w)

    lines = [rule, "ARGOS · individual backtest", rule]
    lines += [f"{k.ljust(key_w)}{v}" for k, v in rows]
    lines += ["-" * (key_w + value_w), "definitions"]
    lines += [
        "  round trips   fully closed positions; partial exits are not modelled",
        "  executions    fills; each buy and each sell counts separately",
        "  drawdown      stored positive, shown negative here",
        "  n/a           metric undefined for this run (see README guards)",
    ]
    return "\n".join(lines)


_METRIC_LABELS = [
    ("final equity", lambda m: money(m.final_equity)),
    ("total return", lambda m: pct(m.total_return)),
    ("CAGR", lambda m: pct(m.cagr)),
    ("annualized volatility", lambda m: pct(m.annualized_volatility, signed=False)),
    ("Sharpe ratio", lambda m: num(m.sharpe_ratio)),
    ("Sortino ratio", lambda m: num(m.sortino_ratio)),
    ("max drawdown", lambda m: pct(-m.max_drawdown)),
    ("Calmar ratio", lambda m: num(m.calmar_ratio)),
    ("exposure", lambda m: pct(m.exposure, signed=False)),
    ("round trips", lambda m: str(m.round_trips)),
    ("executions", lambda m: str(m.executions)),
    ("total commission", lambda m: money(m.total_commission)),
]


def format_comparison(comparison: Comparison) -> str:
    labels = [run.label for run in comparison.runs]
    row_names = [name for name, _ in _METRIC_LABELS] + [
        "metric",
        "requested period",
        "common evaluation period",
        "initial capital",
        "commission per side",
        "slippage per side",
        "excess return",
    ]
    # The label column must fit the longest metric name, not just the strategy
    # headers, otherwise the rules end up narrower than the rows they frame.
    label_w = max(len(x) for x in row_names + labels) + 2
    value_w = 13
    width = label_w + value_w * len(labels)

    head = "metric".ljust(label_w) + "".join(x.rjust(value_w) for x in labels)
    out = [
        "=" * width,
        f"ARGOS · comparison · {comparison.ticker}",
        "=" * width,
        f"{'requested period'.ljust(label_w)}{comparison.requested_period}",
        f"{'common evaluation period'.ljust(label_w)}{comparison.common_period}",
        f"{'initial capital'.ljust(label_w)}{money(comparison.initial_capital)}",
        "-" * width,
        head,
        "-" * width,
    ]

    for name, getter in _METRIC_LABELS:
        line = name.ljust(label_w)
        for run in comparison.runs:
            line += getter(run.metrics).rjust(value_w)
        out.append(line)

    first = comparison.runs[0].result
    out.append("-" * width)
    out.append(
        "commission per side".ljust(label_w)
        + pct(first.commission_pct, signed=False).rjust(value_w)
    )
    out.append(
        "slippage per side".ljust(label_w) + pct(first.slippage_pct, signed=False).rjust(value_w)
    )

    bench = comparison.benchmark
    if bench is not None:
        out.append("-" * width)
        line = "excess return".ljust(label_w)
        for run in comparison.runs:
            line += pct(comparison.excess_return(run.key)).rjust(value_w)
        out.append(line)
        out += _wrap(
            f"(excess return is each strategy minus {bench.label} on the same window)", width
        )

    out.append("-" * width)
    out += _wrap("Return and risk metrics cover the common evaluation period only.", width)
    out += _wrap(
        "Round trips and executions are counted over the full requested period, because a "
        "fill cannot be attributed to the window without re-warming the indicators and "
        "changing the signals themselves. Treat them as activity counts, not as window "
        "statistics.",
        width,
    )

    carried = [
        run.label
        for run in comparison.runs
        if run.carries_position_from_before and not run.trades_inside_window
    ]
    if carried:
        out += _wrap(
            f"Note: {', '.join(carried)} made no trade inside the window, so costs charged "
            "before it cannot reduce the window return. The commission is still reported in "
            "the total commission row above.",
            width,
        )

    return "\n".join(out)


def format_matrix_summary(matrix: MatrixResult, summaries: list[StrategySummary]) -> str:
    width = 78
    out = ["=" * width, "ARGOS · experiment matrix summary (descriptive only)", "=" * width]
    out.append(f"experiments: {len(matrix.rows)}   skipped: {len(matrix.skipped)}")
    if matrix.skipped:
        out.append("")
        out.append("skipped windows (data unavailable; not replaced):")
        out += [f"  - {note}" for note in matrix.skipped]
    out.append("")

    for s in summaries:
        out.append(f"Strategy: {s.label}")
        out.append(f"  experiments                  : {s.experiments}")
        out.append(f"  median CAGR                  : {pct(s.median_cagr)}")
        out.append(f"  median Sharpe                : {num(s.median_sharpe)}")
        out.append(f"  median max drawdown          : {pct(-s.median_max_drawdown)}")
        out.append(
            f"  positive-return experiments  : "
            f"{s.positive_return_experiments}/{s.experiments}"
        )
        out.append(f"  outperforming Buy & Hold     : {s.outperforming_benchmark}/{s.experiments}")
        out.append(f"  worst experiment             : {s.worst}")
        out.append(f"  best experiment              : {s.best}")
        out.append("")

    out.append(
        "These are descriptive statistics over a small, fixed sample of assets and windows."
    )
    out.append("They are not evidence that any strategy works. No composite score is computed")
    out.append(
        "on purpose: combining distinct metrics would imply a ranking the data cannot support."
    )
    return "\n".join(out)