# ARGOS · trading lab

Educational and experimental software-engineering project to study trading
strategies rigorously: what a strategy actually did, what it risked, and what it
would have needed to survive realistic costs.

**Virtual money only. No broker integration, no order routing, no live trading,
and no AI of any kind in this version.**

The purpose is not to find a profitable strategy. It is to build a harness whose
numbers can be trusted, so that a future version can ask harder questions of a
wider set of strategies without first having to prove the measurement itself is
sound.

## Scope

1. Download and validate historical daily market data.
2. Implement strategies that emit target positions from a fixed parameter set.
3. Simulate them bar by bar with an explicit cost model.
4. Measure return, risk, drawdown and time in the market with documented formulas.
5. Compare strategies on a common evaluation window, with a benchmark.
6. Test the critical logic, including the cases that would flatter a result.

Deliberately out of scope: frontend, database, Docker, microservices, LLM,
news, sentiment, machine learning, reinforcement learning, parameter
optimisation, broker integration, paper trading.

## Install

Requires Python 3.12+.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Commands

```bash
# One strategy over the full requested period, its own semantics and warm-up.
python scripts/run_backtest.py backtest

# Every strategy on a common evaluation window, with the benchmark.
python scripts/run_backtest.py compare

# The cost-sensitivity grid: frictionless, 5 bp per side, 10 bp per side.
python scripts/run_backtest.py costs

# The multi-asset, multi-window experiment grid, with a reproducible CSV export.
python scripts/run_backtest.py matrix --scenarios A,B,C --export-csv data/exports/matrix.csv

# Reuse only the local cache; never touches the network.
python scripts/run_backtest.py compare --offline
```

Defaults: `SPY`, daily bars, `2020-01-01` → `2025-12-31`, 10,000 initial capital,
no costs, all three strategies.

Common to every subcommand: `--ticker`, `--start`, `--end`, `--initial-capital`,
`--commission-pct`, `--slippage-pct`, `--annual-rf`, `--strategies`, `--cache-dir`,
`--offline`, `--debug`. `matrix` adds `--tickers`, `--windows`, `--scenarios`,
`--export-csv` and `--summary-only`. `python scripts/run_backtest.py <command>
--help` lists them all.

`--debug` turns a one-line error into a full traceback. Without it, every
expected failure prints exactly one line and exits with code 1.

## Architecture

| Module | Responsibility |
| --- | --- |
| `errors.py` | One root, specific leaves, so a failure names its own kind. |
| `datafeed.py` | Only module that touches the network. Downloads, validates, caches. |
| `indicators.py` | Pure causal functions, `Series -> Series`. No IO, no state. |
| `strategy.py` | Bars to target positions (1 long, 0 flat). Declares its timing. Decides *what*, never *how much*. |
| `portfolio.py` | Sole owner of cash and position. The only mutable state. |
| `engine.py` | Bar-by-bar loop. Applies timing, executes at an open, marks to market. |
| `metrics.py` | Pure functions over the equity curve. |
| `comparison.py` | Common window, rebasing, cost scenarios, comparison result types. |
| `experiment.py` | The multi-asset multi-window matrix, its CSV export and its descriptive summary. |
| `report.py` | Formatting only. No trading logic, no metric computation. |
| `scripts/run_backtest.py` | CLI wiring. |

The dependency direction is one-way: `datafeed → strategy → engine → metrics →
report`. `portfolio` is injected into `engine` as state, and nothing reaches
backwards.

## Temporal semantics

A strategy declares how its decision relates to the bar it is computed on.

**`SignalTiming.DERIVED`** — SMA 20/50 and Momentum 126. The decision consumes
data observed up to and including bar *t*, so it cannot be executed before the
open of bar *t+1*. The engine applies `.shift(1)` and executes at the next open.
Fills therefore pay any overnight gap, and the first actionable bar is the first
valid signal plus one.

**`SignalTiming.PREDEFINED`** — Buy & Hold. The decision exists *before* the data
starts, so there is no bar to observe and no warm-up; it executes at the open of
bar 0. Skipping that first open would hand the benchmark a free gain or loss that
no investor would have experienced.

`PREDEFINED` is not an unchecked promise. The engine verifies that the signal is a
single constant value with no NaN, which makes the class structurally incapable
of leaking future information: a constant cannot depend on data that does not yet
exist. A strategy that misdeclares its timing is rejected, not silently delayed.

The other half of the barrier:

- **Trailing windows only.** Rolling windows look backwards. No centred windows,
  no `bfill`, no forward interpolation of prices.
- **Missing data is an error.** A bar with incomplete OHLCV raises
  `InvalidMarketData` instead of being repaired. Non-trading days simply do not
  appear in the frame.
- **Prefix invariance is tested.** Truncating the future must not change any
  already-computed signal, signal or equity value.

## Comparison mode

An individual backtest respects each strategy's own semantics and warm-up. That is
the right answer to "how does this strategy behave". It is the wrong answer to
"how do these strategies compare", because they would be measured over different
stretches of market: momentum needs 126 bars of history before it can act, so it
would sit in cash while buy-and-hold was already invested.

So `compare` computes the first bar at which *every* strategy could act,
`common_start = max(first_actionable_bar)`, trims every equity curve to
`[common_start, last bar]`, and rescales each curve multiplicatively to 10,000.
Both the requested period and the common evaluation period are printed, and the
requested period is a column in the matrix export.

Rebasing is a pure scale change. Returns, CAGR, volatility, Sharpe, Sortino,
drawdown, Calmar and exposure are all invariant under it, and the tests assert
exactly that rather than assuming it.

Two consequences are reported rather than hidden:

- **Activity counters cover the full requested period.** Return and risk metrics
  describe the window; round trips and executions describe everything the engine
  did, because attributing a fill to the window would require re-warming the
  indicators on the trimmed data, which would change the signals themselves.
- **A strategy that made no fill inside the window cannot show a cost effect
  there.** Buy & Hold is exactly this case: its single entry precedes the window,
  so its commission is reported but cannot reduce the window's return. The report
  says so on the table. A cost grid that hid this would look like evidence that
  costs do not matter.

If a strategy's warm-up exceeds the whole dataset, comparison fails with a clear
error. Dropping it would quietly bias the table.

## Metrics

Computed from daily **simple** returns, `r_t = equity_t / equity_(t-1) - 1`.

| Metric | Definition |
| --- | --- |
| total return | `final / initial - 1` |
| CAGR | `(final / initial)^(365.25 / calendar_days) - 1`, over real elapsed days |
| annualized volatility | `std(r, ddof=1) * sqrt(252)` |
| Sharpe | `(mean(r) - daily_rf) / std(r, ddof=1) * sqrt(252)` |
| Sortino | `(mean(r) - daily_rf) / downside_deviation * sqrt(252)` |
| downside deviation | `sqrt(mean(min(r - daily_rf, 0)^2))` over every observation |
| max drawdown | largest peak-to-trough decline of the equity curve, **stored positive** |
| Calmar | `CAGR / abs(max drawdown)` |
| exposure | `bars holding a position / evaluated bars` |
| commission | percentage of notional, charged on both sides |
| slippage | percentage against the trader: buys fill higher, sells fill lower |

`daily_rf = (1 + annual_rf)^(1/252) - 1`, with `annual_rf` configurable and 0.0 by
default. Drawdown is stored positive internally and shown negative in every
report; one convention, enforced by tests.

**Guards.** A metric that is mathematically undefined returns `NaN`, never `inf`:
zero volatility, zero downside deviation, no drawdown, a window shorter than two
bars, or less than one day of elapsed time. An infinite ratio would be a
presentation bug, and `NaN` makes the cause visible instead of hiding it behind a
huge number. An equity curve at or below zero is an error, not a metric.
Dispersion below `1e-12` counts as zero, because a constant series does not always
produce a standard deviation of exactly `0.0` and dividing by the leftover
floating-point noise would report a Sharpe of 1e16.

## Trade and cost definitions

- **execution** — one fill. Every buy and every sell counts, so a completed round
  trip is 2 executions.
- **round trip** — a position fully closed. Partial exits are not part of the
  model, so `round_trips` only increments on a full exit; if partial exits are ever
  supported the counter must be redesigned to pair entries with closing sizes.
- **Sizing** — 100% of available cash. The commission is funded out of the same
  cash, so the balance reaches exactly zero on entry and no money is created.
- **Open positions** — the last position is never force-closed. Final equity is
  marked to market at the last close.

## Experiment matrix

Fixed before any run, and never tuned after seeing a result:

- **Assets:** SPY, QQQ, AAPL, MSFT.
- **Windows:** 2010–2014, 2015–2019, 2020–2025.
- **Cost scenarios:** A frictionless, B 5 bp per side, C 10 bp per side.
- **Capital:** 10,000 per experiment, fresh in every cell.

Every cell is an independent experiment. A window where the provider has no data
is reported as skipped, never silently replaced, and never dropped because it
looked bad. The CSV is written with `index=False` and a fixed column order.

The summary is descriptive only: median CAGR, median Sharpe, median drawdown, how
many experiments were positive, how many beat the benchmark, and the worst and
best cells. **No composite score and no ranking are computed on purpose.**
Collapsing distinct metrics into one number would imply a comparison the data
cannot support, and a small fixed sample of four assets and three windows is not
enough to establish that any strategy works.

## Test

```bash
pytest
ruff check .
```

The suite covers more than the happy path. It pins exact metric values computed by
hand, checks the drawdown against a slow reference implementation, and hammers
invariants across generated price paths of different shapes — trends, crashes,
gaps, flat series, spiky series and near-constant series — asserting that equity is
never NaN, positions are never short, the accounting identity
`equity == cash + units * close` holds on every single bar, costs monotonically
reduce the outcome, and no metric is ever infinite.

## Known limitations

- **yfinance is not a professional data source.** It is scraper-backed, subject
  to outages and occasional adjustment of already-published history. Suitable for
  a prototype, not for research conclusions. It sits behind `datafeed.py` so it
  can be replaced without touching the engine.
- **Prices are adjusted.** Splits and dividends are folded in, so the equity
  curve has no artificial gaps. The trade-off: the price you see is not literally
  the price that traded.
- **Survivorship and data revision.** The current universe of symbols is used, and
  providers revise history. Delisted assets are absent entirely. Results from a
  single asset over a hand-picked period are easy to overfit.
- **Drawdown is measured on the equity curve at close**, not on intrabar lows. A
  deep wick that recovers within the same bar is invisible to it.
- **Momentum is a pure directional filter.** The magnitude of the move is ignored
  and there is no volatility normalisation, so it is a different bet from
  time-series momentum, not a weaker version of it.
- **No taxes.** Dividends and capital gains taxation are not modelled, so the
  strategy is flattered relative to a taxable investor.
- **Simplified commission and slippage.** Fixed percentages per side, with no
  minimum fees, no market impact, no partial fills, no borrow costs, no spread
  model. The default is 0.0, so the default report describes frictionless trading.
- **Daily bars only.** Intra-bar highs and lows are not used for execution, so
  stops, limits and intraday sequencing cannot be modelled.
- **Fixed 100% position sizing**, one position at a time, no minimum hold.
- **No parameter search.** SMA 20/50 and Momentum 126 are fixed by design.
  Optimising them would turn this into a curve-fitting exercise.
- **A strategy that never warms up cannot be compared.** The error is explicit;
  there is no partial-window fallback.
- **Educational and experimental purpose.** Not financial advice and not suitable
  for trading real money.
