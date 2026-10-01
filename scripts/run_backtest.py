#!/usr/bin/env python3
"""ARGOS command line interface.

Subcommands:
    backtest   one strategy over the full requested period (its own semantics)
    compare    all strategies on a common evaluation window
    costs      the cost-sensitivity grid
    matrix     the multi-asset, multi-window experiment grid
    news       fetch, normalize, analyze, inspect and audit news data

Expected failures print one clear line and exit non-zero. `--debug` adds the
traceback.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
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
from argos.news import (  # noqa: E402
    Manifest,
    NewsManifestBuilder,
    NewsStore,
    analyze_many,
    build_analyzer,
    dedupe_articles,
    normalize_articles,
    provider_for,
    run_audit,
    write_manifest,
)
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

    _add_news_parser(sub)

    return parser


def _add_news_parser(sub) -> None:
    """The `news` group: the raw-to-events path, and the audit that checks it.

    Each stage reads what the previous one wrote, so every command takes the
    same `--dataset` name and none of them re-derives an upstream result. That
    keeps a run reproducible from the store alone.
    """
    p_news = sub.add_parser("news", help="build and audit the news and macro dataset")
    news = p_news.add_subparsers(dest="news_command", required=True)

    p_fetch = news.add_parser("fetch", help="fetch raw provider payloads")
    p_fetch.add_argument("--dataset", required=True)
    p_fetch.add_argument("--provider", default="massive")
    p_fetch.add_argument("--start", type=_parse_date, default=dt.date(2020, 1, 1))
    p_fetch.add_argument("--end", type=_parse_date, default=dt.date.today())
    p_fetch.add_argument("--tickers", default="", help="comma-separated symbols")
    p_fetch.add_argument("--max-pages", type=int, default=10)
    p_fetch.add_argument("--news-root", default="data/news")
    p_fetch.add_argument("--debug", action="store_true")

    p_norm = news.add_parser("normalize", help="turn raw payloads into validated articles")
    p_norm.add_argument("--dataset", required=True)
    p_norm.add_argument(
        "--provider", default="massive",
        help="source name recorded on each article; part of its identity",
    )
    p_norm.add_argument("--news-root", default="data/news")
    p_norm.add_argument("--debug", action="store_true")

    p_analyze = news.add_parser("analyze", help="derive financial events from articles")
    p_analyze.add_argument("--dataset", required=True)
    p_analyze.add_argument("--provider", default="massive")
    p_analyze.add_argument("--news-root", default="data/news")
    p_analyze.add_argument("--analyzer", default="deterministic")
    p_analyze.add_argument("--model", default=None, help="model id, required for --analyzer openai")
    p_analyze.add_argument("--cache", default=None, help="path for the analysis cache")
    p_analyze.add_argument("--debug", action="store_true")

    p_inspect = news.add_parser("inspect", help="summarize what a dataset contains")
    p_inspect.add_argument("--dataset", default=None)
    p_inspect.add_argument("--news-root", default="data/news")
    p_inspect.add_argument("--debug", action="store_true")

    p_audit = news.add_parser("audit", help="check a dataset against the point-in-time rules")
    p_audit.add_argument("--dataset", required=True)
    p_audit.add_argument("--news-root", default="data/news")
    p_audit.add_argument(
        "--expected-start", type=_parse_date, default=None,
        help="fail if coverage begins later than this date; omitting it leaves "
        "coverage gaps advisory",
    )
    p_audit.add_argument(
        "--warnings-as-errors", action="store_true",
        help="exit non-zero on warnings as well as errors",
    )
    p_audit.add_argument("--json", action="store_true", help="emit the report as JSON")
    p_audit.add_argument("--debug", action="store_true")


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


def _now() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


def _manifest_for(store: NewsStore, dataset: str, provider: str) -> NewsManifestBuilder:
    """Start a manifest describing every file the dataset currently has.

    Each stage re-derives this rather than editing the previous manifest, so the
    file on disk always describes the whole dataset. Otherwise a later stage
    overwrites an earlier one's digests, and the audit ends up verifying only
    the stage that ran last -- which is precisely the file an amended row is
    least likely to be hiding in.
    """
    # root makes the recorded paths relative to the store, so a dataset stays
    # auditable after being moved or copied. Without it the paths are absolute
    # (or root-prefixed, when the root is relative) and the audit resolves them
    # against the store root a second time.
    builder = NewsManifestBuilder(name=dataset, provider=provider, root=store.root)
    for path, add in (
        (store.raw_dir / f"{dataset}.json", builder.add_raw),
        (store.normalized_dir / f"{dataset}.jsonl", builder.add_normalized),
        (store.events_dir / f"{dataset}.jsonl", builder.add_events),
    ):
        if path.exists():
            add(path)
    return builder


def _carry_forward(builder: NewsManifestBuilder, previous: Manifest) -> NewsManifestBuilder:
    """Preserve counts a later stage did not itself recompute.

    Rejection and duplicate counts are observed during normalization only. A
    manifest written by `analyze` must still report them, or the record of what
    the data cost disappears once the run is finished.
    """
    builder.record_duplicates(previous.duplicates_dropped)
    for reason, count in sorted(previous.rejection_reasons.items()):
        builder.record_rejections([(index, reason) for index in range(count)])
    return builder


def _previous_manifest(store: NewsStore, dataset: str) -> Manifest | None:
    path = store.root / f"{dataset}.manifest.json"
    if not path.exists():
        return None
    return Manifest.from_dict(json.loads(path.read_text(encoding="utf-8")))


def cmd_news_fetch(args) -> int:
    """Fetch raw payloads. Secrets come from the environment, never argv.

    A key passed as an argument would land in shell history and in any process
    listing, so the CLI only accepts the name of the variable holding it.
    """
    if args.end < args.start:
        raise ConfigurationError(f"--end ({args.end}) must not precede --start ({args.start})")
    if args.max_pages < 1:
        raise ConfigurationError("--max-pages must be at least 1")

    provider = provider_for(args.provider)
    params: dict[str, object] = {}
    if args.start < dt.date(2020, 1, 1):
        print(
            f"note: provider history starts in {args.start:%Y}; earlier dates return nothing",
            file=sys.stderr,
        )
    params["published_utc.gte"] = dt.datetime.combine(args.start, dt.time(), tzinfo=dt.UTC)
    params["published_utc.lte"] = dt.datetime.combine(args.end, dt.time(), tzinfo=dt.UTC)
    if args.tickers:
        params["tickers"] = args.tickers.split(",")

    store = NewsStore(args.news_root)
    total = 0
    for page in provider.iter_pages(max_pages=args.max_pages, **params):
        stats = store.write_raw(args.dataset, page.payload, overwrite=True)
        total += len(page.payload or ())
        print(f"fetched page -> {stats.path} ({stats.written} records)")
    if total == 0:
        print(
            f"no records returned for {args.start}..{args.end}; coverage will start later "
            "than requested and the audit will report it",
            file=sys.stderr,
        )
    return EXIT_OK


def cmd_news_normalize(args) -> int:
    """Validate, dedupe and persist articles, then write the manifest."""
    store = NewsStore(args.news_root)
    retrieved_at = _now()
    payloads = list(store.iter_raw_records(args.dataset))
    if not payloads:
        raise ConfigurationError(
            f"no raw payloads for dataset {args.dataset!r}; run `argos news fetch` first"
        )

    result = normalize_articles(payloads, source=args.provider, retrieved_at=retrieved_at)
    kept, _ = dedupe_articles(result.articles)
    stats = store.write_articles(args.dataset, kept)

    builder = _manifest_for(store, args.dataset, args.provider)
    builder.record_articles(kept)
    builder.record_rejections(result.rejected)
    builder.record_duplicates(len(result.duplicates))
    manifest_path = write_manifest(builder.build(), store.root / f"{args.dataset}.manifest.json")

    print(f"wrote {stats.written} articles -> {stats.path}")
    tally = _counts(result.rejected)
    print(f"  rejected: {sum(tally.values())}   duplicates dropped: {len(result.duplicates)}")
    for reason, count in sorted(tally.items()):
        print(f"    {reason}: {count}")
    print(f"manifest -> {manifest_path}")
    return EXIT_OK


def _counts(rejected) -> dict[str, int]:
    """Tally rejection reasons.

    `rejected` pairs each dropped record's index with its reason. The index
    matters for provenance but not for this summary, so only the reason is
    counted.
    """
    tally: dict[str, int] = {}
    for entry in rejected:
        reason = entry[1] if isinstance(entry, tuple) else str(entry)
        tally[reason] = tally.get(reason, 0) + 1
    return tally


def cmd_news_analyze(args) -> int:
    """Derive events with the chosen analyzer and append them to the store."""
    store = NewsStore(args.news_root)
    articles = store.read_articles(args.dataset)
    if not articles:
        raise ConfigurationError(
            f"no articles for dataset {args.dataset!r}; run `argos news normalize` first"
        )
    analyzer = build_analyzer(
        args.analyzer, model=args.model, cache_path=args.cache
    )
    analyzed_at = _now()
    events = analyze_many(articles, analyzer, analyzed_at=analyzed_at)
    stats = store.write_events(args.dataset, events, append=True)

    previous = _previous_manifest(store, args.dataset)
    builder = _manifest_for(store, args.dataset, args.provider)
    builder.record_articles(articles)
    builder.record_events(events)
    if previous is not None:
        _carry_forward(builder, previous)
    manifest_path = write_manifest(builder.build(), store.root / f"{args.dataset}.manifest.json")

    print(f"analyzer: {args.analyzer}")
    print(f"wrote {stats.written} events -> {stats.path}")
    directions: dict[str, int] = {}
    for event in events:
        directions[event.direction.value] = directions.get(event.direction.value, 0) + 1
    for name, count in sorted(directions.items()):
        print(f"  {name}: {count}")
    print(f"manifest -> {manifest_path}")
    return EXIT_OK


def cmd_news_inspect(args) -> int:
    store = NewsStore(args.news_root)
    description = store.describe()
    print(f"store root: {description['root']}")
    for section in ("raw", "normalized", "events"):
        rows = description[section]["datasets"]
        if args.dataset:
            rows = [r for r in rows if r["name"] == args.dataset]
        if not rows:
            print(f"{section}: (none)")
            continue
        print(f"{section}: {description[section]['dir']}")
        for row in rows:
            size = f"{row['bytes']:,} bytes"
            count = f"{row['rows']:,} rows" if "rows" in row else ""
            print(f"  {row['name']}: {count} {size}".strip())
    return EXIT_OK


def cmd_news_audit(args) -> int:
    """Run every check and report the verdict.

    The manifest is included when present so file identity is verified too; an
    audit that only inspected in-memory rows would not notice an edited file.
    """
    store = NewsStore(args.news_root)
    articles = store.read_articles(args.dataset)
    events = store.read_events(args.dataset)
    manifest_path = store.root / f"{args.dataset}.manifest.json"
    manifest = None
    if manifest_path.exists():
        manifest = Manifest.from_dict(json.loads(manifest_path.read_text(encoding="utf-8")))

    expected_start = (
        dt.datetime.combine(args.expected_start, dt.time(), tzinfo=dt.UTC)
        if args.expected_start
        else None
    )
    report = run_audit(
        articles=articles,
        events=events,
        manifest=manifest,
        expected_start=expected_start,
        manifest_root=str(store.root),
    )

    if args.json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        print(f"dataset {args.dataset!r}: {len(articles)} articles, {len(events)} events")
        for finding in report:
            print(f"  [{finding.severity.value}] {finding.code}: {finding.message}")
        if report.passed:
            print("audit passed")
        else:
            print(f"audit failed: {len(report.errors)} error(s), {len(report.warnings)} warning(s)")

    # Declaring --expected-start is an assertion that the data covers the
    # window, so coverage that falls short contradicts the caller and must fail
    # rather than warn. Without it, truncation stays advisory: the free tier
    # legitimately returns two years, and that is worth knowing, not fatal.
    truncation = [f for f in report.findings if f.code.startswith("coverage-")]

    if not report.passed:
        return EXIT_ERROR
    if args.expected_start and truncation:
        return EXIT_ERROR
    if args.warnings_as_errors and report.warnings:
        return EXIT_ERROR
    return EXIT_OK


NEWS_HANDLERS = {
    "fetch": cmd_news_fetch,
    "normalize": cmd_news_normalize,
    "analyze": cmd_news_analyze,
    "inspect": cmd_news_inspect,
    "audit": cmd_news_audit,
}


def cmd_news(args) -> int:
    return NEWS_HANDLERS[args.news_command](args)


HANDLERS = {
    "backtest": cmd_backtest,
    "compare": cmd_compare,
    "costs": cmd_costs,
    "matrix": cmd_matrix,
    "news": cmd_news,
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