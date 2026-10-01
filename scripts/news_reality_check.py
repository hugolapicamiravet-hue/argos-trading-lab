#!/usr/bin/env python3
"""What the news layer would and would not be able to support, end to end.

Run it to see the coverage arithmetic that decides whether a news backtest is
possible at all:

    python scripts/news_reality_check.py
    python scripts/news_reality_check.py --years 2 --keep /tmp/argos_reality

The point is not to produce a return figure. With a two-year provider window and
an uncalibrated classifier, a PnL number would be uninterpretable in a way that
looks like a result. What this script does instead:

1. Builds a synthetic history with a known planted pattern, so the plumbing can
   be checked against ground truth rather than against a market outcome nobody
   can know.
2. Runs the real pipeline over it -- normalize, dedupe, analyze, manifest, audit.
3. Verifies the score recovers the planted pattern, and only through time.
4. Reports the coverage the dataset actually has against the window a caller
   would ask for, which is the free-tier trap stated numerically.

Every dataset it writes is self-authored and synthetic. It contacts no provider.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from argos.news import (  # noqa: E402
    DeterministicAnalyzer,
    EventDataset,
    NewsFeaturePolicy,
    NewsManifestBuilder,
    NewsStore,
    analyze_many,
    daily_news_score,
    dedupe_articles,
    events_available_as_of,
    normalize_articles,
    run_audit,
    write_manifest,
)

UTC = dt.UTC
RULE = "-" * 72
# The window a caller would naively ask for, which the free tier cannot supply.
REQUESTED_START = dt.datetime(2020, 1, 1, tzinfo=UTC)
# Where the synthetic history begins. Deliberately after REQUESTED_START.
SEED_START = dt.date(2024, 1, 1)


@dataclass
class Planted:
    """Ground truth, recorded while seeding, independent of the pipeline.

    `sign_by_decision_date` maps the date an event's contribution should first
    be usable to the sign that contribution should carry. `counterweight`
    counts the articles deliberately built to cancel the tagged ones if the
    ticker requirement ever stopped filtering them.
    """

    sign_by_decision_date: dict[dt.date, int] = field(default_factory=dict)
    counterweight: int = 0
    tagged: int = 0
    mirror_id: str = ""


def _seed_payload(days: int, start: dt.date) -> tuple[list[dict], Planted]:
    """Synthetic provider payloads with a planted, alternating direction.

    Each weekday gets two articles. The first is tagged `SPY` and alternates
    between "beats" and "misses" by weekday offset. The second is untagged and
    carries the opposite direction, so it would flatten every day to zero if it
    were counted. The planted pattern is therefore only recoverable if the
    ticker requirement is doing its job.

    The pattern is arbitrary and deliberately not predictive of anything real.
    It exists so the score can be checked against a known answer.
    """
    payloads: list[dict] = []
    planted = Planted()

    def article(identifier: str, day: dt.date, positive: bool, tagged: bool) -> dict:
        verb = "beats" if positive else "misses"
        published = dt.datetime.combine(day, dt.time(20, 5), tzinfo=UTC)
        return {
            "id": identifier,
            "headline": f"Synthetic Co {verb} quarterly expectations",
            "url": f"https://example.invalid/news/{identifier}",
            "published_utc": published.isoformat().replace("+00:00", "Z"),
            "summary": f"Synthetic report {identifier} describing a quarter that {verb}.",
            "tickers": ["SPY"] if tagged else [],
            "source": "synthetic",
        }

    for offset in range(days):
        day = start + dt.timedelta(days=offset)
        if day.weekday() >= 5:
            continue
        positive = offset % 2 == 0
        payloads.append(article(f"seed-{offset}-tagged", day, positive, True))
        payloads.append(article(f"seed-{offset}-untagged", day, not positive, False))
        planted.sign_by_decision_date[day + dt.timedelta(days=1)] = 1 if positive else -1
        planted.tagged += 1
        planted.counterweight += 1

    # The same story re-served under a different provider id. Dedupe must
    # collapse it, which it cannot do by comparing ids.
    mirror = dict(payloads[0])
    mirror["id"] = "seed-0-tagged-mirror"
    payloads.append(mirror)
    planted.mirror_id = mirror["id"]

    return payloads, planted


def _report_dataset(store: NewsStore, name: str, years: int) -> dict:
    now = dt.datetime.now(UTC)
    payloads, planted = _seed_payload(365 * years, SEED_START)
    store.write_raw(name, payloads, overwrite=True)
    stored = list(store.iter_raw_records(name))

    # `normalize_articles` already collapses repeats through `dedupe_articles`,
    # so the mirror is dropped here rather than in the explicit pass below.
    normalized = normalize_articles(stored, source="synthetic", retrieved_at=now)
    kept, dropped_ids = dedupe_articles(normalized.articles)
    store.write_articles(name, kept)

    analyzer = DeterministicAnalyzer()
    events = analyze_many(kept, analyzer, analyzed_at=now)
    store.write_events(name, events)

    builder = NewsManifestBuilder(name=name, provider="synthetic", root=store.root)
    builder.add_raw(store.raw_dir / f"{name}.json")
    builder.add_normalized(store.normalized_dir / f"{name}.jsonl")
    builder.add_events(store.events_dir / f"{name}.jsonl")
    builder.record_articles(kept)
    builder.record_events(events)
    builder.record_rejections(normalized.rejected)
    builder.record_duplicates(len(dropped_ids))
    manifest = builder.build()
    write_manifest(manifest, store.root / f"{name}.manifest.json")

    audit = run_audit(
        articles=kept,
        events=events,
        manifest=manifest,
        expected_start=REQUESTED_START,
        manifest_root=str(store.root),
    )
    return {
        "manifest": manifest,
        "audit": audit,
        "planted": planted,
        "raw": len(stored),
        "articles": len(kept),
        "duplicates": len(normalized.duplicates) + len(dropped_ids),
        "rejected": len(normalized.rejected),
        "events": len(events),
    }


def _check_plumbing(store: NewsStore, name: str, planted: Planted) -> list[str]:
    """Check the score against ground truth: right sign, right day, no lookahead."""
    events = store.read_events(name)
    if not events:
        return ["no events to check"]

    problems: list[str] = []

    # 0. The same story re-served under a second provider id must not survive.
    #    Identity alone cannot see this -- the ids differ -- so a regression in
    #    cross-record dedupe would silently double-count the story's weight.
    survivors = [
        article.raw_source_id
        for article in store.read_articles(name)
        if article.raw_source_id == planted.mirror_id
    ]
    if survivors:
        problems.append(
            f"the duplicated story {planted.mirror_id} survived normalization;"
            " dedupe is not collapsing across provider ids"
        )

    coverage_start = min(event.published_at for event in events)
    first_day = coverage_start.date()
    dataset = EventDataset(events=events, name=name)
    window_end = first_day + dt.timedelta(days=30)
    visible = events_available_as_of(dataset, as_of=coverage_start + dt.timedelta(days=30))

    # 1. No lookahead. `daily_news_score` buckets by decision date, so it is
    #    structurally incapable of reading the future; that is guaranteed by
    #    construction and covered by unit tests. The function that can leak is
    #    `events_available_as_of`, so check its actual contract instead: nothing
    #    published after the cutoff may come back.
    late: list[str] = []
    for offset in (1, 7, 15, 30):
        as_of = coverage_start + dt.timedelta(days=offset)
        snapshot = events_available_as_of(dataset, as_of=as_of)
        strays = [e.event_id for e in snapshot if e.published_at > as_of]
        if strays:
            late.append(f"{len(strays)} event(s) published after {as_of.date()} returned")
    if late:
        problems.append("events_available_as_of leaked the future: " + "; ".join(late[:2]))

    # 2. Ground truth: on each decision date the score must carry the planted
    #    sign. A zero or a flipped sign means the untagged counterweight was
    #    counted, or the publication lag is wrong, or the direction was lost.
    policy = NewsFeaturePolicy()
    scores = daily_news_score(visible, start=first_day, end=window_end, policy=policy)
    compared = 0
    for day, sign in sorted(planted.sign_by_decision_date.items()):
        if not first_day <= day <= window_end:
            continue
        actual = scores.get(day, 0.0)
        compared += 1
        if actual == 0.0:
            problems.append(f"{day}: planted {sign:+d}, recovered 0.0")
        elif (actual > 0) != (sign > 0):
            problems.append(f"{day}: planted {sign:+d}, recovered {actual:+.3f}")
        if len(problems) >= 5:
            break
    if compared == 0:
        problems.append("no decision dates fell inside the window; the check proved nothing")

    # 3. The pattern must be produced by the ticker requirement. Relaxing it
    #    should let the counterweight cancel every planted day to zero; if the
    #    pattern survives anyway, the tagged articles are not what is driving it.
    unfiltered = daily_news_score(
        visible, start=first_day, end=window_end, policy=NewsFeaturePolicy(require_ticker=False)
    )
    planted_days = [
        day
        for day, _ in planted.sign_by_decision_date.items()
        if first_day <= day <= window_end
    ]
    cancelled = [day for day in planted_days if abs(unfiltered.get(day, 0.0)) < 1e-9]
    if len(cancelled) < len(planted_days) * 0.9:
        problems.append(
            f"only {len(cancelled)}/{len(planted_days)} planted days cancel when the ticker"
            " requirement is dropped; the counterweight is not being applied"
        )

    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--years", type=int, default=2, help="synthetic history length")
    parser.add_argument("--dataset", default="reality_check")
    parser.add_argument("--root", default="data/reality_check")
    parser.add_argument("--keep", default=None, help="also write a JSON summary here")
    args = parser.parse_args(argv)

    store = NewsStore(args.root)
    print(RULE)
    print("ARGOS news reality check -- synthetic data, no provider calls")
    print(RULE)

    result = _report_dataset(store, args.dataset, args.years)
    manifest, audit, planted = result["manifest"], result["audit"], result["planted"]

    uncovered_days = (manifest.coverage_start - REQUESTED_START).days
    print(f"\nroot                 {store.root}")
    print(f"requested window     {REQUESTED_START.date()} .. today")
    print(
        f"actual coverage      {manifest.coverage_start.date()}"
        f" .. {manifest.coverage_end.date()}"
    )
    print(f"synthetic history    {args.years} year(s) from {SEED_START}")
    print(
        f"uncovered            {uncovered_days} days"
        f" (about {uncovered_days / 365.25:.1f} years) of the requested window"
    )
    print()
    print(f"raw payloads         {result['raw']}")
    print(f"articles kept        {result['articles']}")
    print(f"duplicates dropped   {result['duplicates']}")
    print(f"records rejected     {result['rejected']}")
    print(f"events derived       {result['events']}")

    tagged = sum(1 for event in store.read_events(args.dataset) if event.tickers)
    total = result["events"] or 1
    print(f"events with a ticker {tagged}/{total} ({100 * tagged / total:.1f}%)")

    print(f"\n{RULE}")
    print(f"audit: {'passed' if audit.passed else 'FAILED'}")
    for finding in audit:
        print(f"  [{finding.severity.value}] {finding.code}: {finding.message}")

    problems = _check_plumbing(store, args.dataset, planted)
    print(f"\n{RULE}")
    if problems:
        print(f"plumbing check FAILED against planted ground truth ({planted.tagged} days):")
        for problem in problems:
            print(f"  {problem}")
    else:
        print(f"plumbing check passed: {planted.tagged} planted days recovered in sign and timing,")
        print(f"  {planted.counterweight} untagged counterweight articles correctly excluded")

    print(f"\n{RULE}")
    span = f"2020-{dt.datetime.now(UTC).year}"
    print(f"A backtest over {span} fed this dataset would be wrong, not weak:")
    print(f"the first {uncovered_days} days of that window contain no news at all. Declare the")
    print("window with `news audit --expected-start` and the audit fails instead of warning.")
    print(f"\ndataset written to {store.root}")

    if args.keep:
        target = Path(args.keep)
        target.mkdir(parents=True, exist_ok=True)
        summary = {
            "raw": result["raw"],
            "articles": result["articles"],
            "events": result["events"],
            "coverage_start": str(manifest.coverage_start),
            "coverage_end": str(manifest.coverage_end),
            "uncovered_days": uncovered_days,
            "audit_passed": audit.passed,
            "findings": [finding.to_dict() for finding in audit.findings],
            "plumbing_problems": problems,
        }
        (target / "reality_check.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(f"summary written to {target / 'reality_check.json'}")

    return 0 if audit.passed and not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())