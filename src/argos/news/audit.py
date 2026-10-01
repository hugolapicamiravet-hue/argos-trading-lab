"""Audits: prove a news dataset is internally consistent and causally sound.

An audit answers questions you cannot answer by looking at the data, because
the data is what you are suspicious of. Every check here corresponds to a way
this pipeline could plausibly have lied:

    coverage    does the data actually cover the period the backtest claims?
    causality   could any event influence a decision before it was published?
    provenance  can every event be traced to an article and an analyzer version?
    duplicates  is one article inflating its own weight?
    consistency does what the manifest claims match what is on disk?

Findings carry a severity so a warning about free-tier coverage is not confused
with a broken invariant. `ERROR` means something is wrong and must be fixed
before any result is believed; `WARNING` means a limitation that must be stated
alongside any conclusion drawn from the data.
"""

from __future__ import annotations

import datetime as dt
import hashlib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum

from argos.errors import ArgosError
from argos.news.dedupe import article_key
from argos.news.events import EventDataset, audit_temporal_integrity
from argos.news.features import NewsFeaturePolicy, first_decision_date
from argos.news.manifest import Manifest
from argos.news.models import UTC, Direction, FinancialEvent, NewsArticle

#: Analyzers known to call a language model. Events from these must record which
#: model and which prompt produced them, otherwise the dataset cannot be
#: re-derived when a prompt changes and drift becomes invisible. A new LLM
#: analyzer has to be added here, and the `unaudited-analyzer` warning below is
#: what makes forgetting to do so visible.
MODEL_BACKED_ANALYZERS = frozenset({"openai"})

#: Analyzers whose output is reproducible from its inputs and version alone, so
#: a missing model is expected rather than a defect.
RULE_BASED_ANALYZERS = frozenset({"deterministic"})


class Severity(StrEnum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass(frozen=True, slots=True)
class Finding:
    """One audit observation."""

    code: str
    severity: Severity
    message: str

    def __str__(self) -> str:
        return f"[{self.severity.value.upper()}] {self.code}: {self.message}"

    def to_dict(self) -> dict:
        return {"code": self.code, "severity": self.severity.value, "message": self.message}


@dataclass(frozen=True, slots=True)
class AuditReport:
    """Collected findings plus the verdict they imply."""

    findings: tuple[Finding, ...] = ()

    def __len__(self) -> int:
        return len(self.findings)

    def __iter__(self):
        return iter(self.findings)

    @property
    def errors(self) -> tuple[Finding, ...]:
        return tuple(f for f in self.findings if f.severity is Severity.ERROR)

    @property
    def warnings(self) -> tuple[Finding, ...]:
        return tuple(f for f in self.findings if f.severity is Severity.WARNING)

    @property
    def passed(self) -> bool:
        """True when no invariant is broken. Warnings do not fail the audit."""
        return not self.errors

    def raise_if_failed(self) -> None:
        if not self.passed:
            lines = "\n".join(f"  {finding}" for finding in self.errors)
            raise AuditFailure(f"news dataset audit failed:\n{lines}")

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "errors": len(self.errors),
            "warnings": len(self.warnings),
            "findings": [f.to_dict() for f in self.findings],
        }


class AuditFailure(ArgosError):
    """Raised by `raise_if_failed` when an invariant is broken."""


def audit_timestamps(
    articles: Sequence[NewsArticle], events: Sequence[FinancialEvent]
) -> list[Finding]:
    """The core temporal audit.

    Checks, in order of how badly they would corrupt a backtest:

    1. Every `published_at` is timezone-aware. A naive timestamp would have been
       rejected at normalization, so reaching here means the check is load-bearing
       and worth asserting independently.
    2. No `retrieved_at` precedes its `published_at`.
    3. No event's `analyzed_at` precedes its `published_at`: we cannot have
       judged an article before it existed.
    4. Every event's `published_at` equals its source article's `published_at`,
       so the analyzer cannot have back-dated an event to buy itself lead time.
    """
    findings: list[Finding] = []
    by_id = {a.article_id: a for a in articles}

    for article in articles:
        offset = article.published_at.tzinfo and article.published_at.tzinfo.utcoffset(
            article.published_at
        )
        if offset is None:
            # Record and continue. Every comparison below mixes naive and aware
            # values, and comparing those raises TypeError, so an audit that
            # crashed on the first hand-edited timestamp would report nothing
            # about the file that is most suspect.
            findings.append(
                Finding(
                    "timestamp-naive",
                    Severity.ERROR,
                    f"article {article.article_id} has a naive published_at",
                )
            )
            continue
        if article.retrieved_at < article.published_at:
            findings.append(
                Finding(
                    "retrieval-before-publication",
                    Severity.ERROR,
                    f"article {article.article_id} was retrieved at "
                    f"{article.retrieved_at.isoformat()} before it was published at "
                    f"{article.published_at.isoformat()}",
                )
            )

    for event in events:
        for message in audit_temporal_integrity([event]):
            findings.append(Finding("event-timestamp", Severity.ERROR, message))
        article = by_id.get(event.article_id)
        if article is None:
            findings.append(
                Finding(
                    "orphan-event",
                    Severity.WARNING,
                    f"event {event.event_id} references article {event.article_id}, "
                    "which is not in the article set",
                )
            )
            continue
        if article.published_at.tzinfo is None or event.published_at.tzinfo is None:
            # Already reported above for the article; comparing would raise.
            continue
        if article.published_at != event.published_at:
            findings.append(
                Finding(
                    "publication-time-drift",
                    Severity.ERROR,
                    f"event {event.event_id} is stamped {event.published_at.isoformat()} but its "
                    f"article was published {article.published_at.isoformat()}; an event cannot "
                    "claim to be known earlier than its source",
                )
            )

    return findings


def audit_coverage(
    manifest: Manifest, *, expected_start: dt.datetime | None = None
) -> list[Finding]:
    """Check that coverage matches the requested window.

    This is where the provider's plan limits become visible. On Massive's free
    tier news history is two years, so a request spanning 2020-2025 returns
    whatever exists; if the manifest records coverage that starts in 2024, a
    reader can see immediately that a backtest over 2020 is not running on
    complete news data. Silence here would be the worst option, because it would
    look like the period was fully covered.
    """
    findings: list[Finding] = []

    if manifest.coverage_start is None:
        findings.append(Finding("coverage-empty", Severity.ERROR, "dataset contains no articles"))
        return findings

    if (
        expected_start is not None
        and manifest.coverage_start > expected_start + dt.timedelta(days=1)
    ):
        gap = manifest.coverage_start - expected_start
        findings.append(
            Finding(
                "coverage-truncated-start",
                Severity.WARNING,
                f"coverage starts {manifest.coverage_start.date()}, "
                f"{gap.days} day(s) after the requested start {expected_start.date()}; "
                "the provider may not serve this history on the current plan",
            )
        )

    if manifest.requested_window and manifest.requested_window[1]:
        requested_end = manifest.requested_window[1]
        if manifest.coverage_end is not None and manifest.coverage_end < requested_end:
            gap = requested_end - manifest.coverage_end
            findings.append(
                Finding(
                    "coverage-truncated-end",
                    Severity.WARNING,
                    f"coverage ends {manifest.coverage_end.date()}, {gap.days} day(s) before "
                    f"the requested end {requested_end.date()}",
                )
            )

    history_since = manifest.provider_limits.get("history_since")
    if isinstance(history_since, str) and history_since:
        # Paid plans advertise history from this date; free plans do not. A
        # manifest whose coverage predates the advertised limit is therefore
        # probably not the plan it claims to be.
        try:
            floor = dt.datetime.fromisoformat(history_since).replace(tzinfo=UTC)
        except ValueError:
            floor = None
        if floor and manifest.coverage_start < floor:
            findings.append(
                Finding(
                    "coverage-before-plan-limit",
                    Severity.WARNING,
                    f"coverage starts {manifest.coverage_start.date()}, earlier than the "
                    f"advertised history start {history_since} for this plan; verify the plan",
                )
            )

    return findings


def audit_duplicates(
    articles: Sequence[NewsArticle], events: Sequence[FinancialEvent] = ()
) -> list[Finding]:
    """Detect surviving duplicates that would double-count an article's weight."""
    findings: list[Finding] = []
    by_identity: dict[str, list[NewsArticle]] = {}
    for article in articles:
        by_identity.setdefault(article_key(article), []).append(article)

    for key, group in sorted(by_identity.items()):
        if len(group) > 1:
            findings.append(
                Finding(
                    "duplicate-articles",
                    Severity.ERROR,
                    f"{len(group)} articles share identity {key}: "
                    f"{', '.join(sorted(a.article_id for a in group))}",
                )
            )

    # Same fingerprint under different identities is the softer failure: two
    # distinct provider ids for one wire story, which will not be caught by the
    # identity cascade but will still double the score.
    by_fingerprint: dict[str, list[NewsArticle]] = {}
    for article in articles:
        by_fingerprint.setdefault(article.fingerprint, []).append(article)
    for fingerprint, group in sorted(by_fingerprint.items()):
        if len(group) > 1:
            findings.append(
                Finding(
                    "duplicate-content",
                    Severity.WARNING,
                    f"{len(group)} articles share content fingerprint {fingerprint[:12]} under "
                    f"different identities: {', '.join(sorted(a.article_id for a in group))}",
                )
            )

    # Repeated event ids. Storage appends are idempotent, so these come from a
    # merged or hand-edited event file -- and `daily_news_score` sums one
    # contribution per row, so a duplicate silently scales that day's score.
    repeated: dict[str, int] = {}
    for event in events:
        repeated[event.event_id] = repeated.get(event.event_id, 0) + 1
    for event_id, count in sorted(repeated.items()):
        if count > 1:
            findings.append(
                Finding(
                    "duplicate-events",
                    Severity.ERROR,
                    f"event id {event_id} appears {count} times; each row contributes to the "
                    "daily score, so the duplicate will double-count it",
                )
            )
    return findings


def audit_provenance(events: Sequence[FinancialEvent]) -> list[Finding]:
    """Every event must name its analyzer, version, and source article.

    Model-backed events must additionally name the model and the prompt. An
    LLM inference recorded without them cannot be reproduced when the prompt
    changes, which is how silent drift stays invisible across a long-running
    pipeline. Rule-based analyzers are exempt: their output follows from the
    inputs and the version alone.
    """
    findings: list[Finding] = []
    for event in events:
        if not event.analyzer or not event.analyzer_version:
            findings.append(
                Finding(
                    "missing-analyzer-identity",
                    Severity.ERROR,
                    f"event {event.event_id} has no analyzer identity",
                )
            )
        if not event.article_id:
            findings.append(
                Finding(
                    "missing-article-reference",
                    Severity.ERROR,
                    f"event {event.event_id} has no article reference",
                )
            )

        if (event.model is None) != (event.prompt_ref is None):
            # One without the other means a field was lost on the way to disk,
            # not that the analyzer is rule-based.
            findings.append(
                Finding(
                    "incomplete-model-provenance",
                    Severity.ERROR,
                    f"event {event.event_id} records model={event.model!r} but "
                    f"prompt_ref={event.prompt_ref!r}; both or neither",
                )
            )
        elif event.analyzer in MODEL_BACKED_ANALYZERS and event.model is None:
            findings.append(
                Finding(
                    "missing-model-identity",
                    Severity.ERROR,
                    f"event {event.event_id} came from model-backed analyzer "
                    f"{event.analyzer!r} but records no model",
                )
            )
        elif event.analyzer not in MODEL_BACKED_ANALYZERS | RULE_BASED_ANALYZERS:
            # Not an error: a custom analyzer is legitimate. But its events get
            # no model/provenance checking, and the caller should know that
            # rather than assume they were checked.
            findings.append(
                Finding(
                    "unaudited-analyzer",
                    Severity.WARNING,
                    f"event {event.event_id} used analyzer {event.analyzer!r}, which is not "
                    "classified as model-backed or rule-based, so its model and prompt "
                    "provenance were not verified",
                )
            )
    return findings


def audit_signal_leakage(
    events: Sequence[FinancialEvent], *, policy: NewsFeaturePolicy | None = None
) -> list[Finding]:
    """Verify the causal availability rule on real data.

    Confirms that for every event, its decision date is strictly after its
    publication date under the active policy. An event that could influence a
    decision on its own publication day would make every backtest that used it
    optimistic by construction, so this check is not decorative.

    Also reports events the scoring policy discards without saying so. A
    `require_ticker` policy drops untagged events on the floor, and a dataset
    where most events are untagged can still audit clean while contributing
    nothing to a score. Silence there reads as coverage.
    """
    active = policy or NewsFeaturePolicy()
    findings: list[Finding] = []
    untagged: list[str] = []
    for event in events:
        decision_date = first_decision_date(
            event.published_at, lag_days=active.publication_lag_days
        )
        published_date = event.published_at.astimezone(UTC).date()
        if decision_date <= published_date:
            findings.append(
                Finding(
                    "same-day-signal",
                    Severity.ERROR,
                    f"event {event.event_id} is usable on {decision_date}, its publication "
                    f"date; the policy requires a lag of {active.publication_lag_days} day(s)",
                )
            )
        if active.require_ticker and not event.tickers:
            untagged.append(event.event_id)

    if untagged:
        findings.append(
            Finding(
                "untagged-events-dropped",
                Severity.WARNING,
                f"{len(untagged)} event(s) carry no ticker and will be dropped by the "
                f"require_ticker policy, e.g. {', '.join(sorted(untagged)[:5])}",
            )
        )
    return findings


def audit_direction_consistency(events: Sequence[FinancialEvent]) -> list[Finding]:
    """Check that confident directional events have a usable confidence value.

    A POSITIVE event with `confidence=None` falls back to a full +1.0 weight in
    `contribution()`, which would let an uncalibrated analyzer dominate the
    score. Worth surfacing rather than silently compensating.
    """
    findings: list[Finding] = []
    uncalibrated = [
        e.event_id
        for e in events
        if e.direction in (Direction.POSITIVE, Direction.NEGATIVE) and e.confidence is None
    ]
    if uncalibrated:
        findings.append(
            Finding(
                "uncalibrated-direction",
                Severity.WARNING,
                f"{len(uncalibrated)} directional event(s) have no confidence and will be "
                f"weighted at full sign, e.g. {', '.join(sorted(uncalibrated)[:5])}",
            )
        )
    return findings


def audit_manifest_files(
    manifest: Manifest, *, root: str | None = None
) -> list[Finding]:
    """Verify the files the manifest names still exist with the recorded hashes.

    A manifest pointing at files that changed underneath it is worse than no
    manifest, because it lends false authority to a dataset nobody has verified.
    """
    from pathlib import Path

    findings: list[Finding] = []
    base = Path(root) if root else None

    for group, digests in (
        ("raw", manifest.raw_files),
        ("normalized", manifest.normalized_files),
        ("events", manifest.event_files),
    ):
        for digest in digests:
            candidate = Path(digest.path)
            path = (
                candidate
                if not base or candidate.is_absolute()
                else base / candidate
            )
            if not path.exists():
                findings.append(
                    Finding(
                        "manifest-missing-file",
                        Severity.ERROR,
                        f"{group} file {digest.path} named in the manifest is missing",
                    )
                )
                continue
            actual = path.stat().st_size
            if actual != digest.bytes:
                findings.append(
                    Finding(
                        "manifest-size-mismatch",
                        Severity.ERROR,
                        f"{digest.path} is {actual} bytes, manifest records {digest.bytes}",
                    )
                )
                continue
            # Size alone is not identity. Two datasets of the same length are
            # common -- a fetch that stopped one article short has exactly the
            # byte count of a complete one once row lengths line up -- so the
            # hash is what actually proves the file is the one described.
            content_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest.sha256 and content_hash != digest.sha256:
                findings.append(
                    Finding(
                        "manifest-hash-mismatch",
                        Severity.ERROR,
                        f"{digest.path} content hash {content_hash[:12]} does not match the "
                        f"manifest's {digest.sha256[:12]}",
                    )
                )
    return findings


def run_audit(
    *,
    articles: Iterable[NewsArticle] = (),
    events: Iterable[FinancialEvent] = (),
    dataset: EventDataset | None = None,
    manifest: Manifest | None = None,
    expected_start: dt.datetime | None = None,
    policy: NewsFeaturePolicy | None = None,
    manifest_root: str | None = None,
) -> AuditReport:
    """Run every applicable audit and collect the findings.

    Each input is optional so the same function serves `audit` on a full
    pipeline build, on a normalized-only dataset, or on a bare event file.
    """
    article_list = list(articles)
    event_list = list(events)
    if dataset is not None and not event_list:
        event_list = list(dataset.events)

    findings: list[Finding] = []
    findings.extend(audit_timestamps(article_list, event_list))
    findings.extend(audit_duplicates(article_list, event_list))
    findings.extend(audit_provenance(event_list))
    findings.extend(audit_signal_leakage(event_list, policy=policy))
    findings.extend(audit_direction_consistency(event_list))

    if manifest is not None:
        findings.extend(audit_coverage(manifest, expected_start=expected_start))
        findings.extend(audit_manifest_files(manifest, root=manifest_root))

    if not article_list and not event_list:
        findings.append(
            Finding(
                "nothing-to-audit", Severity.ERROR, "no articles or events supplied"
            )
        )

    return AuditReport(findings=tuple(findings))


def audit_from_store(
    store_root: str, *, dataset_name: str, policy: NewsFeaturePolicy | None = None
) -> AuditReport:
    """Audit a dataset already on disk, reading its manifest.

    Convenience for `argos news audit`, so the CLI does not have to reassemble
    the same four inputs each time.
    """
    from pathlib import Path

    root = Path(store_root)
    articles_path = root / "normalized" / f"{dataset_name}.jsonl"
    events_path = root / "events" / f"{dataset_name}.jsonl"
    manifest_path = root / "manifest.json"

    articles: list[NewsArticle] = []
    events: list[FinancialEvent] = []
    if articles_path.exists():
        articles = list(_read_jsonl(articles_path, NewsArticle.from_dict))
    if events_path.exists():
        events = list(_read_jsonl(events_path, FinancialEvent.from_dict))

    manifest_obj = None
    expected_start: dt.datetime | None = None
    if manifest_path.exists():
        manifest_obj = _manifest_from_dict(manifest_obj, manifest_path)
        if manifest_obj is not None and manifest_obj.requested_window:
            expected_start = manifest_obj.requested_window[0]

    return run_audit(
        articles=articles,
        events=events,
        manifest=manifest_obj,
        expected_start=expected_start,
        policy=policy,
        manifest_root=str(root),
    )


def _read_jsonl(path, builder):
    import json

    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(builder(json.loads(stripped)))
    return rows


def _manifest_from_dict(_existing, path):
    """Rebuild a `Manifest` from its JSON form for audit purposes."""
    import json

    from argos.news.manifest import MANIFEST_SCHEMA_VERSION, FileDigest

    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        return None

    def digest(items) -> tuple:
        out = []
        for item in items or ():
            out.append(
                FileDigest(
                    path=item["path"],
                    bytes=int(item["bytes"]),
                    sha256=item["sha256"],
                )
            )
        return tuple(out)

    def instant(value):
        if not value:
            return None
        return dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))

    window = raw.get("requested_window") or {}
    return Manifest(
        name=raw["name"],
        created_at=instant(raw["created_at"]) or dt.datetime.now(UTC),
        provider=raw.get("provider", "massive"),
        raw_files=digest(raw.get("raw_files")),
        normalized_files=digest(raw.get("normalized_files")),
        event_files=digest(raw.get("event_files")),
        article_count=int(raw.get("article_count", 0)),
        event_count=int(raw.get("event_count", 0)),
        coverage_start=instant(raw.get("coverage_start")),
        coverage_end=instant(raw.get("coverage_end")),
        requested_window=(
            (instant(window.get("start")), instant(window.get("end")))
            if isinstance(window, dict)
            else None
        ),
        tickers=tuple(raw.get("tickers") or ()),
        analyzers=tuple(raw.get("analyzers") or ()),
        provider_limits=dict(raw.get("provider_limits") or {}),
        feature_policy=dict(raw.get("feature_policy") or {}),
        rejected=int(raw.get("rejected", 0)),
        rejection_reasons=dict(raw.get("rejection_reasons") or {}),
        duplicates_dropped=int(raw.get("duplicates_dropped", 0)),
        unknowns=tuple(raw.get("unknowns") or ()),
        notes=tuple(raw.get("notes") or ()),
    )