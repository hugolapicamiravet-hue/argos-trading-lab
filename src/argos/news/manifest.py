"""Dataset manifest: what a news dataset is, and what it is allowed to contain.

A dataset without a manifest is an assertion. A dataset with one is a
falsifiable claim: these articles, from this provider, with this plan's limits,
covering this window, produced by these files with these checksums, under these
policies.

That matters here more than in most backtesting work, because the pipeline has a
provider dependency whose free tier genuinely cannot reach the project's
baseline period. A manifest that records `history_since: 2016-06-22` and
`requests_per_minute: 5` next to the actual coverage makes that gap visible
immediately, instead of letting someone conclude that a 2020-2025 news backtest
ran against complete data when it may have run against two years.

The manifest never contains an API key. Any field that could plausibly embed
one is passed through redaction on the way in.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

from argos.errors import ArgosError
from argos.news.events import EventDataset
from argos.news.features import NewsFeaturePolicy
from argos.news.models import UTC, FinancialEvent, NewsArticle

MANIFEST_SCHEMA_VERSION = 1
MANIFEST_FILENAME = "manifest.json"


class ManifestError(ArgosError):
    """A manifest file is missing, unreadable, or internally inconsistent."""


@dataclass(frozen=True, slots=True)
class FileDigest:
    """One file's identity: path, size, and SHA-256 of its bytes."""

    path: str
    bytes: int
    sha256: str

    def to_dict(self) -> dict:
        return {"path": self.path, "bytes": self.bytes, "sha256": self.sha256}

    @classmethod
    def of(
        cls,
        path: str | os.PathLike[str],
        *,
        root: str | os.PathLike[str] | None = None,
    ) -> FileDigest:
        target = Path(path)
        data = target.read_bytes()
        return cls(
            path=str(target.relative_to(root)) if root else str(target),
            bytes=len(data),
            sha256=hashlib.sha256(data).hexdigest(),
        )


@dataclass(frozen=True, slots=True)
class Manifest:
    """Self-describing record of one dataset build.

    `unknowns` is a first-class field. Research that could not verify something
    from official documentation belongs here as an explicit gap, not as a silent
    default or an optimistic assumption. Entries are free-form strings so a new
    unknown can be recorded without a schema change.
    """

    name: str
    created_at: dt.datetime
    schema_version: int = MANIFEST_SCHEMA_VERSION
    provider: str = "massive"
    source_kind: str = "news"
    raw_files: tuple[FileDigest, ...] = ()
    normalized_files: tuple[FileDigest, ...] = ()
    event_files: tuple[FileDigest, ...] = ()
    article_count: int = 0
    event_count: int = 0
    coverage_start: dt.datetime | None = None
    coverage_end: dt.datetime | None = None
    requested_window: tuple[dt.datetime | None, dt.datetime | None] | None = None
    tickers: tuple[str, ...] = ()
    analyzers: tuple[Mapping[str, object], ...] = ()
    provider_limits: Mapping[str, object] = field(default_factory=dict)
    feature_policy: Mapping[str, object] = field(default_factory=dict)
    rejected: int = 0
    rejection_reasons: Mapping[str, int] = field(default_factory=dict)
    duplicates_dropped: int = 0
    unknowns: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "created_at", self.created_at.astimezone(UTC))

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["created_at"] = self.created_at.isoformat().replace("+00:00", "Z")
        for key in ("coverage_start", "coverage_end"):
            value = payload.get(key)
            if isinstance(value, dt.datetime):
                payload[key] = value.isoformat().replace("+00:00", "Z")
        if self.requested_window:
            start, end = self.requested_window
            payload["requested_window"] = {
                "start": start.isoformat().replace("+00:00", "Z") if start else None,
                "end": end.isoformat().replace("+00:00", "Z") if end else None,
            }
        for key in ("raw_files", "normalized_files", "event_files"):
            payload[key] = [
                item.to_dict() if isinstance(item, FileDigest) else item
                for item in payload[key]
            ]
        return payload

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True, ensure_ascii=False)

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> Manifest:
        """Rebuild a manifest from `to_dict` output.

        The inverse of `to_dict`, so a manifest written to disk can be read back
        and audited later -- which is the only reason to persist one. Timestamps
        must carry an explicit offset: a naive one would compare against UTC
        fields and raise rather than report the problem.
        """

        def moment(value: object) -> dt.datetime | None:
            if value is None:
                return None
            if isinstance(value, dt.datetime):
                parsed = value
            else:
                text = str(value)
                try:
                    parsed = dt.datetime.fromisoformat(text.replace("Z", "+00:00"))
                except ValueError as exc:
                    raise ManifestError(f"unreadable timestamp {value!r}") from exc
            if parsed.tzinfo is None or parsed.tzinfo.utcoffset(parsed) is None:
                raise ManifestError(f"timestamp {value!r} has no UTC offset")
            return parsed.astimezone(UTC)

        def digests(value: object) -> tuple[FileDigest, ...]:
            if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
                return ()
            return tuple(
                FileDigest(
                    path=str(item["path"]),
                    bytes=int(item["bytes"]),
                    sha256=str(item["sha256"]),
                )
                for item in value
            )

        window = payload.get("requested_window")
        requested: tuple[dt.datetime | None, dt.datetime | None] | None = None
        if isinstance(window, Mapping):
            requested = (moment(window.get("start")), moment(window.get("end")))
        elif isinstance(window, Sequence) and not isinstance(window, (str, bytes)):
            start, end = (list(window) + [None, None])[:2]
            requested = (moment(start), moment(end))

        known = {f.name for f in fields(cls)}
        arguments = {key: value for key, value in payload.items() if key in known}
        arguments.update(
            created_at=moment(arguments.get("created_at")) or dt.datetime.now(UTC),
            coverage_start=moment(arguments.get("coverage_start")),
            coverage_end=moment(arguments.get("coverage_end")),
            requested_window=requested,
            raw_files=digests(arguments.get("raw_files")),
            normalized_files=digests(arguments.get("normalized_files")),
            event_files=digests(arguments.get("event_files")),
            tickers=tuple(arguments.get("tickers") or ()),
            analyzers=tuple(arguments.get("analyzers") or ()),
            unknowns=tuple(arguments.get("unknowns") or ()),
            notes=tuple(arguments.get("notes") or ()),
            provider_limits=dict(arguments.get("provider_limits") or {}),
            feature_policy=dict(arguments.get("feature_policy") or {}),
            rejection_reasons=dict(arguments.get("rejection_reasons") or {}),
        )
        return cls(**arguments)

    def summary(self) -> str:
        """One-paragraph human summary, for CLI output."""
        coverage = "empty"
        if self.coverage_start and self.coverage_end:
            coverage = f"{self.coverage_start.date()} to {self.coverage_end.date()}"
        parts = [
            f"{self.name}: {self.article_count} articles, {self.event_count} events, "
            f"coverage {coverage}"
        ]
        if self.requested_window and (self.requested_window[0] or self.requested_window[1]):
            start, end = self.requested_window
            wanted = f"{start.date() if start else '?'} to {end.date() if end else '?'}"
            parts.append(f"requested {wanted}")
        requested_start = self.requested_window[0] if self.requested_window else None
        if (
            self.coverage_start
            and requested_start
            and self.coverage_start > requested_start
        ):
            parts.append(
                "WARNING: coverage starts after the requested window; the provider may not "
                "have this history on the current plan"
            )
        if self.rejected:
            parts.append(f"{self.rejected} records rejected")
        if self.duplicates_dropped:
            parts.append(f"{self.duplicates_dropped} duplicates dropped")
        if self.unknowns:
            parts.append(f"{len(self.unknowns)} unverified item(s) recorded")
        return "; ".join(parts)


class NewsManifestBuilder:
    """Accumulate facts about a build, then emit a `Manifest`.

    Built as a builder rather than assembled by hand at the call site because
    the same facts are needed by `normalize`, `analyze` and `audit`, and a
    hand-assembled manifest is where fields quietly go missing.
    """

    def __init__(
        self,
        name: str,
        *,
        provider: str = "massive",
        requested_window: tuple[dt.datetime | None, dt.datetime | None] | None = None,
        provider_limits: Mapping[str, object] | None = None,
        unknowns: Sequence[str] = (),
        notes: Sequence[str] = (),
        root: str | os.PathLike[str] | None = None,
    ) -> None:
        self.name = name
        self.provider = provider
        self.requested_window = requested_window
        self.provider_limits = dict(provider_limits or {})
        self.unknowns = list(unknowns)
        self.notes = list(notes)
        self.root = Path(root) if root else None

        self._raw: list[FileDigest] = []
        self._normalized: list[FileDigest] = []
        self._events: list[FileDigest] = []
        self._article_count = 0
        self._event_count = 0
        self._coverage: tuple[dt.datetime | None, dt.datetime | None] = (None, None)
        self._tickers: set[str] = set()
        self._analyzers: dict[tuple[str, str, str | None], Mapping[str, object]] = {}
        self._rejected = 0
        self._rejection_reasons: dict[str, int] = {}
        self._duplicates = 0
        self._policy: dict[str, object] = {}

    def add_raw(self, path: str | os.PathLike[str]) -> NewsManifestBuilder:
        self._raw.append(FileDigest.of(path, root=self.root))
        return self

    def add_normalized(self, path: str | os.PathLike[str]) -> NewsManifestBuilder:
        self._normalized.append(FileDigest.of(path, root=self.root))
        return self

    def add_events(self, path: str | os.PathLike[str]) -> NewsManifestBuilder:
        self._events.append(FileDigest.of(path, root=self.root))
        return self

    def record_articles(self, articles: Iterable[NewsArticle]) -> NewsManifestBuilder:
        """Fold a batch of articles into counts, coverage and ticker list."""
        count = 0
        for article in articles:
            count += 1
            self._tickers.update(article.tickers)
            start, end = self._coverage
            self._coverage = (
                article.published_at if start is None else min(start, article.published_at),
                article.published_at if end is None else max(end, article.published_at),
            )
        self._article_count += count
        return self

    def record_events(self, events: Iterable[FinancialEvent]) -> NewsManifestBuilder:
        count = 0
        for event in events:
            count += 1
            self._tickers.update(event.tickers)
            # `prompt_ref` belongs in the key, not just in the payload. Two runs
            # of the same analyzer version and model under different prompts
            # produce different interpretations, and collapsing them into one row
            # would report the first prompt as the one behind every event.
            key = (
                event.analyzer,
                event.analyzer_version,
                event.model,
                event.prompt_ref,
            )
            # Counted per event, not per analyzer. Reading this as "did the first
            # event fail" made a dataset where the model failed on most records
            # report a single failure and look clean.
            failed = 1 if event.rationale is None and event.confidence == 0.0 else 0
            existing = self._analyzers.get(key)
            if existing is None:
                self._analyzers[key] = {
                    "analyzer": event.analyzer,
                    "version": event.analyzer_version,
                    "model": event.model,
                    "prompt_ref": event.prompt_ref,
                    "events": 1,
                    "failed": failed,
                }
            else:
                existing["events"] = int(existing["events"]) + 1
                existing["failed"] = int(existing["failed"]) + failed
            start, end = self._coverage
            self._coverage = (
                event.published_at if start is None else min(start, event.published_at),
                event.published_at if end is None else max(end, event.published_at),
            )
        self._event_count += count
        return self

    def record_rejections(
        self,
        rejected: Mapping[int, str] | Sequence[tuple[int, str]],
    ) -> NewsManifestBuilder:
        """Record rejected raw records with their reason counts.

        Reasons are bucketed by text so the manifest says "12 rows had no
        published_at" rather than 12 separate free-text entries.
        """
        reasons = (
            rejected.values() if isinstance(rejected, Mapping) else [r for _, r in rejected]
        )
        for reason in reasons:
            key = reason.split(":")[0].strip() or "unspecified"
            self._rejection_reasons[key] = self._rejection_reasons.get(key, 0) + 1
            self._rejected += 1
        return self

    def record_duplicates(self, count: int) -> NewsManifestBuilder:
        self._duplicates += int(count)
        return self

    def record_policy(self, policy: NewsFeaturePolicy | None) -> NewsManifestBuilder:
        self._policy = policy.describe() if policy else {}
        return self

    def add_unknown(self, text: str) -> NewsManifestBuilder:
        if text not in self.unknowns:
            self.unknowns.append(text)
        return self

    def add_note(self, text: str) -> NewsManifestBuilder:
        if text not in self.notes:
            self.notes.append(text)
        return self

    def build(self) -> Manifest:
        return Manifest(
            name=self.name,
            created_at=dt.datetime.now(UTC),
            provider=self.provider,
            raw_files=tuple(self._raw),
            normalized_files=tuple(self._normalized),
            event_files=tuple(self._events),
            article_count=self._article_count,
            event_count=self._event_count,
            coverage_start=self._coverage[0],
            coverage_end=self._coverage[1],
            requested_window=self.requested_window,
            tickers=tuple(sorted(self._tickers)),
            analyzers=tuple(
                self._analyzers[key]
                for key in sorted(
                    self._analyzers,
                    key=lambda k: (k[0], k[1], k[2] or "", k[3] or ""),
                )
            ),
            provider_limits=dict(self.provider_limits),
            feature_policy=dict(self._policy),
            rejected=self._rejected,
            rejection_reasons=dict(sorted(self._rejection_reasons.items())),
            duplicates_dropped=self._duplicates,
            unknowns=tuple(self.unknowns),
            notes=tuple(self.notes),
        )


def write_manifest(manifest: Manifest, path: str | os.PathLike[str]) -> Path:
    """Write a manifest atomically, next to the data it describes."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(target.suffix + ".tmp")
    temp.write_text(manifest.to_json() + "\n", encoding="utf-8")
    os.replace(temp, target)
    return target


def read_manifest(path: str | os.PathLike[str]) -> Mapping[str, object]:
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"manifest not found: {target}")
    decoded = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(decoded, Mapping):
        raise ValueError(f"{target} does not contain a manifest object")
    version = decoded.get("schema_version")
    if version != MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            f"{target} has manifest schema version {version}, this build reads "
            f"{MANIFEST_SCHEMA_VERSION}"
        )
    return decoded


def manifest_for_dataset(
    dataset: EventDataset, *, name: str, provider: str = "massive"
) -> NewsManifestBuilder:
    """Convenience builder seeded from an in-memory dataset."""
    builder = NewsManifestBuilder(name, provider=provider)
    builder.record_events(dataset.events)
    return builder