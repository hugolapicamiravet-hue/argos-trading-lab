"""Event dataset queries: the only sanctioned way to read events in time order.

The important function here is `events_available_as_of`. "Available" means the
information was already public at `as_of`, which is judged by the event's
`published_at` and never by `analyzed_at`, by `retrieved_at`, or by the order
rows happen to sit in the file.

That distinction matters because this project is designed around one question:
*what did ARGOS know, and when could it have known it?* Running an analyzer today
over 2020 headlines produces `analyzed_at` in 2026. If any downstream consumer
read that column as the availability time, a 2020 backtest would appear to have
had perfect hindsight over six years of news. The correct reading is that the
2020 decision could have used the 2020 article, and nothing about the 2026
analysis changes that.

Row order is therefore never meaningful: every query sorts explicitly.
"""

from __future__ import annotations

import datetime as dt
import json
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

from argos.news.models import UTC, Direction, EventType, FinancialEvent, _require_aware

DEFAULT_EVENTS_FILENAME = "events.jsonl"


@dataclass(frozen=True, slots=True)
class EventDataset:
    """An in-memory set of events with as-of and filter queries.

    Construction sorts by `(published_at, event_id)`, which is both the audit
    order and the order a feature builder needs. The dataset is immutable, so a
    query cannot perturb what a later query sees.
    """

    events: tuple[FinancialEvent, ...] = ()
    name: str = "events"

    def __post_init__(self) -> None:
        object.__setattr__(self, "events", tuple(sorted(
            self.events, key=lambda e: (e.published_at, e.event_id)
        )))

    def __len__(self) -> int:
        return len(self.events)

    def __iter__(self) -> Iterator[FinancialEvent]:
        return iter(self.events)

    @property
    def span(self) -> tuple[dt.datetime | None, dt.datetime | None]:
        """`(earliest, latest)` publication instant, or `(None, None)` if empty."""
        if not self.events:
            return (None, None)
        return (self.events[0].published_at, self.events[-1].published_at)

    def available_as_of(self, as_of: dt.datetime) -> tuple[FinancialEvent, ...]:
        """Events published at or before `as_of`, earliest first.

        The boundary is inclusive: an article stamped exactly at `as_of` was
        public at that instant, and excluding it would silently discard events
        that a same-moment decision could legitimately have used.
        """
        moment = _require_aware(as_of, "as_of")
        return tuple(e for e in self.events if e.published_at <= moment)

    def for_ticker(self, ticker: str) -> EventDataset:
        symbol = str(ticker).strip().upper()
        return EventDataset(
            events=tuple(e for e in self.events if e.has_ticker(symbol)),
            name=f"{self.name}[{symbol}]",
        )

    def of_type(self, *event_types: EventType) -> EventDataset:
        wanted = {EventType.coerce(t) for t in event_types}
        return EventDataset(
            events=tuple(e for e in self.events if e.event_type in wanted),
            name=self.name,
        )

    def with_direction(self, *directions: Direction) -> EventDataset:
        wanted = {Direction.coerce(d) for d in directions}
        return EventDataset(
            events=tuple(e for e in self.events if e.direction in wanted),
            name=self.name,
        )

    def between(
        self, start: dt.datetime | None = None, end: dt.datetime | None = None
    ) -> EventDataset:
        """Half-open window `(start, end]`, matching the causal availability rule."""
        events = self.events
        if start is not None:
            low = _require_aware(start, "start")
            events = tuple(e for e in events if e.published_at > low)
        if end is not None:
            high = _require_aware(end, "end")
            events = tuple(e for e in events if e.published_at <= high)
        return EventDataset(events=events, name=self.name)

    def stats(self) -> dict[str, object]:
        """Counts by type and direction, for inspection and audit output."""
        by_type: dict[str, int] = {}
        by_direction: dict[str, int] = {}
        by_ticker: dict[str, int] = {}
        for event in self.events:
            by_type[event.event_type.value] = by_type.get(event.event_type.value, 0) + 1
            by_direction[event.direction.value] = by_direction.get(event.direction.value, 0) + 1
            for ticker in event.tickers:
                by_ticker[ticker] = by_ticker.get(ticker, 0) + 1
        earliest, latest = self.span
        return {
            "name": self.name,
            "count": len(self.events),
            "earliest_published_at": earliest.isoformat() if earliest else None,
            "latest_published_at": latest.isoformat() if latest else None,
            "by_type": dict(sorted(by_type.items())),
            "by_direction": dict(sorted(by_direction.items())),
            "by_ticker": dict(sorted(by_ticker.items())),
            "analyzers": dict(sorted(self._analyzer_counts().items())),
        }

    def _analyzer_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for event in self.events:
            counts[event.analyzer] = counts.get(event.analyzer, 0) + 1
        return counts


def build_dataset(
    events: Iterable[FinancialEvent], *, name: str = "events"
) -> EventDataset:
    """Convenience constructor mirroring `EventDataset` for a plain iterable."""
    return EventDataset(events=tuple(events), name=name)


def load_events(
    path: str | Path, *, name: str | None = None
) -> EventDataset:
    """Read a JSONL event file into a sorted dataset.

    Schema mismatches raise rather than being skipped. A silently ignored event
    would shift a feature value without any visible trace, which is the failure
    mode this project's tests are built to catch.
    """
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"event dataset not found: {source}")

    events: list[FinancialEvent] = []
    with source.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{source}:{number} is not valid JSON: {exc}") from exc
            try:
                events.append(FinancialEvent.from_dict(record))
            except (KeyError, ValueError, TypeError) as exc:
                raise ValueError(
                    f"{source}:{number} does not match the event schema: {exc}"
                ) from exc

    return EventDataset(events=tuple(events), name=name or source.stem)


def write_events(
    path: str | Path, events: Iterable[FinancialEvent]
) -> int:
    """Write events as JSONL, sorted, and return the row count."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(events, key=lambda e: (e.published_at, e.event_id))
    with target.open("w", encoding="utf-8") as handle:
        for event in ordered:
            handle.write(json.dumps(event.to_dict(), sort_keys=True, ensure_ascii=False) + "\n")
        handle.flush()
        import os

        os.fsync(handle.fileno())
    return len(ordered)


def events_available_as_of(
    dataset: EventDataset, as_of: dt.datetime
) -> tuple[FinancialEvent, ...]:
    """Module-level alias for `EventDataset.available_as_of`.

    Exists as a free function because the causal-availability question is asked
    from many places, and every one of them should be forced through the same
    implementation.
    """
    return dataset.available_as_of(as_of)


def events_for_trading_range(
    dataset: EventDataset, *, start: dt.datetime, end: dt.datetime
) -> EventDataset:
    """Events published after `start` and at or before `end`.

    Useful for slicing a dataset to the same window as a backtest, so a feature
    cannot see events from outside the period it claims to cover.
    """
    return dataset.between(start, end)


def audit_temporal_integrity(events: Sequence[FinancialEvent]) -> list[str]:
    """Report any event whose timestamps could not have been produced honestly.

    Returns a list of human-readable problems; empty means clean. Checks the
    invariants that are cheap to verify and expensive to discover later:

      * `analyzed_at` earlier than `published_at`, which would mean we judged an
        article before it existed;
      * `analyzed_at` far after publication, which is normal but worth counting,
        since a very long gap means the analyzer ran against text that may since
        have been edited;
      * events with no article reference, which break provenance.
    """
    problems: list[str] = []
    for event in events:
        if event.analyzed_at < event.published_at:
            problems.append(
                f"{event.event_id}: analyzed_at {event.analyzed_at.isoformat()} precedes "
                f"published_at {event.published_at.isoformat()}"
            )
        if not event.article_id:
            problems.append(f"{event.event_id}: no article reference; provenance broken")
    return problems


def utc_now() -> dt.datetime:
    return dt.datetime.now(UTC)