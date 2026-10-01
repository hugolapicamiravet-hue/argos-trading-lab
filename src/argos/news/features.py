"""Causal news features.

The rule this module exists to enforce:

    A news feature for day D may only use events published strictly before the
    decision that consumes it.

ARGOS already resolves signals against the *next* bar's open, so a feature
computed from information available at the close of D produces a signal that is
executed at D+1's open. News is aligned to the same convention and is
deliberately stricter than strictly necessary:

    first_decision_date(published_at) = UTC date of published_at + 1 day

Why the extra day rather than same-day? An article timestamped 09:05 on D is
genuinely public during session D, but a batch pipeline that collects, parses
and analyzes news cannot be assumed to have completed that work before D's close.
Treating publication date as `decision_date - 1` means an article only reaches a
signal once a full day has passed, which is true for any realistic pipeline and
false for none. The cost is one day of latency; the benefit is that a leak would
have to be deliberate to appear.

Weighting is `sign(direction) * confidence`, and `magnitude` is deliberately not
used. Magnitude is an estimate of the same thing confidence already qualifies;
multiplying them would square one model's uncertainty and make the score harder
to reason about than the evidence warrants. This constant was fixed before any
result was observed, which is what stops it being a fitted parameter.

Nothing in this module trades. It produces a number per day; a strategy decides
what to do with it.
"""

from __future__ import annotations

import datetime as dt
from bisect import bisect_right
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from argos.news.events import EventDataset
from argos.news.models import UTC, EventType, FinancialEvent, _require_aware


@dataclass(frozen=True, slots=True)
class NewsFeaturePolicy:
    """Every constant the feature computation depends on, in one place.

    Kept as a frozen record so the policy is inspectable, serialisable into the
    dataset manifest, and impossible to change halfway through a parameter study
    without it showing up in the manifest.
    """

    # Days to wait after publication before an event may influence a decision.
    # 1 means "available from the next calendar day", the strictest value that
    # still uses news at all.
    publication_lag_days: int = 1
    # When True, only events carrying at least one ticker are counted. An
    # article with no ticker association cannot be attributed to any instrument,
    # so counting it would let macro noise dilute a single-name signal.
    require_ticker: bool = True
    # Whether events with direction UNKNOWN contribute. They contribute zero via
    # `Direction.sign`; this flag exists to make the choice explicit in the
    # manifest rather than implicit in the sign function.
    drop_unknown_direction: bool = True
    # Optional filter to restrict the feature to particular event types.
    event_types: tuple[EventType, ...] = ()
    # Optional minimum confidence for an event to count at all.
    min_confidence: float | None = None

    def __post_init__(self) -> None:
        if self.publication_lag_days < 0:
            raise ValueError(
                f"publication_lag_days must be >= 0, got {self.publication_lag_days}; "
                "a negative lag would make news visible before it was published"
            )
        object.__setattr__(
            self,
            "event_types",
            tuple(EventType.coerce(t) for t in self.event_types),
        )
        confidence = self.min_confidence
        if confidence is not None and not 0.0 <= confidence <= 1.0:
            raise ValueError(
                f"min_confidence must lie in [0, 1], got {self.min_confidence}"
            )

    def admits(self, event: FinancialEvent) -> bool:
        """Whether one event contributes to the score under this policy."""
        if self.require_ticker and not event.tickers:
            return False
        if self.drop_unknown_direction and event.direction.value == "UNKNOWN":
            return False
        if self.event_types and event.event_type not in self.event_types:
            return False
        return not (
            self.min_confidence is not None
            and (event.confidence is None or event.confidence < self.min_confidence)
        )

    def describe(self) -> dict[str, object]:
        return {
            "publication_lag_days": self.publication_lag_days,
            "require_ticker": self.require_ticker,
            "drop_unknown_direction": self.drop_unknown_direction,
            "event_types": [t.value for t in self.event_types],
            "min_confidence": self.min_confidence,
            "weighting": "sign(direction) * confidence; magnitude deliberately unused",
        }


def first_decision_date(published_at: dt.datetime, *, lag_days: int = 1) -> dt.date:
    """The first calendar date on which this event may influence a decision.

    `published_at` is normalised to UTC, then advanced by `lag_days`. A negative
    lag is rejected at policy construction; this function rejects it too so a
    direct caller cannot bypass the guard.
    """
    moment = _require_aware(published_at, "published_at")
    if lag_days < 0:
        raise ValueError(f"lag_days must be >= 0, got {lag_days}")
    return (moment.astimezone(UTC) + dt.timedelta(days=lag_days)).date()


def daily_news_score(
    events: Iterable[FinancialEvent],
    *,
    start: dt.date,
    end: dt.date,
    policy: NewsFeaturePolicy | None = None,
) -> dict[dt.date, float]:
    """Map each date in `[start, end]` to the news score available that day.

    The score for date D is the sum of contributions from every event whose
    `first_decision_date` is D, i.e. everything published by the end of D-1.
    Dates with no events get `0.0` rather than being absent, so downstream
    indexing never has to distinguish "no news" from "no row".

    `start` bounds the window by decision date, not by publication date. An
    event published before `start` still counts if its decision date falls
    inside the window, and one published inside it is excluded if its decision
    date falls outside. That is the correct behaviour for a windowed feature and
    needs no guard: nothing is silently lost, because every event either has a
    decision date in the window or belongs to a neighbouring one.
    """
    if end < start:
        raise ValueError(f"end {end} must not precede start {start}")
    active = policy or NewsFeaturePolicy()

    # Bucket by decision date. Events whose decision date lands outside the
    # window are simply not part of this window's feature; they belong to a
    # neighbouring one.
    buckets: dict[dt.date, float] = {}
    for event in events:
        if not active.admits(event):
            continue
        decision_date = first_decision_date(
            event.published_at, lag_days=active.publication_lag_days
        )
        if not start <= decision_date <= end:
            continue
        buckets[decision_date] = buckets.get(decision_date, 0.0) + event.contribution()

    # Materialize every date so "no events" is an explicit 0.0.
    return {day: round(buckets.get(day, 0.0), 12) for day in _dates(start, end)}


def cumulative_news_score(
    events: Iterable[FinancialEvent],
    *,
    start: dt.date,
    end: dt.date,
    policy: NewsFeaturePolicy | None = None,
) -> dict[dt.date, float]:
    """Running total of `daily_news_score` across the window.

    Provided as a separate function rather than a flag because the two are
    genuinely different features and conflating them is a silent modelling
    choice: a windowed score forgets yesterday, a cumulative one remembers it
    forever. Make the choice explicit at the call site.
    """
    daily = daily_news_score(events, start=start, end=end, policy=policy)
    running = 0.0
    result: dict[dt.date, float] = {}
    for day in sorted(daily):
        running += daily[day]
        result[day] = round(running, 12)
    return result


def align_score_to_index(
    score: Mapping[dt.date, float], index_dates: Sequence[object]
) -> list[float]:
    """Project a date-keyed score onto a bar index, carrying the last known value.

    A score exists for calendar days; bars exist for sessions. Carrying the last
    known score forward is correct: news published on a Saturday is still
    relevant on Monday. Zero-filling instead would invent the disappearance of
    news over every weekend.

    The lookup is a backward bisect over sorted dates, so it is structurally
    incapable of reading a future score into a past bar. A forward scan would
    also be correct, and the reason this is not one is that "can I accidentally
    look ahead here?" should be answered by the shape of the code, not by
    reading it carefully.
    """
    if not score:
        # Still parse the index dates. Returning early would silently accept a
        # malformed index just because there happened to be no news, which is a
        # trap that only appears in the one run where you are not looking.
        for raw_date in index_dates:
            _to_date(raw_date)
        return [0.0] * len(index_dates)

    ordered = sorted(score.items())
    days = [day for day, _ in ordered]
    values = [value for _, value in ordered]

    aligned: list[float] = []
    for raw_date in index_dates:
        day = _to_date(raw_date)
        # `right` - 1 is the last entry with date <= day, or -1 if all are later.
        position = bisect_right(days, day) - 1
        aligned.append(values[position] if position >= 0 else 0.0)
    return aligned


def _dates(start: dt.date, end: dt.date) -> Iterable[dt.date]:
    current = start
    while current <= end:
        yield current
        current += dt.timedelta(days=1)


def _to_date(value: object) -> dt.date:
    if isinstance(value, dt.datetime):
        moment = _require_aware(value, "index date")
        return moment.astimezone(UTC).date()
    if isinstance(value, dt.date):
        return value
    text = str(value)
    # Tolerate the ISO date prefix of a timestamp string, and a bare date.
    try:
        return dt.date.fromisoformat(text[:10])
    except ValueError as exc:
        raise ValueError(f"cannot interpret {value!r} as a date") from exc


def feature_for_dataset(
    dataset: EventDataset,
    *,
    start: dt.date,
    end: dt.date,
    policy: NewsFeaturePolicy | None = None,
) -> dict[dt.date, float]:
    """Convenience wrapper accepting an `EventDataset` directly."""
    return daily_news_score(dataset.events, start=start, end=end, policy=policy)