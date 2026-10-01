"""Does the news feature layer actually work, as opposed to merely run?

The obvious way to answer that would be to build a strategy on the score and
report a PnL. That is exactly the wrong experiment here: the available history
is about two years, the analyzer's confidence is uncalibrated, and a number
produced that way would be uninterpretable in a way that looks like a result.

So this module tests the plumbing against a signal it plants, rather than
against a market outcome it cannot know. If the lag were wrong by a day, or if
the score leaked same-day news, or if adding future articles rewrote past
scores, these tests would fail. Passing them says the layer would surface a real
relationship if one existed -- which is the precondition for building a strategy,
and not a claim that the relationship does.
"""

from __future__ import annotations

import datetime as dt

import pytest
from tests.news.conftest import UTC, make_event

from argos.news import (
    Direction,
    FinancialEvent,
    NewsFeaturePolicy,
    build_dataset,
    cumulative_news_score,
    daily_news_score,
    events_available_as_of,
    first_decision_date,
)

START = dt.date(2025, 1, 6)
DAYS = 40


def published(day_offset: int) -> dt.datetime:
    """Publication time late in the trading day, so the lag must be a full day."""
    return dt.datetime.combine(START + dt.timedelta(days=day_offset), dt.time(20, 5), tzinfo=UTC)


def decision(day_offset: int) -> dt.date:
    return first_decision_date(published(day_offset))


def planted(day_offset: int, direction: str = "POSITIVE", confidence: float = 0.8):
    return make_event(
        article_id=f"provider:massive:planted-{day_offset}",
        published_at=published(day_offset),
        direction=direction,
        confidence=confidence,
    )


def end_date() -> dt.date:
    return decision(DAYS - 1)


def score_of(events, **policy_kwargs) -> dict[dt.date, float]:
    return daily_news_score(events, start=START, end=end_date(), **policy_kwargs)


class TestLagIsReal:
    def test_same_day_news_cannot_reach_the_same_day_score(self):
        # A feature that reacted to news published at 20:05 on day t would let a
        # decision taken at that day's open act on it.
        events = [planted(0)]
        scores = score_of(events)
        assert scores[START] == 0.0
        assert scores[decision(0)] == pytest_approx(0.8)

    def test_the_score_lands_on_exactly_one_day(self):
        events = [planted(5)]
        scores = score_of(events)
        touched = [day for day, value in scores.items() if value]
        assert touched == [decision(5)]
        assert decision(5) == published(5).date() + dt.timedelta(days=1)

    def test_a_lone_event_contributes_exactly_once(self):
        scores = score_of([planted(3, confidence=0.8)])
        assert [v for v in scores.values() if v] == [pytest_approx(0.8)]

    def test_opposing_events_cancel(self):
        scores = score_of([planted(0, "POSITIVE", 0.8), planted(0, "NEGATIVE", 0.8)])
        assert scores[decision(0)] == pytest_approx(0.0)

    def test_events_weighted_by_confidence(self):
        scores = score_of([planted(0, "POSITIVE", 0.2), planted(0, "POSITIVE", 0.6)])
        assert scores[decision(0)] == pytest_approx(0.8)

    def test_a_day_with_no_news_scores_zero_not_missing(self):
        # A missing value and a real zero are different facts, and only one of
        # them is safe to trade on.
        scores = score_of([planted(0), planted(9)])
        assert scores[START + dt.timedelta(days=4)] == 0.0
        assert len(scores) > DAYS

    def test_magnitude_is_recorded_but_not_scored(self):
        # Scaling by a magnitude the analyzer has never been calibrated on would
        # let an unvalidated number size a position.
        small = make_event(
            article_id="provider:massive:m1",
            published_at=published(0),
            direction="POSITIVE",
            confidence=0.5,
            magnitude=0.01,
        )
        large = make_event(
            article_id="provider:massive:m2",
            published_at=published(0),
            direction="POSITIVE",
            confidence=0.5,
            magnitude=0.99,
        )
        assert score_of([small])[decision(0)] == score_of([large])[decision(0)]

    def test_unknown_and_neutral_contribute_nothing(self):
        events = [planted(0, "UNKNOWN", 0.9), planted(0, "NEUTRAL", 0.9), planted(0, "MIXED", 0.9)]
        assert score_of(events)[decision(0)] == 0.0


class TestPrefixInvariance:
    def test_appending_future_news_cannot_change_past_scores(self):
        # The property is about *publication time*, not about which events happen
        # to be in the file: an event published on day T may move the score for
        # T+1, but must not move any score at or before T. Comparing whole
        # windows would wrongly flag that legitimate change.
        early = [planted(0), planted(1)]
        # Everything appended is published strictly after `cutoff`, so nothing
        # appended may legitimately move a score on or before that date.
        cutoff = published(1).date()
        later = early + [planted(i) for i in range(2, DAYS)]
        assert min(event.published_at for event in later[2:]).date() > cutoff
        before = score_of(early)
        after = score_of(later)
        checked = 0
        for day, value in before.items():
            if day <= cutoff:
                checked += 1
                assert after[day] == value, f"score changed on {day}"
        assert checked > 0

    def test_a_longer_history_agrees_with_a_shorter_one_over_the_shared_window(self):
        short = [planted(i) for i in range(5)]
        long = [planted(i) for i in range(DAYS)]
        short_scores = score_of(short)
        long_scores = score_of(long)
        cutoff = published(4).date()
        for day, value in short_scores.items():
            if day <= cutoff:
                assert long_scores[day] == value, f"score changed on {day}"

    def test_as_of_bounding_hides_only_the_future(self):
        dataset = build_dataset([planted(i) for i in range(DAYS)], name="planted")
        cutoff = dt.datetime.combine(decision(6), dt.time(), tzinfo=UTC)
        visible = events_available_as_of(dataset, as_of=cutoff)
        # Every event used by an as-of query must itself have been decided by
        # that date; anything later would be information from the future.
        for event in visible:
            assert first_decision_date(event.published_at) <= decision(6)
        assert 0 < len(visible) < len(dataset.events)

    def test_an_as_of_score_equals_the_full_history_score(self):
        # Bounding the history must not change the answer for the window it
        # covers -- that is the whole point of as-of querying.
        dataset = build_dataset([planted(i) for i in range(DAYS)], name="planted")
        as_of = dt.datetime.combine(decision(6), dt.time(), tzinfo=UTC)
        bounded = cumulative_news_score(
            events_available_as_of(dataset, as_of=as_of), start=START, end=decision(6)
        )
        full = cumulative_news_score(
            events_available_as_of(dataset, as_of=dt.datetime(2049, 1, 1, tzinfo=UTC)),
            start=START,
            end=decision(6),
        )
        assert bounded == full


class TestPlantedSignalIsRecoverable:
    """The precondition for building a strategy: the plumbing is not lossy."""

    def test_a_planted_daily_pattern_appears_at_exactly_the_right_lag(self):
        events = [planted(i, "POSITIVE" if i % 2 else "NEGATIVE") for i in range(DAYS)]
        scores = score_of(events)
        for offset in range(0, DAYS - 1):
            expected = 0.8 if offset % 2 else -0.8
            assert scores[decision(offset)] == pytest_approx(expected)

    def test_shifting_the_relationship_breaks_alignment(self):
        # If the lag were not doing work, shifting the planted pattern would not
        # matter. It does, which is the evidence that the one-day offset is
        # applied rather than coincidentally aligned on this fixture.
        aligned = [planted(i, "POSITIVE" if i % 2 else "NEGATIVE") for i in range(DAYS)]
        shifted = [planted(i, "POSITIVE" if (i + 1) % 2 else "NEGATIVE") for i in range(DAYS)]
        a, b = score_of(aligned), score_of(shifted)
        probe = decision(10)
        assert a[probe] == -b[probe]

    def test_the_score_is_confidence_weighted_end_to_end(self):
        events = [planted(i, "POSITIVE", 0.25) for i in range(DAYS)]
        scores = score_of(events)
        assert scores[decision(10)] == pytest_approx(0.25)


class TestPolicyIsHonoured:
    def test_untagged_events_are_dropped_when_required(self):
        event = FinancialEvent(
            event_id="evt:untagged",
            article_id="provider:massive:untagged",
            source="massive",
            tickers=(),
            event_type="EARNINGS",
            direction=Direction.POSITIVE,
            published_at=published(0),
            analyzed_at=published(0),
            analyzer="deterministic",
            analyzer_version="1.0.0",
            confidence=0.9,
        )
        assert score_of([event])[decision(0)] == 0.0

    def test_untagged_events_count_when_the_policy_allows_them(self):
        event = FinancialEvent(
            event_id="evt:untagged",
            article_id="provider:massive:untagged",
            source="massive",
            tickers=(),
            event_type="EARNINGS",
            direction=Direction.POSITIVE,
            published_at=published(0),
            analyzed_at=published(0),
            analyzer="deterministic",
            analyzer_version="1.0.0",
            confidence=0.9,
        )
        strict = score_of([event], policy=NewsFeaturePolicy(require_ticker=True))
        relaxed = score_of([event], policy=NewsFeaturePolicy(require_ticker=False))
        assert strict[decision(0)] == 0.0
        assert relaxed[decision(0)] == pytest_approx(0.9)

    def test_events_below_the_confidence_floor_are_dropped(self):
        events = [planted(0, "POSITIVE", 0.2)]
        assert score_of(events, policy=NewsFeaturePolicy(min_confidence=0.5))[decision(0)] == 0.0
        assert score_of(events, policy=NewsFeaturePolicy(min_confidence=0.1))[decision(0)] == 0.2

    def test_event_type_filtering_narrows_the_score(self):
        event = make_event(
            article_id="provider:massive:legal-1",
            published_at=published(0),
            event_type="LEGAL",
            direction="POSITIVE",
            confidence=0.9,
        )
        assert score_of([event], policy=NewsFeaturePolicy(event_types=()))[decision(0)] == 0.9
        assert (
            score_of(
                [event], policy=NewsFeaturePolicy(event_types=("EARNINGS",))
            )[decision(0)]
            == 0.0
        )


def pytest_approx(value: float):
    return pytest.approx(value, abs=1e-9)