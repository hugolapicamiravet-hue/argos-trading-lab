"""Temporal integrity: the tests that make a news backtest trustworthy.

Everything in this project exists to answer one question: *what did ARGOS know,
and when could it have known it?* These tests are the executable form of that
question. If any of them fails, a backtest built on this data is fiction.

The four properties enforced here:

  1. No event is usable before its own publication time.
  2. Availability is judged by `published_at`, never by when we analyzed it or
     downloaded it.
  3. Extending a dataset into the future must not change the past. (prefix
     invariance)
  4. A score projected onto a bar index cannot read a future value.
"""

from __future__ import annotations

import datetime as dt

import pytest
from tests.news.conftest import UTC, make_event

from argos.news import (
    Direction,
    NewsFeaturePolicy,
    align_score_to_index,
    build_dataset,
    cumulative_news_score,
    daily_news_score,
    events_available_as_of,
    first_decision_date,
)


def _event(day: int, hour: int = 12, **kwargs):
    return make_event(
        article_id=f"provider:massive:d{day}",
        published_at=dt.datetime(2025, 7, day, hour, tzinfo=UTC),
        **kwargs,
    )


class TestFirstDecisionDate:
    def test_lag_of_one_day(self):
        published = dt.datetime(2025, 7, 31, 20, 5, tzinfo=UTC)
        assert first_decision_date(published) == dt.date(2025, 8, 1)

    def test_lag_of_zero_is_same_day(self):
        published = dt.datetime(2025, 7, 31, 20, 5, tzinfo=UTC)
        assert first_decision_date(published, lag_days=0) == dt.date(2025, 7, 31)

    def test_decision_date_uses_utc_not_local_wall_clock(self):
        # 23:30Z is still the 31st in UTC. Converting to a US Eastern wall clock
        # first would call it the 31st at 19:30 local, but the reverse mistake is
        # the dangerous one: a timestamp late in the UTC day must not roll into
        # the next decision date early.
        late = dt.datetime(2025, 7, 31, 23, 30, tzinfo=UTC)
        assert first_decision_date(late) == dt.date(2025, 8, 1)

    def test_early_utc_article_decides_next_day(self):
        early = dt.datetime(2025, 7, 31, 0, 5, tzinfo=UTC)
        assert first_decision_date(early) == dt.date(2025, 8, 1)

    def test_non_utc_offset_is_normalized_first(self):
        # 20:05 at -04:00 is 00:05Z on 1 Aug, so it decides on the 2nd.
        offset = dt.datetime(2025, 7, 31, 20, 5, tzinfo=dt.timezone(dt.timedelta(hours=-4)))
        assert first_decision_date(offset) == dt.date(2025, 8, 2)

    def test_negative_lag_is_rejected(self):
        published = dt.datetime(2025, 7, 31, 20, 5, tzinfo=UTC)
        with pytest.raises(ValueError, match="lag_days must be >= 0"):
            first_decision_date(published, lag_days=-1)

    def test_naive_timestamp_is_rejected(self):
        with pytest.raises(ValueError, match="timezone-aware"):
            first_decision_date(dt.datetime(2025, 7, 31, 20, 5))


class TestEventsAvailableAsOf:
    def test_excludes_future_events(self):
        events = [_event(1), _event(5), _event(10)]
        dataset = build_dataset(events)
        available = events_available_as_of(dataset, dt.datetime(2025, 7, 5, 0, tzinfo=UTC))
        assert [e.article_id for e in available] == ["provider:massive:d1"]

    def test_boundary_is_inclusive(self):
        events = [_event(5, hour=9)]
        dataset = build_dataset(events)
        exact = dt.datetime(2025, 7, 5, 9, 0, tzinfo=UTC)
        assert len(events_available_as_of(dataset, exact)) == 1

    def test_availability_ignores_analyzed_at(self):
        # Analyzed in 2026, published in 2025. A 2025 query must see it, because
        # the *information* was public in 2025. A pipeline that keyed off
        # analyzed_at would wrongly hide it.
        event = _event(5, analyzed_at=dt.datetime(2026, 10, 1, tzinfo=UTC))
        dataset = build_dataset([event])
        assert len(events_available_as_of(dataset, dt.datetime(2025, 7, 6, tzinfo=UTC))) == 1

    def test_naive_as_of_is_rejected(self):
        dataset = build_dataset([_event(5)])
        with pytest.raises(ValueError, match="timezone-aware"):
            events_available_as_of(dataset, dt.datetime(2025, 7, 6))

    def test_as_of_before_everything_returns_empty(self):
        dataset = build_dataset([_event(5), _event(10)])
        assert events_available_as_of(dataset, dt.datetime(2025, 1, 1, tzinfo=UTC)) == ()

    def test_intraday_as_of_excludes_same_day_later_events(self):
        morning = _event(5, hour=9)
        evening = _event(5, hour=21, direction="NEGATIVE")
        dataset = build_dataset([morning, evening], name="x")
        # Make ids unique so dedupe-free build keeps both.
        dataset = build_dataset(
            [make_event(article_id="a", published_at=dt.datetime(2025, 7, 5, 9, tzinfo=UTC)),
             make_event(article_id="b", published_at=dt.datetime(2025, 7, 5, 21, tzinfo=UTC),
                        direction=Direction.NEGATIVE)]
        )
        cutoff = dt.datetime(2025, 7, 5, 12, tzinfo=UTC)
        assert [e.article_id for e in events_available_as_of(dataset, cutoff)] == ["a"]


class TestDailyNewsScore:
    def test_event_lands_on_its_decision_date(self):
        score = daily_news_score(
            [_event(31, confidence=0.5)], start=dt.date(2025, 7, 30), end=dt.date(2025, 8, 2)
        )
        assert score[dt.date(2025, 7, 31)] == 0.0
        assert score[dt.date(2025, 8, 1)] == pytest.approx(0.5)

    def test_every_date_in_window_is_present(self):
        score = daily_news_score([], start=dt.date(2025, 8, 1), end=dt.date(2025, 8, 5))
        assert list(score) == [dt.date(2025, 8, d) for d in range(1, 6)]
        assert all(v == 0.0 for v in score.values())

    def test_weighting_is_sign_times_confidence(self):
        events = [
            make_event(article_id="p", direction="POSITIVE", confidence=0.8),
            make_event(article_id="n", direction="NEGATIVE", confidence=0.3),
        ]
        score = daily_news_score(
            events, start=dt.date(2025, 8, 1), end=dt.date(2025, 8, 1)
        )
        assert score[dt.date(2025, 8, 1)] == pytest.approx(0.5)

    def test_magnitude_is_deliberately_excluded(self):
        low = make_event(article_id="low", magnitude=0.1, confidence=0.5)
        high = make_event(article_id="high", magnitude=1.0, confidence=0.5)
        a = daily_news_score([low], start=dt.date(2025, 8, 1), end=dt.date(2025, 8, 1))
        b = daily_news_score([high], start=dt.date(2025, 8, 1), end=dt.date(2025, 8, 1))
        assert a[dt.date(2025, 8, 1)] == b[dt.date(2025, 8, 1)]

    def test_unknown_direction_contributes_nothing(self):
        event = make_event(direction="UNKNOWN", confidence=0.9)
        score = daily_news_score(
            [event], start=dt.date(2025, 8, 1), end=dt.date(2025, 8, 1)
        )
        assert score[dt.date(2025, 8, 1)] == 0.0

    def test_mixed_direction_contributes_nothing(self):
        event = make_event(direction="MIXED", confidence=0.9)
        score = daily_news_score(
            [event], start=dt.date(2025, 8, 1), end=dt.date(2025, 8, 1)
        )
        assert score[dt.date(2025, 8, 1)] == 0.0

    def test_neutral_contributes_nothing(self):
        event = make_event(direction="NEUTRAL", confidence=0.9)
        score = daily_news_score(
            [event], start=dt.date(2025, 8, 1), end=dt.date(2025, 8, 1)
        )
        assert score[dt.date(2025, 8, 1)] == 0.0

    def test_missing_confidence_uses_full_sign(self):
        # Documented behaviour: an uncalibrated analyzer is weighted at full
        # sign. The audit flags this rather than hiding it.
        event = make_event(direction="POSITIVE", confidence=None)
        score = daily_news_score(
            [event], start=dt.date(2025, 8, 1), end=dt.date(2025, 8, 1)
        )
        assert score[dt.date(2025, 8, 1)] == pytest.approx(1.0)

    def test_untagged_events_are_excluded_by_default(self):
        event = make_event(tickers=(), direction="POSITIVE", confidence=0.5)
        score = daily_news_score(
            [event], start=dt.date(2025, 8, 1), end=dt.date(2025, 8, 1)
        )
        assert score[dt.date(2025, 8, 1)] == 0.0

    def test_untagged_events_included_when_policy_allows(self):
        event = make_event(tickers=(), direction="POSITIVE", confidence=0.5)
        score = daily_news_score(
            [event],
            start=dt.date(2025, 8, 1),
            end=dt.date(2025, 8, 1),
            policy=NewsFeaturePolicy(require_ticker=False),
        )
        assert score[dt.date(2025, 8, 1)] == pytest.approx(0.5)

    def test_events_outside_window_are_not_included(self):
        event = _event(1)
        score = daily_news_score(
            [event], start=dt.date(2025, 9, 1), end=dt.date(2025, 9, 5)
        )
        assert all(v == 0.0 for v in score.values())

    def test_reversed_window_is_rejected(self):
        with pytest.raises(ValueError, match="must not precede"):
            daily_news_score([], start=dt.date(2025, 9, 5), end=dt.date(2025, 9, 1))

    def test_multiple_events_same_day_accumulate(self):
        events = [
            make_event(article_id="a", direction="POSITIVE", confidence=0.2),
            make_event(article_id="b", direction="POSITIVE", confidence=0.3),
            make_event(article_id="c", direction="NEGATIVE", confidence=0.1),
        ]
        score = daily_news_score(
            events, start=dt.date(2025, 8, 1), end=dt.date(2025, 8, 1)
        )
        assert score[dt.date(2025, 8, 1)] == pytest.approx(0.4)

    def test_zero_lag_policy_uses_same_day(self):
        event = _event(31)
        score = daily_news_score(
            [event],
            start=dt.date(2025, 7, 31),
            end=dt.date(2025, 7, 31),
            policy=NewsFeaturePolicy(publication_lag_days=0),
        )
        assert score[dt.date(2025, 7, 31)] == pytest.approx(0.6)

    def test_two_day_lag_delays_decision(self):
        event = _event(31)
        score = daily_news_score(
            [event],
            start=dt.date(2025, 7, 31),
            end=dt.date(2025, 8, 3),
            policy=NewsFeaturePolicy(publication_lag_days=2),
        )
        assert score[dt.date(2025, 8, 1)] == 0.0
        assert score[dt.date(2025, 8, 2)] == pytest.approx(0.6)


class TestPrefixInvariance:
    """Extending a dataset must not change what the past looked like.

    This is the property that catches an entire class of subtle pipeline bugs:
    if computing a feature for January requires February's data, then the
    feature was never really a January feature.
    """

    def test_future_events_do_not_change_past_scores(self):
        early = [_event(1, direction="POSITIVE", confidence=0.5)]
        early_and_late = early + [_event(20, direction="NEGATIVE", confidence=0.9)]

        short = daily_news_score(early, start=dt.date(2025, 7, 1), end=dt.date(2025, 7, 10))
        long = daily_news_score(
            early_and_late, start=dt.date(2025, 7, 1), end=dt.date(2025, 7, 31)
        )
        for day in short:
            assert long[day] == short[day], f"score for {day} changed when future data was added"

    def test_input_order_does_not_change_scores(self):
        events = [_event(d, direction="POSITIVE", confidence=0.1 * d) for d in range(1, 8)]
        forward = daily_news_score(
            events, start=dt.date(2025, 7, 1), end=dt.date(2025, 7, 31)
        )
        backward = daily_news_score(
            list(reversed(events)), start=dt.date(2025, 7, 1), end=dt.date(2025, 7, 31)
        )
        assert forward == backward

    def test_cumulative_score_is_monotonic_prefix_consistent(self):
        events = [_event(1, direction="POSITIVE", confidence=0.5)]
        with_more = events + [_event(15, direction="POSITIVE", confidence=0.5)]
        short = cumulative_news_score(
            events, start=dt.date(2025, 7, 1), end=dt.date(2025, 7, 10)
        )
        long = cumulative_news_score(
            with_more, start=dt.date(2025, 7, 1), end=dt.date(2025, 7, 31)
        )
        for day in short:
            assert long[day] == short[day]

    def test_cumulative_accumulates(self):
        events = [
            make_event(article_id="a", direction="POSITIVE", confidence=0.5),
            make_event(article_id="b", direction="NEGATIVE", confidence=0.2),
        ]
        cumulative = cumulative_news_score(
            events, start=dt.date(2025, 8, 1), end=dt.date(2025, 8, 1)
        )
        assert cumulative[dt.date(2025, 8, 1)] == pytest.approx(0.3)


class TestAlignScoreToIndex:
    INDEX = [
        dt.datetime(2025, 7, 30, tzinfo=UTC),
        dt.datetime(2025, 7, 31, tzinfo=UTC),
        dt.datetime(2025, 8, 1, tzinfo=UTC),
        dt.datetime(2025, 8, 4, tzinfo=UTC),
    ]

    def test_carries_last_value_across_a_weekend(self):
        score = {dt.date(2025, 8, 1): 0.5}
        aligned = align_score_to_index(score, self.INDEX)
        # Friday's score carries into Monday; it does not vanish over the weekend.
        assert aligned == [0.0, 0.0, 0.5, 0.5]

    def test_never_reads_a_future_value(self):
        score = {dt.date(2025, 8, 1): 0.5}
        aligned = align_score_to_index(score, self.INDEX)
        assert aligned[0] == 0.0
        assert aligned[1] == 0.0

    def test_empty_score_yields_zeros(self):
        assert align_score_to_index({}, self.INDEX) == [0.0, 0.0, 0.0, 0.0]

    def test_accepts_date_objects_and_iso_strings(self):
        score = {dt.date(2025, 8, 1): 0.5}
        by_date = align_score_to_index(score, [dt.date(2025, 8, 1)])
        by_string = align_score_to_index(score, ["2025-08-01"])
        assert by_date == by_string == [0.5]

    def test_unsorted_index_dates_are_handled(self):
        score = {dt.date(2025, 8, 1): 0.5, dt.date(2025, 8, 2): 0.9}
        aligned = align_score_to_index(score, ["2025-08-03", "2025-08-01"])
        # Each bar takes the last score at or before it, independent of order.
        assert aligned == [0.9, 0.5]

    def test_unparseable_date_is_rejected(self):
        with pytest.raises(ValueError, match="cannot interpret"):
            align_score_to_index({}, ["not-a-date"])


class TestNoSameBarLeakage:
    """The headline invariant: news cannot influence its own publication bar.

    ARGOS executes a signal at the next bar's open. A feature built from an
    article published during bar t must therefore not be visible in bar t's
    signal at all.
    """

    def test_publication_day_score_is_always_zero(self):
        for day in (1, 5, 15, 28):
            for hour in (0, 6, 12, 18, 23):
                event = make_event(
                    article_id=f"p{day}h{hour}",
                    published_at=dt.datetime(2025, 7, day, hour, tzinfo=UTC),
                    direction="POSITIVE",
                    confidence=0.9,
                )
                score = daily_news_score(
                    [event],
                    start=dt.date(2025, 7, day),
                    end=dt.date(2025, 7, day),
                )
                assert score[dt.date(2025, 7, day)] == 0.0, (
                    f"event published {day} {hour}:00 influenced its own bar"
                )

    def test_first_influence_is_strictly_after_publication(self):
        event = _event(15, hour=23, direction="POSITIVE", confidence=0.9)
        decision = first_decision_date(event.published_at)
        assert decision > event.published_at.astimezone(UTC).date()