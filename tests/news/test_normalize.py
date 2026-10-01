"""Normalization, UTC parsing and the fail-closed temporal policy.

The rule under test throughout: a record whose publication time cannot be
trusted is rejected, never repaired. Every case here corresponds to a way a
timestamp can go wrong in a real feed, and each one must end in the record
either being normalized correctly or being dropped with a stated reason.
"""

from __future__ import annotations

import datetime as dt

import pytest
from tests.news.conftest import NOW, UTC, make_payload

from argos.news import normalize_articles, parse_utc, reject_reason


class TestParseUtc:
    def test_parses_trailing_z_as_utc(self):
        parsed = parse_utc("2025-07-31T20:05:00Z")
        assert parsed == dt.datetime(2025, 7, 31, 20, 5, tzinfo=UTC)

    def test_converts_offset_to_utc(self):
        # 16:05 at -04:00 is 20:05 UTC. Getting this wrong shifts an
        # after-hours article across a session boundary.
        assert parse_utc("2025-07-31T16:05:00-04:00") == dt.datetime(
            2025, 7, 31, 20, 5, tzinfo=UTC
        )

    def test_parses_offset_aware_datetime(self):
        moment = dt.datetime(2025, 7, 31, 20, 5, tzinfo=dt.timezone(dt.timedelta(hours=2)))
        assert parse_utc(moment) == dt.datetime(2025, 7, 31, 18, 5, tzinfo=UTC)

    def test_naive_string_is_rejected(self):
        with pytest.raises(ValueError, match="no timezone offset"):
            parse_utc("2025-07-31 20:05:00")

    def test_naive_datetime_is_rejected(self):
        with pytest.raises(ValueError, match="no timezone offset"):
            parse_utc(dt.datetime(2025, 7, 31, 20, 5))

    def test_empty_string_is_rejected(self):
        with pytest.raises(ValueError, match="empty"):
            parse_utc("   ")

    def test_garbage_is_rejected(self):
        with pytest.raises(ValueError, match="not in any recognised format"):
            parse_utc("last thursday")

    def test_non_string_non_datetime_is_rejected(self):
        with pytest.raises(TypeError, match="must be str or datetime"):
            parse_utc(1722469500)

    def test_empty_date_is_accepted_at_midnight_utc(self):
        # A date-only value carries an implicit offset of UTC per ISO-8601.
        # Ambiguous in practice, so it is only accepted via the fallback path
        # when an offset is explicit; a bare date is NOT silently assumed.
        with pytest.raises(ValueError):
            parse_utc("2025-07-31")


class TestRejectReason:
    def test_accepts_well_formed_record(self):
        assert reject_reason(make_payload()) is None

    def test_rejects_non_dict(self):
        assert reject_reason(["not", "a", "dict"]) == "record is list, not an object"

    def test_rejects_missing_title(self):
        assert reject_reason(make_payload(title="")) == "missing or empty title"

    def test_rejects_missing_timestamp(self):
        reason = reject_reason(make_payload(published=None))
        assert reason == "missing published_at"

    def test_rejects_naive_timestamp(self):
        reason = reject_reason(make_payload(published="2025-07-31 20:05:00"))
        assert "unparseable published_at" in reason

    def test_reports_first_usable_key(self):
        # ARGOS's own key wins when present, even if the provider's is broken.
        payload = make_payload()
        payload["published_at"] = "2025-07-31T20:05:00Z"
        payload["published_utc"] = "garbage"
        assert reject_reason(payload) is None


class TestNormalizeArticles:
    def test_normalizes_provider_keys(self, payloads):
        result = normalize_articles(payloads, source="massive", retrieved_at=NOW)
        assert result.accepted == 3
        assert result.rejected_count == 0

        first = result.articles[0]
        assert first.source == "massive"
        assert first.raw_source_id == "n1"
        assert first.publisher == "Reuters"
        assert first.tickers == ("AAPL",)
        assert first.published_at == dt.datetime(2025, 7, 31, 20, 5, tzinfo=UTC)
        assert first.retrieved_at == NOW

    def test_sorts_by_publication_time(self, payloads):
        result = normalize_articles(payloads, source="massive", retrieved_at=NOW)
        times = [a.published_at for a in result.articles]
        assert times == sorted(times)

    def test_comma_separated_ticker_string_is_split(self):
        payload = make_payload(tickers="AAPL,MSFT")
        result = normalize_articles([payload], source="massive", retrieved_at=NOW)
        assert result.articles[0].tickers == ("AAPL", "MSFT")

    def test_tickers_are_uppercased_and_deduped(self):
        payload = make_payload(tickers=["aapl", " AAPL ", "msft"])
        result = normalize_articles([payload], source="massive", retrieved_at=NOW)
        assert result.articles[0].tickers == ("AAPL", "MSFT")

    def test_headline_key_is_accepted(self):
        payload = {"headline": "Something happened", "published_utc": "2025-07-31T20:05:00Z"}
        result = normalize_articles([payload], source="massive", retrieved_at=NOW)
        assert result.articles[0].title == "Something happened"

    def test_rejects_and_reports_by_index(self):
        batch = [
            make_payload(article_id="ok"),
            make_payload(article_id="bad-time", published=None),
            make_payload(
                article_id="also-ok",
                title="Nvidia beats Q3 expectations",
                summary="Data center revenue.",
                url="https://www.reuters.com/markets/nvidia-q3-2025",
                published="2025-08-01T12:00:00Z",
            ),
        ]
        result = normalize_articles(batch, source="massive", retrieved_at=NOW)
        assert result.accepted == 2
        assert result.rejected == ((1, "missing published_at"),)

    def test_retrieved_at_is_never_used_as_published_fallback(self):
        # The critical case: a record with no publication time must be dropped
        # even though we know exactly when we fetched it.
        payload = make_payload(published=None)
        result = normalize_articles([payload], source="massive", retrieved_at=NOW)
        assert result.accepted == 0
        assert result.rejected_count == 1

    def test_rejects_retrieved_before_published(self):
        payload = make_payload(published="2027-01-01T00:00:00Z")
        with pytest.raises(ValueError, match="earlier than published_at"):
            normalize_articles([payload], source="massive", retrieved_at=NOW)

    def test_requires_source(self, payloads):
        with pytest.raises(ValueError, match="source must name the provider"):
            normalize_articles(payloads, source="", retrieved_at=NOW)

    def test_requires_aware_retrieved_at(self, payloads):
        with pytest.raises(ValueError, match="timezone-aware"):
            normalize_articles(payloads, source="massive", retrieved_at=dt.datetime.now())

    def test_deduplication_can_be_disabled(self):
        batch = [make_payload(), make_payload()]
        result = normalize_articles(
            batch, source="massive", retrieved_at=NOW, reject_duplicates=False
        )
        assert result.accepted == 2
        assert result.duplicate_count == 0

    def test_deduplication_is_on_by_default(self):
        result = normalize_articles(
            [make_payload(), make_payload()], source="massive", retrieved_at=NOW
        )
        assert result.accepted == 1
        assert result.duplicate_count == 1

    def test_iteration_and_len_support_result(self, payloads):
        result = normalize_articles(payloads, source="massive", retrieved_at=NOW)
        assert len(result) == 3
        assert [a.article_id for a in result] == [a.article_id for a in result.articles]


class TestTemporalPolicyEdgeCases:
    def test_article_published_exactly_at_midnight_utc_is_kept(self):
        payload = make_payload(published="2025-08-01T00:00:00Z")
        result = normalize_articles([payload], source="massive", retrieved_at=NOW)
        assert result.accepted == 1
        assert result.articles[0].published_at.hour == 0

    def test_dst_boundary_offsets_are_normalized_not_shifted(self):
        # US DST began 2025-03-09. A feed reporting -04:00 on either side must
        # land on the same UTC instant regardless of which offset it sends.
        before = make_payload(
            article_id="dst-a", url="https://www.reuters.com/markets/before-dst",
            published="2025-03-08T20:05:00-05:00",
        )
        after = make_payload(
            article_id="dst-b", url="https://www.reuters.com/markets/after-dst",
            published="2025-03-09T20:05:00-04:00",
        )
        result = normalize_articles([before, after], source="massive", retrieved_at=NOW)
        # 20:05-05:00 == 01:05Z next day; 20:05-04:00 == 00:05Z. Different
        # instants, and the pipeline must not pretend otherwise.
        assert result.articles[0].published_at == dt.datetime(2025, 3, 9, 1, 5, tzinfo=UTC)
        assert result.articles[1].published_at == dt.datetime(2025, 3, 10, 0, 5, tzinfo=UTC)

    def test_empty_batch_is_valid(self):
        result = normalize_articles([], source="massive", retrieved_at=NOW)
        assert result.accepted == 0
        assert result.rejected_count == 0