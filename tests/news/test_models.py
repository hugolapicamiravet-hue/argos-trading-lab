"""Model invariants: enums, bounds, provenance, and the fact/inference split."""

from __future__ import annotations

import datetime as dt

import pytest
from tests.news.conftest import NOW, UTC, make_article, make_event

from argos.news import Direction, EventType, FinancialEvent, NewsArticle
from argos.news.models import ARTICLE_SCHEMA_VERSION, EVENT_SCHEMA_VERSION


class TestNewsArticleValidation:
    def test_requires_timezone_aware_published_at(self):
        with pytest.raises(ValueError, match="timezone-aware"):
            make_article(published_at=dt.datetime(2025, 7, 31, 20, 5))

    def test_requires_timezone_aware_retrieved_at(self):
        with pytest.raises(ValueError, match="timezone-aware"):
            make_article(retrieved_at=dt.datetime(2026, 1, 1))

    def test_rejects_retrieval_before_publication(self):
        with pytest.raises(ValueError, match="earlier than published_at"):
            make_article(
                published_at=dt.datetime(2025, 7, 31, tzinfo=UTC),
                retrieved_at=dt.datetime(2025, 1, 1, tzinfo=UTC),
            )

    def test_rejects_empty_title(self):
        with pytest.raises(ValueError, match="title must not be empty"):
            make_article(title="   ")

    def test_rejects_empty_article_id(self):
        # Built directly rather than via the fixture: the fixture treats "" as
        # "derive one", which would quietly defeat the assertion.
        with pytest.raises(ValueError, match="article_id must not be empty"):
            NewsArticle(
                article_id="",
                source="massive",
                title="Story",
                published_at=dt.datetime(2025, 7, 31, 20, 5, tzinfo=UTC),
                retrieved_at=dt.datetime(2026, 1, 1, tzinfo=UTC),
            )

    def test_requires_source(self):
        with pytest.raises(ValueError, match="source must name the provider"):
            NewsArticle(
                article_id="provider:x:1",
                source="",
                title="Story",
                published_at=dt.datetime(2025, 7, 31, 20, 5, tzinfo=UTC),
                retrieved_at=dt.datetime(2026, 1, 1, tzinfo=UTC),
            )

    def test_normalizes_tickers(self):
        article = make_article(tickers=("aapl", "aapl", " msft ", ""))
        assert article.tickers == ("AAPL", "MSFT")

    def test_offset_is_normalized_to_utc(self):
        article = make_article(
            published_at=dt.datetime(
                2025, 7, 31, 20, 5, tzinfo=dt.timezone(dt.timedelta(hours=-4))
            ),
            retrieved_at=NOW,
        )
        assert article.published_at.tzinfo is UTC
        assert article.published_at == dt.datetime(2025, 8, 1, 0, 5, tzinfo=UTC)

    def test_fingerprint_is_stable_across_calls(self):
        article = make_article()
        assert article.fingerprint == article.fingerprint
        assert len(article.fingerprint) == 64

    def test_fingerprint_ignores_casing_and_whitespace(self):
        a = make_article(title="Apple Beats Earnings", summary="Record   revenue")
        b = make_article(title="apple beats earnings", summary="Record revenue")
        assert a.fingerprint == b.fingerprint

    def test_fingerprint_changes_with_content(self):
        a = make_article(title="Apple beats earnings")
        b = make_article(title="Apple misses earnings")
        assert a.fingerprint != b.fingerprint

    def test_fingerprint_changes_with_publication_time(self):
        a = make_article(published_at=dt.datetime(2025, 7, 31, 20, 5, tzinfo=UTC))
        b = make_article(published_at=dt.datetime(2025, 7, 31, 20, 6, tzinfo=UTC))
        assert a.fingerprint != b.fingerprint

    def test_has_ticker(self):
        article = make_article(tickers=("AAPL",))
        assert article.has_ticker("aapl")
        assert article.has_ticker("AAPL")
        assert not article.has_ticker("MSFT")


class TestArticleSerialization:
    def test_round_trip_preserves_everything(self):
        article = make_article()
        assert NewsArticle.from_dict(article.to_dict()) == article

    def test_round_trip_with_minimal_fields(self):
        article = NewsArticle(
            article_id="provider:x:1",
            source="massive",
            title="Bare minimum",
            published_at=dt.datetime(2025, 1, 1, tzinfo=UTC),
            retrieved_at=dt.datetime(2025, 1, 2, tzinfo=UTC),
        )
        assert NewsArticle.from_dict(article.to_dict()) == article

    def test_serialized_timestamps_use_z_suffix(self):
        payload = make_article().to_dict()
        assert payload["published_at"].endswith("Z")
        assert payload["retrieved_at"].endswith("Z")

    def test_unknown_schema_version_is_rejected(self):
        payload = make_article().to_dict()
        payload["schema_version"] = ARTICLE_SCHEMA_VERSION + 99
        with pytest.raises(ValueError, match="schema version"):
            NewsArticle.from_dict(payload)

    def test_to_dict_is_json_serializable(self):
        import json

        json.dumps(make_article().to_dict())


class TestEnumCoercion:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("positive", Direction.POSITIVE),
            ("  NEGATIVE  ", Direction.NEGATIVE),
            ("bullish", Direction.UNKNOWN),
            ("", Direction.UNKNOWN),
            (None, Direction.UNKNOWN),
            (42, Direction.UNKNOWN),
        ],
    )
    def test_direction_coercion_is_fail_closed(self, raw, expected):
        # An unrecognized direction must become UNKNOWN, never a guess. Guessing
        # "bullish" as POSITIVE is exactly the kind of silent interpretation
        # this project refuses to do.
        assert Direction.coerce(raw) is expected

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("earnings", EventType.EARNINGS),
            ("MERGER_ACQUISITION", EventType.MERGER_ACQUISITION),
            ("merger acquisition", EventType.MERGER_ACQUISITION),
            ("merger-acquisition", EventType.MERGER_ACQUISITION),
            ("something-new", EventType.OTHER),
            (None, EventType.OTHER),
        ],
    )
    def test_event_type_coercion(self, raw, expected):
        assert EventType.coerce(raw) is expected

    def test_enum_values_are_plain_strings(self):
        assert Direction.POSITIVE == "POSITIVE"
        assert EventType.EARNINGS == "EARNINGS"

    def test_sign_mapping(self):
        assert Direction.POSITIVE.sign == 1.0
        assert Direction.NEGATIVE.sign == -1.0
        assert Direction.NEUTRAL.sign == 0.0
        assert Direction.MIXED.sign == 0.0
        assert Direction.UNKNOWN.sign == 0.0


class TestFinancialEvent:
    def test_requires_article_reference(self):
        with pytest.raises(ValueError, match="must name the source article"):
            FinancialEvent(
                event_id="evt:x",
                article_id="",
                source="massive",
                tickers=("AAPL",),
                event_type=EventType.EARNINGS,
                direction=Direction.POSITIVE,
                published_at=dt.datetime(2025, 7, 31, 20, 5, tzinfo=UTC),
                analyzed_at=dt.datetime(2026, 1, 1, tzinfo=UTC),
                analyzer="deterministic",
                analyzer_version="1.0.0",
            )

    def test_requires_analyzer_identity(self):
        with pytest.raises(ValueError, match="analyzer and analyzer_version"):
            FinancialEvent(
                event_id="evt:x",
                article_id="provider:x:1",
                source="massive",
                tickers=("AAPL",),
                event_type=EventType.EARNINGS,
                direction=Direction.POSITIVE,
                published_at=dt.datetime(2025, 7, 31, 20, 5, tzinfo=UTC),
                analyzed_at=dt.datetime(2026, 1, 1, tzinfo=UTC),
                analyzer="",
                analyzer_version="",
            )

    @pytest.mark.parametrize("field", ["magnitude", "confidence"])
    @pytest.mark.parametrize("value", [-0.1, 1.1, float("nan")])
    def test_rejects_out_of_range_or_nan(self, field, value):
        with pytest.raises(ValueError):
            make_event(**{field: value})

    @pytest.mark.parametrize("field", ["magnitude", "confidence"])
    def test_accepts_boundary_values(self, field):
        assert make_event(**{field: 0.0}) is not None
        assert make_event(**{field: 1.0}) is not None

    def test_known_at_is_publication_time_not_analysis_time(self):
        event = make_event(analyzed_at=dt.datetime(2030, 1, 1, tzinfo=UTC))
        assert event.known_at == event.published_at
        assert event.known_at != event.analyzed_at

    def test_contribution_is_sign_times_confidence(self):
        positive = make_event(direction="POSITIVE", confidence=0.4)
        negative = make_event(direction="NEGATIVE", confidence=0.4)
        assert positive.contribution() == pytest.approx(0.4)
        assert negative.contribution() == pytest.approx(-0.4)

    def test_contribution_without_confidence_is_full_sign(self):
        assert make_event(direction="POSITIVE", confidence=None).contribution() == 1.0

    def test_with_article_reasserts_publication_time(self):
        # Even a buggy analyzer cannot produce an event claiming to be known
        # earlier than its source article.
        article = make_article(published_at=dt.datetime(2025, 7, 31, 20, 5, tzinfo=UTC))
        event = make_event(published_at=dt.datetime(2020, 1, 1, tzinfo=UTC))
        corrected = event.with_article(article)
        assert corrected.published_at == article.published_at
        assert corrected.tickers == article.tickers

    def test_round_trip(self):
        event = make_event()
        assert FinancialEvent.from_dict(event.to_dict()) == event

    def test_unknown_schema_version_is_rejected(self):
        payload = make_event().to_dict()
        payload["schema_version"] = EVENT_SCHEMA_VERSION + 99
        with pytest.raises(ValueError, match="schema version"):
            FinancialEvent.from_dict(payload)

    def test_enum_fields_are_coerced_on_construction(self):
        event = make_event(direction="not-a-direction", event_type="not-a-type")
        assert event.direction is Direction.UNKNOWN
        assert event.event_type is EventType.OTHER


class TestFactInferenceSeparation:
    """The layering guarantee, asserted directly.

    A `NewsArticle` must not carry an opinion, and a `FinancialEvent` must not
    claim to be a provider fact. These are structural properties, so the test is
    structural too: it checks the field names, not just runtime behaviour.
    """

    ARTICLE_FIELDS = {
        "article_id", "source", "title", "published_at", "retrieved_at",
        "publisher", "author", "summary", "url", "raw_source_id", "tickers",
        "schema_version",
    }
    INFERENCE_FIELDS = {
        "event_type", "direction", "magnitude", "confidence", "rationale",
    }

    def test_article_carries_no_inference_fields(self):
        import dataclasses

        names = {f.name for f in dataclasses.fields(NewsArticle)}
        assert not (names & self.INFERENCE_FIELDS)

    def test_event_carries_provenance_fields(self):
        import dataclasses

        names = {f.name for f in dataclasses.fields(FinancialEvent)}
        required = {
            "article_id", "analyzer", "analyzer_version",
            "model", "prompt_ref", "analyzed_at",
        }
        assert required <= names

    def test_event_carries_no_trading_instructions(self):
        # The field set is the enforcement mechanism for "news cannot trade".
        # Adding a size, order or stop field here would be the violation.
        import dataclasses

        names = {f.name for f in dataclasses.fields(FinancialEvent)}
        forbidden = {"size", "quantity", "shares", "order", "stop_loss", "take_profit",
                     "position", "weight", "notional", "side"}
        assert not (names & forbidden)

    def test_article_has_exactly_the_documented_field_set(self):
        import dataclasses

        names = {f.name for f in dataclasses.fields(NewsArticle)}
        assert names == self.ARTICLE_FIELDS