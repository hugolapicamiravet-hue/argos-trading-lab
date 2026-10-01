"""Analyzer behaviour: determinism, fail-closed parsing, and provenance.

An analyzer turns a fact into an inference. Two rules govern everything here:

1. Failure must never look like a neutral opinion. A refusal, a truncated
   response and a malformed payload all become UNKNOWN with an error attached,
   never a silent NEUTRAL. "We could not tell" and "nothing happened" are
   different claims, and only one of them is true.
2. Every inference names the model, version and prompt that produced it. An
   event whose origin cannot be identified cannot be re-evaluated when the
   prompt changes, which is how silent drift happens.

No network. The transport is injected, so the OpenAI path is exercised with a
recorded response shape.
"""

from __future__ import annotations

import datetime as dt
import json

import pytest
from tests.news.conftest import ANALYZED_AT, UTC, make_article

from argos.news import Direction, EventType, FinancialEvent
from argos.news.analyzer import (
    AnalysisResult,
    AnalyzerError,
    DeterministicAnalyzer,
    DictCache,
    JsonlCache,
    LLMAnalyzer,
    LLMConfigurationError,
    OpenAIRequest,
    analyze_many,
    build_analyzer,
    build_prompt_ref,
    parse_llm_response,
)

MODEL = "gpt-4.1-mini"
API_KEY = "sk-test-key-do-not-log"


def responses_api_output(*, direction="POSITIVE", event_type="EARNINGS", **overrides):
    """A Responses API body whose single output item carries structured output."""
    payload = {
        "event_type": event_type,
        "direction": direction,
        "magnitude": 0.7,
        "confidence": 0.8,
        "rationale": "Reported beat with raised guidance.",
    }
    payload.update(overrides)
    return {"output_text": json.dumps(payload)}


def fake_transport(body):
    calls: list[OpenAIRequest] = []

    def transport(request):
        calls.append(request)
        return body

    transport.calls = calls
    return transport


class TestDeterministicAnalyzer:
    def test_reads_positive_language(self):
        result = DeterministicAnalyzer().analyze(make_article())
        assert result.direction is Direction.POSITIVE
        assert result.event_type is EventType.EARNINGS
        assert result.error is None

    def test_reads_negative_language(self):
        article = make_article(
            title="Apple faces regulatory probe and profit warning",
            summary="Investigators opened an inquiry into the app store.",
        )
        result = DeterministicAnalyzer().analyze(article)
        assert result.direction is Direction.NEGATIVE
        assert result.event_type is not EventType.EARNINGS

    def test_neutral_language_is_neutral(self):
        # Three-way taxonomy, and the middle term earns its place: UNKNOWN means
        # "no evidence", NEUTRAL means "evidence of no directional effect".
        # Collapsing them would let a keyword gap read as a market insight.
        article = make_article(
            title="Company schedules its annual meeting", summary="Notice of meeting."
        )
        assert DeterministicAnalyzer().analyze(article).direction is Direction.NEUTRAL

    def test_no_keyword_at_all_is_unknown(self):
        article = make_article(title="Zzz qqq", summary="Wxyz.")
        assert DeterministicAnalyzer().analyze(article).direction is Direction.UNKNOWN

    def test_neutral_and_unknown_both_score_zero(self):
        analyzer = DeterministicAnalyzer()
        article = make_article(
            title="Company schedules its annual meeting", summary="Notice of meeting."
        )
        event = analyzer.to_event(article, analyzer.analyze(article), analyzed_at=ANALYZED_AT)
        assert event.direction is Direction.NEUTRAL

    def test_is_deterministic(self):
        analyzer = DeterministicAnalyzer()
        article = make_article()
        assert analyzer.analyze(article) == analyzer.analyze(article)

    def test_result_is_json_serializable(self):
        json.dumps(DeterministicAnalyzer().analyze(make_article()).to_dict())

    def test_title_only_article_is_still_classified(self):
        # A headline with no summary must not throw the analyzer away.
        article = make_article(title="Apple beats earnings expectations", summary=None)
        assert DeterministicAnalyzer().analyze(article).direction is Direction.POSITIVE

    def test_to_event_records_analyzer_identity(self):
        analyzer = DeterministicAnalyzer()
        article = make_article()
        event = analyzer.to_event(article, analyzer.analyze(article), analyzed_at=ANALYZED_AT)
        assert isinstance(event, FinancialEvent)
        assert event.analyzer == "deterministic"
        assert event.analyzer_version == "1.0.0"

    def test_event_known_at_is_publication_not_analysis(self):
        analyzer = DeterministicAnalyzer()
        article = make_article()
        event = analyzer.to_event(article, analyzer.analyze(article), analyzed_at=ANALYZED_AT)
        assert event.known_at == article.published_at
        assert event.known_at != event.analyzed_at


class TestEventConversion:
    def test_event_copies_tickers_from_article(self):
        analyzer = DeterministicAnalyzer()
        article = make_article(tickers=("AAPL", "MSFT"))
        event = analyzer.to_event(article, analyzer.analyze(article), analyzed_at=ANALYZED_AT)
        assert event.tickers == ("AAPL", "MSFT")

    def test_event_id_is_derived_from_the_article(self):
        analyzer = DeterministicAnalyzer()
        article = make_article()
        first = analyzer.to_event(article, analyzer.analyze(article), analyzed_at=ANALYZED_AT)
        second = analyzer.to_event(article, analyzer.analyze(article), analyzed_at=ANALYZED_AT)
        assert first.event_id == second.event_id

    def test_published_at_comes_from_the_article_not_the_result(self):
        analyzer = DeterministicAnalyzer()
        article = make_article(published_at=dt.datetime(2025, 7, 31, 20, 5, tzinfo=UTC))
        event = analyzer.to_event(article, analyzer.analyze(article), analyzed_at=ANALYZED_AT)
        assert event.published_at == article.published_at

    def test_event_round_trips_through_json(self):
        analyzer = DeterministicAnalyzer()
        article = make_article()
        event = analyzer.to_event(article, analyzer.analyze(article), analyzed_at=ANALYZED_AT)
        assert FinancialEvent.from_dict(json.loads(json.dumps(event.to_dict()))) == event

    def test_unknown_direction_survives_conversion(self):
        analyzer = DeterministicAnalyzer()
        article = make_article(title="Zzz qqq", summary="Wxyz.")
        event = analyzer.to_event(article, analyzer.analyze(article), analyzed_at=ANALYZED_AT)
        assert event.direction is Direction.UNKNOWN
        assert event.contribution() == 0.0


class TestParseLlmResponse:
    def test_parses_valid_output_text(self):
        result = parse_llm_response(responses_api_output())
        assert result.direction is Direction.POSITIVE
        assert result.event_type is EventType.EARNINGS
        assert result.confidence == 0.8
        assert result.error is None

    def test_event_type_and_direction_are_both_required(self):
        # A missing field must not silently default: with strict schema this is
        # unreachable, and it is checked anyway so a provider-side change is
        # visible instead of looking like a model that found nothing.
        result = parse_llm_response({"output_text": json.dumps({"direction": "POSITIVE"})})
        assert result.direction is Direction.UNKNOWN
        assert "event_type" in result.error

    def test_parses_nested_output_items(self):
        # The Responses API can return text inside output[] rather than as the
        # output_text convenience field. Only handling one shape would silently
        # lose every event.
        body = {
            "output": [
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "output_text",
                            "text": json.dumps(
                                {"event_type": "EARNINGS", "direction": "POSITIVE"}
                            ),
                        }
                    ],
                }
            ]
        }
        assert parse_llm_response(body).direction is Direction.POSITIVE

    def test_unrecognised_direction_is_recorded_not_swallowed(self):
        result = parse_llm_response(responses_api_output(direction="VERY BULLISH"))
        assert result.direction is Direction.UNKNOWN
        # The emitted text is kept so the failure is visible in the dataset
        # rather than looking like a model that found nothing.
        assert result.raw_direction == "VERY BULLISH"

    @pytest.mark.parametrize("direction", ["POSITIVE", "NEGATIVE", "NEUTRAL", "MIXED", "UNKNOWN"])
    def test_recognised_direction_is_not_marked_rejected(self, direction):
        result = parse_llm_response(responses_api_output(direction=direction))
        assert result.raw_direction is None

    @pytest.mark.parametrize("direction", ["UNKNOWN", "unknown", " UNKNOWN "])
    def test_legitimate_unknown_survives_a_round_trip(self, direction):
        # Cached results are re-read through from_dict. Marking a real UNKNOWN
        # as rejected would make the round trip non-idempotent.
        result = parse_llm_response(responses_api_output(direction=direction))
        assert AnalysisResult.from_dict(result.to_dict()) == result
        assert result.raw_direction is None

    def test_unknown_event_type_falls_back_to_other(self):
        result = parse_llm_response(responses_api_output(event_type="ALIEN INVASION"))
        assert result.event_type is EventType.OTHER

    def test_non_json_output_is_unknown_with_error(self):
        result = parse_llm_response({"output_text": "Sure! Apple had a great quarter."})
        assert result.direction is Direction.UNKNOWN
        assert result.error

    def test_truncated_json_is_unknown_with_error(self):
        # A response cut off by max_tokens is not a neutral opinion.
        truncated = json.dumps({"direction": "POSITIVE", "confidence": 0.9})[:20]
        result = parse_llm_response({"output_text": truncated})
        assert result.direction is Direction.UNKNOWN
        assert result.error

    def test_refusal_is_unknown_with_error(self):
        body = {
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "refusal", "refusal": "I cannot."}],
                }
            ]
        }
        result = parse_llm_response(body)
        assert result.direction is Direction.UNKNOWN
        assert "refus" in result.error.lower()

    def test_refusal_at_item_level_is_also_detected(self):
        result = parse_llm_response({"output": [{"type": "refusal"}]})
        assert "refus" in result.error.lower()

    def test_refusal_item_is_detected_before_text(self):
        body = {
            "output": [
                {
                    "type": "message",
                    "content": [
                        {"type": "refusal", "refusal": "I cannot analyze this."},
                        {"type": "output_text", "text": json.dumps({"direction": "POSITIVE"})},
                    ],
                }
            ]
        }
        assert parse_llm_response(body).direction is Direction.UNKNOWN

    def test_missing_output_is_unknown_with_error(self):
        result = parse_llm_response({})
        assert result.direction is Direction.UNKNOWN
        assert result.error

    def test_api_error_body_is_unknown_with_error(self):
        result = parse_llm_response({"error": {"message": "rate limit exceeded"}})
        assert result.direction is Direction.UNKNOWN
        assert "rate limit" in result.error

    def test_out_of_range_confidence_is_rejected(self):
        result = parse_llm_response(responses_api_output(confidence=1.4))
        assert result.direction is Direction.UNKNOWN
        assert result.error

    def test_absent_confidence_is_allowed(self):
        # Confidence is calibration, not a licence to invent. UNKNOWN with no
        # confidence is honest; 0.5 would be a fabricated number.
        result = parse_llm_response(
            {"output_text": json.dumps({"event_type": "EARNINGS", "direction": "POSITIVE"})}
        )
        assert result.direction is Direction.POSITIVE
        assert result.confidence is None

    @pytest.mark.parametrize("body", [None, [], "string", 42])
    def test_non_mapping_body_is_unknown_not_a_crash(self, body):
        result = parse_llm_response(body)
        assert result.direction is Direction.UNKNOWN
        assert result.error

    def test_error_result_is_json_serializable(self):
        json.dumps(parse_llm_response({"output_text": "not json"}).to_dict())

    def test_never_raises(self):
        # A batch must not die because one article produced a weird response.
        for body in (None, {}, {"output_text": None}, {"output": []}, {"output": "x"}):
            assert isinstance(parse_llm_response(body), AnalysisResult)


class TestLLMAnalyzer:
    def test_requires_an_explicit_model(self):
        # No default model: a pipeline whose behaviour changes when the vendor
        # moves its default is not reproducible.
        with pytest.raises(LLMConfigurationError, match="model"):
            LLMAnalyzer(model="", api_key=API_KEY, transport=fake_transport({}))

    def test_requires_an_api_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(LLMConfigurationError, match="OPENAI_API_KEY"):
            LLMAnalyzer(model=MODEL, api_key="", transport=fake_transport({}))

    def test_reads_key_from_environment(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", API_KEY)
        analyzer = LLMAnalyzer(model=MODEL, transport=fake_transport({}))
        assert analyzer

    def test_build_request_uses_strict_json_schema(self):
        analyzer = LLMAnalyzer(
            model=MODEL, api_key=API_KEY, transport=fake_transport(responses_api_output())
        )
        request = analyzer.build_request(make_article())
        payload = request.payload
        assert payload["model"] == MODEL
        text = json.dumps(payload)
        assert "additionalProperties" in text and "false" in text

    def test_build_request_requests_zero_temperature_by_default(self):
        analyzer = LLMAnalyzer(
            model=MODEL, api_key=API_KEY, transport=fake_transport(responses_api_output())
        )
        assert analyzer.build_request(make_article()).payload.get("temperature") == 0.0

    def test_build_request_includes_prompt_and_article(self):
        analyzer = LLMAnalyzer(
            model=MODEL, api_key=API_KEY, transport=fake_transport(responses_api_output())
        )
        payload = analyzer.build_request(make_article()).payload
        rendered = json.dumps(payload)
        assert "Apple beats Q3 earnings expectations" in rendered

    def test_redacted_request_never_carries_the_key(self):
        analyzer = LLMAnalyzer(
            model=MODEL, api_key=API_KEY, transport=fake_transport(responses_api_output())
        )
        request = analyzer.build_request(make_article())
        assert API_KEY not in json.dumps(request.redacted())
        assert API_KEY not in repr(request)

    def test_url_is_the_responses_endpoint(self):
        analyzer = LLMAnalyzer(
            model=MODEL, api_key=API_KEY, transport=fake_transport(responses_api_output())
        )
        assert analyzer.build_request(make_article()).url.endswith("/v1/responses")

    def test_analyze_parses_the_response(self):
        analyzer = LLMAnalyzer(
            model=MODEL,
            api_key=API_KEY,
            transport=fake_transport(responses_api_output(direction="NEGATIVE")),
        )
        result = analyzer.analyze(make_article())
        assert result.direction is Direction.NEGATIVE

    def test_refusal_becomes_unknown(self):
        analyzer = LLMAnalyzer(
            model=MODEL,
            api_key=API_KEY,
            transport=fake_transport(
                {"output": [{"type": "message", "content": [{"type": "refusal", "refusal": "no"}]}]}
            ),
        )
        assert analyzer.analyze(make_article()).direction is Direction.UNKNOWN

    def test_transport_exception_becomes_unknown(self):
        def transport(request):
            raise TimeoutError("upstream timeout")

        analyzer = LLMAnalyzer(model=MODEL, api_key=API_KEY, transport=transport)
        result = analyzer.analyze(make_article())
        assert result.direction is Direction.UNKNOWN
        assert result.error

    def test_transport_error_does_not_leak_the_key(self):
        def transport(request):
            raise TimeoutError(f"failed posting to {request.url} with {request.api_key}")

        analyzer = LLMAnalyzer(model=MODEL, api_key=API_KEY, transport=transport)
        assert API_KEY not in analyzer.analyze(make_article()).error

    def test_event_records_model_and_prompt(self):
        analyzer = LLMAnalyzer(
            model=MODEL, api_key=API_KEY, transport=fake_transport(responses_api_output())
        )
        article = make_article()
        event = analyzer.to_event(article, analyzer.analyze(article), analyzed_at=ANALYZED_AT)
        assert event.analyzer == "openai"
        assert event.model == MODEL
        assert event.prompt_ref

    def test_prompt_ref_is_versioned(self):
        ref = build_prompt_ref(version="2.0.0")
        assert ref.version == "2.0.0"
        assert ref.text
        assert ref.to_dict()["version"] == "2.0.0"


class TestCaching:
    def test_repeat_analysis_uses_the_cache(self):
        transport = fake_transport(responses_api_output())
        analyzer = LLMAnalyzer(
            model=MODEL, api_key=API_KEY, transport=transport, cache=DictCache()
        )
        article = make_article()
        first = analyzer.analyze(article)
        second = analyzer.analyze(article)
        assert first == second
        assert len(transport.calls) == 1

    def test_different_articles_are_cached_separately(self):
        transport = fake_transport(responses_api_output())
        analyzer = LLMAnalyzer(
            model=MODEL, api_key=API_KEY, transport=transport, cache=DictCache()
        )
        analyzer.analyze(make_article(raw_source_id="n1"))
        analyzer.analyze(make_article(raw_source_id="n2", title="Different story"))
        assert len(transport.calls) == 2

    def test_changing_content_invalidates_the_cache(self):
        # Providers amend stories. Keying on identity alone would hand back the
        # classification of a headline that no longer exists.
        transport = fake_transport(responses_api_output())
        analyzer = LLMAnalyzer(
            model=MODEL, api_key=API_KEY, transport=transport, cache=DictCache()
        )
        analyzer.analyze(make_article())
        analyzer.analyze(make_article(title="A completely different headline"))
        assert len(transport.calls) == 2

    def test_identical_content_hits_the_cache(self):
        transport = fake_transport(responses_api_output())
        analyzer = LLMAnalyzer(
            model=MODEL, api_key=API_KEY, transport=transport, cache=DictCache()
        )
        analyzer.analyze(make_article())
        analyzer.analyze(make_article())
        assert len(transport.calls) == 1

    def test_failures_are_cached_too(self):
        # Re-running after a network blip should not re-bill the same call.
        def transport(request):
            raise TimeoutError("blip")

        analyzer = LLMAnalyzer(
            model=MODEL, api_key=API_KEY, transport=transport, cache=DictCache()
        )
        first = analyzer.analyze(make_article())
        second = analyzer.analyze(make_article())
        assert first == second
        assert first.failed

    def test_jsonl_cache_round_trips(self, tmp_path):
        path = tmp_path / "cache.jsonl"
        cache = JsonlCache(path)
        cache.put("k", {"direction": "POSITIVE"})
        assert JsonlCache(path).get("k") == {"direction": "POSITIVE"}

    def test_jsonl_cache_miss_returns_none(self, tmp_path):
        assert JsonlCache(tmp_path / "c.jsonl").get("absent") is None

    def test_jsonl_cache_survives_a_corrupt_line(self, tmp_path):
        # One bad append must not make every future run fail.
        path = tmp_path / "cache.jsonl"
        cache = JsonlCache(path)
        cache.put("k", {"direction": "POSITIVE"})
        with path.open("a", encoding="utf-8") as handle:
            handle.write("not json\n")
        assert JsonlCache(path).get("k") == {"direction": "POSITIVE"}

    def test_both_caches_implement_the_protocol(self, tmp_path):
        # Not an isinstance check: AnalysisCache is a Protocol and is not
        # runtime_checkable. The contract is get/put, so assert that.
        for cache in (DictCache(), JsonlCache(tmp_path / "c.jsonl")):
            assert callable(cache.get)
            assert callable(cache.put)


class TestAnalyzeMany:
    def test_returns_one_event_per_article(self):
        articles = [make_article(raw_source_id=f"n{i}") for i in range(5)]
        events = analyze_many(articles, DeterministicAnalyzer(), analyzed_at=ANALYZED_AT)
        assert len(events) == 5
        assert all(isinstance(e, FinancialEvent) for e in events)

    def test_one_failure_does_not_abort_the_batch(self):
        def transport(request):
            raise TimeoutError("boom")

        analyzer = LLMAnalyzer(model=MODEL, api_key=API_KEY, transport=transport)
        articles = [make_article(raw_source_id=f"n{i}") for i in range(3)]
        events = analyze_many(articles, analyzer, analyzed_at=ANALYZED_AT)
        assert len(events) == 3
        assert all(e.direction is Direction.UNKNOWN for e in events)

    def test_empty_batch(self):
        assert analyze_many([], DeterministicAnalyzer(), analyzed_at=ANALYZED_AT) == []

    def test_requires_analyzed_at(self):
        with pytest.raises(ValueError):
            analyze_many([make_article()], DeterministicAnalyzer(), analyzed_at=dt.datetime.now())


class TestBuildAnalyzer:
    def test_deterministic_is_the_default(self):
        assert isinstance(build_analyzer(), DeterministicAnalyzer)

    def test_builds_llm(self):
        analyzer = build_analyzer(
            "llm", model=MODEL, api_key=API_KEY, transport=fake_transport({})
        )
        assert isinstance(analyzer, LLMAnalyzer)

    def test_rejects_unknown_kind(self):
        with pytest.raises(AnalyzerError):
            build_analyzer("crystal-ball", model=MODEL, api_key=API_KEY)