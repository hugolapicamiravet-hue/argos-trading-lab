"""Analysis: turning an article into a structured financial event.

This module holds the boundary between ARGOS and a language model, and the
boundary is the point. A model reads an article and returns a label. It does not
see a portfolio, and nothing it returns can move one. `AnalysisResult` has no
size, no stop, no target and no order field, so even a hallucinating model has
nowhere to put a trade.

Three pieces, in increasing order of sophistication:

    Analyzer           protocol: identity, analyze(article) -> AnalysisResult
    DeterministicAnalyzer  keyword rules, no network, fully reproducible
    LLMAnalyzer        OpenAI Responses API with strict JSON Schema output

`PromptRef` exists because model output is only reproducible if you know what
was asked. An event records the prompt's name, version and content hash, so a
year later it is possible to say exactly what the model was told. Without that,
"the model changed" is indistinguishable from "the strategy stopped working".

`LLMAnalyzer` requires an explicit model name and is offline-testable through an
injected transport. There is no invented default model, because a plausible
guess at a model id is exactly the kind of detail that silently rots.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import hashlib
import json
import os
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Protocol

from argos.news.models import (
    Direction,
    EventType,
    FinancialEvent,
    NewsArticle,
    _check_unit_interval,
    _stable_digest,
)
from argos.news.providers import redact

DETERMINISTIC_ANALYZER_VERSION = "1.0.0"


class AnalyzerError(RuntimeError):
    """Analyzer could not produce a usable result."""


class LLMConfigurationError(RuntimeError):
    """The LLM adapter is missing configuration it cannot invent."""


# --------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PromptRef:
    """Identity of a prompt: name, version, and a hash of the exact text."""

    name: str
    version: str
    text: str

    @property
    def sha256(self) -> str:
        """Hash of the prompt text, so edits are detectable after the fact."""
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()

    @property
    def identifier(self) -> str:
        """Compact form recorded on every event: `name@version#sha8`."""
        return f"{self.name}@{self.version}#{self.sha256[:8]}"

    def to_dict(self) -> dict:
        return {"name": self.name, "version": self.version, "sha256": self.sha256}


@dataclass(frozen=True, slots=True)
class AnalyzerIdentity:
    """Who produced a result: which analyzer, which model, which prompt."""

    analyzer: str
    version: str
    model: str | None = None
    prompt_ref: str | None = None

    def to_dict(self) -> dict:
        return {
            "analyzer": self.analyzer,
            "version": self.version,
            "model": self.model,
            "prompt_ref": self.prompt_ref,
        }

    @property
    def cache_key_suffix(self) -> str:
        return f"{self.analyzer}:{self.version}:{self.model or '-'}:{self.prompt_ref or '-'}"


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AnalysisResult:
    """An analyzer's reading of one article. All inference, no fact.

    `magnitude` and `confidence` are the analyzer's own estimates of importance
    and certainty respectively. Neither is a probability of profit and neither
    is an expected return; the field docstrings exist because those
    misreadings lead to bad strategies downstream.
    """

    event_type: EventType = EventType.OTHER
    direction: Direction = Direction.UNKNOWN
    magnitude: float | None = None
    confidence: float | None = None
    rationale: str | None = None
    raw_event_type: str | None = None
    # The literal string the model emitted for `direction`, kept only when it was
    # not one of the accepted values. Without it a model answering "VERY
    # BULLISH" yields an ordinary-looking UNKNOWN event with no trace that
    # anything went wrong, indistinguishable from having found nothing.
    raw_direction: str | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_type", EventType.coerce(self.event_type))
        object.__setattr__(self, "direction", Direction.coerce(self.direction))
        object.__setattr__(self, "magnitude", _check_unit_interval(self.magnitude, "magnitude"))
        object.__setattr__(self, "confidence", _check_unit_interval(self.confidence, "confidence"))

    @property
    def failed(self) -> bool:
        """True when the analyzer could not produce a real classification."""
        return self.error is not None

    @classmethod
    def failure(cls, reason: str) -> AnalysisResult:
        """A fail-closed result.

        Direction UNKNOWN scores zero in the feature, so an article we could not
        read behaves exactly like an article with no signal, rather than being
        guessed at or dropped from a count that then disagrees with the dataset.
        """
        return cls(
            event_type=EventType.OTHER,
            direction=Direction.UNKNOWN,
            magnitude=None,
            confidence=0.0,
            rationale=None,
            error=reason,
        )

    def to_dict(self) -> dict:
        return {
            "event_type": self.event_type.value,
            "direction": self.direction.value,
            "magnitude": self.magnitude,
            "confidence": self.confidence,
            "rationale": self.rationale,
            "raw_event_type": self.raw_event_type,
            "raw_direction": self.raw_direction,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> AnalysisResult:
        direction = cls._coerce_direction(payload.get("direction"))
        return cls(
            event_type=EventType.coerce(payload.get("event_type", EventType.OTHER)),
            direction=direction,
            magnitude=payload.get("magnitude"),  # type: ignore[arg-type]
            confidence=payload.get("confidence"),  # type: ignore[arg-type]
            rationale=payload.get("rationale"),  # type: ignore[arg-type]
            raw_event_type=payload.get("raw_event_type"),  # type: ignore[arg-type]
            raw_direction=cls._rejected_direction(payload.get("direction"), direction),
            error=payload.get("error"),  # type: ignore[arg-type]
        )

    @staticmethod
    def _coerce_direction(value: object) -> Direction:
        return Direction.coerce(value)

    @staticmethod
    def _rejected_direction(raw: object, coerced: Direction) -> str | None:
        """Return the emitted text only when it was not a value we accept.

        A recognized value that happens to coerce to UNKNOWN -- `UNKNOWN` itself
        -- is not a rejection. Treating it as one would corrupt every cached
        UNKNOWN result on its way back out of `from_dict`.
        """
        if coerced is not Direction.UNKNOWN or not isinstance(raw, str):
            return None
        text = raw.strip()
        if not text:
            return None
        try:
            Direction(text.upper())
        except ValueError:
            return text
        return None


# --------------------------------------------------------------------------
# Interface
# --------------------------------------------------------------------------


class Analyzer(Protocol):
    """Anything that can read an article and return a structured opinion."""

    @property
    def identity(self) -> AnalyzerIdentity: ...

    def analyze(self, article: NewsArticle) -> AnalysisResult: ...

    def to_event(
        self, article: NewsArticle, result: AnalysisResult, *, analyzed_at: dt.datetime
    ) -> FinancialEvent: ...


def to_event(
    article: NewsArticle,
    result: AnalysisResult,
    identity: AnalyzerIdentity,
    *,
    analyzed_at: dt.datetime,
) -> FinancialEvent:
    """Build a `FinancialEvent` with a deterministic id and honest timestamps.

    `published_at` comes from the article, never from the analyzer. The event id
    is a SHA-256 over article id plus analyzer identity, so re-running the same
    analyzer on the same article yields the same id, while changing the prompt or
    model produces a different id rather than silently overwriting the old one.
    """
    event_id = _stable_digest(
        "argos.event.v1",
        article.article_id,
        identity.cache_key_suffix,
    )
    return FinancialEvent(
        event_id=event_id,
        article_id=article.article_id,
        source=article.source,
        tickers=article.tickers,
        event_type=result.event_type,
        direction=result.direction,
        published_at=article.published_at,
        analyzed_at=analyzed_at,
        analyzer=identity.analyzer,
        analyzer_version=identity.version,
        model=identity.model,
        prompt_ref=identity.prompt_ref,
        magnitude=result.magnitude,
        confidence=result.confidence,
        rationale=result.rationale,
        raw_event_type=result.raw_event_type,
    )


# --------------------------------------------------------------------------
# Deterministic baseline
# --------------------------------------------------------------------------

# Rule sets are ordered lists of (event_type, direction, phrase). First match
# wins, so order encodes priority: an article saying "beats earnings and issues
# guidance" is an EARNINGS story before it is a GUIDANCE story.
_NEGATIVE_TERMS = (
    "miss", "misses", "missed", "below expectations", "disappoint", "disappointing",
    "downgrade", "downgraded", "lawsuit", "sues", "sued", "probe", "investigation",
    "recall", "layoff", "layoffs", "job cuts", "bankruptcy", "default", "fraud",
    "halted", "halt", "suspend", "resign", "resigns", "step down", "decline",
    "falls", "fall", "slump", "warning", "cuts guidance", "cut guidance", "fine",
    "penalty", "sanction", "delay", "delayed", "weak", "loss", "losses",
)
_POSITIVE_TERMS = (
    "beat", "beats", "exceed", "exceeds", "upgrade", "upgraded", "record revenue",
    "record profit", "wins approval", "approved", "acquisition", "acquires",
    "acquired", "merger", "buyback", "dividend", "raises guidance", "raised guidance",
    "profit", "growth", "surges", "surge", "jumps", "rally", "strong", "partnership",
)
_NEUTRAL_TERMS = (
    "reports", "announces", "schedules", "appoints", "names", "conference",
    "presentation", "outlook", "quarterly results", "files",
)
_GUIDANCE_TERMS = ("guidance", "outlook", "forecast", "raises", "cuts")
_MERGER_TERMS = (
    "merger", "acquisition", "acquire", "acquires", "acquired", "buyout", "takeover",
)
_REGULATORY_TERMS = (
    "regulator", "regulatory", "ftc", "doj", "sec ", "antitrust", "approval", "fda",
)
_LEGAL_TERMS = (
    "lawsuit", "litigation", "sues", "sued", "court", "settlement", "injunction", "probe",
)
_EARNINGS_TERMS = (
    "earnings", "eps", "revenue", "quarterly results", "q1", "q2", "q3", "q4",
)
_PRODUCT_TERMS = (
    "launch", "launches", "unveil", "announces product", "release", "new product",
)
_MANAGEMENT_TERMS = (
    "ceo", "cfo", "appoints", "resign", "steps down", "board", "chairman",
)
_MACRO_TERMS = (
    "fed", "fomc", "inflation", "cpi", "interest rate", "rate cut", "rate hike",
    "gdp", "jobs report",
)
_ANALYST_TERMS = (
    "price target", "analyst", "rating", "initiates coverage", "downgrade", "upgrade",
)

_DIRECTION_RULES: tuple[tuple[Direction, tuple[str, ...]], ...] = (
    (Direction.NEGATIVE, _NEGATIVE_TERMS),
    (Direction.POSITIVE, _POSITIVE_TERMS),
    (Direction.NEUTRAL, _NEUTRAL_TERMS),
)
_TYPE_RULES: tuple[tuple[EventType, tuple[str, ...]], ...] = (
    (EventType.GUIDANCE, _GUIDANCE_TERMS),
    (EventType.MERGER_ACQUISITION, _MERGER_TERMS),
    (EventType.REGULATORY, _REGULATORY_TERMS),
    (EventType.LEGAL, _LEGAL_TERMS),
    (EventType.EARNINGS, _EARNINGS_TERMS),
    (EventType.PRODUCT, _PRODUCT_TERMS),
    (EventType.MANAGEMENT, _MANAGEMENT_TERMS),
    (EventType.ANALYST_ACTION, _ANALYST_TERMS),
    (EventType.MACRO, _MACRO_TERMS),
)


class DeterministicAnalyzer:
    """Keyword classifier with no network access and no randomness.

    It exists as the reference implementation: it is trivial to read, trivially
    reproducible and trivially wrong in known ways. Any claim that an LLM adds
    value has to beat this baseline, which means the baseline cannot be a
    placeholder.

    Confidence is a small fixed step function of how many matching phrases were
    found, capped well below 1.0 on purpose. These are string matches, not
    understanding, and a number near 1.0 would overstate them.
    """

    name = "deterministic"

    def __init__(self, *, version: str = DETERMINISTIC_ANALYZER_VERSION) -> None:
        self._identity = AnalyzerIdentity(analyzer=self.name, version=version)

    @property
    def identity(self) -> AnalyzerIdentity:
        return self._identity

    def analyze(self, article: NewsArticle) -> AnalysisResult:
        text = _article_text(article)
        lowered = text.lower()

        event_type = EventType.OTHER
        for candidate, terms in _TYPE_RULES:
            if any(term in lowered for term in terms):
                event_type = candidate
                break

        direction = Direction.UNKNOWN
        hits = 0
        for candidate, terms in _DIRECTION_RULES:
            found = sum(1 for term in terms if term in lowered)
            if found and (direction is Direction.UNKNOWN or found > hits):
                direction, hits = candidate, found

        # Match count is weak evidence. Two phrases -> 0.3, five -> 0.6, ten ->
        # 0.8, and never 1.0.
        confidence = min(0.8, round(0.1 * hits, 4)) if hits else 0.0
        magnitude = min(0.8, round(0.08 * hits, 4)) if hits else None

        matched = [
            term
            for _, terms in _DIRECTION_RULES
            for term in terms
            if term in lowered
        ][:5]

        return AnalysisResult(
            event_type=event_type,
            direction=direction,
            magnitude=magnitude,
            confidence=confidence,
            rationale=(
                f"keyword match ({hits} term{'s' if hits != 1 else ''}): {', '.join(matched)}"
                if matched
                else "no directional keywords matched"
            ),
            raw_event_type=event_type.value,
        )

    def to_event(
        self, article: NewsArticle, result: AnalysisResult, *, analyzed_at: dt.datetime
    ) -> FinancialEvent:
        return to_event(article, result, self._identity, analyzed_at=analyzed_at)


def _article_text(article: NewsArticle) -> str:
    return " ".join(part for part in (article.title, article.summary) if part)


# --------------------------------------------------------------------------
# LLM adapter
# --------------------------------------------------------------------------

EVENT_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["event_type", "direction", "magnitude", "confidence", "rationale"],
    "properties": {
        "event_type": {
            "type": "string",
            "enum": [t.value for t in EventType],
            "description": "Best-fitting category for the article.",
        },
        "direction": {
            "type": "string",
            "enum": [d.value for d in Direction],
            "description": (
                "Effect on the subject company or market. NEUTRAL means balanced or "
                "factual with no directional read; UNKNOWN means the article does not "
                "support a judgement. Do not equate a factual tone with NEUTRAL."
            ),
        },
        "magnitude": {
            "type": ["number", "null"],
            "minimum": 0,
            "maximum": 1,
            "description": (
                "Estimated importance of the news from 0.0 (immaterial) to 1.0 "
                "(transformational). This is importance, not expected return."
            ),
        },
        "confidence": {
            "type": ["number", "null"],
            "minimum": 0,
            "maximum": 1,
            "description": (
                "How sure you are about this classification, from 0.0 to 1.0. This is "
                "certainty about the label, not probability of a price move."
            ),
        },
        "rationale": {
            "type": ["string", "null"],
            "description": (
                "One or two sentences citing the specific wording that drove the "
                "classification. Quote or paraphrase the source; do not speculate."
            ),
        },
    },
}

SYSTEM_PROMPT = """\
You are a financial news classifier. You read one article and return a \
structured label describing it.

Rules:
- Classify only what the article states. Do not infer facts it does not contain.
- If the article does not support a judgement, return direction "UNKNOWN". \
Factual tone is not the same as "NEUTRAL"; use NEUTRAL only for genuinely \
balanced coverage.
- magnitude measures importance, not expected price impact. A minor product \
launch is low magnitude even if positive.
- confidence measures certainty about your own label, not likelihood of profit.
- In rationale, cite the specific wording that drove your label.
- You produce research annotations only. You do not give trading advice, \
positions, sizes or orders.
"""

USER_PROMPT_TEMPLATE = """\
Publisher: {publisher}
Published (UTC): {published_at}
Tickers: {tickers}

Headline:
{title}

Body:
{summary}
"""


def build_prompt_ref(*, version: str = "1.0.0") -> PromptRef:
    """The prompt used by the LLM adapter, as a versioned reference."""
    return PromptRef(name="news_classifier", version=version, text=SYSTEM_PROMPT)


@dataclass(frozen=True, slots=True)
class OpenAIRequest:
    """A fully-built Responses API request.

    Exposed as a value so tests can assert on it without a network call, and so
    the exact payload sent to the provider is reviewable.
    """

    payload: Mapping[str, object]
    url: str
    # repr=False: a dataclass repr is what ends up in a debugger, a traceback
    # and a pytest failure report. With the default, any of those prints the key.
    api_key: str = field(repr=False)

    def redacted(self) -> dict:
        """The payload with the key removed, safe to log."""
        return {"url": self.url, "payload": self.payload}


def _default_transport(request: OpenAIRequest) -> Mapping[str, object]:
    """POST the request and return the decoded Responses API body.

    Real network call. Only reached when an `LLMAnalyzer` is constructed without
    an injected transport, which the tests never do.
    """
    http_request = urllib.request.Request(
        request.url,
        data=json.dumps(request.payload).encode("utf-8"),
        method="POST",
        headers={
            "Authorization": f"Bearer {request.api_key}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(http_request, timeout=120) as response:
            body = response.read()
    except urllib.error.HTTPError as exc:
        detail = ""
        # Diagnostics only: an unreadable error body must not mask the failure.
        with contextlib.suppress(Exception):
            detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise AnalyzerError(f"OpenAI request failed with HTTP {exc.code}: {detail}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise AnalyzerError(f"OpenAI request failed: {exc}") from exc

    try:
        decoded = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AnalyzerError(f"OpenAI returned a non-JSON body: {exc}") from exc
    if not isinstance(decoded, Mapping):
        raise AnalyzerError(f"OpenAI returned {type(decoded).__name__}, expected an object")
    return decoded


class LLMAnalyzer:
    """OpenAI Responses API analyzer with strict JSON Schema output.

    Structured output is not a preference here, it is the correctness mechanism.
    With `strict: true` the provider constrains generation to the schema, so
    `direction` cannot come back as "bullish" or `confidence` as the string
    "high". The residual failure modes are refusals, truncated output and
    `incomplete_details`, all of which are handled as explicit failures rather
    than exceptions that would abort a batch mid-way.

    The model id is required, not defaulted: guessing a plausible name would put
    an unverified detail in the provenance of every event this analyzer produces.

    Everything network-facing is injectable, so the whole adapter is testable
    offline. No `openai` dependency is added for this; the endpoint is one POST.
    """

    name = "openai"

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        prompt_ref: PromptRef | None = None,
        transport: Callable[[OpenAIRequest], Mapping[str, object]] | None = None,
        base_url: str = "https://api.openai.com/v1/responses",
        temperature: float | None = 0.0,
        timeout: float = 120.0,
        version: str = "1.0.0",
        cache: AnalysisCache | None = None,
    ) -> None:
        if not model or not model.strip():
            raise LLMConfigurationError(
                "model is required and must be verified against your OpenAI account; "
                "no default is provided because a wrong model id would be recorded in "
                "the provenance of every event"
            )
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise LLMConfigurationError(
                "OPENAI_API_KEY is not set. Export it or pass api_key=...; the key is "
                "never written to disk and never logged."
            )
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.timeout = float(timeout)
        self._transport = transport or _default_transport
        self._prompt = prompt_ref or build_prompt_ref()
        self._cache = cache
        self._identity = AnalyzerIdentity(
            analyzer=self.name,
            version=version,
            model=model,
            prompt_ref=self._prompt.identifier,
        )
        self.call_count = 0
        self.cache_hits = 0

    @property
    def identity(self) -> AnalyzerIdentity:
        return self._identity

    @property
    def prompt(self) -> PromptRef:
        return self._prompt

    def build_request(self, article: NewsArticle) -> OpenAIRequest:
        """Assemble the Responses API payload for one article."""
        user_text = USER_PROMPT_TEMPLATE.format(
            publisher=article.publisher or "unknown",
            published_at=article.published_at.isoformat().replace("+00:00", "Z"),
            tickers=", ".join(article.tickers) or "none",
            title=article.title,
            summary=article.summary or "(no body provided)",
        )
        payload: dict[str, object] = {
            "model": self.model,
            "input": [
                {"role": "system", "content": [{"type": "input_text", "text": SYSTEM_PROMPT}]},
                {"role": "user", "content": [{"type": "input_text", "text": user_text}]},
            ],
            # The documented structured-output envelope. `strict` makes the
            # schema binding a guarantee rather than a request.
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "news_event",
                    "strict": True,
                    "schema": EVENT_SCHEMA,
                }
            },
        }
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        return OpenAIRequest(payload=payload, url=self.base_url, api_key=self.api_key)

    def analyze(self, article: NewsArticle) -> AnalysisResult:
        # The key includes the content fingerprint, not just the article id.
        # Providers do amend stories, and a corrected headline is a different
        # claim deserving a different analysis; keying on identity alone would
        # hand back the classification of the version that no longer exists.
        cache_key = (
            f"{article.article_id}|{article.fingerprint}|{self._identity.cache_key_suffix}"
        )
        if self._cache is not None:
            cached = self._cache.get(cache_key)
            if cached is not None:
                self.cache_hits += 1
                return AnalysisResult.from_dict(cached)

        request = self.build_request(article)
        self.call_count += 1
        try:
            body = self._transport(request)
        except Exception as exc:
            # A timeout or connection reset on article 400 of 500 must not throw
            # away the 399 that already succeeded. Fail this article closed and
            # let the batch finish, so the gap is visible and re-runnable.
            detail = redact(str(exc), self.api_key)
            result = AnalysisResult.failure(f"transport failed: {type(exc).__name__}: {detail}")
        else:
            result = parse_llm_response(body)

        if self._cache is not None:
            # Failures are cached too. Re-running a batch after a network blip
            # should not re-bill the same API call for the same article.
            self._cache.put(cache_key, result.to_dict())
        return result

    def to_event(
        self, article: NewsArticle, result: AnalysisResult, *, analyzed_at: dt.datetime
    ) -> FinancialEvent:
        return to_event(article, result, self._identity, analyzed_at=analyzed_at)


def parse_llm_response(body: Mapping[str, object]) -> AnalysisResult:
    """Turn a Responses API body into an `AnalysisResult`, never raising.

    Every failure becomes a fail-closed `AnalysisResult` carrying the reason.
    Raising would abort a batch after N good articles and lose the run; a failed
    article that scores zero is visible in the dataset and recoverable on a
    retry.
    """
    if not isinstance(body, Mapping):
        # A transport can hand back None, a list or an error object. `.get` on
        # any of those raises, which would abort the batch at the one article
        # that misbehaved.
        return AnalysisResult.failure(
            f"response body was {type(body).__name__}, expected an object"
        )

    if body.get("error"):
        return AnalysisResult.failure(f"provider error: {body['error']}")

    if body.get("incomplete_details"):
        return AnalysisResult.failure(
            f"incomplete response: {json.dumps(body['incomplete_details'], default=str)}"
        )

    text = _extract_output_text(body)
    if text is None:
        if _has_refusal(body):
            return AnalysisResult.failure("model refused to classify the article")
        return AnalysisResult.failure("response contained no output text")

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        return AnalysisResult.failure(f"output was not valid JSON: {exc}")
    if not isinstance(parsed, Mapping):
        return AnalysisResult.failure(f"output was {type(parsed).__name__}, expected an object")

    required = {"event_type", "direction"}
    missing = required - set(parsed)
    if missing:
        # With strict schema this should be unreachable; it is checked anyway
        # because a missing field silently defaulted to UNKNOWN would hide a
        # provider-side change.
        return AnalysisResult.failure(f"output missing required field(s): {sorted(missing)}")

    try:
        result = AnalysisResult.from_dict(parsed)
    except (TypeError, ValueError) as exc:
        return AnalysisResult.failure(f"output failed validation: {exc}")

    if result.failed:
        return result
    if result.direction is Direction.UNKNOWN and result.event_type is EventType.OTHER:
        # A well-formed but entirely empty answer is treated as a soft failure:
        # it carries no information, and logging it as a real result would
        # inflate the dataset's apparent coverage.
        return AnalysisResult.failure("model returned UNKNOWN/OTHER with no rationale")
    return result


def _extract_output_text(body: Mapping[str, object]) -> str | None:
    output = body.get("output")
    if not isinstance(output, list):
        return _extract_convenience_output_text(body)
    for item in output:
        if not isinstance(item, Mapping):
            continue
        if item.get("type") == "refusal":
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, Mapping) and block.get("type") == "output_text":
                value = block.get("text")
                if isinstance(value, str) and value.strip():
                    return value
    return _extract_convenience_output_text(body)


def _extract_convenience_output_text(body: Mapping[str, object]) -> str | None:
    """Accept the flattened `output_text` field when the API returns it.

    The nested `output[].content[].output_text` form is the documented REST
    shape, but SDKs and proxies also surface a single flattened `output_text`.
    Ignoring it means a perfectly good analysis is discarded as UNKNOWN, which
    is indistinguishable from the model having found nothing.
    """
    value = body.get("output_text")
    if isinstance(value, str) and value.strip():
        return value
    return None


def _has_refusal(body: Mapping[str, object]) -> bool:
    """Detect a refusal at either level the API can use.

    A refusal arrives either as an `output[]` item of type `refusal`, or, in the
    documented form, as an `output[].content[]` block of type `refusal`. Only
    checking one of them makes a real refusal report itself as "no output text",
    which sends an operator looking for a schema problem instead of a model that
    declined.
    """
    output = body.get("output")
    if not isinstance(output, list):
        return False
    for item in output:
        if not isinstance(item, Mapping):
            continue
        if item.get("type") == "refusal":
            return True
        content = item.get("content")
        if isinstance(content, list) and any(
            isinstance(block, Mapping) and block.get("type") == "refusal"
            for block in content
        ):
            return True
    return False


# --------------------------------------------------------------------------
# Caching
# --------------------------------------------------------------------------


class AnalysisCache(Protocol):
    """Keyed storage for analysis results, so reruns are free and deterministic."""

    def get(self, key: str) -> Mapping[str, object] | None: ...

    def put(self, key: str, value: Mapping[str, object]) -> None: ...


class DictCache:
    """In-memory cache. Correct, lost on exit, fine for tests and short runs."""

    def __init__(self) -> None:
        self._data: dict[str, Mapping[str, object]] = {}

    def get(self, key: str) -> Mapping[str, object] | None:
        return self._data.get(key)

    def put(self, key: str, value: Mapping[str, object]) -> None:
        self._data[key] = dict(value)

    def __len__(self) -> int:
        return len(self._data)


class JsonlCache:
    """Append-only JSONL cache on disk.

    Keyed by article plus analyzer identity, which means changing the prompt or
    model does not reuse stale results. That is the whole reason the key has that
    shape: the worst outcome here is silently comparing a new prompt's output
    against an old one's.
    """

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = path
        self._loaded = False
        self._data: dict[str, Mapping[str, object]] = {}

    def _load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    record = json.loads(stripped)
                except json.JSONDecodeError:
                    continue  # a corrupt cache line costs a recompute, nothing more
                if isinstance(record, dict) and "key" in record:
                    self._data[str(record["key"])] = record.get("value", {})

    def get(self, key: str) -> Mapping[str, object] | None:
        self._load()
        return self._data.get(key)

    def put(self, key: str, value: Mapping[str, object]) -> None:
        self._load()
        self._data[key] = dict(value)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"key": key, "value": dict(value)}, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def __len__(self) -> int:
        self._load()
        return len(self._data)


# --------------------------------------------------------------------------
# Construction
# --------------------------------------------------------------------------


def build_analyzer(
    kind: str = "deterministic",
    *,
    model: str | None = None,
    prompt_version: str | None = None,
    cache_path: str | os.PathLike[str] | None = None,
    **kwargs: object,
) -> Analyzer:
    """Construct an analyzer by name.

    The two supported kinds are deliberately few. Every additional analyzer is a
    new set of behaviours that needs its own fixtures and its own justification
    against the deterministic baseline, and there is no evidence yet that either
    of these two earns its place.
    """
    name = str(kind).strip().lower()
    cache = JsonlCache(cache_path) if cache_path else None

    if name == "deterministic":
        return DeterministicAnalyzer(**kwargs)  # type: ignore[arg-type]
    if name in ("openai", "llm"):
        if not model:
            raise LLMConfigurationError(
                "--model is required for the openai analyzer; verify the model id against "
                "your account rather than accepting a default"
            )
        prompt_ref = build_prompt_ref(version=prompt_version) if prompt_version else None
        return LLMAnalyzer(model=model, prompt_ref=prompt_ref, cache=cache, **kwargs)  # type: ignore[arg-type]
    raise AnalyzerError(
        f"unknown analyzer {kind!r}; available: ['deterministic', 'openai']"
    )


def analyze_many(
    articles: Iterable[NewsArticle],
    analyzer: Analyzer,
    *,
    analyzed_at: dt.datetime,
) -> list[FinancialEvent]:
    """Run one analyzer over a batch, returning events in input order.

    A single failure is caught and recorded as a fail-closed event rather than
    stopping the run, so one malformed article cannot cost the other thousand
    their analysis.
    """
    events: list[FinancialEvent] = []
    for article in articles:
        try:
            result = analyzer.analyze(article)
            event = analyzer.to_event(article, result, analyzed_at=analyzed_at)
        except (AnalyzerError, ValueError, TypeError) as exc:
            event = to_event(
                article,
                AnalysisResult.failure(f"{type(exc).__name__}: {exc}"),
                getattr(analyzer, "identity", AnalyzerIdentity(analyzer="unknown", version="0")),
                analyzed_at=analyzed_at,
            )
        events.append(event)
    return events