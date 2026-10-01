"""Domain models for news: normalized articles and inferred financial events.

Design rules, in priority order:

1. A timestamp without a timezone is an error, not a convenience. `published_at`
   is the moment the information existed for the market; guessing its zone would
   make every downstream temporal guarantee meaningless.
2. Fact and inference never share a record. `NewsArticle` holds only what the
   provider said. `FinancialEvent` holds only what an analyzer concluded, and
   points back at the article it came from.
3. `retrieved_at` is provenance about us, never a fact about the market. It is
   stored, auditable, and explicitly not usable as a `published_at` substitute.
4. Unknown means unknown. Optional fields are `None`, never invented defaults
   and never a zero that pretends to be a measurement.

A note on `magnitude` and `confidence`, because these are routinely misread:

    magnitude   0.0-1.0, the analyzer's estimate of how much the news matters.
                It is NOT an expected return, a probability, or a price move.
    confidence  0.0-1.0, how sure the analyzer is about its own classification.
                It is NOT the probability that a price will rise.

`direction` is likewise a label about the article's content. It is not a trade
instruction and nothing in this package converts it into one.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
from dataclasses import dataclass, replace
from enum import StrEnum

# Bumped whenever the serialized shape of a record changes meaning. A dataset
# carries the version that produced it so a reader never has to guess.
ARTICLE_SCHEMA_VERSION = 1
EVENT_SCHEMA_VERSION = 1


class SchemaVersions:
    """Named schema versions, so provenance is explicit rather than implied."""

    article = ARTICLE_SCHEMA_VERSION
    event = EVENT_SCHEMA_VERSION


SCHEMA_VERSIONS = SchemaVersions()

UTC = dt.UTC


def _require_aware(value: dt.datetime, field_name: str) -> dt.datetime:
    """Reject naive datetimes and normalise everything to UTC.

    A naive datetime has no defined offset, so "14:37" could be any of 24
    possible instants. Accepting one would let a DST boundary or a mixed-timezone
    feed shift an article across a trading session, which is precisely the class
    of look-ahead this project exists to prevent.
    """
    if not isinstance(value, dt.datetime):
        raise TypeError(f"{field_name} must be a datetime, got {type(value).__name__}")
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        raise ValueError(
            f"{field_name} must be timezone-aware; a naive datetime has no defined "
            "instant. Pass a tz-aware value or use a provider timestamp that "
            "carries an offset."
        )
    return value.astimezone(UTC)


def _iso(value: dt.datetime) -> str:
    return _require_aware(value, "timestamp").isoformat().replace("+00:00", "Z")


def _parse_iso(value: str) -> dt.datetime:
    return _require_aware(dt.datetime.fromisoformat(str(value).replace("Z", "+00:00")), "timestamp")


def _check_unit_interval(value: float | None, field_name: str) -> float | None:
    if value is None:
        return None
    number = float(value)
    if math.isnan(number):
        raise ValueError(f"{field_name} must not be NaN")
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{field_name} must lie in [0.0, 1.0], got {number}")
    return number


class EventType(StrEnum):
    """A deliberately small, explicit taxonomy.

    Every additional category needs a definition, a test fixture and a
    justification. A large hierarchy nobody maintains is worse than a short one
    that people agree on. `OTHER` exists so a real article is never force-fit
    into a category it does not belong to.
    """

    EARNINGS = "EARNINGS"
    GUIDANCE = "GUIDANCE"
    MERGER_ACQUISITION = "MERGER_ACQUISITION"
    PRODUCT = "PRODUCT"
    REGULATORY = "REGULATORY"
    LEGAL = "LEGAL"
    MANAGEMENT = "MANAGEMENT"
    MACRO = "MACRO"
    ANALYST_ACTION = "ANALYST_ACTION"
    OTHER = "OTHER"

    @classmethod
    def coerce(cls, value: object) -> EventType:
        """Map a possibly-unknown string to a member, defaulting to OTHER.

        Analyzer output is not trusted to contain only valid enum values. A
        surprising value is recorded as OTHER rather than crashing a batch, and
        the raw string is preserved by the caller for inspection.
        """
        if isinstance(value, cls):
            return value
        text = str(value).strip().upper().replace(" ", "_").replace("-", "_")
        try:
            return cls(text)
        except ValueError:
            return cls.OTHER


class Direction(StrEnum):
    """The analyzer's read on an article's content.

    `UNKNOWN` is a real answer meaning "the analyzer could not tell", and it is
    deliberately distinct from `NEUTRAL` meaning "the article is balanced". The
    feature layer treats UNKNOWN as contributing nothing rather than as zero
    opinion, so an unclassified article cannot dilute a score.
    """

    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"
    NEUTRAL = "NEUTRAL"
    MIXED = "MIXED"
    UNKNOWN = "UNKNOWN"

    @classmethod
    def coerce(cls, value: object) -> Direction:
        if isinstance(value, cls):
            return value
        text = str(value).strip().upper()
        try:
            return cls(text)
        except ValueError:
            return cls.UNKNOWN

    @property
    def sign(self) -> float:
        """Numeric contribution to a news score.

        Only POSITIVE and NEGATIVE contribute. NEUTRAL, MIXED and UNKNOWN all
        score zero: a mixed article has no single sign, and an unclassified one
        must not be silently treated as balanced opinion.
        """
        if self is Direction.POSITIVE:
            return 1.0
        if self is Direction.NEGATIVE:
            return -1.0
        return 0.0


def _stable_digest(*parts: object) -> str:
    """SHA-256 over a canonical JSON encoding of `parts`.

    Python's built-in `hash()` is randomised per process by PYTHONHASHSEED, so an
    identifier derived from it would change between runs and silently break
    deduplication and event identity. SHA-256 is stable, standard and
    reproducible on any platform, which is the whole requirement.
    """
    payload = json.dumps(parts, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class NewsArticle:
    """A single news article, normalized from one provider response.

    Everything here is provider-supplied fact, apart from `retrieved_at`, which
    is our own record of when the fetch happened. No field on this class is an
    interpretation of the article.
    """

    article_id: str
    source: str
    title: str
    published_at: dt.datetime
    retrieved_at: dt.datetime
    publisher: str | None = None
    author: str | None = None
    summary: str | None = None
    url: str | None = None
    raw_source_id: str | None = None
    tickers: tuple[str, ...] = ()
    schema_version: int = ARTICLE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "published_at", _require_aware(self.published_at, "published_at"))
        object.__setattr__(
            self, "retrieved_at", _require_aware(self.retrieved_at, "retrieved_at")
        )
        if not self.article_id:
            raise ValueError("article_id must not be empty")
        if not self.source:
            raise ValueError("source must name the provider")
        if not str(self.title).strip():
            raise ValueError("title must not be empty: an untitled article cannot be reviewed")
        if self.retrieved_at < self.published_at:
            # Not fatal in principle (a provider clock can be wrong), but it is
            # a strong signal of a timestamp mix-up, and timestamp mix-ups are
            # the bug this project exists to catch.
            raise ValueError(
                f"retrieved_at {self.retrieved_at.isoformat()} is earlier than published_at "
                f"{self.published_at.isoformat()}; ingestion cannot precede publication"
            )
        clean = []
        for ticker in self.tickers:
            symbol = str(ticker).strip().upper()
            if symbol and symbol not in clean:
                clean.append(symbol)
        object.__setattr__(self, "tickers", tuple(clean))

    @property
    def fingerprint(self) -> str:
        """Content hash used as the last-resort deduplication key.

        The exact fields are documented in `argos.news.dedupe` and covered by
        tests, because a fingerprint whose inputs are unclear is a fingerprint
        nobody can reason about when it merges two different articles.
        """
        return _stable_digest(
            "argos.article.fingerprint.v1",
            self.source,
            self.publisher,
            _normalize_text(self.title),
            _normalize_text(self.summary or ""),
            self.published_at.isoformat(),
        )

    def has_ticker(self, ticker: str) -> bool:
        return str(ticker).strip().upper() in self.tickers

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "article_id": self.article_id,
            "source": self.source,
            "publisher": self.publisher,
            "title": self.title,
            "summary": self.summary,
            "url": self.url,
            "author": self.author,
            "published_at": _iso(self.published_at),
            "retrieved_at": _iso(self.retrieved_at),
            "tickers": list(self.tickers),
            "raw_source_id": self.raw_source_id,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> NewsArticle:
        version = int(payload.get("schema_version", ARTICLE_SCHEMA_VERSION))
        if version != ARTICLE_SCHEMA_VERSION:
            raise ValueError(
                f"article schema version {version} cannot be read by this build, which "
                f"understands version {ARTICLE_SCHEMA_VERSION}"
            )
        return cls(
            article_id=payload["article_id"],
            source=payload["source"],
            title=payload["title"],
            published_at=_parse_iso(payload["published_at"]),
            retrieved_at=_parse_iso(payload["retrieved_at"]),
            publisher=payload.get("publisher"),
            author=payload.get("author"),
            summary=payload.get("summary"),
            url=payload.get("url"),
            raw_source_id=payload.get("raw_source_id"),
            tickers=tuple(payload.get("tickers") or ()),
            schema_version=version,
        )


@dataclass(frozen=True, slots=True)
class FinancialEvent:
    """An analyzer's structured reading of one article.

    FACT versus INFERENCE
    ---------------------
    Fact (from `article`): the article exists, its publisher, its text, and the
    `published_at` at which all of that became public. The event inherits that
    timestamp deliberately, so "when was this knowable" and "when was this
    published" are the same question with the same answer.

    Inference (everything else): `event_type`, `direction`, `magnitude`,
    `confidence` and `rationale`. These are one analyzer's opinion, identified
    by `analyzer`, `analyzer_version`, `model` and `prompt_ref`, and dated by
    `analyzed_at`.

    A `FinancialEvent` cannot be interpreted or acted on. It carries no order,
    no size, no stop and no target, and nothing in this package will derive them
    from it. Turning a direction into a position is a strategy decision that
    belongs to a later, separate layer.
    """

    event_id: str
    article_id: str
    source: str
    tickers: tuple[str, ...]
    event_type: EventType
    direction: Direction
    published_at: dt.datetime
    analyzed_at: dt.datetime
    analyzer: str
    analyzer_version: str
    schema_version: int = EVENT_SCHEMA_VERSION
    model: str | None = None
    prompt_ref: str | None = None
    magnitude: float | None = None
    confidence: float | None = None
    rationale: str | None = None
    raw_event_type: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "published_at", _require_aware(self.published_at, "published_at"))
        object.__setattr__(self, "analyzed_at", _require_aware(self.analyzed_at, "analyzed_at"))
        object.__setattr__(self, "magnitude", _check_unit_interval(self.magnitude, "magnitude"))
        object.__setattr__(self, "confidence", _check_unit_interval(self.confidence, "confidence"))
        if not self.event_id:
            raise ValueError("event_id must not be empty")
        if not self.article_id:
            raise ValueError("article_id must name the source article; provenance is required")
        if not self.analyzer or not self.analyzer_version:
            raise ValueError(
                "analyzer and analyzer_version are required: an event without an identified "
                "producer cannot be audited a year later"
            )
        clean = []
        for ticker in self.tickers:
            symbol = str(ticker).strip().upper()
            if symbol and symbol not in clean:
                clean.append(symbol)
        object.__setattr__(self, "tickers", tuple(clean))
        if not isinstance(self.event_type, EventType):
            object.__setattr__(self, "event_type", EventType.coerce(self.event_type))
        if not isinstance(self.direction, Direction):
            object.__setattr__(self, "direction", Direction.coerce(self.direction))

    @property
    def known_at(self) -> dt.datetime:
        """The instant this event's information was public.

        Always the article's publication time, never `analyzed_at`. Running the
        analyzer today does not make 2020 news available to a 2020 decision, and
        a feature that keyed off `analyzed_at` would be a look-ahead bug wearing
        an audit trail.
        """
        return self.published_at

    def has_ticker(self, ticker: str) -> bool:
        return str(ticker).strip().upper() in self.tickers

    def contribution(self) -> float:
        """Signed, confidence-weighted contribution to a news score.

        `sign * confidence`, with an unconfident event shrinking toward zero
        rather than vanishing. The weighting is a fixed, documented convention
        chosen before any result was observed; it is not a fitted parameter.
        """
        if self.confidence is None:
            return self.direction.sign
        return self.direction.sign * self.confidence

    def with_article(self, article: NewsArticle) -> FinancialEvent:
        """Return a copy whose timestamp and tickers come from the article.

        The analyzer is not allowed to invent a publication time. Re-deriving it
        from the article here means even a buggy analyzer cannot emit an event
        that claims to have been known earlier than its source.
        """
        return replace(self, published_at=article.published_at, tickers=article.tickers)

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "event_id": self.event_id,
            "article_id": self.article_id,
            "source": self.source,
            "tickers": list(self.tickers),
            "event_type": self.event_type.value,
            "direction": self.direction.value,
            "published_at": _iso(self.published_at),
            "analyzed_at": _iso(self.analyzed_at),
            "analyzer": self.analyzer,
            "analyzer_version": self.analyzer_version,
            "model": self.model,
            "prompt_ref": self.prompt_ref,
            "magnitude": self.magnitude,
            "confidence": self.confidence,
            "rationale": self.rationale,
            "raw_event_type": self.raw_event_type,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> FinancialEvent:
        version = int(payload.get("schema_version", EVENT_SCHEMA_VERSION))
        if version != EVENT_SCHEMA_VERSION:
            raise ValueError(
                f"event schema version {version} cannot be read by this build, which "
                f"understands version {EVENT_SCHEMA_VERSION}"
            )
        return cls(
            event_id=payload["event_id"],
            article_id=payload["article_id"],
            source=payload["source"],
            tickers=tuple(payload.get("tickers") or ()),
            event_type=EventType.coerce(payload["event_type"]),
            direction=Direction.coerce(payload["direction"]),
            published_at=_parse_iso(payload["published_at"]),
            analyzed_at=_parse_iso(payload["analyzed_at"]),
            analyzer=payload["analyzer"],
            analyzer_version=payload["analyzer_version"],
            schema_version=version,
            model=payload.get("model"),
            prompt_ref=payload.get("prompt_ref"),
            magnitude=payload.get("magnitude"),
            confidence=payload.get("confidence"),
            rationale=payload.get("rationale"),
            raw_event_type=payload.get("raw_event_type"),
        )


def _normalize_text(text: str) -> str:
    """Lowercase and collapse whitespace, for fingerprinting only.

    Publisher headlines differ in casing and spacing far more often than they
    differ in content, so a fingerprint built on the raw string would miss
    obvious duplicates. Applied to fingerprints, never to stored text.
    """
    return " ".join(str(text).lower().split())
