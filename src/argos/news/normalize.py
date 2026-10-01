"""Provider payload -> `NewsArticle`, with a fail-closed temporal policy.

`published_at` is the field everything else depends on. A backtest that assumes
news was available before it actually was produces a beautiful, profitable,
completely fictional equity curve. So the policy here is deliberately strict:

    REJECT, do not guess.

A missing timestamp, a naive timestamp, or a value that cannot be parsed all
result in the article being dropped and counted in a `rejected` list with a
reason. `retrieved_at` is never used as a fallback. "We know when we fetched
it" is a fact about our infrastructure, not about when the market could have
read the news, and treating the former as the latter is the single most common
way a news backtest lies to its author.

Rejected articles are reported rather than silently dropped. A batch that loses
40% of its rows to bad timestamps is a broken feed, and the count is how you
find out.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from argos.news.dedupe import article_identity
from argos.news.models import UTC, NewsArticle

_ISO_TRAILING_Z = "Z"

# Formats attempted in order, after the ISO parser has already failed. Listed
# explicitly rather than guessed by format sniffing so behaviour is testable.
_FALLBACK_FORMATS: tuple[str, ...] = (
    "%Y-%m-%d %H:%M:%S%z",
    "%Y-%m-%dT%H:%M:%S%z",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%dT%H:%M",
    "%Y-%m-%d",
)


def parse_utc(value: object) -> dt.datetime:
    """Parse a provider timestamp into a timezone-aware UTC datetime.

    Accepts an RFC3339/ISO-8601 string with an explicit offset, a trailing `Z`,
    an offset-bearing datetime, or one of the documented fallback layouts. Naive
    input is rejected rather than assumed to be UTC: a feed that omits the zone
    may be reporting exchange-local time, and quietly attaching UTC to it would
    shift every article by hours, potentially across a trading day boundary.

    Raises:
        TypeError: the value is not a string or datetime.
        ValueError: the value is unparseable, or has no timezone information.
    """
    if isinstance(value, dt.datetime):
        parsed = value
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError("timestamp is empty")
        candidate = text[:-1] + "+00:00" if text.endswith(_ISO_TRAILING_Z) else text
        try:
            parsed = dt.datetime.fromisoformat(candidate)
        except ValueError:
            parsed = _parse_with_fallbacks(text)
    else:
        raise TypeError(f"timestamp must be str or datetime, got {type(value).__name__}")

    if parsed.tzinfo is None or parsed.tzinfo.utcoffset(parsed) is None:
        raise ValueError(
            f"timestamp {value!r} has no timezone offset; refusing to assume UTC because "
            "the correct zone is unknown and guessing it would shift the article in time"
        )
    return parsed.astimezone(UTC)


def _parse_with_fallbacks(text: str) -> dt.datetime:
    for fmt in _FALLBACK_FORMATS:
        try:
            return dt.datetime.strptime(text, fmt)
        except ValueError:
            continue
    raise ValueError(f"timestamp {text!r} is not in any recognised format")


def reject_reason(payload: object) -> str | None:
    """Return why a raw record cannot be normalized, or None if it can be.

    Separated from `normalize_articles` so tests and the audit command can ask
    the same question of the same input without duplicating the rules.
    """
    if not isinstance(payload, dict):
        return f"record is {type(payload).__name__}, not an object"

    if _first_text(payload, _TITLE_KEYS) is None:
        # No title means no reviewable content; an article with a body but no
        # headline is not something the pipeline can reason about.
        return "missing or empty title"

    # Any of the known provider spellings counts. Checking only ARGOS's own
    # `published_at` here would reject every real provider payload, since
    # Massive sends `published_utc`.
    candidates = [
        payload[key] for key in _PUBLISHED_KEYS if payload.get(key) is not None
    ]
    if not candidates:
        return "missing published_at"

    last_error = "no recognised format"
    for value in candidates:
        try:
            parse_utc(value)
        except (TypeError, ValueError) as exc:
            last_error = str(exc)
            continue
        return None

    return f"unparseable published_at: {last_error}"


@dataclass(frozen=True, slots=True)
class NormalizationResult:
    """Outcome of a normalization batch, including everything we chose to drop.

    `rejected` keeps `(index, reason)` pairs so a caller can point at the exact
    offending row of the original provider file. The count alone is not enough
    to debug a feed problem.
    """

    articles: tuple[NewsArticle, ...] = ()
    rejected: tuple[tuple[int, str], ...] = ()
    duplicates: tuple[str, ...] = ()

    @property
    def accepted(self) -> int:
        return len(self.articles)

    @property
    def rejected_count(self) -> int:
        return len(self.rejected)

    @property
    def duplicate_count(self) -> int:
        return len(self.duplicates)

    def __iter__(self) -> Iterator[NewsArticle]:
        return iter(self.articles)

    def __len__(self) -> int:
        return len(self.articles)


# Provider payload keys that may carry the publication timestamp, in priority
# order. `published_at` is ARGOS's own name; the rest are the provider spellings
# seen in the wild. The first key present with a usable value wins.
_PUBLISHED_KEYS: tuple[str, ...] = (
    "published_at",
    "published_utc",
    "published",
    "publication_date",
    "pub_date",
)

# Provider payload keys for the article body, most to least preferred.
_TITLE_KEYS: tuple[str, ...] = ("title", "headline")
_SUMMARY_KEYS: tuple[str, ...] = ("summary", "description", "abstract", "body")
_URL_KEYS: tuple[str, ...] = ("article_url", "url", "link")
_PUBLISHER_KEYS: tuple[str, ...] = ("publisher", "source_name", "author_source")
_AUTHOR_KEYS: tuple[str, ...] = ("author", "authors")
_ID_KEYS: tuple[str, ...] = ("id", "article_id", "uuid", "guid")
_TICKER_KEYS: tuple[str, ...] = ("tickers", "symbols", "ticker")


def _first_text(payload: dict, keys: Iterable[str]) -> str | None:
    for key in keys:
        value = payload.get(key)
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            joined = " ".join(str(item).strip() for item in value if str(item).strip())
            if joined:
                return joined
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def _first_ticker(payload: dict) -> tuple[str, ...]:
    for key in _TICKER_KEYS:
        value = payload.get(key)
        if value is None:
            continue
        if isinstance(value, str):
            # Providers use "AAPL,MSFT" and "AAPL, MSFT" interchangeably.
            parts = [chunk.strip() for chunk in value.split(",")]
            cleaned = tuple(p for p in parts if p)
            if cleaned:
                return cleaned
            continue
        if isinstance(value, (list, tuple)):
            cleaned = tuple(str(item).strip() for item in value if str(item).strip())
            if cleaned:
                return cleaned
    return ()


def normalize_articles(
    payloads: Iterable[object],
    *,
    source: str,
    retrieved_at: dt.datetime,
    reject_duplicates: bool = True,
) -> NormalizationResult:
    """Normalize provider payloads into deduplicated, UTC-anchored articles.

    Args:
        payloads: Raw provider records, in any dict shape the provider uses.
        source: Provider name recorded on every article (`"massive"`, etc.).
        retrieved_at: When this batch was fetched. Stored for provenance, never
            used to fill a missing `published_at`.
        reject_duplicates: Collapse repeats via `article_identity`, keeping the
            earliest-published copy of each distinct article.

    Deduplication is on by default because provider pagination overlaps: asking
    for overlapping date ranges returns the same article again, and a duplicated
    article would double its own weight in the news score.
    """
    if not source:
        raise ValueError("source must name the provider")
    if retrieved_at.tzinfo is None:
        raise ValueError("retrieved_at must be timezone-aware")

    articles: list[NewsArticle] = []
    rejected: list[tuple[int, str]] = []

    for index, payload in enumerate(payloads):
        reason = reject_reason(payload)
        if reason is not None:
            rejected.append((index, reason))
            continue
        assert isinstance(payload, dict)  # guaranteed by reject_reason

        published_at = _first_published_at(payload)
        title = _first_text(payload, _TITLE_KEYS)
        assert title is not None  # guaranteed by reject_reason

        article = NewsArticle(
            article_id=article_identity(payload, source=source),
            source=source,
            title=title,
            published_at=published_at,
            retrieved_at=retrieved_at,
            publisher=_first_text(payload, _PUBLISHER_KEYS),
            author=_first_text(payload, _AUTHOR_KEYS),
            summary=_first_text(payload, _SUMMARY_KEYS),
            url=_first_text(payload, _URL_KEYS),
            raw_source_id=_first_text(payload, _ID_KEYS),
            tickers=_first_ticker(payload),
        )
        articles.append(article)

    ordered = tuple(sorted(articles, key=lambda a: (a.published_at, a.article_id)))
    if not reject_duplicates:
        return NormalizationResult(articles=ordered, rejected=tuple(rejected))

    from argos.news.dedupe import dedupe_articles  # avoid an import cycle

    unique, dropped = dedupe_articles(ordered)
    return NormalizationResult(
        articles=unique, rejected=tuple(rejected), duplicates=tuple(dropped)
    )


def _first_published_at(payload: dict) -> dt.datetime:
    """Take the first usable publication timestamp among the known key spellings.

    Provider fields are preferred over ours, and a key that exists but does not
    parse is treated as absent so the next candidate is tried. A key present
    with a bad value is worth knowing about, which is why `rejected` reports the
    failure when no key at all is usable.
    """
    for key in _PUBLISHED_KEYS:
        if key not in payload or payload.get(key) is None:
            continue
        try:
            return parse_utc(payload[key])
        except (TypeError, ValueError):
            continue
    raise ValueError("no usable publication timestamp found")


def default_retrieved_at() -> dt.datetime:
    """Current instant in UTC, for callers that fetch right now."""
    return dt.datetime.now(UTC)
