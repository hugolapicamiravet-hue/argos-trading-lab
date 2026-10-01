"""Shared fixtures for the news test suite.

Every fixture here is synthetic and self-authored. Two reasons, not one:

1. Tests must run with no network and no API key, so a live provider cannot be
   a dependency.
2. Provider payloads in this repository are hand-written approximations of a
   documented response shape. They are test input, not redistributed provider
   data, which keeps the licensing story clean.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import replace

import pytest

from argos.news.dedupe import article_key
from argos.news.models import FinancialEvent, NewsArticle

UTC = dt.UTC

# Fixed "now" for every test, so nothing depends on the wall clock.
NOW = dt.datetime(2026, 10, 1, 12, 0, 0, tzinfo=UTC)
ANALYZED_AT = dt.datetime(2026, 10, 1, 12, 30, 0, tzinfo=UTC)


def make_payload(
    *,
    article_id: str = "n1",
    title: str = "Apple beats Q3 earnings expectations",
    summary: str | None = "Record revenue driven by services.",
    published: object = "2025-07-31T20:05:00Z",
    publisher: str = "Reuters",
    url: str = "https://www.reuters.com/markets/apple-q3-2025",
    tickers: object = ("AAPL",),
) -> dict:
    """One provider-shaped record, with overridable fields.

    `published=None` omits the field entirely, which is how the "missing
    timestamp" cases are expressed.
    """
    payload: dict = {
        "id": article_id,
        "title": title,
        "summary": summary,
        "publisher": publisher,
        "article_url": url,
        "tickers": list(tickers) if isinstance(tickers, (list, tuple)) else tickers,
    }
    if published is not None:
        payload["published_utc"] = published
    return payload


def make_article(
    *,
    article_id: str | None = None,
    title: str = "Apple beats Q3 earnings expectations",
    summary: str | None = "Record revenue driven by services.",
    published_at: dt.datetime = dt.datetime(2025, 7, 31, 20, 5, tzinfo=UTC),
    retrieved_at: dt.datetime = NOW,
    publisher: str | None = "Reuters",
    url: str | None = "https://www.reuters.com/markets/apple-q3-2025",
    tickers: tuple[str, ...] = ("AAPL",),
    raw_source_id: str | None = "n1",
) -> NewsArticle:
    """A normalized article, with `article_id` derived unless given explicitly.

    Deriving it matters: a hardcoded id would make every "distinct" article in a
    test share one identity, and dedupe would then silently collapse them. That
    failure mode looks like a passing test.
    """
    article = NewsArticle(
        article_id=article_id or "__pending__",
        source="massive",
        title=title,
        summary=summary,
        published_at=published_at,
        retrieved_at=retrieved_at,
        publisher=publisher,
        url=url,
        tickers=tickers,
        raw_source_id=raw_source_id,
    )
    if article_id is not None:
        return article
    return replace(article, article_id=article_key(article))


def make_event(
    *,
    article_id: str | None = None,
    published_at: dt.datetime = dt.datetime(2025, 7, 31, 20, 5, tzinfo=UTC),
    direction: object = "POSITIVE",
    event_type: object = "EARNINGS",
    confidence: float | None = 0.6,
    magnitude: float | None = 0.5,
    tickers: tuple[str, ...] = ("AAPL",),
    analyzer: str = "deterministic",
    analyzed_at: dt.datetime = ANALYZED_AT,
    event_id: str | None = None,
    model: str | None = None,
    prompt_ref: str | None = None,
) -> FinancialEvent:
    from argos.news.models import Direction, EventType

    resolved_direction = (
        direction if isinstance(direction, Direction) else Direction.coerce(direction)
    )
    resolved_type = (
        event_type if isinstance(event_type, EventType) else EventType.coerce(event_type)
    )
    resolved_article_id = article_id or make_article().article_id
    return FinancialEvent(
        event_id=event_id or f"evt:{resolved_article_id}",
        article_id=resolved_article_id,
        source="massive",
        tickers=tickers,
        event_type=resolved_type,
        direction=resolved_direction,
        published_at=published_at,
        analyzed_at=analyzed_at,
        analyzer=analyzer,
        analyzer_version="1.0.0",
        model=model,
        prompt_ref=prompt_ref,
        confidence=confidence,
        magnitude=magnitude,
    )


@pytest.fixture
def article() -> NewsArticle:
    return make_article()


@pytest.fixture
def payloads() -> list[dict]:
    """A small batch exercising the normal path plus known bad records."""
    return [
        make_payload(article_id="n1"),
        make_payload(
            article_id="n2",
            title="Apple faces regulatory probe over app store",
            summary="Investigators opened an inquiry.",
            published="2025-08-01T12:00:00Z",
            publisher="CNBC",
            url="https://www.cnbc.com/2025/08/01/apple-probe.html",
        ),
        make_payload(
            article_id="n3",
            title="Microsoft raises guidance for cloud division",
            summary="Outlook improved.",
            published="2025-08-04T13:30:00Z",
            publisher="WSJ",
            url="https://www.wsj.com/articles/microsoft-guidance",
            tickers=("MSFT",),
        ),
    ]