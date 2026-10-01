"""Deterministic article identity and deduplication.

The same story reaches us from three places: a provider's stable id, the article
URL, and the text itself. Each has a failure mode, so identity is a cascade that
only falls back as far as it must:

    1. provider id    (``massive:<1234>``)
       Authoritative when present. Breaks if the provider renumbers ids.
    2. canonical URL  (``url:<aapl-q3-2025>``)
       Survives id renumbering. Breaks on tracking parameters and trailing
       slashes, both stripped here.
    3. fingerprint    (``fp:<sha256>``)
       Title + publisher + summary + publication instant. Survives a URL change,
       but merges two genuinely identical wire stories on the same day, which is
       accepted: two wires carrying one press release are one event.

Every step uses SHA-256 over canonical JSON, never Python's `hash()`. `hash()` is
salted per process (PYTHONHASHSEED), so ids built from it would differ between
two runs of the same pipeline, silently defeating both dedupe and event identity.
Tests pin this; the comment is here so nobody "simplifies" it back.

That cascade defines one article's *identity*. It is deliberately not how
deduplication groups records: two feeds syndicate one story and each gives it
its own provider id, so both records pass step 1 and both survive, splitting one
article into two identities and double-counting it. `dedupe_articles` therefore
groups by *any* shared signal -- identity, canonical URL, or fingerprint -- using
union-find so a record linking two groups merges them and the outcome does not
depend on input order. `article_key` stays a cascade because callers that only
need one key should not get a merge.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from argos.news.models import NewsArticle

ID_PREFIX = "provider:"
URL_PREFIX = "url:"
FINGERPRINT_PREFIX = "fp:"

# Tracking parameters stripped before URL comparison. A story shared on Twitter
# and read directly differs only by these, so keeping them would split one
# article into two identities and double-count it.
_TRACKING_PARAMS = frozenset(
    {
        "utm_source",
        "utm_medium",
        "utm_campaign",
        "utm_term",
        "utm_content",
        "utm_id",
        "utm_source_platform",
        "utm_creative_format",
        "utm_marketing_tactic",
        "gclid",
        "fbclid",
        "mc_cid",
        "mc_eid",
        "igshid",
        "ref",
        "ref_src",
        "referrer",
        "cmpid",
        "smid",
        "partner",
        "source",
        "spm",
        "yclid",
        "_hsenc",
        "_hsmi",
        "trk",
        "trackingId",
    }
)

def canonical_url(url: str | None) -> str | None:
    """Reduce a URL to a comparable identity, or None if there is none.

    Lowercases scheme and host, drops `www.`, removes tracking parameters,
    strips fragments, and sorts remaining query parameters so that parameter
    order cannot create a false distinction.
    """
    if not url:
        return None
    raw = str(url).strip()
    if not raw:
        return None

    # Force a scheme so urlsplit can read the netloc. Some feeds emit
    # "example.com/story" with no scheme at all.
    if "//" not in raw:
        raw = f"https://{raw}"

    try:
        parts = urlsplit(raw)
    except ValueError:
        return None

    if not parts.netloc:
        return None

    scheme = (parts.scheme or "https").lower()
    if scheme == "http":
        # Treat http and https as the same article; feeds disagree constantly.
        scheme = "https"

    # Take the part after any userinfo, which also discards credentials
    # embedded in a URL so they cannot leak into a stored dataset.
    host = parts.netloc.lower().split("@")[-1]
    if host.endswith(":80") or host.endswith(":443"):
        host = host.rsplit(":", 1)[0]
    # Strip only a leading `www.`. The rest of the host is load-bearing:
    # `news.example.com` and `example.com` are different pages, and reducing
    # either to a bare label would let two unrelated sites share an identity.
    if host.startswith("www."):
        host = host[4:]

    path = re.sub(r"/{2,}", "/", parts.path).rstrip("/")

    kept = [
        (key, value)
        for key, value in parse_qsl(parts.query, keep_blank_values=True)
        if key not in _TRACKING_PARAMS
    ]
    query = urlencode(sorted(kept))

    return urlunsplit((scheme, host, path, query, ""))


def article_identity(payload: object, *, source: str) -> str:
    """Compute the stable identity string for one raw provider record.

    Used at normalization time, where the record is still a raw dict and no
    `NewsArticle` exists yet. Mirrors `article_key` below; both are covered by
    the same tests so the two paths cannot drift apart.
    """
    if not source:
        raise ValueError("source must name the provider")
    if not isinstance(payload, dict):
        raise TypeError(f"payload must be a dict, got {type(payload).__name__}")

    for key in ("id", "article_id", "uuid", "guid"):
        value = payload.get(key)
        if value is not None and str(value).strip():
            return f"{ID_PREFIX}{source}:{str(value).strip()}"

    for key in ("article_url", "url", "link"):
        value = payload.get(key)
        url = canonical_url(value if isinstance(value, str) else None)
        if url:
            return f"{URL_PREFIX}{url}"


    return f"{FINGERPRINT_PREFIX}{_fingerprint_from_payload(payload, source)}"


def article_key(article: NewsArticle) -> str:
    """Identity of an already-normalized article, cascade as documented above."""
    if article.raw_source_id:
        return f"{ID_PREFIX}{article.source}:{article.raw_source_id}"
    url = canonical_url(article.url)
    if url:
        return f"{URL_PREFIX}{url}"
    return f"{FINGERPRINT_PREFIX}{article.fingerprint}"


def _fingerprint_from_payload(payload: dict, source: str) -> str:
    from argos.news.models import _stable_digest

    def text(*keys: str) -> str:
        for key in keys:
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return " ".join(value.lower().split())
        return ""

    publisher = ""
    for key in ("publisher", "source_name", "author_source"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            publisher = " ".join(value.lower().split())
            break

    published = str(payload.get("published_utc") or payload.get("published_at") or "")
    return _stable_digest(
        "argos.article.fingerprint.v1", source, publisher, text("title", "headline"),
        text("summary", "description", "abstract", "body"), published,
    )


def _merge_signals(article: NewsArticle) -> set[str]:
    """Every signal by which this article can be recognized as another one.

    Identity alone is not enough to collapse a syndicated story: two feeds give
    the same article different provider ids, so both records pass the id check
    and both survive. Comparing the canonical URL and the content fingerprint
    across records is what actually collapses them, which is the double-count
    this module exists to prevent.
    """
    signals = {article_key(article)}
    url = canonical_url(article.url)
    if url:
        signals.add(f"{URL_PREFIX}{url}")
    signals.add(f"{FINGERPRINT_PREFIX}{article.fingerprint}")
    return signals


def dedupe_articles(
    articles: Iterable[NewsArticle],
) -> tuple[tuple[NewsArticle, ...], tuple[str, ...]]:
    """Collapse duplicates, keeping the earliest-published copy of each.

    Returns `(unique_articles, dropped_identities)`.

    Records are grouped by any shared signal -- identity, canonical URL, or
    content fingerprint -- so a story arriving under two provider ids collapses
    to one. A record may bridge two groups that were previously separate; the
    groups then merge.

    Earliest wins because an article's first appearance is the one closest in
    spirit to what the market saw first. Ties break on article_id so the output
    is fully deterministic regardless of input order.
    """
    order = list(articles)
    parent = list(range(len(order)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    seen: dict[str, int] = {}
    for index, article in enumerate(order):
        for signal in _merge_signals(article):
            previous = seen.get(signal)
            if previous is None:
                seen[signal] = index
            else:
                union(index, previous)

    groups: dict[int, list[NewsArticle]] = {}
    for index, article in enumerate(order):
        groups.setdefault(find(index), []).append(article)

    dropped: list[str] = []
    winners: list[NewsArticle] = []
    for group in groups.values():
        winner = min(group, key=lambda a: (a.published_at, a.article_id))
        winners.append(winner)
        dropped.extend(a.article_id for a in group if a is not winner)

    unique = tuple(sorted(winners, key=lambda a: (a.published_at, a.article_id)))
    return unique, tuple(sorted(dropped))