"""Provider clients for news data.

Only providers whose official documentation was read are implemented. Alpha
Vantage's NEWS_SENTIMENT was evaluated and deliberately left out: its response
schema is not documented at the field level, so building a client for it would
mean writing a parser against assumptions rather than documentation. That
violates the project's rule of recording UNKNOWN rather than guessing, so
Massive is the sole news provider.

Massive specifics that shaped the client:

  * `GET https://api.massive.com/v2/reference/news`, key in the `apiKey` query
    parameter.
  * `published_utc` is RFC3339 with an explicit `Z`, so no zone inference is
    needed on the happy path.
  * Pagination is by `next_url`, an absolute URL returned in the response. Note
    that this URL can itself embed the API key, which is why `redact` exists and
    is applied to every URL this module records or logs.
  * `limit` maxes at 1000.
  * Free tier ("Stocks Basic") allows 5 requests/minute and only two years of
    news history; broader history starts on paid plans. A backtest over the
    2020-2025 baseline cannot be promised on the free tier, and the client does
    not pretend otherwise.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from argos.news.models import UTC, _require_aware

MASSIVE_NEWS_URL = "https://api.massive.com/v2/reference/news"
DEFAULT_LIMIT = 100
MAX_LIMIT = 1000
DEFAULT_TIMEOUT = 30.0
DEFAULT_RETRIES = 3
RETRY_BACKOFF_SECONDS = 1.0
USER_AGENT = "argos-trading-lab/0.1 (educational backtesting lab)"

# Plan ceilings, from Massive's published pricing documentation. Recorded so the
# manifest can state what the fetched batch is actually allowed to contain.
MASSIVE_PLAN_LIMITS: Mapping[str, Mapping[str, object]] = {
    "stocks_basic": {
        "requests_per_minute": 5,
        "news_history_years": 2,
        "history_since": None,
    },
    "stocks_starter": {
        "requests_per_minute": 30,
        "news_history_years": None,
        "history_since": "2016-06-22",
    },
    "stocks_growth": {
        "requests_per_minute": 300,
        "news_history_years": None,
        "history_since": "2016-06-22",
    },
    "business": {
        "requests_per_minute": 500,
        "news_history_years": None,
        "history_since": "2016-06-22",
    },
}


class ProviderError(RuntimeError):
    """Provider request failed after the configured retries."""


class ProviderConfigurationError(RuntimeError):
    """Provider is unusable: missing key, bad arguments, unsupported plan."""


def redact(text: str | None, api_key: str | None) -> str:
    """Remove an API key from any string before it is stored or logged.

    The `next_url` cursor can contain the key, so error messages, manifest
    entries and log lines all pass through here. Cheap insurance against the most
    boring and most expensive mistake in this project: committing a live key.
    """
    if not text:
        return ""
    cleaned = str(text)
    if api_key:
        cleaned = cleaned.replace(api_key, "***")
    # Belt and braces, in case a key reached us through a differently-named
    # parameter or was only partially interpolated.
    cleaned = _API_KEY_PARAM.sub("apiKey=***", cleaned)
    return cleaned


_API_KEY_PARAM = re.compile(r"(apiKey=|apikey=)[^&\s\"']+", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class ProviderResponse:
    """One page of provider output, kept raw and uninterpreted.

    `results` are the provider's own dicts, untouched. Normalization is a
    separate step so the raw layer remains a faithful record of what was
    received, and `next_url` is preserved so a fetch can be resumed exactly.
    """

    results: tuple[Mapping[str, object], ...]
    next_url: str | None
    status: int
    request_url: str
    fetched_at: dt.datetime
    page: int = 1
    params: Mapping[str, object] = field(default_factory=dict)

    @property
    def count(self) -> int:
        return len(self.results)

    @property
    def has_more(self) -> bool:
        return bool(self.next_url)


class NewsProvider(Protocol):
    """The narrow interface `normalize_articles` and the CLI depend on."""

    name: str

    def fetch_page(self, **params: object) -> ProviderResponse: ...

    def iter_pages(self, **params: object) -> Iterator[ProviderResponse]: ...


class MassiveNewsProvider:
    """Client for Massive's `/v2/reference/news` endpoint.

    Deliberately thin: it validates arguments, enforces the documented rate
    limit with a minimum-interval sleep, follows `next_url` up to a caller-set
    page ceiling, and returns raw dicts. It performs no interpretation.
    """

    name = "massive"

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str = MASSIVE_NEWS_URL,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_RETRIES,
        min_interval: float | None = None,
        plan: str = "stocks_basic",
        opener: object | None = None,
        sleep: object | None = None,
    ) -> None:
        self.api_key = api_key or os.environ.get("MASSIVE_API_KEY")
        if not self.api_key:
            raise ProviderConfigurationError(
                "MASSIVE_API_KEY is not set. Export it or pass api_key=...; the key is never "
                "written to disk and never appears in logs."
            )
        self.base_url = base_url
        self.timeout = float(timeout)
        self.max_retries = int(max_retries)
        self.plan = plan
        if plan not in MASSIVE_PLAN_LIMITS:
            raise ProviderConfigurationError(
                f"unknown plan {plan!r}; known plans: {sorted(MASSIVE_PLAN_LIMITS)}"
            )
        # 5 req/min on the free tier is a 12-second minimum interval. Defaulting
        # to the free plan means an accidental burst is throttled by default
        # rather than discovered by getting rate-limited. Callers may still pass
        # min_interval=0.0 to opt out, but they have to say so.
        rpm = MASSIVE_PLAN_LIMITS[plan]["requests_per_minute"]
        self.min_interval = (60.0 / rpm) if min_interval is None else float(min_interval)
        self._opener = opener or urllib.request.urlopen
        self._sleep = sleep or time.sleep
        self._last_request_at: float | None = None
        self.request_count = 0

    # -- request construction ------------------------------------------------

    def _build_url(self, params: Mapping[str, object], page: int = 1) -> str:
        query: dict[str, object] = {"apiKey": self.api_key, "limit": DEFAULT_LIMIT}
        for key, value in params.items():
            if value is None:
                continue
            query[key] = value

        limit = query.get("limit")
        if not isinstance(limit, int) or limit < 1 or limit > MAX_LIMIT:
            raise ProviderConfigurationError(
                f"limit must be an integer in [1, {MAX_LIMIT}] per the Massive docs, got {limit!r}"
            )

        parts = urllib.parse.urlsplit(self.base_url)
        merged = dict(urllib.parse.parse_qsl(parts.query))
        merged.update(query)
        return urllib.parse.urlunsplit(
            (parts.scheme, parts.netloc, parts.path, urlencode_preserving(merged), "")
        )

    def _throttle(self) -> None:
        if self.min_interval <= 0:
            return
        now = time.monotonic()
        if self._last_request_at is not None:
            wait = self.min_interval - (now - self._last_request_at)
            if wait > 0:
                self._sleep(wait)
        self._last_request_at = time.monotonic()

    def _request_json(self, url: str, page: int) -> tuple[Mapping[str, object], int]:
        last_error: Exception | None = None
        # `max_retries` counts retries, so the loop makes one attempt plus that
        # many. Treating it as a total attempt budget would make max_retries=0
        # mean "never send the request", which is not what disabling retries
        # should mean.
        max_attempts = self.max_retries + 1
        for attempt in range(1, max_attempts + 1):
            self._throttle()
            self.request_count += 1
            request = urllib.request.Request(
                url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"}
            )
            try:
                with self._opener(request, timeout=self.timeout) as response:
                    status = int(getattr(response, "status", 200) or 200)
                    body = response.read()
                return _decode_json(body, url, self.api_key, provider_name=self.name), status
            except urllib.error.HTTPError as exc:
                # 4xx is deterministic: retrying the same request only burns rate
                # limit. 429 and 5xx are transient and worth another attempt.
                if exc.code not in (429, 500, 502, 503, 504):
                    detail = _safe_error_detail(exc, self.api_key)
                    raise ProviderError(
                        f"{self.name} request failed with HTTP {exc.code}: {detail}"
                    ) from exc
                last_error = exc
            except (TimeoutError, urllib.error.URLError, OSError) as exc:
                last_error = exc

            if attempt < max_attempts:
                self._sleep(RETRY_BACKOFF_SECONDS * attempt)

        raise ProviderError(
            f"{self.name} request failed after {max_attempts} attempts for "
            f"{redact(url, self.api_key)}: {redact(str(last_error), self.api_key)}"
        )

    # -- public API ----------------------------------------------------------

    def fetch_page(self, **params: object) -> ProviderResponse:
        """Fetch exactly one page, without following `next_url`."""
        url = self._build_url(params, page=1)
        payload, status = self._request_json(url, page=1)
        return self._to_response(payload, status, url, 1, params)

    def iter_pages(
        self, *, max_pages: int = 10, **params: object
    ) -> Iterator[ProviderResponse]:
        """Yield pages, following `next_url` until exhausted or `max_pages`.

        The page ceiling is a safety rail, not a business rule: an unbounded
        loop against a paginated endpoint is how a rate-limited key becomes a
        rate-limited account. Callers that need everything must ask for it
        explicitly.
        """
        if max_pages < 1:
            raise ProviderConfigurationError(f"max_pages must be >= 1, got {max_pages}")

        url = self._build_url(params, page=1)
        page = 0
        while url and page < max_pages:
            payload, status = self._request_json(url, page=page + 1)
            response = self._to_response(payload, status, url, page + 1, params)
            yield response
            page += 1
            # Follow the cursor out of the raw payload, not
            # `response.next_url`. The response copy is redacted for storage and
            # logging, and the redaction replaces the apiKey with "***", which
            # the next request would then send as its credential.
            cursor = payload.get("next_url")
            url = str(cursor) if isinstance(cursor, str) and cursor else ""

    def fetch_all(
        self, *, max_pages: int = 10, **params: object
    ) -> tuple[tuple[Mapping[str, object], ...], int]:
        """Convenience: flatten pages into one result tuple plus the page count."""
        results: list[Mapping[str, object]] = []
        pages = 0
        for response in self.iter_pages(max_pages=max_pages, **params):
            results.extend(response.results)
            pages += 1
        return tuple(results), pages

    def _to_response(
        self,
        payload: Mapping[str, object],
        status: int,
        url: str,
        page: int,
        params: Mapping[str, object],
    ) -> ProviderResponse:
        if not isinstance(payload, Mapping):
            raise ProviderError(
                f"{self.name} returned {type(payload).__name__} where an object was expected"
            )
        if payload.get("status") == "ERROR" or "error" in payload:
            message = payload.get("error") or payload.get("message") or "unknown error"
            raise ProviderError(
                f"{self.name} returned an error payload: {redact(str(message), self.api_key)}"
            )

        raw_results = payload.get("results", [])
        if raw_results is None:
            raw_results = []
        if not isinstance(raw_results, Sequence) or isinstance(raw_results, (str, bytes)):
            raise ProviderError(
                f"{self.name} returned 'results' as {type(raw_results).__name__}; expected a list"
            )

        next_url = payload.get("next_url")
        return ProviderResponse(
            results=tuple(raw_results),
            # Redacted like request_url. The cursor is a full URL including its
            # apiKey, and this object gets written into manifests and logs, so
            # leaving it raw would put a live key in version control.
            next_url=(
                redact(str(next_url), self.api_key)
                if isinstance(next_url, str) and next_url
                else None
            ),
            status=status,
            request_url=redact(url, self.api_key),
            fetched_at=dt.datetime.now(UTC),
            page=page,
            params=dict(params),
        )


def urlencode_preserving(query: Mapping[str, object]) -> str:
    """Encode a query string without mangling ISO timestamps into `+` or `%20`.

    `urlencode`'s default `quote_via=quote_plus` turns a space into `+`, which
    some providers decode differently than expected for date filters. Using
    explicit quoting keeps the value byte-comparable to what we intended.

    Sequences are joined with commas because that is how providers express
    multi-valued filters: Massive wants `tickers=AAPL,MSFT`. Plain
    `str()` on a list would send `tickers=['AAPL', 'MSFT']`, which is not a
    syntax error, is not rejected, and matches nothing. A `--tickers AAPL MSFT`
    that returns zero articles with a 200 is a failure nobody notices.
    """
    parts: list[str] = []
    for key, value in query.items():
        if value is None:
            # An unset filter must be absent, not the literal string "None",
            # which is a value the provider will happily accept and ignore.
            continue
        if isinstance(value, bool):
            text = "true" if value else "false"
        elif isinstance(value, (str, bytes)):
            text = value.decode() if isinstance(value, bytes) else value
        elif isinstance(value, Sequence):
            text = ",".join(str(item) for item in value)
        else:
            text = str(value)
        parts.append(f"{urllib.parse.quote(str(key))}={urllib.parse.quote(text)}")
    return "&".join(parts)


def _decode_json(
    body: bytes, url: str, api_key: str | None, *, provider_name: str = "provider"
) -> Mapping[str, object]:
    try:
        decoded = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProviderError(
            f"{provider_name} returned a non-JSON body from {redact(url, api_key)}: {exc}"
        ) from exc
    if not isinstance(decoded, Mapping):
        raise ProviderError(
            f"{provider_name} returned a JSON {type(decoded).__name__}, expected an object"
        )
    return decoded


def _safe_error_detail(exc: urllib.error.HTTPError, api_key: str | None) -> str:
    try:
        body = exc.read().decode("utf-8", errors="replace")[:500]
    except Exception:  # noqa: BLE001 - best-effort diagnostics must not mask the error
        body = ""
    return redact(body.strip() or exc.reason or "no detail", api_key)


def provider_for(name: str, **kwargs: object) -> NewsProvider:
    """Construct a provider by name.

    Only `massive` is available, and that is intentional: the phase requires
    every integration to rest on read official documentation, and Alpha Vantage
    does not document its NEWS_SENTIMENT response schema at field level.
    """
    key = str(name).strip().lower()
    if key == "massive":
        return MassiveNewsProvider(**kwargs)  # type: ignore[arg-type]
    raise ProviderConfigurationError(
        f"unknown news provider {name!r}; implemented providers: ['massive']. Alpha Vantage was "
        "evaluated but not implemented because its NEWS_SENTIMENT response schema is undocumented."
    )


def require_aware_timestamp(value: dt.datetime, name: str) -> dt.datetime:
    """Re-exported timestamp guard, so callers need not import from models."""
    return _require_aware(value, name)