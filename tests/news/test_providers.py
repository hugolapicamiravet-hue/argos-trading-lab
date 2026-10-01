"""Provider behaviour: configuration, URL building, redaction, retry, paging.

The provider is the only component that touches a credential. Its contract is
therefore mostly about two things:

- never write or log the API key, including the pagination cursor, which is a
  full URL carrying the key;
- obey the documented rate limit without being told to, because the usual way a
  key gets suspended is a burst from a script.

The transport is injected, so every test here runs offline with no API key.
"""

from __future__ import annotations

import datetime as dt
import io
import json
import urllib.error

import pytest

from argos.news.providers import (
    MASSIVE_PLAN_LIMITS,
    MassiveNewsProvider,
    ProviderConfigurationError,
    ProviderError,
    ProviderResponse,
    provider_for,
    redact,
    urlencode_preserving,
)

API_KEY = "test-key-do-not-log-abcdef123456"

# A realistic pagination cursor: a full URL that carries the apiKey, which is
# exactly why response copies of it must be redacted.
CURSOR = f"https://api.massive.com/v2/reference/news?cursor=2&apiKey={API_KEY}"


def page_response(*, results, next_url=None, status="OK"):
    payload = {"status": status, "results": list(results)}
    if next_url is not None:
        payload["next_url"] = next_url
    return payload


def fake_opener(pages, *, status_code=200):
    """Return an opener serving `pages` in order, recording the URLs requested."""
    calls: list[str] = []

    def opener(request, timeout=None):
        url = request.full_url if hasattr(request, "full_url") else str(request)
        calls.append(url)
        index = len(calls) - 1
        if index >= len(pages):
            raise AssertionError(f"unexpected request #{index + 1} to {url}")
        body = pages[index] if isinstance(pages[index], dict) else pages[index]
        raw = json.dumps(body).encode("utf-8") if not isinstance(body, bytes) else body
        return FakeHTTP(raw, status_code)

    opener.calls = calls
    return opener


class FakeHTTP(io.BytesIO):
    def __init__(self, body, status=200):
        super().__init__(body)
        self.status = status
        self.headers = {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


class TestConfiguration:
    def test_requires_an_api_key(self):
        with pytest.raises(ProviderConfigurationError, match="MASSIVE_API_KEY is not set"):
            MassiveNewsProvider(api_key=None, opener=lambda *a, **k: None)

    def test_reads_key_from_environment(self, monkeypatch):
        monkeypatch.setenv("MASSIVE_API_KEY", API_KEY)
        provider = MassiveNewsProvider(opener=lambda *a, **k: None)
        assert provider.api_key == API_KEY

    def test_rejects_unknown_plan(self):
        with pytest.raises(ProviderConfigurationError, match="unknown plan"):
            MassiveNewsProvider(api_key=API_KEY, plan="platinum", opener=lambda *a, **k: None)

    @pytest.mark.parametrize("plan", sorted(MASSIVE_PLAN_LIMITS))
    def test_known_plans_are_accepted(self, plan):
        provider = MassiveNewsProvider(
            api_key=API_KEY, plan=plan, opener=lambda *a, **k: None
        )
        assert provider.plan == plan

    def test_free_plan_throttles_by_default(self):
        # 5 req/min is a 12s interval. The default must throttle, or a burst is
        # discovered the only way one can be: by getting the key suspended.
        provider = MassiveNewsProvider(
            api_key=API_KEY, plan="stocks_basic", opener=lambda *a, **k: None
        )
        assert provider.min_interval == pytest.approx(12.0)

    @pytest.mark.parametrize(
        ("plan", "expected"),
        [("stocks_basic", 12.0), ("stocks_starter", 2.0), ("stocks_growth", 0.2)],
    )
    def test_interval_follows_plan(self, plan, expected):
        provider = MassiveNewsProvider(api_key=API_KEY, plan=plan, opener=lambda *a, **k: None)
        assert provider.min_interval == pytest.approx(expected)

    def test_interval_can_be_overridden(self):
        provider = MassiveNewsProvider(
            api_key=API_KEY, min_interval=0.0, opener=lambda *a, **k: None
        )
        assert provider.min_interval == 0.0

    def test_throttle_sleeps_between_requests(self):
        slept: list[float] = []
        clock = {"t": 0.0}

        def fake_sleep(seconds):
            slept.append(seconds)
            clock["t"] += seconds

        # Both pages need a cursor, or the loop exits after the first one and
        # never reaches the throttle's inter-request branch.
        pages = [
            page_response(results=[{"id": "n1"}], next_url=CURSOR),
            page_response(results=[{"id": "n2"}]),
        ]
        provider = MassiveNewsProvider(
            api_key=API_KEY,
            min_interval=10.0,
            opener=fake_opener(pages),
            sleep=fake_sleep,
        )
        # Pin monotonic so the throttle's arithmetic is observable.
        provider._last_request_at = None
        import argos.news.providers as providers_module

        original = providers_module.time.monotonic
        providers_module.time.monotonic = lambda: clock["t"]
        try:
            list(provider.iter_pages(max_pages=2))
        finally:
            providers_module.time.monotonic = original
        # The first request has no predecessor to wait for.
        assert len(slept) == 1
        assert slept[0] == pytest.approx(10.0, abs=0.01)

    def test_provider_factory_returns_massive(self):
        provider = provider_for("massive", api_key=API_KEY, opener=lambda *a, **k: None)
        assert isinstance(provider, MassiveNewsProvider)

    def test_provider_factory_rejects_unknown(self):
        with pytest.raises((ProviderConfigurationError, KeyError, ValueError)):
            provider_for("alpha_vantage", api_key=API_KEY)


class TestUrlBuilding:
    def test_includes_api_key(self):
        provider = MassiveNewsProvider(
            api_key=API_KEY, min_interval=0.0, opener=lambda *a, **k: None
        )
        assert "apiKey=" in provider._build_url({})

    def test_encodes_ticker_lists_repeatedly(self):
        # Massive takes `tickers=AAPL,MSFT`. A list must be joined, not
        # stringified as "['AAPL', 'MSFT']", which would silently match nothing.
        provider = MassiveNewsProvider(
            api_key=API_KEY, min_interval=0.0, opener=lambda *a, **k: None
        )
        url = provider._build_url({"tickers": ["AAPL", "MSFT"]})
        assert "tickers=AAPL%2CMSFT" in url

    def test_encodes_dates_as_iso(self):
        import datetime as dt

        provider = MassiveNewsProvider(
            api_key=API_KEY, min_interval=0.0, opener=lambda *a, **k: None
        )
        url = provider._build_url(
            {
                "published_utc.gte": dt.date(2025, 1, 2),
                "published_utc.lte": dt.datetime(2025, 3, 4, 5, 6),
            }
        )
        assert "2025-01-02" in url
        assert "2025-03-04" in url

    def test_query_string_is_valid(self):
        provider = MassiveNewsProvider(
            api_key=API_KEY, min_interval=0.0, opener=lambda *a, **k: None
        )
        from urllib.parse import parse_qs, urlsplit

        query = parse_qs(urlsplit(provider._build_url({"tickers": ["AAPL", "MSFT"]})).query)
        assert query["tickers"] == ["AAPL,MSFT"]

    def test_urlencode_preserving_handles_scalars_and_sequences(self):
        assert "a=1" in urlencode_preserving({"a": 1})
        assert "b=1%2C2" in urlencode_preserving({"b": [1, 2]})
        assert urlencode_preserving({"c": True}).startswith("c=")

    def test_none_values_are_omitted(self):
        assert "skipped" not in urlencode_preserving({"skipped": None, "kept": 1})


class TestRedaction:
    def test_removes_the_key(self):
        assert API_KEY not in redact(f"https://x.test/?apiKey={API_KEY}", API_KEY)

    def test_replaces_key_with_marker(self):
        assert "***" in redact(f"https://x.test/?apiKey={API_KEY}", API_KEY)

    def test_handles_no_key_configured(self):
        # Redaction must not crash before an api_key is resolved.
        assert redact("plain text", None) == "plain text"

    def test_handles_empty_input(self):
        assert redact(None, API_KEY) == ""
        assert redact("", API_KEY) == ""

    def test_catches_key_in_other_param_name(self):
        # Belt and braces: a key that reached us via a differently named
        # parameter is still scrubbed.
        assert API_KEY not in redact(f"https://x.test/?token={API_KEY}", API_KEY)

    def test_response_next_url_is_redacted(self):
        # next_url is a complete URL including apiKey, and responses get written
        # into manifests and logs.
        opener = fake_opener(
            [page_response(results=[], next_url=f"https://api.massive.com/v2/reference/news?cursor=abc&apiKey={API_KEY}")]
        )
        provider = MassiveNewsProvider(api_key=API_KEY, min_interval=0.0, opener=opener)
        response = provider.fetch_page()
        assert API_KEY not in response.next_url
        assert API_KEY not in response.request_url

    def test_response_repr_does_not_leak_the_key(self):
        opener = fake_opener(
            [page_response(results=[], next_url=f"https://x.test/?apiKey={API_KEY}")]
        )
        provider = MassiveNewsProvider(api_key=API_KEY, min_interval=0.0, opener=opener)
        assert API_KEY not in repr(provider.fetch_page())

    def test_error_payload_message_is_redacted(self):
        opener = fake_opener(
            [page_response(results=[], status="ERROR") | {"error": f"bad key {API_KEY}"}]
        )
        provider = MassiveNewsProvider(api_key=API_KEY, min_interval=0.0, opener=opener)
        with pytest.raises(ProviderError) as excinfo:
            provider.fetch_page()
        assert API_KEY not in str(excinfo.value)

    def test_http_error_detail_is_redacted(self):
        def opener(request, timeout=None):
            raise urllib.error.HTTPError(
                request.full_url, 401, f"Unauthorized: key {API_KEY}", None, None
            )

        provider = MassiveNewsProvider(
            api_key=API_KEY, min_interval=0.0, opener=opener, max_retries=0
        )
        with pytest.raises(ProviderError) as excinfo:
            provider.fetch_page()
        assert API_KEY not in str(excinfo.value)

    def test_transport_failure_message_is_redacted(self):
        def opener(request, timeout=None):
            raise OSError(f"connection to {request.full_url} failed")

        provider = MassiveNewsProvider(
            api_key=API_KEY, min_interval=0.0, opener=opener, max_retries=0
        )
        with pytest.raises(ProviderError) as excinfo:
            provider.fetch_page()
        assert API_KEY not in str(excinfo.value)


class TestFetchPage:
    def test_returns_results(self):
        opener = fake_opener([page_response(results=[{"id": "n1"}, {"id": "n2"}])])
        provider = MassiveNewsProvider(api_key=API_KEY, min_interval=0.0, opener=opener)
        response = provider.fetch_page()
        assert [r["id"] for r in response.results] == ["n1", "n2"]
        assert response.status == 200
        assert response.page == 1

    def test_passes_query_params_through(self):
        opener = fake_opener([page_response(results=[])])
        provider = MassiveNewsProvider(api_key=API_KEY, min_interval=0.0, opener=opener)
        provider.fetch_page(tickers=["AAPL"])
        assert "tickers=AAPL" in opener.calls[0]

    def test_error_status_payload_raises(self):
        opener = fake_opener([{"status": "ERROR", "error": "unknown ticker"}])
        provider = MassiveNewsProvider(api_key=API_KEY, min_interval=0.0, opener=opener)
        with pytest.raises(ProviderError, match="unknown ticker"):
            provider.fetch_page()

    def test_non_object_payload_raises(self):
        opener = fake_opener([[1, 2, 3]])
        provider = MassiveNewsProvider(api_key=API_KEY, min_interval=0.0, opener=opener)
        with pytest.raises(ProviderError, match="expected an object"):
            provider.fetch_page()

    def test_results_must_be_a_list(self):
        # A `results` dict where a list was expected means the response shape
        # changed. Iterating it would yield keys and lose every article.
        opener = fake_opener([{"status": "OK", "results": {"a": 1}}])
        provider = MassiveNewsProvider(api_key=API_KEY, min_interval=0.0, opener=opener)
        with pytest.raises(ProviderError, match="expected a list"):
            provider.fetch_page()

    def test_missing_results_is_an_empty_page(self):
        opener = fake_opener([{"status": "OK"}])
        provider = MassiveNewsProvider(api_key=API_KEY, min_interval=0.0, opener=opener)
        assert provider.fetch_page().results == ()

    def test_null_results_is_an_empty_page(self):
        opener = fake_opener([{"status": "OK", "results": None}])
        provider = MassiveNewsProvider(api_key=API_KEY, min_interval=0.0, opener=opener)
        assert provider.fetch_page().results == ()

    def test_invalid_json_raises_with_redacted_url(self):
        opener = fake_opener([b"<html>gateway error</html>"])
        provider = MassiveNewsProvider(api_key=API_KEY, min_interval=0.0, opener=opener)
        with pytest.raises(ProviderError) as excinfo:
            provider.fetch_page()
        assert API_KEY not in str(excinfo.value)


class TestRetry:
    @pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
    def test_retries_transient_statuses(self, status):
        attempts = {"n": 0}

        def opener(request, timeout=None):
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise urllib.error.HTTPError(request.full_url, status, "retry", None, None)
            return FakeHTTP(json.dumps(page_response(results=[{"id": "n1"}])).encode())

        provider = MassiveNewsProvider(
            api_key=API_KEY,
            min_interval=0.0,
            opener=opener,
            max_retries=2,
            sleep=lambda s: None,
        )
        assert provider.fetch_page().results[0]["id"] == "n1"
        assert attempts["n"] == 2  # first attempt, then one retry: it succeeded early

    @pytest.mark.parametrize("status", [400, 401, 403, 404])
    def test_does_not_retry_client_errors(self, status):
        # Retrying a bad key or a bad ticker wastes time and looks like the
        # provider is having a transient problem.
        attempts = {"n": 0}

        def opener(request, timeout=None):
            attempts["n"] += 1
            raise urllib.error.HTTPError(request.full_url, status, "no", None, None)

        provider = MassiveNewsProvider(
            api_key=API_KEY,
            min_interval=0.0,
            opener=opener,
            max_retries=3,
            sleep=lambda s: None,
        )
        with pytest.raises(ProviderError):
            provider.fetch_page()
        assert attempts["n"] == 1

    def test_gives_up_after_max_retries(self):
        attempts = {"n": 0}

        def opener(request, timeout=None):
            attempts["n"] += 1
            raise urllib.error.HTTPError(request.full_url, 503, "down", None, None)

        provider = MassiveNewsProvider(
            api_key=API_KEY,
            min_interval=0.0,
            opener=opener,
            max_retries=2,
            sleep=lambda s: None,
        )
        with pytest.raises(ProviderError):
            provider.fetch_page()
        assert attempts["n"] == 3  # one attempt plus the two requested retries

    def test_sleeps_between_retries(self):
        slept: list[float] = []

        def opener(request, timeout=None):
            if len(slept) < 2:
                raise urllib.error.HTTPError(request.full_url, 503, "down", None, None)
            return FakeHTTP(json.dumps(page_response(results=[])).encode())

        provider = MassiveNewsProvider(
            api_key=API_KEY,
            min_interval=0.0,
            opener=opener,
            max_retries=3,
            sleep=slept.append,
        )
        provider.fetch_page()
        assert len(slept) == 2
        assert all(delay >= 0 for delay in slept)

    def test_retry_count_can_be_disabled(self):
        attempts = {"n": 0}

        def opener(request, timeout=None):
            attempts["n"] += 1
            raise urllib.error.HTTPError(request.full_url, 503, "down", None, None)

        # max_retries=0 must still send one request; "no retries" is not
        # "no request".
        provider = MassiveNewsProvider(
            api_key=API_KEY, min_interval=0.0, opener=opener, max_retries=0, sleep=lambda s: None
        )
        with pytest.raises(ProviderError):
            provider.fetch_page()
        assert attempts["n"] == 1


class TestPagination:
    def test_follows_cursor_until_exhausted(self):
        cursor3 = "https://api.massive.com/v2/reference/news?cursor=3&apiKey=" + API_KEY
        opener = fake_opener(
            [
                page_response(results=[{"id": "n1"}], next_url=CURSOR),
                page_response(results=[{"id": "n2"}], next_url=cursor3),
                page_response(results=[{"id": "n3"}]),
            ]
        )
        provider = MassiveNewsProvider(api_key=API_KEY, min_interval=0.0, opener=opener)
        pages = list(provider.iter_pages(max_pages=10))
        assert [p.page for p in pages] == [1, 2, 3]
        assert [r["id"] for p in pages for r in p.results] == ["n1", "n2", "n3"]

    def test_cursor_request_carries_the_real_key(self):
        # The response copy of next_url is redacted, so pagination must follow
        # the raw cursor or the second request would authenticate as "***".
        opener = fake_opener(
            [
                page_response(results=[{"id": "n1"}], next_url=CURSOR),
                page_response(results=[{"id": "n2"}]),
            ]
        )
        provider = MassiveNewsProvider(api_key=API_KEY, min_interval=0.0, opener=opener)
        list(provider.iter_pages(max_pages=2))
        assert "cursor=2" in opener.calls[1]
        assert API_KEY in opener.calls[1]
        assert "***" not in opener.calls[1]

    def test_max_pages_caps_the_loop(self):
        # An unbounded loop against a paginated endpoint is how a rate-limited
        # key becomes a rate-limited account.
        cursor = f"https://api.massive.com/v2/reference/news?cursor=x&apiKey={API_KEY}"
        endless = [page_response(results=[{"id": "n"}], next_url=cursor) for _ in range(50)]
        opener = fake_opener(endless)
        provider = MassiveNewsProvider(api_key=API_KEY, min_interval=0.0, opener=opener)
        pages = list(provider.iter_pages(max_pages=3))
        assert len(pages) == 3

    def test_rejects_zero_max_pages(self):
        provider = MassiveNewsProvider(
            api_key=API_KEY, min_interval=0.0, opener=lambda *a, **k: None
        )
        with pytest.raises(ProviderConfigurationError, match="max_pages must be"):
            list(provider.iter_pages(max_pages=0))

    def test_fetch_all_flattens(self):
        opener = fake_opener(
            [
                page_response(results=[{"id": "n1"}], next_url=CURSOR),
                page_response(results=[{"id": "n2"}]),
            ]
        )
        provider = MassiveNewsProvider(api_key=API_KEY, min_interval=0.0, opener=opener)
        results, pages = provider.fetch_all(max_pages=5)
        assert [r["id"] for r in results] == ["n1", "n2"]
        assert pages == 2

    def test_request_count_is_tracked(self):
        opener = fake_opener(
            [page_response(results=[], next_url=CURSOR), page_response(results=[])]
        )
        provider = MassiveNewsProvider(api_key=API_KEY, min_interval=0.0, opener=opener)
        provider.fetch_all(max_pages=2)
        assert provider.request_count == 2


class TestProviderResponse:
    def make(self, **overrides) -> ProviderResponse:
        kwargs = {
            "results": (),
            "next_url": None,
            "status": 200,
            "request_url": "https://example.test/",
            "fetched_at": dt.datetime(2026, 1, 1, tzinfo=dt.UTC),
        }
        return ProviderResponse(**{**kwargs, **overrides})

    def test_is_immutable(self):
        import dataclasses

        response = self.make()
        with pytest.raises(dataclasses.FrozenInstanceError):
            response.status = 500

    def test_results_are_a_tuple(self):
        # A list here would be mutable shared state carried across pages.
        assert isinstance(self.make(results=({"id": "n1"},)).results, tuple)

    def test_fetched_at_is_aware(self):
        assert self.make().fetched_at.tzinfo is not None