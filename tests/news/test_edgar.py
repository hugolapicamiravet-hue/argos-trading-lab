"""EDGAR as a point-in-time corporate event source.

The claim EDGAR makes is narrow and testable: a filing became public at
`acceptanceDateTime`, and nothing earlier. Everything here is offline -- the
opener is injected, so the suite never touches sec.gov -- but the payloads are
the real column-per-field shape rather than something convenient, because the
transpose is where a wrong implementation hides.
"""

from __future__ import annotations

import datetime as dt
import json
import urllib.error

import pytest

from argos.news.edgar import (
    ARCHIVES_BASE,
    FAIR_ACCESS_MAX_REQUESTS_PER_SECOND,
    RELEVANT_FORMS,
    EdgarClient,
    EdgarConfigurationError,
    EdgarError,
    Filing,
    filing_to_event_type,
)
from argos.news.models import UTC

CONTACT = "argos-trading-lab you@example.com"


def submissions_payload(rows: list[dict], *, length: int | None = None) -> dict:
    """Build the real `filings.recent` layout: parallel arrays plus `length`."""
    names = sorted({key for row in rows for key in row})
    block: dict = {name: [row.get(name) for row in rows] for name in names}
    block["length"] = len(rows) if length is None else length
    return {"filings": {"recent": block}}


def sample_row(**overrides) -> dict:
    row = {
        "accessionNumber": "0000320193-25-000006",
        "form": "8-K",
        "acceptanceDateTime": "2025-01-30T21:30:00.000Z",
        "filingDate": "2025-01-30",
        "reportDate": "2025-01-29",
        "primaryDocument": "aapl-20250130.htm",
        "primaryDocDescription": "8-K",
        "items": "1.01,2.02",
    }
    row.update(overrides)
    return row


class FakeResponse:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


def opener_returning(payload: object, *, log: list | None = None):
    body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()

    def opener(request, timeout=None):
        if log is not None:
            log.append((request.full_url, dict(request.headers)))
        return FakeResponse(body)

    return opener


def client(payload: object, **kwargs) -> EdgarClient:
    kwargs.setdefault("opener", opener_returning(payload))
    kwargs.setdefault("sleep", lambda _s: None)
    return EdgarClient(user_agent=CONTACT, **kwargs)


class TestUserAgentPolicy:
    def test_a_descriptive_user_agent_is_accepted(self):
        assert EdgarClient(user_agent=CONTACT).request_count == 0

    def test_the_default_user_agent_is_refused(self):
        # The default is a placeholder. Accepting it would mean a client ran
        # against sec.gov without contact information and got an opaque 403
        # instead of a sentence explaining what to fix.
        with pytest.raises(EdgarConfigurationError):
            EdgarClient()

    def test_a_user_agent_naming_no_contact_is_refused(self):
        with pytest.raises(EdgarConfigurationError):
            EdgarClient(user_agent="argos-trading-lab/0.1")

    def test_the_user_agent_actually_reaches_the_request(self):
        log: list = []
        client(submissions_payload([sample_row()]), opener=opener_returning(
            submissions_payload([sample_row()]), log=log
        )).submissions("320193")
        assert log, "no request was made"
        assert CONTACT in str(log[0][1])


class TestAcceptanceTimeParsing:
    """The one field the whole module exists to get right."""

    def test_a_z_suffixed_timestamp_is_read_as_utc(self):
        filing = Filing.from_submission(cik="1", record=sample_row())
        assert filing.acceptance_datetime == dt.datetime(2025, 1, 30, 21, 30, tzinfo=UTC)

    def test_an_explicit_offset_is_preserved_not_relabelled(self):
        # EDGAR has published this field with an offset. Relabelling 16:30-05:00
        # as 16:30Z would place the filing five hours before it was public, and
        # a backtest would act on it before the SEC released it.
        filing = Filing.from_submission(
            cik="1", record=sample_row(acceptanceDateTime="2025-01-30T16:30:00-05:00")
        )
        assert filing.acceptance_datetime == dt.datetime(2025, 1, 30, 21, 30, tzinfo=UTC)

    def test_a_naive_timestamp_is_assumed_utc(self):
        filing = Filing.from_submission(
            cik="1", record=sample_row(acceptanceDateTime="2025-01-30T21:30:00")
        )
        assert filing.acceptance_datetime == dt.datetime(2025, 1, 30, 21, 30, tzinfo=UTC)

    def test_a_missing_acceptance_time_yields_no_filing(self):
        assert Filing.from_submission(cik="1", record=sample_row(acceptanceDateTime="")) is None
        assert Filing.from_submission(cik="1", record={"form": "8-K"}) is None

    def test_an_unparseable_acceptance_time_yields_no_filing(self):
        assert (
            Filing.from_submission(cik="1", record=sample_row(acceptanceDateTime="not-a-date"))
            is None
        )

    def test_report_date_never_gates_availability(self):
        # A 10-Q filed 2025-01-30 describing 2024-12-31 is available on the
        # 30th. Using reportDate would move it a month into the past.
        filing = Filing.from_submission(
            cik="1",
            record=sample_row(form="10-Q", filingDate="2025-01-30", reportDate="2024-12-31"),
        )
        assert filing.available_at.date() == dt.date(2025, 1, 30)
        assert filing.report_date == dt.date(2024, 12, 31)


class TestFieldParsing:
    def test_items_are_split_and_stripped(self):
        filing = Filing.from_submission(cik="1", record=sample_row(items=" 1.01 , 2.02 "))
        assert filing.items == ("1.01", "2.02")

    def test_absent_items_become_an_empty_tuple(self):
        assert Filing.from_submission(cik="1", record=sample_row(items=None)).items == ()

    def test_missing_optional_fields_become_none(self):
        filing = Filing.from_submission(
            cik="1",
            record=sample_row(primaryDocument="", primaryDocDescription="   ", reportDate=""),
        )
        assert filing.primary_document is None
        assert filing.primary_doc_description is None
        assert filing.report_date is None

    def test_the_primary_document_url_uses_edgars_own_layout(self):
        filing = Filing.from_submission(
            cik="0000320193", record=sample_row(primaryDocument="aapl-20250130.htm")
        )
        assert filing.primary_document_url == (
            f"{ARCHIVES_BASE}/320193/000032019325000006/aapl-20250130.htm"
        )

    def test_no_primary_document_means_no_url(self):
        filing = Filing.from_submission(cik="1", record=sample_row(primaryDocument=""))
        assert filing.primary_document_url is None

    def test_round_tripping_to_dict_keeps_the_availability_time(self):
        filing = Filing.from_submission(cik="1", record=sample_row())
        payload = filing.to_dict()
        assert payload["acceptance_datetime"] == "2025-01-30T21:30:00Z"
        assert json.loads(json.dumps(payload)) == payload


class TestFilings:
    def test_only_relevant_forms_come_back(self):
        rows = [
            sample_row(form="8-K"),
            sample_row(form="4", acceptanceDateTime="2025-01-30T21:31:00.000Z"),
            sample_row(form="10-Q", acceptanceDateTime="2025-01-30T21:32:00.000Z"),
        ]
        found = client(submissions_payload(rows)).filings("320193")
        assert sorted(f.form for f in found) == ["10-Q", "8-K"]

    def test_forms_are_matched_case_insensitively(self):
        rows = [sample_row(form="8-k")]
        assert len(client(submissions_payload(rows)).filings("320193")) == 1

    def test_the_default_form_set_excludes_ownership_reports(self):
        assert "4" not in RELEVANT_FORMS
        assert "13F-HR" not in RELEVANT_FORMS

    def test_filings_come_back_newest_first(self):
        rows = [
            sample_row(acceptanceDateTime="2025-01-30T21:30:00.000Z", accessionNumber="a"),
            sample_row(acceptanceDateTime="2025-02-03T21:30:00.000Z", accessionNumber="b"),
            sample_row(acceptanceDateTime="2025-01-15T21:30:00.000Z", accessionNumber="c"),
        ]
        found = client(submissions_payload(rows)).filings("320193")
        assert [f.accession_number for f in found] == ["b", "a", "c"]

    def test_since_filters_on_acceptance_time_not_filing_date(self):
        # A filing accepted on the 30th for the 29th must survive `since` at the
        # 29th 23:59, because the public had it. Filtering on filingDate would
        # drop it; filtering on reportDate would be worse.
        rows = [sample_row(acceptanceDateTime="2025-01-30T21:30:00.000Z", filingDate="2025-01-30")]
        found = client(submissions_payload(rows)).filings(
            "320193", since=dt.datetime(2025, 1, 29, 23, 59, tzinfo=UTC)
        )
        assert len(found) == 1

    def test_limit_truncates_after_sorting(self):
        rows = [
            sample_row(acceptanceDateTime="2025-01-15T21:30:00.000Z"),
            sample_row(acceptanceDateTime="2025-02-03T21:30:00.000Z"),
            sample_row(acceptanceDateTime="2025-01-20T21:30:00.000Z"),
        ]
        found = client(submissions_payload(rows)).filings("320193", limit=2)
        assert len(found) == 2
        assert found[0].acceptance_datetime > found[1].acceptance_datetime

    def test_a_naive_since_is_refused(self):
        with pytest.raises(ValueError, match="naive"):
            client(submissions_payload([sample_row()])).filings(
                "320193", since=dt.datetime(2025, 1, 1)
            )

    def test_rows_without_acceptance_times_are_skipped_not_fatal(self):
        rows = [sample_row(), sample_row(accessionNumber="b", acceptanceDateTime="")]
        assert len(client(submissions_payload(rows)).filings("320193")) == 1

    def test_the_cik_is_zero_padded_in_the_request_url(self):
        log: list = []
        client(submissions_payload([sample_row()]), opener=opener_returning(
            submissions_payload([sample_row()]), log=log
        )).submissions("320193")
        assert "CIK0000320193.json" in log[0][0]

    def test_a_cik_given_as_a_padded_string_works_too(self):
        log: list = []
        c = EdgarClient(
            user_agent=CONTACT,
            opener=opener_returning(submissions_payload([sample_row()]), log=log),
            sleep=lambda _s: None,
        )
        c.submissions("0000320193")
        assert "CIK0000320193.json" in log[0][0]


class TestMalformedPayloads:
    def test_a_missing_filings_block_is_an_error(self):
        with pytest.raises(EdgarError, match="filings"):
            client({"name": "APPLE"}).filings("320193")

    def test_a_missing_recent_block_is_an_error(self):
        with pytest.raises(EdgarError, match="filings.recent"):
            client({"filings": {"files": []}}).filings("320193")

    def test_a_column_shorter_than_length_is_an_error(self):
        # Guessing here would silently mis-pair a form with a timestamp.
        payload = submissions_payload([sample_row(), sample_row()])
        payload["filings"]["recent"]["form"] = ["8-K"]
        with pytest.raises(EdgarError, match="rows"):
            client(payload).filings("320193")

    def test_a_missing_length_marker_is_an_error(self):
        payload = submissions_payload([sample_row()])
        del payload["filings"]["recent"]["length"]
        with pytest.raises(EdgarError, match="length"):
            client(payload).filings("320193")

    def test_a_non_json_body_is_an_error(self):
        with pytest.raises(EdgarError, match="non-JSON"):
            client(b"<html>nope</html>").submissions("320193")

    def test_an_http_error_becomes_an_edgar_error(self):
        def opener(request, timeout=None):
            raise urllib.error.HTTPError(request.full_url, 403, "Forbidden", {}, None)

        with pytest.raises(EdgarError, match="403"):
            EdgarClient(user_agent=CONTACT, opener=opener, sleep=lambda _s: None).submissions(
                "320193"
            )

    def test_a_connection_error_becomes_an_edgar_error(self):
        def opener(request, timeout=None):
            raise urllib.error.URLError("no route")

        with pytest.raises(EdgarError, match="failed"):
            EdgarClient(user_agent=CONTACT, opener=opener, sleep=lambda _s: None).submissions(
                "320193"
            )

    def test_a_non_object_payload_is_an_error(self):
        with pytest.raises(EdgarError, match="list"):
            client([1, 2, 3]).submissions("320193")


class TestTickerIndex:
    INDEX = {
        "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
        "1": {"cik_str": 789019, "ticker": "MSFT", "title": "Microsoft Corp"},
        "2": {"cik_str": None, "ticker": "ORPHAN", "title": "No CIK"},
        "3": "not a mapping",
    }

    def test_tickers_map_to_ciks(self):
        assert client(self.INDEX).company_tickers() == {"AAPL": "320193", "MSFT": "789019"}

    def test_an_entry_without_a_cik_is_skipped(self):
        assert "ORPHAN" not in client(self.INDEX).company_tickers()

    def test_ticker_lookup_is_case_insensitive_and_pads_the_cik(self):
        assert client(self.INDEX).cik_for_ticker(" aapl ") == "0000320193"

    def test_an_unknown_ticker_is_an_error(self):
        with pytest.raises(EdgarError, match="not found"):
            client(self.INDEX).cik_for_ticker("NOSUCH")


class TestFairAccess:
    def test_the_default_interval_matches_the_sec_limit(self):
        assert client({}).min_interval == pytest.approx(1.0 / FAIR_ACCESS_MAX_REQUESTS_PER_SECOND)

    def test_consecutive_requests_are_spaced(self):
        import time as real_time

        waits: list[float] = []
        original = real_time.monotonic
        clock = {"now": 0.0}
        real_time.monotonic = lambda: clock["now"]  # type: ignore[assignment]
        try:
            c = EdgarClient(
                user_agent=CONTACT,
                opener=opener_returning(submissions_payload([sample_row()])),
                sleep=lambda s: (waits.append(s), clock.__setitem__("now", clock["now"] + s)),
            )
            c.submissions("1")
            clock["now"] = 0.001
            c.submissions("1")
        finally:
            real_time.monotonic = original  # type: ignore[assignment]
        assert waits and waits[0] == pytest.approx(c.min_interval - 0.001)

    def test_the_request_counter_tracks_every_call(self):
        c = client(submissions_payload([sample_row()]))
        c.submissions("1")
        c.submissions("1")
        assert c.request_count == 2


class TestFilingToEventType:
    def test_known_forms_are_labelled(self):
        assert filing_to_event_type("10-Q") == "periodic_report"
        assert filing_to_event_type("10-K") == "periodic_report"
        assert filing_to_event_type("8-K") == "current_report"

    def test_anything_else_is_other(self):
        assert filing_to_event_type("S-1") == "other"