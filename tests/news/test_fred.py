"""FRED and ALFRED as vintage-aware macro data.

The single property this module has to get right: a macro value may only be used
once the vintage that produced it existed. Everything else is plumbing, but the
plumbing is what decides whether the vintage reaches the caller at all, so it is
tested too.

Offline throughout -- `parse_observations` is static and takes recorded payloads,
and the HTTP layer is exercised through an injected opener.
"""

from __future__ import annotations

import datetime as dt
import json
import urllib.error

import pytest

from argos.news.fred import (
    ALFRED_SERIES_URL,
    FredClient,
    FredConfigurationError,
    FredError,
    MacroObservation,
    check_frequency,
)
from argos.news.models import UTC

KEY = "test-key-not-real"
VINTAGE_START = dt.date(2025, 1, 1)
VINTAGE_END = dt.date(2025, 1, 31)


def payload(rows: list[dict]) -> dict:
    return {"observations": rows}


CPI = payload(
    [
        {"date": "2025-01-01", "value": "318.6"},
        {"date": "2025-02-01", "value": "319.1"},
        {"date": "2025-03-01", "value": "."},
        {"date": "2025-04-01", "value": "320.4"},
    ]
)


class FakeResponse:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


def client(body: object = None, *, log: list | None = None, key: str | None = KEY) -> FredClient:
    if body is None:
        raw = json.dumps(CPI).encode()
    elif isinstance(body, bytes):
        raw = body
    else:
        raw = json.dumps(body).encode()

    def opener(request, timeout=None):
        if log is not None:
            log.append(request.full_url)
        return FakeResponse(raw)

    return FredClient(api_key=key, opener=opener)


def obs(day: int, value: float, *, start=VINTAGE_START, end=VINTAGE_END) -> MacroObservation:
    return MacroObservation(
        series_id="CPIAUCSL",
        observation_date=dt.date(2025, 1, day),
        value=value,
        vintage_start=start,
        vintage_end=end,
    )


class TestConfiguration:
    def test_a_missing_key_is_refused_with_an_actionable_message(self, monkeypatch):
        monkeypatch.delenv("FRED_API_KEY", raising=False)
        with pytest.raises(FredConfigurationError, match="FRED_API_KEY"):
            FredClient()

    def test_the_key_is_read_from_the_environment(self, monkeypatch):
        monkeypatch.setenv("FRED_API_KEY", "from-env")
        assert FredClient().api_key == "from-env"

    def test_an_explicit_key_beats_the_environment(self, monkeypatch):
        monkeypatch.setenv("FRED_API_KEY", "from-env")
        assert FredClient(api_key="explicit").api_key == "explicit"


class TestQueryConstruction:
    def test_the_realtime_window_is_sent_as_a_closed_interval(self):
        log: list = []
        c = client(log=log)
        c.fetch_observations("CPIAUCSL", realtime_start=VINTAGE_START, realtime_end=VINTAGE_END)
        assert f"realtime_start={VINTAGE_START}" in log[0]
        assert f"realtime_end={VINTAGE_END}" in log[0]

    def test_alfred_is_the_default_endpoint(self):
        log: list = []
        client(log=log).fetch_observations(
            "CPIAUCSL", realtime_start=VINTAGE_START, realtime_end=VINTAGE_END
        )
        assert log[0].startswith(ALFRED_SERIES_URL)

    def test_observation_bounds_are_optional(self):
        log: list = []
        client(log=log).fetch_observations(
            "CPIAUCSL", realtime_start=VINTAGE_START, realtime_end=VINTAGE_END
        )
        assert "observation_start" not in log[0]

        log.clear()
        client(log=log).fetch_observations(
            "CPIAUCSL",
            realtime_start=VINTAGE_START,
            realtime_end=VINTAGE_END,
            observation_start=dt.date(2020, 1, 1),
            observation_end=dt.date(2025, 1, 1),
        )
        assert "observation_start=2020-01-01" in log[0]
        assert "observation_end=2025-01-01" in log[0]

    def test_a_reversed_realtime_window_is_refused(self):
        with pytest.raises(FredConfigurationError, match="must not precede"):
            client().fetch_observations(
                "CPIAUCSL", realtime_start=VINTAGE_END, realtime_end=VINTAGE_START
            )

    def test_an_empty_series_id_is_refused(self):
        with pytest.raises(FredConfigurationError, match="series_id"):
            client().fetch_observations(
                "", realtime_start=VINTAGE_START, realtime_end=VINTAGE_END
            )

    def test_the_realtime_window_cannot_be_omitted(self):
        # A defaulted window would make every query silently look-ahead, so the
        # parameters are keyword-only with no default at all.
        with pytest.raises(TypeError):
            client().fetch_observations("CPIAUCSL")

    def test_the_api_key_never_appears_in_an_error_message(self, monkeypatch):
        monkeypatch.delenv("FRED_API_KEY", raising=False)
        with pytest.raises(FredConfigurationError) as caught:
            FredClient().fetch_observations(
                "CPIAUCSL", realtime_start=VINTAGE_START, realtime_end=VINTAGE_END
            )
        assert "FRED_API_KEY is not set" in str(caught.value)
        # The variable name may appear; a value must not.
        assert KEY not in str(caught.value)

    def test_a_transport_failure_does_not_echo_the_key(self):
        def opener(request, timeout=None):
            raise urllib.error.URLError("no route")

        c = FredClient(api_key="super-secret-value", opener=opener)
        with pytest.raises(FredError) as caught:
            c.fetch_observations(
                "CPIAUCSL", realtime_start=VINTAGE_START, realtime_end=VINTAGE_END
            )
        assert "super-secret-value" not in str(caught.value)


class TestParseObservations:
    def test_values_are_parsed_in_date_order(self):
        got = FredClient.parse_observations(
            CPI, series_id="CPIAUCSL", vintage_start=VINTAGE_START, vintage_end=VINTAGE_END
        )
        assert [o.observation_date for o in got] == [
            dt.date(2025, 1, 1),
            dt.date(2025, 2, 1),
            dt.date(2025, 4, 1),
        ]
        assert got[0].value == 318.6

    def test_a_missing_period_is_skipped_not_read_as_zero(self):
        # FRED writes "." for no data. Coercing that to 0.0 would put a real
        # zero into a series that never published one.
        got = FredClient.parse_observations(
            CPI, series_id="CPIAUCSL", vintage_start=VINTAGE_START, vintage_end=VINTAGE_END
        )
        assert 0.0 not in [o.value for o in got]
        assert all(o.value > 0 for o in got)

    def test_empty_and_null_values_are_skipped(self):
        rows = [
            {"date": "2025-01-01", "value": ""},
            {"date": "2025-02-01", "value": None},
            {"date": "2025-03-01", "value": "1.5"},
        ]
        got = FredClient.parse_observations(
            payload(rows), series_id="X", vintage_start=VINTAGE_START, vintage_end=VINTAGE_END
        )
        assert len(got) == 1

    def test_an_unparseable_row_is_skipped_not_fatal(self):
        rows = [
            {"date": "not-a-date", "value": "1.0"},
            {"date": "2025-02-01", "value": "not-a-number"},
            {"date": "2025-03-01", "value": "2.0"},
        ]
        got = FredClient.parse_observations(
            payload(rows), series_id="X", vintage_start=VINTAGE_START, vintage_end=VINTAGE_END
        )
        assert [o.value for o in got] == [2.0]

    def test_every_row_gets_the_requested_vintage(self):
        got = FredClient.parse_observations(
            CPI, series_id="X", vintage_start=VINTAGE_START, vintage_end=VINTAGE_END
        )
        assert {o.vintage_start for o in got} == {VINTAGE_START}
        assert {o.vintage_end for o in got} == {VINTAGE_END}

    def test_a_payload_without_observations_is_an_error(self):
        with pytest.raises(FredError, match="observations"):
            FredClient.parse_observations(
                {}, series_id="X", vintage_start=VINTAGE_START, vintage_end=VINTAGE_END
            )

    def test_a_non_json_body_is_an_error(self):
        with pytest.raises(FredError, match="non-JSON"):
            client(b"<html>nope</html>").fetch_observations(
                "X", realtime_start=VINTAGE_START, realtime_end=VINTAGE_END
            )

    def test_a_json_array_body_is_an_error(self):
        with pytest.raises(FredError, match="list"):
            client(b"[1, 2, 3]").fetch_observations(
                "X", realtime_start=VINTAGE_START, realtime_end=VINTAGE_END
            )

    def test_an_http_error_becomes_a_fred_error(self):
        def opener(request, timeout=None):
            raise urllib.error.HTTPError(request.full_url, 429, "Slow down", {}, None)

        c = FredClient(api_key=KEY, opener=opener)
        with pytest.raises(FredError, match="429"):
            c.fetch_observations("X", realtime_start=VINTAGE_START, realtime_end=VINTAGE_END)


class TestAvailability:
    def test_availability_is_the_close_of_the_realtime_window(self):
        # Within one ALFRED window a value may have been revised at any point,
        # so only the close is certainly safe. Being late can only make a
        # backtest less optimistic.
        assert obs(15, 318.6).available_at == dt.datetime(2025, 1, 31, 23, 59, 59, tzinfo=UTC)

    def test_availability_does_not_depend_on_the_observation_date(self):
        # The observation date says what the number is about. It must never move
        # the moment the number became knowable.
        early = obs(1, 1.0)
        late = obs(28, 2.0)
        assert early.available_at == late.available_at

    def test_a_value_is_invisible_until_its_window_closes(self):
        c = client()
        series = (obs(15, 318.6),)
        at_close = dt.datetime(2025, 1, 31, 23, 59, 59, tzinfo=UTC)
        just_before = dt.datetime(2025, 1, 31, 23, 58, tzinfo=UTC)
        assert len(c.observations_available_as_of(series, at_close)) == 1
        assert len(c.observations_available_as_of(series, just_before)) == 0

    def test_a_naive_as_of_is_refused(self):
        with pytest.raises(ValueError, match="naive"):
            client().observations_available_as_of((obs(1, 1.0),), dt.datetime(2025, 2, 1))


class TestRevisionSafety:
    """The property the module exists for.

    CPI for January is published in February and revised in March. A backtest
    run in June must not see the March number when it asks what was known in
    February.
    """

    ORIGINAL = payload([{"date": "2025-01-01", "value": "318.6"}])
    REVISED = payload([{"date": "2025-01-01", "value": "320.1"}])

    def vintage(self, body: dict, end: dt.date) -> MacroObservation:
        got = FredClient.parse_observations(
            body, series_id="CPIAUCSL", vintage_start=dt.date(2025, 1, 1), vintage_end=end
        )
        assert len(got) == 1
        return got[0]

    def test_a_later_vintage_is_invisible_to_an_earlier_query(self):
        february = self.vintage(self.ORIGINAL, dt.date(2025, 2, 28))
        march = self.vintage(self.REVISED, dt.date(2025, 3, 31))
        as_of_february = dt.datetime(2025, 2, 28, 23, 59, 59, tzinfo=UTC)

        visible = client().observations_available_as_of((february, march), as_of_february)
        assert [o.value for o in visible] == [318.6]

    def test_the_revision_is_visible_once_it_had_happened(self):
        february = self.vintage(self.ORIGINAL, dt.date(2025, 2, 28))
        march = self.vintage(self.REVISED, dt.date(2025, 3, 31))
        as_of_april = dt.datetime(2025, 4, 1, tzinfo=UTC)
        visible = client().observations_available_as_of((february, march), as_of_april)
        assert sorted(o.value for o in visible) == [318.6, 320.1]

    def test_the_two_vintages_are_distinguishable(self):
        february = self.vintage(self.ORIGINAL, dt.date(2025, 2, 28))
        march = self.vintage(self.REVISED, dt.date(2025, 3, 31))
        assert february.observation_date == march.observation_date
        assert february.vintage_end != march.vintage_end
        assert february.value != march.value


class TestSerialization:
    def test_round_tripping_keeps_the_vintage_and_the_availability_time(self):
        original = obs(15, 318.6)
        payload = original.to_dict()
        assert payload["observation_date"] == "2025-01-15"
        assert payload["vintage_start"] == "2025-01-01"
        assert payload["vintage_end"] == "2025-01-31"
        assert payload["available_at"] == "2025-01-31T23:59:59Z"
        assert json.loads(json.dumps(payload)) == payload

    def test_a_missing_retrieval_time_serializes_as_null(self):
        assert obs(1, 1.0).to_dict()["retrieved_at"] is None

    def test_a_retrieval_time_is_recorded_when_known(self):
        stamp = dt.datetime(2025, 2, 1, 12, tzinfo=UTC)
        got = obs(1, 1.0)
        with_stamp = MacroObservation(
            series_id=got.series_id,
            observation_date=got.observation_date,
            value=got.value,
            vintage_start=got.vintage_start,
            vintage_end=got.vintage_end,
            retrieved_at=stamp,
        )
        assert with_stamp.to_dict()["retrieved_at"] == "2025-02-01T12:00:00Z"


class TestCheckFrequency:
    def monthly(self, n: int) -> list[MacroObservation]:
        return [
            MacroObservation(
                series_id="CPIAUCSL",
                observation_date=dt.date(2024, 1, 1) + dt.timedelta(days=31 * i),
                value=float(i),
                vintage_start=VINTAGE_START,
                vintage_end=VINTAGE_END,
            )
            for i in range(n)
        ]

    def test_a_monthly_series_passes(self):
        ok, why = check_frequency(self.monthly(12), "M")
        assert ok, why

    def test_frequency_matching_is_case_insensitive(self):
        assert check_frequency(self.monthly(12), "m")[0]

    def test_a_daily_series_asked_for_as_monthly_fails(self):
        daily = [
            MacroObservation(
                series_id="DGS10",
                observation_date=dt.date(2024, 1, 1) + dt.timedelta(days=i),
                value=1.0,
                vintage_start=VINTAGE_START,
                vintage_end=VINTAGE_END,
            )
            for i in range(60)
        ]
        ok, why = check_frequency(daily, "M")
        assert not ok
        assert "M" in why

    def test_an_unknown_frequency_is_reported_not_assumed(self):
        ok, why = check_frequency(self.monthly(12), "fortnightly")
        assert not ok
        assert "unknown frequency" in why

    def test_too_few_observations_cannot_falsely_fail(self):
        assert check_frequency(self.monthly(2), "D")[0]

    def test_an_empty_series_is_handled(self):
        ok, why = check_frequency([], "M")
        assert ok, why