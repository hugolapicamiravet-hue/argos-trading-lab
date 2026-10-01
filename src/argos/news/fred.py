"""FRED and ALFRED macro series, with vintage-aware querying.

The look-ahead problem in macro data
------------------------------------
FRED's default behaviour returns the *latest* value for each date. If CPI for
March is revised, a plain FRED query run in 2026 returns April 2026's revised
number for that March date. A backtest that read it would be trading on
information that did not exist at the time.

ALFRED exists precisely for this. It serves the vintage as it stood on a given
realtime date, so the correct question is never "what is the value for March?"
but "what did someone querying on 1 April 2026 know about March?" The answer to
that question comes from ALFRED, keyed by `realtime_start` / `realtime_end`,
which describe a closed interval of realtime dates.

The rule encoded here: never read a macro value without also recording the
vintage it was read as of. `MacroObservation` carries both, and `available_at`
is derived from the realtime window, not from the observation date.

Data as an observation
----------------------
Note the asymmetry with news, and do not "fix" it:

    news    a single article published at one instant
    macro   one number for one period, revised across many vintages

So a macro observation has two independent times: `observation_date` (what
period it describes) and `available_at` (when that particular value became
public). Both matter, and conflating them is the FRED-specific version of the
same bug this project keeps guarding against.
"""

from __future__ import annotations

import datetime as dt
import json
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from argos.news.models import UTC

FRED_SERIES_URL = "https://api.stlouisfed.org/fred/series/observations"
ALFRED_SERIES_URL = "https://api.stlouisfed.org/alfred/series/observations"
FRED_SERIES_BASE = "https://fred.stlouisfed.org/graph/fredgraph.csv"

DEFAULT_FRED_TIMEOUT = 30.0
DEFAULT_USER_AGENT = "argos-trading-lab/0.1 (educational backtesting lab)"

# Frequency -> expected days between observations. Used to sanity-check a
# series' shape; a daily series returning monthly spacing means the caller
# asked for the wrong series id.
FREQUENCY_DAYS = {
    "D": 1,
    "W": 7,
    "M": 31,
    "Q": 91,
    "A": 365,
}


class FredError(RuntimeError):
    """FRED/ALFRED request failed."""


class FredConfigurationError(RuntimeError):
    """Missing key, or a query that cannot answer a point-in-time question."""


@dataclass(frozen=True, slots=True)
class MacroObservation:
    """One value for one period, read as of one vintage.

    `observation_date` is what the number describes. `vintage_start` and
    `vintage_end` bracket the realtime window this value was read under;
    `available_at` is the earliest moment this exact value could have been
    published, which is the conservative choice for a backtest.
    """

    series_id: str
    observation_date: dt.date
    value: float
    vintage_start: dt.date
    vintage_end: dt.date
    retrieved_at: dt.datetime | None = None

    @property
    def available_at(self) -> dt.datetime:
        """Earliest instant this value was public.

        Taken as the *end* of the realtime window rather than the start: within
        one ALFRED window, values may have been revised at any point, so the
        conservative assumption is that only the window's close is safe. This
        can only ever make a backtest less optimistic, never more.
        """
        return dt.datetime.combine(
            self.vintage_end, dt.time(23, 59, 59), tzinfo=UTC
        )

    def to_dict(self) -> dict:
        return {
            "series_id": self.series_id,
            "observation_date": self.observation_date.isoformat(),
            "value": self.value,
            "vintage_start": self.vintage_start.isoformat(),
            "vintage_end": self.vintage_end.isoformat(),
            "available_at": self.available_at.isoformat().replace("+00:00", "Z"),
            "retrieved_at": (
                self.retrieved_at.isoformat().replace("+00:00", "Z")
                if self.retrieved_at
                else None
            ),
        }


class FredClient:
    """ALFRED-first client for vintage-aware macro data.

    `fetch_observations` refuses to run without an explicit realtime window. It
    is not being difficult for its own sake: `vintage_end` is what makes the
    result safe to use in a backtest, and letting it default to today would make
    every "no-vintage" query silently look-ahead. If you genuinely want current
    values, ask for them deliberately by passing today's date, and the manifest
    will show a vintage equal to the retrieval date.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        timeout: float = DEFAULT_FRED_TIMEOUT,
        base_url: str = ALFRED_SERIES_URL,
        opener: object | None = None,
    ) -> None:
        import os

        self.api_key = api_key or os.environ.get("FRED_API_KEY")
        if not self.api_key:
            raise FredConfigurationError(
                "FRED_API_KEY is not set. ALFRED's API requires a key even though the "
                "CSV download does not; the key is never written to disk or logged."
            )
        self.timeout = float(timeout)
        self.base_url = base_url
        self._opener = opener or urllib.request.urlopen

    def fetch_observations(
        self,
        series_id: str,
        *,
        realtime_start: dt.date,
        realtime_end: dt.date,
        observation_start: dt.date | None = None,
        observation_end: dt.date | None = None,
    ) -> tuple[MacroObservation, ...]:
        """Fetch series values as they stood during a realtime window.

        The ALFRED realtime window is closed on both ends, matching FRED's
        documented behaviour.
        """
        if not series_id:
            raise FredConfigurationError("series_id is required")
        if realtime_end < realtime_start:
            raise FredConfigurationError(
                f"realtime_end {realtime_end} must not precede realtime_start {realtime_start}"
            )

        query = {
            "series_id": series_id,
            "api_key": self.api_key,
            "file_type": "json",
            "realtime_start": realtime_start.isoformat(),
            "realtime_end": realtime_end.isoformat(),
        }
        if observation_start is not None:
            query["observation_start"] = observation_start.isoformat()
        if observation_end is not None:
            query["observation_end"] = observation_end.isoformat()

        url = f"{self.base_url}?{urllib.parse.urlencode(query)}"
        payload = self._get_json(url)
        return self.parse_observations(
            payload, series_id=series_id, vintage_start=realtime_start, vintage_end=realtime_end
        )

    @staticmethod
    def parse_observations(
        payload: Mapping[str, object],
        *,
        series_id: str,
        vintage_start: dt.date,
        vintage_end: dt.date,
        retrieved_at: dt.datetime | None = None,
    ) -> tuple[MacroObservation, ...]:
        """Parse an ALFRED/FRED observations payload.

        Static so it can be tested against recorded payloads with no key and no
        network. Missing values are skipped rather than coerced to zero: FRED
        reports `.` for a period with no data, and zero is a real number that
        would silently corrupt a series.
        """
        observations = payload.get("observations")
        if not isinstance(observations, list):
            raise FredError(
                f"FRED payload has no 'observations' list (got {type(observations).__name__})"
            )

        results: list[MacroObservation] = []
        for row in observations:
            if not isinstance(row, Mapping):
                continue
            raw_date = row.get("date")
            raw_value = row.get("value")
            if raw_date is None or raw_value in (None, "", "."):
                continue
            try:
                observation_date = dt.date.fromisoformat(str(raw_date))
                value = float(raw_value)
            except (TypeError, ValueError):
                continue
            results.append(
                MacroObservation(
                    series_id=series_id,
                    observation_date=observation_date,
                    value=value,
                    vintage_start=vintage_start,
                    vintage_end=vintage_end,
                    retrieved_at=retrieved_at,
                )
            )

        results.sort(key=lambda o: o.observation_date)
        return tuple(results)

    def observations_available_as_of(
        self,
        observations: Iterable[MacroObservation],
        as_of: dt.datetime,
    ) -> tuple[MacroObservation, ...]:
        """Filter observations whose vintage makes them usable by `as_of`.

        An observation is usable when `as_of` reaches the end of its realtime
        window. This is the macro equivalent of `events_available_as_of`.
        """
        moment = _aware(as_of)
        return tuple(o for o in observations if o.available_at <= moment)

    def _get_json(self, url: str) -> Mapping[str, object]:
        request = urllib.request.Request(
            url, headers={"User-Agent": DEFAULT_USER_AGENT, "Accept": "application/json"}
        )
        try:
            with self._opener(request, timeout=self.timeout) as response:
                body = response.read()
        except urllib.error.HTTPError as exc:
            raise FredError(f"FRED request failed with HTTP {exc.code}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise FredError(f"FRED request failed: {exc}") from exc
        try:
            decoded = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise FredError(f"FRED returned a non-JSON body: {exc}") from exc
        if not isinstance(decoded, Mapping):
            raise FredError(f"FRED returned {type(decoded).__name__}, expected an object")
        return decoded


def check_frequency(
    observations: Sequence[MacroObservation], frequency: str
) -> tuple[bool, str]:
    """Verify a series' spacing matches its declared frequency.

    A mismatch usually means the wrong `series_id`, which is worth catching
    before a feature is built on it. Tolerance is generous because holidays and
    revisions shift the gap by a few days.
    """
    expected = FREQUENCY_DAYS.get(frequency.upper())
    if expected is None:
        return False, f"unknown frequency {frequency!r}"
    if len(observations) < 3:
        return True, "too few observations to check spacing"

    dates = [o.observation_date for o in observations]
    gaps = [
        (dates[i + 1] - dates[i]).days for i in range(len(dates) - 1)
    ]
    if not gaps:
        return True, "no gaps to check"

    average = sum(gaps) / len(gaps)
    # Half the expected period, plus a week, absorbs holidays and gaps.
    tolerance = max(expected * 0.5, 7)
    if abs(average - expected) > tolerance:
        return False, (
            f"average gap {average:.1f} days does not match {frequency.upper()} "
            f"expectation of {expected} days (tolerance {tolerance:.0f})"
        )
    return True, f"average gap {average:.1f} days, consistent with {frequency.upper()}"


class MacroSeriesProvider(Protocol):
    """Interface a caller can rely on without importing FRED specifics."""

    def fetch_observations(
        self,
        series_id: str,
        *,
        realtime_start: dt.date,
        realtime_end: dt.date,
    ) -> tuple[MacroObservation, ...]: ...


def _aware(value: dt.datetime) -> dt.datetime:
    if value.tzinfo is None:
        raise ValueError(f"{value!r} is naive; an as-of comparison needs an offset")
    return value.astimezone(UTC)