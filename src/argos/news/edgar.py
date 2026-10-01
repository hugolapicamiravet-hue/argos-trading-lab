"""SEC EDGAR filings as point-in-time corporate events.

Why EDGAR is worth integrating separately from news
---------------------------------------------------
A press release says what a company chose to announce. A filing says what a
company was required to disclose, on a date the SEC published, with a document
that has not been rewritten since. For anything time-sensitive that distinction
is the whole value proposition: 8-K items are filed within four business days of
an event, whereas the news cycle around the same event is uneven and often
preceded by speculation.

Temporal semantics, which differ from the news pipeline in one important way
---------------------------------------------------------------
EDGAR timestamps are already UTC and already archival, so nothing needs
converting. What matters instead is choosing the right field:

    acceptanceDateTime   when EDGAR accepted the filing: the moment the public
                         could see it. This is the availability time.
    filingDate           the calendar date on the filing header, in ET.
    reportDate           the period the filing *describes*, which can be a
                         quarter earlier. Never use this as availability.

`filingDate` looks more convenient and is the wrong choice: it has no time of
day, so an event dated D could be placed anywhere in a 24-hour window, and using
it to filter "what was known on D" invites an off-by-one-day leak. The engine
consumes `acceptanceDateTime`.

`reportDate` is retained on the event but is documentation, not a temporal
gate. A 10-Q filed on 2025-01-30 with `reportDate` 2024-12-31 is available on
2025-01-30, not on 2024-12-31, and confusing those two is a classic way to
manufacture a look-ahead.

Access policy
-------------
No API key. A descriptive User-Agent with contact information is mandatory, and
SEC fair-access guidance limits clients to 10 requests per second. Both are
enforced here rather than documented and hoped for.
"""

from __future__ import annotations

import datetime as dt
import json
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from argos.errors import ArgosError
from argos.news.models import UTC

SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
ARCHIVES_BASE = "https://www.sec.gov/Archives/edgar/data"
CIK_URL_SUFFIX = ".json"

FAIR_ACCESS_MAX_REQUESTS_PER_SECOND = 10
DEFAULT_EDGAR_TIMEOUT = 30.0
DEFAULT_USER_AGENT = (
    "argos-trading-lab/0.1 (educational backtesting lab; contact: set --user-agent)"
)

# Forms worth turning into events. 4 and 13F/13D are excluded deliberately: they
# are routine ownership and transaction reports with no directional content.
RELEVANT_FORMS = ("10-K", "10-Q", "8-K")


class EdgarError(ArgosError):
    """EDGAR request failed."""


class EdgarConfigurationError(ArgosError):
    """EDGAR client is missing configuration it cannot invent."""


@dataclass(frozen=True, slots=True)
class Filing:
    """One filing as EDGAR reports it.

    Facts only. No interpretation: a filing is not classified as good or bad
    here, it is merely recorded with the timestamp at which it became public.
    """

    cik: str
    accession_number: str
    form: str
    acceptance_datetime: dt.datetime
    filing_date: dt.date | None
    report_date: dt.date | None
    primary_document: str | None = None
    primary_doc_description: str | None = None
    items: tuple[str, ...] = ()

    @property
    def available_at(self) -> dt.datetime:
        """When the public could see this filing: EDGAR acceptance time."""
        return self.acceptance_datetime

    @property
    def primary_document_url(self) -> str | None:
        """Direct URL to the primary document, built the way EDGAR lays it out.

        The path uses the un-padded CIK and the accession number with its dashes
        removed, which is EDGAR's convention rather than an assumption about it.
        """
        if not self.primary_document:
            return None
        cik = str(self.cik).lstrip("0") or "0"
        accession = self.accession_number.replace("-", "")
        return (
            f"{ARCHIVES_BASE}/{cik}/{accession}/{self.primary_document}"
        )

    def to_dict(self) -> dict:
        return {
            "cik": self.cik,
            "accession_number": self.accession_number,
            "form": self.form,
            "acceptance_datetime": self.acceptance_datetime.isoformat().replace(
                "+00:00", "Z"
            ),
            "filing_date": self.filing_date.isoformat() if self.filing_date else None,
            "report_date": self.report_date.isoformat() if self.report_date else None,
            "primary_document": self.primary_document,
            "primary_doc_description": self.primary_doc_description,
            "items": list(self.items),
            "primary_document_url": self.primary_document_url,
        }

    @classmethod
    def from_submission(
        cls, *, cik: str, record: Mapping[str, object]
    ) -> Filing | None:
        """Build a `Filing` from one entry of the `filings.recent` arrays.

        Returns None when `acceptanceDateTime` is absent, since without it the
        filing has no availability time and must not be used. Same fail-closed
        rule as the news pipeline: no timestamp means no event.
        """
        raw_acceptance = record.get("acceptanceDateTime")
        if not raw_acceptance:
            return None
        try:
            accepted = dt.datetime.fromisoformat(str(raw_acceptance)).replace(tzinfo=UTC)
        except ValueError:
            return None

        raw_filing = record.get("filingDate")
        raw_report = record.get("reportDate")
        raw_items = record.get("items") or ""
        return cls(
            cik=str(cik),
            accession_number=str(record.get("accessionNumber") or ""),
            form=str(record.get("form") or ""),
            acceptance_datetime=accepted,
            filing_date=dt.date.fromisoformat(str(raw_filing)) if raw_filing else None,
            report_date=dt.date.fromisoformat(str(raw_report)) if raw_report else None,
            primary_document=_optional_str(record.get("primaryDocument")),
            primary_doc_description=_optional_str(record.get("primaryDocDescription")),
            items=tuple(
                chunk.strip() for chunk in str(raw_items).split(",") if chunk.strip()
            ),
        )


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


class EdgarClient:
    """Thin, well-mannered EDGAR client.

    Injects its opener so every test in this project runs without a network
    call, which is the only way a test suite with an external dependency stays
    trustworthy.
    """

    def __init__(
        self,
        *,
        user_agent: str | None = None,
        timeout: float = DEFAULT_EDGAR_TIMEOUT,
        min_interval: float | None = None,
        opener: object | None = None,
        sleep: object | None = None,
    ) -> None:
        self.user_agent = user_agent or DEFAULT_USER_AGENT
        lowered = self.user_agent.lower()
        # SEC asks for an identifying User-Agent with contact information. An
        # email address satisfies that; the word "contact" is accepted too so a
        # placeholder User-Agent still fails loudly rather than being blocked
        # later with an opaque 403.
        has_contact = "@" in self.user_agent or "contact" in lowered
        if not has_contact or "set --user-agent" in lowered:
            raise EdgarConfigurationError(
                "SEC access policy requires a User-Agent identifying this project and a "
                "contact address. Pass user_agent='argos-trading-lab you@example.com'."
            )
        self.timeout = float(timeout)
        default_interval = 1.0 / FAIR_ACCESS_MAX_REQUESTS_PER_SECOND
        self.min_interval = default_interval if min_interval is None else float(min_interval)
        self._opener = opener or urllib.request.urlopen
        self._sleep = sleep or __import__("time").sleep
        self._last_at: float | None = None
        self.request_count = 0

    def _throttle(self) -> None:
        import time

        now = time.monotonic()
        if self._last_at is not None:
            wait = self.min_interval - (now - self._last_at)
            if wait > 0:
                self._sleep(wait)
        self._last_at = time.monotonic()

    def _get(self, url: str) -> Mapping[str, object]:
        self._throttle()
        self.request_count += 1
        request = urllib.request.Request(
            url, headers={"User-Agent": self.user_agent, "Accept": "application/json"}
        )
        try:
            with self._opener(request, timeout=self.timeout) as response:
                body = response.read()
        except urllib.error.HTTPError as exc:
            raise EdgarError(f"EDGAR request to {url} failed with HTTP {exc.code}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise EdgarError(f"EDGAR request to {url} failed: {exc}") from exc
        try:
            decoded = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise EdgarError(f"EDGAR returned a non-JSON body from {url}: {exc}") from exc
        if not isinstance(decoded, Mapping):
            raise EdgarError(f"EDGAR returned {type(decoded).__name__} from {url}")
        return decoded

    def submissions(self, cik: str | int) -> Mapping[str, object]:
        """Fetch `submissions/CIK##########.json` for a company."""
        numeric = _normalize_cik(cik)
        return self._get(SUBMISSIONS_URL.format(cik=numeric))

    def filings(
        self,
        cik: str | int,
        *,
        forms: Sequence[str] = RELEVANT_FORMS,
        since: dt.datetime | None = None,
        limit: int | None = None,
    ) -> tuple[Filing, ...]:
        """Return relevant filings, newest first, filtered by acceptance time.

        `since` compares against `acceptanceDateTime`, not `filingDate`, so the
        filter means "publicly available since".
        """
        payload = self.submissions(cik)
        filings_block = payload.get("filings")
        if not isinstance(filings_block, Mapping):
            raise EdgarError("EDGAR submissions payload has no 'filings' object")

        # EDGAR splits filings across `filings.recent` and one or more
        # `filings.files` shards for long histories. Only `recent` is read here;
        # the shards would need their own index fetches, and a half-implemented
        # path to them is worse than none. A caller needing older filings should
        # extend this deliberately.
        recent = filings_block.get("recent")
        if not isinstance(recent, Mapping):
            raise EdgarError(
                "EDGAR submissions payload has no 'filings.recent' object; the response "
                "shape may have changed"
            )

        rows = _rows_from_block(recent)
        wanted = {f.strip().upper() for f in forms}
        normalized_cik = _normalize_cik(cik)

        filings: list[Filing] = []
        for record in rows:
            form = str(record.get("form") or "").strip().upper()
            if wanted and form not in wanted:
                continue
            filing = Filing.from_submission(cik=normalized_cik, record=record)
            if filing is None:
                continue
            if since is not None and filing.acceptance_datetime < _aware(since):
                continue
            filings.append(filing)

        filings.sort(key=lambda f: f.acceptance_datetime, reverse=True)
        if limit is not None:
            filings = filings[:limit]
        return tuple(filings)

    def company_tickers(self) -> Mapping[str, str]:
        """Map ticker -> CIK from EDGAR's public ticker index.

        A plain dict because callers want lookup, not the raw array.
        """
        payload = self._get(TICKER_MAP_URL)
        result: dict[str, str] = {}
        for entry in payload.values():
            if not isinstance(entry, Mapping):
                continue
            ticker = entry.get("ticker")
            cik = entry.get("cik_str")
            if ticker and cik is not None:
                result[str(ticker).upper()] = str(cik)
        return result

    def cik_for_ticker(self, ticker: str) -> str:
        """Resolve one ticker to a zero-padded CIK."""
        symbol = str(ticker).strip().upper()
        found = self.company_tickers().get(symbol)
        if not found:
            raise EdgarError(f"ticker {symbol!r} not found in EDGAR's ticker index")
        return found.zfill(10)


def _rows_from_block(block: Mapping[str, object]) -> list[Mapping[str, object]]:
    """Reassemble EDGAR's column-per-field layout into row dicts.

    EDGAR returns parallel arrays (`accessionNumber: [...]`, `form: [...]`) plus
    an `length` marker. Transposing is the only sensible reading, and a missing
    or mismatched array means the payload is not what we think it is, so it
    raises rather than guessing.
    """
    length = block.get("length")
    columns = {k: v for k, v in block.items() if isinstance(v, list) and k != "length"}
    if length is None:
        raise EdgarError("EDGAR submissions block has no 'length'")
    count = int(length)
    for name, values in columns.items():
        if len(values) != count:
            raise EdgarError(
                f"EDGAR submissions column {name!r} has {len(values)} rows, "
                f"expected {count}"
            )
    names = sorted(columns)
    return [
        {name: columns[name][index] for name in names}
        for index in range(count)
    ]


def _normalize_cik(cik: str | int) -> int:
    digits = "".join(ch for ch in str(cik) if ch.isdigit())
    if not digits:
        raise EdgarConfigurationError(f"CIK {cik!r} contains no digits")
    return int(digits)


def _aware(value: dt.datetime) -> dt.datetime:
    if value.tzinfo is None:
        raise ValueError(
            f"{value!r} is naive; comparisons against EDGAR acceptance times need an offset"
        )
    return value.astimezone(UTC)


def filing_to_event_type(form: str) -> str:
    """Map a form type to a coarse label, for display only.

    8-K is split by its `items` codes in principle; here it stays 8-K because the
    item-to-meaning mapping is long and only worth encoding if a caller actually
    branches on it.
    """
    key = str(form).strip().upper()
    if key in ("10-K", "10-Q"):
        return "periodic_report"
    if key == "8-K":
        return "current_report"
    return "other"


class FilingEventFactory(Protocol):
    """What a caller must supply to turn filings into events."""

    def __call__(self, filing: Filing) -> object: ...