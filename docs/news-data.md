# News and macro data

How ARGOS gets from raw provider payloads to dated financial events, and what
each step refuses to do.

The whole layer is built around one question: **at what moment could this
information have been acted on?** Everything else — schemas, storage, the audit
— exists to make that answer checkable rather than assumed.

- [The pipeline](#the-pipeline)
- [Point-in-time rules](#point-in-time-rules)
- [Schemas](#schemas)
- [Providers](#providers)
- [Coverage you cannot get](#coverage-you-cannot-get)
- [Running it](#running-it)
- [What the audit checks](#what-the-audit-checks)
- [Known unknowns](#known-unknowns)

---

## The pipeline

```
raw payload ──▶ normalize ──▶ articles ──▶ dedupe ──▶ events ──▶ features
                     │                                              │
                     └── rejected (with reason)                     └── audit
```

Each stage reads what the previous one wrote and never re-derives an upstream
result, so a run can be reproduced from the store alone. Each stage writes a
manifest describing **every** file the dataset has, not just its own — otherwise
the last stage to run would erase the digests of everything before it.

| Stage | Command | Writes |
|---|---|---|
| fetch | `argos news fetch` | `data/news/raw/<dataset>.json` |
| normalize | `argos news normalize` | `normalized/<dataset>.jsonl`, manifest |
| analyze | `argos news analyze` | `events/<dataset>.jsonl`, manifest |
| audit | `argos news audit` | verdict, exit code |

`argos news inspect` summarizes what a dataset contains without judging it.

### Why facts and inferences are separate types

`NewsArticle` records what a provider published. `FinancialEvent` records what
an analyzer concluded from it. They are different objects because they have
different failure modes and different trust levels.

The separation is enforced structurally: an event carries no `size`, no `order`,
no `stop_loss`, no target price, and no cash field. There is no version of this
layer that can tell the engine how much to buy, because the type has nowhere to
put it. A strategy that wants news influence composes a signal from events; it
does not read them directly.

### Identity and deduplication

An article's identity is resolved in three steps, first match wins:

1. **Provider id** — `provider:<source>:<raw_source_id>`
2. **Canonical URL** — `url:<scheme+host+path+query>`, reduced so that the same
   article reached two ways shares an identity: scheme lowercased, `http` folded
   to `https`, `www.` dropped, default ports removed, duplicate and trailing
   slashes collapsed, tracking parameters removed, remaining query parameters
   sorted, fragment discarded. Credentials embedded in a URL are stripped rather
   than stored. Only a leading `www.` is removed — `news.example.com` and
   `example.com` are different pages, and collapsing either to a bare label would
   let unrelated sites share an identity.
3. **Content fingerprint** — `fp:<sha256>` of normalized title and summary text

Deduplication keeps the earliest `published_at` — the moment the information
first became public — and sorts deterministically by id. Fingerprints use
SHA-256, never the built-in `hash()`, whose values vary per process and would
make a store non-reproducible across runs.

Two provider ids for one wire story survive step 1 and 2, so the audit reports
them as `duplicate-content`: it cannot tell a syndicated story from two genuinely
different reports with the same text.

---

## Point-in-time rules

These are the rules that make a news backtest honest. Each is enforced in code
and covered by a test.

**`published_at` is the only thing that orders information.** It is required,
timezone-aware, and normalized to UTC. A missing, naive or unparseable timestamp
rejects the record with a reason. `retrieved_at` records when *we* fetched it and
never stands in for publication.

**No fallback to `retrieved_at`.** This is the single most common way a news
backtest lies: an undated record gets stamped with the crawl time, which is
always present, and then looks perfectly well-dated.

**Availability is judged by `published_at`, never `analyzed_at`.** An LLM that
takes four seconds to classify an article does not delay the information by four
seconds. The article was public at 20:05; a decision made at 20:05:04 could act
on it.

**Events inherit their article's `published_at`.** An analyzer cannot back-date a
conclusion to manufacture lead time. `audit_timestamps` flags any drift.

**A feature is usable the day after publication**, not the day of.
`first_decision_date` takes a UTC date and adds the policy's lag. With the default
one-day lag, a 20:05 UTC article can inform a decision taken at the next day's
open, and nothing sooner. A zero lag is rejected by the policy, and an event that
would be usable on its own publication day is reported as `same-day-signal`.

**Prefix invariance.** Truncating the future must not change any already-computed
score. This is tested for price features and is equally true for news: adding
tomorrow's articles cannot alter today's.

### What a news feature is allowed to do

A news feature is an *input* to a strategy, nothing more. It cannot set position
size, override a risk limit, place an order, choose a stop, or bypass the risk
engine. `NewsFeaturePolicy` is a plain dataclass of thresholds — lag, minimum
confidence, weighting, whether a ticker is required — and it holds no execution
field of any kind.

`daily_news_score` deliberately excludes `UNKNOWN` events and ignores `magnitude`.
Weighting is `sign(direction) × confidence`, and only `POSITIVE` and `NEGATIVE`
carry a sign, so `NEUTRAL`, `MIXED` and `UNKNOWN` contribute nothing. Magnitude is
recorded and audited but not scored, because a confidence value that has never
been calibrated against outcomes should not be allowed to scale a position.

Events with no ticker are dropped when `require_ticker` is set, which is the
default. That is a real loss of coverage, so the audit reports how many events
the policy discarded instead of letting it pass unnoticed.

---

## Schemas

### `NewsArticle`

| Field | Notes |
|---|---|
| `article_id` | resolved identity: `provider:`, `url:` or `fp:` prefixed |
| `source` | provider name, part of identity |
| `title`, `summary`, `url` | as published |
| `published_at` | required, UTC-aware |
| `retrieved_at` | when we fetched it; never an ordering input |
| `tickers` | as tagged by the provider, not inferred |
| `raw_source_id`, `fingerprint` | provider id and content hash |

### `FinancialEvent`

| Field | Notes |
|---|---|
| `event_id`, `article_id` | event id, and the article it came from |
| `event_type` | closed taxonomy: `EARNINGS`, `GUIDANCE`, `MERGER_ACQUISITION`, `PRODUCT`, `REGULATORY`, `LEGAL`, `MANAGEMENT`, `MACRO`, `ANALYST_ACTION`, `OTHER` |
| `direction` | `POSITIVE`, `NEGATIVE`, `NEUTRAL`, `MIXED`, `UNKNOWN` |
| `confidence`, `magnitude` | recorded; only confidence contributes to a score |
| `analyzer`, `analyzer_version` | required — an event with no analyzer is unauditable |
| `model`, `prompt_ref` | required for model-backed analyzers |
| `published_at`, `analyzed_at` | inherited and observed times |

`UNKNOWN` is a first-class direction, not a missing value. An analyzer that
cannot tell, or refuses to answer, says so and the event is excluded from the
score rather than guessed at. `NEUTRAL` and `MIXED` are equally explicit: they
are statements that the news is not directional, not a hedge on a missing
answer. Only `POSITIVE` and `NEGATIVE` carry a sign.

Every event records how it was produced. That is what lets a dataset be
re-derived when a prompt changes instead of drifting silently.

### `MacroObservation`

FRED/ALFRED series, using `realtime_start`/`realtime_end` for availability and
`vintage_end` to pin which revision was known at the time. EDGAR filings use
`acceptanceDateTime`, which is when the filing was accepted, not its period.

---

## Providers

| Provider | Needs | What it gives |
|---|---|---|
| Massive | `MASSIVE_API_KEY` | news articles with `published_utc` |
| OpenAI | `OPENAI_API_KEY` + explicit `--model` | event classification |
| FRED / ALFRED | `FRED_API_KEY` | macro series with vintages |
| EDGAR | a descriptive `User-Agent` | filings with acceptance times |

Keys are read from the environment only — never from command-line arguments,
which would land in shell history and in any process listing.

**Massive** paginates by cursor. `ProviderResponse.next_url` is redacted before
it is persisted or logged, so a cursor cannot leak a key into a stored file,
while `iter_pages` still follows the raw cursor internally. The default rate
limit is derived from the plan rather than hard-coded: `Stocks Basic` allows about
5 requests/minute, which is 12 seconds between calls. Retries count after the
initial attempt, so `max_retries=3` means four requests.

**OpenAI** uses the Responses API with a strict JSON Schema. There is no default
model id: a silently wrong model would make every recorded provenance entry a
lie. Refusals, truncated output, malformed bodies and transport errors all fail
closed to `UNKNOWN` and are cached, so a retry storm cannot turn one outage into
an LLM bill. Transport errors are redacted.

**EDGAR** needs no key but does require a descriptive User-Agent, and SEC
fair-access guidance asks for no more than 10 requests/second.

---

## Coverage you cannot get

**The free tier cannot reproduce a 2020–2025 backtest.** Massive's `Stocks Basic`
plan serves roughly two years of news history. A pull for 2020 returns nothing
for that year, and nothing fails — the dataset simply starts in 2024.

This is the failure mode the audit exists to catch. Declaring the window you
meant to cover makes it fatal:

```bash
argos news audit --dataset demo --expected-start 2020-01-01
```

Without that flag the gap is only a warning, because the operator has claimed no
window and a short history may be perfectly fine. With it, `coverage-truncated-start`
fails the command and exits non-zero.

Any conclusion drawn from a two-year window applies to two years. A backtest over
six years fed two years of news is not a weak result, it is a wrong one.

**Vintage data is not free either.** ALFRED revisions matter: the number FRED
shows today may not be the number available in 2023. Using current-vintage macro
data in a historical backtest is look-ahead, and `vintage_end` is what prevents it.

---

## Running it

```bash
cp .env.example .env          # fill in what you have

argos news fetch     --dataset demo --start 2024-01-01 --end 2025-12-31
argos news normalize --dataset demo
argos news analyze   --dataset demo --analyzer deterministic
argos news inspect   --dataset demo
argos news audit     --dataset demo --expected-start 2024-01-01
```

Everything under `data/news/raw/`, `data/news/normalized/` and
`data/news/events/` is ignored by git. Only `.gitkeep` and the synthetic
fixtures under `data/news/fixtures/` are tracked, so no provider payload — and
nothing licensed — can reach a commit by accident.

---

## What the audit checks

`argos news audit` exits non-zero on any error. Run it after every pipeline
change and before trusting a dataset.

| Finding | Severity | Means |
|---|---|---|
| `timestamp-naive` | error | a `published_at` has no UTC offset |
| `retrieval-before-publication` | error | retrieved before it was published |
| `publication-time-drift` | error | an event is stamped earlier than its article |
| `orphan-event` | warning | an event references an article not in the set |
| `duplicate-articles` | error | same identity, more than once |
| `duplicate-events` | error | same event id twice; the score would double-count it |
| `duplicate-content` | warning | same fingerprint under different identities |
| `missing-analyzer-identity` | error | no analyzer or version recorded |
| `missing-article-reference` | error | no source article |
| `incomplete-model-provenance` | error | model recorded without prompt, or vice versa |
| `missing-model-identity` | error | model-backed analyzer with no model recorded |
| `unaudited-analyzer` | warning | an analyzer nothing knows how to verify |
| `same-day-signal` | error | usable on its own publication day |
| `untagged-events-dropped` | warning | events the policy discards for having no ticker |
| `uncalibrated-direction` | warning | directional event with no confidence, weighted at full sign |
| `coverage-empty` | error | the manifest describes no coverage |
| `coverage-truncated-start` | warning, fatal with `--expected-start` | coverage begins after the requested window |
| `coverage-truncated-end` | warning | coverage ends before the requested window |
| `coverage-before-plan-limit` | warning | coverage predates the plan's advertised history |
| `manifest-missing-file` | error | a file the manifest lists is gone |
| `manifest-size-mismatch` | error | file size differs from the manifest |
| `manifest-hash-mismatch` | error | content differs at the same size |
| `event-timestamp` | error | an event's own timestamp is inconsistent |
| `nothing-to-audit` | error | nothing was supplied; not the same as nothing wrong |

Errors mean the dataset should not be used. Warnings mean use it knowing this.

The hash check matters more than its size suggests. Size alone is not identity: a
fetch that stopped one record short has almost the same byte count as a complete
one, and an amended row of the same length is indistinguishable. Only the
SHA-256 proves the file is the one the manifest describes.

---

## Known unknowns

Stated plainly rather than papered over:

- **Licensing and retention terms for Massive/Polygon are unresolved.** Whether
  raw payloads may be stored, redistributed or used to publish derived research
  has not been established. Nothing here assumes a permission that has not been
  confirmed.
- **No licensed corpus is committed.** Only self-authored synthetic fixtures.
- **Free-tier coverage is roughly two years**, so no pre-2024 baseline can be
  claimed from this pipeline as configured.
- **Alpha Vantage `NEWS_SENTIMENT` was evaluated and deferred.** Its schema is
  too thin to support point-in-time reconstruction.
- **Ticker tags come from the provider and are not verified** against an
  instrument master. A mis-tagged article scores the wrong symbol.
- **No de-duplication against other providers.** The same story arriving via two
  feeds is only caught when the text matches.
- **LLM event quality is uncalibrated.** `confidence` comes from the model, not
  from a scoring study against realized outcomes, which is why `magnitude` is
  deliberately not used to scale anything.