"""News data: raw information, normalized articles, and inferred events.

The layering is the whole point of this package, so it is worth stating plainly.
Three kinds of thing live here and they are not interchangeable:

    RAW        exactly what the provider returned, byte-for-byte as JSON
    NORMALIZED  provider data mapped into ARGOS's own schema, validated, deduped
    INFERRED   a model's interpretation of an article, stored separately

An article's `publisher`, `published_at` and `title` are provider facts. An
event's `direction`, `magnitude` and `confidence` are model inferences. Mixing
them in one record would make it impossible, later, to tell what was observed
from what was guessed. Every `FinancialEvent` therefore carries the id of the
article it came from plus the analyzer identity that produced it.

The temporal contract, which the tests exist to protect:

    An article may influence a decision only at or after its own published_at.

`retrieved_at` records when *we* fetched it and is never a substitute for
`published_at`. A backtest that used retrieval time would silently assume the
news was available in 2020 because we downloaded it in 2026, which is exactly the
error this package is designed to make impossible.
"""

from argos.news.analyzer import (
    EVENT_SCHEMA,
    SYSTEM_PROMPT,
    AnalysisCache,
    AnalysisResult,
    Analyzer,
    AnalyzerError,
    AnalyzerIdentity,
    DeterministicAnalyzer,
    DictCache,
    JsonlCache,
    LLMAnalyzer,
    LLMConfigurationError,
    PromptRef,
    analyze_many,
    build_analyzer,
    build_prompt_ref,
    parse_llm_response,
    to_event,
)
from argos.news.audit import (
    AuditFailure,
    AuditReport,
    Finding,
    Severity,
    audit_from_store,
    run_audit,
)
from argos.news.dedupe import (
    article_identity,
    article_key,
    canonical_url,
    dedupe_articles,
)
from argos.news.edgar import (
    RELEVANT_FORMS,
    EdgarClient,
    EdgarConfigurationError,
    EdgarError,
    Filing,
    filing_to_event_type,
)
from argos.news.events import (
    EventDataset,
    build_dataset,
    events_available_as_of,
    events_for_trading_range,
    load_events,
    write_events,
)
from argos.news.features import (
    NewsFeaturePolicy,
    align_score_to_index,
    cumulative_news_score,
    daily_news_score,
    feature_for_dataset,
    first_decision_date,
)
from argos.news.fred import (
    FredClient,
    FredConfigurationError,
    FredError,
    MacroObservation,
    check_frequency,
)
from argos.news.manifest import (
    FileDigest,
    Manifest,
    NewsManifestBuilder,
    read_manifest,
    write_manifest,
)
from argos.news.models import (
    SCHEMA_VERSIONS,
    Direction,
    EventType,
    FinancialEvent,
    NewsArticle,
)
from argos.news.normalize import (
    NormalizationResult,
    normalize_articles,
    parse_utc,
    reject_reason,
)
from argos.news.providers import (
    MassiveNewsProvider,
    NewsProvider,
    ProviderConfigurationError,
    ProviderError,
    ProviderResponse,
    provider_for,
    redact,
)
from argos.news.storage import NewsStore, StorageError, WriteStats

__all__ = [
    "EVENT_SCHEMA",
    "SCHEMA_VERSIONS",
    "SYSTEM_PROMPT",
    "Analyzer",
    "AnalysisCache",
    "AnalyzerError",
    "AnalyzerIdentity",
    "AnalysisResult",
    "AuditFailure",
    "AuditReport",
    "DictCache",
    "DeterministicAnalyzer",
    "Direction",
    "EdgarClient",
    "EdgarConfigurationError",
    "EdgarError",
    "EventDataset",
    "EventType",
    "FileDigest",
    "Filing",
    "FinancialEvent",
    "FredClient",
    "FredConfigurationError",
    "FredError",
    "Finding",
    "JsonlCache",
    "LLMAnalyzer",
    "LLMConfigurationError",
    "MacroObservation",
    "Manifest",
    "MassiveNewsProvider",
    "NewsArticle",
    "NewsFeaturePolicy",
    "NewsManifestBuilder",
    "NewsProvider",
    "NewsStore",
    "NormalizationResult",
    "PromptRef",
    "ProviderConfigurationError",
    "ProviderError",
    "ProviderResponse",
    "RELEVANT_FORMS",
    "Severity",
    "StorageError",
    "WriteStats",
    "align_score_to_index",
    "analyze_many",
    "article_identity",
    "article_key",
    "audit_from_store",
    "build_analyzer",
    "build_dataset",
    "build_prompt_ref",
    "canonical_url",
    "check_frequency",
    "cumulative_news_score",
    "daily_news_score",
    "dedupe_articles",
    "events_available_as_of",
    "events_for_trading_range",
    "feature_for_dataset",
    "filing_to_event_type",
    "first_decision_date",
    "load_events",
    "normalize_articles",
    "parse_llm_response",
    "parse_utc",
    "provider_for",
    "read_manifest",
    "redact",
    "reject_reason",
    "run_audit",
    "to_event",
    "write_events",
    "write_manifest",
]
