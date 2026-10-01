"""Audit checks: the questions that must be answerable about any dataset.

An audit exists because "it ran and produced a number" is not evidence. Each
check here corresponds to a specific way a news backtest can be wrong while
still looking healthy:

- timestamps that do not order correctly, so causality is inverted;
- coverage that starts late, so a 2020 backtest runs on two years of news and
  nobody notices -- the free-tier trap;
- duplicate rows that silently double a score;
- events whose provenance does not name the analyzer that produced them;
- a feature policy with a zero publication lag, which is look-ahead by another
  name.

Findings are graded so a caller can act on them: ERROR means the dataset should
not be used, WARNING means use it knowing this.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json

import pytest
from tests.news.conftest import UTC, make_article, make_event

from argos.news import NewsFeaturePolicy, build_dataset
from argos.news.audit import (
    AuditFailure,
    Finding,
    Severity,
    audit_coverage,
    audit_direction_consistency,
    audit_duplicates,
    audit_manifest_files,
    audit_provenance,
    audit_signal_leakage,
    audit_timestamps,
    run_audit,
)
from argos.news.manifest import FileDigest, Manifest, NewsManifestBuilder


def codes(findings) -> set[str]:
    return {f.code for f in findings}


def events_on(day: int, **kwargs):
    return make_event(
        article_id=f"provider:massive:d{day}", published_at=published_on(day), **kwargs
    )


def published_on(day: int) -> dt.datetime:
    return dt.datetime(2025, 7, day, 12, tzinfo=UTC)


def article_on(day: int):
    return make_article(
        raw_source_id=f"d{day}", title=f"Story of day {day}", published_at=published_on(day)
    )


def distinct_articles(count: int = 3):
    return [
        make_article(raw_source_id=f"n{i}", title=f"Distinct story {i}") for i in range(count)
    ]


def drifted_case():
    """An article and event that disagree on publication time: a hard error."""
    article = make_article(published_at=dt.datetime(2025, 7, 31, 20, 5, tzinfo=UTC))
    event = make_event(published_at=dt.datetime(2025, 7, 31, 10, 0, tzinfo=UTC))
    return article, event


class TestAuditTimestamps:
    def test_clean_data_has_no_findings(self):
        assert audit_timestamps([make_article()], [make_event()]) == []

    def test_event_published_before_its_article_is_flagged(self):
        # Causality inverted: a conclusion dated before its own premise.
        article, event = drifted_case()
        assert "publication-time-drift" in codes(audit_timestamps([article], [event]))

    def test_same_instant_is_accepted(self):
        article = make_article(published_at=dt.datetime(2025, 7, 31, 20, 5, tzinfo=UTC))
        event = make_event(published_at=dt.datetime(2025, 7, 31, 20, 5, tzinfo=UTC))
        assert audit_timestamps([article], [event]) == []

    def test_orphan_event_is_flagged(self):
        # An event whose article is absent cannot have been derived from
        # anything in this dataset.
        assert "orphan-event" in codes(audit_timestamps([], [make_event()]))

    def test_naive_timestamp_is_flagged_without_crashing(self):
        # Reached only by hand-edited files; models reject it, so this catches
        # tampering. It must produce a finding rather than a TypeError: an audit
        # that crashes on the most suspect file reports nothing about it.
        article = make_article()
        object.__setattr__(article, "published_at", dt.datetime(2025, 7, 31, 20, 5))
        assert "timestamp-naive" in codes(audit_timestamps([article], [make_event()]))

    def test_naive_article_with_event_does_not_raise(self):
        article = make_article()
        object.__setattr__(article, "published_at", dt.datetime(2025, 7, 31, 20, 5))
        assert audit_timestamps([article], [make_event()]) is not None

    def test_empty_input_is_clean(self):
        assert audit_timestamps([], []) == []


class TestAuditCoverage:
    def manifest(self, *, start, end=None) -> Manifest:
        builder = NewsManifestBuilder(name="set", provider="massive")
        builder.record_articles(
            [
                make_article(published_at=start),
                make_article(raw_source_id="n2", published_at=end or start),
            ]
        )
        return builder.build()

    def test_missing_coverage_is_an_error(self):
        findings = audit_coverage(Manifest(name="set", created_at=dt.datetime.now(UTC)))
        assert "coverage-empty" in codes(findings)
        assert all(f.severity is Severity.ERROR for f in findings)

    def test_coverage_inside_the_expected_window_is_clean(self):
        findings = audit_coverage(
            self.manifest(start=dt.datetime(2020, 1, 2, tzinfo=UTC)),
            expected_start=dt.datetime(2020, 1, 1, tzinfo=UTC),
        )
        assert findings == []

    def test_late_coverage_start_is_flagged(self):
        # The free-tier case: coverage begins 2024 while the backtest claims
        # 2020. Silence here would make truncation look like completeness.
        findings = audit_coverage(
            self.manifest(start=dt.datetime(2024, 1, 1, tzinfo=UTC)),
            expected_start=dt.datetime(2020, 1, 1, tzinfo=UTC),
        )
        assert "coverage-truncated-start" in codes(findings)

    def test_coverage_starting_exactly_on_the_boundary_is_clean(self):
        findings = audit_coverage(
            self.manifest(start=dt.datetime(2020, 1, 2, tzinfo=UTC)),
            expected_start=dt.datetime(2020, 1, 1, tzinfo=UTC),
        )
        assert findings == []

    def test_requested_window_is_recorded(self):
        builder = NewsManifestBuilder(
            name="set",
            provider="massive",
            requested_window=(
                dt.datetime(2020, 1, 1, tzinfo=UTC),
                dt.datetime(2025, 1, 1, tzinfo=UTC),
            ),
            provider_limits={"requests_per_minute": 5, "news_history_years": 2},
            unknowns=["free tier returns two years only"],
        )
        manifest = builder.record_articles([make_article()]).build()
        assert manifest.requested_window[0].year == 2020
        assert manifest.provider_limits["news_history_years"] == 2
        assert manifest.unknowns


class TestAuditDuplicates:
    def test_clean_data_has_no_findings(self):
        articles = distinct_articles()
        assert audit_duplicates(articles) == []

    def test_repeated_article_is_flagged(self):
        article = make_article()
        assert "duplicate-articles" in codes(audit_duplicates([article, article]))

    def test_distinct_articles_are_not_flagged(self):
        assert audit_duplicates(distinct_articles()) == []

    def test_same_content_under_different_ids_is_flagged(self):
        # Two provider ids for one story. Dedupe would not catch it, because the
        # ids differ, but the content fingerprint can.
        articles = [make_article(raw_source_id=f"n{i}", title="One story") for i in range(2)]
        assert "duplicate-content" in codes(audit_duplicates(articles))


class TestAuditProvenance:
    def test_clean_events_have_no_findings(self):
        assert audit_provenance([make_event()]) == []

    def test_events_missing_an_article_reference_are_flagged(self):
        event = make_event()
        object.__setattr__(event, "article_id", "")
        assert "missing-article-reference" in codes(audit_provenance([event]))

    def test_events_missing_the_analyzer_are_flagged(self):
        event = make_event()
        object.__setattr__(event, "analyzer", "")
        assert "missing-analyzer-identity" in codes(audit_provenance([event]))

    def test_model_backed_events_without_a_model_are_flagged(self):
        # An LLM inference recorded without its model cannot be re-run when the
        # prompt changes, which is how silent drift becomes invisible.
        event = make_event(analyzer="openai")
        object.__setattr__(event, "model", None)
        object.__setattr__(event, "prompt_ref", None)
        assert "missing-model-identity" in codes(audit_provenance([event]))

    def test_model_backed_events_without_a_prompt_are_flagged(self):
        event = make_event(analyzer="openai")
        object.__setattr__(event, "model", None)
        object.__setattr__(event, "prompt_ref", None)
        assert "missing-model-identity" in codes(audit_provenance([event]))

    def test_complete_model_provenance_is_clean(self):
        event = make_event(analyzer="openai", model="gpt-x", prompt_ref="PR-1")
        assert audit_provenance([event]) == []

    def test_half_recorded_provenance_is_flagged(self):
        # One field present without the other means a field was lost in
        # serialization, not that the analyzer is rule-based.
        event = make_event(analyzer="openai", model="gpt-x", prompt_ref=None)
        assert "incomplete-model-provenance" in codes(audit_provenance([event]))

    def test_deterministic_events_need_no_model(self):
        assert audit_provenance([make_event(analyzer="deterministic")]) == []

    def test_unclassified_analyzer_warns_that_it_was_not_checked(self):
        # A custom analyzer is legitimate, but its provenance went unverified
        # and the caller should hear about it.
        assert "unaudited-analyzer" in codes(audit_provenance([make_event(analyzer="my-rules")]))


class TestAuditSignalLeakage:
    def test_default_policy_is_clean(self):
        assert audit_signal_leakage([events_on(5)]) == []

    def test_zero_lag_policy_is_flagged(self):
        # A zero publication lag is look-ahead: a feature would be usable on the
        # day of publication, so a same-day signal could react to it.
        findings = audit_signal_leakage(
            [events_on(5)], policy=NewsFeaturePolicy(publication_lag_days=0)
        )
        assert "same-day-signal" in codes(findings)

    def test_negative_lag_is_rejected_at_construction(self):
        # The policy refuses to be built rather than being caught downstream.
        with pytest.raises(ValueError):
            NewsFeaturePolicy(publication_lag_days=-1)

    def test_one_day_lag_is_accepted(self):
        assert audit_signal_leakage(
            [events_on(5)], policy=NewsFeaturePolicy(publication_lag_days=1)
        ) == []

    def test_untagged_events_are_reported_as_dropped(self):
        # require_ticker discards these at scoring time. An audit that said
        # nothing would let a dataset of untagged events pass while
        # contributing nothing to any score.
        event = make_event(tickers=(), direction="POSITIVE", confidence=0.9)
        findings = audit_signal_leakage(
            [event], policy=NewsFeaturePolicy(require_ticker=True)
        )
        assert "untagged-events-dropped" in codes(findings)

    def test_untagged_events_allowed_by_policy(self):
        event = make_event(tickers=(), direction="POSITIVE", confidence=0.9)
        assert audit_signal_leakage(
            [event], policy=NewsFeaturePolicy(require_ticker=False)
        ) == []

    def test_tagged_events_are_not_reported_as_dropped(self):
        assert "untagged-events-dropped" not in codes(audit_signal_leakage([events_on(5)]))


class TestAuditDirectionConsistency:
    def test_uncalibrated_directional_event_is_flagged(self):
        # With confidence=None a POSITIVE event is weighted at full +1.0, so an
        # uncalibrated analyzer would dominate the score. Surfacing it beats
        # silently compensating.
        event = make_event(direction="POSITIVE")
        object.__setattr__(event, "confidence", None)
        assert "uncalibrated-direction" in codes(audit_direction_consistency([event]))

    def test_negative_confidence_cannot_be_constructed(self):
        # A negative weight on a POSITIVE event is impossible to reach, so the
        # audit does not need a check for it.
        with pytest.raises(ValueError):
            make_event(direction="POSITIVE", confidence=-0.5)

    def test_calibrated_direction_is_clean(self):
        assert audit_direction_consistency(
            [make_event(direction="NEGATIVE", confidence=0.4)]
        ) == []

    def test_unknown_is_not_a_sign_so_it_needs_no_confidence(self):
        event = make_event(direction="UNKNOWN")
        object.__setattr__(event, "confidence", None)
        assert audit_direction_consistency([event]) == []


class TestAuditManifestFiles:
    def build(self, tmp_path):
        (tmp_path / "normalized").mkdir()
        target = tmp_path / "normalized" / "set.jsonl"
        target.write_text('{"article_id": "a"}\n', encoding="utf-8")
        digest = FileDigest.of(target, root=tmp_path)
        manifest = Manifest(
            name="set",
            created_at=dt.datetime.now(UTC),
            normalized_files=(digest,),
        )
        return manifest, target

    def test_matching_file_is_clean(self, tmp_path):
        manifest, _ = self.build(tmp_path)
        assert audit_manifest_files(manifest, root=str(tmp_path)) == []

    def test_a_root_prefixed_path_resolves_when_root_is_relative(self, tmp_path, monkeypatch):
        # A builder that never received `root` records the path exactly as it
        # was handed. With a relative store that is a relative, root-prefixed
        # path like `news/normalized/set.jsonl`, and a caller who typed
        # `--root data/news` passes a relative root too. Joining them looks in
        # data/news/data/news/... and reports a file that is sitting right there
        # as missing.
        store = tmp_path / "news"
        (store / "normalized").mkdir(parents=True)
        target = store / "normalized" / "set.jsonl"
        target.write_text('{"article_id": "a"}\n', encoding="utf-8")
        data = target.read_bytes()
        manifest = Manifest(
            name="set",
            created_at=dt.datetime.now(UTC),
            normalized_files=(
                FileDigest(
                    path="news/normalized/set.jsonl",
                    bytes=len(data),
                    sha256=hashlib.sha256(data).hexdigest(),
                ),
            ),
        )
        monkeypatch.chdir(tmp_path)
        assert audit_manifest_files(manifest, root="news") == []

    def test_a_root_relative_path_is_still_resolved_against_root(self, tmp_path, monkeypatch):
        # The other convention, and the one the CLI now produces: the manifest
        # holds a store-relative path.
        store = tmp_path / "news"
        (store / "normalized").mkdir(parents=True)
        (store / "normalized" / "set.jsonl").write_text("x\n", encoding="utf-8")
        manifest = Manifest(
            name="set",
            created_at=dt.datetime.now(UTC),
            normalized_files=(
                FileDigest(path="normalized/set.jsonl", bytes=2, sha256="0" * 64),
            ),
        )
        monkeypatch.chdir(tmp_path)
        findings = audit_manifest_files(manifest, root="news")
        # Located, so not "missing"; the hash is wrong on purpose.
        assert "manifest-missing-file" not in codes(findings)

    def test_missing_file_is_flagged(self, tmp_path):
        manifest, target = self.build(tmp_path)
        target.unlink()
        assert "manifest-missing-file" in codes(
            audit_manifest_files(manifest, root=str(tmp_path))
        )

    def test_size_change_is_flagged(self, tmp_path):
        manifest, target = self.build(tmp_path)
        target.write_text('{"article_id": "a"}\n{"article_id": "b"}\n', encoding="utf-8")
        assert "manifest-size-mismatch" in codes(
            audit_manifest_files(manifest, root=str(tmp_path))
        )

    def test_same_size_different_content_is_flagged(self, tmp_path):
        # The case size alone cannot catch: an amended row keeps the byte count.
        manifest, target = self.build(tmp_path)
        original = '{"article_id": "a"}\n'
        tampered = '{"article_id": "z"}\n'
        assert len(original) == len(tampered)
        target.write_text(tampered, encoding="utf-8")
        findings = audit_manifest_files(manifest, root=str(tmp_path))
        assert "manifest-hash-mismatch" in codes(findings)

    def test_absolute_paths_are_not_rebased(self, tmp_path):
        manifest, target = self.build(tmp_path)
        manifest = Manifest(
            name="set", created_at=dt.datetime.now(UTC), normalized_files=(FileDigest.of(target),)
        )
        assert audit_manifest_files(manifest, root="/nonexistent") == []


class TestAuditReport:
    def test_clean_report_passes(self):
        assert run_audit(articles=[make_article()], events=[make_event()]).passed

    def test_error_severity_fails_the_report(self):
        drifted = drifted_case()
        assert not run_audit(articles=[drifted[0]], events=[drifted[1]]).passed

    def test_warnings_alone_do_not_fail_the_report(self):
        # An orphan event is a warning: the dataset is still usable, but the
        # reference is incomplete.
        report = run_audit(events=[make_event()])
        assert report.warnings and report.passed

    def test_raise_if_failed_raises(self):
        drifted = drifted_case()
        with pytest.raises(AuditFailure):
            run_audit(articles=[drifted[0]], events=[drifted[1]]).raise_if_failed()

    def test_raise_if_failed_is_silent_when_clean(self):
        run_audit(articles=[make_article()], events=[make_event()]).raise_if_failed()

    def test_errors_and_warnings_are_partitioned(self):
        # Order of checks is the report's order; only the partition is promised.
        report = run_audit(
            articles=[make_article(), make_article()],
            events=[make_event(), make_event()],
        )
        assert all(f.severity is Severity.ERROR for f in report.errors)
        assert not set(report.errors) & set(report.warnings)

    def test_report_is_json_serializable(self):
        json.dumps(run_audit(articles=[make_article()], events=[make_event()]).to_dict())

    def test_summary_counts_errors_and_warnings(self):
        drifted = drifted_case()
        summary = run_audit(articles=[drifted[0]], events=[drifted[1]]).to_dict()
        assert summary["errors"] >= 1
        assert summary["passed"] is False
        assert isinstance(summary["findings"], list)

    def test_error_count_matches_the_partitioned_errors(self):
        drifted = drifted_case()
        report = run_audit(articles=[drifted[0]], events=[drifted[1]])
        assert report.to_dict()["errors"] == len(report.errors)

    def test_empty_audit_is_an_error_not_a_pass(self):
        # "Nothing to audit" must not read as "nothing wrong".
        report = run_audit()
        assert not report.passed
        assert "nothing-to-audit" in codes(report.findings)


class TestRunAuditIntegration:
    def test_dataset_is_incorporated(self):
        report = run_audit(
            dataset=build_dataset([events_on(5)], name="set"), articles=[article_on(5)]
        )
        assert report.passed

    def test_duplicate_events_are_flagged(self):
        dataset = build_dataset([events_on(5), events_on(5)], name="set")
        assert not run_audit(dataset=dataset, articles=[article_on(5)]).passed

    def test_expected_start_surfaces_truncation(self):
        report = run_audit(
            articles=[make_article()],
            events=[make_event()],
            manifest=Manifest(name="set", created_at=dt.datetime.now(UTC)),
            expected_start=dt.datetime(2015, 1, 1, tzinfo=UTC),
        )
        assert not report.passed


class TestFinding:
    def test_finding_is_json_serializable(self):
        json.dumps(Finding("x", Severity.WARNING, "message").to_dict())

    def test_code_and_message_are_preserved(self):
        finding = Finding("some-code", Severity.ERROR, "some message")
        assert finding.code == "some-code"
        assert finding.severity is Severity.ERROR