"""The manifest as a falsifiable claim, and stable serialization.

A manifest exists so that "this dataset is what it says it is" can be checked
rather than believed. That makes two properties load-bearing:

- determinism, because the manifest records SHA-256 of files whose bytes must
  reproduce across runs, and a hash that changes for identical input makes the
  whole record meaningless;
- honest counting, because a field that under-reports failure is worse than one
  that does not exist.

Both are tested here against the real classes, offline.
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import replace

import pytest

from argos.news.manifest import (
    MANIFEST_SCHEMA_VERSION,
    FileDigest,
    Manifest,
    ManifestError,
    NewsManifestBuilder,
    read_manifest,
    write_manifest,
)
from argos.news.models import UTC, Direction, FinancialEvent, NewsArticle

NOW = dt.datetime(2025, 8, 1, 12, 0, tzinfo=UTC)


def article(index: int, *, tickers=("AAPL",), published=None) -> NewsArticle:
    return NewsArticle(
        article_id=f"a{index}",
        source="synthetic",
        title=f"Headline {index}",
        summary=f"Body {index}.",
        published_at=published or dt.datetime(2025, 7, 31, 20, 5, tzinfo=UTC),
        retrieved_at=NOW,
        publisher="Reuters",
        url=f"https://example.invalid/{index}",
        tickers=tickers,
        raw_source_id=f"n{index}",
    )


def event(
    index: int,
    *,
    analyzer="openai",
    version="1",
    model="gpt-test",
    prompt="p1",
    failed: bool = False,
    tickers=("AAPL",),
) -> FinancialEvent:
    return FinancialEvent(
        event_id=f"e{index}",
        article_id=f"a{index}",
        source="synthetic",
        published_at=dt.datetime(2025, 7, 31, 20, 5, tzinfo=UTC),
        event_type="earnings",
        direction=Direction.UNKNOWN if failed else Direction.POSITIVE,
        magnitude=None,
        confidence=0.0 if failed else 0.7,
        rationale=None if failed else "keyword match",
        tickers=tickers,
        analyzer=analyzer,
        analyzer_version=version,
        model=model,
        prompt_ref=prompt,
        analyzed_at=NOW,
    )


class TestFileDigest:
    def test_the_digest_hashes_content_not_metadata(self, tmp_path):
        target = tmp_path / "a.json"
        target.write_text('{"x": 1}', encoding="utf-8")
        first = FileDigest.of(target)

        target.touch()  # same bytes, new mtime
        assert FileDigest.of(target) == first

    def test_changed_bytes_change_the_digest(self, tmp_path):
        target = tmp_path / "a.json"
        target.write_text('{"x": 1}', encoding="utf-8")
        first = FileDigest.of(target)
        target.write_text('{"x": 2}', encoding="utf-8")
        assert FileDigest.of(target).sha256 != first.sha256

    def test_the_size_field_matches_the_bytes_on_disk(self, tmp_path):
        target = tmp_path / "a.json"
        target.write_text('{"x": 1}', encoding="utf-8")
        assert FileDigest.of(target).bytes == target.stat().st_size

    def test_a_root_makes_the_recorded_path_relocatable(self, tmp_path):
        target = tmp_path / "raw" / "a.json"
        target.parent.mkdir()
        target.write_text("{}", encoding="utf-8")
        assert FileDigest.of(target, root=tmp_path).path == "raw/a.json"

    def test_without_a_root_the_path_is_kept_as_given(self, tmp_path):
        target = tmp_path / "a.json"
        target.write_text("{}", encoding="utf-8")
        assert FileDigest.of(target).path == str(target)

    def test_a_missing_file_raises_rather_than_recording_an_empty_digest(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            FileDigest.of(tmp_path / "nope.json")


class TestStableSerialization:
    def test_the_same_manifest_serializes_to_identical_bytes_twice(self):
        manifest = Manifest(name="d", created_at=NOW, article_count=3, event_count=2)
        assert manifest.to_json() == manifest.to_json()

    def test_key_order_does_not_affect_the_output(self):
        a = Manifest(name="d", created_at=NOW, provider_limits={"b": 2, "a": 1})
        b = Manifest(name="d", created_at=NOW, provider_limits={"a": 1, "b": 2})
        assert a.to_json() == b.to_json()

    def test_the_output_is_sorted(self):
        raw = Manifest(name="d", created_at=NOW).to_json()
        keys = [line.split('"')[1] for line in raw.splitlines() if ": " in line]
        assert keys == sorted(keys)

    def test_two_identically_built_manifests_agree_except_for_created_at(self):
        # Everything the pipeline controls must be byte-identical; only the build
        # timestamp may move.
        def build() -> Manifest:
            builder = NewsManifestBuilder("d", provider="synthetic")
            builder.record_articles([article(1), article(2)])
            builder.record_events([event(1)])
            builder.record_rejections({0: "missing published_at", 1: "missing published_at"})
            builder.record_duplicates(2)
            return builder.build()

        first, second = build(), build()
        stamped = (replace(first, created_at=NOW), replace(second, created_at=NOW))
        assert stamped[0].to_json() == stamped[1].to_json()

    def test_repeated_pipeline_runs_produce_byte_identical_files(self, tmp_path):
        # The property that makes SHA-256 in a manifest worth recording. The two
        # runs live in different directories, so only the content can match --
        # which is the point: digests identify content, not locations.
        def build(where: str) -> tuple[int, str]:
            root = tmp_path / where
            (root / "normalized").mkdir(parents=True)
            target = root / "normalized" / "d.jsonl"
            target.write_text(
                "".join(
                    json.dumps(a.to_dict(), sort_keys=True, ensure_ascii=False) + "\n"
                    for a in [article(1), article(2), article(3)]
                ),
                encoding="utf-8",
            )
            digest = FileDigest.of(target)
            return digest.bytes, digest.sha256

        assert build("run-one") == build("run-two")

    def test_writing_a_manifest_is_atomic_and_leaves_no_temp_file(self, tmp_path):
        target = tmp_path / "nested" / "manifest.json"
        write_manifest(Manifest(name="d", created_at=NOW), target)
        assert target.exists()
        assert not list(tmp_path.rglob("*.tmp"))


class TestRoundTrip:
    def test_to_dict_then_from_dict_reproduces_the_manifest(self):
        builder = NewsManifestBuilder(
            "d",
            provider="synthetic",
            requested_window=(dt.datetime(2020, 1, 1, tzinfo=UTC), None),
            provider_limits={"requests_per_minute": 5},
            unknowns=("two-year history limit unverified",),
            notes=("synthetic",),
        )
        builder.record_articles([article(1)])
        builder.record_events([event(1)])
        builder.record_rejections({0: "missing published_at"})
        builder.record_duplicates(1)
        original = builder.build()

        restored = Manifest.from_dict(original.to_dict())
        assert restored.name == original.name
        assert restored.coverage_start == original.coverage_start
        assert restored.requested_window == original.requested_window
        assert restored.unknowns == original.unknowns
        assert restored.provider_limits == original.provider_limits
        assert restored.analyzers == original.analyzers
        assert restored.rejected == original.rejected
        assert restored.rejection_reasons == original.rejection_reasons
        assert restored.duplicates_dropped == original.duplicates_dropped

    def test_the_written_form_reads_back_identically(self, tmp_path):
        # Compared through json.loads, because tuples become lists on the way
        # through a JSON document; anything else would be a formatting artifact.
        manifest = Manifest(name="d", created_at=NOW, article_count=1)
        write_manifest(manifest, tmp_path / "manifest.json")
        assert read_manifest(tmp_path / "manifest.json") == json.loads(manifest.to_json())

    def test_timestamps_are_normalized_to_utc_on_construction(self):
        moment = dt.datetime(2025, 8, 1, 12, 0, tzinfo=dt.timezone(dt.timedelta(hours=2)))
        assert Manifest(name="d", created_at=moment).created_at == dt.datetime(
            2025, 8, 1, 10, 0, tzinfo=UTC
        )

    def test_a_naive_timestamp_is_refused_rather_than_assumed_utc(self):
        # Assuming UTC here would silently shift every coverage boundary.
        with pytest.raises(ManifestError, match="no UTC offset"):
            Manifest.from_dict(
                {"name": "d", "created_at": "2025-08-01T12:00:00", "schema_version": 1}
            )

    def test_an_unreadable_timestamp_is_refused(self):
        with pytest.raises(ManifestError, match="unreadable"):
            Manifest.from_dict(
                {"name": "d", "created_at": "yesterday", "schema_version": 1}
            )

    def test_a_missing_created_at_is_stamped_now_and_says_so_by_being_aware(self):
        restored = Manifest.from_dict({"name": "d", "schema_version": 1})
        assert restored.created_at.tzinfo is not None
        assert restored.created_at <= dt.datetime.now(UTC)

    def test_unknown_fields_in_a_stored_manifest_are_ignored(self):
        # A manifest written by a newer build must still be readable by this one
        # rather than raising on a field it has never heard of.
        payload = {"name": "d", "schema_version": 1, "created_at": NOW.isoformat(),
                   "some_future_field": {"nested": 1}}
        assert Manifest.from_dict(payload).name == "d"


class TestReadManifest:
    def test_a_missing_file_is_a_filenotfounderror(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            read_manifest(tmp_path / "nope.json")

    def test_a_foreign_schema_version_is_refused(self, tmp_path):
        target = tmp_path / "manifest.json"
        target.write_text(json.dumps({"name": "d", "schema_version": 999}), encoding="utf-8")
        with pytest.raises(ValueError, match="schema version"):
            read_manifest(target)

    def test_the_current_schema_version_is_accepted(self, tmp_path):
        target = tmp_path / "manifest.json"
        target.write_text(
            json.dumps({"name": "d", "schema_version": MANIFEST_SCHEMA_VERSION}), encoding="utf-8"
        )
        assert read_manifest(target)["name"] == "d"

    def test_a_non_object_document_is_refused(self, tmp_path):
        target = tmp_path / "manifest.json"
        target.write_text("[1, 2, 3]", encoding="utf-8")
        with pytest.raises(ValueError, match="manifest object"):
            read_manifest(target)


class TestAnalyzerAccounting:
    def test_failures_are_counted_per_event_not_once_per_analyzer(self):
        # A dataset where the model failed on every record must not report one
        # failure, which is what reading this as "did the first event fail" did.
        builder = NewsManifestBuilder("d")
        builder.record_events([event(i, failed=True) for i in range(3)])
        record = builder.build().analyzers[0]
        assert record["events"] == 3
        assert record["failed"] == 3

    def test_successes_are_not_counted_as_failures(self):
        builder = NewsManifestBuilder("d")
        builder.record_events([event(i) for i in range(4)])
        assert builder.build().analyzers[0]["failed"] == 0

    def test_a_mix_of_successes_and_failures_is_counted_separately(self):
        builder = NewsManifestBuilder("d")
        builder.record_events(
            [event(1), event(2, failed=True), event(3), event(4, failed=True)]
        )
        record = builder.build().analyzers[0]
        assert (record["events"], record["failed"]) == (4, 2)

    def test_different_analyzers_get_their_own_rows(self):
        builder = NewsManifestBuilder("d")
        builder.record_events([event(1, analyzer="openai"), event(2, analyzer="deterministic")])
        rows = builder.build().analyzers
        assert sorted(r["analyzer"] for r in rows) == ["deterministic", "openai"]

    def test_different_prompts_get_their_own_rows(self):
        # Same model, same version, different prompt: different interpretations,
        # so reporting one row would attribute every event to the first prompt.
        builder = NewsManifestBuilder("d")
        builder.record_events([event(1, prompt="p1"), event(2, prompt="p2")])
        rows = builder.build().analyzers
        assert len(rows) == 2
        assert sorted(r["prompt_ref"] for r in rows) == ["p1", "p2"]

    def test_rows_are_ordered_deterministically(self):
        builder = NewsManifestBuilder("d")
        builder.record_events(
            [event(1, analyzer="openai", model="b"), event(2, analyzer="openai", model="a"),
             event(3, analyzer="deterministic", model=None)]
        )
        again = NewsManifestBuilder("d")
        again.record_events(
            [event(1, analyzer="openai", model="b"), event(2, analyzer="openai", model="a"),
             event(3, analyzer="deterministic", model=None)]
        )
        assert builder.build().analyzers == again.build().analyzers


class TestBuilder:
    def test_coverage_spans_the_earliest_and_latest_article(self):
        builder = NewsManifestBuilder("d")
        builder.record_articles(
            [
                article(1, published=dt.datetime(2025, 3, 1, tzinfo=UTC)),
                article(2, published=dt.datetime(2025, 7, 1, tzinfo=UTC)),
                article(3, published=dt.datetime(2025, 5, 1, tzinfo=UTC)),
            ]
        )
        manifest = builder.build()
        assert manifest.coverage_start == dt.datetime(2025, 3, 1, tzinfo=UTC)
        assert manifest.coverage_end == dt.datetime(2025, 7, 1, tzinfo=UTC)

    def test_tickers_are_unioned_and_sorted(self):
        builder = NewsManifestBuilder("d")
        builder.record_articles([article(1, tickers=("MSFT", "AAPL")), article(2, tickers=())])
        assert builder.build().tickers == ("AAPL", "MSFT")

    def test_counts_accumulate_across_calls(self):
        builder = NewsManifestBuilder("d")
        builder.record_articles([article(1), article(2)])
        builder.record_articles([article(3)])
        builder.record_events([event(1)])
        builder.record_events([event(2)])
        builder.record_duplicates(1)
        builder.record_duplicates(2)
        manifest = builder.build()
        assert (manifest.article_count, manifest.event_count, manifest.duplicates_dropped) == (
            3, 2, 3,
        )

    def test_rejection_reasons_are_bucketed_by_text(self):
        builder = NewsManifestBuilder("d")
        builder.record_rejections(
            {
                0: "missing published_at",
                1: "missing published_at",
                2: "bad ticker list",
            }
        )
        manifest = builder.build()
        assert manifest.rejected == 3
        assert manifest.rejection_reasons == {"missing published_at": 2, "bad ticker list": 1}

    def test_rejection_reasons_accept_a_sequence_of_pairs(self):
        builder = NewsManifestBuilder("d")
        builder.record_rejections([(0, "missing published_at"), (1, "missing published_at")])
        assert builder.build().rejection_reasons == {"missing published_at": 2}

    def test_a_reason_with_a_detail_suffix_is_bucketed_by_its_head(self):
        builder = NewsManifestBuilder("d")
        builder.record_rejections({0: "missing published_at: got None"})
        assert builder.build().rejection_reasons == {"missing published_at": 1}

    def test_the_built_in_unknown_is_not_repeated(self):
        builder = NewsManifestBuilder("d")
        builder.add_unknown("same").add_unknown("same").add_note("n").add_note("n")
        manifest = builder.build()
        assert manifest.unknowns == ("same",)
        assert manifest.notes == ("n",)

    def test_a_builder_with_a_root_records_relative_digests(self, tmp_path):
        (tmp_path / "raw").mkdir()
        target = tmp_path / "raw" / "d.json"
        target.write_text("{}", encoding="utf-8")
        manifest = NewsManifestBuilder("d", root=tmp_path).add_raw(target).build()
        assert manifest.raw_files[0].path == "raw/d.json"


class TestSummary:
    def test_the_summary_leads_with_the_facts(self):
        text = Manifest(name="d", created_at=NOW, article_count=5, event_count=3).summary()
        assert "d" in text and "5 articles" in text and "3 events" in text

    def test_a_late_coverage_start_is_called_out(self):
        manifest = Manifest(
            name="d",
            created_at=NOW,
            coverage_start=dt.datetime(2024, 1, 1, tzinfo=UTC),
            requested_window=(dt.datetime(2020, 1, 1, tzinfo=UTC), None),
        )
        assert "WARNING" in manifest.summary()

    def test_coverage_matching_the_request_says_nothing(self):
        manifest = Manifest(
            name="d",
            created_at=NOW,
            coverage_start=dt.datetime(2020, 1, 1, tzinfo=UTC),
            requested_window=(dt.datetime(2020, 1, 1, tzinfo=UTC), None),
        )
        assert "WARNING" not in manifest.summary()

    def test_an_empty_dataset_says_so_instead_of_claiming_coverage(self):
        assert "empty" in Manifest(name="d", created_at=NOW).summary()

    def test_rejections_and_duplicates_are_surfaced(self):
        manifest = Manifest(name="d", created_at=NOW, rejected=4, duplicates_dropped=2)
        text = manifest.summary()
        assert "4 records rejected" in text
        assert "2 duplicates dropped" in text

    def test_recorded_unknowns_are_counted(self):
        text = Manifest(name="d", created_at=NOW, unknowns=("a", "b")).summary()
        assert "2 unverified item(s)" in text