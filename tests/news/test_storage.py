"""Storage round-trips, idempotency, immutability and path safety.

Three properties matter for a store that a backtest depends on:

1. What goes in comes back out byte-for-byte identical in meaning. A schema
   change that silently drops a field is a silent data loss.
2. Re-running a fetch is safe. Operators re-run commands; append must not
   duplicate rows, because duplicated news events double the score.
3. A dataset name from the command line cannot escape the data directory.
"""

from __future__ import annotations

import json

import pytest
from tests.news.conftest import make_article, make_event

from argos.news import NewsArticle, StorageError
from argos.news.storage import NewsStore


@pytest.fixture
def store(tmp_path) -> NewsStore:
    return NewsStore(tmp_path / "news")


class TestRawStorage:
    def test_write_and_read_round_trip(self, store):
        payload = {"results": [{"id": "n1", "title": "Story"}], "status": "OK"}
        store.write_raw("capture", payload)
        assert store.read_raw("capture") == payload

    def test_raw_is_written_as_valid_json_with_trailing_newline(self, store):
        store.write_raw("capture", {"results": []})
        text = store.read_raw("capture") and (store.raw_dir / "capture.json").read_text()
        assert text.endswith("\n")
        assert json.loads(text) == {"results": []}

    def test_each_write_creates_only_its_own_directory(self, store):
        # Lazy creation: a fetch should not litter empty normalized/events dirs
        # that `news inspect` would then have to explain.
        assert not store.root.exists()
        store.write_raw("capture", {"results": []})
        assert store.raw_dir.is_dir()
        assert not store.normalized_dir.exists()
        store.write_articles("set", [make_article()])
        assert store.normalized_dir.is_dir()
        assert not store.events_dir.exists()

    def test_ensure_dirs_creates_the_full_layout(self, store):
        store.ensure_dirs()
        assert store.raw_dir.is_dir()
        assert store.normalized_dir.is_dir()
        assert store.events_dir.is_dir()

    def test_raw_is_immutable_by_default(self, store):
        # A raw capture is evidence of what the provider said at a moment in
        # time. Overwriting it silently would let a manifest describe a dataset
        # that no longer exists.
        store.write_raw("capture", {"results": [{"id": "n1"}]})
        with pytest.raises(StorageError, match="immutable"):
            store.write_raw("capture", {"results": [{"id": "DIFFERENT"}]})
        assert store.read_raw("capture") == {"results": [{"id": "n1"}]}

    def test_overwrite_requires_explicit_opt_in(self, store):
        store.write_raw("capture", {"results": []})
        store.write_raw("capture", {"results": [{"id": "n2"}]}, overwrite=True)
        assert store.read_raw("capture") == {"results": [{"id": "n2"}]}

    def test_missing_raw_raises(self, store):
        with pytest.raises(StorageError, match="no raw dataset"):
            store.read_raw("absent")

    def test_iter_raw_records_from_wrapped_payload(self, store):
        store.write_raw("capture", {"results": [{"id": "a"}, {"id": "b"}]})
        assert [r["id"] for r in store.iter_raw_records("capture")] == ["a", "b"]

    def test_iter_raw_records_from_bare_list(self, store):
        store.write_raw("capture", [{"id": "a"}])
        assert [r["id"] for r in store.iter_raw_records("capture")] == ["a"]

    def test_iter_raw_records_from_single_object(self, store):
        store.write_raw("capture", {"id": "a", "title": "one story"})
        assert [r["id"] for r in store.iter_raw_records("capture")] == ["a"]

    def test_iter_raw_records_skips_non_dict_rows(self, store):
        store.write_raw("capture", {"results": [{"id": "a"}, "junk", None, 42]})
        assert [r["id"] for r in store.iter_raw_records("capture")] == ["a"]

    def test_object_without_a_list_is_read_as_one_record(self, store):
        # A capture holding a single story, rather than a batch. Supporting this
        # lets a hand-built fixture normalize without a conversion step.
        store.write_raw("capture", {"id": "n1", "title": "one story"})
        assert [r["id"] for r in store.iter_raw_records("capture")] == ["n1"]

    def test_scalar_payload_is_rejected(self, store):
        store.write_raw("capture", 42)
        with pytest.raises(StorageError, match="cannot extract records"):
            list(store.iter_raw_records("capture"))


class TestNormalizedStorage:
    def test_articles_round_trip(self, store, article):
        store.write_articles("set", [article])
        assert store.read_articles("set") == (article,)

    def test_multiple_articles_round_trip(self, store, payloads):
        from tests.news.conftest import NOW

        from argos.news import normalize_articles

        articles = normalize_articles(payloads, source="massive", retrieved_at=NOW).articles
        store.write_articles("set", articles)
        assert store.read_articles("set") == articles

    def test_read_returns_models_not_dicts(self, store, article):
        store.write_articles("set", [article])
        assert all(isinstance(a, NewsArticle) for a in store.read_articles("set"))

    def test_write_reports_count(self, store):
        stats = store.write_articles("set", [make_article(raw_source_id=f"n{i}") for i in range(3)])
        assert stats.written == 3
        assert stats.mode == "w"

    def test_write_mode_replaces(self, store):
        store.write_articles("set", [make_article(raw_source_id="n1")])
        store.write_articles("set", [make_article(raw_source_id="n2")])
        assert len(store.read_articles("set")) == 1

    def test_append_skips_already_present_ids(self, store):
        # Re-running a fetch over an overlapping range must not duplicate rows.
        # Duplicate events would double the news score for that day.
        store.write_articles("set", [make_article(raw_source_id="n1")])
        stats = store.write_articles("set", [make_article(raw_source_id="n1")], append=True)
        assert stats.written == 0
        assert len(store.read_articles("set")) == 1

    def test_append_adds_only_new_ids(self, store):
        store.write_articles("set", [make_article(raw_source_id="n1")])
        store.write_articles(
            "set",
            [make_article(raw_source_id="n1"), make_article(raw_source_id="n2")],
            append=True,
        )
        assert {a.raw_source_id for a in store.read_articles("set")} == {"n1", "n2"}

    def test_append_preserves_existing_order(self, store):
        store.write_articles("set", [make_article(raw_source_id="n1")])
        store.write_articles("set", [make_article(raw_source_id="n2")], append=True)
        assert [a.raw_source_id for a in store.read_articles("set")] == ["n1", "n2"]

    def test_append_to_missing_file_creates_it(self, store):
        stats = store.write_articles("fresh", [make_article()], append=True)
        assert stats.written == 1

    def test_repeated_appends_converge(self, store):
        articles = [make_article(raw_source_id=f"n{i}") for i in range(3)]
        for _ in range(5):
            store.write_articles("set", articles, append=True)
        assert len(store.read_articles("set")) == 3

    def test_missing_dataset_raises(self, store):
        with pytest.raises(StorageError, match="no normalized dataset"):
            store.read_articles("absent")

    def test_malformed_line_reports_line_number(self, store):
        path = store.normalized_dir / "broken.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('not json\n{"article_id": "a"}\n', encoding="utf-8")
        with pytest.raises(StorageError, match=":1 is not valid JSON"):
            store.read_articles("broken")

    def test_incomplete_record_reports_the_missing_field(self, store):
        path = store.normalized_dir / "incomplete.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"article_id": "a"}\n', encoding="utf-8")
        with pytest.raises(KeyError, match="source"):
            store.read_articles("incomplete")

    def test_non_object_line_is_rejected(self, store):
        path = store.normalized_dir / "broken.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('["not", "an", "object"]\n', encoding="utf-8")
        with pytest.raises(StorageError, match="expected an object"):
            store.read_articles("broken")

    def test_blank_lines_are_skipped(self, store, article):
        store.write_articles("set", [article])
        path = store.normalized_dir / "set.jsonl"
        path.write_text(path.read_text(encoding="utf-8") + "\n\n", encoding="utf-8")
        assert len(store.read_articles("set")) == 1

    def test_unicode_is_preserved(self, store):
        article = make_article(title="Bancrectificación del banco — €1.234M")
        store.write_articles("set", [article])
        assert store.read_articles("set")[0].title == "Bancrectificación del banco — €1.234M"

    def test_utf8_is_written_literally_not_escaped(self, store):
        store.write_articles("set", [make_article(title="Ação Corretiva €1M")])
        raw = (store.normalized_dir / "set.jsonl").read_text(encoding="utf-8")
        assert "Ação Corretiva €1M" in raw


class TestEventStorage:
    def test_events_round_trip(self, store):
        event = make_event()
        store.write_events("set", [event])
        assert store.read_events("set") == (event,)

    def test_append_skips_by_event_id_not_article_id(self, store):
        # Both records serialize an article_id. Indexing an event file by
        # article_id would match nothing and let every append duplicate.
        store.write_events("set", [make_event()])
        stats = store.write_events("set", [make_event()], append=True)
        assert stats.written == 0
        assert len(store.read_events("set")) == 1

    def test_distinct_events_for_one_article_both_persist(self, store):
        # Two analyzers disagreeing about one article is a real scenario: the
        # provenance columns exist precisely so both can be recorded.
        first = make_event(analyzer="deterministic")
        second = make_event(analyzer="llm", direction="NEGATIVE")
        store.write_events("set", [first, second])
        assert len(store.read_events("set")) == 2

    def test_repeated_appends_converge(self, store):
        events = [make_event(article_id=f"provider:massive:n{i}") for i in range(4)]
        for _ in range(5):
            store.write_events("set", events, append=True)
        assert len(store.read_events("set")) == 4


class TestPathSafety:
    @pytest.mark.parametrize(
        "name",
        ["../escape", "../../etc/passwd", "sub/dir", "back\\slash", "", ".", ".."],
    )
    def test_traversal_names_are_refused(self, store, name):
        # A dataset name arrives from the CLI. Without this check,
        # `--name ../../etc/passwd` writes outside the data directory.
        with pytest.raises(StorageError, match="invalid dataset name|dataset name must"):
            store.write_raw(name, {"results": []})
        with pytest.raises(StorageError, match="invalid dataset name|dataset name must"):
            store.write_articles(name, [make_article()])
        with pytest.raises(StorageError, match="invalid dataset name|dataset name must"):
            store.write_events(name, [make_event()])

    def test_legitimate_names_are_accepted(self, store):
        for name in ("aapl-2025", "capture_01", "news.v2"):
            store.write_raw(name, {"results": []})

    def test_reads_are_also_guarded(self, store):
        for reader in (store.read_raw, store.read_articles, store.read_events):
            with pytest.raises(StorageError):
                reader("../escape")


class TestDescribe:
    def test_reports_layout_and_row_counts(self, store, article):
        store.write_raw("capture", {"results": []})
        store.write_articles("set", [article, make_article(raw_source_id="n2")])
        store.write_events("set", [make_event()])
        described = store.describe()
        assert described["root"] == str(store.root)
        assert described["raw"]["datasets"][0]["name"] == "capture"
        assert described["normalized"]["datasets"][0]["rows"] == 2
        assert described["events"]["datasets"][0]["rows"] == 1

    def test_describe_works_on_empty_store(self, store):
        described = store.describe()
        assert described["raw"]["datasets"] == []
        assert described["normalized"]["datasets"] == []

    def test_describe_is_json_serializable(self, store):
        store.write_articles("set", [make_article()])
        json.dumps(store.describe())