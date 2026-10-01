"""The `news` CLI, exercised as a subprocess against a synthetic dataset.

These tests cover the operator-facing contract: the stages compose, each one
reads what the previous wrote, and the audit fails the command for the failure
modes that matter. Nothing here touches the network -- raw payloads are seeded
directly, which is what `fetch` would have written.
"""

from __future__ import annotations

import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_backtest.py"

PAYLOAD = {
    "results": [
        {
            "id": "n1",
            "headline": "Apple beats Q3 earnings expectations",
            "url": "https://www.reuters.com/markets/apple-q3-2025",
            "published_utc": "2025-07-31T20:05:00Z",
            "summary": "Apple beats Q3 earnings expectations.",
            "tickers": ["AAPL"],
            "source": "reuters",
        },
        {
            "id": "n2",
            "headline": "Fed holds rates steady",
            "url": "https://www.reuters.com/markets/fed-rates-2025",
            "published_utc": "2025-08-01T18:00:00Z",
            "summary": "Fed holds rates steady as inflation cools.",
            "tickers": ["SPY"],
            "source": "reuters",
        },
        {
            # No timestamp: the record that makes rejection counts worth having.
            "id": "n3",
            "headline": "Undated record",
            "url": "https://www.reuters.com/markets/undated",
            "summary": "no timestamp here",
            "tickers": ["AAPL"],
        },
    ]
}


def news(*argv: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "news", *argv], capture_output=True, text=True, timeout=180
    )


@pytest.fixture
def store(tmp_path: Path) -> Path:
    """A store with raw payloads already fetched."""
    root = tmp_path / "news"
    (root / "raw").mkdir(parents=True)
    (root / "raw" / "demo.json").write_text(json.dumps(PAYLOAD), encoding="utf-8")
    return root


@pytest.fixture
def built(store: Path) -> Path:
    """A store that has been normalized and analyzed."""
    news("normalize", "--dataset", "demo", "--news-root", str(store)).check_returncode()
    news("analyze", "--dataset", "demo", "--news-root", str(store)).check_returncode()
    return store


class TestPipeline:
    def test_normalize_writes_articles_and_a_manifest(self, store: Path) -> None:
        result = news("normalize", "--dataset", "demo", "--news-root", str(store))
        assert result.returncode == 0, result.stderr
        assert "wrote 2 articles" in result.stdout
        assert (store / "normalized" / "demo.jsonl").exists()
        assert (store / "demo.manifest.json").exists()

    def test_normalize_reports_why_records_were_dropped(self, store: Path) -> None:
        result = news("normalize", "--dataset", "demo", "--news-root", str(store))
        # A rejection count without the reason tells the operator nothing about
        # whether the feed is malformed or the schema changed.
        assert "missing published_at: 1" in result.stdout

    def test_analyze_derives_events(self, built: Path) -> None:
        events = (built / "events" / "demo.jsonl").read_text(encoding="utf-8")
        assert len(events.strip().splitlines()) == 2

    def test_analyze_counts_directions(self, store: Path) -> None:
        news("normalize", "--dataset", "demo", "--news-root", str(store)).check_returncode()
        result = news("analyze", "--dataset", "demo", "--news-root", str(store))
        assert result.returncode == 0, result.stderr
        assert "UNKNOWN: 1" in result.stdout

    def test_inspect_reports_row_counts(self, built: Path) -> None:
        result = news("inspect", "--dataset", "demo", "--news-root", str(built))
        assert result.returncode == 0, result.stderr
        assert "2 rows" in result.stdout

    def test_inspect_filters_to_one_dataset(self, built: Path) -> None:
        (built / "raw" / "other.json").write_text(json.dumps(PAYLOAD), encoding="utf-8")
        news("normalize", "--dataset", "other", "--news-root", str(built)).check_returncode()
        result = news("inspect", "--dataset", "demo", "--news-root", str(built))
        assert "demo" in result.stdout
        assert "other" not in result.stdout

    def test_the_manifest_describes_every_stage(self, built: Path) -> None:
        # analyze runs last, so it is the stage most likely to drop the earlier
        # stages' digests -- and then the audit could only verify its own output.
        manifest = json.loads((built / "demo.manifest.json").read_text(encoding="utf-8"))
        assert manifest["raw_files"] and manifest["normalized_files"] and manifest["event_files"]
        assert manifest["article_count"] == 2
        assert manifest["event_count"] == 2

    def test_the_manifest_keeps_counts_the_last_stage_cannot_see(self, built: Path) -> None:
        # Rejections are observed only during normalization. If the manifest
        # written by analyze dropped them, the record of what the data cost
        # would vanish as soon as the run finished.
        manifest = json.loads((built / "demo.manifest.json").read_text(encoding="utf-8"))
        assert manifest["rejected"] == 1
        assert manifest["rejection_reasons"] == {"missing published_at": 1}

    def test_repeated_normalize_yields_the_same_articles(self, built: Path) -> None:
        def articles() -> list[dict]:
            return [
                json.loads(line)
                for line in (built / "normalized" / "demo.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]

        before = articles()
        news("normalize", "--dataset", "demo", "--news-root", str(built)).check_returncode()
        after = articles()
        # The byte streams differ because each run re-stamps retrieved_at, which
        # is correct: it records when *this* fetch happened. What must not
        # change is which articles exist or when they were published.
        assert [a["article_id"] for a in after] == [a["article_id"] for a in before]
        assert [a["published_at"] for a in after] == [a["published_at"] for a in before]


class TestAuditCommand:
    def test_a_clean_dataset_passes(self, built: Path) -> None:
        result = news("audit", "--dataset", "demo", "--news-root", str(built))
        assert result.returncode == 0, result.stdout
        assert "audit passed" in result.stdout

    def test_a_missing_dataset_is_one_clear_line(self, store: Path) -> None:
        result = news("analyze", "--dataset", "ghost", "--news-root", str(store))
        assert result.returncode == 1
        assert "error [StorageError]" in result.stderr
        assert "Traceback" not in result.stderr

    def test_coverage_short_of_an_explicit_window_fails(self, built: Path) -> None:
        # The free-tier trap: a 2020 backtest fed two years of news. Passing
        # --expected-start is an assertion about the window, so falling short of
        # it has to be fatal rather than advisory.
        result = news(
            "audit", "--dataset", "demo", "--news-root", str(built),
            "--expected-start", "2020-01-01",
        )
        assert result.returncode == 1
        assert "coverage-truncated-start" in result.stdout

    def test_coverage_gap_is_advisory_without_the_flag(self, built: Path) -> None:
        # Without --expected-start the operator has not claimed a window, so a
        # short history is information rather than a contradiction.
        result = news("audit", "--dataset", "demo", "--news-root", str(built))
        assert result.returncode == 0
        assert "audit passed" in result.stdout

    def test_a_window_the_data_covers_passes(self, built: Path) -> None:
        # A day of slack is allowed, so the boundary case is the start date
        # itself rather than the day before.
        result = news(
            "audit", "--dataset", "demo", "--news-root", str(built),
            "--expected-start", "2025-07-31",
        )
        assert result.returncode == 0, result.stdout

    def test_an_amended_row_is_caught_by_the_manifest_hash(self, built: Path) -> None:
        # Same byte count, different content: the case a size check cannot see
        # and the reason the manifest records a hash.
        path = built / "normalized" / "demo.jsonl"
        lines = path.read_text(encoding="utf-8").splitlines()
        target = lines[0]
        index = target.index("AAPL")
        lines[0] = target[:index] + "AAQL" + target[index + 4 :]
        assert len(lines[0].encode()) == len(target.encode())
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        result = news("audit", "--dataset", "demo", "--news-root", str(built))
        assert result.returncode == 1
        assert "manifest-hash-mismatch" in result.stdout

    def test_warnings_as_errors_only_bites_when_warnings_exist(self, built: Path) -> None:
        strict = news(
            "audit", "--dataset", "demo", "--news-root", str(built), "--warnings-as-errors"
        )
        assert strict.returncode == 0, strict.stdout

    def test_json_output_is_machine_readable(self, built: Path) -> None:
        result = news("audit", "--dataset", "demo", "--news-root", str(built), "--json")
        summary = json.loads(result.stdout)
        assert summary == {"errors": 0, "findings": [], "passed": True, "warnings": 0}


class TestValidation:
    def test_openai_analyzer_requires_an_explicit_model(self, built: Path) -> None:
        # Never accept a default model id: a silent wrong model would make the
        # events unreproducible and the provenance lie.
        result = news(
            "analyze", "--dataset", "demo", "--news-root", str(built),
            "--analyzer", "openai",
        )
        assert result.returncode == 1
        assert "--model is required" in result.stderr
        assert "Traceback" not in result.stderr

    def test_an_unknown_analyzer_is_reported(self, built: Path) -> None:
        result = news(
            "analyze", "--dataset", "demo", "--news-root", str(built), "--analyzer", "gpt"
        )
        assert result.returncode == 1
        assert "Traceback" not in result.stderr

    def test_normalizing_nothing_is_a_clear_error(self, store: Path) -> None:
        result = news("normalize", "--dataset", "absent", "--news-root", str(store))
        assert result.returncode == 1
        assert "error [StorageError]" in result.stderr

    def test_fetch_rejects_an_inverted_window(self, store: Path) -> None:
        result = news(
            "fetch", "--dataset", "demo", "--news-root", str(store),
            "--start", "2025-01-01", "--end", "2024-01-01",
        )
        assert result.returncode == 1
        assert "must not precede" in result.stderr

    def test_fetch_rejects_zero_pages(self, store: Path) -> None:
        result = news(
            "fetch", "--dataset", "demo", "--news-root", str(store), "--max-pages", "0"
        )
        assert result.returncode == 1
        assert "--max-pages" in result.stderr


class TestManifestRoundTrip:
    def test_a_written_manifest_can_be_read_back(self, built: Path) -> None:
        # The audit reloads the manifest from disk, so a lossy round-trip would
        # silently drop the digests it depends on.
        from argos.news import Manifest

        raw = json.loads((built / "demo.manifest.json").read_text(encoding="utf-8"))
        restored = Manifest.from_dict(raw)
        # Compare through JSON: `to_dict` keeps tuples where the file has lists.
        assert json.loads(restored.to_json()) == raw
        assert restored.normalized_files[0].sha256 == raw["normalized_files"][0]["sha256"]

    def test_manifest_timestamps_survive_the_round_trip(self, built: Path) -> None:
        from argos.news import Manifest

        raw = json.loads((built / "demo.manifest.json").read_text(encoding="utf-8"))
        restored = Manifest.from_dict(raw)
        assert restored.coverage_start == dt.datetime(2025, 7, 31, 20, 5, tzinfo=dt.UTC)

    def test_a_naive_manifest_timestamp_is_refused(self) -> None:
        # Comparing a naive value against UTC ones raises a TypeError with no
        # hint about which field was wrong; a named error is actionable.
        from argos.news import Manifest, ManifestError

        with pytest.raises(ManifestError, match="no UTC offset"):
            Manifest.from_dict({"name": "s", "coverage_start": "2025-07-31T20:05:00"})