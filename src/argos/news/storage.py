"""Local, gitignored storage for the news pipeline.

Three separate stores, because mixing them destroys the audit trail:

    data/news/raw/         provider responses, byte-faithful JSON
    data/news/normalized/  `NewsArticle` records in ARGOS's schema
    data/news/events/      `FinancialEvent` records, each naming its article

JSONL rather than one big JSON file: appending a fetch must not require reading
and rewriting everything already collected, and a truncated write should cost
one line rather than the dataset. Writes go through a temp file plus `os.replace`
so a crash mid-write cannot leave a half-written file where a valid one was.

Nothing here is tracked by git. News data is large, licensed, and reproducible
from the provider; the code that produces it is the artifact worth versioning.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import tempfile
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

from argos.news.models import FinancialEvent, NewsArticle

RAW_DIRNAME = "raw"
NORMALIZED_DIRNAME = "normalized"
EVENTS_DIRNAME = "events"


class StorageError(RuntimeError):
    """Storage operation failed in a way the caller should surface, not hide."""


@dataclass(frozen=True, slots=True)
class WriteStats:
    """Counts returned by every write, so a caller can log what happened."""

    path: Path
    written: int
    mode: str

    @property
    def skipped(self) -> int:
        return 0


class NewsStore:
    """Filesystem-backed store for raw, normalized and event data.

    Layout, created on demand:

        <root>/raw/<name>.json
        <root>/normalized/<name>.jsonl
        <root>/events/<name>.jsonl

    `root` defaults to `<repo>/data/news`, which `.gitignore` excludes.
    """

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root)

    # -- paths ---------------------------------------------------------------

    @property
    def raw_dir(self) -> Path:
        return self.root / RAW_DIRNAME

    @property
    def normalized_dir(self) -> Path:
        return self.root / NORMALIZED_DIRNAME

    @property
    def events_dir(self) -> Path:
        return self.root / EVENTS_DIRNAME

    def ensure_dirs(self) -> None:
        for path in (self.raw_dir, self.normalized_dir, self.events_dir):
            path.mkdir(parents=True, exist_ok=True)

    def _target(self, directory: Path, name: str, suffix: str) -> Path:
        # A caller-supplied name becomes a filename, so refuse traversal rather
        # than letting `--name ../../etc/passwd` escape the data directory.
        if not name or name in (".", ".."):
            raise StorageError(f"invalid dataset name: {name!r}")
        if Path(name).name != name or "/" in name or "\\" in name:
            raise StorageError(f"dataset name must be a bare filename, got {name!r}")
        return directory / f"{name}{suffix}"

    # -- raw ----------------------------------------------------------------

    def write_raw(self, name: str, payload: object, *, overwrite: bool = False) -> WriteStats:
        """Persist a provider response verbatim.

        Overwrite is refused by default. A raw capture is evidence of what the
        provider said at a moment in time; silently replacing it would mean the
        manifest can describe a dataset that no longer exists.
        """
        path = self._target(self.raw_dir, name, ".json")
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and not overwrite:
            raise StorageError(
                f"{path} already exists; raw captures are immutable by default. "
                "Pass overwrite=True only if you are deliberately re-capturing."
            )
        body = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False)
        _atomic_write_text(path, body + "\n")
        return WriteStats(path=path, written=1, mode="w" if overwrite else "x")

    def read_raw(self, name: str) -> object:
        path = self._target(self.raw_dir, name, ".json")
        if not path.exists():
            raise StorageError(f"no raw dataset at {path}")
        return json.loads(path.read_text(encoding="utf-8"))

    def iter_raw_records(self, name: str) -> Iterator[dict]:
        """Yield records from a raw capture, accepting either container shape.

        Providers wrap their rows in `{"results": [...]}`; hand-built fixtures are
        often bare lists. Both are accepted so `normalize` works on a live fetch
        and on a checked-in sample without a conversion step.
        """
        payload = self.read_raw(name)
        if isinstance(payload, dict):
            results = payload.get("results")
            if results is None:
                # A single-record capture, or a wrapper with a different key.
                for value in payload.values():
                    if isinstance(value, list):
                        results = value
                        break
            if results is None:
                yield payload
                return
            payload = results
        if not isinstance(payload, list):
            raise StorageError(
                f"raw dataset {name} is neither a list nor an object containing one; "
                "cannot extract records"
            )
        for record in payload:
            if isinstance(record, dict):
                yield record

    # -- normalized ---------------------------------------------------------

    def write_articles(
        self, name: str, articles: Iterable[NewsArticle], *, append: bool = False
    ) -> WriteStats:
        """Write articles as JSONL.

        Append mode skips ids already present in the file, so re-running a fetch
        over an overlapping date range adds only what is new instead of
        duplicating rows that would later double-count in the news score.
        """
        path = self._target(self.normalized_dir, name, ".jsonl")
        path.parent.mkdir(parents=True, exist_ok=True)

        seen = self._read_ids(path, "article_id") if append else set()
        rows = 0
        duplicates = 0
        with _open_append_or_write(path, append) as handle:
            for article in articles:
                if append and article.article_id in seen:
                    duplicates += 1
                    continue
                handle.write(
                    json.dumps(article.to_dict(), sort_keys=True, ensure_ascii=False) + "\n"
                )
                seen.add(article.article_id)
                rows += 1
        if rows == 0 and duplicates:
            return WriteStats(path=path, written=0, mode="a")
        return WriteStats(path=path, written=rows, mode="a" if append else "w")

    def read_articles(self, name: str) -> tuple[NewsArticle, ...]:
        path = self._target(self.normalized_dir, name, ".jsonl")
        if not path.exists():
            raise StorageError(f"no normalized dataset at {path}")
        return tuple(self.iter_articles(name))

    def iter_articles(self, name: str) -> Iterator[NewsArticle]:
        path = self._target(self.normalized_dir, name, ".jsonl")
        if not path.exists():
            raise StorageError(f"no normalized dataset at {path}")
        for record in self._iter_jsonl(path):
            yield NewsArticle.from_dict(record)

    # -- events -------------------------------------------------------------

    def write_events(
        self, name: str, events: Iterable[FinancialEvent], *, append: bool = False
    ) -> WriteStats:
        path = self._target(self.events_dir, name, ".jsonl")
        path.parent.mkdir(parents=True, exist_ok=True)

        seen = self._read_ids(path, "event_id") if append else set()
        rows = 0
        with _open_append_or_write(path, append) as handle:
            for event in events:
                if append and event.event_id in seen:
                    continue
                handle.write(
                    json.dumps(event.to_dict(), sort_keys=True, ensure_ascii=False) + "\n"
                )
                seen.add(event.event_id)
                rows += 1
        return WriteStats(path=path, written=rows, mode="a" if append else "w")

    def read_events(self, name: str) -> tuple[FinancialEvent, ...]:
        path = self._target(self.events_dir, name, ".jsonl")
        if not path.exists():
            raise StorageError(f"no event dataset at {path}")
        return tuple(self.iter_events(name))

    def iter_events(self, name: str) -> Iterator[FinancialEvent]:
        path = self._target(self.events_dir, name, ".jsonl")
        if not path.exists():
            raise StorageError(f"no event dataset at {path}")
        for record in self._iter_jsonl(path):
            yield FinancialEvent.from_dict(record)

    # -- shared -------------------------------------------------------------

    @staticmethod
    def _iter_jsonl(path: Path) -> Iterator[dict]:
        with path.open("r", encoding="utf-8") as handle:
            for number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    record = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise StorageError(
                        f"{path}:{number} is not valid JSON: {exc}"
                    ) from exc
                if not isinstance(record, dict):
                    raise StorageError(
                        f"{path}:{number} is {type(record).__name__}, expected an object"
                    )
                yield record

    @staticmethod
    def _read_ids(path: Path, id_field: str) -> set[str]:
        """Collect the values of `id_field` from an existing JSONL file.

        The field name must be explicit. Articles and events both serialize an
        `article_id`, so a generic "whatever id is present" lookup would index an
        event file by article id, match nothing, and let an append duplicate
        every row it was meant to skip.
        """
        if not path.exists():
            return set()
        ids: set[str] = set()
        for record in NewsStore._iter_jsonl(path):
            value = record.get(id_field)
            if value:
                ids.add(str(value))
        return ids

    def describe(self) -> dict[str, object]:
        """Inventory of the store, for `news inspect` and the audit command."""
        def count(directory: Path) -> dict[str, object]:
            files = sorted(directory.glob("*.jsonl")) if directory.exists() else []
            return {
                "dir": str(directory),
                "datasets": [
                    {
                        "name": path.stem,
                        "rows": sum(1 for _ in NewsStore._iter_jsonl(path)),
                        "bytes": path.stat().st_size,
                    }
                    for path in files
                ],
            }

        raws = sorted(self.raw_dir.glob("*.json")) if self.raw_dir.exists() else []
        return {
            "root": str(self.root),
            "raw": {
                "dir": str(self.raw_dir),
                "datasets": [
                    {"name": path.stem, "bytes": path.stat().st_size} for path in raws
                ],
            },
            "normalized": count(self.normalized_dir),
            "events": count(self.events_dir),
            "generated_at": dt.datetime.now(dt.UTC).isoformat().replace("+00:00", "Z"),
        }


class _open_append_or_write:
    """Context manager choosing append or truncate, creating the file if absent."""

    def __init__(self, path: Path, append: bool) -> None:
        self.path = path
        self.append = append
        self.handle = None

    def __enter__(self):
        exists = self.path.exists()
        # Truncating an existing file is intentional in non-append mode, but only
        # after the rows have been produced upstream, so a failure mid-write does
        # not destroy a good dataset.
        self.handle = self.path.open("a" if self.append or not exists else "w", encoding="utf-8")
        return self.handle

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.handle is not None:
            self.handle.flush()
            import os as _os

            _os.fsync(self.handle.fileno())
            self.handle.close()


def _atomic_write_text(path: Path, text: str) -> None:
    """Write via temp file + `os.replace`, so readers never see a partial file."""
    directory = path.parent
    directory.mkdir(parents=True, exist_ok=True)
    handle_fd, temp_name = tempfile.mkstemp(
        dir=directory, prefix=f".{path.name}.", suffix=".tmp"
    )
    temp_path = Path(temp_name)
    try:
        # Close the raw descriptor and reopen through a text handle: the
        # descriptor exists only so mkstemp can hand back a unique name.
        os.close(handle_fd)
        with temp_path.open("w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    except BaseException:
        # A failed write must not leave a partial file where a valid one was.
        temp_path.unlink(missing_ok=True)
        raise