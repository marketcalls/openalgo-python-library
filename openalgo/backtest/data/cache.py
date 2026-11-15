"""Local cache of raw history rows.

The cache stores exactly what a source returned: epoch seconds and values, per
source and broker. Nothing in it depends on a calendar, a session filter, a
timestamp shift or the normalisation code: every load normalises the cached rows
again, with its own calendar and options.

Layout under the cache directory:

    v2/<namespace>/exchange=<X>/symbol=<S>/interval=<I>/<YYYY-MM>.npy
    v2/<namespace>/exchange=<X>/symbol=<S>/interval=<I>/index.json

One `.npy` file holds the rows of one IST calendar month of one series, as a
structured array without pickle. `index.json` records, per file, a SHA-256 content
hash and, per IST date, the time of the fetch that supplied the rows of that date.
The namespace keeps sources and brokers apart ("openalgo-db",
"openalgo-api-zerodha").

What is trusted:

- A date counts as complete only when it is settled (`is_settled`): its rows were
  fetched a safety margin after its session closed and, for a broker, on a later IST
  date. Anything else is fetched again by the loader. A date the source answered with
  no rows can be recorded the same way, with no rows in its month file, so that a
  holiday the calendar does not know is not asked for on every load (which dates the
  loader records is `_empty_answers`). Caches written before that are read as they
  are: such a date is asked for once more, then recorded.
- Every read verifies the hash, including the read that precedes a merge. A file
  that fails is moved aside (quarantined), never served and never deleted, and its
  dates are fetched again.

The cache directory is, in order: the `cache_dir` argument, the environment
variable OPENALGO_BT_CACHE, then `~/.openalgo/backtest/cache`. Every file is
written to a temporary name and then moved into place, the index last. Two
processes writing one series are not coordinated: the later index wins, and a file
it does not describe fails verification and is fetched again.
"""

import datetime as dt
import hashlib
import json
import os
import time
import uuid
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO

import numpy as np

from .._calendar_convert import from_day, to_day
from .._layout import NANOS_PER_SECOND
from ..errors import ConfigError, DataError
from ._sessions import ist_days
from .fetching import RAW_DTYPE, bucket_days

CACHE_ENV = "OPENALGO_BT_CACHE"
FORMAT_VERSION = 2
INDEX_NAME = "index.json"

# A session is taken as complete only when its rows were fetched this long after its
# close: sources keep correcting and completing a day for some hours (a local store
# is filled by a download job, a broker revises its candles in the evening).
SETTLE_MARGIN_NANOS = 6 * 3_600 * NANOS_PER_SECOND

_DEFAULT_CACHE_PARTS = (".openalgo", "backtest", "cache")
# Mixed into every hash, so that a change of the stored form changes the hashes.
_HASH_HEADER = b"openalgo.backtest raw rows v2\n"
_SAFE_NAME_CHARACTERS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
)
# What numpy raises for a file that is missing, truncated or not an array.
_READ_ERRORS = (OSError, ValueError, EOFError)
# On Windows a file that another process holds open cannot be replaced. The move is
# tried again a few times, with a short growing pause, before giving up.
_REPLACE_ATTEMPTS = 5
_REPLACE_PAUSE_SECONDS = 0.05
_UNIQUE_SUFFIX_LENGTH = 8  # hex characters that keep two writers' files apart

DateRange = tuple[dt.date, dt.date]


def resolve_cache_dir(cache_dir: str | os.PathLike | None = None) -> Path:
    """The cache directory: the argument, then OPENALGO_BT_CACHE, then the per-user
    default `~/.openalgo/backtest/cache`."""
    if cache_dir is not None:
        return Path(cache_dir)
    from_environment = os.environ.get(CACHE_ENV)
    if from_environment:
        return Path(from_environment)
    return Path.home().joinpath(*_DEFAULT_CACHE_PARTS)


def is_settled(day: dt.date, fetched_at_s: int | None, close_ns: int, source: str) -> bool:
    """Whether the cached rows of a session can be taken as complete.

    day: the IST date of the session. fetched_at_s: epoch seconds of the fetch that
    supplied its rows (None when there are none). close_ns: the session close in UTC
    nanoseconds. source: "db" or "api"; a broker is never trusted for a session on
    the IST date it was asked.
    """
    if fetched_at_s is None:
        return False
    fetched_ns = fetched_at_s * NANOS_PER_SECOND
    if fetched_ns < close_ns + SETTLE_MARGIN_NANOS:
        return False
    fetched_day = int(ist_days(fetched_ns))
    return source != "api" or fetched_day > to_day(day)


def content_hash(records: np.ndarray) -> str:
    """SHA-256 of the bytes of raw rows, as 'sha256:<hex>'."""
    digest = hashlib.sha256(_HASH_HEADER)
    digest.update(np.ascontiguousarray(records, dtype=RAW_DTYPE).tobytes())
    return "sha256:" + digest.hexdigest()


@dataclass(frozen=True)
class SeriesKey:
    """Which cached series: the namespace (one per source and broker) and the instrument."""

    namespace: str
    exchange: str
    symbol: str
    interval: str


@dataclass
class CachedRows:
    """Verified raw rows of a date range.

    records: the rows, sorted by timestamp. fetched_at: IST date to the epoch seconds
    of the fetch that supplied its rows. notes: what failed verification on the way.
    """

    records: np.ndarray
    fetched_at: dict[dt.date, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


class RawCache:
    """Read and write the cached raw rows of bar series (see the module docstring)."""

    def __init__(self, cache_dir: str | os.PathLike | None = None) -> None:
        self.root = resolve_cache_dir(cache_dir)

    def series_dir(self, key: SeriesKey) -> Path:
        """The directory that holds the partitions and the index of a series."""
        return (
            self.root
            / f"v{FORMAT_VERSION}"
            / safe_name(key.namespace)
            / f"exchange={safe_name(key.exchange)}"
            / f"symbol={safe_name(key.symbol)}"
            / f"interval={safe_name(key.interval)}"
        )

    def read(self, key: SeriesKey, first: dt.date, last: dt.date) -> CachedRows:
        """The cached rows stamped on the IST dates `first` to `last`, verified.

        A partition or index that fails verification is quarantined and reported in
        `notes`; its rows are not returned. Raises DataError only when a damaged
        file cannot be moved aside.
        """
        folder = self.series_dir(key)
        found = CachedRows(np.zeros(0, dtype=RAW_DTYPE))
        index = self._index(folder, key, found.notes)
        low, high = to_day(first), to_day(last)
        parts = []
        for month in _months_between(first, last):
            records = self._partition(folder, index, month, found.notes)
            if records is None:
                continue
            days = bucket_days(records)
            parts.append(records[(days >= low) & (days <= high)])
            for text, fetched_at_s in index["partitions"][month]["fetched"].items():
                day = dt.date.fromisoformat(text)
                if first <= day <= last:
                    found.fetched_at[day] = fetched_at_s
        if parts:
            found.records = np.concatenate(parts)
        return found

    def write(
        self,
        key: SeriesKey,
        records: np.ndarray,
        ranges: Iterable[DateRange],
        fetched_at_s: int,
        *,
        source: str,
        broker: str | None,
        empty_days: Iterable[dt.date] = (),
    ) -> list[str]:
        """Store the rows a source returned for the inclusive IST date `ranges`.

        The rows of every date that received at least one row replace the cached
        rows of that date; dates that received nothing keep what they had, and rows
        outside `ranges` are ignored. `fetched_at_s` (epoch seconds) is recorded for
        each date that was replaced, and for each of `empty_days` (dates the source
        answered with no rows) that has no cached rows. Returns notes about files
        that failed verification while merging.

        Raises ConfigError when the cache directory cannot be used.
        """
        folder = self.series_dir(key)
        notes: list[str] = []
        index = self._index(folder, key, notes)  # may move a damaged directory aside
        try:
            folder.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ConfigError(
                f"the cache directory {self.root} cannot be used ({exc}). Pass cache_dir=, "
                f"set {CACHE_ENV}, or load with cache=False."
            ) from exc
        days = bucket_days(records)
        asked = np.zeros(len(records), dtype=bool)
        for first, last in ranges:
            asked |= (days >= to_day(first)) & (days <= to_day(last))
        records, days = records[asked], days[asked]
        months = np.array([from_day(day).strftime("%Y-%m") for day in days], dtype=object)
        empty_by_month: dict[str, list[int]] = {}
        for day in empty_days:
            empty_by_month.setdefault(day.strftime("%Y-%m"), []).append(to_day(day))
        for month in sorted(set(months.tolist()) | set(empty_by_month)):
            of_month = months == month
            empty = empty_by_month.get(month, [])
            self._merge(
                folder, index, month, records[of_month], days[of_month], empty, fetched_at_s, notes
            )
        index.update(source=source, broker=broker)
        self._save_index(folder, index)
        return notes

    def _merge(
        self,
        folder: Path,
        index: dict,
        month: str,
        fresh: np.ndarray,
        fresh_days: np.ndarray,
        empty_days: list[int],
        fetched_at_s: int,
        notes: list[str],
    ) -> None:
        """Replace, in one month partition, the rows of the dates that `fresh` covers,
        and record the `empty_days` (day numbers) that have no rows as fetched."""
        cached = self._partition(folder, index, month, notes)
        fetched = dict(index["partitions"].get(month, {}).get("fetched", {}))
        pieces = [fresh]
        if cached is not None:
            pieces.append(cached[~np.isin(bucket_days(cached), fresh_days)])
        merged = np.concatenate(pieces)
        merged = merged[np.argsort(merged["timestamp"], kind="stable")]
        with_rows = set(bucket_days(merged).tolist())
        recorded = [*np.unique(fresh_days).tolist(), *(d for d in empty_days if d not in with_rows)]
        if not recorded:
            return
        path = folder / f"{month}.npy"
        atomic_write(path, lambda handle: np.save(handle, merged, allow_pickle=False))
        fetched.update({str(from_day(day)): fetched_at_s for day in recorded})
        index["partitions"][month] = {
            "file": path.name,
            "sha256": content_hash(merged),
            "rows": len(merged),
            "fetched": fetched,
        }

    def _index(self, folder: Path, key: SeriesKey, notes: list[str]) -> dict:
        """The index of a series; a fresh one when there is none or it is unusable."""
        fresh = {
            "version": FORMAT_VERSION,
            "namespace": key.namespace,
            "exchange": key.exchange,
            "symbol": key.symbol,
            "interval": key.interval,
            "partitions": {},
        }
        path = folder / INDEX_NAME
        try:
            index = json.loads(path.read_bytes().decode("utf-8"))
            same_series = isinstance(index, dict) and all(
                index.get(name) == value for name, value in fresh.items() if name != "partitions"
            )
            usable = same_series and isinstance(index.get("partitions"), dict)
            problem = None if usable else "it belongs to another series or format"
        except (FileNotFoundError, NotADirectoryError):
            return fresh
        except (OSError, ValueError) as exc:
            problem = f"it cannot be read ({exc})"
        if problem is None:
            return index
        moved = _quarantine(folder)
        notes.append(
            f"the cache index {path} failed verification ({problem}); the series directory "
            f"was moved to {moved.name} and the series is fetched again"
        )
        return fresh

    def _partition(
        self, folder: Path, index: dict, month: str, notes: list[str]
    ) -> np.ndarray | None:
        """The verified rows of one month, or None when it is not cached or was
        quarantined just now."""
        entry = index["partitions"].get(month)
        if entry is None:
            return None
        path = folder / entry["file"]
        try:
            records = np.load(path, allow_pickle=False)
            matches = records.dtype == RAW_DTYPE and content_hash(records) == entry["sha256"]
            problem = None if matches else "its content does not match the hash in the index"
        except _READ_ERRORS as exc:
            problem = f"it cannot be read ({exc})"
        if problem is None:
            return records
        moved = _quarantine(path)
        del index["partitions"][month]
        self._save_index(folder, index)
        notes.append(
            f"the cache partition {path} failed verification ({problem}); it was moved to "
            f"{moved.name} and its sessions are fetched again"
        )
        return None

    def _save_index(self, folder: Path, index: dict) -> None:
        payload = json.dumps(index, indent=1, sort_keys=True).encode("utf-8")
        atomic_write(folder / INDEX_NAME, lambda handle: handle.write(payload))


def safe_name(text: str) -> str:
    """A directory-name form of `text` that is reversible and valid on every OS.

    Letters, digits, "-" and "_" stay; every other character becomes %XX of its
    UTF-8 bytes ("M&M" is "M%26M"), so no symbol can produce a path separator or a
    name the file system treats specially.
    """
    return "".join(
        character
        if character in _SAFE_NAME_CHARACTERS
        else "".join(f"%{byte:02X}" for byte in character.encode("utf-8"))
        for character in str(text)
    )


def atomic_write(path: Path, write: Callable[[BinaryIO], object]) -> None:
    """Write a file through a temporary file in the same directory, then move it
    into place, so that a reader never sees a half-written file.

    `write` receives the open binary handle. Raises OSError when the file cannot be
    written or moved; the temporary file is removed in every case.
    """
    unique = uuid.uuid4().hex[:_UNIQUE_SUFFIX_LENGTH]
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{unique}.tmp")
    try:
        with open(temporary, "wb") as handle:
            write(handle)
            handle.flush()
            os.fsync(handle.fileno())
        for attempt in range(1, _REPLACE_ATTEMPTS + 1):
            try:
                os.replace(temporary, path)
                break
            except PermissionError:
                if attempt == _REPLACE_ATTEMPTS:
                    raise
                time.sleep(_REPLACE_PAUSE_SECONDS * attempt)
    finally:
        temporary.unlink(missing_ok=True)


def _quarantine(path: Path) -> Path:
    """Move a damaged file or directory aside, under a name no reader looks at.

    Raises DataError when it cannot be moved: it must then be deleted by hand.
    """
    moved = path.with_name(f"{path.name}.quarantined-{uuid.uuid4().hex[:_UNIQUE_SUFFIX_LENGTH]}")
    try:
        if path.exists():
            os.replace(path, moved)
    except OSError as exc:
        raise DataError(
            f"the damaged cache entry {path} could not be moved aside ({exc}). Delete it and "
            "load again."
        ) from exc
    return moved


def _months_between(start: dt.date, end: dt.date) -> list[str]:
    """'YYYY-MM' keys of every month the inclusive date range touches."""
    months = []
    year, month = start.year, start.month
    while (year, month) <= (end.year, end.month):
        months.append(f"{year:04d}-{month:02d}")
        year, month = (year + month // 12, month % 12 + 1)
    return months


__all__ = [
    "CACHE_ENV",
    "SETTLE_MARGIN_NANOS",
    "CachedRows",
    "RawCache",
    "SeriesKey",
    "content_hash",
    "is_settled",
    "resolve_cache_dir",
]
