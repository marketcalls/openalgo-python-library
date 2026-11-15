"""The stored form of recorded ticks (private to `recorder`): the record layout, the raw
files a recording appends to, and where the files live in the package's cache.

Layout under the cache directory (`data.cache.resolve_cache_dir`, which must be an
absolute folder here: recorded ticks cannot be fetched again), names made safe with
`data.cache.safe_name`:

    v2/ticks/recorder.lock                                      held while one writes
    v2/ticks/exchange=<X>/symbol=<S>/<YYYY-MM-DD>.<id>.raw      while a recording runs
    v2/ticks/exchange=<X>/symbol=<S>/<YYYY-MM-DD>.<id>.npy      the day, once finalized
    v2/ticks/exchange=<X>/symbol=<S>/index.json

A tick is one record of `TICK_DTYPE`, fixed size and little-endian, its times in UTC
nanoseconds. `recording` names the recording that took it and `seq` its place in that
recording's receive order, across all symbols: the writer numbers each message as it
takes it off the queue, so (recording, seq) orders ticks exactly where receive times tie
or the wall clock stepped back. A missing last trade time is `NO_EVENT_TIME` (0, which
the live bars read as missing too); a missing volume, bid or ask is NaN (`NO_VALUE`). A
volume of 0 is stored as 0: the SDK fills a quote volume the server left out with 0, and
the live bars apply their own rule to it.

A raw file holds the ticks of one instrument and one IST day (the day of their receive
time) from one recording: a header of `RAW_HEADER_SIZE` bytes (`RAW_MAGIC`, the
recording's id, the file's id, both in hex, and the recording's start in UTC
nanoseconds) and the records, appended in the order the writer took them. A crash can
leave a part of the last record: a read cuts it off and reports how many bytes it cut,
and after a write that failed the next append cuts it off first, so a tear is only ever
at the end of a file.
"""

from __future__ import annotations

import datetime as dt
import math
import os
import struct
import urllib.parse
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import numpy as np

from .._calendar_convert import SECONDS_PER_DAY, from_day
from .._layout import IST_OFFSET_NANOS, NANOS_PER_SECOND
from ..data.cache import CACHE_ENV, FORMAT_VERSION, resolve_cache_dir, safe_name
from ..errors import ConfigError, DataError
from ._types import Tick

#: One recorded tick. `recording` is the number of the recording that took it (its hex
#: id read as an integer), `seq` its place in that recording's receive order;
#: `ts_recv_ns` is `Tick.ts_recv_ns` (the wall clock when the writer took it off its
#: queue), `ts_queued_ns` when the feed's callback queued it, `ts_event_ns` the last
#: trade time; prices rupees, volume units.
TICK_DTYPE = np.dtype(
    [
        ("recording", "<u8"),
        ("seq", "<i8"),
        ("ts_recv_ns", "<i8"),
        ("ts_queued_ns", "<i8"),
        ("ts_event_ns", "<i8"),
        ("ltp", "<f8"),
        ("volume", "<f8"),
        ("bid", "<f8"),
        ("ask", "<f8"),
    ]
)
#: A record's `ts_event_ns` when the feed gave no last trade time.
NO_EVENT_TIME = 0
#: A record's volume, bid or ask when the feed gave none.
NO_VALUE = math.nan

#: The cache's folder of recorded ticks, beside its history namespaces.
TICKS_FOLDER = "ticks"
#: The first bytes of a raw file; the digit is the version of the record layout.
RAW_MAGIC = b"OA-TICKS-RAW-v2\n"
# Hex characters of a recording's or a file's id: 64 bits, so that two never share one.
_ID_CHARS = 16
_HEX_DIGITS = frozenset("0123456789abcdef")
_START = struct.Struct("<q")  # the recording's start, UTC nanoseconds
#: Bytes before the first record of a raw file: the magic, the two ids and the start.
RAW_HEADER_SIZE = len(RAW_MAGIC) + 2 * _ID_CHARS + _START.size
RAW_SUFFIX = ".raw"
DAY_SUFFIX = ".npy"
_NANOS_PER_DAY = SECONDS_PER_DAY * NANOS_PER_SECOND
_EXCHANGE_PREFIX = "exchange="
_SYMBOL_PREFIX = "symbol="


def new_id() -> str:
    """A new id for a recording or a file: 16 hex digits."""
    return uuid.uuid4().hex[:_ID_CHARS]


def is_id(text: str) -> bool:
    """Whether `text` is an id `new_id` could have made."""
    return isinstance(text, str) and len(text) == _ID_CHARS and set(text) <= _HEX_DIGITS


def recording_number(recording: str) -> int:
    """The number a recording's records carry: its hex id read as an integer."""
    return int(recording, 16)


def recording_name(number: int) -> str:
    """The hex id of the recording whose records carry `number`."""
    return f"{int(number):0{_ID_CHARS}x}"


def ticks_root(cache_dir: str | os.PathLike | None) -> Path:
    """The folder of recorded ticks in the cache directory (`resolve_cache_dir`, a
    leading `~` expanded). Raises ConfigError when that is not an absolute folder: a
    relative one would move with the folder Python was started from, and recorded ticks
    cannot be fetched again."""
    given = resolve_cache_dir(cache_dir)
    folder = given.expanduser()
    if not folder.is_absolute():
        source = "cache_dir" if cache_dir is not None else CACHE_ENV
        raise ConfigError(
            f"recorded ticks are kept in an absolute folder only, and {source} is "
            f"{os.fspath(given)!r}: pass cache_dir= an absolute folder, or set {CACHE_ENV} "
            "to one. Recorded ticks cannot be fetched again, so they must not land in a "
            "folder that moves with the folder Python was started from"
        )
    return folder / f"v{FORMAT_VERSION}" / TICKS_FOLDER


def instrument_folder(root: Path, exchange: str, symbol: str) -> Path:
    """The folder of one instrument's tick files."""
    return (
        root / f"{_EXCHANGE_PREFIX}{safe_name(exchange)}" / f"{_SYMBOL_PREFIX}{safe_name(symbol)}"
    )


def instrument_folders(root: Path) -> list[Path]:
    """Every instrument's folder in the folder of recorded ticks, in name order."""
    return sorted(
        path for path in root.glob(f"{_EXCHANGE_PREFIX}*/{_SYMBOL_PREFIX}*") if path.is_dir()
    )


def instrument_of(folder: Path) -> tuple[str, str]:
    """(exchange, symbol) of an instrument's folder: the inverse of `instrument_folder`
    (`safe_name` writes every other character as %XX of its UTF-8 bytes)."""
    exchange = folder.parent.name.removeprefix(_EXCHANGE_PREFIX)
    symbol = folder.name.removeprefix(_SYMBOL_PREFIX)
    return urllib.parse.unquote(exchange), urllib.parse.unquote(symbol)


def raw_files_of(root: Path, day: dt.date) -> list[Path]:
    """Every raw file of an IST day in the cache, in name order."""
    pattern = f"{_EXCHANGE_PREFIX}*/{_SYMBOL_PREFIX}*/{day.isoformat()}.*{RAW_SUFFIX}"
    return sorted(root.glob(pattern))


def pending_raw_files(folder: Path, day: dt.date) -> list[Path]:
    """The raw files of an instrument's IST day not finalized yet, in name order."""
    return sorted(folder.glob(f"{day.isoformat()}.*{RAW_SUFFIX}"))


def day_files(folder: Path, key: str) -> list[Path]:
    """Every day file of the IST day `key` (an ISO date) in an instrument's folder."""
    return sorted(folder.glob(f"{key}.*{DAY_SUFFIX}"))


def ist_day(instant_ns: int) -> dt.date:
    """The IST date of an instant in UTC nanoseconds."""
    return from_day((instant_ns + IST_OFFSET_NANOS) // _NANOS_PER_DAY)


def day_bounds(day: dt.date) -> tuple[int, int]:
    """[start, end) of an IST date in UTC nanoseconds."""
    midnight = (day - dt.date(1970, 1, 1)).days * _NANOS_PER_DAY - IST_OFFSET_NANOS
    return midnight, midnight + _NANOS_PER_DAY


def tick_row(tick: Tick, *, recording: int, seq: int, queued_ns: int) -> tuple:
    """The fields of a tick's record, in `TICK_DTYPE` order."""
    return (
        recording,
        seq,
        tick.ts_recv_ns,
        queued_ns,
        NO_EVENT_TIME if tick.ts_event_ns is None else tick.ts_event_ns,
        tick.ltp,
        *(NO_VALUE if value is None else value for value in (tick.volume, tick.bid, tick.ask)),
    )


def records_of(rows: list[tuple]) -> np.ndarray:
    """Records of `TICK_DTYPE` from rows made by `tick_row`."""
    return np.array(rows, dtype=TICK_DTYPE)


def ticks_of(records: np.ndarray, exchange: str, symbol: str) -> list[Tick]:
    """The ticks of records of `TICK_DTYPE`, in their order."""
    ticks = []
    for _, _, recv, _, event, ltp, volume, bid, ask in records.tolist():
        ticks.append(
            Tick(
                symbol,
                exchange,
                ltp,
                recv,
                None if event == NO_EVENT_TIME else event,
                *(None if math.isnan(value) else value for value in (volume, bid, ask)),
            )
        )
    return ticks


class RawFile:
    """A raw file open for appending, made by `create`. Only the writer that made it
    writes it; each append reaches the system at once (the file is unbuffered).

    After a write that failed, the next append first cuts the file back to its last
    whole record; when that cut fails too, `cut_failed` is set and nothing more may be
    written to it: the writer moves to a new raw file.
    """

    def __init__(self, path: Path, handle: BinaryIO, recording: str, file_id: str) -> None:
        self.path = path
        self.recording = recording
        self.file_id = file_id
        self.cut_failed = False
        self._handle = handle
        self._torn = False

    @classmethod
    def create(cls, folder: Path, day: dt.date, *, recording: str, start_ns: int) -> RawFile:
        """Create a new raw file of an IST day in an instrument's folder for the recording
        `recording` (its id) that started at `start_ns`. Raises OSError when it cannot be
        made."""
        folder.mkdir(parents=True, exist_ok=True)
        file_id = new_id()
        path = folder / f"{day.isoformat()}.{file_id}{RAW_SUFFIX}"
        header = RAW_MAGIC + f"{recording}{file_id}".encode("ascii") + _START.pack(start_ns)
        # No `with` block: the file stays open across appends until `close`.
        handle = open(path, "xb", buffering=0)  # noqa: SIM115
        try:
            _write_all(handle, memoryview(header))
        except BaseException:
            handle.close()
            path.unlink(missing_ok=True)
            raise
        return cls(path, handle, recording, file_id)

    @property
    def closed(self) -> bool:
        return self._handle.closed

    def append(self, records: np.ndarray) -> None:
        """Append records. Raises OSError when they cannot be written, or when the end a
        failed write tore cannot be cut off (`cut_failed` is then set)."""
        if self._torn:
            self._cut_torn_end()
        data = memoryview(np.ascontiguousarray(records, dtype=TICK_DTYPE).tobytes())
        try:
            self._write_bytes(data)
        except BaseException:
            # Part of the batch may be in the file: the next append cuts back first.
            self._torn = True
            raise

    def close(self) -> None:
        """Close the file (once; a second call does nothing); it stays for finalize.
        Raises OSError when the system reports an error on closing."""
        if not self._handle.closed:
            self._handle.close()

    def _write_bytes(self, data: memoryview) -> None:
        """Write every byte of `data`: the one write of records."""
        _write_all(self._handle, data)

    def _truncate(self, size: int) -> None:
        """Cut the file to `size` bytes: the one truncation."""
        self._handle.truncate(size)

    def _cut_torn_end(self) -> None:
        try:
            size = os.fstat(self._handle.fileno()).st_size
            body = max(size - RAW_HEADER_SIZE, 0)
            whole = RAW_HEADER_SIZE + body - body % TICK_DTYPE.itemsize
            if size != whole:
                self._truncate(whole)
            self._handle.seek(0, os.SEEK_END)
        except OSError:
            self.cut_failed = True
            raise
        self._torn = False


def _write_all(handle: BinaryIO, data: memoryview) -> None:
    while data:
        written = handle.write(data)
        data = data[written:]


@dataclass(frozen=True)
class RawContent:
    """What a raw file holds: the recording's id and start (UTC nanoseconds), the file's
    id, its whole records, and the bytes of a torn last record that were cut off (0 when
    there was none)."""

    recording: str
    file_id: str
    start_ns: int
    records: np.ndarray
    torn_bytes: int


def read_raw(path: Path) -> RawContent:
    """Read a raw file. Raises DataError when its header is not a raw file's of this
    version, OSError when it cannot be read."""
    blob = path.read_bytes()
    ids = blob[len(RAW_MAGIC) : len(RAW_MAGIC) + 2 * _ID_CHARS].decode("ascii", errors="replace")
    recording, file_id = ids[:_ID_CHARS], ids[_ID_CHARS:]
    ours = len(blob) >= RAW_HEADER_SIZE and blob.startswith(RAW_MAGIC)
    if not (ours and is_id(recording) and is_id(file_id)):
        raise DataError(f"{path} is not a raw tick file of this version: its header differs")
    (start_ns,) = _START.unpack_from(blob, RAW_HEADER_SIZE - _START.size)
    body = memoryview(blob)[RAW_HEADER_SIZE:]
    whole = len(body) - len(body) % TICK_DTYPE.itemsize
    records = np.frombuffer(body[:whole], dtype=TICK_DTYPE).copy()
    return RawContent(recording, file_id, start_ns, records, len(body) - whole)


__all__ = [
    "DAY_SUFFIX",
    "NO_EVENT_TIME",
    "NO_VALUE",
    "RAW_HEADER_SIZE",
    "RAW_MAGIC",
    "RAW_SUFFIX",
    "TICKS_FOLDER",
    "TICK_DTYPE",
    "RawContent",
    "RawFile",
    "day_bounds",
    "day_files",
    "instrument_folder",
    "instrument_folders",
    "instrument_of",
    "ist_day",
    "is_id",
    "new_id",
    "pending_raw_files",
    "raw_files_of",
    "read_raw",
    "recording_name",
    "recording_number",
    "records_of",
    "tick_row",
    "ticks_of",
    "ticks_root",
]
