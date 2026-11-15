"""A recording's raw file (private to `recorder`): what the writer appends to while a
recording runs, and what finalize reads back.

A raw file is a header (`RAW_MAGIC`, then the length of a JSON document and the document:
the recording's number, the file's part, the IST day, the recording's start in UTC
nanoseconds and the list of its instruments) followed by records of `RAW_DTYPE`, in the
order the writer took them. Every append ends with an fsync, so a power loss costs at
most the last flush.

What a crash or a power loss can leave is cut off or skipped on read, never guessed: a
part of the last record (counted in bytes), and records that cannot be the recording's
(counted): a zero-filled tail, records of another recording number, a sequence number
that does not increase from 1, an instrument the header does not list, a last price that
is not a finite positive number. After a write that failed, the next append first cuts
the file back to its last whole record, so a tear is only ever at the end of a file.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np

from ..errors import DataError
from ._tick_files import RAW_DTYPE

#: The first bytes of a raw file; the digit is the version of the layout.
RAW_MAGIC = b"OA-TICKS-RAW-v3\n"
_LENGTH = struct.Struct("<I")  # bytes of the header's JSON document
_PREFIX = len(RAW_MAGIC) + _LENGTH.size
# A header longer than this is no header of ours (a list of 10 000 instruments is less).
_MAX_HEADER_BYTES = 4 * 1024 * 1024


@dataclass(frozen=True)
class RawHeader:
    """What a raw file says about itself: its recording's number, its part (0 first),
    the IST day, the recording's start (UTC nanoseconds) and the (exchange, symbol) of
    each instrument, whose index the records carry."""

    recording: int
    part: int
    day: dt.date
    start_ns: int
    instruments: tuple[tuple[str, str], ...]

    def encoded(self) -> bytes:
        """The header as the file begins."""
        document = {
            "recording": self.recording,
            "part": self.part,
            "day": self.day.isoformat(),
            "start_ns": self.start_ns,
            "instruments": [list(instrument) for instrument in self.instruments],
        }
        payload = json.dumps(document, sort_keys=True).encode("utf-8")
        return RAW_MAGIC + _LENGTH.pack(len(payload)) + payload


@dataclass(frozen=True)
class RawContent:
    """What a raw file holds: its header, its records that can be the recording's (in
    file order), how many records were skipped as not the recording's, and the bytes of
    a torn last record cut off."""

    header: RawHeader
    records: np.ndarray
    rejected: int
    torn_bytes: int


class RawFile:
    """A raw file open for appending, made by `create`. Only the writer that made it
    writes it.

    After a write that failed, the next append first cuts the file back to its last
    whole record; when that cut fails too, `cut_failed` is set and nothing more may be
    written to it: the writer goes on in the recording's next part.
    """

    def __init__(self, path: Path, handle: BinaryIO, header: RawHeader, body_start: int) -> None:
        self.path = path
        self.header = header
        self.cut_failed = False
        self._handle = handle
        self._body_start = body_start
        self._torn = False

    @classmethod
    def create(cls, path: Path, header: RawHeader) -> RawFile:
        """Create a new raw file with its header written and synced. Raises OSError when it
        cannot be made."""
        path.parent.mkdir(parents=True, exist_ok=True)
        encoded = header.encoded()
        # No `with` block: the file stays open across appends until `close`.
        handle = open(path, "xb", buffering=0)  # noqa: SIM115
        try:
            _write_all(handle, memoryview(encoded))
            os.fsync(handle.fileno())
        except BaseException:
            handle.close()
            path.unlink(missing_ok=True)
            raise
        return cls(path, handle, header, len(encoded))

    @property
    def closed(self) -> bool:
        """Whether the file was closed (by `close`): nothing more is written to it."""
        return self._handle.closed

    def append(self, records: np.ndarray) -> None:
        """Append records and sync them to the disk. Raises OSError when they cannot be
        written or synced, or when the end a failed write tore cannot be cut off
        (`cut_failed` is then set)."""
        if self._torn:
            self._cut_torn_end()
        data = memoryview(np.ascontiguousarray(records, dtype=RAW_DTYPE).tobytes())
        try:
            self._write_bytes(data)
            self._sync()
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

    def _sync(self) -> None:
        """Hand what was written to the disk: the one fsync of an append."""
        os.fsync(self._handle.fileno())

    def _truncate(self, size: int) -> None:
        """Cut the file to `size` bytes: the one truncation."""
        self._handle.truncate(size)

    def _cut_torn_end(self) -> None:
        try:
            size = os.fstat(self._handle.fileno()).st_size
            body = max(size - self._body_start, 0)
            whole = self._body_start + body - body % RAW_DTYPE.itemsize
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


def read_raw_header(path: Path) -> RawHeader:
    """The header of a raw file. Raises DataError when it is not a raw file's of this
    version, OSError when it cannot be read."""
    with open(path, "rb") as handle:
        length = _header_length(path, handle.read(_PREFIX))
        return _header(path, handle.read(length))


def read_raw(path: Path) -> RawContent:
    """A raw file's header and the records that can be the recording's (see the module
    docstring). Raises DataError when its header is not a raw file's of this version,
    OSError when it cannot be read."""
    blob = path.read_bytes()
    length = _header_length(path, blob[:_PREFIX])
    header = _header(path, blob[_PREFIX : _PREFIX + length])
    body = memoryview(blob)[_PREFIX + length :]
    whole = len(body) - len(body) % RAW_DTYPE.itemsize
    records = np.frombuffer(body[:whole], dtype=RAW_DTYPE)
    keep = _possible(records, header)
    return RawContent(header, records[keep].copy(), int((~keep).sum()), len(body) - whole)


def _header_length(path: Path, prefix: bytes) -> int:
    """The length of the header's document, from the magic and the length field."""
    if len(prefix) < _PREFIX or not prefix.startswith(RAW_MAGIC):
        raise DataError(f"{path} is not a raw tick file of this version: its header differs")
    (length,) = _LENGTH.unpack_from(prefix, len(RAW_MAGIC))
    if length > _MAX_HEADER_BYTES:
        raise DataError(f"{path} is not a raw tick file of this version: its header is too long")
    return length


def _header(path: Path, payload: bytes) -> RawHeader:
    """The header from its document (a torn one cannot be parsed)."""
    try:
        document = json.loads(bytes(payload).decode("utf-8"))
        header = RawHeader(
            recording=_count(document["recording"]),
            part=_count(document["part"]),
            day=dt.date.fromisoformat(document["day"]),
            start_ns=_count(document["start_ns"]),
            instruments=tuple(_instrument(item) for item in document["instruments"]),
        )
    except (ValueError, KeyError, TypeError) as exc:
        raise DataError(
            f"{path} is not a raw tick file of this version: its header is torn ({exc})"
        ) from exc
    return header


def _count(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{value!r} is no count")
    return value


def _instrument(item: Any) -> tuple[str, str]:
    if not (
        isinstance(item, list) and len(item) == 2 and all(isinstance(v, str) and v for v in item)
    ):
        raise ValueError(f"{item!r} is no instrument")
    return item[0], item[1]


def _possible(records: np.ndarray, header: RawHeader) -> np.ndarray:
    """Which records can be the recording's: its number, an instrument it lists, a
    finite positive last price, and a sequence number of 1 or more above every earlier
    such record's (a batch written twice keeps its first copy)."""
    ltp = records["ltp"]
    keep = (
        (records["recording"] == header.recording)
        & (records["instrument"] < len(header.instruments))
        & np.isfinite(ltp)
        & (ltp > 0)
        & (records["seq"] >= 1)
    )
    return increasing(records, keep)


def increasing(records: np.ndarray, keep: np.ndarray | None = None) -> np.ndarray:
    """Which of the records (those `keep` allows, all when None) have a sequence number
    above every earlier one's: a batch written twice keeps its first copy."""
    keep = np.ones(len(records), dtype=bool) if keep is None else keep
    seqs = np.where(keep, records["seq"], 0)
    before = np.concatenate([[0], np.maximum.accumulate(seqs)[:-1]]) if len(seqs) else seqs
    return keep & (seqs > before)


__all__ = [
    "RAW_MAGIC",
    "RawContent",
    "RawFile",
    "RawHeader",
    "increasing",
    "read_raw",
    "read_raw_header",
]
