"""A recording's raw file (private to `recorder`): what the writer appends to while a
recording runs, and what finalize reads back. Readers of a day never open it.

A raw file is a header (`RAW_MAGIC`, then the length of a JSON document and the document:
the recording's number and identity, the file's part, the IST day, the recording's start
in UTC nanoseconds and the list of its instruments) followed by frames, one per flush. A
frame is a small header (`_FRAME_MAGIC`, the record count, the CRC-32 of the records and
the CRC-32 of those three) followed by the records of `RAW_DTYPE`, in the order the
writer took them. Every append ends with an fsync, so a power loss costs at most the
last flush.

What a crash or a power loss can leave is dropped whole and counted, never guessed: a
frame whose records are cut short or do not match their CRC (a zero-filled page inside
it) is dropped and reading goes on at the next frame, since its header says where that
is; a frame header that cannot be read (torn, zero-filled) ends the file, its bytes
counted. In the frames that verify, a record that cannot be the recording's (another
recording number, an instrument the header does not list, a last price that is not a
finite positive number, a sequence number below 1) is skipped and counted. After a write
that failed, the next append first cuts the file back to the end of its last whole
frame, so a tear is only ever at the end of a file.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np

from ..errors import DataError
from ._tick_files import RAW_DTYPE, is_identity

#: The first bytes of a raw file; the digit is the version of the layout.
RAW_MAGIC = b"OA-TICKS-RAW-v4\n"
_LENGTH = struct.Struct("<I")  # bytes of the header's JSON document
_PREFIX = len(RAW_MAGIC) + _LENGTH.size
# A header longer than this is no header of ours (a list of 10 000 instruments is less).
_MAX_HEADER_BYTES = 4 * 1024 * 1024
#: The first bytes of every frame.
_FRAME_MAGIC = b"OAFR"
_FRAME_FIELDS = struct.Struct("<4sII")  # magic, record count, CRC-32 of the records
_FRAME_CHECK = struct.Struct("<I")  # CRC-32 of the three fields before it
FRAME_HEADER_SIZE = _FRAME_FIELDS.size + _FRAME_CHECK.size


@dataclass(frozen=True)
class RawHeader:
    """What a raw file says about itself: its recording's number and random identity,
    its part (0 first), the IST day, the recording's start (UTC nanoseconds) and the
    (exchange, symbol) of each instrument, whose index the records carry."""

    recording: int
    identity: str
    part: int
    day: dt.date
    start_ns: int
    instruments: tuple[tuple[str, str], ...]

    def encoded(self) -> bytes:
        """The header as the file begins."""
        document = {
            "recording": self.recording,
            "identity": self.identity,
            "part": self.part,
            "day": self.day.isoformat(),
            "start_ns": self.start_ns,
            "instruments": [list(instrument) for instrument in self.instruments],
        }
        payload = json.dumps(document, sort_keys=True).encode("utf-8")
        return RAW_MAGIC + _LENGTH.pack(len(payload)) + payload


@dataclass(frozen=True)
class RawContent:
    """What a raw file holds: its header; the records of its verified frames that can be
    the recording's, in file order; `rejected`, the records skipped (those of frames
    dropped whole, and those that could not be the recording's); `dropped_frames`, the
    frames dropped; and `torn_bytes`, the bytes after the last readable frame header."""

    header: RawHeader
    records: np.ndarray
    rejected: int
    dropped_frames: int
    torn_bytes: int


def frame_of(records: np.ndarray) -> bytes:
    """One flush's records as a frame."""
    body = np.ascontiguousarray(records, dtype=RAW_DTYPE).tobytes()
    fields = _FRAME_FIELDS.pack(_FRAME_MAGIC, len(records), zlib.crc32(body))
    return fields + _FRAME_CHECK.pack(zlib.crc32(fields)) + body


class RawFile:
    """A raw file open for appending, made by `create`. Only the writer that made it
    writes it.

    After a write that failed, the next append first cuts the file back to the end of
    its last whole frame; when that cut fails too, `cut_failed` is set and nothing more
    may be written to it: the writer goes on in the recording's next part.
    """

    def __init__(self, path: Path, handle: BinaryIO, header: RawHeader, body_start: int) -> None:
        self.path = path
        self.header = header
        self.cut_failed = False
        self._handle = handle
        self._whole_end = body_start  # the end of the last frame written and synced
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
        """Append records as one frame and sync it to the disk. Raises OSError when it
        cannot be written or synced, or when the end a failed write tore cannot be cut
        off (`cut_failed` is then set)."""
        if self._torn:
            self._cut_torn_end()
        frame = frame_of(records)
        try:
            self._write_bytes(memoryview(frame))
            self._sync()
        except BaseException:
            # Part of the frame may be in the file: the next append cuts back first.
            self._torn = True
            raise
        self._whole_end += len(frame)

    def close(self) -> None:
        """Close the file (once; a second call does nothing); it stays for finalize.
        Raises OSError when the system reports an error on closing."""
        if not self._handle.closed:
            self._handle.close()

    def _write_bytes(self, data: memoryview) -> None:
        """Write every byte of `data`: the one write of a frame."""
        _write_all(self._handle, data)

    def _sync(self) -> None:
        """Hand what was written to the disk: the one fsync of an append."""
        os.fsync(self._handle.fileno())

    def _truncate(self, size: int) -> None:
        """Cut the file to `size` bytes: the one truncation."""
        self._handle.truncate(size)

    def _cut_torn_end(self) -> None:
        try:
            if os.fstat(self._handle.fileno()).st_size != self._whole_end:
                self._truncate(self._whole_end)
            self._handle.seek(0, os.SEEK_END)
        except OSError:
            self.cut_failed = True
            raise
        self._torn = False


def _write_all(handle: BinaryIO, data: memoryview) -> None:
    while data:
        written = handle.write(data)
        data = data[written:]


def read_raw(path: Path) -> RawContent:
    """A raw file's header and the records of its verified frames that can be the
    recording's (see the module docstring). Raises DataError when its header is not a
    raw file's of this version, OSError when it cannot be read."""
    blob = path.read_bytes()
    length = _header_length(path, blob[:_PREFIX])
    header = _header(path, blob[_PREFIX : _PREFIX + length])
    frames, dropped, dropped_records, torn = _frames(memoryview(blob), _PREFIX + length)
    records = np.concatenate(frames) if frames else np.zeros(0, dtype=RAW_DTYPE)
    keep = _possible(records, header)
    rejected = dropped_records + int((~keep).sum())
    return RawContent(header, records[keep].copy(), rejected, dropped, torn)


def _frames(blob: memoryview, start: int) -> tuple[list[np.ndarray], int, int, int]:
    """(the records of each verified frame, frames dropped, their records, torn bytes)."""
    frames, dropped, dropped_records = [], 0, 0
    position = start
    while position < len(blob):
        fields = bytes(blob[position : position + _FRAME_FIELDS.size])
        check = bytes(blob[position + _FRAME_FIELDS.size : position + FRAME_HEADER_SIZE])
        if len(check) < _FRAME_CHECK.size or _FRAME_CHECK.unpack(check)[0] != zlib.crc32(fields):
            break  # a frame header that cannot be read ends the file
        magic, count, crc = _FRAME_FIELDS.unpack(fields)
        if magic != _FRAME_MAGIC:
            break
        body_start = position + FRAME_HEADER_SIZE
        end = body_start + count * RAW_DTYPE.itemsize
        body = blob[body_start:end]
        position = end
        if end > len(blob) or zlib.crc32(body) != crc:
            dropped, dropped_records = dropped + 1, dropped_records + count
            continue
        frames.append(np.frombuffer(body, dtype=RAW_DTYPE))
    return frames, dropped, dropped_records, max(len(blob) - position, 0)


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
        identity = document["identity"]
        if not is_identity(identity):
            raise ValueError(f"{identity!r} is no identity")
        header = RawHeader(
            recording=_count(document["recording"]),
            identity=identity,
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
    named = isinstance(item, list) and len(item) == 2
    if not (named and all(isinstance(value, str) and value for value in item)):
        raise ValueError(f"{item!r} is no instrument")
    return item[0], item[1]


def _possible(records: np.ndarray, header: RawHeader) -> np.ndarray:
    """Which records can be the recording's: its number, an instrument it lists, a
    finite positive last price and a sequence number of 1 or more."""
    ltp = records["ltp"]
    return (
        (records["recording"] == header.recording)
        & (records["instrument"] < len(header.instruments))
        & np.isfinite(ltp)
        & (ltp > 0)
        & (records["seq"] >= 1)
    )


def increasing(records: np.ndarray) -> np.ndarray:
    """Which records have a sequence number above every earlier one's: a batch written
    twice (a Ctrl+C right after its write, a part that repeats a torn batch) keeps its
    first copy."""
    seqs = records["seq"]
    before = np.concatenate([[0], np.maximum.accumulate(seqs)[:-1]]) if len(seqs) else seqs
    return seqs > before


__all__ = [
    "FRAME_HEADER_SIZE",
    "RAW_MAGIC",
    "RawContent",
    "RawFile",
    "RawHeader",
    "frame_of",
    "increasing",
    "read_raw",
]
