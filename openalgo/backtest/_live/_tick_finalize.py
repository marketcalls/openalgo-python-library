"""Finalizing a recording (private to `recorder`): its raw file becomes one finalized
file and sidecar per instrument, written once and never rewritten, and only then is the
raw file removed.

How the recording stopped is read from the stop sidecar beside its raw file
(`write_stop`, written before finalizing): its stop time, why it stopped and its feed
outages. Without one it ended by a crash, at its greatest receive time, its outages not
known. For each instrument the recording's file is checked first: one that exists and
verifies is kept as it is; one that is absent is written (the ticks, then the sidecar,
each with `atomic_write`) and checked; one that is damaged is reported and its raw file
kept, for `repair` to move it aside and a later finalize to write it again. The raw file
(all its parts) and the stop sidecar are removed only once every instrument's file
exists and verifies, so a crash at any point is redone safely: the next finalize finds
the files already written and only removes the raw file. A raw file is the only file
the recorder ever deletes.
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..data.cache import atomic_write
from ..errors import DataError
from ._tick_files import (
    finalized_paths,
    finalized_records,
    instrument_folder,
    raw_parts_of,
    set_aside,
    stop_path,
)
from ._tick_raw import RawContent, increasing, read_raw
from ._tick_sidecar import State, check, read_file, sidecar_bytes
from ._tick_spans import Outage, Span, Stop, outages_from_json, outages_to_json

_STOP_VERSION = 3


@dataclass(frozen=True)
class FinalizedFile:
    """One recording's finalized ticks of an instrument: the ticks file, the instrument,
    the IST day, the recording's number and its tick count."""

    path: Path
    exchange: str
    symbol: str
    day: dt.date
    recording: int
    ticks: int


@dataclass(frozen=True)
class RecordingFinalized:
    """What finalizing one recording did: its finalized files, the records its raw file
    held that could not be the recording's (`rejected`, a zero-filled tail after a power
    loss among them), the bytes of torn last records cut off, and whether its raw file
    was removed (False while a finalized file of it is damaged or a removal failed)."""

    recording: int
    day: dt.date
    files: tuple[FinalizedFile, ...]
    rejected: int
    torn_bytes: int
    removed: bool


@dataclass(frozen=True)
class StopFacts:
    """How a recording stopped: why, when (UTC nanoseconds) and its feed outages."""

    stopped_by: Stop
    stop_ns: int
    outages: tuple[Outage, ...]


def write_stop(root: Path, day: dt.date, number: int, facts: StopFacts) -> None:
    """Write the stop sidecar beside a recording's raw file. Raises OSError when it
    cannot be written."""
    document = {
        "version": _STOP_VERSION,
        "recording": number,
        "day": day.isoformat(),
        "stopped_by": facts.stopped_by.value,
        "stop_ns": facts.stop_ns,
        "outages": outages_to_json(facts.outages),
    }
    payload = json.dumps(document, indent=1, sort_keys=True).encode("utf-8")
    atomic_write(stop_path(root, day, number), lambda handle: handle.write(payload))


def finalize_pending(root: Path, day: dt.date, notes: list[str]) -> list[RecordingFinalized]:
    """Under the recorder's lock: finalize every recording of an IST day whose raw file
    is still there, in number order. What was found on the way is added to `notes`.
    Raises OSError naming the path when a file cannot be read, written or moved."""
    done = []
    for number, parts in raw_parts_of(root, day).items():
        result = finalize_recording(root, day, number, parts, notes)
        if result is not None:
            done.append(result)
    return done


def finalize_recording(
    root: Path, day: dt.date, number: int, parts: list[Path], notes: list[str]
) -> RecordingFinalized | None:
    """Finalize one recording from its raw file's parts (see the module docstring).
    Returns None when no part is a raw file of it (each is moved aside and noted)."""
    contents = _contents(parts, day, number, notes)
    if not contents:
        return None
    header = contents[0].header
    joined = np.concatenate([content.records for content in contents])
    once = increasing(joined)
    records = joined[once]
    span = _span(root, day, number, header.start_ns, records, notes)
    files = []
    for index, (exchange, symbol) in enumerate(header.instruments):
        mine = finalized_records(records[records["instrument"] == index])
        written = _finalize_instrument(root, day, exchange, symbol, mine, span, notes)
        if written is not None:
            files.append(written)
    whole = len(files) == len(header.instruments)
    removed = whole and _remove([*parts, stop_path(root, day, number)], notes)
    rejected = sum(content.rejected for content in contents) + int((~once).sum())
    torn = sum(content.torn_bytes for content in contents)
    return RecordingFinalized(number, day, tuple(files), rejected, torn, removed)


def _contents(parts: list[Path], day: dt.date, number: int, notes: list[str]) -> list[RawContent]:
    """The parts that are this recording's raw files; any other is moved aside."""
    contents: list[RawContent] = []
    for path in parts:
        try:
            content = read_raw(path)
            header = content.header
            first = contents[0].header.instruments if contents else header.instruments
            if (header.recording, header.day, header.instruments) != (number, day, first):
                raise DataError(f"{path} does not hold recording {number} of {day} its name says")
        except DataError as exc:
            notes.append(f"{exc}; it was moved to {set_aside(path).name}")
            continue
        contents.append(content)
    return contents


def _span(
    root: Path, day: dt.date, number: int, start_ns: int, records: np.ndarray, notes: list[str]
) -> Span:
    """The recording's span, from its stop sidecar, or ended by a crash without one."""
    greatest = int(records["ts_recv_ns"].max()) if len(records) else start_ns
    stopped = _read_stop(stop_path(root, day, number), day, number, notes)
    if stopped is None:
        return Span(number, start_ns, max(greatest, start_ns), Stop.CRASH, None)
    stop_ns = max(stopped.stop_ns, greatest, start_ns)
    return Span(number, start_ns, stop_ns, stopped.stopped_by, stopped.outages)


def _read_stop(path: Path, day: dt.date, number: int, notes: list[str]) -> StopFacts | None:
    """The stop sidecar, or None without one; one that cannot be parsed is noted and the
    recording counts as crashed. Raises OSError naming the path when it cannot be read."""
    try:
        blob = read_file(path)
    except FileNotFoundError:
        return None
    try:
        stored = json.loads(blob.decode("utf-8"))
        if stored["recording"] != number or stored["day"] != day.isoformat():
            raise ValueError("it belongs to another recording")
        stop_ns, outages = stored["stop_ns"], outages_from_json(stored["outages"])
        if not isinstance(stop_ns, int) or isinstance(stop_ns, bool) or outages is None:
            raise ValueError("its stop or outages are malformed")
        return StopFacts(Stop(stored["stopped_by"]), stop_ns, outages)
    except (ValueError, KeyError, TypeError) as exc:
        notes.append(
            f"the stop sidecar {path} cannot be parsed ({exc}): the recording counts as crashed"
        )
        return None


def _finalize_instrument(
    root: Path,
    day: dt.date,
    exchange: str,
    symbol: str,
    records: np.ndarray,
    span: Span,
    notes: list[str],
) -> FinalizedFile | None:
    """The recording's finalized file of one instrument: kept when it verifies, written
    when absent, reported when damaged (None)."""
    folder = instrument_folder(root, exchange, symbol)
    ticks_path, sidecar_path = finalized_paths(folder, day, span.recording)
    found = check(ticks_path, sidecar_path, day, span.recording)
    if found.state is State.ABSENT:
        folder.mkdir(parents=True, exist_ok=True)
        sidecar = sidecar_bytes(exchange, symbol, day, records, span)
        atomic_write(ticks_path, lambda handle: np.save(handle, records, allow_pickle=False))
        atomic_write(sidecar_path, lambda handle: handle.write(sidecar))
        found = check(ticks_path, sidecar_path, day, span.recording)
    if found.state is State.VERIFIED:
        ticks = found.finalized.ticks
        return FinalizedFile(ticks_path, exchange, symbol, day, span.recording, ticks)
    notes.append(
        f"the finalized ticks {ticks_path} of recording {span.recording} are damaged "
        f"({found.problem}); the recording's raw file is kept: repair moves the damaged "
        "file aside, then finalize writes it again"
    )
    return None


def _remove(paths: list[Path], notes: list[str]) -> bool:
    """Remove a finalized recording's raw file parts, then its stop sidecar; a removal
    that fails is noted (a later finalize only removes them) and stops the rest."""
    for path in paths:
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:
            notes.append(
                f"{path} could not be removed ({exc}); its ticks are finalized, and a later "
                "finalize removes it"
            )
            return False
    return True


__all__ = [
    "FinalizedFile",
    "RecordingFinalized",
    "StopFacts",
    "finalize_pending",
    "finalize_recording",
    "write_stop",
]
