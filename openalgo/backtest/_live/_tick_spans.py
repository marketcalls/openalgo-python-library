"""What a recorded day says about its own coverage (private to `recorder`, which
re-exports the types): the span of each recording that wrote it, the feed outages
inside a span, and a day read back with them.

A moment of a day is in one of three states: recorded (inside a recording's span, the
feed up), a feed outage (inside a span, the feed down), or not recorded (outside every
span: no recording ran, or one had crashed). A span or its outages can be unknown: a
recording that crashed lost its outages with it, and an index rebuilt from the day files
knows no spans at all.
"""

from __future__ import annotations

import datetime as dt
import enum
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from ._tick_files import is_id


@dataclass(frozen=True, order=True)
class Outage:
    """A span in which the feed was down, UTC nanoseconds: from the feed's disconnect to
    its reconnect, or to the end of the recording when it was still down then."""

    start_ns: int
    end_ns: int

    def holds(self, instant_ns: int) -> bool:
        """Whether the feed was down at `instant_ns`."""
        return self.start_ns <= instant_ns < self.end_ns


class Ending(enum.Enum):
    """How a recording's span on a day ended."""

    STOP = "stop"  # it stopped and finalized: its stop and outages are known
    CRASH = "crash"  # it did not finalize (killed, crashed, failed): it stops at its last tick
    UNKNOWN = "unknown"  # the index was rebuilt from the day files: nothing is known


class Coverage(enum.Enum):
    """What a recorded day says about a moment (see the module docstring)."""

    RECORDED = "recorded"
    OUTAGE = "outage"
    NOT_RECORDED = "not_recorded"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Span:
    """The time one recording covered on a day, UTC nanoseconds, both ends included.

    `recording` is its id. For a recording that stopped, `stop_ns` is its stop; for one
    that crashed, the receive time of its last tick of the day (its start when it had
    none). `start_ns` and `stop_ns` are cut to the day. `outages` are the feed's outages
    that started inside it, None when not known (a crash loses them).
    """

    recording: str
    start_ns: int | None
    stop_ns: int | None
    ended_by: Ending
    outages: tuple[Outage, ...] | None

    @property
    def known(self) -> bool:
        """Whether its start and stop are known."""
        return self.start_ns is not None and self.stop_ns is not None

    def contains(self, instant_ns: int) -> bool:
        """Whether a known span covers `instant_ns`."""
        return self.known and self.start_ns <= instant_ns <= self.stop_ns


@dataclass(frozen=True)
class RecordedDay:
    """One instrument's recorded ticks of an IST day.

    `records` are of `TICK_DTYPE`, in receive order (the recordings in the order of
    their start, each in its sequence); `spans` the recordings that wrote the day, in the
    same order; `pending` is True when a raw file of the day waits to be finalized (a
    recording runs, or one crashed): its ticks are not in `records` yet.
    """

    exchange: str
    symbol: str
    day: dt.date
    records: np.ndarray
    spans: tuple[Span, ...]
    pending: bool

    @property
    def outages(self) -> tuple[Outage, ...] | None:
        """Every feed outage of the day in start order, None when a span's are unknown."""
        if any(span.outages is None for span in self.spans):
            return None
        return tuple(sorted(outage for span in self.spans for outage in span.outages))

    def coverage(self, instant_ns: int) -> Coverage:
        """What the day says about the moment `instant_ns` (UTC nanoseconds)."""
        for span in self.spans:
            if span.contains(instant_ns):
                if span.outages is None:
                    return Coverage.UNKNOWN
                down = any(outage.holds(instant_ns) for outage in span.outages)
                return Coverage.OUTAGE if down else Coverage.RECORDED
        if any(not span.known for span in self.spans):
            return Coverage.UNKNOWN
        return Coverage.NOT_RECORDED


def joined(spans: Iterable[Span]) -> list[Span]:
    """The spans with those of one recording joined into one, in first-seen order: the
    earliest known start, the latest known stop, the best known ending, and the outages
    of all its parts (None when one part's are unknown)."""
    parts: dict[str, list[Span]] = {}
    for span in spans:
        parts.setdefault(span.recording, []).append(span)
    return [_join(recording, of) for recording, of in parts.items()]


def _join(recording: str, parts: list[Span]) -> Span:
    if len(parts) == 1:
        return parts[0]
    starts = [part.start_ns for part in parts if part.start_ns is not None]
    stops = [part.stop_ns for part in parts if part.stop_ns is not None]
    endings = {part.ended_by for part in parts}
    ended_by = next(ending for ending in Ending if ending in endings)
    unknown = any(part.outages is None for part in parts)
    outages = None if unknown else tuple(sorted({o for part in parts for o in part.outages}))
    return Span(recording, min(starts, default=None), max(stops, default=None), ended_by, outages)


def span_to_json(span: Span) -> dict[str, Any]:
    """A span as the index stores it."""
    return {
        "recording": span.recording,
        "start_ns": span.start_ns,
        "stop_ns": span.stop_ns,
        "ended_by": span.ended_by.value,
        "outages": None
        if span.outages is None
        else [[outage.start_ns, outage.end_ns] for outage in span.outages],
    }


def span_from_json(stored: Mapping[str, Any]) -> Span:
    """A span from the index. Raises ValueError, TypeError or KeyError when it is
    malformed."""
    recording = stored["recording"]
    start, stop = stored["start_ns"], stored["stop_ns"]
    outages = stored["outages"]
    if not is_id(recording) or not all(_instant_or_none(value) for value in (start, stop)):
        raise ValueError("a span's recording or times are malformed")
    if outages is not None:
        outages = tuple(Outage(*_instants(pair)) for pair in outages)
    return Span(recording, start, stop, Ending(stored["ended_by"]), outages)


def _instant_or_none(value: Any) -> bool:
    return value is None or (isinstance(value, int) and not isinstance(value, bool))


def _instants(pair: Any) -> tuple[int, int]:
    whole = isinstance(pair, list) and len(pair) == 2
    if not (whole and all(value is not None and _instant_or_none(value) for value in pair)):
        raise ValueError("an outage is malformed")
    return pair[0], pair[1]


__all__ = [
    "Coverage",
    "Ending",
    "Outage",
    "RecordedDay",
    "Span",
    "joined",
    "span_from_json",
    "span_to_json",
]
