"""What a recording and a finalize report (private to `recorder`, which re-exports the
types), and the log lines a recording ends with.
"""

from __future__ import annotations

import enum
import logging
from collections.abc import Mapping
from dataclasses import dataclass

from .._calendar_convert import format_ist
from ._tick_days import DayWritten
from ._tick_spans import Outage
from ._types import LOGGER_NAME

logger = logging.getLogger(LOGGER_NAME)


class Stop(enum.Enum):
    """Why a recording stopped."""

    CLOSE = "close"  # the day's close from the calendar, the default stop
    STOP_AT = "stop_at"  # the `stop_at_ns` the call gave
    INTERRUPT = "interrupt"  # Ctrl+C (KeyboardInterrupt)
    WRITE_FAILED = "write_failed"  # a write of recorded ticks failed (`RecordSummary.error`)


@dataclass(frozen=True)
class RecordSummary:
    """What a recording did.

    `recording` is its id (its spans carry it). `ticks`: ticks written per (exchange,
    symbol), every instrument asked for included. `outages`: the feed's outages, in
    order. `files`: the day files written, one per instrument and day, with each day's
    tick count. `dropped`: feed messages that made no tick (`ticks.TickTally`).
    `event_times_rejected`: ticks kept without an implausible last trade time.
    `overflowed`: messages lost because the writer's queue was full. `lost`: messages
    taken off the queue that were neither written nor dropped (writes that kept failing).
    `error`: the first write or finalize error, None when there was none; the raw files
    it left stay for `finalize`. `notes`: files that failed verification or could not be
    removed, and calls that failed while stopping (a second Ctrl+C among them).
    """

    started_ns: int
    stopped_ns: int
    stopped_by: Stop
    recording: str
    ticks: Mapping[tuple[str, str], int]
    outages: tuple[Outage, ...]
    files: tuple[DayWritten, ...]
    dropped: int
    event_times_rejected: int
    overflowed: int
    lost: int
    error: OSError | None
    notes: tuple[str, ...]


@dataclass(frozen=True)
class FinalizeReport:
    """What `finalize` did: the day files it wrote and what it found on the way."""

    files: tuple[DayWritten, ...]
    notes: tuple[str, ...]


def log_summary(summary: RecordSummary) -> None:
    """The log lines a recording ends with: what it wrote, then each thing to look at."""
    logger.info(
        "recorded %d ticks of %d instruments from %s to %s (%s), %d feed outages",
        sum(summary.ticks.values()),
        len(summary.ticks),
        format_ist(summary.started_ns),
        format_ist(summary.stopped_ns),
        summary.stopped_by.value,
        len(summary.outages),
    )
    for (exchange, symbol), count in summary.ticks.items():
        if count == 0:
            logger.warning("no tick of %s:%s was recorded: check the symbol", exchange, symbol)
    for count, text in (
        (summary.overflowed, "feed messages were lost: the writer fell behind the feed"),
        (summary.lost, "ticks taken off the queue could not be written"),
        (
            summary.event_times_rejected,
            "ticks carried a last trade time that is no plausible instant; they were kept "
            "without it",
        ),
    ):
        if count:
            logger.warning("%d %s", count, text)
    if summary.error is not None:
        logger.error("the recording ended by an error: %s", summary.error)


__all__ = ["FinalizeReport", "RecordSummary", "Stop", "log_summary"]
