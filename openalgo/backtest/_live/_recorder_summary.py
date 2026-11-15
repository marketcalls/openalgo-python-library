"""What a recording and a finalize report (private to `recorder`, which re-exports the
types), and the log lines a recording ends with.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .._calendar_convert import format_ist
from ._tick_finalize import FinalizedFile, RecordingFinalized
from ._tick_spans import Outage, Stop
from ._types import LOGGER_NAME

logger = logging.getLogger(LOGGER_NAME)


@dataclass(frozen=True)
class RecordSummary:
    """What a recording did.

    `recording` is its number (its spans carry it). `ticks`: ticks written per (exchange,
    symbol), every instrument asked for included. `outages`: the feed's outages, in
    order. `files`: the finalized files written or found, one per instrument and
    recording finalized (a crashed recording's of the same day among them), with their
    tick counts. `dropped`: feed messages that made no tick (`ticks.TickTally`).
    `not_asked`: ticks of instruments the recording did not ask for, not recorded.
    `event_times_rejected`: ticks kept without an implausible last trade time.
    `overflowed`: messages lost because the writer's queue was full. `lost`: messages
    taken off the queue that were not written (writes that kept failing). `rejected`:
    records the raw files held that could not be the recordings' (a zero-filled tail
    after a power loss). `error`: the first write or finalize error, None when there was
    none; a raw file it left stays for `finalize`. `notes`: what to look at (files
    damaged or left pending, calls that failed while stopping, a second Ctrl+C).
    """

    started_ns: int
    stopped_ns: int
    stopped_by: Stop
    recording: int
    ticks: Mapping[tuple[str, str], int]
    outages: tuple[Outage, ...]
    files: tuple[FinalizedFile, ...]
    dropped: int
    not_asked: int
    event_times_rejected: int
    overflowed: int
    lost: int
    rejected: int
    error: OSError | None
    notes: tuple[str, ...]


@dataclass(frozen=True)
class FinalizeReport:
    """What `finalize` did: the finalized files written or found, the records the raw
    files held that could not be the recordings' (`rejected`), the bytes of torn last
    records cut off, the recordings whose raw file stays (a damaged file or a removal
    that failed: see `notes`), and what it found on the way."""

    files: tuple[FinalizedFile, ...]
    rejected: int
    torn_bytes: int
    pending: tuple[int, ...]
    notes: tuple[str, ...]


def report_of(finalized: Sequence[RecordingFinalized], notes: Sequence[str]) -> FinalizeReport:
    """The report of finalized recordings."""
    return FinalizeReport(
        files=tuple(file for done in finalized for file in done.files),
        rejected=sum(done.rejected for done in finalized),
        torn_bytes=sum(done.torn_bytes for done in finalized),
        pending=tuple(done.recording for done in finalized if not done.removed),
        notes=tuple(notes),
    )


def log_summary(summary: RecordSummary) -> None:
    """The log lines a recording ends with: what it wrote, then each thing to look at."""
    logger.info(
        "recording %d: %d ticks of %d instruments from %s to %s (%s), %d feed outages",
        summary.recording,
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
        (summary.rejected, "records of the raw file could not be the recording's (a power loss?)"),
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


__all__ = ["FinalizeReport", "RecordSummary", "log_summary", "report_of"]
