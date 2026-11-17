"""What `bt.record` runs (16-session-design 1.3; unit 13): the tick recorder (`recorder.record`)
on a feed client of its own, on the wall clock, until a time of day the user names.

- **Its own client.** The SDK keeps one data callback per feed mode, so recording on a
  client someone else feeds would take their callback: the recording's feed client is a
  new `openalgo.api` with the user's settings (`clients.settings_of`), closed at the end.
  The user's client is only read.
- **The clocks.** The wall clock stamps the ticks (`time.time_ns`); a monotonic clock that
  never steps back schedules the writes (`time.monotonic_ns`).
- **Until when** (`stop_at_of`): an IST "HH:MM" of the recording's day, a datetime (aware,
  or naive and read as IST), or None for the latest close of the instruments' exchanges
  that day (the recorder's default).
- **A write that fails** stops the recording (`Stop.WRITE_FAILED`): the ticks it could not
  write cannot be fetched again, so it raises LiveError chained from the error, naming
  the raw files kept, which the next recording of the day finalizes.

Times are UTC nanoseconds.
"""

from __future__ import annotations

import datetime as dt
import os
import time
from collections.abc import Mapping, Sequence
from typing import Any

from .._calendar_convert import IST, format_ist
from ..errors import ConfigError, LiveError
from ._recorder_loop import Monotonic
from ._recorder_summary import RecordSummary
from ._tick_files import ist_day, raw_parts_of, ticks_root
from ._tick_spans import Stop
from ._types import Now, Sleep
from .clients import settings_of
from .recorder import record

#: How `until` is written as a time of day: "HH:MM", IST.
TIME_OF_DAY = "%H:%M"
#: The instant UTC nanoseconds count from.
EPOCH = dt.datetime(1970, 1, 1, tzinfo=dt.UTC)


def stop_at_of(until: Any, started_ns: int) -> int | None:
    """The instant (UTC nanoseconds) a recording begun at `started_ns` stops at, from
    `until` (module docstring); None for the recorder's default. Raises ConfigError for
    any other value, saying what to pass."""
    if until is None:
        return None
    if isinstance(until, str):
        try:
            clock = dt.datetime.strptime(until, TIME_OF_DAY).time()
        except ValueError:
            raise ConfigError(
                f"until takes an IST time of day as 'HH:MM' (for example '15:30'), not {until!r}"
            ) from None
        until = dt.datetime.combine(ist_day(started_ns), clock)
    if isinstance(until, dt.datetime):
        moment = until if until.tzinfo is not None else until.replace(tzinfo=IST)
        return (moment - EPOCH) // dt.timedelta(microseconds=1) * 1_000
    raise ConfigError(
        f"until takes an IST time of day ('15:30'), a datetime, or None for the day's close, "
        f"not {until!r}"
    )


def record_day(
    client: Any,
    instruments: Sequence[Mapping[str, str]],
    *,
    mode: str = "quote",
    cache_dir: str | os.PathLike[str] | None = None,
    until: Any = None,
    now: Now = time.time_ns,
    monotonic: Monotonic = time.monotonic_ns,
    wait: Sleep | None = None,
    feed: Any = None,
) -> RecordSummary:
    """Record the ticks of `instruments` until `until` (module docstring) and return the
    recording's summary. `now`, `monotonic`, `wait` and `feed` (a feed client used as it
    is, not derived from `client`) are the test seam.

    Raises ConfigError for a client that is not an `openalgo.api` client and for unusable
    arguments, LiveError when a write failed (chained from it) and what `recorder.record`
    raises."""
    stop_at_ns = stop_at_of(until, now())
    own = feed is None
    if own:
        settings = settings_of(client)
        feed = settings.build(timeout_s=settings.timeout_s)
    try:
        summary = record(
            feed,
            instruments,
            now=now,
            monotonic=monotonic,
            mode=mode,
            cache_dir=cache_dir,
            stop_at_ns=stop_at_ns,
            wait=wait,
        )
    finally:
        if own:
            feed.close()
    if summary.stopped_by is Stop.WRITE_FAILED:
        raise _write_failed(summary, cache_dir)
    return summary


def _write_failed(summary: RecordSummary, cache_dir: str | os.PathLike[str] | None) -> LiveError:
    """The error of a recording whose writes failed, chained from the summary's error."""
    day = ist_day(summary.started_ns)
    kept = raw_parts_of(ticks_root(cache_dir), day).get(summary.recording, [])
    files = ", ".join(os.fspath(path) for path in kept) or "none left"
    error = LiveError(
        f"recording {summary.recording} stopped at {format_ist(summary.stopped_ns)} because "
        f"its ticks could not be written ({summary.error}); ticks received since cannot be "
        f"fetched again. Raw files kept: {files}. Fix the folder (free space, permissions), "
        "then record again today: the next recording of the day finalizes them"
    )
    error.__cause__ = summary.error
    return error


__all__ = ["TIME_OF_DAY", "record_day", "stop_at_of"]
