"""The session loop's lag (16-session-design 2.9; unit 8), split from `_session_loop`: the
age of the message the loop takes, from when it was put on the inbox. Past
`LOOP_LAG_WARN_S` it is logged and written to the event stream, at most once every
`LOOP_LAG_WARN_EVERY_S`; past `LOOP_LAG_ALERT_S` it is an ERROR alert too. The loop
watches it with threads only: under the test seam a worker's call runs on the loop's
thread, so the messages it posts meanwhile wait for nothing but that call.

Times are UTC nanoseconds. Used by the loop only.
"""

from __future__ import annotations

import logging
from typing import Any

from .._layout import NANOS_PER_SECOND
from ._types import LOGGER_NAME, AlertKind
from .inbox import Envelope

logger = logging.getLogger(LOGGER_NAME)

#: The loop's lag (2.9): a warning past the first, an ERROR alert past the second.
LOOP_LAG_WARN_S = 1.0
LOOP_LAG_ALERT_S = 5.0
#: Seconds between two warnings of the lag, so a slow spell is not logged per message.
LOOP_LAG_WARN_EVERY_S = 60.0

_NS = NANOS_PER_SECOND


class LoopLag:
    """The lag watch of `session`, a `_session_loop.Session` (module docstring)."""

    def __init__(self, session: Any) -> None:
        self._session = session
        self._warned_ns: int | None = None

    def note(self, envelope: Envelope) -> None:
        """The loop takes `envelope`: a warning, and an ERROR alert, when it waited too
        long on the inbox."""
        session = self._session
        now_ns = session.now()
        lag_s = (now_ns - envelope.put_ns) / _NS
        if lag_s <= LOOP_LAG_WARN_S:
            return
        warned = self._warned_ns
        if warned is None or now_ns - warned >= round(LOOP_LAG_WARN_EVERY_S * _NS):
            self._warned_ns = now_ns
            logger.warning("the session loop is %.1f s behind its inbox", lag_s)
            session.event("loop_lag", seconds=lag_s)
        if lag_s > LOOP_LAG_ALERT_S:
            session.alert(
                AlertKind.ERROR,
                f"the session loop is more than {LOOP_LAG_ALERT_S:g} s behind: a slow "
                "strategy or a stuck journal delays ticks and order news",
            )


__all__ = ["LOOP_LAG_ALERT_S", "LOOP_LAG_WARN_EVERY_S", "LOOP_LAG_WARN_S", "LoopLag"]
