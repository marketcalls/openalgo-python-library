"""When OpenAlgo is silent (16-session-design 4.9): after a client timeout or a dropped
connection the order path sends nothing for any instrument until OpenAlgo answers a call
again (`_order_answers.ServerWatch`), and only the order path's own calls lift that: its
polls, its order-book reads for unbound intents, its settlement reads and a probe.

- **The probe.** While OpenAlgo is silent, a protective request (a stop, an exit, a
  protective modify or cancel) waits in the order worker's queue, and the order path has
  called OpenAlgo for nothing else in the last `PROBE_EVERY_S` (its other duties are
  not due), the session asks the worker for a probe (`_worker_queue.Probe`, one
  order-book read), at most one at a time: the hold lifts as soon as OpenAlgo answers,
  and the waiting request goes then. With nothing waiting no probe is asked: the next
  call due lifts the hold.
- **The alert.** Once the silence has lasted `SILENT_ALERT_AFTER_S`, one ERROR alert
  says for how long, how many positions are open, and how many of them have no confirmed
  stop at the broker covering them; it is sent again at most every
  `SILENT_ALERT_EVERY_S` while the same silence lasts.

The broker's stops are the protection meanwhile (D31); the unprotected windows go on
counting (4.2, `stops.StopsSupervisor.review`). The worker's view of the silence is read
through `OrderWorker.silent_since_ns` and `OrderWorker.asked_ns`. Times are UTC
nanoseconds. Used by the loop only.
"""

from __future__ import annotations

import logging
from typing import Any

from .._layout import NANOS_PER_SECOND
from ._types import LOGGER_NAME, AlertKind
from ._venue_requests import OrderRequest, Priority
from ._worker_queue import Probe

logger = logging.getLogger(LOGGER_NAME)

#: Seconds between two probes while a protective request waits on a silent OpenAlgo
#: (16-session-design 4.9).
PROBE_EVERY_S = 2.0
#: Seconds of silence after which the operator is alerted (4.9).
SILENT_ALERT_AFTER_S = 10.0
#: Seconds between two alerts of one silence, so a long one is not alerted per step.
SILENT_ALERT_EVERY_S = 60.0

_NS = NANOS_PER_SECOND


class SilentServer:
    """What the session does while OpenAlgo is silent (module docstring). `session` is the
    `_session_loop.Session`, `stops` its `StopsSupervisor` (for the stops it counts)."""

    def __init__(self, session: Any, stops: Any) -> None:
        self._session = session
        self._stops = stops
        # The protective requests the worker put back, by request ID.
        self._waiting: set[int] = set()
        self._probe_asked_ns: int | None = None
        # The silence alerted (its start) and when its next alert is due.
        self._silence_ns: int | None = None
        self._alert_due_ns = 0

    def held(self, request: OrderRequest) -> None:
        """The worker put `request` back to send later (OpenAlgo silent, or an intent of
        its instrument unbound): a protective one waits."""
        if request.priority is Priority.PROTECT:
            self._waiting.add(request.request_id)

    def done(self, request_id: int) -> None:
        """The request numbered `request_id` was sent, or will not be: it waits no more."""
        self._waiting.discard(request_id)

    def due(self, now_ns: int) -> None:
        """Step D: the probe and the alert due while OpenAlgo is silent."""
        worker = self._session.parts.worker
        since_ns = worker.silent_since_ns
        if since_ns is None:
            self._silence_ns = self._probe_asked_ns = None
            return
        if since_ns != self._silence_ns:
            self._silence_ns = since_ns
            self._alert_due_ns = since_ns + round(SILENT_ALERT_AFTER_S * _NS)
        if now_ns >= self._alert_due_ns:
            self._alert_due_ns = now_ns + round(SILENT_ALERT_EVERY_S * _NS)
            self._alert(now_ns - since_ns)
        if self._waiting and self._probe_due(now_ns, worker.asked_ns):
            self._probe_asked_ns = now_ns
            self._session.hand_over([Probe()])

    def _probe_due(self, now_ns: int, asked_ns: int | None) -> bool:
        """Whether a probe is due: the order path has called OpenAlgo for nothing for
        `PROBE_EVERY_S`, and the probe asked last has been made (a call came since)."""
        asked_probe = self._probe_asked_ns
        if asked_probe is not None and (asked_ns is None or asked_ns < asked_probe):
            return False
        return asked_ns is None or now_ns - asked_ns >= round(PROBE_EVERY_S * _NS)

    def _alert(self, silent_ns: int) -> None:
        stops = self._stops
        units = [
            (trader, round(abs(trader.open_units())))
            for trader in self._session.traders
            if trader.open_units() != 0
        ]
        bare = sum(1 for trader, held in units if stops.covered(trader) < held)
        self._session.alert(
            AlertKind.ERROR,
            f"OpenAlgo has not answered for {silent_ns / _NS:.0f} s: {len(units)} "
            f"position{'' if len(units) == 1 else 's'} open, {bare} without a confirmed "
            "stop at the broker",
        )


__all__ = ["PROBE_EVERY_S", "SILENT_ALERT_AFTER_S", "SILENT_ALERT_EVERY_S", "SilentServer"]
