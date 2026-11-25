"""The deadline of a stop never confirmed (16-session-design 4.2; 08-live-trading 6.2): a
leg's stop accepted with an order ID and with no word since that says it works, after
`UNPROTECTED_EXIT_AFTER_S`, exits the leg (`request_exit(unprotected)`: the kernel cancels
the stop, then exits at market), with an ERROR alert.

It is not acted on while the order worker is inside a call, has order words it has not
posted, or the loop's inbox holds messages: a confirmation may be among them. Under
protect by state (`_stops_protect`), which asks for no exit of the leg, the deadline only
sends the ERROR alert: an exit there would cancel the stop protect by state placed, and
the stop would be placed and cancelled again at every deadline. Each stop meets its
deadline once.

A CNC or NRML leg's stop GTT meets the same deadline from the hand-over of its place to
the order worker until the worker's answer (20-positional-live 2, "Day 1, entry fills"; the
lead's ruling N3b-2): the worker answers PLACED only once a read of the GTT book shows it
resting. The exit then waits for the GTT's cancel, which waits for the place's answer.

Times are UTC nanoseconds. Used by the loop only.
"""

from __future__ import annotations

from typing import Any

from .._layout import NANOS_PER_SECOND
from ._margin_reject import HELD_EXIT_WORDS, HELD_FOR_READ
from ._session_trader import Trader
from ._stops_orders import Sent
from ._types import AlertKind

#: Seconds a stop accepted with no confirming word may leave its leg before the leg is
#: exited (08-live-trading 6.2).
UNPROTECTED_EXIT_AFTER_S = 10.0
#: The kernel's reason for that exit (its RISK_EXIT_REASONS).
UNPROTECTED = "unprotected"

_DEADLINE_NS = round(UNPROTECTED_EXIT_AFTER_S * NANOS_PER_SECOND)


class Deadlines:
    """The deadlines of the stops of one session (module docstring). `session` is the
    `_session_loop.Session`, `stops` its `StopsSupervisor`."""

    def __init__(self, session: Any, stops: Any) -> None:
        self._session = session
        self._stops = stops
        self._met: set[tuple[str, int]] = set()
        self._gtts_met: set[tuple[str, int]] = set()

    def due(self, now_ns: int) -> None:
        """Act on each stop past its deadline (module docstring)."""
        session = self._session
        worker = session.parts.worker
        for trader in session.traders:
            leg = self._stops.leg(trader)
            stop = leg.orders.get(leg.stop) if leg.stop is not None else None
            if stop is None or stop.state is not Sent.BOUND or stop.bound_ns is None:
                continue
            key = (trader.name, stop.kernel_order)
            if key in self._met or now_ns < stop.bound_ns + _DEADLINE_NS:
                continue
            # Read in this order: each is raised before the one ahead of it clears.
            if worker.busy_since_ns is not None or worker.raw_pending or session.inbox_waiting:
                continue
            self._met.add(key)
            if self._stops.protection.holds(trader):
                self._alert(
                    trader,
                    stop.order_id,
                    "the session asks for no exit while it protects the leg: check the order book",
                )
                continue
            asked = session.margin.request_exit(trader, UNPROTECTED)
            session.event("stop_unconfirmed", strategy=trader.name, outcome=asked)
            what = "the position is exited (the stop cancelled first)"
            self._alert(trader, stop.order_id, HELD_EXIT_WORDS if asked == HELD_FOR_READ else what)
        self._gtts_due(now_ns)

    def _gtts_due(self, now_ns: int) -> None:
        """Act on each leg's stop GTT past its deadline (module docstring)."""
        session = self._session
        worker = session.parts.worker
        for trader in session.traders:
            placing = [] if trader.gtts is None else trader.gtts.placing()
            for gtt_id, since_ns in placing:
                key = (trader.name, gtt_id)
                if key in self._gtts_met or now_ns < since_ns + _DEADLINE_NS:
                    continue
                if worker.busy_since_ns is not None or worker.raw_pending or session.inbox_waiting:
                    continue
                self._gtts_met.add(key)
                if self._stops.protection.holds(trader):
                    self._alert(trader, "GTT", "the session asks for no exit while it protects it")
                    continue
                asked = session.margin.request_exit(trader, UNPROTECTED)
                session.event("stop_unconfirmed", strategy=trader.name, outcome=asked, gtt=gtt_id)
                what = "the position is exited once the GTT's place is answered and it is cancelled"
                self._alert(trader, "GTT", what)

    def _alert(self, trader: Trader, order_id: str, what: str) -> None:
        self._session.alert(
            AlertKind.ERROR,
            f"{trader.name}: stop {order_id} of {trader.symbol} {trader.exchange} was "
            f"accepted and not confirmed working within {UNPROTECTED_EXIT_AFTER_S:g} s: "
            f"{what}",
        )


__all__ = ["UNPROTECTED", "UNPROTECTED_EXIT_AFTER_S", "Deadlines"]
