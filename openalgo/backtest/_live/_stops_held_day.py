"""An instrument held for the day, and the square-off it cannot have (16-session-design
4.9 and 6.2; 18-multi-strategy 4.4 and F105; 2.4.0 item P3-3).

A placement the order book cannot tell apart from another order (an AMBIGUOUS resolution)
stays unbound for the rest of the day: the order book only grows, so the orders it cannot
tell apart stay so. While it is unbound the order path sends nothing for its symbol,
exchange and product, for any strategy of the session that trades it, so no order of the
session can take one of the candidates and settle it later, and a restart refuses to start
while an intent is ambiguous (`_restart_books`). The instrument is held for the day
(`holder`), for every member of its group (`_groups`), a strategy alone included, and the
hold is final: a cancel kept here is never handed over.

So the square-off cannot close such a position: its cancel-then-exit (3.4, the kernel's
exit owner) would cancel the leg's stop first, and that cancel could never be sent. For an
instrument held for the day the square-off's cancel-then-exit is not started at the
broker: a cancel of a member's stop the kernel decides at or after the member's square-off
is kept here, never handed to the order worker (`admit`), and the stop keeps working at
the broker and is reported as working (`StopsSupervisor.cancelling` leaves it out, so its
units count as covered and no unprotected window opens for it). The kernel's exit, which
waits for that cancel, is never sent either. Every other square-off is unchanged.

Once the square-off has come, one ERROR alert per member with a position says that no
square-off is sent for it, then the note of its position left open (`review`,
`_stops_report.left_open`), which ends with why and what to do (`words`): nothing can be
sent for the instrument today, so cancel the leg's stop working at the broker (a sale by
hand beside it would let it sell again), check the order book when an order may still
work, then close it by hand in OpenAlgo, with the broker's own MIS square-off as the
backstop. That advice holds while the session runs, unlike the note of another leg: the
session sends nothing more for the instrument today, so it cannot sell beside the
operator.

Times are UTC nanoseconds. Used by the loop only.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ._session_trader import Trader
from ._stops_orders import Sent
from ._stops_report import left_open, leg_words, stop_state
from ._types import AlertKind, Purpose
from ._venue_requests import CancelRequest


class DayHolds:
    """The instruments held for the day and the square-off cancels kept back for them
    (module docstring), a part of the stops supervisor `stops` of `session` (a
    `_session_loop.Session`)."""

    def __init__(self, session: Any, stops: Any) -> None:
        self._session = session
        self._stops = stops
        self._kept: dict[str, list[CancelRequest]] = {}
        self._alerted: set[str] = set()

    def holder(self, trader: Trader) -> str | None:
        """The order that holds `trader`'s instrument for the day, in words ("LateTrend's
        stop order, intent 20261116-live-1-LateTrend-o2"), or None while nothing holds it:
        the first placement of a member of its group, in call order, that the order book
        cannot tell apart."""
        for member in self._session.groups.members(trader):
            for watched in self._stops.leg(member).orders.values():
                if watched.state is Sent.AMBIGUOUS:
                    purpose = watched.purpose.value.lower()
                    return f"{member.name}'s {purpose} order, intent {watched.intent_id}"
        return None

    def words(self, trader: Trader, *, check: bool) -> str | None:
        """What a note of `trader`'s position says of its instrument held for the day, or
        None when it is not held: why nothing can be sent for it, then what to do (module
        docstring), the order book checked first when `check` (an order may still work, or
        the stop may be at the broker without covering the position)."""
        holder = self.holder(trader)
        if holder is None:
            return None
        stop_id, confirmed = stop_state(self._stops, trader)
        steps = [f"cancel its stop {stop_id} at the broker"] if confirmed else []
        if check:
            other = "else " if steps else ""
            steps.append(f"check the order book, cancelling what {other}still works")
        steps.append("close it by hand in OpenAlgo")
        return (
            f"{trader.symbol} {trader.exchange} is held for the day by {holder}, which the "
            "order book cannot tell apart from another order: nothing can be sent for it "
            f"today, so {', then '.join(steps)}; the broker's own MIS square-off remains the "
            "backstop"
        )

    def admit(self, requests: Sequence[Any]) -> list[Any]:
        """`requests` in order, but the square-off's cancels of a stop on an instrument held
        for the day, which are kept (module docstring)."""
        admitted = []
        for request in requests:
            if self._keeps(request):
                self._keep(request)
            else:
                admitted.append(request)
        return admitted

    def held_cancels(self, name: str) -> frozenset[int]:
        """The kernel orders of strategy `name` whose cancel is kept back: they work at the
        broker."""
        return frozenset(request.kernel_order for request in self._kept.get(name, []))

    def review(self, now_ns: int) -> None:
        """Once a member's square-off has come at `now_ns`, one ERROR alert for each member
        with a position on an instrument held for the day (module docstring)."""
        session = self._session
        for trader in session.traders:
            if trader.name in self._alerted or now_ns < trader.square_off_ns:
                continue
            units = trader.open_units()
            if not units or self.holder(trader) is None:
                continue
            self._alerted.add(trader.name)
            note = left_open(self._stops, trader)
            session.alert(
                AlertKind.ERROR,
                f"{leg_words(trader, units)}: no square-off is sent for it: {note}",
            )

    def _keeps(self, request: Any) -> bool:
        if not isinstance(request, CancelRequest) or request.intent.purpose is not Purpose.STOP:
            return False
        trader = self._session.trader_named(request.intent.strategy)
        past = self._session.now() >= trader.square_off_ns
        return past and self.holder(trader) is not None

    def _keep(self, request: CancelRequest) -> None:
        name = request.intent.strategy
        self._kept.setdefault(name, []).append(request)
        self._session.event("square_off_held", strategy=name, kernel_order=request.kernel_order)


__all__ = ["DayHolds"]
