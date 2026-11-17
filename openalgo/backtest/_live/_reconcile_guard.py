"""No exit beyond what the broker holds (16-session-design 5.4 "As built", 5.3; unit 12):
while the broker's books show a leg holding fewer units on the engine's side than the
engine counts, none, or the other side, no closing order of any owner (a strategy's exit,
the safe stop's flatten, the square-off, the kill switch's exit, a stop placed again) is
sent for more than the broker holds on the engine's side.

The engine exits a whole position and sizes a stop for its whole position, so such an
order would take the broker through zero. The guard holds it back from the order worker
instead: a placement of a stop or an exit, or a modify that raises a closing order's units,
beyond the room (`_reconcile_rules.exit_room`). One ERROR alert per held order says so.
When a later comparison agrees (the broker's books caught up, or the engine adopted an
explained difference), the leg's held requests that still fit what the engine holds go to
the worker in their order; the others are refused as not sent (`close`). A cancel, and a
modify that keeps or lowers the units, always go: they never add exposure. A held order
is not at the broker, so it does not keep its leg in flight (`held_orders`).

Quantities are units. Used by the loop only.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from ._reconcile_rules import exit_room
from ._types import LOGGER_NAME, AlertKind, Purpose
from ._venue_requests import ModifyRequest, PlaceRequest
from .inbox import NotSent, NotSentReason

logger = logging.getLogger(LOGGER_NAME)

#: The purposes of the orders that close a position, a leg's stop among them.
CLOSING = frozenset({Purpose.STOP, Purpose.EXIT, Purpose.RISK_EXIT, Purpose.SQUARE_OFF})


class ExitGuard:
    """The guard of one session (module docstring); `session` is the
    `_session_loop.Session`."""

    def __init__(self, session: Any) -> None:
        self._session = session
        # The broker's units of each guarded leg, by strategy, as the last read showed.
        self._broker: dict[str, int] = {}
        self._held: dict[str, list[Any]] = {}
        self._alerted: set[tuple[str, int]] = set()

    def guards(self, name: str) -> bool:
        """True while strategy `name`'s leg is guarded."""
        return name in self._broker

    def open(self, name: str, broker_units: int) -> None:
        """Guard strategy `name`'s leg: the broker holds `broker_units` (signed)."""
        if self._broker.get(name) != broker_units:
            self._session.event("exit_guard", strategy=name, broker_units=broker_units)
        self._broker[name] = broker_units

    def close(self, name: str) -> None:
        """The leg agrees with the broker again: each held request that still fits what
        the engine holds goes to the worker, in order; any other is refused, as a request
        the worker did not send (`NotSent`): a stop is then placed again for the units
        now held (R3), an exit asked for again by the retry schedule, a modify refused."""
        if self._broker.pop(name, None) is None:
            return
        self._session.event("exit_guard_lifted", strategy=name)
        session = self._session
        trader = session.trader_named(name)
        fits = []
        for request in self._held.pop(name, []):
            if self._fits(trader, request):
                fits.append(request)
                continue
            why = "held back while the broker held fewer units, and no longer fits the position"
            session.news.take(NotSent(request, NotSentReason.REFUSED, why))
        if fits:
            session.hand_over(fits)

    def _fits(self, trader: Any, request: Any) -> bool:
        """Whether a held request still fits: its order works in the engine with no change
        of it waiting, and its units are within what the engine holds."""
        kernel_order = request.kernel_order
        if kernel_order not in trader.working():
            return False
        units = request.terms.quantity if isinstance(request, ModifyRequest) else None
        if isinstance(request, PlaceRequest):
            if trader.venue.changing(kernel_order):
                return False
            units = request.intent.quantity
        return units is not None and units <= abs(round(trader.open_units()))

    def admit(self, requests: Sequence[Any]) -> list[Any]:
        """`requests` but those the guard holds (module docstring), in order."""
        admitted = []
        for request in requests:
            if self._beyond(request):
                self._hold(request)
            else:
                admitted.append(request)
        return admitted

    def held_orders(self, name: str) -> frozenset[int]:
        """The kernel orders of strategy `name` whose requests the guard holds."""
        return frozenset(request.kernel_order for request in self._held.get(name, []))

    def left(self) -> list[Any]:
        """Every request still held (the session's close reports them)."""
        return [request for held in self._held.values() for request in held]

    def _beyond(self, request: Any) -> bool:
        if not isinstance(request, PlaceRequest | ModifyRequest):
            return False
        intent = request.intent
        if intent.purpose not in CLOSING or intent.strategy not in self._broker:
            return False
        if isinstance(request, ModifyRequest):
            units = request.terms.quantity
            if units <= request.replaces.quantity:
                return False
        else:
            units = intent.quantity
        trader = self._session.trader_named(intent.strategy)
        room = exit_room(round(trader.open_units()), self._broker[intent.strategy])
        return units > room

    def _hold(self, request: Any) -> None:
        intent = request.intent
        self._held.setdefault(intent.strategy, []).append(request)
        key = (intent.strategy, request.kernel_order)
        self._session.event(
            "exit_held", strategy=intent.strategy, kernel_order=request.kernel_order
        )
        if key in self._alerted:
            return
        self._alerted.add(key)
        broker = self._broker[intent.strategy]
        what = "stop" if intent.purpose is Purpose.STOP else intent.purpose.value.replace("_", " ")
        self._session.alert(
            AlertKind.ERROR,
            f"{intent.strategy}: the {what} of {intent.symbol} {intent.exchange} "
            f"({intent.action} {intent.quantity}) is held back: OpenAlgo's position book "
            f"shows {broker} units, and it would take the broker's position through zero. "
            "It is sent once the books agree; check the position in OpenAlgo",
        )


__all__ = ["CLOSING", "ExitGuard"]
