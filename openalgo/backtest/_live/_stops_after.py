"""What becomes of a leg's broker-held stop after the session's last lines
(16-session-design 3.3 step 5 "As built", unit 12).

Once the close has printed its last lines (`AfterLines.end`), the session sends nothing
those lines did not name. The close's final book read takes in the books and the order
news only: no price reaches the engine or the price rule, and the stops supervisor's
duties no longer run (`stops.StopsSupervisor`), so no target, trail or price rule acts;
the exit guard sends only the reconciler's own safe corrections (`_reconcile_guard`).

A word in that window that ends a leg's stop (filled, cancelled or rejected at the
broker) is therefore not acted on: what the engine makes of it (the stop placed again, an
exit at market) is not sent. It is noted (`heard`), so the result's note of the position
left open first says what happened to the stop, then the leg's state as
`_stops_report.left_open` gives it (`note`), and one ERROR alert per leg says the same
(`alert`): the last line is never left claiming a protection the leg no longer has. A leg
its stop's fill left flat is named too, since the engine's other orders on it may still
work at the broker.

Quantities are units. Used by the loop only.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

from ._session_trader import Trader
from ._stops_orders import LegOrders
from ._stops_report import left_open, leg_words
from ._types import AlertKind, OrderStatus, OrderUpdate

#: How a note says a stop ended, by the status of its last word.
_ENDED_WORDS: Mapping[OrderStatus, str] = MappingProxyType(
    {
        OrderStatus.COMPLETE: "filled",
        OrderStatus.CANCELLED: "cancelled",
        OrderStatus.REJECTED: "rejected",
    }
)
#: What the alert of a leg its stop's fill left flat asks of the operator.
FLAT_NOW = (
    "the engine holds no position now: check the order book in OpenAlgo, cancelling what "
    "still works"
)


class AfterLines:
    """What happened to each leg's stop after the session's last lines (module
    docstring)."""

    def __init__(self) -> None:
        self._ended = False
        # By strategy: what happened to its stop, in a note's words.
        self._words: dict[str, str] = {}

    @property
    def ended(self) -> bool:
        """True once the close has printed its last lines."""
        return self._ended

    def end(self) -> None:
        """The close has printed its last lines: from now a word that ends a leg's stop is
        noted."""
        self._ended = True

    def heard(self, trader: Trader, leg: LegOrders, kernel_order: int, update: OrderUpdate) -> bool:
        """A word of `trader`'s order `kernel_order`, before `leg` takes it: True when it
        ends the leg's stop after the last lines, which is then noted."""
        if not self._ended or kernel_order != leg.stop or not update.status.terminal:
            return False
        name = leg.orders[kernel_order].name()
        ended = _ENDED_WORDS[update.status]
        self._words[trader.name] = (
            f"its stop {name} {ended} at the broker after the session's last lines"
        )
        return True

    def note(self, stops: Any, trader: Trader) -> str:
        """The note of `trader`'s position left open: what happened to its stop after the
        last lines, when anything did, then the leg's state (`left_open`); `stops` is the
        stops supervisor."""
        rest = left_open(stops, trader)
        words = self._words.get(trader.name)
        return rest if words is None else f"{words}; {rest}"

    def alert(self, session: Any) -> None:
        """One ERROR alert per leg whose stop ended after the last lines, naming the leg and
        its note; `session` is the `_session_loop.Session`."""
        for trader in session.traders:
            words = self._words.get(trader.name)
            if words is None:
                continue
            units = trader.open_units()
            if units:
                text = f"{leg_words(trader, units)}: {self.note(session.stops, trader)}"
            else:
                text = f"{trader.name} {trader.symbol} {trader.exchange}: {words}; {FLAT_NOW}"
            session.alert(AlertKind.ERROR, text)


__all__ = ["FLAT_NOW", "AfterLines"]
