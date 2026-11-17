"""What becomes of a leg and its broker-held stop after the session's last lines
(16-session-design 3.3 step 5 "As built", unit 12).

Once the close has printed its last lines (`AfterLines.end`), the session sends nothing
those lines did not name. The close's final book read takes in the books and the order
news only: no price reaches the engine or the price rule, and the stops supervisor's
duties no longer run (`stops.StopsSupervisor`), so no target, trail or price rule acts;
the exit guard sends only what takes exposure away (`_reconcile_guard`), and the read
waits for its answers before the order worker closes.

A word in that window that ends a leg's stop (filled, cancelled or rejected at the
broker) is therefore not acted on: what the engine makes of it (the stop placed again, an
exit at market) is not sent. It is noted (`heard`), and so is a leg whose units changed
in that window (a fill whose answer came then, a sale by hand the reconciler adopted), as
the engine held them when the lines were printed (`end`). The result's note of a position
left open first says what happened, then the leg's state as `_stops_report.left_open`
gives it (`note`), and one ERROR alert per leg says the same (`alert`): the last line is
never left claiming a position or a protection the leg no longer has.

A leg whose last comparison saw a difference with OpenAlgo's books for the first time
(fewer units, none, more or the other side: `Reconciler.unconfirmed`) has no later read to
confirm it once the session has ended (16-session-design 5.4 "As built", round 7). It is
reported whatever the engine holds, flat included: the note, and one ERROR alert, name
OpenAlgo's units and the engine's (`differs_words`), that nothing was sent for the
difference, then what works toward the leg and its stop with the order book to check
before closing it by hand (`_stops_report.left_open`); the result lists it at OpenAlgo's
units (`_session_close`). A leg left open that the exit guard still holds after a
difference seen twice (`CLOSED_DIFFERING`) is alerted too: no comparison agreed after it,
so a line printed before the final read may have said its stop protects it; the alert and
the note name the units OpenAlgo's books showed and the stop's own. A leg left flat is
alerted when anything happened to it, or when an order of it still works in the engine
(`_stops_report.still_working`), since that order may work at the broker; no line named a
flat leg. Each alert names the changes of the leg's orders the exit guard held back (a
held placement's order is named by the note as not sent). Nothing is sent for any of it.

Quantities are signed units. Used by the loop only.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any

from ._reconcile_findings import Sighting
from ._session_trader import Trader
from ._stops_orders import LegOrders
from ._stops_report import left_open, leg_words, still_working
from ._types import AlertKind, OrderStatus, OrderUpdate
from ._venue_requests import CancelRequest, ModifyRequest

#: How a note says a stop ended, by the status of its last word.
_ENDED_WORDS: Mapping[OrderStatus, str] = MappingProxyType(
    {
        OrderStatus.COMPLETE: "filled",
        OrderStatus.CANCELLED: "cancelled",
        OrderStatus.REJECTED: "rejected",
    }
)
#: What the alert of a leg left flat asks of the operator.
FLAT_NOW = (
    "the engine holds no position now: check the order book in OpenAlgo, cancelling what "
    "still works"
)
#: What the alert of a leg the exit guard still holds at the close says first.
CLOSED_DIFFERING = "the session closed before OpenAlgo's books agreed with the engine"
#: What the note and the alert of a leg whose difference no later read confirms say of
#: it (`differs_words`).
DIFFERS = (
    "OpenAlgo's books showed {broker} units against the engine's {engine} at the last "
    "read, and the session ended before they agreed: nothing was sent for the difference"
)
#: When it happened, in a note's words.
_AFTER_THE_LINES = "after the session's last lines"


class AfterLines:
    """What happened to each leg and its stop after the session's last lines (module
    docstring)."""

    def __init__(self) -> None:
        self._ended = False
        # By strategy: the engine's signed units when the last lines were printed.
        self._units: dict[str, float] = {}
        # By strategy: its stop's name and how it ended after them.
        self._stops: dict[str, tuple[str, OrderStatus]] = {}

    @property
    def ended(self) -> bool:
        """True once the close has printed its last lines."""
        return self._ended

    def end(self, traders: Sequence[Trader]) -> None:
        """The close has printed its last lines: the units each of `traders` holds now are
        kept, and from now a word that ends a leg's stop is noted."""
        self._ended = True
        self._units = {trader.name: trader.open_units() for trader in traders}

    def heard(self, trader: Trader, leg: LegOrders, kernel_order: int, update: OrderUpdate) -> bool:
        """A word of `trader`'s order `kernel_order`, before `leg` takes it: True when it
        ends the leg's stop after the last lines, which is then noted."""
        if not self._ended or kernel_order != leg.stop or not update.status.terminal:
            return False
        self._stops[trader.name] = (leg.orders[kernel_order].name(), update.status)
        return True

    def note(self, session: Any, trader: Trader) -> str:
        """The note of `trader`'s position left open: what happened to the leg after the
        last lines, when anything did, how OpenAlgo's books differed from the engine when
        no later read confirms it (`differs_words`), then its state (`left_open`);
        `session` is the `_session_loop.Session`."""
        seen = session.reconcile.unconfirmed(trader.name)
        if seen is None:
            return "; ".join([*self._happened(trader), left_open(session.stops, trader)])
        state = left_open(session.stops, trader, books=False)
        return "; ".join([*self._happened(trader), differs_words(seen), state])

    def alert(self, session: Any) -> None:
        """One ERROR alert per leg whose difference with OpenAlgo's books no later read
        confirms, per leg left open that anything happened to after the last lines or that
        the exit guard still holds, and per leg left flat that anything happened to or that
        still has an order working (module docstring); `session` is the
        `_session_loop.Session`."""
        held = session.reconcile.guard.left()
        for trader in session.traders:
            seen = session.reconcile.unconfirmed(trader.name)
            text = self._alert_text(session.stops, trader, held, seen)
            if text is not None:
                session.alert(AlertKind.ERROR, text)

    def _alert_text(
        self, stops: Any, trader: Trader, held: Sequence[Any], seen: Sighting | None
    ) -> str | None:
        """The ERROR alert of `trader`'s leg, or None when it needs none (`alert`)."""
        happened, not_sent = self._happened(trader), _held_changes(trader, held)
        if seen is not None:
            state = left_open(stops, trader, books=False)
            parts = [*happened, differs_words(seen), *not_sent, state]
            return f"{trader.name} {trader.symbol} {trader.exchange}: {'; '.join(parts)}"
        units = trader.open_units()
        if units:
            differs = [CLOSED_DIFFERING] if stops.broker_units(trader) is not None else []
            if not (happened or differs):
                return None
            parts = [*happened, *differs, *not_sent, left_open(stops, trader)]
            return f"{leg_words(trader, units)}: {'; '.join(parts)}"
        working = still_working(stops, trader)
        if not (happened or working):
            return None
        parts = [*happened, *working, *not_sent, FLAT_NOW]
        return f"{trader.name} {trader.symbol} {trader.exchange}: {'; '.join(parts)}"

    def _happened(self, trader: Trader) -> list[str]:
        """What happened to the leg after the last lines, in a note's words: its units
        changed (but by its stop's own fill, which says it), its stop ended, or both."""
        now = trader.open_units()
        before = self._units.get(trader.name, now)
        stop = self._stops.get(trader.name)
        if stop is None:
            return [] if before == now else [_moved(before, now)]
        name, status = stop
        ended = f"its stop {name} {_ENDED_WORDS[status]} at the broker"
        if before == now or status is OrderStatus.COMPLETE:
            return [f"{ended} {_AFTER_THE_LINES}"]
        return [f"{_moved(before, now)}, and {ended}"]


def differs_words(seen: Sighting) -> str:
    """How a note and an alert name a difference with OpenAlgo's books that no later read
    confirms: both figures, and that nothing was sent for it (`DIFFERS`)."""
    return DIFFERS.format(broker=seen.broker_units, engine=seen.engine_units)


def _moved(before: float, now: float) -> str:
    """A change of a leg's signed units after the last lines, in a note's words."""
    return f"its position went from {before:g} to {now:g} units {_AFTER_THE_LINES}"


def _held_changes(trader: Trader, held: Sequence[Any]) -> list[str]:
    """The modifies and cancels of `trader`'s orders the exit guard held back (`held` is
    every request it still holds), in an alert's words."""
    names = [
        f"the {'cancel' if isinstance(request, CancelRequest) else 'modify'} of "
        f"{request.intent.purpose.value.replace('_', ' ')} order "
        f"{trader.order_name(request.kernel_order)}"
        for request in held
        if request.intent.strategy == trader.name
        and isinstance(request, ModifyRequest | CancelRequest)
    ]
    return [f"not sent: {', '.join(names)}"] if names else []


__all__ = ["CLOSED_DIFFERING", "DIFFERS", "FLAT_NOW", "AfterLines", "differs_words"]
