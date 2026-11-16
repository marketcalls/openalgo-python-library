"""What the operator is told of a leg and its broker-held stop (16-session-design 1.4
`open_positions`, 3.3 step 4, 3.5, 6.2; the unit 8 hand-off on the left-open note): the
stop's order ID and whether it is confirmed, the stop in an alert's words, and the note
of a position left open.

A position left open names each order still working toward it, by its broker order ID
once bound, else its side, units, purpose and intent ID: an exit or an entry sent and
unanswered, held back or ambiguous is "in flight" (it may still fill, so the order book
comes before closing by hand), one bound is "resting at the broker", one the kernel made
and never sent is "not sent". Then the stop: confirmed, not confirmed, ambiguous, or none.

Used by the loop only, over the stops supervisor (`stops.StopsSupervisor`).
"""

from __future__ import annotations

from typing import Any

from ._order_state import ENTRY_PURPOSES
from ._session_trader import Trader
from ._stops_orders import IN_FLIGHT, Sent, Watched
from ._types import Purpose

#: How an alert says a leg has no stop.
NO_STOP = "no stop at the broker"
#: How far an order in flight got, in words.
_IN_FLIGHT_WORDS = {
    Sent.HANDED: "not answered yet",
    Sent.HELD: "held back, not sent",
    Sent.UNANSWERED: "sent and unanswered",
    Sent.AMBIGUOUS: "ambiguous in the order book",
}


def stop_state(stops: Any, trader: Trader) -> tuple[str, bool]:
    """The stop of `trader`'s leg as the supervisor `stops` knows it: its broker order ID
    ("" for none, or one not bound) and whether it is confirmed working at the broker."""
    watched = _stop(stops, trader)
    return ("", False) if watched is None else (watched.order_id, stops.covered(trader) > 0)


def stop_words(stops: Any, trader: Trader) -> str:
    """The stop of `trader`'s leg in an alert's words: confirmed, not confirmed, ambiguous,
    or none."""
    watched = _stop(stops, trader)
    if watched is None:
        return NO_STOP
    if watched.state is Sent.AMBIGUOUS:
        return f"ambiguous stop order (intent {watched.intent_id}): close it by hand"
    name = watched.order_id or f"(intent {watched.intent_id})"
    if stops.covered(trader) > 0:
        return f"stop {name} confirmed at the broker"
    return f"stop {name} not confirmed at the broker"


def left_open(stops: Any, trader: Trader) -> str:
    """The note of `trader`'s position left open (module docstring)."""
    leg, working = stops.leg(trader), set(trader.working())
    parts, in_flight = [], False
    for watched in leg.orders.values():
        stop = watched.purpose is Purpose.STOP
        if stop or watched.state is Sent.ENDED or watched.kernel_order not in working:
            continue
        flying = watched.state in IN_FLIGHT
        in_flight = in_flight or flying
        kind = "entry" if watched.purpose in ENTRY_PURPOSES else "exit"
        words = (
            f"in flight, {_IN_FLIGHT_WORDS[watched.state]}" if flying else "resting at the broker"
        )
        parts.append(f"{kind} order {watched.name()} {words}")
    unsent = [kernel_order for kernel_order in working if kernel_order not in leg.orders]
    if unsent:
        parts.append(f"order {', '.join(f'o{k}' for k in unsent)} not sent")
    stop_part = stop_words(stops, trader)
    if in_flight:
        # A leg that never had a stop: the order in flight is what the operator checks.
        advice = "check the order book before closing it by hand"
        parts.append(f"{stop_part}; {advice}" if leg.guarded else advice)
    elif stop_part == NO_STOP:
        parts.append(f"{NO_STOP}, close it by hand")
    else:
        parts.append(stop_part)
    return "; ".join(parts)


def _stop(stops: Any, trader: Trader) -> Watched | None:
    leg = stops.leg(trader)
    return leg.orders.get(leg.stop) if leg.stop is not None else None


__all__ = ["NO_STOP", "left_open", "stop_state", "stop_words"]
