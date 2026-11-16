"""What the operator is told of a leg and its broker-held stop (16-session-design 1.4
`open_positions`, 3.3 step 4, 3.5, 6.2; the unit 8 hand-off on the left-open note): the
stop's order ID and whether it is confirmed, the stop in an alert's words, and the note
of a position left open.

A position left open names each order still working toward it, by its broker order ID
once bound, else its side, units, purpose and intent ID: an exit or an entry sent and
unanswered, held back or ambiguous, or a MARKET order accepted with no final word yet, is
"in flight"; any other one bound is "resting at the broker"; one the kernel made and
never sent is "not sent". Then the stop: confirmed, not confirmed, being cancelled,
ambiguous, or none. While any order works (or the stop's cancel has no answer) the note
ends by saying to check the order book, cancelling what still works, before closing the
position by hand: a sale by hand beside the session's own would sell short. Only a leg
with nothing working and no stop is told plainly to close it by hand.

Used by the loop only, over the stops supervisor (`stops.StopsSupervisor`).
"""

from __future__ import annotations

from typing import Any

from ._order_state import ENTRY_PURPOSES
from ._session_trader import Trader
from ._stops_kernel import order_of
from ._stops_orders import IN_FLIGHT, Sent, Watched
from ._types import Purpose

#: How an alert says a leg has no stop.
NO_STOP = "no stop at the broker"
#: What a note of a position with an order still working tells the operator to do.
CHECK_FIRST = "check the order book before closing it by hand, cancelling what still works"
#: How far an order in flight got, in words.
_IN_FLIGHT_WORDS = {
    Sent.HANDED: "not answered yet",
    Sent.HELD: "held back, not sent",
    Sent.UNANSWERED: "sent and unanswered",
    Sent.AMBIGUOUS: "ambiguous in the order book",
}
#: The words of a MARKET order accepted with no final word yet.
_ACCEPTED_WORDS = "accepted with no final word yet"


def stop_state(stops: Any, trader: Trader) -> tuple[str, bool]:
    """The stop of `trader`'s leg as the supervisor `stops` knows it: its broker order ID
    ("" for none, or one not bound) and whether it is confirmed working at the broker."""
    watched = _stop(stops, trader)
    return ("", False) if watched is None else (watched.order_id, stops.covered(trader) > 0)


def stop_words(stops: Any, trader: Trader) -> str:
    """The stop of `trader`'s leg in an alert's words: confirmed, not confirmed, being
    cancelled (with the order book to check), ambiguous, or none."""
    words = _stop_words(stops, trader)
    return f"{words}: check the order book" if _cancelling(stops, trader) else words


def left_open(stops: Any, trader: Trader) -> str:
    """The note of `trader`'s position left open (module docstring)."""
    leg, working = stops.leg(trader), set(trader.working())
    parts = []
    for watched in leg.orders.values():
        stop = watched.purpose is Purpose.STOP
        if stop or watched.state is Sent.ENDED or watched.kernel_order not in working:
            continue
        kind = "entry" if watched.purpose in ENTRY_PURPOSES else "exit"
        parts.append(f"{kind} order {watched.name()} {_how_far(trader, watched)}")
    works = bool(parts)
    unsent = [kernel_order for kernel_order in working if kernel_order not in leg.orders]
    if unsent:
        parts.append(f"order {', '.join(f'o{k}' for k in unsent)} not sent")
    stop_part = _stop_words(stops, trader)
    works = works or _cancelling(stops, trader)
    if not works:
        parts.append(f"{NO_STOP}, close it by hand" if stop_part == NO_STOP else stop_part)
        return "; ".join(parts)
    if stop_part != NO_STOP or leg.guarded:
        # With an order working on a leg that never had a stop, that order is what the
        # operator checks, not a stop it never was to have.
        parts.append(stop_part)
    parts.append(CHECK_FIRST)
    return "; ".join(parts)


def _how_far(trader: Trader, watched: Watched) -> str:
    """How far a working order got, in a note's words: in flight (sent and unanswered,
    held back, ambiguous, or a MARKET order accepted with no final word), or resting."""
    if watched.state in IN_FLIGHT:
        return f"in flight, {_IN_FLIGHT_WORDS[watched.state]}"
    order = order_of(trader, watched.kernel_order)
    if order is not None and order["order_type"] == "MARKET":
        return f"in flight, {_ACCEPTED_WORDS}"
    return "resting at the broker"


def _stop_words(stops: Any, trader: Trader) -> str:
    """The stop of the leg in words, with no advice (`stop_words`)."""
    watched = _stop(stops, trader)
    if watched is None:
        return NO_STOP
    if watched.state is Sent.AMBIGUOUS:
        return f"ambiguous stop order (intent {watched.intent_id}): close it by hand"
    name = watched.order_id or f"(intent {watched.intent_id})"
    if watched.cancel_sent:
        return f"stop {name} being cancelled, with no final word yet"
    if stops.covered(trader) > 0:
        return f"stop {name} confirmed at the broker"
    return f"stop {name} not confirmed at the broker"


def _cancelling(stops: Any, trader: Trader) -> bool:
    """Whether a cancel of the leg's stop was sent and has no final word yet: the stop
    may still be at the broker, or gone."""
    watched = _stop(stops, trader)
    return watched is not None and watched.cancel_sent


def _stop(stops: Any, trader: Trader) -> Watched | None:
    leg = stops.leg(trader)
    return leg.orders.get(leg.stop) if leg.stop is not None else None


__all__ = ["CHECK_FIRST", "NO_STOP", "left_open", "stop_state", "stop_words"]
