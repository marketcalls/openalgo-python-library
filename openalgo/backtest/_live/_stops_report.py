"""What the operator is told of a leg and its broker-held stop (16-session-design 1.4
`open_positions`, 3.3 step 4, 3.5, 6.2; the unit 8 hand-off on the left-open note): the
stop's order ID and whether it is confirmed, the stop in an alert's words, and the note
of a position left open (the safe stop's lines, the result's `open_positions` and the
emergency stop's alert).

A position left open names each order still working toward it, by its broker order ID
once bound, else its side, units, purpose and intent ID: an exit or an entry sent and
unanswered, held back or ambiguous, or a MARKET order accepted with no final word yet, is
"in flight"; any other one bound is "resting at the broker"; one the kernel made and
never sent is "not sent". Then the stop: confirmed (for part of the position, or all of
it), repriced to a LIMIT resting at the broker, being cancelled (a cancel sent, or one the
kernel decided and the session has not handed over yet), ambiguous, handed to the order
worker with no answer, held back and not sent, sent and unanswered, not confirmed, or
none. Then what to do, since a sale by hand beside the session's own orders would sell
short:

- no stop and nothing sent working (an order the kernel made and never sent is no
  order at the broker): close it by hand;
- a confirmed stop covering the whole position and nothing else working: the stop
  protects it, and must be cancelled before any sale by hand (`protected`);
- anything else (an order working, or a stop that may be at the broker without covering
  the position): check the order book before closing it by hand, cancelling what still
  works.

The notes are taken once the loop step's commands are handed over (`_session_stop`): an
order not sent then was not handed to the order worker (after a guard lock, or an exit
the kernel holds until a cancel it waits for ends, which the note names too). Used by
the loop only, over the stops supervisor (`stops.StopsSupervisor`).
"""

from __future__ import annotations

from typing import Any

from ._order_state import ENTRY_PURPOSES
from ._session_trader import Trader
from ._stops_kernel import cancelling, order_of
from ._stops_orders import IN_FLIGHT, Sent, Watched
from ._types import Purpose

#: How an alert says a leg has no stop.
NO_STOP = "no stop at the broker"
#: What a note of a position with an order still working tells the operator to do.
CHECK_FIRST = "check the order book before closing it by hand, cancelling what still works"
#: What a note of a position its confirmed stop covers whole, with nothing else working,
#: tells the operator: a sale by hand beside the stop would let the stop sell again.
STOP_PROTECTS = "it protects the position, so cancel the stop before any sale by hand"
#: How far an order in flight got, in words.
_IN_FLIGHT_WORDS = {
    Sent.HANDED: "not answered yet",
    Sent.HELD: "held back, not sent",
    Sent.UNANSWERED: "sent and unanswered",
    Sent.AMBIGUOUS: "ambiguous in the order book",
}
#: How far a stop not answered got, in words: one handed to the order worker may wait in
#: its queue (dropped at its close) or be in its call.
_STOP_UNANSWERED_WORDS = {
    Sent.HANDED: "handed to the order worker, not answered: it may not have been sent",
    Sent.HELD: "held back, not sent",
    Sent.UNANSWERED: "sent and unanswered",
}
#: The words of a MARKET order accepted with no final word yet.
_ACCEPTED_WORDS = "accepted with no final word yet"


def stop_state(stops: Any, trader: Trader) -> tuple[str, bool]:
    """The stop of `trader`'s leg as the supervisor `stops` knows it: its broker order ID
    ("" for none, or one not bound) and whether it is confirmed working at the broker."""
    watched = _stop(stops, trader)
    return ("", False) if watched is None else (watched.order_id, _covering(stops, trader) > 0)


def stop_words(stops: Any, trader: Trader) -> str:
    """The stop of `trader`'s leg in an alert's words: confirmed, repriced, being
    cancelled or ambiguous (these two with the order book to check), not sent, not
    confirmed, or none."""
    words = _stop_words(stops, trader)
    watched = _stop(stops, trader)
    unsure = watched is not None and (
        watched.state is Sent.AMBIGUOUS or _being_cancelled(stops, trader, watched)
    )
    return f"{words}: check the order book" if unsure else words


def protected(stops: Any, trader: Trader) -> bool:
    """True when the leg's confirmed stop covers its whole position and no other order
    works toward it: nothing for the operator to do but leave the stop or cancel it
    before a sale by hand."""
    units = round(abs(trader.open_units()))
    return 0 < units <= _covering(stops, trader) and not _working(stops, trader)


def left_open(stops: Any, trader: Trader) -> str:
    """The note of `trader`'s position left open (module docstring)."""
    leg, working = stops.leg(trader), trader.working()
    sent = _working(stops, trader)
    unsent = [kernel_order for kernel_order in working if kernel_order not in leg.orders]
    parts = [*sent]
    if unsent:
        parts.append(f"order {', '.join(f'o{k}' for k in unsent)} not sent")
    stop_part = _stop_words(stops, trader)
    if stop_part == NO_STOP and not sent:
        # Nothing sent works toward the position: only a sale by hand closes it.
        return "; ".join([*parts, f"{NO_STOP}, close it by hand"])
    if protected(stops, trader):
        return "; ".join([*parts, f"{stop_part}: {STOP_PROTECTS}"])
    if stop_part != NO_STOP or leg.guarded:
        # With an order working on a leg that never had a stop, that order is what the
        # operator checks, not a stop it never was to have.
        parts.append(stop_part)
    parts.append(CHECK_FIRST)
    return "; ".join(parts)


def _working(stops: Any, trader: Trader) -> list[str]:
    """Each order other than the stop sent and still working toward the position, in a
    note's words."""
    leg, working = stops.leg(trader), set(trader.working())
    parts = []
    for watched in leg.orders.values():
        stop = watched.purpose is Purpose.STOP
        if stop or watched.state is Sent.ENDED or watched.kernel_order not in working:
            continue
        kind = "entry" if watched.purpose in ENTRY_PURPOSES else "exit"
        parts.append(f"{kind} order {watched.name()} {_how_far(trader, watched)}")
    return parts


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
    """The stop of the leg in words, with no advice (module docstring)."""
    watched = _stop(stops, trader)
    if watched is None:
        return NO_STOP
    if watched.state is Sent.AMBIGUOUS:
        return f"ambiguous stop order (intent {watched.intent_id})"
    name = watched.order_id or f"(intent {watched.intent_id})"
    if _being_cancelled(stops, trader, watched):
        return f"stop {name} being cancelled, with no final word yet"
    covered, units = stops.covered(trader), round(abs(trader.open_units()))
    if covered and covered >= units:
        return f"stop {name} confirmed at the broker"
    if covered:
        return f"stop {name} confirmed at the broker for {covered} of {units} units"
    if watched.state in _STOP_UNANSWERED_WORDS:
        return f"stop {name} {_STOP_UNANSWERED_WORDS[watched.state]}"
    limit = _repriced_limit(trader, watched)
    if limit is not None:
        return f"stop {name} repriced to a LIMIT at {limit:.2f}, resting at the broker"
    return f"stop {name} not confirmed at the broker"


def _repriced_limit(trader: Trader, watched: Watched) -> float | None:
    """The limit in rupees of a leg stop the price rule repriced to a LIMIT that the
    broker accepted and that still works, else None (a leg stop is SL or SL-M until a
    reprice is applied)."""
    if watched.state not in (Sent.BOUND, Sent.CONFIRMED):
        return None
    if watched.kernel_order not in trader.working():
        return None
    order = order_of(trader, watched.kernel_order)
    if order is None or order["order_type"] != "LIMIT":
        return None
    return float(order["limit_rupees"])


def _being_cancelled(stops: Any, trader: Trader, watched: Watched) -> bool:
    """Whether a cancel of the leg's stop `watched` was sent, or the kernel decided one the
    session has not handed over yet (its cancelling list): either may be in effect at the
    broker by the time the operator reads the note. Only a stop that covers units is
    looked up in the kernel: one not confirmed covers nothing either way."""
    if watched.cancel_sent:
        return True
    return stops.covered(trader) > 0 and watched.kernel_order in cancelling(trader)


def _covering(stops: Any, trader: Trader) -> int:
    """The units the leg's confirmed stop covers at the broker for a note: none while it
    is being cancelled (`_being_cancelled`)."""
    watched = _stop(stops, trader)
    if watched is None or _being_cancelled(stops, trader, watched):
        return 0
    return int(stops.covered(trader))


def _stop(stops: Any, trader: Trader) -> Watched | None:
    leg = stops.leg(trader)
    return leg.orders.get(leg.stop) if leg.stop is not None else None


__all__ = [
    "CHECK_FIRST",
    "NO_STOP",
    "STOP_PROTECTS",
    "left_open",
    "protected",
    "stop_state",
    "stop_words",
]
