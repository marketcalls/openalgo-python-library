"""What the operator is told of a leg and its broker-held stop (16-session-design 1.4
`open_positions`, 3.3 step 4, 3.5, 6.2; the unit 8 hand-off on the left-open note): how a
line names the leg, the stop's order ID and whether it is confirmed, the stop in an
alert's words, and the note of a position left open (the safe stop's lines, the result's
`open_positions` and the emergency stop's alert), and what still works on a leg left flat
(`still_working`, the close's alert, `_stops_after`).

A position left open names each order still working toward it, by its broker order ID
once bound, else its side, units, purpose and intent ID: an exit or an entry sent and
unanswered, held back or ambiguous, or a MARKET order accepted with no final word yet, is
"in flight"; any other one bound is "resting at the broker"; one the kernel made and
never sent is "not sent". Then the stop: confirmed (for part of the position, all of it,
or more units than it holds), repriced to a LIMIT resting at the broker, being cancelled
(a cancel sent, or one the kernel decided and the session has not handed over yet),
ambiguous, handed to the order worker with no answer, held back and not sent, sent and
unanswered, not confirmed, or none. A stop's units are the terms the broker confirmed
(`StopsSupervisor.covered`), never the kernel's while a modify or cancel of it is held
back or unanswered (3.3 step 5 "As built"). Then what to do, since a sale by hand beside
the session's own orders would sell short:

- no stop and nothing sent working (an order the kernel made and never sent is no
  order at the broker): close it by hand;
- a confirmed stop for exactly the position's units and nothing else working: the stop
  protects it, and must be cancelled before any sale by hand (`protected`). A stop
  larger than the position (a sale by hand adopted before its cut or cancel went) never
  protects it: a trigger would sell beyond it;
- anything else (an order working, or a stop that may be at the broker without covering
  the position): check the order book before closing it by hand, cancelling what still
  works.

While the reconciler's exit guard holds the leg (its last read showed the broker holding
fewer units on the engine's side, none, or the other side), and after the close's last
lines for a difference no later read will confirm (`StopsSupervisor.broker_units`), the
note names the units OpenAlgo's books showed (unless the caller names them itself) and a
confirmed stop's own units, and sends the operator to the order book: the broker's
position is not the engine's, so a stop sized for the engine's units may sell beyond it,
and no stop is said to protect the leg or reported confirmed (`stop_state`).

That advice holds once the session has ended. While it runs, the engine sends a rejected
exit again at a later bar's open and places a stop again that ended at the broker with no
request of its own (16-session-design 4.8, 4.10), so a sale by hand beside it sells
twice: an alert sent while the session runs (the give-up of an exit, a safe stop's
reminder) says to end the session first, then what the leg asks for (`while_running`,
`open_while_running`).

The notes are taken once the loop step's commands are handed over (`_session_stop`): an
order not sent then was not handed to the order worker (after a guard lock, or an exit
the kernel holds until a cancel it waits for ends, which the note names too). Used by
the loop only, over the stops supervisor (`stops.StopsSupervisor`).
"""

from __future__ import annotations

from typing import Any

from ._order_state import ENTRY_PURPOSES
from ._session_trader import Trader
from ._stops_cover import CLOSE_REJECTED, REFUSED_FOR_MARGIN
from ._stops_kernel import PROTECT_PLACED, order_of
from ._stops_orders import IN_FLIGHT, Sent, Watched
from ._types import Purpose

#: How an alert says a leg has no stop.
NO_STOP = "no stop at the broker"
#: What a note of a position with an order still working tells the operator to do.
CHECK_FIRST = "check the order book before closing it by hand, cancelling what still works"
#: What a note of a position its confirmed stop covers whole, with nothing else working,
#: tells the operator: a sale by hand beside the stop would let the stop sell again.
STOP_PROTECTS = "it protects the position, so cancel the stop before any sale by hand"
#: How an alert sent while the session runs tells the operator to end it before any close
#: by hand (module docstring): a first Ctrl+C with on_stop="flatten" may wait for its
#: exits, and a second one ends the session.
END_SESSION_FIRST = "end the session first (Ctrl+C, and a second Ctrl+C if it does not end)"
#: When the step after the end applies: the default on_stop="flatten" may close the leg
#: as the session ends, and the lines printed then say what is still open.
IF_STILL_OPEN = "if its closing lines still show the leg open"
#: Why, as such an alert says it.
WHILE_IT_RUNS = "while the session runs, the engine may send its exit or its stop again"
#: What a leg whose stop works asks of the operator once the session has ended.
CANCEL_THAT_STOP = "cancel that stop"
#: What a leg with an order working, or a stop that may be at the broker, asks of the
#: operator once the session has ended.
CHECK_THE_BOOK = "check the order book, cancelling what still works"
#: What a note advises by the leg's state (module docstring): close it by hand, its stop
#: protects it, or check the order book first.
_CLOSE, _PROTECTS, _CHECK = "close", "protects", "check"
#: What each advice asks of the operator once the session has ended, for an alert sent
#: while it runs (`open_while_running`).
_AFTER_THE_END = {_CLOSE: "", _PROTECTS: CANCEL_THAT_STOP, _CHECK: CHECK_THE_BOOK}
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


def leg_words(trader: Trader, units: float) -> str:
    """How a line names a strategy's leg: the strategy, the instrument, signed units."""
    return f"{trader.name} {trader.symbol} {trader.exchange} {units:g} units"


def stop_state(stops: Any, trader: Trader) -> tuple[str, bool]:
    """The stop of `trader`'s leg as the supervisor `stops` knows it: its broker order ID
    ("" for none, or one not bound) and whether it is confirmed working at the broker for
    no more units than the position holds (all of them or part), never while the exit
    guard holds the leg (module docstring)."""
    watched = _stop(stops, trader)
    if watched is None:
        return "", False
    covering = _covering(stops, trader)
    guarded = stops.broker_units(trader) is not None
    return watched.order_id, not guarded and 0 < covering <= round(abs(trader.open_units()))


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
    """True when the leg's confirmed stop is for exactly its position's units, no other
    order works toward it and the exit guard does not hold the leg (module docstring):
    nothing for the operator to do but leave the stop or cancel it before a sale by
    hand."""
    if stops.broker_units(trader) is not None:
        return False
    units = round(abs(trader.open_units()))
    return 0 < units == _covering(stops, trader) and not _working(stops, trader)


def left_open(stops: Any, trader: Trader, *, books: bool = True) -> str:
    """The note of `trader`'s position left open once the session has ended (module
    docstring); with `books` False it leaves out the units OpenAlgo's books showed, for a
    caller whose words name them."""
    parts, advice = _note(stops, trader, books=books)
    if advice == _CHECK:
        return "; ".join([*parts, CHECK_FIRST])
    *before, last = parts
    words = f"{last}, close it by hand" if advice == _CLOSE else f"{last}: {STOP_PROTECTS}"
    return "; ".join([*before, words])


def still_working(stops: Any, trader: Trader) -> list[str]:
    """Each order of `trader`'s leg sent and still working in the engine, its stop among
    them, in a note's words: what may still work at the broker, for a leg left flat."""
    parts = _working(stops, trader)
    watched = _stop(stops, trader)
    if (
        watched is not None
        and watched.state is not Sent.ENDED
        and watched.kernel_order in trader.working()
    ):
        parts.append(_stop_words(stops, trader))
    return parts


def open_while_running(stops: Any, trader: Trader) -> str:
    """The note of `trader`'s open position for an alert sent while the session runs:
    what works toward it and its stop, as `left_open` names them, then `while_running`
    with what the leg asks for once the session has ended (module docstring)."""
    parts, advice = _note(stops, trader)
    return "; ".join([*parts, while_running(_AFTER_THE_END[advice])])


#: What the broker holds for a leg after `protect`, by its outcome (the kernel's
#: `PROTECT_OUTCOMES`), with what the leg asks of the operator once the session has ended
#: (`while_running`): a stop working is cancelled before a sale by hand, and an exit
#: waiting or working sends the operator to the order book first.
_PROTECT_WORDS = {
    **dict.fromkeys(
        PROTECT_PLACED,
        ("its stop is placed again at the broker and protects the position", CANCEL_THAT_STOP),
    ),
    "stop_works": ("its stop works at the broker and protects the position", CANCEL_THAT_STOP),
    "exiting": ("an exit of it waits or works", CHECK_THE_BOOK),
}


def protect_words(outcome: str | None) -> tuple[str, str]:
    """What an alert says the broker holds for a leg after `protect` answered `outcome`,
    and what the leg asks of the operator once the session has ended (for
    `while_running`). Any other outcome (no stop level, the market closed, a failed call
    given as None) leaves no stop at the broker and only the sale by hand."""
    return _PROTECT_WORDS.get(outcome or "", (NO_STOP, ""))


def while_running(then: str = "") -> str:
    """What an alert sent while the session runs tells an operator who would close a leg
    by hand: end the session first, then `then` (what the leg asks for once the session
    has ended, `CANCEL_THAT_STOP` or `CHECK_THE_BOOK`; nothing when only the sale by hand
    is left) if the lines printed as it ends still show the leg open, since the default
    flatten may close it on the way out, and why (module docstring)."""
    step = f", then, {IF_STILL_OPEN}, {then}" if then else ""
    return f"to close it by hand, {END_SESSION_FIRST}{step}: {WHILE_IT_RUNS}"


def _note(stops: Any, trader: Trader, *, books: bool = True) -> tuple[list[str], str]:
    """What works toward `trader`'s position and its stop, in a note's words, and the
    advice the leg's state asks for (`_CLOSE`, `_PROTECTS` or `_CHECK`); with `_CLOSE`
    and `_PROTECTS` the last part is the stop's. `books`: whether it names the units
    OpenAlgo's books showed when they differ from the engine's (`left_open`)."""
    leg, working = stops.leg(trader), trader.working()
    sent = _working(stops, trader)
    unsent = [kernel_order for kernel_order in working if kernel_order not in leg.orders]
    parts = [*_held_words(stops, trader), *sent]
    if unsent:
        parts.append(f"order {', '.join(f'o{k}' for k in unsent)} not sent")
    stop_part = _stop_words(stops, trader)
    broker = stops.broker_units(trader)
    if broker is not None:
        # The broker holds other units than the engine counts, or the other side: neither
        # a sale by hand of the engine's units nor its stop fits what the broker holds.
        shown = [f"OpenAlgo's books showed {broker} units at the last read"] if books else []
        return [*parts, stop_part, *shown], _CHECK
    if stop_part == NO_STOP and not sent:
        # Nothing sent works toward the position: only a sale by hand closes it.
        return [*parts, NO_STOP], _CLOSE
    if protected(stops, trader):
        return [*parts, stop_part], _PROTECTS
    if stop_part != NO_STOP or leg.guarded:
        # With an order working on a leg that never had a stop, that order is what the
        # operator checks, not a stop it never was to have.
        parts.append(stop_part)
    return parts, _CHECK


def _held_words(stops: Any, trader: Trader) -> list[str]:
    """The leg's unprotected reason when a closing order's rejection on a shared
    instrument holds it (18-multi-strategy 4.14: "listed in open_positions"), as the
    first part of its note; none otherwise."""
    window = stops.windows.current(trader.name)
    if window is None or window.reason not in (REFUSED_FOR_MARGIN, CLOSE_REJECTED):
        return []
    return [f"{window.reason}: the broker may not hold these units"]


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
    if covered and stops.broker_units(trader) is not None:
        # The broker's position is not the engine's (the note names it): the stop's own
        # units are what the operator weighs against it.
        return f"stop {name} confirmed at the broker for {covered} units"
    if covered and units in (0, covered):
        # A flat leg's stop working on is named so: its alert sends the operator to the
        # order book to cancel what still works.
        return f"stop {name} confirmed at the broker"
    if covered > units:
        # A sale by hand adopted before the stop's cut or cancel went: never a protection.
        more = f"for {covered} units, more than the {units} held"
        return f"stop {name} confirmed at the broker {more}"
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
    session has not handed over yet (its cancelling list, but a cancel the exit guard holds
    back): either may be in effect at the broker by the time the operator reads the note.
    Only a stop that covers units is looked up in the kernel: one not confirmed covers
    nothing either way."""
    if watched.cancel_sent:
        return True
    return stops.covered(trader) > 0 and watched.kernel_order in stops.cancelling(trader)


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
    "CANCEL_THAT_STOP",
    "CHECK_FIRST",
    "CHECK_THE_BOOK",
    "END_SESSION_FIRST",
    "IF_STILL_OPEN",
    "NO_STOP",
    "STOP_PROTECTS",
    "WHILE_IT_RUNS",
    "leg_words",
    "left_open",
    "open_while_running",
    "protect_words",
    "protected",
    "still_working",
    "stop_state",
    "stop_words",
    "while_running",
]
