"""A CNC or NRML leg's stop GTT as the stops supervisor sees it (20-positional-live 2 and
10.4; 16-session-design 4.2; the lead's ruling N3b-2): the leg's stop is a SINGLE GTT the
broker holds, not a stop order, so what covers the leg, whether it ever had a stop, and how
its stop is named in an alert or a line are read from the kernel's GTTs of the position
(`Engine.gtts`) and the strategy's GTT map (`_venue_gtts`, the broker's trigger ID).

- **Covered**: the units of the leg's stop GTTs that rest confirmed (the kernel's
  "Active", and "ModifyPending", whose old terms rest until the modify's answer). One
  whose cancel was handed to the order worker covers nothing (the broker may have
  cancelled it already; `_venue_gtts.KernelGtt.cancel_sent`), nor one not confirmed yet
  ("Requested"), nor one the close's read of the GTT book did not confirm after the
  kernel's session closed (`KernelGtt.unconfirmed`, `_session_carry`: the kernel still
  holds it as resting); one whose cancel the kernel decided and the session never handed
  over (after a guard lock) still rests.
- **Guarded**: the leg was given a stop GTT, resting, ended or fired.
- **Waiting**: a stop GTT's place or modify waits for its answer; **cancelling**: a stop
  GTT's cancel does.

Quantities are whole units, prices rupees. Used by the loop only.
"""

from __future__ import annotations

from typing import Any

from .._engine import call
from .._products import OVERNIGHT_PRODUCTS
from ._plan_catch_up import INSTRUMENT

#: The kernel's states (the binding's `GTT_STATES`) of a stop GTT that rests confirmed,
#: one not confirmed yet, and one whose cancel waits for its answer.
RESTING = frozenset({"Active", "ModifyPending"})
REQUESTED = "Requested"
CANCEL_PENDING = "CancelPending"
#: What a carried leg whose stop GTT the close did not confirm asks of the operator.
PLACE_GTT_BY_HAND = "place its stop GTT in OpenAlgo by hand before the next open"
#: The same in a line of a leg left open, which the operator reads once the session ended:
#: a stop GTT gone from the book may have fired, so the books come first.
CHECK_THEN_PLACE = f"check the order book and the position, then {PLACE_GTT_BY_HAND}"


def overnight(trader: Any) -> bool:
    """Whether `trader` trades CNC or NRML: its leg's stop is a GTT."""
    return trader.product in OVERNIGHT_PRODUCTS


def stop_gtts(trader: Any) -> list[dict[str, Any]]:
    """The kernel's stop GTTs of `trader`'s position, in GTT order (its kernel session
    answers the query after its close too)."""
    gtts = call(trader.engine.gtts, INSTRUMENT, trader.product)
    return [gtt for gtt in gtts if gtt["purpose"] == "stop"]


def gtt_guarded(trader: Any) -> bool:
    """Whether `trader`'s leg was given a stop GTT (module docstring)."""
    return bool(stop_gtts(trader))


def gtt_cover(trader: Any) -> int:
    """The units the leg's confirmed stop GTTs cover at the broker (module docstring)."""
    return sum(round(gtt["quantity_units"]) for gtt in resting_gtts(trader))


def resting_gtts(trader: Any) -> list[dict[str, Any]]:
    """The leg's stop GTTs that rest confirmed at the broker (module docstring)."""
    return [gtt for gtt in stop_gtts(trader) if _rests(trader, gtt)]


def _rests(trader: Any, gtt: dict[str, Any]) -> bool:
    if _unconfirmed(trader, gtt):
        return False
    if gtt["state"] in RESTING:
        return True
    return gtt["state"] == CANCEL_PENDING and not _cancel_sent(trader, gtt)


def _unconfirmed(trader: Any, gtt: dict[str, Any]) -> str:
    """Why the close's read did not confirm `gtt`, or "" (module docstring)."""
    known = None if trader.gtts is None else trader.gtts.gtt(int(gtt["gtt_id"]))
    return "" if known is None else known.unconfirmed


def _cancel_sent(trader: Any, gtt: dict[str, Any]) -> bool:
    known = None if trader.gtts is None else trader.gtts.gtt(int(gtt["gtt_id"]))
    return known is not None and known.cancel_sent


def gtt_waiting(trader: Any) -> bool:
    """Whether a stop GTT of the leg waits for the answer to its place or modify."""
    return any(gtt["state"] in (REQUESTED, "ModifyPending") for gtt in stop_gtts(trader))


def gtt_cancelling(trader: Any) -> bool:
    """Whether a stop GTT of the leg waits for the answer to a cancel handed over."""
    return any(
        gtt["state"] == CANCEL_PENDING and _cancel_sent(trader, gtt) for gtt in stop_gtts(trader)
    )


def gtt_words(trader: Any) -> str:
    """The leg's stop GTT in an alert's words: confirmed (its trigger ID, trigger and
    units), being cancelled, not confirmed yet, or "" for none."""
    gtts = stop_gtts(trader)
    resting = resting_gtts(trader)
    if resting:
        gtt = resting[-1]
        name = _name(trader, gtt)
        return (
            f"{name} confirmed at the broker, trigger {gtt['trigger_price']:.2f} for "
            f"{round(gtt['quantity_units'])} units"
        )
    if gtt_cancelling(trader):
        return f"{_name(trader, gtts[-1])} being cancelled, with no final word yet"
    unconfirmed = [gtt for gtt in gtts if _unconfirmed(trader, gtt)]
    if unconfirmed:
        gtt = unconfirmed[-1]
        return (
            f"{_name(trader, gtt)} not confirmed at the broker after the close "
            f"({_unconfirmed(trader, gtt)})"
        )
    if any(gtt["state"] == REQUESTED for gtt in gtts):
        return f"{_name(trader, gtts[-1])} not confirmed at the broker yet"
    return ""


def gtt_unconfirmed(trader: Any) -> bool:
    """Whether the leg has a stop GTT the close's read did not confirm and none resting
    confirmed: its line asks the operator to check the books, then place it by hand."""
    gtts = stop_gtts(trader)
    return not resting_gtts(trader) and any(_unconfirmed(trader, gtt) for gtt in gtts)


def gtt_trigger_id(trader: Any) -> str:
    """The broker's trigger ID of the leg's latest stop GTT that rests confirmed, else of
    its latest one the close's read did not confirm, or ""."""
    named = resting_gtts(trader) or [g for g in stop_gtts(trader) if _unconfirmed(trader, g)]
    return _trigger_id(trader, named[-1]) if named else ""


def _trigger_id(trader: Any, gtt: dict[str, Any]) -> str:
    known = None if trader.gtts is None else trader.gtts.gtt(int(gtt["gtt_id"]))
    return "" if known is None else known.trigger_id


def _name(trader: Any, gtt: dict[str, Any]) -> str:
    trigger_id = _trigger_id(trader, gtt)
    return f"stop GTT {trigger_id}" if trigger_id else "stop GTT"


__all__ = [
    "CANCEL_PENDING",
    "CHECK_THEN_PLACE",
    "PLACE_GTT_BY_HAND",
    "REQUESTED",
    "RESTING",
    "gtt_cancelling",
    "gtt_cover",
    "gtt_guarded",
    "gtt_trigger_id",
    "gtt_unconfirmed",
    "gtt_waiting",
    "gtt_words",
    "overnight",
    "resting_gtts",
    "stop_gtts",
]
