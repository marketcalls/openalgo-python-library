"""A CNC or NRML leg's stop GTT as the stops supervisor sees it (20-positional-live 2 and
10.4; 16-session-design 4.2; the lead's ruling N3b-2): the leg's stop is a SINGLE GTT the
broker holds, not a stop order, so what covers the leg, whether it ever had a stop, and how
its stop is named in an alert or a line are read from the kernel's GTTs of the position
(`Engine.gtts`) and the strategy's GTT map (`_venue_gtts`, the broker's trigger ID).

- **Covered**: the units of the leg's stop GTTs that rest confirmed (the kernel's
  "Active", and "ModifyPending", whose old terms rest until the modify's answer). One
  whose cancel was sent covers nothing (the broker may have cancelled it already), nor
  one not confirmed yet ("Requested").
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


def overnight(trader: Any) -> bool:
    """Whether `trader` trades CNC or NRML: its leg's stop is a GTT."""
    return trader.product in OVERNIGHT_PRODUCTS


def stop_gtts(trader: Any) -> list[dict[str, Any]]:
    """The kernel's stop GTTs of `trader`'s position, in GTT order (none once its kernel
    session finished)."""
    if trader.finished:
        return []
    gtts = call(trader.engine.gtts, INSTRUMENT, trader.product)
    return [gtt for gtt in gtts if gtt["purpose"] == "stop"]


def gtt_guarded(trader: Any) -> bool:
    """Whether `trader`'s leg was given a stop GTT (module docstring)."""
    return bool(stop_gtts(trader))


def gtt_cover(trader: Any) -> int:
    """The units the leg's confirmed stop GTTs cover at the broker (module docstring)."""
    return sum(round(gtt["quantity_units"]) for gtt in stop_gtts(trader) if gtt["state"] in RESTING)


def gtt_waiting(trader: Any) -> bool:
    """Whether a stop GTT of the leg waits for the answer to its place or modify."""
    return any(gtt["state"] in (REQUESTED, "ModifyPending") for gtt in stop_gtts(trader))


def gtt_cancelling(trader: Any) -> bool:
    """Whether a stop GTT of the leg waits for the answer to its cancel."""
    return any(gtt["state"] == CANCEL_PENDING for gtt in stop_gtts(trader))


def gtt_words(trader: Any) -> str:
    """The leg's stop GTT in an alert's words: confirmed (its trigger ID, trigger and
    units), being cancelled, not confirmed yet, or "" for none."""
    gtts = stop_gtts(trader)
    resting = [gtt for gtt in gtts if gtt["state"] in RESTING]
    if resting:
        gtt = resting[-1]
        name = _name(trader, gtt)
        return (
            f"{name} confirmed at the broker, trigger {gtt['trigger_price']:.2f} for "
            f"{round(gtt['quantity_units'])} units"
        )
    if any(gtt["state"] == CANCEL_PENDING for gtt in gtts):
        return f"{_name(trader, gtts[-1])} being cancelled, with no final word yet"
    if any(gtt["state"] == REQUESTED for gtt in gtts):
        return f"{_name(trader, gtts[-1])} not confirmed at the broker yet"
    return ""


def gtt_trigger_id(trader: Any) -> str:
    """The broker's trigger ID of the leg's latest stop GTT that rests confirmed, or ""."""
    resting = [gtt for gtt in stop_gtts(trader) if gtt["state"] in RESTING]
    return _trigger_id(trader, resting[-1]) if resting else ""


def _trigger_id(trader: Any, gtt: dict[str, Any]) -> str:
    known = None if trader.gtts is None else trader.gtts.gtt(int(gtt["gtt_id"]))
    return "" if known is None else known.trigger_id


def _name(trader: Any, gtt: dict[str, Any]) -> str:
    trigger_id = _trigger_id(trader, gtt)
    return f"stop GTT {trigger_id}" if trigger_id else "stop GTT"


__all__ = [
    "CANCEL_PENDING",
    "REQUESTED",
    "RESTING",
    "gtt_cancelling",
    "gtt_cover",
    "gtt_guarded",
    "gtt_trigger_id",
    "gtt_waiting",
    "gtt_words",
    "overnight",
    "stop_gtts",
]
