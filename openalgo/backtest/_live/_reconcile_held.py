"""What the reconciler's exit guard (`_reconcile_guard`) tells the operator of an order it
holds back on a guarded leg (16-session-design 5.4 "As built"; unit 12): one ERROR alert
per held order, naming the leg, the units OpenAlgo's position book shows and why the order
waits.

Neither a held cancel of the stop nor the exit it makes way for is promised once the books
agree: protect by state keeps the cancel of a stop that still fits and asks for no exit,
and after the session's last lines only what a sale by hand adopted frees goes (the
guard's "A sale by hand adopted" and "At the close").

Quantities are units. Used by the loop only.
"""

from __future__ import annotations

from typing import Any

from ._order_state import ENTRY_PURPOSES
from ._types import Purpose
from ._venue_requests import CancelRequest


def held_words(request: Any, broker_units: int) -> str:
    """The ERROR alert of `request` (a placement, modify or cancel the guard holds) of a
    guarded leg whose broker holds `broker_units` (signed) by OpenAlgo's position book."""
    intent = request.intent
    leg = f"{intent.symbol} {intent.exchange} ({intent.action} {intent.quantity})"
    book = f"OpenAlgo's position book shows {broker_units} units"
    if isinstance(request, CancelRequest):
        return (
            f"{intent.strategy}: the cancel of the stop of {leg} is held back, so the stop "
            f"stays at the broker: {book}, and the exit it makes way for would take the "
            "broker's position through zero. Both wait for the books to agree, and go only "
            "if the session still asks for them then; check the position in OpenAlgo"
        )
    if intent.purpose in ENTRY_PURPOSES:
        return (
            f"{intent.strategy}: the entry of {leg} is held back: {book}, and it would close "
            "more than the broker holds. It is sent once the books agree, unless entries "
            "are locked by then; check the position in OpenAlgo"
        )
    what = "stop" if intent.purpose is Purpose.STOP else intent.purpose.value.replace("_", " ")
    return (
        f"{intent.strategy}: the {what} of {leg} is held back: {book}, and it would take "
        "the broker's position through zero. It is sent once the books agree; check the "
        "position in OpenAlgo"
    )


__all__ = ["held_words"]
