"""What the reconciler's exit guard (`_reconcile_guard`) tells the operator of an order it
holds back on a guarded leg (16-session-design 5.4 "As built"; unit 12): one ERROR alert
per held order, naming the leg, the units OpenAlgo's position book shows and why the order
waits. On an instrument several strategies of the session share, the book's units are the
net of all of them, and every closing order of each waits while the net differs from
their sum (18-multi-strategy 4.8, unit M6).

Neither a held cancel of the stop nor the exit it makes way for is promised once the books
agree: protect by state keeps the cancel of a stop that still fits and asks for no exit,
and after the session's last lines only what a sale by hand adopted frees goes (the
guard's "A sale by hand adopted" and "At the close").

Quantities are units. Used by the loop only.
"""

from __future__ import annotations

from typing import Any

from ._gtt_types import GttAction, GttRequest
from ._order_state import ENTRY_PURPOSES
from ._types import Purpose
from ._venue_requests import CancelRequest


def held_words(request: Any, broker_units: int, *, shared: bool = False) -> str:
    """The ERROR alert of `request` (a placement, modify or cancel the guard holds) of a
    guarded leg whose broker holds `broker_units` (signed) by OpenAlgo's position book;
    `shared` for a member of an instrument several strategies of the session share
    (module docstring)."""
    intent = request.intent
    leg = f"{intent.symbol} {intent.exchange} ({intent.action} {intent.quantity})"
    book = f"OpenAlgo's position book shows {broker_units} units"
    through_zero = "would take the broker's position through zero"
    if shared:
        book += f" of {intent.symbol} for the strategies sharing it, not what they hold together"
        through_zero = "waits with every closing order of them while the books differ"
    if isinstance(request, CancelRequest):
        return (
            f"{intent.strategy}: the cancel of the stop of {leg} is held back, so the stop "
            f"stays at the broker: {book}, and the exit it makes way for {through_zero}. "
            "Both wait for the books to agree, and go only if the session still asks for "
            "them then; check the position in OpenAlgo"
        )
    if intent.purpose in ENTRY_PURPOSES:
        return (
            f"{intent.strategy}: the entry of {leg} is held back: {book}, and it would close "
            "more than the broker holds. It is sent once the books agree, unless entries "
            "are locked by then; check the position in OpenAlgo"
        )
    what = "stop" if intent.purpose is Purpose.STOP else intent.purpose.value.replace("_", " ")
    return (
        f"{intent.strategy}: the {what} of {leg} is held back: {book}, and it {through_zero}. "
        "It is sent once the books agree; check the position in OpenAlgo"
    )


def held_gtt_words(
    request: GttRequest, trigger_id: str, broker_units: int, *, shared: bool = False
) -> str:
    """The ERROR alert of `request`, a GTT call of a CNC or NRML leg the guard holds
    (`_reconcile_guard_gtt`), whose GTT the broker knows as `trigger_id` ("" before its
    place was answered), on a guarded leg whose broker holds `broker_units` (signed) by
    OpenAlgo's books; `shared` as for `held_words`."""
    kind = "stop GTT" if request.purpose is Purpose.STOP else "GTT order"
    name = f"{kind} {trigger_id}" if trigger_id else kind
    leg = f"{request.symbol} {request.exchange}"
    book = f"OpenAlgo's books show {broker_units} units"
    if shared:
        book += f" of {request.symbol} for the strategies sharing it"
    if request.action is GttAction.CANCEL:
        return (
            f"{request.strategy}: the cancel of the {name} of {leg} is held back, so the GTT "
            f"stays at the broker: {book}, and the exit it makes way for would close more "
            "than the broker holds. Both wait for the books to agree; check the position in "
            "OpenAlgo"
        )
    return (
        f"{request.strategy}: the {name} of {leg} ({request.side} {request.quantity}) is held "
        f"back: {book}, and its child would close more than the broker holds. It is sent "
        "once the books agree; check the position in OpenAlgo"
    )


__all__ = ["held_gtt_words", "held_words"]
