"""The alerts the stops supervisor (`stops`) sends about the requests of a leg
(16-session-design 4.2 and 4.3; 18-multi-strategy 4.4, unit M6):

- a stop, an exit or a protective modify the order worker holds back (`held_back`), once
  per request: OpenAlgo silent, or an order of the instrument unresolved, which on a
  shared instrument may be another member's unanswered placement (`instrument_held_by`);
- an add whose answer was lost on a guarded leg (`unbound_add`): its units may be filled
  and uncovered;
- a trailing modify of the stop the broker accepted (`stop_moved`, STOP_MOVED).

Quantities are units, prices rupees. Used by the loop only.
"""

from __future__ import annotations

from typing import Any

from ._session_trader import Trader
from ._stops_cover import held_by_other
from ._types import AlertKind, Purpose
from ._venue_requests import STOP_TYPES, ModifyRequest, OrderRequest, PlaceRequest
from .inbox import NotSent, NotSentReason


def instrument_held_by(session: Any, stops: Any, trader: Trader) -> tuple[str, str] | None:
    """Another strategy of `trader`'s instrument with a placement unanswered, which holds
    every send for the instrument, as (its name, the placement's words); None when there
    is none (`session` the `_session_loop.Session`, `stops` its supervisor)."""
    members = session.groups.members(trader)
    return held_by_other((m.name, stops.leg(m)) for m in members if m is not trader)


def held_back(
    session: Any, stops: Any, trader: Trader, request: OrderRequest, message: NotSent
) -> None:
    """The ERROR alert of a protective request the order worker held back (module
    docstring): what waits, why, and the units left without a confirmed stop."""
    what = "stop" if request.intent.purpose is Purpose.STOP else "exit"
    if isinstance(request, ModifyRequest):
        what = f"change of the {what}"
    units = round(abs(trader.open_units())) - stops.covered(trader)
    why = f"{message.reason.value}: {message.error}"
    holder = None
    if message.reason is NotSentReason.HELD:
        holder = instrument_held_by(session, stops, trader)
    if holder is not None:
        why = f"{holder[0]}'s order {holder[1]} of the instrument has no answer yet"
    session.alert(
        AlertKind.ERROR,
        f"{trader.name}: the {what} of {trader.symbol} {trader.exchange} cannot be sent "
        f"now ({why}); it waits, and {max(units, 0)} units have no confirmed stop at the "
        "broker meanwhile",
    )


def unbound_add(session: Any, stops: Any, trader: Trader, request: PlaceRequest) -> None:
    """The ERROR alert of an add whose answer was lost, on a guarded leg only (module
    docstring)."""
    if not stops.leg(trader).guarded:
        return
    session.alert(
        AlertKind.ERROR,
        f"{trader.name}: the add of {request.intent.quantity} {trader.symbol} "
        f"{trader.exchange} got no answer: if it filled, those units have no stop at the "
        "broker, and the change of the stop is held, until the order book shows it",
    )


def stop_moved(session: Any, trader: Trader, request: ModifyRequest) -> None:
    """The STOP_MOVED alert of a trailing modify of the leg's stop the broker accepted:
    its units kept, its trigger moved; nothing for any other modify."""
    old, new = request.replaces, request.terms
    trailing = old.quantity == new.quantity and {old.price_type, new.price_type} <= STOP_TYPES
    if request.intent.purpose is Purpose.STOP and trailing:
        session.alert(
            AlertKind.STOP_MOVED,
            f"{trader.name}: stop of {trader.symbol} {trader.exchange} moved to "
            f"{new.trigger_price:.2f} (order {request.order_id})",
        )


__all__ = ["held_back", "instrument_held_by", "stop_moved", "unbound_add"]
