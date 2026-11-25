"""The exit guard's rules for the GTT calls of a CNC or NRML leg (16-session-design 5.4 "As
built"; 20-positional-live 7.2; the lead's ruling N3b-4), split from `_reconcile_guard` so
each module keeps one job.

A CNC or NRML leg's stop is a SINGLE GTT the broker holds, so the guard's rules for a
leg's stop order apply to its stop GTT:

- **Guarded or awaiting a read**, with the engine holding a position: the cancel of the
  stop GTT is held (a target's, a strategy's or the safe stop's exit asks for it first,
  and that exit would be held), so the GTT stays at the broker protecting what it holds
  and the exit waits in the engine for the cancel's answer. A GTT order of the strategy
  placed against its position is held as an entry against it is.
- **Guarded**: a stop GTT placed again, or modified to more units than rest, beyond the
  room (`_reconcile_rules.exit_room`) is held, as a stop order placed again is: with CNC
  its child would sell the user's own holdings, with NRML it would take the position
  through zero. A trail (the same units) goes.
- **Released** once the leg is neither guarded nor awaiting a read: each held GTT call
  whose kernel GTT still waits for it goes, in order (the books agree then, and the
  kernel fits its stop to an adopted sale by a modify that waits behind a held place);
  one the kernel no longer waits for (the GTT fired or ended meanwhile) is dropped.
- **After the session's last lines**: only what takes exposure away goes, as for orders:
  the cancel of a GTT order, the cancel of a stop GTT once the engine holds the leg flat,
  and a modify that only lowers a stop GTT's units.

Quantities are whole units. Used by the loop only.
"""

from __future__ import annotations

from typing import Any

from .._engine import call
from ._gtt_types import GttAction, GttRequest
from ._reconcile_rules import exit_room, reduces
from ._types import Purpose

#: The kernel's state of a GTT (the binding's `GTT_STATES`) that waits for each call.
_AWAITS = {
    GttAction.PLACE: "Requested",
    GttAction.MODIFY: "ModifyPending",
    GttAction.CANCEL: "CancelPending",
}


def gtt_holds(
    trader: Any, request: GttRequest, *, awaiting: bool, broker: int | None, hold_all: bool
) -> bool:
    """Whether the guard holds `request` of `trader`'s leg (module docstring) while the leg
    awaits a read (`awaiting`) or is guarded (`broker`, the broker's signed units, not
    None); `hold_all` for a member of a shared instrument."""
    if not awaiting and broker is None:
        return False
    engine = round(trader.open_units())
    if request.purpose is not Purpose.STOP:
        return request.action is GttAction.PLACE and reduces(request.side, engine)
    if request.action is GttAction.CANCEL:
        return engine != 0
    if broker is None:
        # A stop placed again rests until triggered: only a guarded leg's is held.
        return False
    if request.action is GttAction.MODIFY and request.quantity <= _resting_units(trader, request):
        return False
    return hold_all or request.quantity > exit_room(engine, broker)


def gtt_takes_away(trader: Any, request: GttRequest) -> bool:
    """Whether `request` only takes exposure away, so it goes after the session's last
    lines (module docstring)."""
    if request.action is GttAction.CANCEL:
        return request.purpose is not Purpose.STOP or round(trader.open_units()) == 0
    if request.action is GttAction.MODIFY and request.purpose is Purpose.STOP:
        resting = _view(trader, request)
        if resting is None:
            return False
        same = (resting["trigger_price"], resting["price"]) == (
            request.trigger_price,
            request.limit_price,
        )
        return same and request.quantity < round(float(resting["quantity_units"]))
    return False


def gtt_awaited(trader: Any, request: GttRequest) -> bool:
    """Whether `trader`'s kernel still waits for `request`'s answer, so a held call may go
    (module docstring)."""
    view = _view(trader, request)
    return view is not None and view["state"] == _AWAITS[request.action]


def _view(trader: Any, request: GttRequest) -> dict[str, Any] | None:
    view = call(trader.engine.gtt, request.gtt_id)
    return None if view is None else dict(view)


def _resting_units(trader: Any, request: GttRequest) -> int:
    """The units of `request`'s GTT as they rest now (a modify's are the kernel's
    `pending` until answered)."""
    view = _view(trader, request)
    return 0 if view is None else round(float(view["quantity_units"]))


__all__ = ["gtt_awaited", "gtt_holds", "gtt_takes_away"]
