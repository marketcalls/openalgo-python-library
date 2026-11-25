"""The GTTs a same-day restart takes back (16-session-design 3.6; 20-positional-live 3; the
lead's ruling N3b-2, review finding 1 of unit N3b): each GTT the kernel restores from the
saved state, joined to the journal's GTT of the same strategy, day and kernel GTT id, so
the strategy's GTT map (`_venue_gtts.GttMap.take_back`) sends the kernel's later commands
about it and applies the follower's findings.

The state is saved after the loop takes a batch of commands and before the order worker
sends them, and again at each slice it releases; the journal is written before each GTT
call and with each answer (`_gtt_send`). So the journal tells what became of a call the
saved state still waits for (`settle`), by the kernel's state and the journal's GTT:

- Requested, none in the journal: its place never reached the worker; placed now.
- Requested, bound with the same terms: placed before the crash; confirmed
  (`apply_gtt_placed`) and taken back.
- Active, bound with the same terms: taken back.
- ModifyPending, bound with the resting terms: its modify never reached the worker; taken
  back and modified now.
- ModifyPending, bound with the modify's terms: modified before the crash; taken back
  and confirmed (`apply_gtt_modified`).
- CancelPending, bound with the resting terms: its cancel never reached the worker; taken
  back and cancelled now.
- Anything else: unsettled.

"Bound" is a journal's GTT bound to the broker's trigger ID that has not ended (placed,
resting, or fired with its child not resolved, which the follower finds). A call the
crash caught under way (a place, modify or cancel sent and not answered), a GTT that fired
with its child followed, or terms that agree with neither is **unsettled**: the GTT map
sends nothing more about it, the strategy's entries are locked with an ERROR alert naming
it, and nothing is exited (as for an unresolved GTT, 16-session-design 5.3): the user
settles it by hand. GTTs of earlier days are unit N5's (the follower follows them).

Quantities are whole units, prices rupees. Run by the main thread at a restart.
"""

from __future__ import annotations

import enum
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .._engine import call
from ._gtt_types import GttRequest
from ._journal_gtt_rows import GttBinding, GttState
from ._plan_catch_up import INSTRUMENT
from ._session_gtts import lock_unresolved
from ._session_trader import Trader
from ._stops_gtt import overnight
from ._venue_gtts import GttMap

#: The kernel's states (the binding's `GTT_STATES`) of a GTT that has not ended.
_LIVE_STATES = frozenset({"Requested", "Active", "ModifyPending", "CancelPending", "Triggered"})
#: The journal's states of a GTT bound at the broker that has not ended.
_BOUND = frozenset({GttState.PLACED, GttState.ACTIVE, GttState.TRIGGERED})
#: Half a paisa: two trigger prices closer than it are the same price.
_PAISA = 0.005


class Settle(enum.Enum):
    """What a restart does with one restored GTT (module docstring)."""

    PLACE = "place"
    CONFIRM_PLACE = "confirm_place"
    TAKE_BACK = "take_back"
    MODIFY = "modify"
    CONFIRM_MODIFY = "confirm_modify"
    CANCEL = "cancel"
    UNSETTLED = "unsettled"


@dataclass(frozen=True)
class Settled:
    """What `settle` decided for a restored GTT, and why when it is unsettled."""

    settle: Settle
    why: str = ""


def settle(view: Mapping[str, Any], row: GttBinding | None) -> Settled:
    """What a restart does with the kernel's restored GTT `view` (an `Engine.gtt` dict of a
    GTT that has not ended) given the journal's GTT of it (`row`, None for none): module
    docstring's table."""
    state = view["state"]
    bound = row is not None and row.trigger_id is not None and row.state in _BOUND
    if state == "Requested" and row is None:
        return Settled(Settle.PLACE)
    if bound and state in ("Requested", "Active") and _same(view, row):
        return Settled(Settle.CONFIRM_PLACE if state == "Requested" else Settle.TAKE_BACK)
    if bound and state == "ModifyPending" and _same(view, row):
        return Settled(Settle.MODIFY)
    if bound and state == "ModifyPending" and _same(view["pending"], row):
        return Settled(Settle.CONFIRM_MODIFY)
    if bound and state == "CancelPending" and _same(view, row):
        return Settled(Settle.CANCEL)
    seen = "none in the journal" if row is None else f"the journal's says {row.state.value}"
    return Settled(Settle.UNSETTLED, f"the engine's GTT was {state} and {seen}")


def restore_gtts(session: Any, trader: Trader) -> None:
    """Take back `trader`'s restored GTTs (module docstring) and hand the order worker the
    calls that never reached it, after the kernel's restore and before its leg is
    protected; `session` is the `_session_loop.Session`."""
    gtts = trader.gtts
    if gtts is None or not overnight(trader):
        return
    views = [dict(v) for v in call(trader.engine.gtts, INSTRUMENT, trader.product)]
    views = [view for view in views if view["state"] in _LIVE_STATES]
    if not views:
        return
    day = session.parts.spec.day
    rows = {
        row.gtt_id: row
        for row in session.parts.journal.open_gtts()
        if row.strategy == trader.name and row.day == day
    }
    requests: list[GttRequest] = []
    for view in views:
        row = rows.get(int(view["gtt_id"]))
        decided = settle(view, row)
        requests += _apply(session, trader, gtts, view, row, decided)
        session.event(
            "gtt_restored", strategy=trader.name, gtt=view["gtt_id"], settle=decided.settle.value
        )
    session.hand_over(requests)


def _apply(
    session: Any,
    trader: Trader,
    gtts: GttMap,
    view: dict[str, Any],
    row: GttBinding | None,
    decided: Settled,
) -> list[GttRequest]:
    """Apply one decision (module docstring) to `trader`'s GTT map `gtts`; the GTT calls to
    send now."""
    gtt_id = int(view["gtt_id"])
    if decided.settle is Settle.PLACE:
        return gtts.requests({**view, "command": "place_gtt"})
    seq = None if row is None else row.seq
    trigger_id = "" if row is None or row.trigger_id is None else row.trigger_id
    unsettled = decided.settle is Settle.UNSETTLED
    gtts.take_back(view, seq=seq, trigger_id=trigger_id, unsettled=unsettled)
    if unsettled:
        kind = "stop GTT" if view["purpose"] == "stop" else "GTT order"
        name = f"{kind} {trigger_id}" if trigger_id else f"{kind} g{gtt_id}"
        lock_unresolved(
            session, trader, f"{name} cannot be taken back by the restart ({decided.why})"
        )
        return []
    if decided.settle is Settle.CONFIRM_PLACE:
        call(trader.engine.apply_gtt_placed, gtt_id)
    elif decided.settle is Settle.CONFIRM_MODIFY:
        call(trader.engine.apply_gtt_modified, gtt_id)
    elif decided.settle is Settle.MODIFY:
        return gtts.requests({**view["pending"], "command": "modify_gtt", "gtt_id": gtt_id})
    elif decided.settle is Settle.CANCEL:
        return gtts.requests({**view, "command": "cancel_gtt"})
    return []


def _same(terms: Mapping[str, Any] | None, row: GttBinding | None) -> bool:
    """Whether the kernel's `terms` are the journal's GTT's: side, units and trigger (the
    limit is the broker's own rounding, 20-positional-live 3.6)."""
    if terms is None or row is None:
        return False
    return (
        terms["side"] == row.terms.side
        and round(float(terms["quantity_units"])) == row.terms.quantity
        and abs(float(terms["trigger_price"]) - row.terms.trigger_price) < _PAISA
    )


__all__ = ["Settle", "Settled", "restore_gtts", "settle"]
