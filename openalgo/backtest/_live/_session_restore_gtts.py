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
- CancelPending, the journal's cancelled with the same terms: cancelled before the crash;
  taken back and confirmed (`apply_gtt_cancelled`), so the kernel goes on with what waited
  for the cancel (a target's exit). Left unsettled, that exit would never be sent, beside
  a stop the broker no longer holds (review finding 5 of round 3 of unit N3b).
- Requested, the journal's not placed (refused by the broker, or never reached it): placed
  again; a second refusal is answered as one (`apply_gtt_refused`: the leg is exited and
  entries locked for the day), and a transient one is placed.
- Requested, or ModifyPending (a replace's place after its cancel), the journal's
  requested and bound to no trigger ID (its place under way when the crash came: it may
  rest at the broker, which only a read of the GTT book can tell): settled by the order
  worker as an unanswered place (`_carry_restore.settle_place`; the lead's ruling N5a-5):
  one of its shape resting is bound, none with no fall since a reference from before the
  place is placed again once, anything else is unresolved.
- Anything else: unsettled.

"Bound" is a journal's GTT bound to the broker's trigger ID that has not ended (placed,
resting, or fired with its child not resolved, which the follower finds). The journal's
GTT of a restored one is the latest of the day's GTTs of the strategy with its kernel id
that has not ended, else the latest that has (`Journal.day_gtts`), else, for a GTT carried
from an earlier day, the one the saved state binds it to (`_carry_restore.gtt_bindings`).
A modify or cancel the crash caught under way, a GTT that fired or ended with its child
followed, or terms that agree with neither is **unsettled**: the GTT map sends nothing
more about it, the strategy's entries are locked with an ERROR alert naming it, and
nothing is exited (as for an unresolved GTT, 16-session-design 5.3): the user settles it
by hand.

Quantities are whole units, prices rupees. Run by the main thread at a restart.
"""

from __future__ import annotations

import enum
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .._engine import call
from ._carry_restore import settle_place
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
    CONFIRM_CANCEL = "confirm_cancel"
    SETTLE_PLACE = "settle_place"
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
    if row is not None and row.state.terminal:
        return _ended(view, row)
    bound = row is not None and row.trigger_id is not None and row.state in _BOUND
    if state == "Requested" and row is None:
        return Settled(Settle.PLACE)
    under_way = row is not None and row.state is GttState.REQUESTED and row.trigger_id is None
    if state in ("Requested", "ModifyPending") and under_way:
        # A place, or a replace's place after its cancel, under way: settled (module docstring).
        return Settled(Settle.SETTLE_PLACE)
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


def _ended(view: Mapping[str, Any], row: GttBinding) -> Settled:
    """What a restart does with the kernel's restored GTT `view` whose journal's GTT `row`
    has ended (module docstring's table)."""
    state = view["state"]
    if state == "CancelPending" and row.state is GttState.CANCELLED and _same(view, row):
        return Settled(Settle.CONFIRM_CANCEL)
    if state == "Requested" and row.state is GttState.NOT_PLACED:
        return Settled(Settle.PLACE)
    return Settled(
        Settle.UNSETTLED, f"the engine's GTT was {state} and the journal's says {row.state.value}"
    )


def restore_gtts(session: Any, trader: Trader, bindings: Mapping[int, int] | None = None) -> None:
    """Take back `trader`'s restored GTTs (module docstring) and hand the order worker the
    calls that never reached it, after the kernel's restore and before its leg is
    protected; `session` is the `_session_loop.Session`, `bindings` the journal's GTT of
    each kernel GTT the saved state kept (`_carry_restore.gtt_bindings`), by which a GTT
    carried from an earlier day is found. Used for a saved state of today only: a carry
    from an earlier day is taken back by `_carry_restore`, and a GTT its kernel places
    since goes out with the loop's commands. A GTT the strategy's GTT map holds already is
    left as it is."""
    gtts = trader.gtts
    if gtts is None or not overnight(trader):
        return
    views = [dict(v) for v in call(trader.engine.gtts, INSTRUMENT, trader.product)]
    views = [v for v in views if v["state"] in _LIVE_STATES and gtts.gtt(int(v["gtt_id"])) is None]
    if not views:
        return
    journal = session.parts.journal
    today = journal.day_gtts(trader.name, day=session.parts.spec.day)
    # By kernel GTT id, the latest that has not ended, else the latest (module docstring).
    rows = {row.gtt_id: row for row in today if row.state.terminal}
    rows.update({row.gtt_id: row for row in today if not row.state.terminal})
    for gtt_id, seq in (bindings or {}).items():
        if gtt_id not in rows and (carried := journal.gtt(seq)) is not None:
            rows[gtt_id] = carried
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
    if decided.settle is Settle.SETTLE_PLACE and row is not None:
        return settle_place(gtts, view, row)
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
    elif decided.settle is Settle.CONFIRM_CANCEL:
        call(trader.engine.apply_gtt_cancelled, gtt_id)
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
