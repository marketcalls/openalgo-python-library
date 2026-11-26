"""The GTTs of a session that stopped without its day-close record, rebuilt (`_carry_rebuild`;
20-positional-live 3 and 4.2 step 2; the lead's ruling N5c-2): what became of each GTT of
the last save after it, read from the journal, and given to the kernel session rebuilt for
that day through the calls a same-day restart gives them by.

The journal's GTT of each kernel GTT is `_session_restore_gtts.journal_rows`'s. Before the
kernel takes the save (`unsent`), each request the save waits for that never reached the
broker is found: a place the journal never recorded, or noted not placed; a modification
or cancellation the journal shows the GTT resting on its old terms for. The kernel drops
them (`Engine.restore_stopped_state`): a stop GTT never placed is carried without one, and
`protect` places it the next day.

Then each GTT's later fate (`apply`), by the kernel's state at the save and the journal's:

- Requested, bound to a trigger ID: `apply_gtt_placed`, then its later fate. Under way (no
  trigger ID): nothing, it is carried on the terms asked and the next start settles the
  place by the book.
- ModifyPending, modifying: nothing, carried on the terms asked. Resting on the
  modification's terms: `apply_gtt_modified`, then its later fate.
- CancelPending, cancelling: nothing, it rests until the next start's read. Cancelled:
  `apply_gtt_cancelled`.
- Resting or waiting, fired with its child bound: `apply_gtt_triggered`, then the child's
  fills from the journal; fired with no child found (the fall applied as its fill):
  `apply_gtt_triggered`, the child filled at the GTT's limit, flagged `price_unverified`.
  Fired before the save with its child bound after it: the child's fills.
- Triggered (gone, its child not resolved): nothing, the next start's follower judges it,
  as a record's.
- Resting or waiting, ended outside the session: `apply_gtt_ended` (expired, rejected) or
  `apply_gtt_cancelled`.
- Anything else is not accounted for: the start is refused.

Each one given is an update applied; each left to the next start is named with why. Times
are UTC nanoseconds, quantities whole units, prices rupees. Run by the main thread at the
start.
"""

from __future__ import annotations

import enum
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .._engine import call
from ._fill_ledger import FillLedger
from ._gtt_book import Seen
from ._gtt_followed import ENDINGS
from ._journal_gtt_rows import GttBinding, GttState
from ._session_restore_gtts import same_terms
from ._types import Purpose
from ._venue_children import missed_child_id
from .journal import Journal

#: The flag of a fired GTT's child fill priced at the GTT's limit (20-positional-live 3.3).
PRICE_UNVERIFIED = "price_unverified"
#: The kernel's states of a GTT whose request waits for the broker's answer.
_WAITING = frozenset({"Requested", "ModifyPending", "CancelPending"})
#: The kernel's states of a GTT that rests or waits (not ended, not fired).
_AT_VENUE = frozenset({"Active", *_WAITING})
#: The journal's states of a GTT that rests at the broker.
_RESTING = frozenset({GttState.PLACED, GttState.ACTIVE})
#: The journal's states of a GTT that fired, its child resolved or not.
_FIRED = (GttState.TRIGGERED, GttState.FIRED)
#: How the journal notes an ending the kernel takes as `apply_gtt_ended`, by its note.
_ENDED_HOW = {ENDINGS[Seen.EXPIRED][1]: "expired", ENDINGS[Seen.REJECTED][1]: "rejected"}


@dataclass
class GttsApplied:
    """What `apply` did: the updates applied, the words of each one left to the next
    start, the words of each one not accounted for, and the latest journal time read (UTC
    nanoseconds)."""

    applied: int = 0
    left: list[str] = field(default_factory=list)
    unaccounted: list[str] = field(default_factory=list)
    latest_ns: int = 0


class _Step(enum.Enum):
    """What one GTT's waiting request became (module docstring)."""

    ANSWERED = "answered"
    UNDER_WAY = "under_way"
    NEVER_SENT = "never_sent"
    UNACCOUNTED = "unaccounted"


@dataclass(frozen=True)
class GttContext:
    """What `apply` reads besides the kernel: the `journal` (handle of the session's
    ledger), the instrument's `tick` (rupees, which prices a child's fills) and the kernel
    orders the rebuild's order step gave their fills already (`handled`)."""

    journal: Journal
    tick: float
    handled: Collection[int] = ()


def unsent(views: Sequence[Mapping[str, Any]], rows: Mapping[int, GttBinding]) -> list[int]:
    """The kernel GTTs of `views` (`Engine.gtts` dicts of the save) whose waiting request
    never reached the broker (module docstring), by the journal's GTT of each (`rows`)."""
    return [
        int(view["gtt_id"])
        for view in views
        if view["state"] in _WAITING
        and _waiting(view, rows.get(int(view["gtt_id"])))[0] is _Step.NEVER_SENT
    ]


def apply(
    engine: Any,
    views: Sequence[Mapping[str, Any]],
    rows: Mapping[int, GttBinding],
    context: GttContext,
) -> GttsApplied:
    """Give `engine` (the rebuilt session) what became of each GTT of `views` (`Engine.gtts`
    dicts read before the venue's results) after the save (module docstring), by the
    journal's GTT of each (`rows`). Raises JournalError when the journal cannot be read."""
    done = GttsApplied()
    for view in views:
        row = rows.get(int(view["gtt_id"]))
        if row is not None:
            done.latest_ns = max(done.latest_ns, row.changed_ns)
        if view["state"] == "Triggered" and row is not None:
            _child_fills(engine, view, row, context, done)
            continue
        if view["state"] not in _AT_VENUE:
            continue
        step, why = _waiting(view, row)
        if step is _Step.NEVER_SENT:
            continue
        if row is None or step is _Step.UNACCOUNTED:
            done.unaccounted.append(f"{_name(view, row)}: {why}")
        elif step is _Step.UNDER_WAY:
            done.left.append(f"{_name(view, row)}: {why}")
        else:
            _answer(engine, view, row, done)
            _later(engine, view, row, context, done)
    return done


def _waiting(view: Mapping[str, Any], row: GttBinding | None) -> tuple[_Step, str]:
    """What the request GTT `view` waits for at the save became, by its journal's GTT `row`
    (module docstring): answered (or none waits), under way, never sent, or not
    accounted for, and why for the last two."""
    state = view["state"]
    if row is None and state == "Requested":
        return _Step.NEVER_SENT, ""
    if row is None:
        return _Step.UNACCOUNTED, "the journal holds no GTT of it"
    if state not in _WAITING:
        return _Step.ANSWERED, ""
    if row.state is GttState.NOT_PLACED and state == "Requested":
        return _Step.NEVER_SENT, ""
    under_way = {
        "Requested": (GttState.REQUESTED, "its place"),
        "ModifyPending": (GttState.MODIFYING, "its modification"),
        "CancelPending": (GttState.CANCELLING, "its cancellation"),
    }
    journal_state, what = under_way[state]
    if row.state is journal_state:
        return _Step.UNDER_WAY, f"{what} was under way when the session stopped"
    if state == "ModifyPending" and row.state in _RESTING:
        modified = same_terms(view["pending"], row)
        return (_Step.ANSWERED, "") if modified else (_Step.NEVER_SENT, "")
    if state == "CancelPending" and row.state in _RESTING:
        return _Step.NEVER_SENT, ""
    return _Step.ANSWERED, ""


def _answer(engine: Any, view: Mapping[str, Any], row: GttBinding, done: GttsApplied) -> None:
    """Give the kernel the answer to the request `view` waited for (module docstring)."""
    gtt_id, state = int(view["gtt_id"]), view["state"]
    if state == "Requested":
        call(engine.apply_gtt_placed, gtt_id)
        done.applied += 1
    elif state == "ModifyPending" and row.state in _RESTING:
        call(engine.apply_gtt_modified, gtt_id)
        done.applied += 1
    elif state == "CancelPending" and row.state is GttState.CANCELLED:
        call(engine.apply_gtt_cancelled, gtt_id)
        done.applied += 1


def _later(
    engine: Any, view: Mapping[str, Any], row: GttBinding, context: GttContext, done: GttsApplied
) -> None:
    """Give the kernel what became of GTT `view` once it rested (module docstring)."""
    gtt_id = int(view["gtt_id"])
    if row.state is GttState.TRIGGERED:
        done.left.append(
            f"{_name(view, row)}: it went from the GTT book after the save, its child not "
            "found: this start's read of the GTT book judges it"
        )
    elif row.state is GttState.FIRED:
        fired_ns = _state_ns(context.journal, row, _FIRED)
        child = call(engine.apply_gtt_triggered, gtt_id, fired_ns)
        done.applied += 1
        _child_fills(engine, {**view, "child_order_id": child}, row, context, done)
    elif row.state is GttState.ENDED:
        how = _ENDED_HOW.get(_last_note(context.journal, row))
        if how is None:
            call(engine.apply_gtt_cancelled, gtt_id)
        else:
            call(engine.apply_gtt_ended, gtt_id, how)
        done.applied += 1
    elif row.state not in _RESTING and not (
        view["state"] == "CancelPending" and row.state is GttState.CANCELLED
    ):
        why = f"the engine's was {view['state']} and the journal's says {row.state.value}"
        done.unaccounted.append(f"{_name(view, row)}: {why}")


def _child_fills(
    engine: Any, view: Mapping[str, Any], row: GttBinding, context: GttContext, done: GttsApplied
) -> None:
    """The fills of the child of fired GTT `view` (its kernel order `child_order_id`): from
    the journal's updates of the child bound to its journal GTT `row`, or, for a child
    never found (none bound, or the session's own ID for one it could not find), its units
    at the GTT's limit, flagged (module docstring). None for a child the order step gave
    its fills already, or one the save holds filled."""
    child = view.get("child_order_id")
    if child is None or int(child) in context.handled:
        return
    order = call(engine.order, int(child))
    if order is None or int(child) not in call(engine.working_orders):
        return
    bound = context.journal.gtt_child(row.seq)
    missed = row.trigger_id is not None and missed_child_id(row.trigger_id)
    if bound is None or bound.order_id == missed:
        if row.state is not GttState.FIRED:
            return
        fired_ns = _state_ns(context.journal, row, _FIRED)
        units, price = float(row.terms.quantity), row.terms.limit_price
        call(engine.apply_fill, int(child), units, price, fired_ns, flags=[PRICE_UNVERIFIED])
        done.applied += 1
        done.left.append(
            f"{_name(view, row)}: it fired after the save with its child order not found: "
            f"its {row.terms.quantity} units are recorded at its limit, {price:.2f}, flagged "
            f"{PRICE_UNVERIFIED}"
        )
        return
    update = context.journal.last_update(bound.order_id, day=bound.day)
    if update is None:
        return
    done.latest_ns = max(done.latest_ns, update.ts_recv_ns)
    filled = round(float(order["filled_units"]))
    value = filled * float(order.get("average_fill_rupees") or 0.0)
    ledger = FillLedger(
        engine,
        int(child),
        tick=context.tick,
        alerts=NamedWords(done.left),
        filled_units=filled,
        filled_value=value,
    )
    result = ledger.apply(update)
    if result.increment is not None or result.ended is not None:
        done.applied += 1


def _last_note(journal: Journal, row: GttBinding) -> str:
    states = journal.gtt_states(row.seq)
    return states[-1][1] if states else ""


def _state_ns(journal: Journal, row: GttBinding, states: tuple[GttState, ...]) -> int:
    """When `row` first took one of `states`, by the journal (its last change if none)."""
    found = [at for state, _, at in journal.gtt_states(row.seq) if state in states]
    return min(found) if found else row.changed_ns


@dataclass
class NamedWords:
    """An alerts sink that keeps each text as a word for the banner (`FillLedger`'s
    refusals: the next start's reconciliation reads the books again)."""

    words: list[str]

    def send(self, kind: Any, text: str) -> None:
        """Keep `text` (an alert of `kind`)."""
        self.words.append(text)


def _name(view: Mapping[str, Any], row: GttBinding | None) -> str:
    """A GTT in the banner's words: its kind and the broker's trigger ID (its kernel id
    when none)."""
    stop = view.get("purpose") == "stop" or (row is not None and row.purpose is Purpose.STOP)
    kind = "stop GTT" if stop else "GTT order"
    trigger = None if row is None else row.trigger_id
    return f"its {kind} {trigger or 'g' + str(view['gtt_id'])}"


__all__ = ["PRICE_UNVERIFIED", "GttContext", "GttsApplied", "NamedWords", "apply", "unsent"]
