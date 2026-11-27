"""What a close's day-close record adds to the carried state of a session rebuilt after it
stopped without its close (`_carry_rebuild`; 20-positional-live 4.1 and 4.2 step 2; the
lead's ruling N5c-2), split from it so each module keeps one job.

- **The leg's reference** of the broker's units: the save's last reconciliation read, when
  it still holds (settled, no order of the session working when it stopped, the units the
  leg's, and nothing of the day's orders filled after the read). With none, a stop GTT
  gone from the next start's book is a mismatch, as for any record without one.
- **Each carried GTT that reference confirms**: the journal shows a read of the GTT book
  found it resting on the terms it kept since before the reference's read, so it had not
  fired then. The others are named not confirmed (`NOT_CONFIRMED`: no close read was
  made, so the start's first read of the GTT book judges them).
- **The journal's GTT each carried GTT is bound to**, by kernel id (`BINDINGS_KEY`).

Times are UTC nanoseconds, quantities whole units. Run by the main thread at the start.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping, Sequence
from typing import Any

from ._broker_units import strategy_units
from ._carry_start import BINDINGS_KEY, CONFIRMED_KEY, REFERENCES_KEY, UNCONFIRMED_KEY
from ._journal_gtt_rows import GttBinding, GttState
from .journal import Journal

#: Why a rebuilt record's stop GTTs are not confirmed.
NOT_CONFIRMED = "the session stopped before its close read the GTT book"


def add_record_parts(
    carried: dict[str, Any],
    rows: Mapping[int, GttBinding],
    saved_state: Mapping[str, Any],
    journal: Journal,
    day: dt.date,
    working: set[int],
) -> None:
    """Add to `carried` (the kernel's carried state of the session of `day`) what a close's
    record adds (module docstring): the leg's reference, from the save's `saved_state`
    while no kernel order was `working` when the session stopped; each carried GTT it
    confirms, and each other named not confirmed; and the journal's GTT each carried GTT
    is bound to (`rows`, by kernel id). Raises JournalError when the journal cannot be
    read."""
    reference = _reference(saved_state, carried, journal, day, working)
    gtts = [int(gtt["gtt"]) for gtt in carried.get("gtts", []) if int(gtt["gtt"]) in rows]
    read_ns = int(reference[0]["read_ns"]) if reference else None
    confirmed = [gtt for gtt in gtts if _confirmed(rows[gtt], read_ns)]
    carried[REFERENCES_KEY] = reference
    carried[CONFIRMED_KEY] = confirmed
    carried[UNCONFIRMED_KEY] = [
        {"gtt": gtt, "trigger_id": rows[gtt].trigger_id or "", "why": NOT_CONFIRMED}
        for gtt in gtts
        if gtt not in confirmed
    ]
    carried[BINDINGS_KEY] = [{"gtt": gtt, "seq": rows[gtt].seq} for gtt in gtts]


def _reference(
    state: Mapping[str, Any],
    carried: Mapping[str, Any],
    journal: Journal,
    day: dt.date,
    working: set[int],
) -> list[dict[str, Any]]:
    """The leg's reference of the broker's units the rebuilt record keeps (module
    docstring): the save's, when it still holds (no order of the session's still `working`
    when it stopped); else none."""
    references = list(state.get(REFERENCES_KEY) or [])
    if len(references) != 1 or working:
        return []
    reference = dict(references[0])
    if not reference.get("settled"):
        return []
    held = sum(round(float(p["quantity_units"])) for p in carried.get("positions", []))
    broker = int(reference["units"])
    if strategy_units(str(reference["product"]), held, broker) != held:
        return []
    read_ns = int(reference["read_ns"])
    for order in journal.bound_orders(day=day):
        if _filled_after(journal.updates(order.order_id, day=day), read_ns):
            return []
    return [reference]


def _filled_after(words: Sequence[Any], read_ns: int) -> bool:
    """Whether an order's journal `words` (its updates, cumulative) show units filled after
    `read_ns` (UTC nanoseconds): more filled in all of them than in those received by then
    (a later word of a fill made before, a poll's, moves nothing)."""
    before = max((w.filled_quantity for w in words if w.ts_recv_ns <= read_ns), default=0)
    return max((w.filled_quantity for w in words), default=0) > before


def _confirmed(row: GttBinding, read_ns: int | None) -> bool:
    """Whether the leg's reference read at `read_ns` (None for none) confirms the GTT the
    journal's `row` is: a read of the GTT book showed it resting (ACTIVE) on terms it kept
    from before the reference's read, so it had not fired by then (the reference's units
    hold the leg's, which a fire would have lowered)."""
    resting = row.state is GttState.ACTIVE and row.trigger_id is not None
    return read_ns is not None and resting and row.changed_ns <= read_ns


__all__ = ["NOT_CONFIRMED", "add_record_parts"]
