"""The start's read of what a session continues from earlier days (`open_carry`, on a
journal handle opened for it before the preflight; `_carry_start`), and what it cannot
continue, refused before anything is sent (20-positional-live 4.2 step 2; the lead's
ruling N5a-1): a capital other than a carried strategy's (`_carry_start.capital_refusals`),
and, since a session continues a position from an earlier day only from its day-close
record,

- a strategy whose latest session, of an earlier day, stopped without writing its record
  (the process ended before the close) while its last save of that day holds a position or
  a GTT is refused: its state then is that save moved by the journal's later words, which
  this version does not carry into today's kernel; and
- a strategy holding a GTT of the journal from an earlier day that has not ended and that
  no carried record binds (one a session placed and never recorded at a close) is
  refused: started afresh it would trade beside a GTT no engine follows.

Each refusal names what is held and what to do. Days are IST trading days. Run by the
main thread at the start, before the preflight.
"""

from __future__ import annotations

import datetime as dt
import pathlib
from collections.abc import Sequence
from typing import Any

from ._carry_start import CarryAtStart, capital_refusals, gtt_seqs, read_carry
from ._journal_gtt_rows import GttBinding
from ._preflight_plan import refused
from ._types import Mode, Now, Purpose
from .journal import Journal

#: What a start refused for what a strategy holds from an earlier day asks of the operator.
CHECK_BY_HAND = (
    "check the position and its GTTs in OpenAlgo, close or cancel them by hand, then start again"
)


def open_carry(
    spec: Any, *, path: pathlib.Path, account: str, broker: str, mode: Mode, now: Now
) -> CarryAtStart:
    """Read the carry of the session `spec` (a `plan.SessionSpec`) from the journal at
    `path` of `account` at `broker`, in the ledger of `mode`, on a handle opened for the
    read and closed after it (module docstring), and refuse a start whose capital is not a
    carried strategy's or that holds what no record carries. Raises PreflightError for such
    a start, JournalError when the journal cannot be read, and LiveError for a record that
    cannot be read."""
    names = [each.name for each in spec.strategies]
    with Journal.open(path, account=account, broker=broker, mode=mode, now=now) as journal:
        carry = read_carry(journal, names, spec.day, mode)
        refusals = uncarried_refusals(journal, names, spec.day, carry)
    capitals = {each.name: each.capital for each in spec.strategies}
    refusals = capital_refusals(carry, capitals) + refusals
    if refusals:
        raise refused(tuple(refusals))
    return carry


def uncarried_refusals(
    journal: Journal, names: Sequence[str], day: dt.date, carry: CarryAtStart
) -> list[str]:
    """A refusal for each of the strategies `names` that holds from an earlier day what no
    day-close record carries into the session of `day` (module docstring). Raises
    JournalError when the journal cannot be read."""
    open_gtts = journal.open_gtts()
    refusals = []
    for name in names:
        carried = carry.of(name)
        held = _unrecorded_save(journal, name, day, carry)
        bound = set() if carried is None else set(gtt_seqs(carried.record).values())
        held += [
            _gtt_words(gtt)
            for gtt in open_gtts
            if gtt.strategy == name and gtt.day < day and gtt.seq not in bound
        ]
        if held:
            refusals.append(f"{name}: {'; '.join(held)}: {CHECK_BY_HAND}")
    return refusals


def _unrecorded_save(journal: Journal, name: str, day: dt.date, carry: CarryAtStart) -> list[str]:
    """Words for strategy `name`'s last save of an earlier day after its latest record
    that holds a position or a GTT (module docstring); none otherwise."""
    last = journal.last_ran(name, day)
    carried = carry.of(name)
    if last is None or (carried is not None and carried.record.day >= last):
        return []
    record = journal.day_close(name)
    if record is not None and record.day >= last:
        return []
    saved = journal.rms_state(name, day=last)
    state = {} if saved is None else dict(saved.get("live_state", {}))
    positions = [p for p in state.get("positions", []) if float(p.get("quantity_units", 0))]
    if not positions and not state.get("gtts"):
        return []
    return [
        f"its session of {last} stopped without its day-close record (the process ended "
        "before the close) while holding a position or a GTT, which this version continues "
        "only from a day-close record"
    ]


def _gtt_words(gtt: GttBinding) -> str:
    """A GTT of an earlier day that has not ended and that no record carries."""
    kind = "stop GTT" if gtt.purpose is Purpose.STOP else "GTT order"
    name = gtt.trigger_id or f"g{gtt.gtt_id} (its place was not answered)"
    return (
        f"its {kind} {name} of {gtt.day} on {gtt.exchange}:{gtt.symbol} ({gtt.product}) has "
        "not ended and no day-close record carries it"
    )


__all__ = ["CHECK_BY_HAND", "open_carry", "uncarried_refusals"]
