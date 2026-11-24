"""What a read of the GTT book says of a GTT call that waits for its confirmation
(20-positional-live 3.1, "Confirmations" and "Unanswered calls"; the lead's ruling N3a-1).
The order worker's GTT sender (`_gtt_send`) reads the book at once after each call and
every `READ_AGAIN_AFTER_S` while one waits (`Pending`), and acts on the `Verdict` of
`judge`; nothing else is sent for that GTT meanwhile.

- **A place** is confirmed by a success with a trigger ID and the GTT resting in the read
  (active or transit; an unknown status counts as resting): PLACED. A GTT the read shows
  fired or ended is placed all the same, and the follower judges it from then; one the
  book does not show is read again, and after `MISSES_BEFORE_DECIDING` reads it is taken as
  gone. An unanswered place is looked for by its shape (symbol, exchange, product, side,
  trigger and units) among the rows no GTT of the journal holds: one found is bound; none
  after `MISSES_BEFORE_DECIDING` reads, it is placed again once, then UNRESOLVED; several
  are UNRESOLVED, never a guess.
- **A modify** is confirmed by the read showing its trigger and units, compared by value
  (MODIFIED). The old terms after as many reads: an unanswered modify was not made
  (MODIFY_REFUSED); an answered one cannot be told. A modify that cannot be told, or that
  half-applied (other terms), is cancelled and the GTT placed again (REPLACE). A GTT that
  is no longer resting is UNRESOLVED: the follower judges it.
- **A cancel** is confirmed by the GTT absent or ended without firing, with no child order
  found for it (CHILD_CHECK: the sender reads the order book); fired, it is
  CANCEL_REFUSED; still resting after as many reads, CANCEL_REFUSED.

Times are UTC nanoseconds, prices rupees, quantities whole units.
"""

from __future__ import annotations

import enum
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from ._gtt_book import GttRow, Seen, seen_of
from ._gtt_types import GttAnswerKind, GttRequest
from ._journal_gtt_rows import GttTerms

#: Seconds between two reads while a call waits for its confirmation (3.1: "two reads 2
#: seconds apart").
READ_AGAIN_AFTER_S = 2.0
#: Reads that miss a GTT, or show a modify's old terms or a cancelled GTT resting, before
#: the confirmation decides (3.1).
MISSES_BEFORE_DECIDING = 2


class Stage(enum.Enum):
    """What a call waiting for its confirmation was."""

    PLACE = "place"
    MODIFY = "modify"
    CANCEL = "cancel"
    #: The cancel of a GTT whose modify cannot be told, before it is placed again.
    REPLACE_CANCEL = "replace_cancel"
    #: The place of that GTT again, with the modify's terms.
    REPLACE_PLACE = "replace_place"


@dataclass
class Pending:
    """A call waiting for its confirmation: its stage and request, the journal's GTT and
    the broker's trigger ID ("" for an unanswered place not found yet), whether the call
    answered (False: unanswered), a modify's terms before it, whether it is a place sent
    again (3.1), the reads that missed it, and when to read again."""

    stage: Stage
    request: GttRequest
    seq: int
    trigger_id: str = ""
    answered: bool = True
    old_terms: GttTerms | None = None
    again: bool = False
    misses: int = 0
    next_ns: int = 0


class Step(enum.Enum):
    """What the sender does on a read (module docstring)."""

    WAIT = "wait"
    ANSWER = "answer"
    BIND = "bind"
    PLACE_AGAIN = "place_again"
    #: A place unanswered twice that no read found: never placed, as far as can be told.
    NOT_FOUND = "not_found"
    REPLACE = "replace"
    CHILD_CHECK = "child_check"


@dataclass(frozen=True)
class Verdict:
    """What a read says of a pending call: the step, the answer's kind (ANSWER), the GTT's
    status when it no longer rests (the follower judges it from then), the trigger ID to
    bind (BIND), and the words of the answer."""

    step: Step
    kind: GttAnswerKind | None = None
    seen: Seen | None = None
    trigger_id: str = ""
    message: str = ""


_WAIT = Verdict(Step.WAIT)


def judge(
    pending: Pending, rows: Mapping[str, GttRow] | None, bound: Callable[[str], bool]
) -> Verdict:
    """What `rows`, a read of the GTT book (None: it failed), says of `pending` (module
    docstring); `bound(trigger_id)` is True for a broker GTT the journal holds already.
    A read that misses what confirms the call is counted in `pending.misses`."""
    if rows is None:
        return _WAIT
    if pending.stage in (Stage.PLACE, Stage.REPLACE_PLACE):
        return _place(pending, rows, bound)
    if pending.stage is Stage.MODIFY:
        return _modify(pending, rows)
    return _cancel(pending, rows)


def _missed(pending: Pending) -> bool:
    """Count a read that missed; True while another read is due before deciding."""
    pending.misses += 1
    return pending.misses < MISSES_BEFORE_DECIDING


def _place(pending: Pending, rows: Mapping[str, GttRow], bound: Callable[[str], bool]) -> Verdict:
    request = pending.request
    if not pending.trigger_id:
        where = (request.symbol, request.exchange, request.product)
        found = [
            row.trigger_id
            for row in rows.values()
            if row.has_shape(*where, request.terms) and not bound(row.trigger_id)
        ]
        if len(found) == 1:
            return Verdict(Step.BIND, trigger_id=found[0])
        if found:
            words = f"GTTs {', '.join(found)} each have its shape, so it is not bound to one"
            return Verdict(Step.ANSWER, GttAnswerKind.UNRESOLVED, message=words)
        if _missed(pending):
            return _WAIT
        if not pending.again:
            return Verdict(Step.PLACE_AGAIN)
        words = "its place was unanswered twice and the GTT book shows neither"
        return Verdict(Step.NOT_FOUND, GttAnswerKind.UNRESOLVED, message=words)
    seen = seen_of(rows, pending.trigger_id)
    if seen.rests:
        return Verdict(Step.ANSWER, GttAnswerKind.PLACED)
    if seen is Seen.GONE and _missed(pending):
        return _WAIT
    return Verdict(Step.ANSWER, GttAnswerKind.PLACED, seen=seen)


def _modify(pending: Pending, rows: Mapping[str, GttRow]) -> Verdict:
    row = rows.get(pending.trigger_id)
    seen = Seen.GONE if row is None else row.seen
    if row is None or not seen.rests:
        words = f"the GTT is {seen.value} before its modify was confirmed"
        return Verdict(Step.ANSWER, GttAnswerKind.UNRESOLVED, seen=seen, message=words)
    if row.shows(pending.request.terms):
        return Verdict(Step.ANSWER, GttAnswerKind.MODIFIED)
    old = pending.old_terms
    if old is None or not row.shows(old):
        return Verdict(Step.REPLACE, message="the GTT book shows other terms than either")
    if _missed(pending):
        return _WAIT
    if pending.answered:
        return Verdict(Step.REPLACE, message="the GTT book still shows its old terms")
    words = "its modify was unanswered and the GTT book shows its old terms"
    return Verdict(Step.ANSWER, GttAnswerKind.MODIFY_REFUSED, message=words)


def _cancel(pending: Pending, rows: Mapping[str, GttRow]) -> Verdict:
    seen = seen_of(rows, pending.trigger_id)
    if seen in (Seen.GONE, Seen.ENDED):
        return Verdict(Step.CHILD_CHECK)
    if seen in (Seen.EXPIRED, Seen.REJECTED):
        return Verdict(Step.ANSWER, GttAnswerKind.CANCELLED)
    if seen is Seen.TRIGGERED:
        return Verdict(Step.ANSWER, GttAnswerKind.CANCEL_REFUSED, seen=seen, message="it fired")
    if _missed(pending):
        return _WAIT
    words = "the GTT book still shows it resting"
    return Verdict(Step.ANSWER, GttAnswerKind.CANCEL_REFUSED, message=words)


__all__ = [
    "MISSES_BEFORE_DECIDING",
    "READ_AGAIN_AFTER_S",
    "Pending",
    "Stage",
    "Step",
    "Verdict",
    "judge",
]
