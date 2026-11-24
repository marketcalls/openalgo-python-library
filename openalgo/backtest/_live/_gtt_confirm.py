"""What a read of the GTT book says of a GTT call that waits for its confirmation
(20-positional-live 3.1, "Confirmations" and "Unanswered calls"; the lead's ruling N3a-1).
The order worker's GTT sender (`_gtt_send`) reads the book at once after each call and
every `READ_AGAIN_AFTER_S` while one waits (`Pending`), and acts on the `Verdict` of
`judge`; nothing else is sent for that GTT meanwhile.

**When a read counts as a miss** (`misses_from_ns`, the order path's rule for an unbound
intent, `_order_resolve`). A read that misses what confirms a call counts only from 5 s
after its answer when the server has finished with it (a broker's book can show a change a
few seconds late), and from its send plus 150 s when the server may still be working on it
(a client timeout, a dropped connection, a 5xx page): until then the call can still land
at the broker, and deciding sooner would place a second GTT beside it. `MISSES_BEFORE_DECIDING`
reads that count decide. An unanswered modify the server may still apply counts at once:
it is cancelled and placed again, which a late modify cannot undo. A read whose row of
the GTT cannot be read (`Seen.UNREADABLE`), or for an unanswered place one with a row no
GTT of the journal holds that cannot be read (it may be this place), is no miss: it says
nothing of the GTT.

- **A place** is confirmed by a success with a trigger ID and the GTT resting in the read
  (active or transit; an unknown status counts as resting): PLACED. A GTT the read shows
  fired or ended is placed all the same, and the follower judges it from then; one the
  book does not show is read again until misses decide it gone. An unanswered place is
  looked for by its shape (symbol, exchange, product, side, trigger and units) among the
  resting rows no GTT of the journal holds: one found is bound; several are UNRESOLVED,
  never a guess. A row of its shape that ended is never the session's. None once misses
  decide: it is placed again once, then UNRESOLVED; but not when a row of its shape that
  no GTT of the journal holds has fired, which may be this place, landed and fired
  (UNRESOLVED: a second GTT could exit the leg twice).
- **A modify** is confirmed by the read showing its trigger and units, compared by value
  (MODIFIED). The old terms once misses decide: an unanswered modify the server has
  finished with was not made (MODIFY_REFUSED); an answered one, or one the server may
  still apply, cannot be told. A modify that cannot be told, or that half-applied (other
  terms), is cancelled and the GTT placed again (REPLACE). A GTT that is no longer
  resting is UNRESOLVED: the follower judges it.
- **A cancel** is confirmed by the GTT absent or ended without firing, with no child order
  found for it (CHILD_CHECK: the sender reads the order book, `child_verdict`); fired, it
  is CANCEL_REFUSED; still resting once misses decide, CANCEL_REFUSED. A GTT gone from the
  book (not shown cancelled) may have fired just before the cancel: the leg's units moved
  by its units say it fired, and otherwise the order book is read `CHILD_READS_OF_A_GONE_GTT`
  times, `READ_AGAIN_AFTER_S` apart, before the cancel is confirmed.

Times are UTC nanoseconds, prices rupees, quantities whole units.
"""

from __future__ import annotations

import enum
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from ._gtt_book import GttRow, Seen, seen_of
from ._gtt_child import Candidates
from ._gtt_types import GttAnswerKind, GttRequest
from ._journal_gtt_rows import GttTerms
from ._order_resolve import BROKER_WINDOW_NS, NOT_PLACED_AFTER_NS

#: Seconds between two reads while a call waits for its confirmation (3.1: "two reads 2
#: seconds apart").
READ_AGAIN_AFTER_S = 2.0
#: Reads that miss a GTT, or show a modify's old terms or a cancelled GTT resting, once
#: misses count, before the confirmation decides (3.1).
MISSES_BEFORE_DECIDING = 2
#: Order-book reads that must miss the child of a cancelled GTT gone from the book before
#: its cancel is confirmed (3.1, 3.2): a broker's order book can list a new order late.
CHILD_READS_OF_A_GONE_GTT = 2


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
    answered (False: unanswered) and whether the server has finished with it, a modify's
    terms before it, whether it is a place sent again (3.1), from when a read that misses
    counts (`misses_from_ns`), the reads that missed it, the order-book reads that found no
    child of a cancelled GTT, and when to read again."""

    stage: Stage
    request: GttRequest
    seq: int
    trigger_id: str = ""
    answered: bool = True
    old_terms: GttTerms | None = None
    again: bool = False
    finished: bool = True
    misses_from_ns: int = 0
    misses: int = 0
    child_reads: int = 0
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
    status when it no longer rests (the follower judges it from then; for CHILD_CHECK,
    gone or ended), the trigger ID to bind (BIND), and the words of the answer."""

    step: Step
    kind: GttAnswerKind | None = None
    seen: Seen | None = None
    trigger_id: str = ""
    message: str = ""


_WAIT = Verdict(Step.WAIT)


def misses_from_ns(stage: Stage, *, finished: bool, sent_ns: int, answered_ns: int) -> int:
    """From when (UTC ns) a read that misses counts for a call of `stage` sent at
    `sent_ns` and answered at `answered_ns`, the server `finished` with it or not (module
    docstring)."""
    if finished:
        return answered_ns + NOT_PLACED_AFTER_NS
    if stage is Stage.MODIFY:
        return answered_ns
    return sent_ns + BROKER_WINDOW_NS


def judge(
    pending: Pending,
    rows: Mapping[str, GttRow] | None,
    bound: Callable[[str], bool],
    read_ns: int,
) -> Verdict:
    """What `rows`, a read of the GTT book begun at `read_ns` (None: it failed), says of
    `pending` (module docstring); `bound(trigger_id)` is True for a broker GTT the journal
    holds already. A read that misses what confirms the call is counted in
    `pending.misses` once misses count."""
    if rows is None:
        return _WAIT
    if pending.stage in (Stage.PLACE, Stage.REPLACE_PLACE):
        return _place(pending, rows, bound, read_ns)
    if pending.stage is Stage.MODIFY:
        return _modify(pending, rows, read_ns)
    return _cancel(pending, rows, read_ns)


def child_verdict(pending: Pending, seen: Seen | None, found: Candidates, moved: bool) -> Verdict:
    """What the order book says of a cancelled GTT a read showed `seen` (GONE or ENDED):
    `found`, the orders that may be its child; `moved`, whether the leg's units moved by
    the GTT's units since the last read that showed it resting (module docstring)."""
    if found.every:
        words = f"it fired: orders {', '.join(found.every)} may be its child"
        return Verdict(Step.ANSWER, GttAnswerKind.CANCEL_REFUSED, seen=seen, message=words)
    if seen is Seen.GONE and moved:
        words = "it fired: the position moved by its units, its child not in the order book yet"
        return Verdict(Step.ANSWER, GttAnswerKind.CANCEL_REFUSED, seen=seen, message=words)
    pending.child_reads += 1
    if seen is Seen.GONE and pending.child_reads < CHILD_READS_OF_A_GONE_GTT:
        return _WAIT
    return Verdict(Step.ANSWER, GttAnswerKind.CANCELLED)


def _missed(pending: Pending, read_ns: int) -> bool:
    """Count a read that missed, once misses count; True while another read is due before
    deciding."""
    if read_ns >= pending.misses_from_ns:
        pending.misses += 1
    return pending.misses < MISSES_BEFORE_DECIDING


def _place(
    pending: Pending, rows: Mapping[str, GttRow], bound: Callable[[str], bool], read_ns: int
) -> Verdict:
    if not pending.trigger_id:
        return _unanswered_place(pending, rows, bound, read_ns)
    seen = seen_of(rows, pending.trigger_id)
    if seen.rests:
        return Verdict(Step.ANSWER, GttAnswerKind.PLACED)
    if seen is Seen.UNREADABLE or (seen is Seen.GONE and _missed(pending, read_ns)):
        return _WAIT
    return Verdict(Step.ANSWER, GttAnswerKind.PLACED, seen=seen)


def _unanswered_place(
    pending: Pending, rows: Mapping[str, GttRow], bound: Callable[[str], bool], read_ns: int
) -> Verdict:
    request = pending.request
    where = (request.symbol, request.exchange, request.product)
    shaped = [
        row
        for row in rows.values()
        if row.has_shape(*where, request.terms) and not bound(row.trigger_id)
    ]
    found = [row.trigger_id for row in shaped if row.seen.rests]
    if len(found) == 1:
        return Verdict(Step.BIND, trigger_id=found[0])
    if found:
        words = f"GTTs {', '.join(found)} each have its shape, so it is not bound to one"
        return Verdict(Step.ANSWER, GttAnswerKind.UNRESOLVED, message=words)
    unread = any(row.seen is Seen.UNREADABLE and not bound(row.trigger_id) for row in rows.values())
    if unread or _missed(pending, read_ns):
        return _WAIT
    fired = [row.trigger_id for row in shaped if row.seen is Seen.TRIGGERED]
    if fired:
        words = (
            f"its place was unanswered and GTT {', '.join(fired)} of its shape has fired: "
            "it may be this place, so it is not placed again"
        )
        return Verdict(Step.ANSWER, GttAnswerKind.UNRESOLVED, message=words)
    if not pending.again:
        return Verdict(Step.PLACE_AGAIN)
    words = "its place was unanswered twice and the GTT book shows neither"
    return Verdict(Step.NOT_FOUND, GttAnswerKind.UNRESOLVED, message=words)


def _modify(pending: Pending, rows: Mapping[str, GttRow], read_ns: int) -> Verdict:
    row = rows.get(pending.trigger_id)
    seen = Seen.GONE if row is None else row.seen
    if seen is Seen.UNREADABLE:
        return _WAIT
    if row is None or not seen.rests:
        words = f"the GTT is {seen.value} before its modify was confirmed"
        return Verdict(Step.ANSWER, GttAnswerKind.UNRESOLVED, seen=seen, message=words)
    if row.shows(pending.request.terms):
        return Verdict(Step.ANSWER, GttAnswerKind.MODIFIED)
    old = pending.old_terms
    if old is None or not row.shows(old):
        return Verdict(Step.REPLACE, message="the GTT book shows other terms than either")
    if _missed(pending, read_ns):
        return _WAIT
    if pending.answered:
        return Verdict(Step.REPLACE, message="the GTT book still shows its old terms")
    if not pending.finished:
        words = "its modify was unanswered and the server may still apply it"
        return Verdict(Step.REPLACE, message=words)
    words = "its modify was unanswered and the GTT book shows its old terms"
    return Verdict(Step.ANSWER, GttAnswerKind.MODIFY_REFUSED, message=words)


def _cancel(pending: Pending, rows: Mapping[str, GttRow], read_ns: int) -> Verdict:
    seen = seen_of(rows, pending.trigger_id)
    if seen is Seen.UNREADABLE:
        return _WAIT
    if seen in (Seen.GONE, Seen.ENDED):
        return Verdict(Step.CHILD_CHECK, seen=seen)
    if seen in (Seen.EXPIRED, Seen.REJECTED):
        return Verdict(Step.ANSWER, GttAnswerKind.CANCELLED)
    if seen is Seen.TRIGGERED:
        return Verdict(Step.ANSWER, GttAnswerKind.CANCEL_REFUSED, seen=seen, message="it fired")
    if _missed(pending, read_ns):
        return _WAIT
    words = "the GTT book still shows it resting"
    return Verdict(Step.ANSWER, GttAnswerKind.CANCEL_REFUSED, message=words)


__all__ = [
    "CHILD_READS_OF_A_GONE_GTT",
    "MISSES_BEFORE_DECIDING",
    "READ_AGAIN_AFTER_S",
    "Pending",
    "Stage",
    "Step",
    "Verdict",
    "child_verdict",
    "judge",
    "misses_from_ns",
]
