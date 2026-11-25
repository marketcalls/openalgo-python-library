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
the GTT cannot be read (`Seen.UNREADABLE`) is no miss: it says nothing of the GTT.

- **A place** is confirmed by a success with a trigger ID and the GTT resting in the read
  (active or transit; an unknown status counts as resting): PLACED. A GTT the read shows
  fired or ended is placed all the same, and the follower judges it from then; one the
  book does not show is read again until misses decide it gone. An unanswered place is
  looked for by its shape (symbol, exchange, product, side, trigger and units) among the
  resting rows no GTT of the journal holds: one found is bound; several are UNRESOLVED,
  never a guess. A row of its shape that ended is never the session's. None once misses
  decide: it is placed again once, then UNRESOLVED; but never while it may have landed
  and fired (UNRESOLVED: a second GTT could exit the leg twice). Three signs say it may
  have, since four of seven brokers list active GTTs only and a GTT that fired is simply
  gone from their books (3.1): a row of its shape no GTT of the journal holds has fired;
  the leg's units moved by its units on its side (`_gtt_units.judge`) against the leg's
  latest reference made before the place was sent (the lead's ruling M3 on N3a's review
  round 5: `GttFollower.placing`); or the order book holds an order of its child's shape
  placed since its send (FIRED_CHECK: the sender reads the order book,
  `_gtt_checks.again_verdict`).
  A fall is judged only on a settled read: once misses decide, a read made while the
  session's own fills on the leg had not settled (`_own_fills`) judges nothing, the leg
  is read again every `READ_AGAIN_AFTER_S`, and still not settled by
  `_gtt_reads.UNITS_WAIT_S` (`BookRead.waited_out`) it is UNRESOLVED. With no reference
  from before the place was sent, whether it landed and fired cannot be told: UNRESOLVED
  once misses decide, never placed again. A row on its symbol and exchange (or naming
  none) that no GTT of the journal holds and that cannot be read may be this place too:
  it is never placed again beside one (UNRESOLVED once misses decide); a row of another
  instrument says nothing of it.
- **A modify** is confirmed by the read showing its trigger and units, compared by value
  (MODIFIED). A book that leaves its row out is read again until misses decide, as for a
  place: a broker's book can drop a row from one read. The old terms once misses decide:
  an unanswered modify the server has finished with was not made (MODIFY_REFUSED); an
  answered one, or one the server may still apply, cannot be told. A modify that cannot
  be told, or that half-applied (other terms), is cancelled and the GTT placed again
  (REPLACE). A GTT that is no longer resting (shown so, or still left out once misses
  decide) is UNRESOLVED: the follower judges it.
- **A cancel** is confirmed by the GTT absent or ended without firing, with no child order
  found for it (CHILD_CHECK: the sender reads the order book, `child_verdict`); fired, it
  is CANCEL_REFUSED; still resting once misses decide, CANCEL_REFUSED. A GTT a book with
  history rows shows cancelled is CANCELLED once an order-book read finds no child; one
  gone from the book (absent) may have fired just before the cancel, and its leg's units
  decide with the order book (`_gtt_checks.child_verdict`, which states the rule and the
  one case it leaves).

Times are UTC nanoseconds, prices rupees, quantities whole units.
"""

from __future__ import annotations

import enum
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from ._gtt_book import GttRow, Seen, seen_of
from ._gtt_reads import UNITS_WAIT_S, BookRead
from ._gtt_types import GttAnswerKind, GttRequest
from ._gtt_units import UnitsSign
from ._journal_gtt_rows import GttTerms
from ._order_resolve import BROKER_WINDOW_NS, NOT_PLACED_AFTER_NS

#: Seconds between two reads while a call waits for its confirmation (3.1: "two reads 2
#: seconds apart").
READ_AGAIN_AFTER_S = 2.0
#: Reads that miss a GTT, or show a modify's old terms or a cancelled GTT resting, once
#: misses count, before the confirmation decides (3.1).
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
    answered (False: unanswered) and whether the server has finished with it, a modify's
    terms before it, whether it is a place sent again (3.1), when it was sent, from when a
    read that misses counts (`misses_from_ns`), the reads that missed it, the order-book
    reads that found no child of a cancelled GTT, and when to read again; and when the
    first read that found the leg's units not settled was made (None: none has), from
    which the wait for them is bounded (`units_waited_out`)."""

    stage: Stage
    request: GttRequest
    seq: int
    trigger_id: str = ""
    answered: bool = True
    old_terms: GttTerms | None = None
    again: bool = False
    finished: bool = True
    sent_ns: int = 0
    misses_from_ns: int = 0
    misses: int = 0
    child_reads: int = 0
    next_ns: int = 0
    units_wait_ns: int | None = None

    def units_waited_out(self, read: BookRead) -> bool:
        """Whether the wait for the leg's units to settle, which begins at `read` unless it
        began before, is over at `read` (`BookRead.waited_out`)."""
        request = self.request
        if self.units_wait_ns is None:
            self.units_wait_ns = read.read_ns
        leg = (request.symbol, request.exchange, request.product)
        return read.waited_out(leg, self.units_wait_ns)


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
    #: Read the order book for a child of an unanswered place before it is placed again.
    FIRED_CHECK = "fired_check"


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


#: The verdict that reads again.
WAIT = Verdict(Step.WAIT)


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
    pending: Pending, read: BookRead, bound: Callable[[str], bool], fall: UnitsSign
) -> Verdict:
    """What `read`, a read of the GTT book (its rows None when it failed), says of
    `pending` (module docstring); `bound(trigger_id)` is True for a broker GTT the journal
    holds already, and `fall` is what the leg's units in `read` say of the place's GTT
    against its reference (`GttFollower.units_of`), which only an unanswered place reads.
    A read that misses what confirms the call is counted in `pending.misses` once misses
    count."""
    rows = read.rows
    if rows is None:
        return WAIT
    if pending.stage in (Stage.PLACE, Stage.REPLACE_PLACE):
        return _place(pending, read, rows, bound, fall)
    if pending.stage is Stage.MODIFY:
        return _modify(pending, rows, read.read_ns)
    return _cancel(pending, rows, read.read_ns)


def _missed(pending: Pending, read_ns: int) -> bool:
    """Count a read that missed, once misses count; True while another read is due before
    deciding."""
    if read_ns >= pending.misses_from_ns:
        pending.misses += 1
    return pending.misses < MISSES_BEFORE_DECIDING


def _place(
    pending: Pending,
    read: BookRead,
    rows: Mapping[str, GttRow],
    bound: Callable[[str], bool],
    fall: UnitsSign,
) -> Verdict:
    if not pending.trigger_id:
        return _unanswered_place(pending, read, rows, bound, fall)
    seen = seen_of(rows, pending.trigger_id)
    if seen.rests:
        return Verdict(Step.ANSWER, GttAnswerKind.PLACED)
    if seen is Seen.UNREADABLE or (seen is Seen.GONE and _missed(pending, read.read_ns)):
        return WAIT
    return Verdict(Step.ANSWER, GttAnswerKind.PLACED, seen=seen)


def _unanswered_place(
    pending: Pending,
    read: BookRead,
    rows: Mapping[str, GttRow],
    bound: Callable[[str], bool],
    fall: UnitsSign,
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
    if _missed(pending, read.read_ns):
        return WAIT
    fired = [row.trigger_id for row in shaped if row.seen is Seen.TRIGGERED]
    unread = [row.trigger_id for row in rows.values() if _may_be(request, row, bound)]
    if fired or unread or fall is UnitsSign.FIRED:
        return Verdict(Step.ANSWER, GttAnswerKind.UNRESOLVED, message=_may_have(fired, unread))
    if fall is UnitsSign.NO_REFERENCE:
        words = (
            "its place was unanswered and no settled read of its leg's units was made before "
            "it was sent: whether it landed and fired cannot be told, so it is not placed again"
        )
        return Verdict(Step.ANSWER, GttAnswerKind.UNRESOLVED, message=words)
    if fall is UnitsSign.UNSETTLED:
        if not pending.units_waited_out(read):
            return WAIT
        words = (
            "its place was unanswered and the session's own orders on its leg had not "
            f"settled {UNITS_WAIT_S:g} s on: whether it landed and fired cannot be told, so "
            "it is not placed again"
        )
        return Verdict(Step.ANSWER, GttAnswerKind.UNRESOLVED, message=words)
    if pending.again:
        words = "its place was unanswered twice and the GTT book shows neither"
        return Verdict(Step.NOT_FOUND, GttAnswerKind.UNRESOLVED, message=words)
    return Verdict(Step.FIRED_CHECK)


def _may_be(request: GttRequest, row: GttRow, bound: Callable[[str], bool]) -> bool:
    """Whether `row`, which cannot be read, may be the GTT of an unanswered place
    `request`: no GTT of the journal holds it, and it names the request's symbol and
    exchange, or none."""
    if row.seen is not Seen.UNREADABLE or bound(row.trigger_id):
        return False
    return row.symbol in ("", request.symbol) and row.exchange in ("", request.exchange)


def _may_have(fired: list[str], unread: list[str]) -> str:
    """Why an unanswered place may have landed and fired (module docstring)."""
    if fired:
        return (
            f"its place was unanswered and GTT {', '.join(fired)} of its shape has fired: "
            "it may be this place, so it is not placed again"
        )
    if unread:
        return (
            f"its place was unanswered and GTT {', '.join(unread)} on its instrument cannot "
            "be read: it may be this place, so it is not placed again"
        )
    return (
        "its place was unanswered and the position fell by its units since: it may have "
        "landed and fired, so it is not placed again"
    )


def _modify(pending: Pending, rows: Mapping[str, GttRow], read_ns: int) -> Verdict:
    row = rows.get(pending.trigger_id)
    seen = Seen.GONE if row is None else row.seen
    if seen is Seen.UNREADABLE or (row is None and _missed(pending, read_ns)):
        return WAIT
    if row is None or not seen.rests:
        words = f"the GTT is {seen.value} before its modify was confirmed"
        return Verdict(Step.ANSWER, GttAnswerKind.UNRESOLVED, seen=seen, message=words)
    if row.shows(pending.request.terms):
        return Verdict(Step.ANSWER, GttAnswerKind.MODIFIED)
    old = pending.old_terms
    if old is None or not row.shows(old):
        return Verdict(Step.REPLACE, message="the GTT book shows other terms than either")
    if _missed(pending, read_ns):
        return WAIT
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
        return WAIT
    if seen in (Seen.GONE, Seen.ENDED):
        return Verdict(Step.CHILD_CHECK, seen=seen)
    if seen in (Seen.EXPIRED, Seen.REJECTED):
        return Verdict(Step.ANSWER, GttAnswerKind.CANCELLED)
    if seen is Seen.TRIGGERED:
        return Verdict(Step.ANSWER, GttAnswerKind.CANCEL_REFUSED, seen=seen, message="it fired")
    if _missed(pending, read_ns):
        return WAIT
    words = "the GTT book still shows it resting"
    return Verdict(Step.ANSWER, GttAnswerKind.CANCEL_REFUSED, message=words)


__all__ = [
    "MISSES_BEFORE_DECIDING",
    "READ_AGAIN_AFTER_S",
    "WAIT",
    "Pending",
    "Stage",
    "Step",
    "Verdict",
    "judge",
    "misses_from_ns",
]
