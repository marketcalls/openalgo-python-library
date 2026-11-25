"""What the order book and a GTT's leg's units say when a GTT call's confirmation needs
them (20-positional-live 3.1, 3.2; the lead's rulings N3a-1, and M2 on N3a's review round
5): the cancel of a GTT the GTT book no longer shows resting (`child_verdict`), and an
unanswered place before it is placed again (`again_verdict`). Pure: the order worker's GTT
sender (`_gtt_send`) reads the order book after a read of the GTT book whose verdict
(`_gtt_confirm`) is CHILD_CHECK or FIRED_CHECK, and acts on these.

**A cancel** (`child_verdict`). Any order that may be the GTT's child (`_gtt_child`) makes
it CANCEL_REFUSED: it fired. With none, a GTT a book with history rows shows cancelled
(ENDED) is CANCELLED. A GTT gone from the book (absent) may have fired just before the
cancel, whatever the cancel call answered (a broker may accept the cancel of a GTT that
has just fired and drop its row), so its leg's units decide with the order book, against
the GTT's reference (`_gtt_units`): with no reference it is CANCEL_REFUSED at once; a
settled read that shows the units moved by its units on its side makes it CANCEL_REFUSED;
it is CANCELLED only when `CHILD_READS_OF_A_GONE_GTT` order-book reads,
`_gtt_confirm.READ_AGAIN_AFTER_S` apart, found no child and a settled read shows no such
move. That read is never the first that showed the GTT gone, so its units were read after
the GTT went. A read made while the session's own fills on the leg had not settled
(`_own_fills`), or whose units could not be read, judges nothing, whatever its units
show: the leg is read again every `READ_AGAIN_AFTER_S` until a read is settled, and with
none `_gtt_reads.UNITS_WAIT_S` after the later of the first unsettled read and the newest
own fill on the leg (`BookRead.waited_out`), it is CANCEL_REFUSED.

**The one case left.** A position book that lags a broker fill of the child itself (not
the session's own fills, which the wait covers) by more than the child reads can still
confirm the cancel of a GTT that fired, its child not listed in the order book yet; the
reconciler then sees the child's fill as an extra fill (16-session-design 5.3).

**An unanswered place** (`again_verdict`) is placed again only when the order book holds no
order of its child's shape placed since it was sent: one there may be its child, the
place landed and fired (UNRESOLVED).

Quantities are whole units.
"""

from __future__ import annotations

from ._gtt_book import Seen
from ._gtt_child import Candidates
from ._gtt_confirm import WAIT, Pending, Step, Verdict
from ._gtt_reads import BookRead
from ._gtt_types import GttAnswerKind
from ._gtt_units import UNTOLD, UnitsSign

#: Order-book reads that must miss the child of a cancelled GTT gone from the book before
#: its cancel is confirmed (3.1, 3.2): a broker's order book can list a new order late.
CHILD_READS_OF_A_GONE_GTT = 2


def child_verdict(
    pending: Pending, seen: Seen | None, found: Candidates, sign: UnitsSign, read: BookRead
) -> Verdict:
    """What the order book says of a cancelled GTT `read` showed `seen` (GONE or ENDED):
    `found`, the orders that may be its child; `sign`, what its leg's units in `read` say
    of it against its reference (`GttFollower.units_of`; module docstring)."""
    if found.every:
        words = f"it fired: orders {', '.join(found.every)} may be its child"
        return Verdict(Step.ANSWER, GttAnswerKind.CANCEL_REFUSED, seen=seen, message=words)
    pending.child_reads += 1
    if seen is not Seen.GONE:
        return Verdict(Step.ANSWER, GttAnswerKind.CANCELLED)
    if sign is UnitsSign.UNSETTLED and not pending.units_waited_out(read):
        return WAIT
    words = _cancel_refusal(sign)
    if words:
        return Verdict(Step.ANSWER, GttAnswerKind.CANCEL_REFUSED, seen=seen, message=words)
    if pending.child_reads < CHILD_READS_OF_A_GONE_GTT:
        return WAIT
    return Verdict(Step.ANSWER, GttAnswerKind.CANCELLED)


def _cancel_refusal(sign: UnitsSign) -> str:
    """Why the cancel of a GTT gone from the book with no child found is refused, by what
    its leg's units say (module docstring), or "" when they show no move by its units:
    FIRED, no reference, or not settled by the end of the wait."""
    if sign is UnitsSign.FIRED:
        return "it fired: the position moved by its units, its child not in the order book yet"
    if sign in UNTOLD:
        return f"{UNTOLD[sign]}: whether it fired cannot be told"
    return ""


def again_verdict(pending: Pending, found: Candidates) -> Verdict:
    """What the order book says of an unanswered place before it is placed again: `found`,
    its orders of the child's shape placed since the place was sent (module docstring)."""
    if found.every:
        words = (
            f"its place was unanswered and orders {', '.join(found.every)} of its child's "
            "shape were placed since: it may have landed and fired, so it is not placed again"
        )
        return Verdict(Step.ANSWER, GttAnswerKind.UNRESOLVED, message=words)
    return Verdict(Step.PLACE_AGAIN)


__all__ = ["CHILD_READS_OF_A_GONE_GTT", "again_verdict", "child_verdict"]
