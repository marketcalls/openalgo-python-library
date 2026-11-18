"""The invariants of a live session on the fake OpenAlgo (16-session-design 7.1, unit 14),
checked after every session test: `rig` (session_samples) records each fake's order changes
and session runs (`session_trace`, `session_hooks`) and has the fixture run
`assert_session_invariants` when the test ends. A test that cannot keep one check waives
that check alone, with its reason (`SessionTrace.waive`); every other check still runs.

At every change of an order at the fake, as it happened (`session_invariants_changes`):

1. **One stop.** At most one stop of the session's works per leg.
2. **Exits within the position.** The session's working exits never hold more units than
   the position; the exits a sale by hand overtook, by no more than the units sold, until
   they fit the position again.
3. **Never the other side.** No fill of the session's orders takes the position to the
   other side of the leg.

When the last run ends (its close returned, or the test ended after a crash), of that run
(`session_invariants_end`):

4. **No orphan.** Every order of the session's working at the fake is a working kernel
   order of the run, and no stop works for a flat position.
5. **Protected.** Each position the fake holds has a confirmed working stop covering it,
   unless its leg has no stop rule or the test leaves it unprotected by a row 7.1 lists
   (F35, F36, F60, the F77 variant, F80) or a decided end, which the run then names.
6. **Books agree.** The engine's units and the fake's are equal, or the run's
   reconciliations or the close's named legs hold the difference.

Over each run's event stream and the fake's log (`session_invariants_stream`):

7. **Every Modify and Cancel answered.**
8. **Nothing after a guard lock.**
9. **Priced fills.**

A sale by hand is the one thing a session learns of only by its comparisons, so what one
leaves at the fake beyond 1 to 6 stands only by a decided end the test names
(`SessionTrace.kept_stop_by_decision`, `SessionTrace.oversold_by_decision`), each such
test listed in FAULT_ROWS.md for the lead.

Quantities are signed units, prices rupees, times UTC nanoseconds.
"""

from __future__ import annotations

from fake_openalgo import FakeOpenAlgo
from session_invariants_changes import one_stop_per_leg, sides_of, within_the_position
from session_invariants_end import (
    at_the_end,
    books_agree,
    closing_now,
    no_orphan,
    working_at_the_end,
)
from session_invariants_stream import (
    modifies_and_cancels_answered,
    nothing_after_a_guard_lock,
    priced_fills,
)
from session_trace import (
    ANSWERED,
    BOOKS_AGREE,
    GUARD_LOCK,
    NO_ORPHAN,
    ONE_STOP,
    OTHER_SIDE,
    PRICED,
    WITHIN,
    SessionTrace,
)


class InvariantBroken(AssertionError):
    """An invariant of 16-session-design 7.1 does not hold; the message names it."""


def assert_session_invariants(session: SessionTrace, fake: FakeOpenAlgo) -> None:
    """Check every invariant of the module docstring over `session`, the trace of the
    sessions run on `fake`, but those the test waived. Raises InvariantBroken naming the
    first one broken."""
    if not session.runs:
        return
    try:
        _check(session, fake)
    except InvariantBroken:
        raise
    except AssertionError as error:
        raise InvariantBroken(str(error)) from None


def _check(session: SessionTrace, fake: FakeOpenAlgo) -> None:
    waived = session.waived
    mine = session.session_clients()
    last = session.runs[-1]
    # The changes up to the last run's end: what a test does after it (an operator's
    # orders once the session has ended) is not the session's.
    changes = session.changes[: last.ended_at]
    own = [c for c in changes if c.client in mine]
    sides = sides_of(own)
    if ONE_STOP not in waived:
        one_stop_per_leg(changes, mine)
    rooms = within_the_position(
        changes,
        mine,
        sides,
        session.oversold,
        within=WITHIN not in waived,
        side=OTHER_SIDE not in waived,
    )
    closing = last.closing or closing_now(last)
    working = working_at_the_end(own)
    held = {c.instrument: c.position for c in changes}
    if NO_ORPHAN not in waived:
        no_orphan(last, closing, working, held, rooms, session.kept_stops)
    at_the_end(session, last, closing, working, held, sides, rooms)
    if BOOKS_AGREE not in waived and last.outcome is not None:
        books_agree(last, closing, held)
    for at, run in enumerate(session.runs):
        if ANSWERED not in waived:
            modifies_and_cancels_answered(run)
        if GUARD_LOCK not in waived:
            end = run.log_to
            if end is None and at + 1 < len(session.runs):
                end = session.runs[at + 1].log_from
            nothing_after_a_guard_lock(run, fake, end)
        if PRICED not in waived:
            priced_fills(run, fake)


__all__ = ["InvariantBroken", "assert_session_invariants"]
