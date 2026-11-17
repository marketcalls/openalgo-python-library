"""The close's final book read (16-session-design 3.3 step 5; 5.4 "As built", rounds 7
and 9; unit 12), made by the reconciler (`reconcile.Reconciler`) for the close
(`_session_close`).

The read is asked for at once and taken in by the loop's protective steps, each waiting
as little as the loop waits, for at most the close's seconds. They take in the books and
the order news only: after the last lines no price reaches the engine and the stops
supervisor's duties do not run (`_stops_after`). Once the books came, the steps go on
within the same seconds until the order worker is quiet (`worker_quiet`): what the
comparisons handed over (a safe correction, an entry's cancel bound meanwhile) is
answered before the worker closes, and the order words it took during the read are taken
in.

A read whose trade book gave the kernel a fill of a leg's own orders does not judge that
leg (its position book was read first and may predate the fill: `_reconcile_findings`).
While the close's seconds allow, one more read is then made of those legs alone, so the
close judges them from books read after the fill; the legs the first read judged are not
compared again, so what it was the first to see stays as the close reports it (round 7).
A leg still unjudged when the seconds end is reported as such (`_reconcile_close`).

Times are UTC nanoseconds. Used by the reconciler only.
"""

from __future__ import annotations

from typing import Any

from .._layout import NANOS_PER_SECOND


def final_read(reconcile: Any, session: Any, wait_s: float) -> None:
    """The close's read of every leg, then, while `wait_s` seconds from now allow, one of
    the legs it left unjudged (module docstring). `reconcile` is the session's
    `reconcile.Reconciler`, whose `end` came first, `session` the `_session_loop.Session`.
    Nothing is read once the mode guard has locked or the order worker stopped reading."""
    if session.locked or reconcile.stopped:
        return
    until_ns = session.now() + round(wait_s * NANOS_PER_SECOND)
    _read(reconcile, session, None, until_ns)
    again = frozenset(t.name for t in session.traders if reconcile.unjudged(t.name))
    if again and session.now() < until_ns and not (session.locked or reconcile.stopped):
        _read(reconcile, session, again, until_ns)


def _read(reconcile: Any, session: Any, names: frozenset[str] | None, until_ns: int) -> None:
    """One read of the legs of strategies `names` (all for None), taken in until the
    worker is quiet or `until_ns`."""
    reconcile.ask_of(names)
    while reconcile.waiting and session.now() < until_ns:
        session.step(session.now(), protecting=True)
    while not (session.locked or worker_quiet(session)) and session.now() < until_ns:
        session.step(session.now(), protecting=True)


def worker_quiet(session: Any) -> bool:
    """Whether the order worker has executed every request handed to it that can still be
    sent (what it put back before the last lines is withdrawn, not counted:
    `OrderWorker.withdraw_put_back`), is in no call, and every order word it took and
    every message it posted have been taken. Read in this order: each is raised before
    the one ahead of it clears."""
    worker = session.parts.worker
    if worker.unfinished or worker.busy_since_ns is not None:
        return False
    return not (worker.raw_pending or session.inbox_waiting)


__all__ = ["final_read", "worker_quiet"]
