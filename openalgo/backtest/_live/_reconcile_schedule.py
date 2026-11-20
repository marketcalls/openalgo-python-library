"""When the reconciler (`reconcile`) reads the broker's books (16-session-design 5.1;
unit 12, split from the reconciler in unit M6).

**Schedule** (5.1): a read when trading begins, every `RECONCILE_EVERY_S`,
`RECONCILE_AFTER_FILL_S` after a fill, of the session's orders or, on an instrument it
trades, another client's (`other_fill`: a sale by hand is seen within seconds, 7.1 "As
built (unit 14"), `RECONCILE_AGAIN_AFTER_S` after a comparison that
found a difference the one before it did not (the second confirms it; one seen twice in a
row waits for the schedule), `SQUARE_OFF_READ_BEFORE_S` before and
`SQUARE_OFF_READ_AFTER_S` after each strategy's square-off, when the safe stop's flatten
begins (`read_before_exits`), and at the common close (`_reconcile_final`); never after
the mode guard locked. One read at a time, numbered: the order worker reads the position,
order and trade books and the funds (`ReadBooks`) and answers `Books`. Before the flatten
and the square-off the exit guard holds each leg's exits until a read begun after the
request has compared it (a sale by hand since the last read is not known before), read
again every `RECONCILE_AGAIN_AFTER_S` while a leg cannot be compared.

Times are UTC nanoseconds. Used by the loop only.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from .._layout import NANOS_PER_SECOND
from ._session_trader import Trader
from ._worker_queue import ReadBooks

#: Seconds between two reads (08-live-trading section 7).
RECONCILE_EVERY_S = 30.0
#: Seconds after the last fill, and after a comparison that differed, before a read.
RECONCILE_AFTER_FILL_S = 3.0
RECONCILE_AGAIN_AFTER_S = 3.0
#: Seconds after a strategy's square-off before the books are read: the session's own
#: check that every leg is flat then (16-session-design 3.4).
SQUARE_OFF_READ_AFTER_S = 30.0
#: Seconds before a strategy's square-off at which the books are read, so its exit knows
#: what the broker holds (5.4 "As built"): a read of the four books takes a second or two.
SQUARE_OFF_READ_BEFORE_S = 5.0
#: Seconds after which a read asked for and never answered is asked for again: well past
#: the orders client's own wait (the SDK's 120 s), as for the kill switch's reads.
RECONCILE_READ_LOST_AFTER_S = 300.0

_NS = NANOS_PER_SECOND


class ReadSchedule:
    """The schedule of a `reconcile.Reconciler`'s reads (module docstring). The
    reconciler holds the session (`_session`), its exit guard (`guard`), the read's
    number and state (`_generation`, `_asked_ns`, `_answered`, `_stopped`), the
    instants due (`_next_ns`, `_soon_ns`, `_square_offs`), each leg's quiet time
    (`_quiet`) and the read's filter (`_only`); its step D (`duties`) notes each leg's
    flight and asks for a read when one is due."""

    _session: Any
    guard: Any
    _generation: int
    _asked_ns: int | None
    _answered: bool
    _stopped: bool
    _next_ns: int | None
    _soon_ns: int | None
    _square_offs: list[int]
    _quiet: dict[str, int | None]
    _only: frozenset[str] | None

    def filled(self, trader: Trader, at_ns: int) -> None:
        """A fill of `trader`'s reached the kernel at `at_ns`: a read is due shortly, and
        only a read begun after it compares the leg."""
        if self._quiet.get(trader.name) is not None:
            self._quiet[trader.name] = at_ns
        self._soon(at_ns + _s(RECONCILE_AFTER_FILL_S))

    def other_fill(self, at_ns: int) -> None:
        """Another client's fill on an instrument the session trades reached the order
        stream at `at_ns`: a read is due as after the session's own fills, so a sale by
        hand is seen within seconds instead of at the next scheduled read."""
        self._soon(at_ns + _s(RECONCILE_AFTER_FILL_S))

    def not_read(self) -> None:
        """The worker did not read the books (the guard locked): nothing more is read."""
        self._stopped = True

    @property
    def waiting(self) -> bool:
        """True while a read asked for has not come."""
        return not self._answered and not self._stopped

    @property
    def stopped(self) -> bool:
        """True once the worker did not read the books (`not_read`)."""
        return self._stopped

    # ----- when a read is due -----------------------------------------------------------

    def read_before_exits(self, names: Iterable[str] | None = None) -> None:
        """The exits of strategies `names` (all for None) wait for a comparison of a read
        asked for now (module docstring): the safe stop's flatten and the square-off. A
        finished kernel session is never compared again, so it waits for nothing. The
        read is asked for in this step, so the exits wait one read and no more; while one
        is asked for already, its answer asks for the next at once when it began earlier
        (`books`)."""
        session = self._session
        now_ns = session.now()
        for trader in session.traders:
            if not trader.finished and (names is None or trader.name in names):
                self.guard.await_read(trader.name, now_ns)
        if not self._answered:
            return
        self._soon(now_ns)
        if self._reads(session):
            self._ask_when_due(now_ns)

    def ask(self, now_ns: int) -> None:
        """Ask the order worker for a read now, of every leg."""
        self._only = None
        self._generation += 1
        self._asked_ns, self._answered = now_ns, False
        self._next_ns, self._soon_ns = now_ns + _s(RECONCILE_EVERY_S), None
        self._square_offs = [instant for instant in self._square_offs if instant > now_ns]
        self._session.hand_over([ReadBooks(self._generation)])

    def ask_of(self, names: frozenset[str] | None) -> None:
        """Ask for a read now that compares the legs of strategies `names` alone (all for
        None), each out of flight from now: the close's reads (`_reconcile_final`), after
        the last lines, when no duty notes flight."""
        now_ns = self._session.now()
        traders = self._session.traders
        self._quiet.update({t.name: now_ns for t in traders if names is None or t.name in names})
        self.ask(now_ns)
        self._only = names

    def _reads(self, session: Any) -> bool:
        """Whether reads are made: once trading began, never after a guard lock."""
        return session.trading and not (session.locked or self._stopped)

    def _ask_when_due(self, now_ns: int) -> None:
        # While OpenAlgo is silent the order worker's probe asks it (4.9): a read could
        # hold the worker for the client's whole timeout, ahead of the stops and exits.
        if self._session.parts.worker.silent_since_ns is not None:
            return
        if self._answered:
            if self._due(now_ns):
                self.ask(now_ns)
        elif self._asked_ns is None or now_ns >= self._asked_ns + _s(RECONCILE_READ_LOST_AFTER_S):
            self.ask(now_ns)

    def _due(self, now_ns: int) -> bool:
        due = [self._next_ns if self._next_ns is not None else now_ns]
        due += [self._soon_ns] if self._soon_ns is not None else []
        due += [instant for instant in self._square_offs if instant <= now_ns]
        return now_ns >= min(due)

    def _soon(self, due_ns: int) -> None:
        self._soon_ns = due_ns if self._soon_ns is None else min(self._soon_ns, due_ns)


def _s(seconds: float) -> int:
    return round(seconds * _NS)


__all__ = [
    "RECONCILE_AFTER_FILL_S",
    "RECONCILE_AGAIN_AFTER_S",
    "RECONCILE_EVERY_S",
    "RECONCILE_READ_LOST_AFTER_S",
    "SQUARE_OFF_READ_AFTER_S",
    "SQUARE_OFF_READ_BEFORE_S",
    "ReadSchedule",
]
