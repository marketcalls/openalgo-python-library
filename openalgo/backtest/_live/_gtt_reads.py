"""The book reads of the order worker's GTT parts (20-positional-live 3.1 to 3.3, 7.1; the
lead's rulings N3a-1 and N3a-2): the GTT book with the broker's units of the legs read
just before it, the order book a fired GTT's child is looked for in, and the trade book a
missed child's price is read from.

**Positions first.** A GTT book read is preceded by a position-book read (and the
holdings, when a leg trades CNC: `_broker_units`): a GTT the book shows resting had not
fired when the positions were read, so those units are what its leg held before it
could fire, the baseline a GTT that is later gone is judged against (3.3). With them goes
what the session's own orders had filled on each leg by then, as the order path knows it,
whether those fills had settled and when the newest of them was heard (`_own_fills`,
`OrderPath.own_fills`): a later move of the leg's units is the GTT's only net of the
session's own fills since (an exit sold part of the leg while its stop rested), and units
read while they had not settled decide nothing yet (the position book and the order path's
words may not agree in time). They are waited for, until `UNITS_WAIT_S` after the later
of when the wait began and that newest fill (`BookRead.waited_out`); still not settled
then, the GTT parts give the answer that can send no second exit and no second GTT.

Every answer goes through the mode guard (the analyzer switch routes the books too) and
is told to the order path's server watch (`note`: OpenAlgo's own answer lifts the hold of
a silent server, as the order path's own book reads do); once the guard has locked
nothing more is read, and what was not read is None. A failed
read is None too, never taken as zero units or a GTT gone (7.1). The GTT book is read with
`status="all"`, so a broker that keeps history shows how a GTT ended; one that does not
shows it gone (3.1). Times are UTC nanoseconds, quantities signed whole units. Used by
the worker's thread only.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from typing import Any

from .._layout import NANOS_PER_SECOND
from .._products import CNC
from ..errors import LiveError
from ._answers import server_answer
from ._broker_units import broker_units
from ._gtt_book import GttRow, gtt_rows
from ._gtt_child import CHILD_SEARCH_S
from ._order_book import book_rows
from ._own_fills import POSITION_BOOK_LAG_S, LegKey, OwnFills
from ._types import GuardLike, Now

#: The GTT book's status filter that lists the GTTs that ended too, where a broker keeps
#: them (openalgo/gtt.py, `gttorderbook`).
ALL_GTTS = "all"
#: Seconds the GTT parts wait at most for a leg's units to settle (`_own_fills`) before
#: they give the answer that can send no second exit and no second GTT (the lead's ruling
#: L1 on N3a's review round 4), counted from the later of when the wait began and the
#: newest own fill on the leg: the `POSITION_BOOK_LAG_S` that fill takes to settle, and a
#: child search's `CHILD_SEARCH_S` more for an own order still working on the leg to end.
UNITS_WAIT_S = POSITION_BOOK_LAG_S + CHILD_SEARCH_S


@dataclass(frozen=True)
class BookRead:
    """A read of the GTT book: its rows by trigger ID (None when it could not be read),
    the broker's units of each leg asked for as read just before it (None for a leg whose
    books could not be read), when the GTT book's read began, the units the session's own
    orders had filled on each of those legs when the broker's were read, the legs whose
    own fills had settled then (`_own_fills`: only their units decide anything), and when
    the newest own fill on each leg was heard (a leg with none is left out)."""

    rows: Mapping[str, GttRow] | None
    read_ns: int
    units: Mapping[LegKey, int | None] = field(default_factory=dict)
    own: Mapping[LegKey, int] = field(default_factory=dict)
    settled: frozenset[LegKey] = frozenset()
    fills_ns: Mapping[LegKey, int] = field(default_factory=dict)

    def waited_out(self, leg: LegKey, since_ns: int) -> bool:
        """Whether a wait for `leg`'s units to settle that began at `since_ns` (UTC ns) is
        over at this read: `UNITS_WAIT_S` have passed since the later of `since_ns` and
        the newest own fill on the leg (module docstring)."""
        start_ns = max(since_ns, self.fills_ns.get(leg, since_ns))
        return self.read_ns - start_ns >= round(UNITS_WAIT_S * NANOS_PER_SECOND)


class GttReader:
    """The reads (module docstring) through `client` (the orders client), each answer
    checked by `guard` and told to `note` (`OrderPath.note_answer`); `own` is the order
    path's view of the session's own fills on each leg (`OrderPath.own_fills`), kept as
    the public `own` for the GTT parts; `calling` marks the worker busy for a call, `now`
    gives UTC nanoseconds."""

    def __init__(
        self,
        client: Any,
        *,
        guard: GuardLike,
        note: Callable[[Mapping[str, Any]], None],
        own: OwnFills,
        now: Now,
        calling: Callable[[], AbstractContextManager[None]],
    ) -> None:
        self._client = client
        self._guard = guard
        self._note = note
        self.own = own
        self._now = now
        self._calling = calling

    def gtt_book(self, legs: Iterable[LegKey]) -> BookRead:
        """The GTT book, with the units of `legs` read just before it and the session's
        own on them then, settled or not (module docstring). Raises JournalError when the
        unbound intents cannot be read."""
        wanted = set(legs)
        own = {leg: self.own.units(leg) for leg in wanted}
        # Judged before the positions are read: a fill heard just within the lag is not
        # settled for a book read a moment later.
        settled = frozenset(leg for leg in wanted if self.own.settled(leg))
        newest = {leg: self.own.newest_fill_ns(leg) for leg in wanted}
        fills_ns = {leg: heard for leg, heard in newest.items() if heard is not None}
        units = self.units(wanted)
        read_ns = self._now()
        answer = self._read("gttorderbook", status=ALL_GTTS)
        rows = None if answer is None else gtt_rows(answer)
        return BookRead(rows, read_ns, units, own, settled, fills_ns)

    def units(self, legs: Iterable[LegKey]) -> dict[LegKey, int | None]:
        """The broker's units of each of `legs` now (`_broker_units`), None for a leg
        whose books could not be read."""
        wanted = set(legs)
        if not wanted:
            return {}
        positions = self._read("positionbook")
        holdings = self._read("holdings") if any(leg[2] == CNC for leg in wanted) else None
        if positions is None:
            return dict.fromkeys(wanted)
        return {leg: broker_units(positions, holdings, *leg) for leg in wanted}

    def order_rows(self) -> list[Mapping[str, Any]] | None:
        """The order book's rows, or None when it could not be read."""
        answer = self._read("orderbook")
        if answer is None or answer.get("status") != "success":
            return None
        try:
            return book_rows(answer)
        except LiveError:
            return None

    def trade_book(self) -> Mapping[str, Any] | None:
        """The trade book's answer, or None when it was not read (the guard locked)."""
        return self._read("tradebook")

    def _read(self, name: str, **fields: Any) -> Mapping[str, Any] | None:
        if self._guard.locked:
            return None
        with self._calling():
            answer = server_answer(getattr(self._client, name)(**fields))
        self._note(answer)
        self._guard.check_response(answer)
        return None if self._guard.locked else answer


__all__ = ["ALL_GTTS", "UNITS_WAIT_S", "BookRead", "GttReader", "LegKey"]
