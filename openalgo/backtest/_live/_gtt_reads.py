"""The book reads of the order worker's GTT parts (20-positional-live 3.1 to 3.3, 7.1; the
lead's rulings N3a-1 and N3a-2): the GTT book with the broker's units of the legs read
just before it, the order book a fired GTT's child is looked for in, and the trade book a
missed child's price is read from.

**Positions first.** A GTT book read is preceded by a position-book read (and the
holdings, when a leg trades CNC: `_broker_units`): a GTT the book shows resting had not
fired when the positions were read, so those units are what its leg held before it
could fire, the baseline a GTT that is later gone is judged against (3.3). With them goes
what the session's own orders had filled on each leg by then, as the order path knows it
(`OrderPath.filled_units`): a later move of the leg's units is the GTT's only net of the
session's own fills since (an exit sold part of the leg while its stop rested).

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

from .._products import CNC
from ..errors import LiveError
from ._answers import server_answer
from ._broker_units import broker_units
from ._gtt_book import GttRow, gtt_rows
from ._order_book import book_rows
from ._types import GuardLike, Now

#: A leg as the GTT parts key its units: (symbol, exchange, product).
LegKey = tuple[str, str, str]
#: The GTT book's status filter that lists the GTTs that ended too, where a broker keeps
#: them (openalgo/gtt.py, `gttorderbook`).
ALL_GTTS = "all"


@dataclass(frozen=True)
class BookRead:
    """A read of the GTT book: its rows by trigger ID (None when it could not be read),
    the broker's units of each leg asked for as read just before it (None for a leg whose
    books could not be read), when the GTT book's read began, and the units the session's
    own orders had filled on each of those legs when the broker's were read."""

    rows: Mapping[str, GttRow] | None
    read_ns: int
    units: Mapping[LegKey, int | None] = field(default_factory=dict)
    own: Mapping[LegKey, int] = field(default_factory=dict)


class GttReader:
    """The reads (module docstring) through `client` (the orders client), each answer
    checked by `guard` and told to `note` (`OrderPath.note_answer`); `own(leg)` gives the
    units the session's own orders have filled on a leg now (`OrderPath.filled_units`),
    kept as the public `own` for the GTT parts; `calling` marks the worker busy for a
    call, `now` gives UTC nanoseconds."""

    def __init__(
        self,
        client: Any,
        *,
        guard: GuardLike,
        note: Callable[[Mapping[str, Any]], None],
        own: Callable[[LegKey], int],
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
        own on them then (module docstring)."""
        wanted = set(legs)
        own = {leg: self.own(leg) for leg in wanted}
        units = self.units(wanted)
        read_ns = self._now()
        answer = self._read("gttorderbook", status=ALL_GTTS)
        return BookRead(None if answer is None else gtt_rows(answer), read_ns, units, own)

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


__all__ = ["ALL_GTTS", "BookRead", "GttReader", "LegKey"]
