"""The book reads the order worker makes for the loop (16-session-design 2.4, 5.2, 4.11),
split from `_worker_send` so each module keeps one job: the books of a reconciliation
(`ReadBooks`, posted as `inbox.Books`), and the position book of a held cancel, with the
order book after it when asked for (`ReadPositions`, posted as `inbox.PositionRows`).

Each answer goes to the mode guard as it comes (the analyzer switch routes the books too);
once the guard locks nothing more is read and the request is `NotSent(locked)`. A session
with a CNC strategy reads the holdings right after the position book (`_broker_units`: a
CNC leg's units are in both). The position book and holdings of each reconciliation read,
the start's among them, are handed to `positions_read` as they come (the GTT follower's
references, `_gtt_follow.GttFollower.reference`). Times are UTC nanoseconds. Used by the
worker's thread only.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import Any

from ._answers import server_answer
from ._worker_queue import ReadBooks, ReadPositions
from .guard import ModeGuard
from .inbox import Books, Message, NotSent, NotSentReason, PositionRows

#: The books a reconciliation reads, as `inbox.Books` names them (5.2).
_BOOKS = ("positionbook", "orderbook", "tradebook", "funds")
#: The books of a held cancel's read, in order: the position book, then the order book
#: when asked for (`ReadPositions.orders`).
_HELD_CANCEL_BOOKS = ("positionbook", "orderbook")
#: The holdings, read right after the position book when the session reads them: before
#: the trade book, so a fill that lands during a read shows in the trade book as it would
#: in the position book (16-session-design 5.4, round 9).
_HOLDINGS = "holdings"


class BookReads:
    """The reads (module docstring) through `client` (the orders client), each answer
    checked by `guard`; `post` puts a message on the loop's inbox, `calling` marks the
    worker busy for a call, `now` gives UTC nanoseconds; `read_holdings` is True for a
    session with a CNC strategy; `positions_read(positionbook, holdings, read_ns)` is told
    each reconciliation read's position book, holdings (None when not read) and the time
    its read began (UTC ns), before the read is posted."""

    def __init__(
        self,
        client: Any,
        *,
        guard: ModeGuard,
        post: Callable[[Message], None],
        calling: Callable[[], AbstractContextManager[None]],
        now: Callable[[], int],
        read_holdings: bool,
        positions_read: Callable[[Any, Any, int], None] | None = None,
    ) -> None:
        self._positions_read = positions_read
        self._client = client
        self._guard = guard
        self._post = post
        self._calling = calling
        self._now = now
        self._read_holdings = read_holdings

    def read(self, request: ReadBooks | ReadPositions) -> None:
        """Read the books `request` asks for and post them (module docstring)."""
        started = self._now()
        names = _BOOKS
        if isinstance(request, ReadPositions):
            names = _HELD_CANCEL_BOOKS if request.orders else _HELD_CANCEL_BOOKS[:1]
        if self._read_holdings:
            names = (names[0], _HOLDINGS, *names[1:])
        answers = {}
        for name in names:
            if self._guard.locked:
                break
            with self._calling():
                answers[name] = server_answer(getattr(self._client, name)())
            self._guard.check_response(answers[name])
        if self._guard.locked:
            why = "the mode guard is locked: the books answer from the other mode"
            self._post(NotSent(request, NotSentReason.LOCKED, why))
        elif isinstance(request, ReadBooks):
            if self._positions_read is not None:
                self._positions_read(answers["positionbook"], answers.get(_HOLDINGS), started)
            ended = self._now()
            self._post(Books(request.generation, **answers, started_ns=started, ended_ns=ended))
        else:
            positions, orders = answers["positionbook"], answers.get("orderbook")
            self._post(PositionRows(positions, started, orders, answers.get(_HOLDINGS)))


__all__ = ["BookReads"]
