"""The broker's books a start reads to judge what a strategy holds from an earlier day that no
day-close record carries (`_carry_unrecorded`; 20-positional-live 4.2 step 2): the position
book, the holdings when a CNC leg is judged, the order and trade books, and the GTT book of
every status, read once for every such strategy of the start, as the SDK answers them.
Split from `_carry_unrecorded` so each module keeps one job.

Quantities are signed whole units. Run by the main thread at the start: only reads.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from .._products import CNC
from ._answers import server_answer
from ._gtt_reads import ALL_GTTS
from ._reconcile_rules import LegBooks, leg_books

#: The books the judgement reads, by the SDK call that reads each.
_BOOKS = ("positionbook", "orderbook", "tradebook")

#: A leg: (symbol, exchange, product).
Leg = tuple[str, str, str]


@dataclass(frozen=True)
class BookReads:
    """The broker's books the judgement reads (module docstring), as the SDK answered
    them; a book not read, or not a success, is None."""

    positionbook: Mapping[str, Any] | None
    holdings: Mapping[str, Any] | None
    orderbook: Mapping[str, Any] | None
    tradebook: Mapping[str, Any] | None
    gttbook: Mapping[str, Any] | None


def read_books(client: Any, legs: Iterable[Leg]) -> BookReads:
    """The broker's books for judging the strategies whose `legs` these are (module
    docstring), read through `client` (the session's data client); only reads."""
    answers = {name: _answer(client, name) for name in _BOOKS}
    cnc = any(leg[2] == CNC for leg in legs)
    holdings = _answer(client, "holdings") if cnc else None
    gttbook = _answer(client, "gttorderbook", status=ALL_GTTS)
    return BookReads(holdings=holdings, gttbook=gttbook, **answers)


def books_of(books: BookReads, leg: Leg) -> LegBooks:
    """The books' view of `leg` (symbol, exchange, product): its units at the broker (a
    CNC leg's with the holdings) and its order rows."""
    symbol, exchange, product = leg
    return leg_books(
        books.positionbook,
        books.tradebook,
        books.orderbook,
        symbol,
        exchange,
        product=product,
        holdings=books.holdings,
    )


def _answer(client: Any, name: str, **fields: Any) -> Mapping[str, Any] | None:
    """The SDK's answer to read `name`, or None when it is not a success."""
    answer = server_answer(getattr(client, name)(**fields))
    return answer if answer.get("status") == "success" else None


__all__ = ["BookReads", "Leg", "books_of", "read_books"]
