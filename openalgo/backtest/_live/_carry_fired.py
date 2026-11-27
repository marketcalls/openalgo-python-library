"""Whether a stop GTT of a session that stopped without its day-close record may have fired
since, by the broker's books at the next start (20-positional-live 3.3 and 4.3; the lead's
rulings N5c-2 and N5c-3 (d)): the judgement `_carry_unrecorded` makes before it takes what
the books show closed as settled by hand.

A stop GTT the save holds may have fired when the GTT book read at the start shows it gone
or triggered (a book of active GTTs only shows a GTT that fired as gone, as one cancelled
by hand) and the leg's units at the broker have moved by exactly its units on its side
from the save's reference of them (`_session_saved`: the leg's latest settled
reconciliation read before the save): the rule of 3.3 by which any carried stop GTT gone
from the book is taken as fired. Such a strategy is not taken as settled by hand: its
session is rebuilt and continued as from a record, and the GTT is followed as a record's
is (a fire on a day no session ran is recorded at the child's limit, flagged
`price_unverified` and `missed_day`). With no such reference the books cannot tell a GTT
that fired from one cancelled by hand, and N5a's rule stands: what the books show closed
was settled by hand.

Quantities are signed whole units. Run by the main thread at the start.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ._carry_books import BookReads, books_of
from ._carry_start import REFERENCES_KEY
from ._gtt_book import Seen, gtt_rows, seen_of
from ._types import Purpose
from .journal import Journal

#: What a GTT book's row says of a GTT that may have fired.
_MAY_HAVE_FIRED = frozenset({Seen.GONE, Seen.TRIGGERED})


def may_have_fired(held: Any, books: BookReads, journal: Journal) -> bool:
    """Whether a stop GTT of `held` (a `_carry_unrecorded.Unrecorded` with a save) may have
    fired by `books` (a `_carry_books.BookReads`), against the save's reference read
    through `journal` (module docstring). Raises JournalError when the journal cannot be
    read."""
    read = (books.gttbook, books.positionbook, books.orderbook, books.tradebook)
    if held.save_day is None or any(book is None for book in read):
        return False
    rows = gtt_rows(books.gttbook)
    saved = journal.rms_state(held.name, day=held.save_day) or {}
    references = saved.get(REFERENCES_KEY) or []
    if rows is None or len(references) != 1 or not references[0].get("settled"):
        return False
    reference = references[0]
    for gtt in held.gtts:
        if gtt.purpose is not Purpose.STOP or gtt.trigger_id is None:
            continue
        if seen_of(rows, gtt.trigger_id) not in _MAY_HAVE_FIRED:
            continue
        if _moved_by(gtt, reference, books):
            return True
    return False


def _moved_by(gtt: Any, reference: Mapping[str, Any], books: BookReads) -> bool:
    """Whether the books show `gtt`'s leg moved from `reference` by exactly its units on
    its side (a SELL lowers the units, a BUY raises them)."""
    leg = (reference["symbol"], reference["exchange"], reference["product"])
    if leg != (gtt.symbol, gtt.exchange, gtt.product):
        return False
    found = books_of(books, leg)
    if found.units is None:
        return False
    signed = gtt.terms.quantity if gtt.terms.side == "BUY" else -gtt.terms.quantity
    return found.units == int(reference["units"]) + signed


__all__ = ["may_have_fired"]
