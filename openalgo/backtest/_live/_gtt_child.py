"""The child order of a GTT that fired, and what a GTT that is gone with no child found
says (20-positional-live 3.2, 3.3; the lead's ruling N3a-2). Pure: the GTT follower
(`_gtt_follow`) and the order worker's GTT confirmations (`_gtt_send`) read the books and
act on what these give.

- **The child** (`child_candidates`, 3.2 rule 2): an order of the order book on the GTT's
  symbol, exchange, product and side, for its units, placed at or after the last read that
  showed the GTT resting (less the clock skew the preflight allows, as an unbound intent's
  order is looked for: `_order_book.CLOCK_SKEW_LIMIT_NS`), and taken by nothing of the
  session (no intent of the journal is bound to it, nor is it another GTT's child). A row
  whose units or time cannot be read is a possible child the engine cannot tell apart
  (`unsure`): with any, or with more than one sure, the engine never guesses (rule 5).
- **A GTT gone with no child** (`move_of`, 3.3): the broker's units of the leg against
  its baseline, those read just before the last read that showed it resting, moved by
  what the session's own orders filled on the leg since (`_gtt_reads`). Moved by exactly
  the GTT's units on its side (a SELL lowers them, a BUY raises them: a long's stop, a
  short's stop and a strategy's GTT entry alike, review M4): it fired and its child was
  missed. Unchanged: it ended outside the session. Anything else: the engine cannot tell.
- **A missed child's price** (`missed_price`): the trade book's trades of the leg on the
  GTT's side by orders the session did not send, newest first, for the GTT's units,
  averaged; None when the trade book holds none (the GTT's limit stands in, flagged).

Times are UTC nanoseconds, prices rupees, quantities signed whole units.
"""

from __future__ import annotations

import enum
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .._layout import NANOS_PER_SECOND
from ..errors import LiveError
from ._journal_gtt_rows import GttBinding, GttTerms
from ._order_book import CLOCK_SKEW_LIMIT_NS, book_time_ns
from ._order_updates import units
from ._reconcile_rules import adoption_price, leg_books


@dataclass(frozen=True)
class Candidates:
    """The orders that may be a GTT's child: `sure` ones, and `unsure` ones whose units or
    time cannot be read."""

    sure: tuple[str, ...] = ()
    unsure: tuple[str, ...] = ()

    @property
    def child(self) -> str | None:
        """The child when exactly one order can be it, else None."""
        return self.sure[0] if len(self.sure) == 1 and not self.unsure else None

    @property
    def every(self) -> tuple[str, ...]:
        """Every order that may be the child, sure ones first."""
        return self.sure + self.unsure


class Move(enum.Enum):
    """How the broker's units of a leg moved while its GTT went from the book."""

    #: By exactly the GTT's units on its side: it fired.
    FIRED = "fired"
    #: Not at all: it ended outside the session.
    UNCHANGED = "unchanged"
    #: Otherwise: the engine cannot tell.
    OTHER = "other"


def child_candidates(
    gtt: GttBinding,
    rows: Sequence[Mapping[str, Any]],
    *,
    since_ns: int | None,
    taken: Callable[[str], bool],
) -> Candidates:
    """The orders of `rows` (the order book's) that may be `gtt`'s child (module
    docstring), placed at or after `since_ns` (any time for None)."""
    earliest = None
    if since_ns is not None:
        earliest = since_ns - since_ns % NANOS_PER_SECOND - CLOCK_SKEW_LIMIT_NS
    sure: list[str] = []
    unsure: list[str] = []
    for row in rows:
        order_id = str(row.get("orderid") or "")
        if not order_id or taken(order_id) or not _same_leg(gtt, row):
            continue
        try:
            quantity = units(row, "quantity", order_id)
        except LiveError:
            unsure.append(order_id)
            continue
        if quantity != gtt.terms.quantity:
            continue
        placed_ns = book_time_ns(row.get("timestamp"))
        if placed_ns is None:
            unsure.append(order_id)
        elif earliest is None or placed_ns >= earliest:
            sure.append(order_id)
    return Candidates(tuple(sure), tuple(unsure))


def move_of(terms: GttTerms, before: int, now: int) -> Move:
    """How the broker's units of the leg of a GTT of `terms` moved from `before`, its
    baseline, to `now` (module docstring)."""
    signed = terms.quantity if terms.side == "BUY" else -terms.quantity
    if now - before == signed:
        return Move.FIRED
    return Move.UNCHANGED if now == before else Move.OTHER


def missed_price(
    gtt: GttBinding, tradebook: Mapping[str, Any] | None, taken: Callable[[str], bool]
) -> float | None:
    """The price of `gtt`'s missed child from the trade book's answer (module docstring);
    None when it holds no such trade or could not be read."""
    if tradebook is None:
        return None
    books = leg_books({}, tradebook, {}, gtt.symbol, gtt.exchange, product=gtt.product)
    others = [trade for trade in books.trades or () if not taken(trade.order_id)]
    try:
        return adoption_price(others, gtt.terms.quantity, gtt.terms.side)
    except ValueError:
        return None


def _same_leg(gtt: GttBinding, row: Mapping[str, Any]) -> bool:
    """Whether an order-book row is on `gtt`'s symbol, exchange, product and side; a row
    that names no product is taken as the GTT's (a book that leaves it out)."""
    product = str(row.get("product") or gtt.product).upper()
    return (
        row.get("symbol") == gtt.symbol
        and row.get("exchange") == gtt.exchange
        and product == gtt.product
        and str(row.get("action") or "").upper() == gtt.terms.side
    )


__all__ = ["Candidates", "Move", "child_candidates", "missed_price", "move_of"]
