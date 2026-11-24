"""The session's own fills on a leg, as the order worker's GTT parts net them out of the
broker's units (20-positional-live 3.3), and whether the broker's books can be taken to
show them yet (`OwnFills.settled`). Split from the order path (`orders.OrderPath.own_fills`)
so each module keeps one job.

**Units** (`OwnFills.units`): each followed order on the leg by its latest word (the one
held for the trade book included), bought units positive and sold negative. A later move
of the leg's broker units is a GTT's only net of these.

**Settled.** The broker's units and the order path's words are read apart, so they agree
in time only once the session's own fills on the leg have settled: no intent of the leg is
unbound (its order may be at the broker, filled), no order of the leg is working or sent
with no word yet (a fill of it can show in the position book before its word reaches the
order path: the stream's word is on its way, or the stream is silent until the order path
polls), and the newest word with fills of an order of the leg was heard at least
`POSITION_BOOK_LAG_S` ago (a broker's position book can lag a fill the order path has
heard, 16-session-design F78). Until then a read of the leg's units may count an own fill
the order path has not heard of, or miss one it has: 16-session-design 5.3 takes a position
difference while an own order works as a lag of the broker's books, and the GTT parts do
the same (the baseline is moved by the session's own fills only, and no decision rests on
the units).

Times are UTC nanoseconds, quantities signed whole units. Used by the thread that owns the
order path.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

from .._layout import NANOS_PER_SECOND
from ._types import JournalLike, Now, OrderIntent, OrderUpdate

#: Seconds after the order path heard an own fill on a leg during which a read of the
#: broker's position book may not show it yet (16-session-design F78). No broker publishes
#: how far its position book lags a fill; 30 s spans three of the GTT follower's reads on
#: the clock (every 10 s, 20-positional-live 3.1), so a book that lags by less never sets
#: a GTT's baseline or decides what a GTT did. A book that lags longer still can.
POSITION_BOOK_LAG_S = 30.0

#: A leg as the GTT parts key it: (symbol, exchange, product).
LegKey = tuple[str, str, str]
#: An order the order path follows: its intent and its latest word (None before any).
OwnOrder = tuple[OrderIntent, OrderUpdate | None]


class OwnFills:
    """The session's own fills on each leg (module docstring): `orders()` gives every order
    the order path follows, terminal ones included; `journal` holds the unbound intents;
    `now` gives UTC nanoseconds."""

    def __init__(
        self, orders: Callable[[], Iterable[OwnOrder]], *, journal: JournalLike, now: Now
    ) -> None:
        self._orders = orders
        self._journal = journal
        self._now = now

    def units(self, leg: LegKey) -> int:
        """The units the session's own orders have filled on `leg`, signed (module
        docstring)."""
        filled = 0
        for intent, word in self._of(leg):
            if word is not None:
                filled += word.filled_quantity if intent.action == "BUY" else -word.filled_quantity
        return filled

    def settled(self, leg: LegKey) -> bool:
        """Whether the broker's books can be taken to show every own fill on `leg` (module
        docstring). Raises JournalError when the unbound intents cannot be read."""
        if any(_on(intent, leg) for intent in self._journal.unbound()):
            return False
        lag_ns = round(POSITION_BOOK_LAG_S * NANOS_PER_SECOND)
        now = self._now()
        for _, word in self._of(leg):
            if word is None or not word.status.terminal:
                return False
            if word.filled_quantity > 0 and now - word.ts_recv_ns < lag_ns:
                return False
        return True

    def _of(self, leg: LegKey) -> list[OwnOrder]:
        return [(intent, word) for intent, word in self._orders() if _on(intent, leg)]


def _on(intent: OrderIntent, leg: LegKey) -> bool:
    return (intent.symbol, intent.exchange, intent.product) == leg


__all__ = ["POSITION_BOOK_LAG_S", "LegKey", "OwnFills", "OwnOrder"]
