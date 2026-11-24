"""The session's own fills on a leg, as the order worker's GTT parts net them out of the
broker's units (20-positional-live 3.3), and whether the broker's books can be taken to
show them yet (`OwnFills.settled`). Split from the order path (`orders.OrderPath.own_fills`)
so each module keeps one job.

**Units** (`OwnFills.units`): each followed order on the leg by its latest word (the one
held for the trade book included), bought units positive and sold negative. A later move
of the leg's broker units is a GTT's only net of these.

**Settled** (`OwnFills.settled`, the one definition the GTT parts use). The broker's units
and the order path's words are read apart, so they agree in time only once the session's
own fills on the leg have settled: no order of the session's own on the leg is without its
terminal word (an unbound intent's order may be at the broker, filled; a working order, or
one sent with no word yet, can show a fill in the position book before its word reaches
the order path: the stream's word is on its way, or the stream is silent until the order
path polls), and the newest own fill on the leg (`OwnFills.newest_fill_ns`: when the order
path heard the newest word with fills of an order of the leg) is at least
`POSITION_BOOK_LAG_S` old (a broker's position book can lag a fill the order path has
heard, 16-session-design F78). Until then a read of the leg's units may count an own fill
the order path has not heard of, or miss one it has: 16-session-design 5.3 takes a position
difference while an own order works as a lag of the broker's books. The GTT parts never
ignore such units: they wait for them, for at most `_gtt_reads.UNITS_WAIT_S`, and then
give the answer that can send no second exit or second GTT (the baseline meanwhile is
moved by the session's own fills only).

Times are UTC nanoseconds, quantities signed whole units. Used by the thread that owns the
order path.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

from .._layout import NANOS_PER_SECOND
from ._types import JournalLike, Now, OrderIntent, OrderUpdate

#: Seconds after the order path heard an own fill on a leg during which a read of the
#: broker's position book may not show it yet (16-session-design F78). Not a broker's
#: figure: no broker publishes how far its position book lags a fill. 30 s spans three of
#: the GTT follower's reads on the clock (every 10 s, 20-positional-live 3.1), so a book
#: that lags by less never sets a GTT's baseline or decides what a GTT did; a book that
#: lags longer still can. To be measured in the sandbox's overnight holds (20-positional-
#: live V3) and on the maintainer's broker (V1), and set from what they show.
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
        """Whether `leg`'s units are settled: the broker's books can be taken to show every
        own fill on it (module docstring). No order of the session's own on the leg is
        without its terminal word, and its newest own fill is at least
        `POSITION_BOOK_LAG_S` old. Raises JournalError when the unbound intents cannot be
        read."""
        if any(_on(intent, leg) for intent in self._journal.unbound()):
            return False
        if any(word is None or not word.status.terminal for _, word in self._of(leg)):
            return False
        newest = self.newest_fill_ns(leg)
        lag_ns = round(POSITION_BOOK_LAG_S * NANOS_PER_SECOND)
        return newest is None or self._now() - newest >= lag_ns

    def newest_fill_ns(self, leg: LegKey) -> int | None:
        """When the order path heard the newest word with fills of an order on `leg` (UTC
        ns), or None when no own order has filled on it."""
        heard = [
            word.ts_recv_ns
            for _, word in self._of(leg)
            if word is not None and word.filled_quantity > 0
        ]
        return max(heard, default=None)

    def _of(self, leg: LegKey) -> list[OwnOrder]:
        return [(intent, word) for intent, word in self._orders() if _on(intent, leg)]


def _on(intent: OrderIntent, leg: LegKey) -> bool:
    return (intent.symbol, intent.exchange, intent.product) == leg


__all__ = ["POSITION_BOOK_LAG_S", "LegKey", "OwnFills", "OwnOrder"]
