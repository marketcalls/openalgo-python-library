"""What the order path (`orders`) knows of each order it follows, and the rules that
read an update in its light: when the order is next polled, a stop reported open, a
terminal update with fewer fills than known. Times are UTC nanoseconds, quantities units."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from .._layout import NANOS_PER_SECOND
from ._types import OrderStatus, OrderUpdate

#: An order with no word 2 seconds after it was sent is polled (08-live-trading 6.1) ...
POLL_AFTER_NS = 2 * NANOS_PER_SECOND
#: ... then once a second until it is terminal or rests; rounds are this far apart.
POLL_EVERY_NS = 1 * NANOS_PER_SECOND
#: A resting order the stream has spoken for is polled this often, as a safety net for a
#: stream that falls silent.
SAFETY_POLL_EVERY_NS = 30 * NANOS_PER_SECOND
#: The price types of a stop: some brokers' books write one waiting for its trigger as
#: open, so it counts as triggered only on the stream's word or a fill.
STOP_PRICE_TYPES = frozenset({"SL", "SL-M"})


@dataclass(frozen=True)
class HeldFill:
    """An order update the engine has not been given yet: `filled_quantity` units (as
    reported; 0 for a cancel the tradebook has not confirmed), held since `held_since_ns`
    (UTC nanoseconds) for its price or the tradebook's confirmation."""

    order_id: str
    filled_quantity: int
    held_since_ns: int


@dataclass
class Followed:
    """An order of the session and what is known of it: the last update handed over
    (`seen`), the one waiting for the tradebook (`held`), and what the stream said."""

    strategy: str
    price_type: str
    sent_ns: int
    seen: OrderUpdate | None = None
    held: OrderUpdate | None = None
    held_since_ns: int = 0
    alerted: bool = False
    streamed: bool = False
    stream_pending: bool = False
    polled_ns: int | None = None

    def poll_due_ns(self) -> int | None:
        """When the order is next polled; None once its last word is terminal."""
        word = self.held or self.seen
        if word is None:
            if self.polled_ns is None:
                return self.sent_ns + POLL_AFTER_NS
            return self.polled_ns + POLL_EVERY_NS
        if word.status.terminal:
            return None
        rests = (self.price_type == "LIMIT" and word.status is OrderStatus.OPEN) or (
            self.price_type in STOP_PRICE_TYPES and word.status is OrderStatus.TRIGGER_PENDING
        )
        every = SAFETY_POLL_EVERY_NS if rests and self.streamed else POLL_EVERY_NS
        return max(self.polled_ns or 0, word.ts_recv_ns) + every


def as_followed(order: Followed, update: OrderUpdate) -> OrderUpdate:
    """A stop reported open with no fill is still waiting for its trigger, unless the
    stream said it was waiting and now says open: some brokers' books, and the streams
    the server makes from them, write a waiting stop as open."""
    if (
        order.price_type in STOP_PRICE_TYPES
        and update.status is OrderStatus.OPEN
        and update.filled_quantity == 0
        and not (update.source == "stream" and order.stream_pending)
    ):
        return dataclasses.replace(update, status=OrderStatus.TRIGGER_PENDING)
    return update


def with_floor(order: Followed, update: OrderUpdate) -> OrderUpdate:
    """A terminal update keeps the fills already known of its order (a polled stream
    says 0 filled): with fewer, it takes them, unpriced, so the tradebook settles it."""
    known = max(
        order.seen.filled_quantity if order.seen else 0,
        order.held.filled_quantity if order.held else 0,
    )
    if not update.status.terminal or update.filled_quantity >= known:
        return update
    return dataclasses.replace(update, filled_quantity=known, pending_quantity=0, average_price=0.0)


__all__ = [
    "POLL_AFTER_NS",
    "POLL_EVERY_NS",
    "SAFETY_POLL_EVERY_NS",
    "STOP_PRICE_TYPES",
    "Followed",
    "HeldFill",
    "as_followed",
    "with_floor",
]
