"""What the order path (`orders`) knows of each order it follows: when it is next polled,
when the tradebook is next read for a held word, and a terminal update with fewer fills
than known. Times are UTC nanoseconds, quantities units."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from .._layout import NANOS_PER_SECOND
from ._types import OrderIntent, OrderStatus, OrderUpdate, Purpose

#: An order with no word 2 seconds after it was sent is polled (08-live-trading 6.1) ...
POLL_AFTER_NS = 2 * NANOS_PER_SECOND
#: ... then once a second until it is terminal or rests; rounds are this far apart.
POLL_EVERY_NS = 1 * NANOS_PER_SECOND
#: A resting order the stream has spoken for is polled this often, as a safety net for a
#: stream that falls silent.
SAFETY_POLL_EVERY_NS = 30 * NANOS_PER_SECOND
#: A resting order no stream speaks for (18 brokers have no order stream, or the server
#: runs with ORDER_UPDATES_ENABLED=FALSE), and a held word's tradebook retries, back off
#: through these waits and stay at the last. Every exit cancels the stop first and waits
#: for the cancel's word, so a fill seen up to 5 s late is never exited twice.
BACKOFF_NS = (1 * NANOS_PER_SECOND, 2 * NANOS_PER_SECOND, 5 * NANOS_PER_SECOND)
#: An entry's or an add's fill places the protective stop, so its back-off stops at 2 s.
ENTRY_BACKOFF_NS = (1 * NANOS_PER_SECOND, 2 * NANOS_PER_SECOND)
#: The purposes that open or add to a position: an entry and an add.
ENTRY_PURPOSES = frozenset({Purpose.ENTRY, Purpose.ADD})
#: A market order reported open is working, not resting: it is polled every second.
_MARKET = "MARKET"


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
    """An order of the session and what is known of it: its intent, when it was sent, the
    last update handed over (`seen`), the one waiting for the tradebook (`held`), whether
    the stream has spoken for it, and its polls and tradebook reads. `price_type` is the
    order's now (a modify may change it): a market order open is working, not resting."""

    intent: OrderIntent
    sent_ns: int
    price_type: str
    seen: OrderUpdate | None = None
    held: OrderUpdate | None = None
    held_since_ns: int = 0
    alerted: bool = False
    streamed: bool = False
    polled_ns: int | None = None
    quiet_polls: int = 0
    trades_read_ns: int | None = None
    trade_reads: int = 0

    @classmethod
    def of(cls, intent: OrderIntent, *, sent_ns: int) -> Followed:
        """An order just bound to `intent`, sent at `sent_ns`."""
        return cls(intent, sent_ns, intent.price_type)

    @property
    def strategy(self) -> str:
        """The strategy the order was sent for."""
        return self.intent.strategy

    def poll_due_ns(self) -> int | None:
        """When the order is next polled; None once its last word is terminal.

        No word yet: from 2 s after the send, every second. Resting (open or waiting for
        a trigger, as reported, and not a market order): every 30 s once the stream has
        spoken for it, else backing off (`BACKOFF_NS`, `ENTRY_BACKOFF_NS`) while polls
        bring no news. Any other working order: every second.
        """
        word = self.held or self.seen
        if word is None:
            if self.polled_ns is None:
                return self.sent_ns + POLL_AFTER_NS
            return self.polled_ns + POLL_EVERY_NS
        if word.status.terminal:
            return None
        last_ns = max(self.polled_ns or 0, word.ts_recv_ns)
        rests = word.status is OrderStatus.TRIGGER_PENDING or self.price_type != _MARKET
        if not rests:
            return last_ns + POLL_EVERY_NS
        if self.streamed:
            return last_ns + SAFETY_POLL_EVERY_NS
        return last_ns + self._backoff_ns(self.quiet_polls)

    def trades_due_ns(self) -> int | None:
        """When the tradebook is next read for the held word; None with nothing held.
        The first read is at once, later ones back off as the polls do."""
        if self.held is None:
            return None
        if self.trades_read_ns is None:
            return self.held_since_ns
        return self.trades_read_ns + self._backoff_ns(self.trade_reads - 1)

    def hold(self, update: OrderUpdate, now: int) -> None:
        """Hold `update` for the tradebook; a first hold starts its clock and retries."""
        if self.held is None:
            self.held_since_ns, self.alerted = now, False
            self.trades_read_ns, self.trade_reads = None, 0
        self.held = update

    def release(self) -> None:
        """Nothing is held any more."""
        self.held, self.trades_read_ns, self.trade_reads = None, None, 0

    def _backoff_ns(self, tries: int) -> int:
        steps = ENTRY_BACKOFF_NS if self.intent.purpose in ENTRY_PURPOSES else BACKOFF_NS
        return steps[min(max(tries, 0), len(steps) - 1)]


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
    "BACKOFF_NS",
    "ENTRY_BACKOFF_NS",
    "ENTRY_PURPOSES",
    "POLL_AFTER_NS",
    "POLL_EVERY_NS",
    "SAFETY_POLL_EVERY_NS",
    "Followed",
    "HeldFill",
    "with_floor",
]
