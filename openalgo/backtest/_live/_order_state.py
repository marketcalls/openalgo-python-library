"""What the order path (`orders`) knows of each order it follows: when it is next polled,
when the tradebook is next read for a held word, a terminal update with fewer fills than
known, and a held cancel handed over with the fills the session found for it. Times are
UTC nanoseconds, quantities units, prices rupees."""

from __future__ import annotations

import dataclasses
import math
import numbers
from dataclasses import dataclass

from .._layout import NANOS_PER_SECOND
from ..errors import LiveError
from ._types import MARKET, OrderIntent, OrderStatus, OrderUpdate, Purpose

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
#: An order whose cancel the broker accepted is polled every second, from a second after
#: the acceptance, for this long while no word ends it: a lost cancel word must not hold
#: the exit that waits for it until the next safety poll (16-session-design 4.5, 3.3).
CANCEL_POLL_FOR_NS = 10 * NANOS_PER_SECOND
#: The purposes that open or add to a position: an entry and an add.
ENTRY_PURPOSES = frozenset({Purpose.ENTRY, Purpose.ADD})


@dataclass(frozen=True)
class HeldFill:
    """An order update the engine has not been given yet: `filled_quantity` units (as
    reported, or the units already handed over when a terminal word reports fewer; 0 for
    a cancel the tradebook has not confirmed, of an order with nothing handed over), held
    since `held_since_ns` (UTC nanoseconds) for its price or the tradebook's confirmation.
    `average_price` is the held word's own average price in rupees (a polled fill carries
    `orderstatus`'s), 0.0 when it gives none: the price to release it at when positive."""

    order_id: str
    filled_quantity: int
    held_since_ns: int
    average_price: float = 0.0


@dataclass
class Followed:
    """An order of the session and what is known of it: its intent, when it was sent, the
    last update handed over (`seen`), the one waiting for the tradebook (`held`), when the
    stream last spoke for it, its polls (and a poll asked for out of turn) and its
    tradebook reads. `price_type` is the order's now (a modify may change it): a market
    order open is working, not resting."""

    intent: OrderIntent
    sent_ns: int
    price_type: str
    seen: OrderUpdate | None = None
    held: OrderUpdate | None = None
    held_since_ns: int = 0
    alerted: bool = False
    streamed_ns: int | None = None
    polled_ns: int | None = None
    poll_asked_ns: int | None = None
    cancel_accepted_ns: int | None = None
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

    @property
    def streamed(self) -> bool:
        """True once the stream has spoken for the order."""
        return self.streamed_ns is not None

    @property
    def last_word(self) -> OrderUpdate | None:
        """The latest word of the order: the one held for the tradebook when there is one
        (a word is held only when it tells more than the last one handed over), else the
        last handed over."""
        return self.held or self.seen

    def heard_from_stream(self, ts_recv_ns: int) -> None:
        """Note a word of the stream received at `ts_recv_ns`."""
        self.streamed_ns = max(self.streamed_ns or ts_recv_ns, ts_recv_ns)

    def heard(self, update: OrderUpdate) -> None:
        """Note a word of the order as it is taken: a stream word tells when the stream
        last spoke for it; a polled word read at or after a poll was asked for out of turn
        answers that poll (it gives the order's whole state at that time)."""
        if update.source == "stream":
            self.heard_from_stream(update.ts_recv_ns)
        elif self.poll_asked_ns is not None and update.ts_recv_ns >= self.poll_asked_ns:
            self.poll_asked_ns = None

    def polled(self, now: int) -> None:
        """Note a poll made at `now`. A poll asked for out of turn stays asked until a
        polled word answers it (`heard`): a read that fails or gives no word does not."""
        self.polled_ns, self.quiet_polls = now, self.quiet_polls + 1

    def cancel_accepted(self, now: int) -> None:
        """Note that a cancel of the order was accepted at `now`: it is polled every
        second for `CANCEL_POLL_FOR_NS` while no word ends it."""
        self.cancel_accepted_ns = now

    def ask_poll(self, now: int) -> None:
        """Make the order due for a poll from `now`, whatever its schedule."""
        self.poll_asked_ns = now if self.poll_asked_ns is None else min(self.poll_asked_ns, now)

    def missed_since(self, since_ns: int) -> bool:
        """Whether the stream may have missed news of the working order since `since_ns`:
        no stream word of it has arrived since then. A word of the stream gives the
        order's whole state (its cumulative fills), so one after a gap covers it."""
        word = self.last_word
        if word is not None and word.status.terminal:
            return False
        return self.streamed_ns is None or self.streamed_ns < since_ns

    def poll_due_ns(self) -> int | None:
        """When the order is next polled; None once its last word is terminal.

        No word yet: from 2 s after the send, every second. A cancel of it accepted:
        every second from a second after the acceptance, for `CANCEL_POLL_FOR_NS`. Resting
        (open or waiting for a trigger, as reported, and not a market order): every 30 s
        once the stream has spoken for it, else backing off (`BACKOFF_NS`,
        `ENTRY_BACKOFF_NS`) while polls bring no news. Any other working order: every
        second. A poll asked for out of
        turn (`ask_poll`) is due from when it was asked, whichever comes first, and a
        second after each poll since that did not answer it, so it takes its turn in the
        rounds with the other orders.
        """
        word = self.last_word
        if word is not None and word.status.terminal:
            return None
        due_ns = self._scheduled_ns(word)
        asked_ns = self.poll_asked_ns
        if asked_ns is None:
            return due_ns
        if self.polled_ns is not None and self.polled_ns >= asked_ns:
            asked_ns = self.polled_ns + POLL_EVERY_NS
        return min(due_ns, asked_ns)

    def _scheduled_ns(self, word: OrderUpdate | None) -> int:
        """When the order's schedule polls it next (see `poll_due_ns`)."""
        if word is None:
            if self.polled_ns is None:
                return self.sent_ns + POLL_AFTER_NS
            return self.polled_ns + POLL_EVERY_NS
        last_ns = max(self.polled_ns or 0, word.ts_recv_ns)
        cancelled_ns = self.cancel_accepted_ns
        if cancelled_ns is not None and last_ns < cancelled_ns + CANCEL_POLL_FOR_NS:
            return max(last_ns, cancelled_ns) + POLL_EVERY_NS
        # A market order reported open is working, not resting: it is polled every second.
        rests = word.status is OrderStatus.TRIGGER_PENDING or self.price_type != MARKET
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

    def handed_over(self, update: OrderUpdate) -> None:
        """Note `update` handed to the engine. A held word it tells as much as is released,
        but a terminal held word (a cancel the tradebook must confirm) outlives a working
        update, and takes the fills handed over since it was held (`with_floor`), so it is
        never released with fewer."""
        self.seen, self.quiet_polls = update, 0
        held = self.held
        if held is None:
            return
        if update.filled_quantity >= held.filled_quantity and (
            update.status.terminal or not held.status.terminal
        ):
            self.release()
        else:
            self.held = with_floor(self, held)

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


def released_fill(held: OrderUpdate, average_price: float) -> OrderUpdate:
    """A held word at `average_price` rupees, or as it is when it reports no fill. Raises
    LiveError for a fill at a price that is not a positive number."""
    if not held.filled_quantity:
        return held
    _check_price(average_price)
    return dataclasses.replace(held, average_price=average_price)


def released_cancel(held: OrderUpdate, filled_quantity: int, average_price: float) -> OrderUpdate:
    """A held cancel with the `filled_quantity` units the session found it filled in all
    before it ended (from the position book), at `average_price` rupees, their average
    (unused for 0 units).

    Raises LiveError unless `held` is a cancel, `filled_quantity` is a whole number from
    the units already reported filled up to the order's units, and the price of a fill is
    a positive number.
    """
    if held.status is not OrderStatus.CANCELLED:
        raise LiveError(
            f"order {held.order_id} holds a {held.status.value} word, not a cancel: release "
            "it with release_fill"
        )
    whole = isinstance(filled_quantity, numbers.Integral) and not isinstance(filled_quantity, bool)
    if not (whole and held.filled_quantity <= filled_quantity <= held.quantity):
        raise LiveError(
            f"a cancel of order {held.order_id} is released with a whole number of units "
            f"from the {held.filled_quantity} reported filled up to its {held.quantity}, not "
            f"{filled_quantity!r}"
        )
    if filled_quantity:
        _check_price(average_price)
    return dataclasses.replace(
        held,
        filled_quantity=int(filled_quantity),
        pending_quantity=0,
        average_price=average_price if filled_quantity else 0.0,
    )


def _check_price(average_price: float) -> None:
    if not (math.isfinite(average_price) and average_price > 0):
        raise LiveError(f"a fill is released at a positive price, not {average_price!r}")


__all__ = [
    "BACKOFF_NS",
    "CANCEL_POLL_FOR_NS",
    "ENTRY_BACKOFF_NS",
    "ENTRY_PURPOSES",
    "POLL_AFTER_NS",
    "POLL_EVERY_NS",
    "SAFETY_POLL_EVERY_NS",
    "Followed",
    "HeldFill",
    "released_cancel",
    "released_fill",
    "with_floor",
]
