"""The GTT part of the fake's venue (`venue.Venue` takes it in): the GTTs each book holds,
what the broker does to the next GTT call when a test asks (refuse it, apply a modify in
part), what a test does to a resting GTT as the broker would (cancel it by hand, expire
it, reject it), and the firing of a GTT on a trade, which places its child order.

A GTT fires in the books whose orders prices work (`Venue._working_books`: the broker's
always, the sandbox's while the switch is on and the broker answers its quotes), on a
trade that reaches its trigger (`gtts.fires`). Its child is placed as an order of the
book's, with no client (no API session sent it), so its updates, its fill, its trade-book
row and its position follow as any order's.

Prices are rupees, quantities units, times UTC nanoseconds.
"""

from __future__ import annotations

from collections import deque
from typing import Any

from openalgo.backtest._live._types import Mode

from .book import Book, OrderTerms
from .gtts import ACTIVE, CHILD_PRICE_TYPE, TRIGGERED, FakeGtt, fires
from .log import GTT
from .market import MarketState

#: The GTT calls a test can have the broker refuse.
GTT_CALLS = ("placegttorder", "modifygttorder", "cancelgttorder")


class GttVenue:
    """The venue's GTTs (module docstring). The venue gives `books`, `clock`, `log`,
    `accept` and `_working_books`; `gtt_history` is whether a broker's GTT book keeps the
    GTTs that ended (the sandbox's always does)."""

    books: dict[Mode, Book]
    clock: Any
    log: Any

    def init_gtts(self, *, history: bool) -> None:
        """Start with no GTT and nothing scheduled; `history` as the class docstring."""
        self.gtt_history = history
        self._gtt_refusals: dict[str, deque[str]] = {call: deque() for call in GTT_CALLS}
        self._half_modify = False

    def refuse_next_gtt(self, call: str, reason: str) -> None:
        """The broker refuses the next `call` (one of `GTT_CALLS`) with `reason`. Raises
        ValueError for another call."""
        if call not in GTT_CALLS:
            raise ValueError(f"a GTT refusal is for one of {GTT_CALLS}, not {call!r}")
        self._gtt_refusals[call].append(reason)

    def take_gtt_refusal(self, call: str) -> str | None:
        """The refusal scheduled for `call`, taken off the schedule; None for none."""
        waiting = self._gtt_refusals[call]
        return waiting.popleft() if waiting else None

    def half_apply_next_gtt_modify(self) -> None:
        """The next GTT modify changes the trigger only, keeps the units, and fails."""
        self._half_modify = True

    def take_half_modify(self) -> bool:
        """Whether the GTT modify now is the one applied in part, taken off the schedule."""
        half, self._half_modify = self._half_modify, False
        return half

    def gtt(self, book: Book, trigger_id: str) -> FakeGtt | None:
        """The GTT of `book` with `trigger_id`, ended or not; None for none."""
        return book.gtts.get(trigger_id)

    def set_gtt_status(self, trigger_id: str, status: str) -> None:
        """What the broker or the user does to a resting GTT, outside any call of the
        session: its book row shows `status` from now (cancelled by hand, expired,
        rejected after placement, or any word a broker uses), and no child is placed.
        Raises ValueError for an unknown trigger ID or a GTT that is not active."""
        for book in self.books.values():
            gtt = book.gtts.get(trigger_id)
            if gtt is not None:
                if not gtt.active:
                    raise ValueError(f"GTT {trigger_id} is {gtt.status}, not {ACTIVE}")
                gtt.status, gtt.updated_ns = status, self.clock.now()
                return
        raise ValueError(f"neither book holds a GTT {trigger_id}")

    def fire_gtts(self, book: Book, market: MarketState) -> None:
        """Fire every active GTT of `book` on `market`'s instrument that its last trade
        reaches (module docstring), in the order they were placed."""
        price, instrument = market.last_price, market.instrument
        if price is None:
            return
        for gtt in list(book.gtts.values()):
            if (gtt.symbol, gtt.exchange) == instrument.key and fires(gtt, price):
                self._fire(book, gtt)

    def _fire(self, book: Book, gtt: FakeGtt) -> None:
        now = self.clock.now()
        gtt.status, gtt.updated_ns = TRIGGERED, now
        terms = OrderTerms(
            strategy=gtt.strategy,
            symbol=gtt.symbol,
            exchange=gtt.exchange,
            action=gtt.action,
            quantity=gtt.quantity,
            price_type=CHILD_PRICE_TYPE,
            product=gtt.product,
            price=gtt.price,
            trigger_price=0.0,
        )
        child = self.accept(book, terms)  # type: ignore[attr-defined]
        if book.mode is Mode.SANDBOX:
            gtt.child_order_id = child.order_id
        self.log.record(now, GTT, gtt.trigger_id, mode=book.mode.value, child=child.order_id)


__all__ = ["GTT_CALLS", "GttVenue"]
