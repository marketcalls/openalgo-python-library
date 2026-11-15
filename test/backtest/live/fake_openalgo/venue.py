"""The state of the fake OpenAlgo: its instruments and their prices, the live broker's book
and the sandbox's, the analyzer and semi-auto switches, the broker's login, the Telegram
bot, and what the broker does to the next orders when a test asks (reject one, fill one
in part, report a fill at price 0).

Every change of an order becomes an order update, sent on the feed once the venue's lock
is released (`flush`), so a REST answer and the updates it caused never wait on each
other's locks. Prices work the broker's orders whatever the switch says, and the
sandbox's only while the switch is on (the server stops the sandbox's execution engine
while it is off, services/analyzer_service.py, `_reconcile_sandbox`) and the broker's
login holds (the sandbox prices every fill from the broker's quotes).

Prices are rupees, quantities units, times UTC nanoseconds.
"""

from __future__ import annotations

import itertools
import threading
from collections import deque
from dataclasses import dataclass
from typing import Any

from openalgo.backtest._live._types import Mode

from .book import FILLED, Book, Change, OrderTerms
from .clock import ManualClock
from .feed import Feed
from .ids import BrokerIds, SandboxIds
from .log import FILL, SWITCH, FakeLog
from .market import Instrument, MarketState
from .messages import order_update
from .orders import FakeOrder


@dataclass(frozen=True)
class Rejection:
    """A refusal the broker gives the next order (of `price_type` when given): at
    placement (the place call fails) or right after accepting it (a rejected update)."""

    reason: str
    price_type: str | None
    at_placement: bool


@dataclass(frozen=True)
class TelegramMessage:
    """A notification the Telegram endpoint took: to `username`, at `ts_ns`."""

    username: str
    message: str
    priority: int
    ts_ns: int


class Venue:
    """The fake broker and server. `username` is the OpenAlgo user (the order updates'
    `user_id`, and linked to Telegram); `live_cash` and `sandbox_cash` are the available
    cash of each book, rupees."""

    def __init__(
        self,
        *,
        clock: ManualClock,
        log: FakeLog,
        feed: Feed,
        broker: str,
        username: str,
        live_cash: float,
        sandbox_cash: float,
        analyzer: bool,
    ) -> None:
        self.clock, self.log, self.feed = clock, log, feed
        self.broker, self.username = broker, username
        self.lock = threading.RLock()
        self.markets: dict[tuple[str, str], MarketState] = {}
        self.books = {
            Mode.LIVE: Book(Mode.LIVE, BrokerIds(), live_cash),
            Mode.SANDBOX: Book(Mode.SANDBOX, SandboxIds(), sandbox_cash),
        }
        self.analyzer = analyzer
        self.semi_auto = False
        self.session_expired = False
        self.login_revoked = False
        self.telegram_linked = {username}
        self.telegram_bot_active = True
        self.telegram_sent: list[TelegramMessage] = []
        self.analyzer_logs = 0
        self._pending_ids = itertools.count(1)
        self._rejections: list[Rejection] = []
        self._split_next: int | None = None
        self._zero_price_next = False
        self._outbox: deque[dict[str, Any]] = deque()
        self._flushing = threading.Lock()

    # -- reading the state ------------------------------------------------

    @property
    def mode(self) -> Mode:
        """The mode the analyzer switch selects now."""
        return Mode.SANDBOX if self.analyzer else Mode.LIVE

    @property
    def book(self) -> Book:
        """The book the analyzer switch selects now."""
        return self.books[self.mode]

    @property
    def broker_reachable(self) -> bool:
        """Whether the broker answers OpenAlgo: its login has neither expired nor been
        revoked. The sandbox prices its orders from the broker's quotes, so it cannot
        fill without it."""
        return not (self.session_expired or self.login_revoked)

    def market(self, symbol: str, exchange: str) -> MarketState | None:
        """The price path of an instrument, None when the broker does not know it."""
        return self.markets.get((symbol, exchange))

    def has_position_row(self, book: Book, symbol: str, exchange: str, product: str) -> bool:
        """Whether `book` has a position row (open or flat) for an instrument and product,
        priced by a trade: the sandbox prices a MARKET order from it when no quote comes
        (sandbox/order_manager.py:323-333)."""
        market = self.market(symbol, exchange)
        if market is None or market.last_price is None:
            return False
        return any(
            (row.symbol, row.exchange, row.product) == (symbol, exchange, product)
            for row in book.positions.rows()
        )

    def next_pending_order_id(self) -> int:
        """The ID the Action Center gives the next queued order (database/
        action_center_db.py: the row's integer key)."""
        return next(self._pending_ids)

    # -- the market -------------------------------------------------------

    def add_instrument(self, instrument: Instrument) -> None:
        """Make an instrument known. Raises ValueError for one known already."""
        with self.lock:
            if instrument.key in self.markets:
                raise ValueError(f"{instrument.exchange}:{instrument.symbol} is known already")
            self.markets[instrument.key] = MarketState(instrument)

    def trade(
        self,
        symbol: str,
        exchange: str,
        price: float,
        *,
        volume: float = 0.0,
        bid: float | None = None,
        ask: float | None = None,
    ) -> None:
        """A trade print now of `volume` units at `price`: the price moves, the working
        orders of that instrument are worked (the broker's always, the sandbox's while
        the switch is on), and a tick is sent (`bid` and `ask` give the best quotes, else
        they are the price)."""
        with self.lock:
            market = self._known(symbol, exchange)
            market.trade(self.clock.now(), price, volume, bid, ask)
            for book in self._working_books():
                for order in book.working_orders(symbol, exchange):
                    self._publish_all(book, self._work(book, order, market, arriving=False))
        self.feed.tick(market)
        self.flush()

    # -- orders -----------------------------------------------------------

    def accept(self, book: Book, terms: OrderTerms, *, priced: bool = True) -> FakeOrder:
        """Accept an order into `book`, announce it, apply a rejection the test asked
        for (a broker's only), and work it against the market. With `priced` False (the
        sandbox could fetch no quote) the order rests unworked, a stop trigger pending.
        Returns the order."""
        market = self._known(terms.symbol, terms.exchange)
        placed = book.place(terms, self.clock.now(), market.last_price if priced else None)
        order = book.orders[placed.order.order_id]
        self._publish(book, placed)
        rejection = None
        if book.mode is Mode.LIVE:
            rejection = self.take_rejection(terms.price_type, at_placement=False)
        if rejection is not None:
            self._publish(book, book.reject(order, rejection.reason))
        elif priced:
            self._publish_all(book, self._work(book, order, market, arriving=True))
        return order

    def reject_on_arrival(self, book: Book, terms: OrderTerms, reason: str) -> FakeOrder:
        """Record an order the sandbox refuses at placement: kept as rejected with no
        units pending, and announced as rejected only (sandbox/order_manager.py:632-680).
        Returns the order."""
        placed = book.place(terms, self.clock.now(), None)
        order = book.orders[placed.order.order_id]
        self._publish(book, book.reject(order, reason))
        return order

    def modify(
        self,
        book: Book,
        order: FakeOrder,
        *,
        quantity: int,
        price_type: str,
        price: float,
        trigger: float,
    ) -> None:
        """Change a working order's terms, announce it (a broker does; the sandbox does
        not, sandbox/order_manager.py) and work it against the market. At a broker the
        modified order arrives anew; the sandbox leaves it to its engine, which fills a
        limit at its limit (sandbox/execution_engine.py:532-540)."""
        change = book.modify(
            order, quantity=quantity, price_type=price_type, price=price, trigger=trigger
        )
        live = book.mode is Mode.LIVE
        if live:
            self._publish(book, change)
        if book in self._working_books():
            market = self._known(order.symbol, order.exchange)
            self._publish_all(book, self._work(book, order, market, arriving=live))

    def cancel(self, book: Book, order: FakeOrder) -> None:
        """Cancel a working order and announce it."""
        self._publish(book, book.cancel(order))

    def fill(self, order_id: str, units: int | None, price: float | None) -> None:
        """Fill a working order of either book now (all that is pending when `units` is
        None, at the last price when `price` is None). Raises ValueError for an unknown
        or finished order, a part fill in the sandbox, or no price."""
        with self.lock:
            book = next((b for b in self.books.values() if order_id in b.orders), None)
            if book is None:
                raise ValueError(f"neither book holds an order {order_id}")
            order = book.orders[order_id]
            market = self._known(order.symbol, order.exchange)
            at = price if price is not None else market.last_price
            if at is None:
                raise ValueError(f"no price to fill order {order_id} at: push a trade first")
            count = order.pending if units is None else units
            if book.mode is Mode.SANDBOX and count != order.pending:
                raise ValueError("the OpenAlgo sandbox fills whole orders only")
            self._publish(book, book.fill(order, count, at, self.clock.now()))
        self.flush()

    def take_rejection(self, price_type: str, *, at_placement: bool) -> Rejection | None:
        """The first rejection the test asked for that applies to an order of
        `price_type` at this stage, taken off the list."""
        for index, rejection in enumerate(self._rejections):
            fits = rejection.price_type in (None, price_type)
            if fits and rejection.at_placement is at_placement:
                return self._rejections.pop(index)
        return None

    def schedule_rejection(self, rejection: Rejection) -> None:
        """Have the broker refuse the next order that fits `rejection`. Raises ValueError
        for a rejection after acceptance in sandbox mode: the sandbox refuses only at
        placement (sandbox/order_manager.py:670-680)."""
        with self.lock:
            if self.mode is Mode.SANDBOX and not rejection.at_placement:
                raise ValueError(
                    "the OpenAlgo sandbox refuses an order only at placement: pass "
                    "at_placement=True"
                )
            self._rejections.append(rejection)

    def split_next_fill(self, units: int) -> None:
        """The next fill of the live book takes only `units`; the rest fills on a later
        price. Raises ValueError in sandbox mode or for fewer than 1 unit."""
        with self.lock:
            if self.mode is Mode.SANDBOX:
                raise ValueError("the OpenAlgo sandbox fills whole orders only")
            if units < 1:
                raise ValueError(f"a part fill is at least 1 unit, not {units}")
            self._split_next = units

    def zero_price_next_fill(self) -> None:
        """The next fill of the live book is reported on the stream at an average price of
        0. Raises ValueError in sandbox mode, which reports its fills at their price."""
        with self.lock:
            if self.mode is Mode.SANDBOX:
                raise ValueError("the OpenAlgo sandbox reports every fill at its price")
            self._zero_price_next = True

    # -- switches ---------------------------------------------------------

    def set_switch(self, name: str, on: bool) -> None:
        """Flip the analyzer switch, the semi-auto switch, the broker's login expiring
        ("expired"), OpenAlgo revoking it ("revoked") or the Telegram bot
        ("telegram_bot")."""
        with self.lock:
            if name == "analyzer":
                self.analyzer = on
            elif name == "semi_auto":
                self.semi_auto = on
            elif name == "expired":
                self.session_expired = on
            elif name == "revoked":
                self.login_revoked = on
            elif name == "telegram_bot":
                self.telegram_bot_active = on
            else:
                raise ValueError(f"unknown switch {name!r}")
            self.feed.set_login(self.broker_reachable)
            self.log.record(self.clock.now(), SWITCH, name, on=on)

    # -- delivery ---------------------------------------------------------

    def flush(self) -> None:
        """Send the order updates waiting in the outbox, oldest first. Called outside the
        venue's lock."""
        with self._flushing:
            while True:
                with self.lock:
                    if not self._outbox:
                        return
                    message = self._outbox.popleft()
                self.feed.order_update(message)

    def _working_books(self) -> list[Book]:
        """The books whose orders prices work now: the broker's always (the exchange
        holds them), the sandbox's while the switch is on and the broker answers its
        quote requests (sandbox/execution_engine.py, `_fetch_quote`)."""
        books = [self.books[Mode.LIVE]]
        if self.analyzer and self.broker_reachable:
            books.append(self.books[Mode.SANDBOX])
        return books

    def _work(
        self, book: Book, order: FakeOrder, market: MarketState, *, arriving: bool
    ) -> list[Change]:
        cap = self._split_next if book.mode is Mode.LIVE else None
        changes = book.work(order, market, cap, self.clock.now(), arriving=arriving)
        if any(change.event == FILLED for change in changes):
            self._split_next = None
        return changes

    def _publish_all(self, book: Book, changes: list[Change]) -> None:
        for change in changes:
            self._publish(book, change)

    def _publish(self, book: Book, change: Change) -> None:
        zero = False
        if change.fill is not None:
            fill = change.fill
            self.log.record(
                fill.ts_ns,
                FILL,
                fill.order_id,
                mode=book.mode.value,
                units=fill.quantity,
                price=fill.price,
                trade_id=fill.trade_id,
            )
            if book.mode is Mode.LIVE and self._zero_price_next:
                zero, self._zero_price_next = True, False
        message = order_update(
            change, book.mode, broker=self.broker, user_id=self.username, zero_price=zero
        )
        self._outbox.append(message)

    def _known(self, symbol: str, exchange: str) -> MarketState:
        market = self.market(symbol, exchange)
        if market is None:
            raise ValueError(f"the fake broker does not know {exchange}:{symbol}: add it first")
        return market


__all__ = ["Rejection", "TelegramMessage", "Venue"]
