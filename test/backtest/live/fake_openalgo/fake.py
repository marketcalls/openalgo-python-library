"""`FakeOpenAlgo`: an in-process OpenAlgo server, broker and sandbox for the live layer's
tests, with every fault the fault suite schedules.

A test builds one (through the `fake_openalgo` fixture, which installs its WebSocket
proxy and fails the test on any problem), gets real SDK clients from `client()`, pushes
prices with `trade`, and drives the broker and the switches. The live layer runs against
it unchanged: REST through the SDK's own HTTP client on an in-process transport, ticks
and order updates through the SDK's own `FeedAPI` on the feed tests' in-process socket.
Nothing opens a socket.

Times are UTC nanoseconds on the fake's manual clock, prices rupees, quantities units
(never lots).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Iterable
from typing import Any

import httpx
from feed_fakes import retire, shorten

from openalgo import api
from openalgo.backtest._live._types import Mode

from .clock import ManualClock
from .direct_feed import DirectFeed
from .faults import Faults
from .feed import Feed
from .feed_server import FeedServer
from .gtts import FakeGtt
from .log import FakeLog
from .market import Candle, Instrument
from .orders import FakeOrder, Fill
from .rest import CLIENT_HEADER, ROUTES, RestHandler
from .venue import Rejection, TelegramMessage, Venue

#: The fake's API key: a placeholder, never a real key.
API_KEY = "fake-openalgo-key-not-a-real-one"
#: The host the clients are built for: a name that cannot resolve, so a request that
#: escaped the in-process transport could reach nothing.
HOST = "http://openalgo.invalid"
#: The broker OpenAlgo is logged in to, and the OpenAlgo user.
BROKER = "zerodha"
USERNAME = "trader1"
#: Each book's available cash, rupees: a broker account and the sandbox's default pool.
LIVE_CASH = 250_000.0
SANDBOX_CASH = 10_000_000.0
#: The feed's waits made short for tests, seconds (the SDK's one test seam).
_FEED_TIMING = {"poll": 0.002, "backoff": (0.01,), "backoff_slice": 0.005}


class FakeOpenAlgo:
    """The fake server. `start_ns` starts its clock (UTC nanoseconds); `analyzer` is the
    analyzer switch at the start (False: live mode); `gtt` whether OpenAlgo ships GTT
    orders for the broker (False: its GTT calls answer HTTP 501 in live mode), and
    `gtt_history` whether the broker's GTT book keeps the GTTs that ended (False: it lists
    the active ones whatever is asked, as four of the seven brokers with GTT orders do;
    the sandbox's book always keeps them).

    Attributes: `clock` (the manual clock to hand the live layer as `Now` and `Sleep`),
    `log` (everything that happened), `faults` (the faults a test schedules) and
    `feed_server` (the WebSocket proxy, installed by the fixture).
    """

    def __init__(
        self,
        *,
        start_ns: int,
        analyzer: bool = False,
        broker: str = BROKER,
        username: str = USERNAME,
        live_cash: float = LIVE_CASH,
        sandbox_cash: float = SANDBOX_CASH,
        gtt: bool = True,
        gtt_history: bool = True,
    ) -> None:
        self.clock = ManualClock(start_ns)
        self.log = FakeLog()
        self.feed_server = FeedServer(API_KEY, broker, username, knows=self._knows)
        feed = Feed(self.feed_server, self.log, self.clock.now, broker)
        self._venue = Venue(
            clock=self.clock,
            log=self.log,
            feed=feed,
            broker=broker,
            username=username,
            live_cash=live_cash,
            sandbox_cash=sandbox_cash,
            analyzer=analyzer,
            gtt_supported=gtt,
            gtt_history=gtt_history,
        )
        self.faults = Faults(self._venue, feed, frozenset(ROUTES))
        self._handler = RestHandler(self._venue, self.faults, API_KEY)
        self._clients: list[api] = []
        #: Checks to run when the test ends (`at_test_end`), oldest first.
        self.end_checks: list[Callable[[], None]] = []

    # -- wiring -----------------------------------------------------------

    def client(self, *, api_key: str = API_KEY, auto_reconnect: bool = True) -> api:
        """A real `openalgo.api` whose REST calls this fake answers in-process, and whose
        feed connects to its proxy (installed by the fixture). Closed by `close`. A key
        other than `API_KEY` is refused as the server refuses an unknown one.

        Each client has a name (`client_name`) its calls carry: the log notes it with
        every call, and an order keeps the name of the client that placed it
        (`FakeOrder.client`), so a test can tell the session's calls from another
        client's."""
        client = api(api_key=api_key, host=HOST, auto_reconnect=auto_reconnect)
        client.client.close()
        name = f"client-{len(self._clients) + 1}"
        client.client = httpx.Client(
            transport=httpx.MockTransport(self._handler), headers={CLIENT_HEADER: name}
        )
        shorten(client, **_FEED_TIMING)
        self._clients.append(client)
        return client

    def owns(self, client: api) -> bool:
        """True for a client this fake made (`client`)."""
        return any(client is made for made in self._clients)

    def direct_feed(self) -> DirectFeed:
        """A feed the fake calls directly, with the part of the SDK's `FeedAPI` a live
        session uses: it delivers ticks (quote mode) and order updates synchronously on
        the thread of the fake's call that caused them, in a fixed order, with no socket
        and no thread (see `direct_feed`). Each call makes a new one."""
        venue = self._venue
        direct = DirectFeed(
            broker=venue.broker,
            user_id=venue.username,
            knows=self._knows,
            refusing=lambda: venue.feed.refusing,
        )
        venue.feed.add_direct(direct)
        return direct

    def refuse_feed(self) -> None:
        """Refuse every new feed connection from now, of the SDK's socket and of a direct
        feed alike, until `accept_feed`: the SDK's `connect()` answers False and its own
        retries fail too (a connection already open stays open)."""
        self._venue.feed.refuse_connections(True)

    def accept_feed(self) -> None:
        """Accept feed connections again after `refuse_feed`: the SDK's own retry gets
        through, and a direct feed whose last `connect()` was refused (and that was not
        disconnected since) connects at once, as that retry would."""
        self._venue.feed.refuse_connections(False)

    def close(self) -> None:
        """Retire every client: their feed threads end and nothing reconnects."""
        self.feed_server.answer_close = True
        self.feed_server.gate.set()
        for client in self._clients:
            retire(client)
        self._clients.clear()

    def at_test_end(self, check: Callable[[], None]) -> None:
        """Run `check` when the test that built this fake ends (the fixture runs each, in
        order, before it retires the clients); a check fails the test by raising."""
        self.end_checks.append(check)

    @property
    def mode(self) -> Mode:
        """The mode the analyzer switch selects now."""
        return self._venue.mode

    def _knows(self, symbol: str, exchange: str) -> bool:
        """Whether the broker knows an instrument (for the proxy's subscriptions)."""
        with self._venue.lock:
            return self._venue.market(symbol, exchange) is not None

    # -- the market -------------------------------------------------------

    def add_instrument(
        self,
        symbol: str,
        exchange: str,
        *,
        lot_size: int = 1,
        tick_size: float = 0.05,
        freeze_qty: int = 0,
        instrument_type: str = "EQ",
        expiry: str = "",
        strike: float = 0.0,
    ) -> None:
        """Make an instrument known to the broker and OpenAlgo's symbol table (sizes in
        units, prices in rupees; see `market.Instrument`)."""
        instrument = Instrument(
            symbol, exchange, lot_size, tick_size, freeze_qty, instrument_type, expiry, strike
        )
        self._venue.add_instrument(instrument)

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
        """A trade of `volume` units at `price` rupees now: orders are worked, the candle
        grows and a tick is sent to the subscribers (`bid` and `ask`: the best quotes,
        else the price)."""
        self._venue.trade(symbol, exchange, price, volume=volume, bid=bid, ask=ask)

    def seed_candles(
        self,
        symbol: str,
        exchange: str,
        candles: Iterable[tuple[int, float, float, float, float, float]],
    ) -> None:
        """Put in broker candles of the past for `history`: each (start UTC nanoseconds on
        the minute, open, high, low, close, volume in units)."""
        with self._venue.lock:
            market = self._venue.market(symbol, exchange)
            if market is None:
                raise ValueError(f"the fake broker does not know {exchange}:{symbol}")
            for start, open_, high, low, close, volume in candles:
                market.seed(Candle(start, open_, high, low, close, volume))

    # -- the broker -------------------------------------------------------

    def fill(self, order_id: str, units: int | None = None, *, price: float | None = None) -> None:
        """Fill a working order of either book now: all that is pending (or `units` of
        it, live only), at the last price (or `price`). Raises ValueError for an unknown
        or finished order or when no price has traded yet."""
        self._venue.fill(order_id, units, price)

    def split_next_fill(self, units: int) -> None:
        """The next fill of a live order takes only `units`; the rest fills at a later
        price. Raises ValueError in sandbox mode, which fills whole orders only."""
        self._venue.split_next_fill(units)

    def reject_next(
        self, reason: str, *, price_type: str | None = None, at_placement: bool = False
    ) -> None:
        """The broker refuses the next order (of `price_type` when given) with `reason`:
        at placement (the place call answers HTTP 400; in sandbox mode with the order ID
        of the rejected order it keeps, sandbox/order_manager.py:670-680) or, by default,
        right after accepting it (an order ID, then a rejected update). Raises
        ValueError for a rejection after acceptance in sandbox mode, which refuses only
        at placement."""
        self._venue.schedule_rejection(Rejection(reason, price_type, at_placement))

    def lag_positionbook(self) -> None:
        """The position book answers the positions as they stand now until
        `catch_up_positionbook`: fills from now on do not show in it, while the order and
        trade books show them (a broker's position book lagging its fills,
        16-session-design F78)."""
        with self._venue.lock:
            self._venue.lag_positions()

    def catch_up_positionbook(self) -> None:
        """The position book shows every fill again after `lag_positionbook`."""
        with self._venue.lock:
            self._venue.catch_up_positions()

    def set_master_expiry(self, symbol: str, exchange: str, expiry: str) -> None:
        """Make OpenAlgo's master contract give a known instrument `expiry` ("DD-MMM-YY",
        or "" for none)."""
        market = self._venue.markets[(symbol, exchange)]
        market.instrument = dataclasses.replace(market.instrument, expiry=expiry)

    def set_holding(
        self,
        symbol: str,
        exchange: str,
        quantity: int,
        *,
        average_price: float = 0.0,
        mode: Mode | None = None,
    ) -> None:
        """Hold `quantity` units of a known instrument for delivery (CNC) at
        `average_price` rupees in a book (the selected one by default), as `holdings`
        answers them; 0 units holds none. Raises ValueError for an unknown instrument."""
        with self._venue.lock:
            if self._venue.market(symbol, exchange) is None:
                raise ValueError(f"the fake broker does not know {exchange}:{symbol}")
            book = self._venue.books[mode or self.mode]
            book.holdings[(symbol, exchange)] = (quantity, average_price)

    def end_day(self) -> None:
        """A day boundary in both books (`book.Book.end_day`): each CNC position's units
        move into the holdings, NRML positions carry over, the day's orders and trades are
        cleared, and GTTs stay. The test moves the clock to the next session itself.
        Raises ValueError for an MIS position still open."""
        with self._venue.lock:
            for book in self._venue.books.values():
                book.end_day()

    # -- GTT orders ---------------------------------------------------------

    def refuse_next_gtt(self, reason: str, *, call: str = "placegttorder") -> None:
        """The broker (or the sandbox) refuses the next GTT `call` ("placegttorder",
        "modifygttorder" or "cancelgttorder") with `reason`, HTTP 400: a trigger too near
        the price, a segment it takes no GTT on, whatever its words. Raises ValueError for
        another call."""
        with self._venue.lock:
            self._venue.refuse_next_gtt(call, reason)

    def half_apply_next_gtt_modify(self) -> None:
        """The next GTT modify changes the trigger only, keeps the units, and is answered
        as the server's internal error (HTTP 500): a broker module that failed part way.
        Raises ValueError in sandbox mode, whose modify is all or nothing."""
        with self._venue.lock:
            if self._venue.mode is Mode.SANDBOX:
                raise ValueError("the OpenAlgo sandbox changes a GTT all at once or not at all")
            self._venue.half_apply_next_gtt_modify()

    def set_gtt_status(self, trigger_id: str, status: str) -> None:
        """What the broker or the user does to a resting GTT outside the session's calls:
        its row shows `status` from now ("cancelled" by hand, "expired", "rejected" after
        placement, "deleted", "disabled", "transit", or any word), and no child is placed.
        Raises ValueError for an unknown trigger ID or a GTT that is not active."""
        with self._venue.lock:
            self._venue.set_gtt_status(trigger_id, status)

    def gtts(self, mode: Mode | None = None) -> list[FakeGtt]:
        """Copies of a book's GTTs (the selected one by default), ended ones included,
        oldest first."""
        with self._venue.lock:
            book = self._venue.books[mode or self.mode]
            return [dataclasses.replace(gtt) for gtt in book.gtts.values()]

    def set_holidays(self, rows: Iterable[dict[str, Any]]) -> None:
        """Replace OpenAlgo's market holidays (`market/holidays` rows, as
        `holidays.HOLIDAYS_2026` holds them)."""
        self._venue.holidays = tuple(dict(row) for row in rows)

    def set_analyzer(self, on: bool) -> None:
        """Flip OpenAlgo's analyzer switch (True: sandbox mode)."""
        self._venue.set_switch("analyzer", on)

    def set_semi_auto(self, on: bool) -> None:
        """Turn semi-auto (Action Center) mode on or off."""
        self._venue.set_switch("semi_auto", on)

    def set_telegram_bot(self, on: bool) -> None:
        """Start or stop OpenAlgo's Telegram bot."""
        self._venue.set_switch("telegram_bot", on)

    # -- what the test reads ------------------------------------------------

    def orders(self, mode: Mode | None = None) -> list[FakeOrder]:
        """Copies of a book's orders (the selected one by default), oldest first."""
        book = self._venue.books[mode or self.mode]
        with self._venue.lock:
            return [dataclasses.replace(order) for order in book.orders.values()]

    def fills(self, mode: Mode | None = None) -> list[Fill]:
        """A book's fills (the selected one by default), oldest first."""
        with self._venue.lock:
            return list(self._venue.books[mode or self.mode].fills)

    def position(self, symbol: str, exchange: str, mode: Mode | None = None) -> int:
        """A book's net position in an instrument, signed units."""
        with self._venue.lock:
            return self._venue.books[mode or self.mode].positions.net(symbol, exchange)

    @property
    def telegram_sent(self) -> list[TelegramMessage]:
        """The notifications the Telegram endpoint took, oldest first."""
        with self._venue.lock:
            return list(self._venue.telegram_sent)


def client_name(client: api) -> str:
    """The name of a client `FakeOpenAlgo.client` made, which its calls carry ("" for a
    client of no fake)."""
    return client.client.headers.get(CLIENT_HEADER, "")


__all__ = [
    "API_KEY",
    "BROKER",
    "HOST",
    "LIVE_CASH",
    "SANDBOX_CASH",
    "USERNAME",
    "FakeOpenAlgo",
    "client_name",
]
