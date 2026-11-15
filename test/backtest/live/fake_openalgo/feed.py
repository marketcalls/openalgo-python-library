"""The fake OpenAlgo's WebSocket feed: ticks and order updates sent to the SDK's real
`FeedAPI` on the open socket of `FeedServer`, with the stream's faults.

It sends on the newest socket that has authenticated, the way the proxy fans a tick out
to its subscribers (websocket_proxy/server.py), and loses what the proxy would lose: a
tick or an update sent while no socket is open, or to a socket that did not subscribe,
is not replayed. Every fault is logged when it is set or when it acts.

A message is delivered on the caller's thread and handed to the SDK's socket thread,
which processes it before the send returns. The SDK runs its callbacks on that thread,
which must only queue what they receive: a call into the fake from it raises.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

from feed_fakes import FakeSocket

from .feed_server import FeedServer
from .log import FAULT, ORDER_UPDATE, TICK, FakeLog
from .market import MarketState
from .messages import market_data


class Feed:
    """Sends ticks and order updates, applying the stream's faults. Times are UTC
    nanoseconds from `now`."""

    def __init__(
        self, server: FeedServer, log: FakeLog, now: Callable[[], int], broker: str
    ) -> None:
        self.server = server
        self._log = log
        self._now = now
        self._broker = broker
        self._sending = threading.Lock()
        self._dead = False
        self._gaps: dict[tuple[str, str], int] = {}
        self._silent = False
        self._held: list[dict[str, Any]] | None = None
        self._drop_next = 0
        self._duplicate_next = 0

    # -- what the venue sends ---------------------------------------------

    def tick(self, market: MarketState) -> None:
        """Send the instrument's last trade to every mode the open socket subscribed."""
        key = market.instrument.key
        detail = {"symbol": key[0], "exchange": key[1], "ltp": market.last_price}
        if self._dead:
            self._log.record(self._now(), TICK, "lost", reason="feed dead", **detail)
            return
        if self._gaps.get(key, 0) > 0:
            self._gaps[key] -= 1
            self._fault("tick_gap", **detail)
            self._log.record(self._now(), TICK, "lost", reason="tick gap", **detail)
            return
        sock = self.server.live_socket()
        entries = self.server.subscriptions(sock)[0] if sock is not None else set()
        modes = sorted(mode for symbol, exchange, mode in entries if (symbol, exchange) == key)
        if sock is None or not modes:
            self._log.record(self._now(), TICK, "not sent", reason="no subscriber", **detail)
            return
        for mode in modes:
            self._deliver(sock, market_data(market, mode, broker=self._broker, ts_ns=self._now()))
        self._log.record(self._now(), TICK, "sent", modes=modes, **detail)

    def order_update(self, message: dict[str, Any]) -> None:
        """Send one order update on the order stream, unless a fault takes it."""
        if self._silent:
            self._note(message, "lost", reason="order stream silent")
            return
        if self._held is not None:
            self._held.append(message)
            self._note(message, "held")
            return
        if self._drop_next:
            self._drop_next -= 1
            self._fault("drop_order_update", orderid=message["orderid"])
            self._note(message, "lost", reason="update dropped")
            return
        copies = 1
        if self._duplicate_next:
            self._duplicate_next -= 1
            self._fault("duplicate_order_update", orderid=message["orderid"])
            copies = 2
        for _ in range(copies):
            self._send_update(message)

    # -- the stream's faults ----------------------------------------------

    def set_dead(self, dead: bool) -> None:
        """Stop sending ticks while the connection stays up (True), or start again."""
        self._dead = dead
        self._fault("feed_dead" if dead else "feed_alive")

    def gap(self, key: tuple[str, str], trades: int) -> None:
        """The next `trades` trades of an instrument send no tick. Raises ValueError for
        fewer than 1."""
        if trades < 1:
            raise ValueError(f"a tick gap is at least 1 trade, not {trades}")
        self._gaps[key] = self._gaps.get(key, 0) + trades

    def set_silent(self, silent: bool) -> None:
        """Lose every order update from now (True: the stream has gone silent), or send
        them again."""
        self._silent = silent
        self._fault("order_updates_silent" if silent else "order_updates_resumed")

    def drop_next_update(self) -> None:
        """The next order update is lost."""
        self._drop_next += 1

    def duplicate_next_update(self) -> None:
        """The next order update is sent twice."""
        self._duplicate_next += 1

    def hold_updates(self) -> None:
        """Keep order updates back until `release_updates`."""
        if self._held is None:
            self._held = []
            self._fault("order_updates_held")

    def release_updates(self, *, newest_first: bool = False) -> int:
        """Send the held order updates, oldest first (newest first when asked: out of
        order), and stop holding. Returns how many were held."""
        held, self._held = self._held or [], None
        self._fault("order_updates_released", count=len(held), newest_first=newest_first)
        for message in reversed(held) if newest_first else held:
            self._send_update(message)
        return len(held)

    def drop_connection(self, *, hold_reconnect: bool) -> None:
        """Drop the open socket; the SDK reconnects at once, or only after
        `let_reconnect` when `hold_reconnect`. Raises ValueError with no open socket."""
        sock = self.server.live_socket()
        if sock is None:
            raise ValueError("the feed has no open connection to drop")
        if hold_reconnect:
            self.server.gate.clear()
        self._fault("drop_feed", hold_reconnect=hold_reconnect)
        sock.drop()

    def let_reconnect(self) -> None:
        """Let a held reconnect through."""
        self._fault("feed_reconnect_allowed")
        self.server.gate.set()

    # -- delivery -----------------------------------------------------------

    def refuse_feed_thread(self) -> None:
        """Raise RuntimeError when called on a socket thread of the SDK's feed, where
        callbacks run: they must only queue, and a call into the fake from there could
        wait on a delivery that waits on that very thread."""
        if any(threading.current_thread() is sock.thread for sock in list(self.server.sockets)):
            raise RuntimeError(
                "the fake OpenAlgo was called from the feed's thread: a feed callback must "
                "only queue what it receives"
            )

    def _send_update(self, message: dict[str, Any]) -> None:
        sock = self.server.live_socket()
        if sock is None or not self.server.subscriptions(sock)[1]:
            self._note(message, "lost", reason="no order-stream subscriber")
            return
        self._deliver(sock, message)
        self._note(message, "sent")

    def _deliver(self, sock: FakeSocket, message: dict[str, Any]) -> None:
        self.refuse_feed_thread()
        with self._sending:
            sock.deliver(message)

    def _fault(self, name: str, **detail: Any) -> None:
        self._log.record(self._now(), FAULT, name, **detail)

    def _note(self, message: dict[str, Any], what: str, **detail: Any) -> None:
        self._log.record(
            self._now(),
            ORDER_UPDATE,
            what,
            orderid=message["orderid"],
            order_status=message["order_status"],
            message=dict(message),
            **detail,
        )


__all__ = ["Feed"]
