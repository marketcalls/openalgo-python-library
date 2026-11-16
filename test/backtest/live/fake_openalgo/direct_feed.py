"""`DirectFeed`: the fake OpenAlgo's feed handed straight to the code under test
(16-session-design 2.11 and 7.3), with the part of the SDK's `FeedAPI` a live session
uses and no socket and no thread.

Its surface: `connect`, `disconnect`, the hooks `on_connect`, `on_disconnect`,
`on_reconnect` and `on_error` with `remove_hook`, `subscribe_quote`, `subscribe_orders`,
`get_subscriptions`, and the flags `connected` and `authenticated`. What it hands over
is what the SDK's callbacks and hooks receive, built with the SDK's own shaping
(`openalgo._feed_ticks`, `openalgo._feed_hooks`, `openalgo._feed_errors`): a quote
tick, an order update, the connect and disconnect payloads with their connection number
`seq`, and an instrument the broker does not know refused as the server refuses it.

It delivers synchronously, on the thread of the fake's call that caused it (`trade`, an
order call, a fill, a released update), in the order the fake makes them: a trade's
tick, then the order updates the trade caused. Every fault of the stream and of the
market data acts on it as on the socket (`feed.Feed`), and `FakeOpenAlgo.refuse_feed`
refuses its connections. While it delivers, a call into the fake from the same thread is
refused as one from the socket's thread is: a callback must only queue. It never drops
by itself: a dropped connection is the socket's (`Faults.drop_feed`).

A `connect()` the fake refused is retried as the SDK retries one, but at once: when the
fake accepts connections again (`FakeOpenAlgo.accept_feed`), a direct feed whose last
`connect()` was refused, and that was not disconnected since, connects.

Prices are rupees, quantities units, times as the proxy's messages carry them.
"""

from __future__ import annotations

import contextlib
import logging
import threading
from collections.abc import Callable, Iterator
from typing import Any

from openalgo import _feed_ticks as sdk_ticks
from openalgo._feed_errors import server_rejection
from openalgo._feed_hooks import fresh_copy

from .messages import QUOTE, subscribe_reply, token_not_found

#: The URL a direct feed reports: it opens nothing.
DIRECT_URL = "direct://fake-openalgo"
#: The depth the SDK asks for in a subscribe frame.
_DEPTH_LEVELS = 5
_EVENTS = ("connect", "disconnect", "reconnect", "error")

# Where the SDK writes the feed's errors that no error hook takes.
feed_logger = logging.getLogger("openalgo.feed")


class DirectFeed:
    """A feed that the fake calls directly. `broker` and `user_id` are what the proxy's
    authentication reply names; `knows(symbol, exchange)` says whether the broker has an
    instrument; `refusing()` whether connections are refused now.

    `frames` holds what the code under test asked of the feed, as the frames the SDK
    would send for it (`subscribe`, `subscribe_orders`), in order.
    """

    def __init__(
        self,
        *,
        broker: str,
        user_id: str,
        knows: Callable[[str, str], bool],
        refusing: Callable[[], bool],
    ) -> None:
        self.ws_url = DIRECT_URL
        self.connected = False
        self.authenticated = False
        self.frames: list[dict[str, Any]] = []
        self._broker, self._user_id = broker, user_id
        self._knows, self._refusing = knows, refusing
        self._seq = 0
        self._hooks: dict[str, list[Callable]] = {event: [] for event in _EVENTS}
        # Requested subscriptions by (symbol, exchange), and those the broker accepted on
        # the connection that is up.
        self._quotes: dict[tuple[str, str], dict[str, str]] = {}
        self._accepted: set[tuple[str, str]] = set()
        self._orders = False
        self._quote_callback: Callable | None = None
        self._order_callback: Callable | None = None
        self._retrying = False  # the last connect() was refused, and no disconnect since
        self._here = threading.local()

    # -- the FeedAPI surface a session uses -----------------------------------

    def connect(self) -> bool:
        """Connect: refused (False, nothing fires, and it is retried when the fake
        accepts again) while the fake refuses connections; else the connection gets the
        next `seq`, the connect hooks fire, and the subscriptions requested before are
        answered, as the SDK sends them on authentication. True while connected."""
        if self.authenticated:
            return True
        self._retrying = self._refusing()
        if self._retrying:
            return False
        self._seq += 1
        self.connected = self.authenticated = True
        self._post(
            "connect",
            {
                "event": "connect",
                "seq": self._seq,
                "url": DIRECT_URL,
                "reconnect": False,
                "broker": self._broker,
                "user_id": self._user_id,
            },
        )
        if self._quotes:
            self._send_subscribe(list(self._quotes.values()))
        if self._orders:
            self.frames.append({"action": "subscribe_orders"})
        return True

    def disconnect(self) -> None:
        """Disconnect, as the SDK's `disconnect()` does: the requested subscriptions are
        forgotten, and a connection that was up fires the disconnect hooks with
        `requested` True."""
        was_up = self.authenticated
        self.connected = self.authenticated = self._retrying = False
        self._quotes.clear()
        self._accepted.clear()
        self._orders = False
        if was_up:
            self._post(
                "disconnect",
                {
                    "event": "disconnect",
                    "seq": self._seq,
                    "url": DIRECT_URL,
                    "code": None,
                    "reason": "disconnect() called",
                    "error": None,
                    "requested": True,
                    "will_reconnect": False,
                },
            )

    def on_connect(self, callback: Callable) -> Callable:
        """Register a connect hook; returns it."""
        return self._add("connect", callback)

    def on_disconnect(self, callback: Callable) -> Callable:
        """Register a disconnect hook; returns it."""
        return self._add("disconnect", callback)

    def on_reconnect(self, callback: Callable) -> Callable:
        """Register a reconnect hook (a direct feed never reconnects); returns it."""
        return self._add("reconnect", callback)

    def on_error(self, callback: Callable) -> Callable:
        """Register an error hook, called as `callback(error, context)`; returns it."""
        return self._add("error", callback)

    def remove_hook(self, callback: Callable) -> bool:
        """Remove a hook from every event; whether it was registered."""
        removed = False
        for listeners in self._hooks.values():
            while callback in listeners:
                listeners.remove(callback)
                removed = True
        return removed

    def subscribe_quote(
        self, instruments: list[dict[str, Any]], on_data_received: Callable | None = None
    ) -> bool:
        """Request quote ticks of `instruments` (the callback, when given, replaces the
        last). Whether it was sent now: False while not connected, when it is sent on
        `connect`. An instrument the broker does not know is refused through the error
        hooks and never ticks."""
        if on_data_received is not None:
            self._quote_callback = on_data_received
        valid = [
            {"symbol": entry["symbol"], "exchange": entry["exchange"]}
            for entry in instruments
            if entry.get("symbol") and entry.get("exchange")
        ]
        for entry in valid:
            self._quotes[(entry["symbol"], entry["exchange"])] = entry
        if not (self.authenticated and valid):
            return False
        self._send_subscribe(valid)
        return True

    def subscribe_orders(self, on_order_update: Callable | None = None) -> bool:
        """Request the order stream (the callback, when given, replaces the last).
        Whether it was sent now: False while not connected."""
        if on_order_update is not None:
            self._order_callback = on_order_update
        self._orders = True
        if not self.authenticated:
            return False
        self.frames.append({"action": "subscribe_orders"})
        return True

    def get_subscriptions(self) -> dict[str, Any]:
        """What was requested, in the SDK's `get_subscriptions()` shape."""
        return {
            "ltp": [],
            "quote": [dict(entry) for entry in self._quotes.values()],
            "depth": [],
            "orders": self._orders,
        }

    # -- what the fake's feed calls -------------------------------------------

    def wants_quote(self, key: tuple[str, str]) -> bool:
        """Whether a tick of the instrument `key` (symbol, exchange) goes to it now."""
        return self.authenticated and key in self._accepted

    def wants_orders(self) -> bool:
        """Whether order updates go to it now."""
        return self.authenticated and self._orders

    def deliver_quote(self, message: dict[str, Any]) -> None:
        """Hand the proxy's quote-mode message to the quote callback, shaped as the SDK's
        callback receives it."""
        update = sdk_ticks.quote_update(message.get("data", {}))
        payload = sdk_ticks.callback_message(message, update)
        self._call("quote", self._quote_callback, payload, message)

    def deliver_update(self, message: dict[str, Any]) -> None:
        """Hand an order update to the order callback, as its own copy."""
        self._call("order", self._order_callback, fresh_copy(message), message)

    def connections_accepted(self) -> None:
        """The fake accepts connections again: connect now when the last `connect()` was
        refused and the feed was not disconnected since, as the SDK's own retry does."""
        if self._retrying:
            self.connect()

    def delivering_here(self) -> bool:
        """Whether this thread is inside one of its deliveries now."""
        return getattr(self._here, "depth", 0) > 0

    # -- inside ---------------------------------------------------------------

    def _add(self, event: str, callback: Callable) -> Callable:
        if not callable(callback):
            raise TypeError(f"on_{event} expects a callable, got {type(callback).__name__}")
        self._hooks[event].append(callback)
        return callback

    def _send_subscribe(self, entries: list[dict[str, str]]) -> None:
        """Send a subscribe frame and take the broker's answer: known instruments tick
        from now, the others are refused through the error hooks."""
        frame = {
            "action": "subscribe",
            "symbols": [dict(entry) for entry in entries],
            "mode": QUOTE,
            "depth": _DEPTH_LEVELS,
        }
        self.frames.append(frame)
        rows = []
        for entry in entries:
            symbol, exchange = entry["symbol"], entry["exchange"]
            known = self._knows(symbol, exchange)
            if known:
                self._accepted.add((symbol, exchange))
            refusal = "" if known else token_not_found(symbol, exchange)
            rows.append((symbol, exchange, QUOTE, refusal))
        reply = subscribe_reply(rows, self._broker)
        rejection = server_rejection(reply)
        if rejection is not None:
            kind, error, details = rejection
            context = {"kind": kind, "callback": None, "code": error.code, "message": reply}
            self._report(error, {**context, **details})

    def _call(self, label: str, callback: Callable | None, payload: Any, message: Any) -> None:
        if callback is None:
            return
        with self._delivering():
            try:
                callback(payload)
            # As the SDK: a callback that raises reaches the error hooks, never the fake.
            except Exception as error:
                self._report(error, {"kind": "callback", "callback": label, "message": message})

    def _post(self, event: str, payload: dict[str, Any]) -> None:
        """Deliver a hook's payload to each listener in turn, each its own copy."""
        with self._delivering():
            for listener in list(self._hooks[event]):
                try:
                    listener(fresh_copy(payload))
                # As the SDK: a hook that raises reaches the error hooks.
                except Exception as error:
                    context = {"kind": "hook", "callback": f"on_{event}", "message": payload}
                    self._report(error, context)

    def _report(self, error: BaseException, context: dict[str, Any]) -> None:
        """Hand an error to the error hooks, or write it to the feed's log as the SDK does
        when none is registered; an error hook that raises is written there too."""
        context = {**context, "seq": self._seq}
        listeners = list(self._hooks["error"])
        if not listeners:
            feed_logger.error("OpenAlgo feed (direct): %s: %s", type(error).__name__, error)
            return
        with self._delivering():
            for listener in listeners:
                try:
                    listener(error, fresh_copy(context))
                except Exception as failure:
                    feed_logger.error("OpenAlgo feed (direct): an error hook raised: %r", failure)

    @contextlib.contextmanager
    def _delivering(self) -> Iterator[None]:
        self._here.depth = getattr(self._here, "depth", 0) + 1
        try:
            yield
        finally:
            self._here.depth -= 1


__all__ = ["DIRECT_URL", "DirectFeed"]
