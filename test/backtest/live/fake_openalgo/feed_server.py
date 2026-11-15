"""OpenAlgo's WebSocket proxy as the SDK's socket sees it, built on the in-process fake
socket of the feed tests (test/feed_fakes.py, `FakeServer` and `FakeSocket`, reused as
they are).

`FeedServer` is installed as the `websocket` module of `openalgo.feed` (the fixture does
it), so the SDK's connect builds a `FakeSocket`. It answers the SDK's frames as
websocket_proxy/server.py does, and tells which socket is open and what it subscribed
from the frames the SDK sent on it.
"""

from __future__ import annotations

from typing import Any

from feed_fakes import KEY_REFUSED, FakeServer, FakeSocket

from .messages import auth_reply, orders_reply

_ORDER_ACTIONS = ("subscribe_orders", "unsubscribe_orders")


class FeedServer(FakeServer):
    """The proxy: its authentication reply names the broker and the user, a key other
    than `api_key` is refused, and the order-stream replies are the proxy's own words.
    Market-data subscriptions are answered by `FakeServer`, in the proxy's shape."""

    def __init__(self, api_key: str, broker: str, user_id: str) -> None:
        super().__init__()
        self._api_key = api_key
        self.auth_reply = auth_reply(broker, user_id)

    def reply_to(self, frame: dict[str, Any]) -> dict[str, Any] | None:
        """The proxy's reply to a frame the SDK sent (None: no reply)."""
        action = frame.get("action")
        if action == "authenticate" and frame.get("api_key") != self._api_key:
            return KEY_REFUSED
        if action in _ORDER_ACTIONS:
            return orders_reply(action)
        return super().reply_to(frame)

    def live_socket(self) -> FakeSocket | None:
        """The newest socket, when it is open and has authenticated; else None."""
        if not self.sockets:
            return None
        newest = self.sockets[-1]
        if newest.finished.is_set() or not newest.frames("authenticate"):
            return None
        return newest

    @staticmethod
    def subscriptions(sock: FakeSocket) -> tuple[set[tuple[str, str, int]], bool]:
        """What a socket subscribed: its (symbol, exchange, mode) entries and whether it
        takes order updates, from the frames the SDK sent on it, in order."""
        entries: set[tuple[str, str, int]] = set()
        orders = False
        for frame in list(sock.sent):
            action = frame.get("action")
            symbols = frame.get("symbols", [])
            keys = {(s["symbol"], s["exchange"], frame.get("mode")) for s in symbols}
            if action == "subscribe":
                entries |= keys
            elif action == "unsubscribe":
                entries -= keys
            elif action in _ORDER_ACTIONS:
                orders = action == "subscribe_orders"
        return entries, orders


__all__ = ["FeedServer"]
