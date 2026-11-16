"""OpenAlgo's WebSocket proxy as the SDK's socket sees it, built on the in-process fake
socket of the feed tests (test/feed_fakes.py, `FakeServer` and `FakeSocket`, reused as
they are).

`FeedServer` is installed as the `websocket` module of `openalgo.feed` (the fixture does
it), so the SDK's connect builds a `FakeSocket`. It answers the SDK's frames as
websocket_proxy/server.py does: the key is checked, a subscription to an instrument the
broker does not know is refused in the broker adapter's words
(broker/zerodha/streaming/zerodha_adapter.py:282-284) and never ticks, and the
order-stream replies are the proxy's own. It tells which sockets are open and
authenticated, and what each subscribed, from the frames the SDK sent on it and what it
answered them.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from feed_fakes import KEY_REFUSED, FakeServer, FakeSocket

from .messages import auth_reply, orders_reply, subscribe_reply, token_not_found

_ORDER_ACTIONS = ("subscribe_orders", "unsubscribe_orders")


class FeedServer(FakeServer):
    """The proxy for one user: `api_key` is the only key it accepts, `knows(symbol,
    exchange)` says whether the broker has the instrument."""

    def __init__(
        self, api_key: str, broker: str, user_id: str, knows: Callable[[str, str], bool]
    ) -> None:
        super().__init__()
        self._api_key = api_key
        self._broker = broker
        self._knows = knows
        self.auth_reply = auth_reply(broker, user_id)
        # What the proxy answered each frame, by the frame's identity: a frame stays in
        # its socket's `sent` list, so the identity holds while the socket lives.
        self._accepted: dict[int, set[tuple[str, str, int]]] = {}
        self._authenticated: set[int] = set()

    def reply_to(self, frame: dict[str, Any]) -> dict[str, Any] | None:
        """The proxy's reply to a frame the SDK sent (None: no reply)."""
        action = frame.get("action")
        if action == "authenticate":
            if frame.get("api_key") != self._api_key:
                return KEY_REFUSED
            self._authenticated.add(id(frame))
            return self.auth_reply
        if action == "subscribe" and self.answer_subscribe:
            return self._subscribed(frame)
        if action in _ORDER_ACTIONS:
            return orders_reply(action)
        return super().reply_to(frame)

    def open_sockets(self) -> list[FakeSocket]:
        """Every socket that is open and whose key the proxy accepted, oldest first."""
        return [
            sock
            for sock in list(self.sockets)
            if not sock.finished.is_set()
            and any(id(frame) in self._authenticated for frame in sock.frames("authenticate"))
        ]

    def live_socket(self) -> FakeSocket | None:
        """The newest socket, when it is open and authenticated; else None."""
        if not self.sockets:
            return None
        newest = self.sockets[-1]
        return newest if newest in self.open_sockets() else None

    def subscriptions(self, sock: FakeSocket) -> tuple[set[tuple[str, str, int]], bool]:
        """What a socket subscribed: its accepted (symbol, exchange, mode) entries and
        whether it takes order updates, from its frames in order."""
        entries: set[tuple[str, str, int]] = set()
        orders = False
        for frame in list(sock.sent):
            action = frame.get("action")
            if action == "subscribe":
                entries |= self._accepted.get(id(frame), set())
            elif action == "unsubscribe":
                mode = frame.get("mode")
                entries -= {(s["symbol"], s["exchange"], mode) for s in frame.get("symbols", [])}
            elif action in _ORDER_ACTIONS:
                orders = action == "subscribe_orders"
        return entries, orders

    def _subscribed(self, frame: dict[str, Any]) -> dict[str, Any]:
        mode = frame.get("mode")
        accepted: set[tuple[str, str, int]] = set()
        entries = []
        for instrument in frame.get("symbols", []):
            symbol, exchange = instrument["symbol"], instrument["exchange"]
            refusal = "" if self._knows(symbol, exchange) else token_not_found(symbol, exchange)
            if not refusal:
                accepted.add((symbol, exchange, mode))
            entries.append((symbol, exchange, mode, refusal))
        self._accepted[id(frame)] = accepted
        return subscribe_reply(entries, self._broker)


__all__ = ["FeedServer"]
