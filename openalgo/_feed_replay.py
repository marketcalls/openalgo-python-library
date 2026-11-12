"""Which subscriptions the server accepted after a replay (private to openalgo.feed)."""

from collections.abc import Callable
from typing import Any, NamedTuple

# The modes in the order their frames are sent, by their get_subscriptions() key.
KEYS = ("ltp", "quote", "depth")
# How a reply names the mode of an accepted entry: the label, or the number.
_KEY_BY_LABEL = {
    "ltp": "ltp",
    "quote": "quote",
    "depth": "depth",
    "1": "ltp",
    "2": "quote",
    "3": "depth",
}


class ReplayTracker:
    """
    Follows the server's replies to one replay of the requested subscriptions.

    requested has the format of FeedAPI.get_subscriptions(). accepted has the
    same format and fills up as the replies come in; done turns True once
    every instrument, and the order stream when it was requested, has been
    answered either way.
    """

    def __init__(self, requested: dict[str, Any]) -> None:
        self._waiting = {
            key: [(entry["exchange"], entry["symbol"]) for entry in requested[key]] for key in KEYS
        }
        self._orders_waiting = bool(requested["orders"])
        self.accepted: dict[str, Any] = {"ltp": [], "quote": [], "depth": [], "orders": False}

    @property
    def done(self) -> bool:
        return not self._orders_waiting and not any(self._waiting.values())

    def subscribe_reply(self, reply: dict[str, Any]) -> None:
        """Take in a reply of type "subscribe"."""
        for entry in reply.get("subscriptions") or []:
            instrument = (entry.get("exchange"), entry.get("symbol"))
            key = self._answered_key(instrument, entry.get("mode"))
            if key is None:
                continue
            self._waiting[key].remove(instrument)
            if entry.get("status") == "success":
                self.accepted[key].append({"symbol": instrument[1], "exchange": instrument[0]})

    def orders_reply(self, reply: dict[str, Any]) -> None:
        """Take in a reply of type "subscribe_orders"."""
        if self._orders_waiting:
            self._orders_waiting = False
            self.accepted["orders"] = reply.get("status") == "success"

    def unanswered(self) -> dict[str, Any]:
        """What the server has not answered yet, in the get_subscriptions() format."""
        result: dict[str, Any] = {
            key: [{"symbol": symbol, "exchange": exchange} for exchange, symbol in waiting]
            for key, waiting in self._waiting.items()
        }
        result["orders"] = self._orders_waiting
        return result

    def _answered_key(self, instrument: tuple[Any, Any], mode: Any) -> str | None:
        """
        The mode an entry answers, or None when it answers nothing we wait for.

        The server names the mode only on an accepted entry. A refused one
        counts for the first mode still waiting for that instrument: replies
        arrive in the order the frames were sent.
        """
        if mode is not None:
            key = _KEY_BY_LABEL.get(str(mode).lower())
            return key if key is not None and instrument in self._waiting[key] else None
        return next((key for key in KEYS if instrument in self._waiting[key]), None)


class PendingReplay(NamedTuple):
    """A replay whose replies the feed awaits before it fires the reconnect hook."""

    tracker: ReplayTracker
    payload: dict[str, Any]  # what the reconnect hook receives, without "subscriptions"
    still_live: Callable[[], bool]  # False once the connection it belongs to is gone
