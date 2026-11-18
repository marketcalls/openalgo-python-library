"""What the session's feed link needs of a feed (16-session-design 2.11, 3.1 step 14): the
`FeedAPI` of an `openalgo.api` client, or the fake OpenAlgo's direct feed in tests
(`FeedSource`), and the check that a source has it (`missing_calls`). Used by `feedlink`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

# The calls the link makes of its feed.
_FEED_CALLS = (
    "connect",
    "disconnect",
    "on_connect",
    "on_disconnect",
    "on_reconnect",
    "on_error",
    "remove_hook",
    "subscribe_quote",
    "subscribe_orders",
)


class FeedSource(Protocol):
    """What the link needs of a feed: the `FeedAPI` of an `openalgo.api` client (the
    session's own feed client), or the fake OpenAlgo's direct feed in tests. Besides the
    calls, the link reads `authenticated` (the feed is connected and its API key taken,
    now) and `ws_url` (named when the start fails)."""

    ws_url: str
    authenticated: bool

    def connect(self) -> bool:
        """Connect and authenticate; whether that succeeded."""

    def disconnect(self) -> None:
        """Disconnect, stop reconnecting and forget the subscriptions."""

    def on_connect(self, callback: Callable[[dict[str, Any]], None]) -> Callable:
        """Register a connect hook."""

    def on_disconnect(self, callback: Callable[[dict[str, Any]], None]) -> Callable:
        """Register a disconnect hook."""

    def on_reconnect(self, callback: Callable[[dict[str, Any]], None]) -> Callable:
        """Register a reconnect hook."""

    def on_error(self, callback: Callable[[BaseException, dict[str, Any]], None]) -> Callable:
        """Register an error hook."""

    def remove_hook(self, callback: Callable) -> bool:
        """Remove a hook; whether it was registered."""

    def subscribe_quote(
        self, instruments: list[dict[str, str]], on_data_received: Callable | None = None
    ) -> bool:
        """Subscribe instruments in quote mode; whether the frame was sent now."""

    def subscribe_orders(self, on_order_update: Callable | None = None) -> bool:
        """Subscribe the order stream; whether the frame was sent now."""


def missing_calls(source: Any) -> list[str]:
    """The feed calls and flags a link uses that `source` lacks, in order."""
    missing = [name for name in _FEED_CALLS if not callable(getattr(source, name, None))]
    if not hasattr(source, "authenticated"):
        missing.append("authenticated")
    if not isinstance(getattr(source, "ws_url", None), str):
        missing.append("ws_url")
    return missing


__all__ = ["FeedSource", "missing_calls"]
