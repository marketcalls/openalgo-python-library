"""The session's side of the WebSocket feed (16-session-design 2.3, 2.5 A.12, 3.1 step
14, 6.5): the feed client's callbacks and hooks, the connection, the subscriptions, and
the spans in which the feed was down.

Threads. The callbacks and hooks run on the feed's threads (the SDK's socket thread, its
timer, or the thread that called `disconnect`) and only queue and stamp: a quote
callback puts the market-data message on the loop's inbox as a `FeedTick` with the clock
read when it ran; the order callback puts the order update on the order worker's
`RawUpdates` as a `RawOrderUpdate`; each hook puts a `FeedEvent` with the hook's payload
and its connection number `seq`. They take no lock, make no call to the feed and never
wait: both queues are unbounded. `connect`, `note` and `close` run on the loop's thread.

Outages. The loop hands each `FeedEvent` it takes to `note`, in order. A disconnect the
session did not ask for opens an outage at the loop's clock (a second drop while one is
open keeps its start), and the reconnect that restores the session closes it: `note`
returns the `Outage`, for the loop to `resume` every bar builder over it and to ask the
order worker for `stream_gap(outage.start_ns)`. The ticks' receive times come from the
same clock as the loop takes them, so no tick taken before the disconnect or after the
reconnect falls inside the span. A feed that stops ticking while its connection stays up
is dead, not down: the bars' `FeedHealth` finds that from the ticks (6.5).

Subscriptions. `connect` registers the hooks, connects and subscribes the instruments in
quote mode (08-live-trading section 9 step 6) and the order stream. After a reconnect
the SDK replays them itself; what the server did not take back in the replay (it refused
it, or did not answer in time) `note` subscribes again, once per reconnect.

Times are UTC nanoseconds.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol

from .._layout import NANOS_PER_SECOND
from ..errors import ConfigError, LiveError
from ._preflight_plan import positive_amount
from ._recorder_args import instruments_of
from ._tick_spans import Outage
from ._types import LOGGER_NAME, Now, Sleep
from .inbox import FeedEvent, FeedEventKind, FeedTick, Inbox, RawOrderUpdate, RawUpdates

#: Seconds the feed has to connect at the session's start before the start fails
#: (16-session-design 1.2 and 3.1 step 14).
FEED_CONNECT_TIMEOUT_S = 30.0
#: Seconds between two looks at whether the feed's own retries have connected it: the
#: loop's longest wait (16-session-design 2.5 G), so a session that waits for its feed
#: through its own sleep keeps its loop turning as often as it always does.
CONNECT_POLL_S = 0.1

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

logger = logging.getLogger(LOGGER_NAME)


class FeedSource(Protocol):
    """What the link needs of a feed: the `FeedAPI` of an `openalgo.api` client (the
    session's own feed client), or the fake OpenAlgo's direct feed in tests."""

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


class FeedLink:
    """The feed of one live session (see the module docstring).

    `source` is the session's feed client; `instruments` the session's instruments,
    `[{"exchange": "NSE", "symbol": "SBIN"}, ...]` (upper-cased as OpenAlgo writes them);
    `inbox` the loop's inbox and `updates` the order worker's raw updates; `now` the
    session's clock and `sleep` its wait (the session passes one that keeps its loop
    turning); `connect_timeout_s` the most seconds `connect` waits.

    Raises ConfigError for a source without the feed calls a link makes, unusable
    instruments, or a timeout that is not a positive number of seconds.
    """

    def __init__(
        self,
        source: FeedSource,
        *,
        instruments: Sequence[Mapping[str, str]],
        inbox: Inbox,
        updates: RawUpdates,
        now: Now,
        sleep: Sleep,
        connect_timeout_s: float = FEED_CONNECT_TIMEOUT_S,
    ) -> None:
        missing = [name for name in _FEED_CALLS if not callable(getattr(source, name, None))]
        if missing:
            raise ConfigError(
                f"the feed has no {', '.join(missing)}: give the session's openalgo.api feed client"
            )
        if positive_amount(connect_timeout_s) is None:
            raise ConfigError(
                f"connect_timeout_s is a positive number of seconds, got {connect_timeout_s!r}"
            )
        self._source = source
        self._instruments = instruments_of(instruments)
        self._inbox, self._updates = inbox, updates
        self._now, self._sleep = now, sleep
        self._timeout_s = float(connect_timeout_s)
        self._hooks = (
            (source.on_connect, self._on_connect),
            (source.on_disconnect, self._on_disconnect),
            (source.on_reconnect, self._on_reconnect),
            (source.on_error, self._on_error),
        )
        self._connected = False
        self._closed = False  # read by the callbacks
        self._last_error: str | None = None  # written by the error hook
        self._down_since_ns: int | None = None
        self._outages: list[Outage] = []
        self._errors = 0
        self._resubscriptions = 0

    # ----- what the loop reads --------------------------------------------------------

    @property
    def instruments(self) -> tuple[tuple[str, str], ...]:
        """The (exchange, symbol) pairs subscribed, in upper case, in order."""
        return self._instruments

    @property
    def down_since_ns(self) -> int | None:
        """When the outage under way began by the loop's clock, or None while the feed is
        up (or was never connected)."""
        return self._down_since_ns

    @property
    def outages(self) -> tuple[Outage, ...]:
        """The outages that ended, in order."""
        return tuple(self._outages)

    @property
    def errors(self) -> int:
        """The feed's errors `note` has taken in."""
        return self._errors

    @property
    def resubscriptions(self) -> int:
        """Subscriptions sent again after reconnects: one per instrument, and one for the
        order stream."""
        return self._resubscriptions

    # ----- the loop's thread -----------------------------------------------------------

    def connect(self) -> None:
        """Register the hooks, connect the feed, and subscribe the instruments in quote
        mode and the order stream.

        Waits at most `connect_timeout_s` by the session's clock. When the feed's own
        `connect()` (which waits up to its own connect and authentication times) does not
        succeed, the SDK goes on trying by itself, and the link looks every
        `CONNECT_POLL_S` seconds, through `sleep`, whether it has. A subscription that
        could not be sent now is logged: the feed sends it when it connects again.

        Raises LiveError when the feed is not connected in time, naming the feed's last
        error, and when the link was connected or closed before. Then, and on anything
        `sleep` or the feed raises (the session's stop), the feed is disconnected (so it
        stops trying) and the hooks are removed before the error goes on.
        """
        if self._connected or self._closed:
            raise LiveError("the feed link connects once: a new session makes a new link")
        for register, hook in self._hooks:
            register(hook)
        try:
            self._wait_connected()
            self._subscribe()
        except BaseException:
            self._stop_feed()
            raise
        self._connected = True

    def note(self, event: FeedEvent) -> Outage | None:
        """Take in a feed event the loop took off its inbox, in the order they came.

        Returns the outage a reconnect closed, else None. A disconnect the session did
        not ask for opens an outage now, by the session's clock; a reconnect closes it
        and subscribes again what the server did not take back; an error is counted.
        Each is logged. Raises TypeError for anything but a `FeedEvent`.
        """
        if not isinstance(event, FeedEvent):
            raise TypeError(f"note takes a FeedEvent, not {type(event).__name__}")
        if event.kind is FeedEventKind.DISCONNECT:
            self._went_down(event)
        elif event.kind is FeedEventKind.RECONNECT:
            return self._came_back(event)
        elif event.kind is FeedEventKind.ERROR:
            self._errors += 1
            logger.warning(
                "feed error (%s, connection %s): %s",
                event.payload.get("kind"),
                event.seq,
                event.error,
            )
        else:
            logger.info("the feed connected (connection %s)", event.seq)
        return None

    def close(self) -> None:
        """Stop the feed: from now no callback queues anything, an outage under way ends
        now, the feed is disconnected (it stops reconnecting and forgets its
        subscriptions) and the hooks are removed. Safe to call twice. An error of the
        feed's disconnect is raised once the hooks are removed."""
        if self._closed:
            return
        if self._down_since_ns is not None:
            self._end_outage()
        self._stop_feed()

    # ----- inside, on the loop's thread -------------------------------------------------

    def _wait_connected(self) -> None:
        deadline_ns = self._now() + round(self._timeout_s * NANOS_PER_SECOND)
        if self._source.connect():
            return
        while self._now() < deadline_ns:
            self._sleep(CONNECT_POLL_S)
            if self._source.authenticated:
                return
        last = f" (its last error: {self._last_error})" if self._last_error else ""
        raise LiveError(
            f"the feed did not connect to {self._source.ws_url} within {self._timeout_s:g} s"
            f"{last}: check that OpenAlgo and its WebSocket proxy are running and that the "
            "API key is valid, then start the session again"
        )

    def _subscribe(self) -> None:
        quotes_sent = self._source.subscribe_quote(_requests(self._instruments), self._on_quote)
        orders_sent = self._source.subscribe_orders(self._on_order)
        if not (quotes_sent and orders_sent):
            logger.warning(
                "the feed's subscriptions were not all sent now (the connection dropped?): "
                "the feed sends them when it connects again"
            )

    def _stop_feed(self) -> None:
        self._closed = True
        try:
            self._source.disconnect()
        finally:
            for _, hook in self._hooks:
                self._source.remove_hook(hook)

    def _went_down(self, event: FeedEvent) -> None:
        if event.requested or self._down_since_ns is not None:
            return
        self._down_since_ns = self._now()
        will = (
            "it reconnects by itself"
            if event.payload.get("will_reconnect")
            else "nothing reconnects it"
        )
        logger.warning(
            "the feed is down (connection %s: %s); %s",
            event.seq,
            event.payload.get("reason"),
            will,
        )

    def _came_back(self, event: FeedEvent) -> Outage | None:
        outage = None if self._down_since_ns is None else self._end_outage()
        if outage is not None:
            seconds = (outage.end_ns - outage.start_ns) / NANOS_PER_SECOND
            logger.warning("the feed is back (connection %s) after %.1f s down", event.seq, seconds)
        self._resubscribe(event.payload.get("subscriptions"))
        return outage

    def _end_outage(self) -> Outage:
        start_ns = self._down_since_ns
        # The wall clock may step back between two reads: an outage never ends before it
        # starts, or the bar builders would refuse it.
        outage = Outage(start_ns, max(self._now(), start_ns))
        self._outages.append(outage)
        self._down_since_ns = None
        return outage

    def _resubscribe(self, accepted: Any) -> None:
        """Subscribe again what the reconnect's accepted subscriptions (the SDK's
        `get_subscriptions()` shape) lack."""
        accepted = accepted if isinstance(accepted, Mapping) else {}
        taken = {
            (str(entry.get("exchange")).upper(), str(entry.get("symbol")).upper())
            for entry in accepted.get("quote") or ()
            if isinstance(entry, Mapping)
        }
        missing = [pair for pair in self._instruments if pair not in taken]
        if missing:
            names = ", ".join(f"{exchange}:{symbol}" for exchange, symbol in missing)
            logger.warning(
                "the feed did not take %s back after reconnecting: subscribing again", names
            )
            self._resubscriptions += len(missing)
            self._source.subscribe_quote(_requests(missing), self._on_quote)
        if accepted.get("orders") is not True:
            logger.warning("the feed did not take the order stream back: subscribing again")
            self._resubscriptions += 1
            self._source.subscribe_orders(self._on_order)

    # ----- the feed's threads: queue and stamp only -------------------------------------

    def _on_quote(self, message: dict[str, Any]) -> None:
        if not self._closed:
            self._inbox.put(FeedTick(message, self._now()))

    def _on_order(self, message: dict[str, Any]) -> None:
        if not self._closed:
            self._updates.put(RawOrderUpdate(message, self._now()))

    def _on_connect(self, payload: dict[str, Any]) -> None:
        self._event(FeedEventKind.CONNECT, payload)

    def _on_disconnect(self, payload: dict[str, Any]) -> None:
        self._event(FeedEventKind.DISCONNECT, payload)

    def _on_reconnect(self, payload: dict[str, Any]) -> None:
        self._event(FeedEventKind.RECONNECT, payload)

    def _on_error(self, error: BaseException, context: dict[str, Any]) -> None:
        # Kept for the message of a connect that fails: the loop may not run meanwhile.
        self._last_error = f"{type(error).__name__}: {error}"
        self._event(FeedEventKind.ERROR, context, error)

    def _event(
        self, kind: FeedEventKind, payload: dict[str, Any], error: BaseException | None = None
    ) -> None:
        if not self._closed:
            self._inbox.put(FeedEvent(kind, payload, payload.get("seq"), self._now(), error))


def _requests(pairs: Sequence[tuple[str, str]]) -> list[dict[str, str]]:
    """The SDK's subscription entries for (exchange, symbol) pairs."""
    return [{"exchange": exchange, "symbol": symbol} for exchange, symbol in pairs]


__all__ = ["CONNECT_POLL_S", "FEED_CONNECT_TIMEOUT_S", "FeedLink", "FeedSource"]
