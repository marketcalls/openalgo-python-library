"""The faults a test can schedule on the fake OpenAlgo, one call each.

REST faults act on the next call to one endpoint (the path after `/api/v1/`, as the SDK
posts it: "placeorder", "analyzer", "telegram/notify"), each once, in the order they were
scheduled. A transport fault can be told the call reached the broker first
(`reached_broker`): the order is then placed, modified or cancelled although its answer
never comes back, which is what makes an answer ambiguous. The stream's and the feed's
faults act on the WebSocket (`feed.Feed`). Every fault is logged when it acts or is set
(`FakeLog.faults`).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from .feed import Feed
from .log import FAULT
from .venue import Venue

#: The REST faults.
TIMEOUT = "timeout"
BAD_GATEWAY_PAGE = "bad_gateway_page"
GATEWAY_ERROR = "gateway_error"
DROPPED_CONNECTION = "dropped_connection"
REFUSED_CONNECTION = "refused_connection"
LOST_ANSWER = "lost_answer"
SLOW_ANSWER = "slow_answer"
RATE_LIMITED = "rate_limited"
#: The endpoints whose broker answer the server can lose after the broker acted.
ORDER_WRITES = frozenset(
    {
        "placeorder",
        "modifyorder",
        "cancelorder",
        "placegttorder",
        "modifygttorder",
        "cancelgttorder",
    }
)
#: A gateway's status when it gives up waiting for the server.
GATEWAY_TIMEOUT_STATUS = 504


@dataclass(frozen=True)
class RestFault:
    """A fault on one REST call: `seconds` the answer takes (a slow answer), `status`
    the gateway answers with (a gateway error)."""

    name: str
    endpoint: str
    reached_broker: bool = False
    seconds: float = 0.0
    status: int = 0


class Faults:
    """Every fault a test can schedule. `endpoints` are the REST endpoints the fake
    answers; a fault on another raises ValueError."""

    def __init__(self, venue: Venue, feed: Feed, endpoints: frozenset[str]) -> None:
        self._venue = venue
        self._feed = feed
        self._endpoints = endpoints
        self._rest: dict[str, deque[RestFault]] = {}

    # -- REST -------------------------------------------------------------

    def timeout(self, endpoint: str, *, reached_broker: bool = False) -> None:
        """The next call to `endpoint` times out in the client: the fake's clock moves on
        by the client's read timeout (the SDK's 120 seconds by default), then the SDK
        answers its "timeout_error"."""
        self._schedule(RestFault(TIMEOUT, endpoint, reached_broker))

    def refuse_connection(self, endpoint: str) -> None:
        """The next call to `endpoint` finds no server to connect to: the SDK answers its
        "connection_error", and nothing reached the server."""
        self._schedule(RestFault(REFUSED_CONNECTION, endpoint))

    def bad_gateway_page(self, endpoint: str, *, reached_broker: bool = False) -> None:
        """The next call to `endpoint` is answered HTTP 502 with a gateway's HTML page."""
        self._schedule(RestFault(BAD_GATEWAY_PAGE, endpoint, reached_broker))

    def gateway_error(
        self,
        endpoint: str,
        *,
        status: int = GATEWAY_TIMEOUT_STATUS,
        reached_broker: bool = False,
    ) -> None:
        """The next call to `endpoint` is answered `status` (500 or more) with a gateway's
        JSON body that has no `status`. Raises ValueError for a status below 500."""
        if status < 500:
            raise ValueError(f"a gateway error is HTTP 500 or more, not {status}")
        self._schedule(RestFault(GATEWAY_ERROR, endpoint, reached_broker, status=status))

    def drop_connection(self, endpoint: str, *, reached_broker: bool = False) -> None:
        """The next call to `endpoint` loses its connection before an answer: the SDK
        answers an "http_error" with no status code."""
        self._schedule(RestFault(DROPPED_CONNECTION, endpoint, reached_broker))

    def lose_answer(self, endpoint: str) -> None:
        """The next order write succeeds at the broker but the server loses the broker's
        answer: HTTP 502 with the server's "may have been placed" words for a placement
        (services/place_order_service.py), HTTP 500 "due to internal error" for a modify
        or cancel and for each GTT call (the GTT is placed, changed or cancelled all the
        same). Raises ValueError for an endpoint that writes no order, and in sandbox
        mode, which sends nothing to a broker."""
        if endpoint not in ORDER_WRITES:
            raise ValueError(f"only {sorted(ORDER_WRITES)} can lose a broker's answer")
        if self._venue.analyzer:
            raise ValueError("the OpenAlgo sandbox sends nothing to a broker: no answer to lose")
        self._schedule(RestFault(LOST_ANSWER, endpoint, reached_broker=True))

    def slow(self, endpoint: str, seconds: float) -> None:
        """The next call to `endpoint` takes `seconds`: the fake's clock moves on by that
        much before it answers as usual. An answer slower than the client's read timeout
        is a timeout: the server still does the work, the clock moves on by the timeout,
        and the SDK answers its "timeout_error"."""
        if seconds <= 0:
            raise ValueError(f"a slow answer takes more than 0 seconds, not {seconds}")
        self._schedule(RestFault(SLOW_ANSWER, endpoint, seconds=seconds))

    def rate_limit(self, endpoint: str) -> None:
        """The next call to `endpoint` is refused by the server's rate limiter: HTTP 429
        with the REST layer's own body (`answers.rate_limited`)."""
        self._schedule(RestFault(RATE_LIMITED, endpoint))

    def expire_session(self) -> None:
        """The broker's login expires while OpenAlgo still holds it: `ping` still
        answers, while the broker's calls fail as the server reports them (funds empty,
        quotes, history, orders and books refused), the sandbox, which prices from the
        broker's quotes, fills nothing, and the feed stops ticking and sends no live
        order update (the broker's order stream is down too), until `restore_session`."""
        self._venue.set_switch("expired", True)
        self._venue.log.record(self._venue.clock.now(), FAULT, "session_expired")

    def restore_session(self) -> None:
        """The broker's login is valid again (the user logged in)."""
        self._venue.set_switch("expired", False)
        self._venue.log.record(self._venue.clock.now(), FAULT, "session_restored")

    def revoke_login(self) -> None:
        """OpenAlgo revokes the broker's login, as at the daily session rollover
        (app.py:572-583): every call that needs it, `ping` and `analyzerstatus` included,
        answers HTTP 403 "Invalid openalgo apikey", and the feed stops ticking and sends no
        live order update, until `restore_login`. `telegram/notify` checks the API key
        alone and still answers (restx_api/telegram_bot.py, SendNotification)."""
        self._venue.set_switch("revoked", True)
        self._venue.log.record(self._venue.clock.now(), FAULT, "login_revoked")

    def restore_login(self) -> None:
        """The user logged in to the broker again through OpenAlgo."""
        self._venue.set_switch("revoked", False)
        self._venue.log.record(self._venue.clock.now(), FAULT, "login_restored")

    def take(self, endpoint: str) -> RestFault | None:
        """The next fault scheduled on `endpoint`, taken off the schedule."""
        queue = self._rest.get(endpoint)
        return queue.popleft() if queue else None

    # -- the order-updates stream -------------------------------------------

    def silence_order_updates(self) -> None:
        """The order-updates stream goes silent: every update from now is lost."""
        self._feed.set_silent(True)

    def resume_order_updates(self) -> None:
        """The order-updates stream sends again (what it lost stays lost)."""
        self._feed.set_silent(False)

    def drop_next_order_update(self) -> None:
        """The next order update is lost."""
        self._feed.drop_next_update()

    def duplicate_next_order_update(self) -> None:
        """The next order update arrives twice."""
        self._feed.duplicate_next_update()

    def hold_order_updates(self) -> None:
        """Order updates wait until `release_order_updates` (a lagging stream)."""
        self._feed.hold_updates()

    def release_order_updates(self, *, newest_first: bool = False) -> int:
        """Send the held updates, oldest first, or newest first (out of order). Returns
        how many there were."""
        return self._feed.release_updates(newest_first=newest_first)

    def zero_price_next_fill(self) -> None:
        """The next live fill is reported on the stream at an average price of 0 (the
        books keep its price). Raises ValueError in sandbox mode."""
        self._venue.zero_price_next_fill()

    # -- the market-data feed -------------------------------------------------

    def feed_dead(self) -> None:
        """Ticks stop on every instrument while the connection stays up."""
        self._feed.set_dead(True)

    def feed_alive(self) -> None:
        """Ticks flow again."""
        self._feed.set_dead(False)

    def drop_feed(self, *, hold_reconnect: bool = False) -> None:
        """The WebSocket connection drops. The SDK reconnects by itself at once, or only
        after `let_feed_reconnect` when `hold_reconnect`."""
        self._feed.drop_connection(hold_reconnect=hold_reconnect)

    def let_feed_reconnect(self) -> None:
        """Let a reconnect held by `drop_feed(hold_reconnect=True)` through."""
        self._feed.let_reconnect()

    def tick_gap(self, symbol: str, exchange: str, trades: int = 1) -> None:
        """The next `trades` trades of one instrument send no tick (the broker's candles
        still hold them)."""
        self._feed.gap((symbol, exchange), trades)

    def _schedule(self, fault: RestFault) -> None:
        if fault.endpoint not in self._endpoints:
            raise ValueError(
                f"the fake answers no endpoint {fault.endpoint!r}: use one of "
                f"{sorted(self._endpoints)}"
            )
        self._rest.setdefault(fault.endpoint, deque()).append(fault)


__all__ = [
    "BAD_GATEWAY_PAGE",
    "DROPPED_CONNECTION",
    "GATEWAY_ERROR",
    "GATEWAY_TIMEOUT_STATUS",
    "LOST_ANSWER",
    "ORDER_WRITES",
    "RATE_LIMITED",
    "REFUSED_CONNECTION",
    "SLOW_ANSWER",
    "TIMEOUT",
    "Faults",
    "RestFault",
]
