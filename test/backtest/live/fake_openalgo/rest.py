"""The fake OpenAlgo's REST API: the handler of an `httpx.MockTransport` that a real
`openalgo.api` sends its calls to, so the SDK's own request and error handling runs
unchanged and nothing leaves the process.

Each call: the scheduled fault of its endpoint (`faults`), the route's rate limit on the
fake's clock (`limiter`), the API key and OpenAlgo's hold on the broker's login, then the
route, under the venue's lock; the order updates the route caused are sent once the lock
is released. A timeout costs the client's read timeout on the fake's clock. Every answer
carries an HTTP `Date` from the fake's clock, which the
preflight's clock check reads, and so does a HEAD of the server's root (the web app's
page, blueprints/react_app.py). A call the fake does not model, and any failure inside
the fake, is logged as a problem and raised: the SDK turns a raised error into an
"unknown_error" answer, so the fixture fails the test on any problem.
"""

from __future__ import annotations

import email.utils
import json
import math
import threading
from collections.abc import Callable
from typing import Any

import httpx

from openalgo.backtest._layout import NANOS_PER_SECOND

from . import routes_account, routes_data, routes_orders
from .answers import (
    CANCEL_FAILED,
    DROPPED_WORDS,
    INVALID_KEY,
    MODIFY_FAILED,
    REFUSED_WORDS,
    TIMEOUT_WORDS,
    UNANSWERED_ORDER_MESSAGE,
    Unmodelled,
    bad_gateway_page,
    gateway_json,
    rate_limited,
    response,
    service,
)
from .faults import (
    BAD_GATEWAY_PAGE,
    DROPPED_CONNECTION,
    GATEWAY_ERROR,
    LOST_ANSWER,
    RATE_LIMITED,
    REFUSED_CONNECTION,
    SLOW_ANSWER,
    TIMEOUT,
    Faults,
    RestFault,
)
from .limiter import RateLimiter
from .log import CALL, FAULT, PROBLEM
from .venue import Venue

Route = Callable[[Venue, dict[str, Any]], tuple[dict[str, Any], int]]

#: The HTTP header that names the client of the fake a request came from
#: (`FakeOpenAlgo.client`): the fake's own test seam, no server reads it.
CLIENT_HEADER = "x-fake-openalgo-client"
#: The endpoints the fake answers, by the path after `/api/v1/` the SDK posts to.
ROUTES: dict[str, Route] = {
    "placeorder": routes_orders.place,
    "modifyorder": routes_orders.modify,
    "cancelorder": routes_orders.cancel,
    "orderstatus": routes_orders.status,
    "orderbook": routes_account.orderbook,
    "tradebook": routes_account.tradebook,
    "positionbook": routes_account.positionbook,
    "funds": routes_account.funds,
    "analyzer": routes_account.analyzer,
    "ping": routes_account.ping,
    "symbol": routes_data.symbol,
    "intervals": routes_data.intervals,
    "quotes": routes_data.quotes,
    "history": routes_data.history,
    "market/holidays": routes_data.holidays,
    "telegram/notify": routes_data.telegram_notify,
}
API_PREFIX = "/api/v1/"
#: The Telegram route checks the key itself, with its own words and status
#: (restx_api/telegram_bot.py, SendNotification).
_TELEGRAM = "telegram/notify"
_TELEGRAM_KEY_REFUSED = ({"status": "error", "message": "Invalid or missing API key"}, 401)
#: The server's answer when it lost the broker's answer to an order write after the
#: broker acted (services/place_order_service.py, modify_order_service.py,
#: cancel_order_service.py).
_LOST = {
    "placeorder": ({"status": "error", "message": UNANSWERED_ORDER_MESSAGE}, 502),
    "modifyorder": ({"status": "error", "message": MODIFY_FAILED}, 500),
    "cancelorder": ({"status": "error", "message": CANCEL_FAILED}, 500),
}


class RestHandler:
    """Answers the SDK's HTTP requests for the venue. `api_key` is the only key known."""

    def __init__(self, venue: Venue, faults: Faults, api_key: str) -> None:
        self._venue = venue
        self._faults = faults
        self._api_key = api_key
        self._limiter = RateLimiter()
        # The client of the call being answered on each thread, for the log.
        self._calling = threading.local()

    def __call__(self, request: httpx.Request) -> httpx.Response:
        """The answer to one request; raises the client's own transport errors for a
        timeout, a dropped or a refused connection (the faults), and re-raises (after
        logging a problem) anything else."""
        self._calling.client = request.headers.get(CLIENT_HEADER, "")
        try:
            return self._answer(request)
        except httpx.TransportError:
            raise
        except Exception as error:
            now = self._venue.clock.now()
            self._venue.log.record(now, PROBLEM, request.url.path, error=repr(error))
            raise

    def _answer(self, request: httpx.Request) -> httpx.Response:
        self._venue.feed.refuse_feed_thread()
        path = request.url.path
        if request.method == "HEAD" and path == "/":
            return response((200, "text/html", ""), self._date())
        endpoint = path.removeprefix(API_PREFIX)
        if request.method != "POST" or endpoint == path or endpoint not in ROUTES:
            raise Unmodelled(f"the fake OpenAlgo does not answer {request.method} {path}")
        payload = json.loads(request.content)
        fault = self._faults.take(endpoint)
        if fault is not None:
            self._venue.log.record(self._venue.clock.now(), FAULT, fault.name, endpoint=endpoint)
            if fault.name == REFUSED_CONNECTION:
                self._note(endpoint, payload, 0, None, fault=fault.name)
                raise httpx.ConnectError(REFUSED_WORDS, request=request)
            if fault.name == LOST_ANSWER and self._venue.analyzer:
                raise Unmodelled("the OpenAlgo sandbox sends nothing to a broker to lose")
        if fault is not None and fault.name == RATE_LIMITED:
            return self._sent(endpoint, payload, rate_limited(endpoint), None)
        if not self._limiter.allow(endpoint, self._venue.clock.now()):
            return self._sent(endpoint, payload, rate_limited(endpoint), None, limited=True)
        read_timeout = _read_timeout(request)
        times_out = fault is not None and (
            fault.name == TIMEOUT or (fault.name == SLOW_ANSWER and fault.seconds > read_timeout)
        )
        if fault is not None and fault.name == SLOW_ANSWER and not times_out:
            self._venue.clock.advance(fault.seconds)
        body, status = None, 0
        acts = fault is None or fault.name == SLOW_ANSWER or fault.reached_broker
        if acts:
            body, status = self._route(endpoint, payload, self._calling.client)
            self._venue.flush()
        if times_out:
            # The client gives up after its read timeout, whatever the server does.
            self._venue.clock.advance(read_timeout)
            self._note(endpoint, payload, 0, None, fault=TIMEOUT)
            raise httpx.ReadTimeout(TIMEOUT_WORDS, request=request)
        if fault is not None and fault.name != SLOW_ANSWER:
            return self._faulty(endpoint, payload, fault, request)
        return self._sent(endpoint, payload, service(body, status), body)

    def _route(
        self, endpoint: str, payload: dict[str, Any], client: str
    ) -> tuple[dict[str, Any], int]:
        if endpoint == _TELEGRAM:
            # The notify route checks the API key alone (restx_api/telegram_bot.py).
            if payload.get("apikey") != self._api_key:
                return _TELEGRAM_KEY_REFUSED
        elif payload.get("apikey") != self._api_key or self._venue.login_revoked:
            # services/*: no key, or a login OpenAlgo revoked (get_auth_token_broker).
            return {"status": "error", "message": INVALID_KEY}, 403
        with self._venue.lock:
            self._venue.calling_client = client
            try:
                return ROUTES[endpoint](self._venue, payload)
            finally:
                self._venue.calling_client = ""

    def _faulty(
        self, endpoint: str, payload: dict[str, Any], fault: RestFault, request: httpx.Request
    ) -> httpx.Response:
        if fault.name == DROPPED_CONNECTION:
            self._note(endpoint, payload, 0, None, fault=fault.name)
            raise httpx.RemoteProtocolError(DROPPED_WORDS, request=request)
        if fault.name == BAD_GATEWAY_PAGE:
            return self._sent(endpoint, payload, bad_gateway_page(), None)
        if fault.name == GATEWAY_ERROR:
            return self._sent(endpoint, payload, gateway_json(fault.status), None)
        if fault.name == LOST_ANSWER:
            body, status = _LOST[endpoint]
            return self._sent(endpoint, payload, service(body, status), body)
        raise Unmodelled(f"no answer for the fault {fault.name!r}")

    def _sent(
        self,
        endpoint: str,
        payload: dict[str, Any],
        answer: tuple[int, str, str],
        body: dict[str, Any] | None,
        **detail: Any,
    ) -> httpx.Response:
        shown = body if body is not None else answer[2]
        self._note(endpoint, payload, answer[0], shown, **detail)
        return response(answer, self._date())

    def _note(
        self, endpoint: str, payload: dict[str, Any], status: int, answer: Any, **detail: Any
    ) -> None:
        request = {key: value for key, value in payload.items() if key != "apikey"}
        self._venue.log.record(
            self._venue.clock.now(),
            CALL,
            endpoint,
            request=request,
            status=status,
            answer=answer,
            mode=self._venue.mode.value,
            client=getattr(self._calling, "client", ""),
            **detail,
        )

    def _date(self) -> str:
        return email.utils.formatdate(self._venue.clock.now() // NANOS_PER_SECOND, usegmt=True)


def _read_timeout(request: httpx.Request) -> float:
    """The seconds the client waits for an answer (the SDK passes its `timeout`, 120 by
    default, openalgo/base.py); infinite when it set none."""
    seconds = request.extensions.get("timeout", {}).get("read")
    return math.inf if seconds is None else float(seconds)


__all__ = ["API_PREFIX", "ROUTES", "RestHandler"]
