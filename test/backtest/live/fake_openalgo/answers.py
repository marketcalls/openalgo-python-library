"""The HTTP answers of the fake OpenAlgo, encoded as the server and what stands in front of
it write them, and the server's own words in them.

- A service's answer is the web framework's `jsonify` of the service's dict: keys
  sorted, no spaces, a closing newline (restx_api/*.py return
  `make_response(jsonify(response_data), status_code)`; checked against the framework
  version the server pins in requirements.txt).
- The rate limiter's 429 on a REST route is the REST layer's own error body,
  `{"message": "<the limit>"}` with no `status`, written with the standard separators and
  a closing newline; the limiter describes the limit as "10 per 1 second" (checked
  against the server's pinned libraries, as the alerts' 429 was).
- A gateway in front of the server answers a 502 with an HTML page, or a 5xx with a JSON
  body of its own that has no `status`.
"""

from __future__ import annotations

import http
import json
from typing import Any

import httpx

from .limiter import limit_of

#: services/*: an API key the server does not know, or a session it revoked.
INVALID_KEY = "Invalid openalgo apikey"
#: The sandbox's words for the same (services/sandbox_service.py).
SANDBOX_INVALID_KEY = "Invalid API key"
#: utils/httpx_client.py: the broker's answer to a placement was lost after it was sent
#: (services/place_order_service.py answers it with HTTP 502).
UNANSWERED_ORDER_MESSAGE = (
    "The connection to the broker dropped after the order was sent, so it may "
    "have been placed. Check the order book before placing it again."
)
#: services/modify_order_service.py and cancel_order_service.py: the broker module
#: failed part way (HTTP 500).
MODIFY_FAILED = "Failed to modify order due to internal error"
CANCEL_FAILED = "Failed to cancel order due to internal error"
#: services/place_gtt_order_service.py, modify_gtt_order_service.py and
#: cancel_gtt_order_service.py: the broker module failed (HTTP 500), possibly after the
#: broker acted.
PLACE_GTT_FAILED = "Failed to place GTT due to internal error"
MODIFY_GTT_FAILED = "Failed to modify GTT due to internal error"
CANCEL_GTT_FAILED = "Failed to cancel GTT due to internal error"
#: services/order_router_service.py, queue_order: semi-auto mode queued a placement.
QUEUED_FOR_APPROVAL = "Order queued for approval in Action Center"
SEMI_AUTO_MODE = "semi_auto"
#: services/modify_order_service.py and cancel_order_service.py: semi-auto in live mode.
SEMI_AUTO_NO_MODIFY = (
    "Modify order operation is not allowed in Semi-Auto mode. Please switch to Auto mode "
    "to modify orders."
)
SEMI_AUTO_NO_CANCEL = (
    "Cancel order operation is not allowed in Semi-Auto mode. Please switch to Auto mode "
    "to cancel orders."
)
#: A broker's refusal of a token it no longer honours (an expired or replaced login),
#: in the words of the reference broker (broker/zerodha/api/data.py, funds.py).
EXPIRED_TOKEN = "Incorrect `api_key` or `access_token`."
#: Where the broker's API stands in the client errors the server passes on: the real
#: text names the reference broker's own API address.
BROKER_API = "https://broker.invalid"
#: The paths of the reference broker's books (broker/zerodha/api/order_api.py:93-106).
BOOK_PATHS = {
    "orders": "/orders",
    "trades": "/trades",
    "positions": "/portfolio/positions",
    "holdings": "/portfolio/holdings",
}
#: services/gtt_orderbook_service.py (and the other GTT services): a broker that ships no
#: GTT module, answered HTTP 501.
GTT_NOT_SUPPORTED = "GTT orders are not supported for broker '{broker}' yet"
#: restx_api/gtt_orderbook.py: the schema's refusal of a `status` other than these two.
GTT_BOOK_STATUSES = ("active", "all")
#: The reference broker's cancel refuses through its HTTP client's error, wrapped
#: (broker/zerodha/api/order_api.py:520, 536); the server answers it HTTP 500.
CANCEL_REFUSED = "Failed to cancel order: "
#: The words of quotes and history for an instrument not in the master contract
#: (services/quotes_service.py and history_service.py, validate_symbol_exchange).
UNKNOWN_SYMBOL = (
    "Symbol '{symbol}' not found for exchange '{exchange}'. Please verify the symbol name "
    "and ensure master contracts are downloaded."
)
#: The client's names of HTTP error classes (its `raise_for_status`).
_ERROR_CLASSES = {4: "Client error", 5: "Server error"}


def broker_http_error(status: int, path: str) -> str:
    """The first line of the client's words for a broker's HTTP error on `path`, which
    the server passes on (broker/zerodha/api/order_api.py:76-91 re-raises it for the
    books; services/orderbook_service.py:182-184 answers `str(error)`)."""
    phrase = http.HTTPStatus(status).phrase
    return f"{_ERROR_CLASSES[status // 100]} '{status} {phrase}' for url '{BROKER_API}{path}'"


#: A gateway's JSON body for a request it gave up on.
GATEWAY_TIMEOUT_BODY = {"message": "Endpoint request timed out"}
#: The HTML page a gateway answers a 502 with.
BAD_GATEWAY_PAGE = (
    "<html>\r\n<head><title>502 Bad Gateway</title></head>\r\n<body>\r\n"
    "<center><h1>502 Bad Gateway</h1></center>\r\n</body>\r\n</html>\r\n"
)
#: The client's words when a request times out, the connection drops, or no connection
#: can be made (httpx).
TIMEOUT_WORDS = "timed out"
DROPPED_WORDS = "Server disconnected without sending a response."
REFUSED_WORDS = "[Errno 111] Connection refused"


class Unmodelled(Exception):
    """A request the fake does not model (an endpoint, an interval, a source): the test
    fails, as the fake cannot say what the server would answer."""


def service(body: dict[str, Any], status: int) -> tuple[int, str, str]:
    """A service's answer: (status, content type, text)."""
    return (
        status,
        "application/json",
        json.dumps(body, sort_keys=True, separators=(",", ":")) + "\n",
    )


def rate_limited(endpoint: str) -> tuple[int, str, str]:
    """The limiter's 429 on `endpoint`."""
    body = {"message": limit_of(endpoint).words}
    return 429, "application/json", json.dumps(body) + "\n"


def bad_gateway_page() -> tuple[int, str, str]:
    """A gateway's HTML 502 page."""
    return 502, "text/html", BAD_GATEWAY_PAGE


def gateway_json(status: int) -> tuple[int, str, str]:
    """A gateway's JSON 5xx without a `status`."""
    return status, "application/json", json.dumps(GATEWAY_TIMEOUT_BODY)


def response(answer: tuple[int, str, str], date: str) -> httpx.Response:
    """The HTTP response of an answer, dated `date` (an HTTP date)."""
    status, content_type, text = answer
    headers = {"content-type": content_type, "date": date}
    return httpx.Response(status, headers=headers, text=text)


__all__ = [
    "BAD_GATEWAY_PAGE",
    "BOOK_PATHS",
    "BROKER_API",
    "CANCEL_FAILED",
    "CANCEL_GTT_FAILED",
    "CANCEL_REFUSED",
    "DROPPED_WORDS",
    "EXPIRED_TOKEN",
    "GATEWAY_TIMEOUT_BODY",
    "GTT_BOOK_STATUSES",
    "GTT_NOT_SUPPORTED",
    "INVALID_KEY",
    "MODIFY_FAILED",
    "MODIFY_GTT_FAILED",
    "PLACE_GTT_FAILED",
    "QUEUED_FOR_APPROVAL",
    "REFUSED_WORDS",
    "SANDBOX_INVALID_KEY",
    "SEMI_AUTO_MODE",
    "SEMI_AUTO_NO_CANCEL",
    "SEMI_AUTO_NO_MODIFY",
    "TIMEOUT_WORDS",
    "UNANSWERED_ORDER_MESSAGE",
    "UNKNOWN_SYMBOL",
    "Unmodelled",
    "bad_gateway_page",
    "broker_http_error",
    "gateway_json",
    "rate_limited",
    "response",
    "service",
]
