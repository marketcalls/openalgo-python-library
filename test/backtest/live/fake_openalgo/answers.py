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

import json
from typing import Any

import httpx

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
#: A gateway's JSON body for a request it gave up on.
GATEWAY_TIMEOUT_BODY = {"message": "Endpoint request timed out"}
#: The HTML page a gateway answers a 502 with.
BAD_GATEWAY_PAGE = (
    "<html>\r\n<head><title>502 Bad Gateway</title></head>\r\n<body>\r\n"
    "<center><h1>502 Bad Gateway</h1></center>\r\n</body>\r\n</html>\r\n"
)
#: The rate limits of the REST routes, as the limiter describes them: the order and data
#: routes ("10 per second", restx_api/place_order.py and the others) and the Telegram
#: routes ("30 per minute", restx_api/telegram_bot.py).
DEFAULT_LIMIT = "10 per 1 second"
ROUTE_LIMITS = {"telegram/notify": "30 per 1 minute"}
#: The client's words when a request times out or the connection drops (httpx).
TIMEOUT_WORDS = "timed out"
DROPPED_WORDS = "Server disconnected without sending a response."


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
    body = {"message": ROUTE_LIMITS.get(endpoint, DEFAULT_LIMIT)}
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
    "CANCEL_FAILED",
    "DEFAULT_LIMIT",
    "DROPPED_WORDS",
    "EXPIRED_TOKEN",
    "GATEWAY_TIMEOUT_BODY",
    "INVALID_KEY",
    "MODIFY_FAILED",
    "QUEUED_FOR_APPROVAL",
    "ROUTE_LIMITS",
    "SANDBOX_INVALID_KEY",
    "SEMI_AUTO_MODE",
    "SEMI_AUTO_NO_CANCEL",
    "SEMI_AUTO_NO_MODIFY",
    "TIMEOUT_WORDS",
    "UNANSWERED_ORDER_MESSAGE",
    "Unmodelled",
    "bad_gateway_page",
    "gateway_json",
    "rate_limited",
    "response",
    "service",
]
