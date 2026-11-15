"""What an answer to `placeorder`, `modifyorder` or `cancelorder` says became of the
request, and the server's own answer inside the SDK's error, for the order path
(`orders`).

Read from the SDK's answers (openalgo/orders.py `_make_request`) and the server's order
services: a refusal went no further, while a transport failure or a server error leaves
it unknown whether the request reached the broker.
"""

from __future__ import annotations

import enum
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

#: The SDK's `error_type` for an answer that never came from OpenAlgo's order service: a
#: timeout, a lost connection, an unreadable body or an unexpected failure in the client.
_TRANSPORT_ERROR_TYPES = frozenset(
    {"timeout_error", "connection_error", "json_error", "unknown_error"}
)
#: The SDK's `error_type` for an HTTP failure: with a status code the server answered,
#: without one the exchange broke off (openalgo/orders.py `_make_request`).
_HTTP_ERROR_TYPE = "http_error"
#: From HTTP 500 on, the server failed part way or the broker did not answer (a 502 is the
#: server's "may have been placed", services/place_order_service.py).
_FIRST_SERVER_ERROR_CODE = 500
#: Semi-auto mode queues an order in the Action Center instead of placing it
#: (services/order_router_service.py `queue_order`).
_SEMI_AUTO_MODE = "semi_auto"
_PENDING_ORDER_ID = "pending_order_id"
#: The SDK's message for a non-200 reply: "HTTP <code>: <the reply's text>"
#: (openalgo/orders.py `_handle_response`).
_SDK_HTTP_MESSAGE = re.compile(r"HTTP (\d{3}): (.*)", re.DOTALL)


class Outcome(enum.Enum):
    """What became of a request sent once."""

    #: OpenAlgo took it: an order placed, modified or cancelled.
    ACCEPTED = "accepted"
    #: OpenAlgo refused it; nothing reached the broker.
    REJECTED = "rejected"
    #: No answer to act on (a 502, a timeout, a lost connection): it may have reached the
    #: broker. A placement stays unbound in the journal until it is resolved.
    UNANSWERED = "unanswered"
    #: Semi-auto mode queued it in OpenAlgo's Action Center: not a broker order.
    QUEUED = "queued"


@dataclass(frozen=True)
class SendOutcome:
    """The outcome of one place, modify or cancel. `order_id` is set when ACCEPTED,
    `reason` (OpenAlgo's message) otherwise, and `pending_order_id` when QUEUED."""

    kind: Outcome
    order_id: str = ""
    reason: str = ""
    pending_order_id: str = ""


def server_answer(answer: Mapping[str, Any]) -> Mapping[str, Any]:
    """The server's own answer inside an SDK error, with its HTTP status as `code`.

    For a non-200 reply the SDK keeps only "HTTP <code>: <text>" as the message, which
    hides the server's JSON body: its `mode` ("analyze" on a sandbox rejection) and its
    own message. That body is returned when the text is a JSON object, and `answer`
    unchanged otherwise (a success, a transport failure, a body that is not JSON).
    """
    found = _SDK_HTTP_MESSAGE.fullmatch(str(answer.get("message") or ""))
    if answer.get("status") != "error" or found is None:
        return answer
    try:
        body = json.loads(found.group(2))
    except ValueError:
        return answer
    if not isinstance(body, dict):
        return answer
    return {"status": "error", **body, "code": int(found.group(1))}


def outcome_of(answer: Mapping[str, Any], order_id: str = "") -> SendOutcome:
    """The outcome an answer gives; `order_id` is the order a modify or cancel was for.

    A success without an order ID is UNANSWERED: the order it reports cannot be found
    by its ID, so it is looked for in the order book like a lost answer.
    """
    if _PENDING_ORDER_ID in answer or answer.get("mode") == _SEMI_AUTO_MODE:
        pending = str(answer.get(_PENDING_ORDER_ID, ""))
        reason = f"queued for approval in OpenAlgo's Action Center (semi-auto), as {pending}"
        return SendOutcome(Outcome.QUEUED, reason=reason, pending_order_id=pending)
    if answer.get("status") == "success":
        accepted = str(answer.get("orderid") or order_id)
        if accepted:
            return SendOutcome(Outcome.ACCEPTED, order_id=accepted)
        return SendOutcome(Outcome.UNANSWERED, reason="OpenAlgo answered success, no order ID")
    reason = str(answer.get("message") or "OpenAlgo answered an error with no message")
    kind = Outcome.UNANSWERED if _unanswered(answer) else Outcome.REJECTED
    return SendOutcome(kind, reason=reason)


def _unanswered(answer: Mapping[str, Any]) -> bool:
    """Whether an error answer leaves it unknown if the request reached the broker. A 4xx,
    or an error OpenAlgo answered with 200, is a refusal of a request that went no further
    (a bad field, the key, the rate limiter, a broker refusal)."""
    if answer.get("error_type") in _TRANSPORT_ERROR_TYPES:
        return True
    code = answer.get("code")
    if isinstance(code, int):
        return code >= _FIRST_SERVER_ERROR_CODE
    return answer.get("error_type") == _HTTP_ERROR_TYPE


__all__ = ["Outcome", "SendOutcome", "outcome_of", "server_answer"]
