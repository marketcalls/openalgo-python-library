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

from ._guard_answers import PENDING_ORDER_ID_FIELD, SEMI_AUTO_MODE_FIELD

#: The SDK's `error_type` when its own timeout ended the wait (openalgo/orders.py).
_TIMEOUT_ERROR_TYPE = "timeout_error"
#: The SDK's `error_type` when no connection could be made: the request never left.
_CONNECTION_ERROR_TYPE = "connection_error"
#: The SDK's `error_type` for an answer that never came from OpenAlgo's order service: a
#: timeout, a refused connection, an unreadable body or an unexpected client failure.
_TRANSPORT_ERROR_TYPES = frozenset(
    {_TIMEOUT_ERROR_TYPE, _CONNECTION_ERROR_TYPE, "json_error", "unknown_error"}
)
#: The SDK's `error_type` for an HTTP failure: with a status code the server answered,
#: without one the exchange broke off (openalgo/orders.py `_make_request`).
_HTTP_ERROR_TYPE = "http_error"
#: From HTTP 500 on, the server failed part way or the broker did not answer (a 502 is the
#: server's "may have been placed", services/place_order_service.py).
_FIRST_SERVER_ERROR_CODE = 500
#: The statuses of an answer OpenAlgo wrote itself: every answer on the order path has its
#: own `status` (restx_api/place_order.py, services/place_order_service.py).
_SERVER_STATUSES = frozenset({"success", "error"})
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
    own message. That body is returned when the text is a JSON object with OpenAlgo's
    own `status`, and `answer` unchanged otherwise: a success, a transport failure, a
    body that is not JSON, or a JSON body without a status, which a gateway in front of
    OpenAlgo writes (`{"message": "Endpoint request timed out"}` on a 504).
    """
    found = _SDK_HTTP_MESSAGE.fullmatch(str(answer.get("message") or ""))
    if answer.get("status") != "error" or found is None:
        return answer
    try:
        body = json.loads(found.group(2))
    except ValueError:
        return answer
    if not isinstance(body, dict) or body.get("status") not in _SERVER_STATUSES:
        return answer
    return {**body, "code": int(found.group(1))}


def outcome_of(answer: Mapping[str, Any], order_id: str = "") -> SendOutcome:
    """The outcome an answer gives; `order_id` is the order a modify or cancel was for.

    A success without an order ID, and an answer with neither "success" nor "error" as
    its status, are UNANSWERED: they say nothing of the order, so it is looked for in the
    order book like a lost answer.
    """
    status = answer.get("status")
    if PENDING_ORDER_ID_FIELD in answer or answer.get("mode") == SEMI_AUTO_MODE_FIELD:
        pending = str(answer.get(PENDING_ORDER_ID_FIELD, ""))
        reason = f"queued for approval in OpenAlgo's Action Center (semi-auto), as {pending}"
        return SendOutcome(Outcome.QUEUED, reason=reason, pending_order_id=pending)
    if status == "success":
        accepted = str(answer.get("orderid") or order_id)
        if accepted:
            return SendOutcome(Outcome.ACCEPTED, order_id=accepted)
        return SendOutcome(Outcome.UNANSWERED, reason="OpenAlgo answered success, no order ID")
    if status != "error":
        return SendOutcome(
            Outcome.UNANSWERED, reason=f"OpenAlgo's answer has no status: {dict(answer)!r}"
        )
    reason = str(answer.get("message") or "OpenAlgo answered an error with no message")
    kind = Outcome.UNANSWERED if _unanswered(answer) else Outcome.REJECTED
    return SendOutcome(kind, reason=reason)


def server_has_finished(answer: Mapping[str, Any]) -> bool:
    """Whether an answer that leaves a placement unknown shows the server is no longer
    working on it: the connection was refused (the request never reached the server), or
    OpenAlgo answered in its own JSON (`answered_by_server`). The SDK's own words show
    nothing of the kind: a client timeout, a dropped connection, a 5xx page or a
    gateway's JSON without a status. The server may still be placing it."""
    return answer.get("error_type") == _CONNECTION_ERROR_TYPE or answered_by_server(answer)


def answered_by_server(answer: Mapping[str, Any]) -> bool:
    """Whether OpenAlgo itself wrote the answer, whatever its status code: it carries its
    own `status` and no SDK `error_type` (`server_answer` recovers a non-200 one)."""
    return "error_type" not in answer and answer.get("status") in _SERVER_STATUSES


def went_silent(answer: Mapping[str, Any]) -> bool:
    """Whether the answer shows the server may be hung: the client's own timeout, or a
    connection that broke off with no status code."""
    error_type = answer.get("error_type")
    if error_type == _TIMEOUT_ERROR_TYPE:
        return True
    return error_type == _HTTP_ERROR_TYPE and answer.get("code") is None


class ServerWatch:
    """Whether OpenAlgo has fallen silent: a call of the order path ended in a client
    timeout or a dropped connection, and no call has been answered in OpenAlgo's own JSON
    since (`silent_since_ns`, UTC nanoseconds, else None). A hung server may still be
    placing what it was sent, so nothing is sent for any instrument meanwhile."""

    def __init__(self) -> None:
        self.silent_since_ns: int | None = None

    @property
    def silent(self) -> bool:
        """True while OpenAlgo has not answered since a timeout or a dropped connection."""
        return self.silent_since_ns is not None

    def note(self, answer: Mapping[str, Any], now_ns: int) -> None:
        """Take in the answer to a call made at `now_ns`."""
        if went_silent(answer):
            if self.silent_since_ns is None:
                self.silent_since_ns = now_ns
        elif answered_by_server(answer):
            self.silent_since_ns = None


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


__all__ = [
    "Outcome",
    "SendOutcome",
    "ServerWatch",
    "answered_by_server",
    "outcome_of",
    "server_answer",
    "server_has_finished",
    "went_silent",
]
