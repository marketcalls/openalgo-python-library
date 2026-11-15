"""The server's own answer inside the SDK's error, for every module of the live layer that
reads OpenAlgo's answers (the order path, the alerts).

The SDK's REST mixins answer a non-200 reply alike: `{"status": "error", "message":
"HTTP <code>: <the reply's text>", "code": <code>, "error_type": "http_error"}`
(`_handle_response` in openalgo/orders.py and openalgo/telegram.py).
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any

#: The statuses of an answer OpenAlgo wrote itself: every answer of its REST API has its
#: own `status` (restx_api/place_order.py, restx_api/telegram_bot.py).
_SERVER_STATUSES = frozenset({"success", "error"})
#: The SDK's message for a non-200 reply: "HTTP <code>: <the reply's text>".
_SDK_HTTP_MESSAGE = re.compile(r"HTTP (\d{3}): (.*)", re.DOTALL)


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


def answered_by_server(answer: Mapping[str, Any]) -> bool:
    """Whether OpenAlgo itself wrote the answer, whatever its status code: it carries its
    own `status` and no SDK `error_type` (`server_answer` recovers a non-200 one)."""
    return "error_type" not in answer and answer.get("status") in _SERVER_STATUSES


__all__ = ["answered_by_server", "server_answer"]
