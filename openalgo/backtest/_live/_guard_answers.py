"""How the mode guard (`guard`) reads OpenAlgo's answers: the analyzer switch of an
`analyzerstatus` answer, and the marks of semi-auto mode on an order answer.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..errors import LiveError
from ._types import Mode

#: The field and the mode OpenAlgo puts on an order it queued in the Action Center
#: instead of sending it (services/order_router_service.py, queue_order).
PENDING_ORDER_ID_FIELD = "pending_order_id"
_SEMI_AUTO_MODE_FIELD = "semi_auto"
#: The status of an answer that succeeded.
SUCCESS = "success"
# Lower case, part of the message of the 403 that refuses a modify or cancel while
# semi-auto is on in live mode ("Modify order operation is not allowed in Semi-Auto
# mode. ...": services/modify_order_service.py and services/cancel_order_service.py,
# the semi-auto check of their API-key path). Matched in the server's body and in the
# SDK's "HTTP 403: <body>" wrapper alike.
_SEMI_AUTO_REFUSAL = "is not allowed in semi-auto mode"


def switch_mode(response: Mapping[str, Any]) -> Mode:
    """The mode an `analyzerstatus` answer reports: `data.analyze_mode`, a boolean
    (services/analyzer_service.py). Raises LiveError for an error or any other answer."""
    if response.get("status") != SUCCESS:
        reason = response.get("message") or "an error without a message"
        raise LiveError(f"OpenAlgo did not report its mode (analyzerstatus: {reason})")
    data = response.get("data")
    analyze_mode = data.get("analyze_mode") if isinstance(data, Mapping) else None
    if not isinstance(analyze_mode, bool):
        raise LiveError(
            "OpenAlgo did not report its mode (analyzerstatus answered without data.analyze_mode)"
        )
    return Mode.SANDBOX if analyze_mode else Mode.LIVE


def semi_auto_seen(response: Mapping[str, Any]) -> bool:
    """True for an answer of semi-auto mode: an order queued in the Action Center, or a
    modify or cancel the server refused because semi-auto is on."""
    if PENDING_ORDER_ID_FIELD in response or response.get("mode") == _SEMI_AUTO_MODE_FIELD:
        return True
    message = response.get("message")
    return (
        response.get("status") != SUCCESS
        and isinstance(message, str)
        and _SEMI_AUTO_REFUSAL in message.lower()
    )


__all__ = ["PENDING_ORDER_ID_FIELD", "SUCCESS", "semi_auto_seen", "switch_mode"]
