"""Why a live session's order closes a position, as its EXIT alert and the event of its
request name it (2.4.0 item M11-X3): an audit tells the engine's exit at a leg's target
from a strategy's own exit, a stop, the square-off or the kill switch without reading the
kernel's records.

The reason is the exit reason the order's fills record in the trades' `exit_reason`, the
code a backtest's trade of that exit carries: the kernel's exit reason of the order
(`Engine.order_exit_reason`), or `SIGNAL` for a strategy's order that names none (a sell
that reduces, as the kernel's fill path records it). An entry or an add closes nothing and
has none: its alert and event are as they were. The event carries the code, the alert its
words in the trader's terms (`EXIT_REASON_WORDS`), but for the session's own risk exits the
kernel records as "manual" (Ctrl+C or a retried exit, a stop not confirmed or rejected,
the strategy's kill switch), which the alert words by the reason the session gave the
kernel for them (`Trader.reasons`, `MANUAL_RISK_EXIT_WORDS`).
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

from .._engine import call
from ..errors import LiveError
from ._order_state import ENTRY_PURPOSES
from ._types import Purpose
from ._venue_requests import PlaceRequest

#: The exit reason the kernel's fill path records for a closing order that names none
#: (rust/oa_bt/src/engine/venue_fills.rs: `exit_reason.unwrap_or(ExitReason::Signal)`).
SIGNAL = "signal"

#: The exit reason the kernel records for a risk exit the session asks for that has no
#: reason of its own (rust/oa_bt/src/engine/venue_roles.rs) and for a strategy's own exit
#: naming "manual".
MANUAL = "manual"

#: Each of the kernel's exit reasons (rust/oa_bt/src/rms/vocabulary.rs, `ExitReason`) in
#: the trader's terms.
EXIT_REASON_WORDS: Mapping[str, str] = MappingProxyType(
    {
        "leg_sl": "stop level",
        "leg_target": "target level",
        "time_exit": "time exit",
        "overall_sl": "combined stop",
        "overall_target": "combined target",
        "lock_profit": "profit lock",
        "daily_loss_limit": "daily loss limit",
        "scheduler": "square-off time",
        MANUAL: "manual exit",
        "eod": "end of the session",
        "drawdown_halt": "drawdown halt",
        "margin_liquidation": "margin call",
        "expiry": "expiry",
        "signal": "strategy exit",
        "account_loss_limit": "kill switch at the account daily loss limit",
        "daily_profit_target": "daily profit target",
        "account_profit_target": "kill switch at the account daily profit target",
        "expiry_exit": "exit before the contract's expiry",
    }
)
#: The session's own risk exits the kernel records as `MANUAL`, in the trader's terms, by
#: the reason the session gave the kernel (its `RISK_EXIT_REASONS`; the place command's
#: "reason"). A strategy's exit asked for again after a rejection is "operator_stop".
MANUAL_RISK_EXIT_WORDS: Mapping[str, str] = MappingProxyType(
    {
        "operator_stop": "Ctrl+C or a retried exit",
        "unprotected": "stop not confirmed in time",
        "stop_rejected": "stop rejected",
        "kill_switch": "kill switch",
    }
)


def exit_reason(trader: Any, kernel_order: int, purpose: Purpose) -> str | None:
    """The exit reason of `trader`'s kernel order `kernel_order`, placed for `purpose`
    (module docstring): None for an entry or an add. Raises what the kernel raises for an
    order it does not know."""
    if purpose in ENTRY_PURPOSES:
        return None
    return call(trader.engine.order_exit_reason, kernel_order) or SIGNAL


def exit_words(trader: Any, kernel_order: int, code: str) -> str:
    """Exit reason `code` of `trader`'s kernel order `kernel_order` in the trader's terms
    (module docstring): a risk exit the kernel records as `MANUAL` by the reason the
    session gave it. Raises LiveError as `reason_words` does."""
    risk = trader.reasons.get(kernel_order) if code == MANUAL else None
    return MANUAL_RISK_EXIT_WORDS.get(risk or "") or reason_words(code)


def reason_words(code: str) -> str:
    """An exit reason's code in the trader's terms. Raises LiveError for a code the kernel
    does not have, naming it: a new reason needs its words here."""
    words = EXIT_REASON_WORDS.get(code)
    if words is None:
        raise LiveError(
            f"the kernel gave the exit reason {code!r}, which the live layer has no words "
            "for: add it to EXIT_REASON_WORDS"
        )
    return words


def request_reason(traders: Any, request: object) -> dict[str, str]:
    """The exit reason the event of `request` carries (module docstring): {"reason": code}
    for the placement of an order that closes a position, else nothing. `traders` are the
    session's, whose names the requests' intents carry."""
    if not isinstance(request, PlaceRequest):
        return {}
    for trader in traders:
        if trader.name == request.intent.strategy:
            code = exit_reason(trader, request.kernel_order, request.intent.purpose)
            return {} if code is None else {"reason": code}
    return {}


__all__ = [
    "EXIT_REASON_WORDS",
    "MANUAL",
    "MANUAL_RISK_EXIT_WORDS",
    "SIGNAL",
    "exit_reason",
    "exit_words",
    "reason_words",
    "request_reason",
]
