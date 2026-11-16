"""Cancels the venue refused (16-session-design 4.5): a cancel of a stop or an exit that
was refused or unanswered, and whose order the worker's read found still working, has
been taken off the kernel's cancelling list (`apply_cancel_refused`). The session sends
it again (`cancel`, R7) after `CANCEL_RETRY_DELAYS_S`, then every `CANCEL_RETRY_EVERY_S`,
with an ERROR alert after the `CANCEL_REFUSALS_ALERT`th refusal naming the leg, the
order and the reason. The exit owner keeps waiting meanwhile, and the stop keeps
protecting.

Nothing is sent again for an order that ended, after the guard locked, once protect by
state holds the leg (a refused cancel then leaves the stop protecting, 3.3), or for a
cancel refused for good (the order path cannot read the order: one ERROR alert from the
worker, and the stop stays at the broker).

Times are UTC nanoseconds. Used by the loop only.
"""

from __future__ import annotations

import logging
from typing import Any

from .._layout import NANOS_PER_SECOND
from ._session_trader import Trader
from ._types import LOGGER_NAME, AlertKind

logger = logging.getLogger(LOGGER_NAME)

#: Seconds after the first, second and third refusal before the cancel is sent again.
CANCEL_RETRY_DELAYS_S = (1.0, 2.0, 5.0)
#: Seconds between two sends after the third refusal.
CANCEL_RETRY_EVERY_S = 5.0
#: The refusal after which an ERROR alert names the leg, the order and the reason.
CANCEL_REFUSALS_ALERT = 3

_Key = tuple[str, int]


class CancelRetries:
    """The cancels to send again in one session (module docstring). `session` is the
    `_session_loop.Session` whose strategies it watches."""

    def __init__(self, session: Any) -> None:
        self._session = session
        self._due: dict[_Key, int] = {}
        self._refusals: dict[_Key, int] = {}
        self._reasons: dict[_Key, str] = {}
        self._halted: set[str] = set()

    def reason(self, trader: Trader, kernel_order: int, why: str) -> None:
        """A cancel of the order was refused or unanswered for `why` (kept for the alert, on
        one line: an HTTP body may carry line breaks)."""
        self._reasons[(trader.name, kernel_order)] = " ".join(why.split()) or "no reason given"

    def refused(self, trader: Trader, kernel_order: int, name: str, now_ns: int) -> None:
        """The order (named `name` in alerts) works on after its cancel was refused:
        schedule the cancel again."""
        key = (trader.name, kernel_order)
        refusals = self._refusals.get(key, 0) + 1
        self._refusals[key] = refusals
        if trader.name in self._halted:
            return
        delays = CANCEL_RETRY_DELAYS_S
        delay = delays[refusals - 1] if refusals <= len(delays) else CANCEL_RETRY_EVERY_S
        self._due[key] = now_ns + round(delay * NANOS_PER_SECOND)
        if refusals == CANCEL_REFUSALS_ALERT:
            why = self._reasons.get(key, "no reason given")
            self._session.alert(
                AlertKind.ERROR,
                f"{trader.name}: the cancel of order {name} of {trader.symbol} "
                f"{trader.exchange} was refused {refusals} times ({why}): it is sent again "
                f"every {CANCEL_RETRY_EVERY_S:g} s; the stop protects the position and its "
                "exit waits",
            )

    def for_good(self, trader: Trader, kernel_order: int) -> None:
        """The cancel cannot be settled by any read: it is not sent again."""
        key = (trader.name, kernel_order)
        self._due.pop(key, None)
        self._refusals[key] = self._refusals.get(key, 0) + 1

    def ended(self, trader: Trader, kernel_order: int) -> None:
        """The order ended: nothing more is sent for it."""
        key = (trader.name, kernel_order)
        self._due.pop(key, None)
        self._refusals.pop(key, None)
        self._reasons.pop(key, None)

    def halt(self, strategy: str) -> None:
        """Protect by state holds the leg of `strategy`: no cancel is sent again for it."""
        self._halted.add(strategy)
        for key in [key for key in self._due if key[0] == strategy]:
            del self._due[key]

    def due(self, now_ns: int) -> None:
        """Send again each cancel whose wait has passed."""
        session = self._session
        for key, due_ns in list(self._due.items()):
            if due_ns > now_ns or session.locked:
                continue
            del self._due[key]
            trader = session.trader_named(key[0])
            if key[1] not in trader.working():
                continue
            trader.cancel(key[1])
            session.event("cancel_again", strategy=trader.name, kernel_order=key[1])
            logger.info("%s: the cancel of o%d is sent again", trader.name, key[1])


__all__ = [
    "CANCEL_REFUSALS_ALERT",
    "CANCEL_RETRY_DELAYS_S",
    "CANCEL_RETRY_EVERY_S",
    "CancelRetries",
]
