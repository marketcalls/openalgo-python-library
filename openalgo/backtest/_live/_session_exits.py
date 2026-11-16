"""Exits that fail (16-session-design 4.8, FR-LIVE-07; unit 8): an exit the broker or
OpenAlgo rejects while its position is still open is asked for again, on a schedule.

An exit (a signal exit, a risk exit, a square-off) that ends rejected while its strategy
still holds a position starts the schedule: the session asks the kernel for the exit again
(`request_exit`) after `EXIT_RETRY_DELAYS_S` (1, 2, 5, 10 seconds after each rejection),
so five attempts in all (08-live-trading section 6.3). `request_exit` exits the whole
position, so only an exit that closes all of it is asked for again: a strategy's sale of
part of its position (a reduce, an EXIT order for fewer units than the position) stays
rejected, as the strategy saw it in `on_order`. After the fifth rejection the
strategy's entries are locked for the rest of the session ("exit_failed") and an ERROR
alert says to close the position by hand in OpenAlgo. The schedule starts again once the
position is flat. A risk exit is asked for again with its own reason; a signal exit or a
square-off with "operator_stop", the kernel's reason for an exit the session asks for
(its trade's exit reason reads "manual"). The kernel's own re-send of a rejected engine
exit at each later open stays as a backstop; its exit owner keeps the two from
overlapping. The broker-held stop the design re-places after the fifth failure comes with
the stops supervisor (unit 10). After the second Ctrl+C (`halt`) no exit is asked for
again: the session only waits for what is in flight (16-session-design 3.3 step 4).

Times are UTC nanoseconds. Used by the loop only.
"""

from __future__ import annotations

import logging
from typing import Any

from .._layout import NANOS_PER_SECOND
from ._session_trader import Trader
from ._types import LOGGER_NAME, AlertKind, OrderIntent, Purpose

logger = logging.getLogger(LOGGER_NAME)

#: Seconds after each rejection of an exit before it is asked for again (4.8).
EXIT_RETRY_DELAYS_S = (1.0, 2.0, 5.0, 10.0)
#: Rejections of an exit after which the session gives up on it.
EXIT_ATTEMPTS = len(EXIT_RETRY_DELAYS_S) + 1
#: The entry lock of a strategy whose exit failed every attempt (6.2).
EXIT_FAILED_LOCK = "exit_failed"
#: The kernel's reason for an exit the session asks for again when the rejected one had
#: none of its own (a signal exit, a square-off).
RETRY_REASON = "operator_stop"
#: The purposes of the orders that close a position.
EXIT_PURPOSES = frozenset({Purpose.EXIT, Purpose.RISK_EXIT, Purpose.SQUARE_OFF})


class ExitRetries:
    """The exit retry schedule of one session (module docstring). `session` is the
    `_session_loop.Session` whose strategies it watches."""

    def __init__(self, session: Any) -> None:
        self._session = session
        self._rejections: dict[str, int] = {}
        self._due_ns: dict[str, int] = {}
        self._reason: dict[str, str] = {}
        self._halted = False

    def rejected(self, trader: Trader, kernel_order: int, intent: OrderIntent, why: str) -> None:
        """An order of `trader` (kernel order `kernel_order`, placed as `intent`) ended
        rejected for `why`: an exit of the whole open position is asked for again on the
        schedule, or given up on after the last attempt; a reduce is not (module
        docstring)."""
        units = abs(trader.open_units())
        if intent.purpose not in EXIT_PURPOSES or units == 0:
            return
        session, name = self._session, trader.name
        if intent.purpose is Purpose.EXIT and intent.quantity < units:
            session.event("reduce_rejected", strategy=name, units=intent.quantity, reason=why)
            logger.info(
                "%s: a reduce of %d units was rejected: not asked for again", name, intent.quantity
            )
            return
        rejections = self._rejections.get(name, 0) + 1
        self._rejections[name] = rejections
        session.event("exit_rejected", strategy=name, attempt=rejections, reason=why)
        if rejections > EXIT_ATTEMPTS:
            # The kernel's own re-send at a later open, rejected again: alerted already.
            return
        if rejections == EXIT_ATTEMPTS:
            self._due_ns.pop(name, None)
            if trader.lock(EXIT_FAILED_LOCK):
                session.event("lock", strategy=name, reason=EXIT_FAILED_LOCK)
            session.alert(
                AlertKind.ERROR,
                f"exit of {trader.symbol} {trader.exchange} failed {rejections} times ({why}): "
                f"{name}'s entries are locked; close the position by hand in OpenAlgo",
            )
            return
        if self._halted:
            return
        delay_ns = round(EXIT_RETRY_DELAYS_S[rejections - 1] * NANOS_PER_SECOND)
        self._due_ns[name] = session.now() + delay_ns
        self._reason[name] = trader.reasons.get(kernel_order) or RETRY_REASON

    def owns(self, name: str) -> bool:
        """True while the exit of strategy `name` is on the schedule: a retry is due, or
        every attempt failed. The schedule ends once the strategy is flat."""
        return name in self._rejections

    def retry_due(self, name: str) -> bool:
        """True while a retry of the exit of strategy `name` is due."""
        return name in self._due_ns

    def halt(self) -> None:
        """Ask for no exit again from now (the second Ctrl+C): the retries due are
        dropped, and a later rejection is counted but not asked for again."""
        self._halted = True
        self._due_ns.clear()

    def duties(self, now_ns: int) -> None:
        """Ask again for each exit whose retry is due; forget the schedule of a strategy
        that is flat."""
        session = self._session
        for trader in session.traders:
            name = trader.name
            if name in self._rejections and trader.open_units() == 0:
                self._rejections.pop(name)
                self._due_ns.pop(name, None)
            due_ns = self._due_ns.get(name)
            if due_ns is None or now_ns < due_ns or session.locked:
                continue
            del self._due_ns[name]
            what = trader.request_exit(self._reason[name])
            session.event("exit_retried", strategy=name, outcome=what)
            logger.info("%s: exit asked for again (%s)", name, what)


__all__ = [
    "EXIT_ATTEMPTS",
    "EXIT_FAILED_LOCK",
    "EXIT_PURPOSES",
    "EXIT_RETRY_DELAYS_S",
    "RETRY_REASON",
    "ExitRetries",
]
