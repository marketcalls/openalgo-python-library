"""Exits that fail (16-session-design 4.8, FR-LIVE-07; unit 8): an exit the broker or
OpenAlgo rejects while its position is still open is asked for again, on a schedule.

An exit (a signal exit, a risk exit, a square-off) that ends rejected while its strategy
still holds a position starts the schedule: the session asks the kernel for the exit again
(`request_exit`) after `EXIT_RETRY_DELAYS_S` (1, 2, 5, 10 seconds after each rejection),
so five attempts in all (08-live-trading section 6.3). After the fifth rejection the
strategy's entries are locked for the rest of the session ("exit_failed"), the leg's stop
is placed again (`protect`, R7) so the broker holds one, and an ERROR alert says what
protect found: the stop placed again, or one working already, protects the position; an
exit waits or works; or no stop is at the broker. A sale by hand is never advised beside a
working stop (16-session-design 3.5, 4.8), nor while the session runs
(`_stops_report.while_running`): until it ends the kernel sends the exit again at each
later open (a backstop its exit owner keeps from overlapping the schedule) and places the
stop again should the operator cancel it (4.10), so a sale by hand would sell twice. The
alert says to end the session first, then to cancel the stop, or to read the order book
first while an exit waits or works. A later rejection (that backstop's) places the stop
again too, with no alert. While the retries run the leg has no stop at the broker (it was
cancelled for the exit) and counts as unprotected. The schedule starts again once the
position is flat. A risk exit is asked for again with its own reason; a signal exit, a
target or a square-off with "operator_stop", the kernel's reason for an exit the session
asks for (its trade's exit reason reads "manual").

A partial exit is not asked for again (decided for unit 10): `request_exit`, the only
retry call, exits the whole position, so a rejected sale of part of it (a `reduce()`, an
EXIT order for fewer units than the position, a partial target) stays rejected, as the
strategy saw it in `on_order`, with the REJECT alert sent for every rejection. Turning it
into a full exit would close more than the strategy asked for. Part or all is judged
against the position when the exit was placed (`Trader.units_at_place`), so the outcome
does not depend on whether another exit's fill is heard before the rejection. The units
it was for stay protected: when the partial exit alone was working, the kernel cut the
stop down only for its units and puts it back over the whole position once it has ended
unfilled (R4); when it and the exits beside it closed the whole position (two targets on
one tick, a `reduce()` then an `exit()`), the kernel cancelled the stop for them and does
not place it again, so the stops supervisor does (`_stops_bare`: `protect`, R7, with an
ERROR alert) once the leg holds units with nothing working to close them. After the
second Ctrl+C, or once protect by state holds a leg (`halt`), no exit is asked for again:
the session only waits for what is in flight (16-session-design 3.3).

Times are UTC nanoseconds. Used by the loop only.
"""

from __future__ import annotations

import logging
from typing import Any

from .._layout import NANOS_PER_SECOND
from ._session_trader import Trader
from ._stops_kernel import PROTECT_PLACED, protect
from ._stops_report import protect_words, while_running
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
        self._halted: set[str] = set()
        self._all_halted = False

    def rejected(self, trader: Trader, kernel_order: int, intent: OrderIntent, why: str) -> None:
        """An order of `trader` (kernel order `kernel_order`, placed as `intent`) ended
        rejected for `why`: an exit of the whole open position is asked for again on the
        schedule, or given up on after the last attempt; a reduce is not (module
        docstring)."""
        units = abs(trader.open_units())
        if intent.purpose not in EXIT_PURPOSES or units == 0:
            return
        session, name = self._session, trader.name
        if intent.purpose is Purpose.EXIT and intent.quantity < _units_at_place(
            trader, kernel_order, units
        ):
            session.event("reduce_rejected", strategy=name, units=intent.quantity, reason=why)
            logger.info(
                "%s: a reduce of %d units was rejected: not asked for again", name, intent.quantity
            )
            return
        rejections = self._rejections.get(name, 0) + 1
        self._rejections[name] = rejections
        session.event("exit_rejected", strategy=name, attempt=rejections, reason=why)
        if rejections >= EXIT_ATTEMPTS:
            self._give_up(trader, rejections, why)
            return
        if self.halted(name):
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

    def halt(self, strategy: str | None = None) -> None:
        """Ask for no exit of `strategy` again from now (every strategy's when None: the
        second Ctrl+C): its retries due are dropped, and a later rejection is counted but
        not asked for again."""
        if strategy is None:
            self._all_halted = True
            self._due_ns.clear()
            return
        self._halted.add(strategy)
        self._due_ns.pop(strategy, None)

    def halted(self, strategy: str) -> bool:
        """True once no exit of `strategy` is asked for again."""
        return self._all_halted or strategy in self._halted

    def _give_up(self, trader: Trader, rejections: int, why: str) -> None:
        """The last attempt failed, or a later exit did: the broker holds a stop again
        (`protect`), alerted once with the entries locked, in words of what protect
        achieved (module docstring)."""
        session, name = self._session, trader.name
        self._due_ns.pop(name, None)
        protected = protect(trader)
        session.event("protect", strategy=name, outcome=protected)
        if rejections > EXIT_ATTEMPTS:
            return
        if trader.lock(EXIT_FAILED_LOCK):
            session.event("lock", strategy=name, reason=EXIT_FAILED_LOCK)
        if protected not in PROTECT_PLACED:
            logger.warning("%s: no stop placed again (%s)", name, protected)
        held, then = protect_words(protected)
        session.alert(
            AlertKind.ERROR,
            f"exit of {trader.symbol} {trader.exchange} failed {rejections} times ({why}): "
            f"{name}'s entries are locked; {held}; {while_running(then)}",
        )

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


def _units_at_place(trader: Trader, kernel_order: int, units_now: float) -> float:
    """The position's units when the exit was placed, which tell an exit of part of it
    from one of all of it whatever came between (another exit's fill heard before this
    one's rejection); the units now for an order the session did not see placed."""
    return trader.units_at_place.get(kernel_order, units_now)


__all__ = [
    "EXIT_ATTEMPTS",
    "EXIT_FAILED_LOCK",
    "EXIT_PURPOSES",
    "EXIT_RETRY_DELAYS_S",
    "RETRY_REASON",
    "ExitRetries",
]
