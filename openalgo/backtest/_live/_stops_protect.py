"""Protect by state (16-session-design 3.3, design rule 5): what the session does for each
leg with a position when it must end with no leg left without a broker-held stop while
one can be sent. The rule throughout: never a working exit and a new stop at once.

- Flat, or no stop rule (the leg never had a stop): nothing.
- A confirmed stop for exactly its units and no cancel of it waits: nothing, it is
  protected. One for more units (a sale by hand adopted before its cut went) waits for
  the cut's word, as one not confirmed does; it never protects the leg.
- A cancel of the stop was sent: wait for its word. Cancelled, the cases below apply;
  refused (the stop works on), it is protected.
- The stop not confirmed yet, or held back: wait for its word.
- No stop, and an exit the kernel made that the session has not sent: the exit is
  withdrawn (`cancel`, never sent), then the stop is placed again.
- No stop, and an exit sent with no answer, held back or ambiguous: nothing new (it may
  fill), and an ERROR alert to check the order book.
- No stop, and an exit working at the broker: wait for it. Filled, the leg is flat;
  ended unfilled, the stop is placed again.
- No stop and no exit: `protect`, then wait for the stop's word.

It begins for every leg at the second Ctrl+C, at the safe stop's timeout, with
`on_stop="keep"`, and in the emergency stop (`protect_and_wait`). From then on the leg's
exits are not asked for again and its refused cancels are not sent again. The caller
bounds the wait; `settled` says when a leg needs nothing more, and `may_end` when a
bounded wait may end: at its bound, unless a step of protect by state's own is left
(`acting`: an exit to withdraw, a stop to place, a stop placed that the order worker
has not answered, or whose first word it has not passed on yet), which the wait
finishes for at most `PROTECT_FINISH_WAIT_S` more. A stop placed in the wait's last step
would otherwise be handed to a worker that closes before it sends it.

Times are UTC nanoseconds, quantities units. Used by the loop only.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any

from .._layout import NANOS_PER_SECOND
from ._session_trader import Trader
from ._stops_kernel import PROTECT_PLACED, order_of, protect
from ._stops_orders import IN_FLIGHT, LegOrders, Sent
from ._types import LOGGER_NAME, AlertKind
from ._venue_requests import STOP_TYPES

logger = logging.getLogger(LOGGER_NAME)

#: What a leg is, for protect by state (module docstring).
FLAT, NO_STOP_RULE, PROTECTED = "flat", "no stop rule", "protected"
CANCEL_WAITS, STOP_WAITS = "cancel waits", "stop waits"
EXIT_UNSENT, EXIT_IN_FLIGHT, EXIT_WORKING = "exit unsent", "exit in flight", "exit working"
NO_COVER = "no cover"
#: Seconds a bounded wait goes on past its bound while a step of protect by state's own
#: is left (module docstring): the order worker sends a protective placement within a
#: step, unless it is inside a call, which this bounds.
PROTECT_FINISH_WAIT_S = 5.0
#: The legs that need nothing more.
_SETTLED = frozenset({FLAT, NO_STOP_RULE, PROTECTED})
#: What `protect` answers when nothing can be placed for the leg.
_GIVEN_UP = frozenset({"no_stop_level", "market_closed", "flat"})

_FINISH_NS = round(PROTECT_FINISH_WAIT_S * NANOS_PER_SECOND)


class Protection:
    """Protect by state for the legs of one session (module docstring). `session` is the
    `_session_loop.Session`, `stops` its `StopsSupervisor`."""

    def __init__(self, session: Any, stops: Any) -> None:
        self._session = session
        self._stops = stops
        self._protecting: set[str] = set()
        self._outcome: dict[str, str | None] = {}
        self._withdrawn: set[tuple[str, int]] = set()
        self._alerted: set[tuple[str, int]] = set()

    def begin(self, traders: Iterable[Trader], now_ns: int) -> None:
        """Protect by state from now for each of `traders`: their exit retries and cancel
        retries stop, and the first actions are taken."""
        for trader in traders:
            if trader.name in self._protecting:
                continue
            self._protecting.add(trader.name)
            self._session.exits.halt(trader.name)
            self._stops.cancels.halt(trader.name)
            self._session.event("protect_by_state", strategy=trader.name)
        self.step(now_ns)

    def holds(self, trader: Trader) -> bool:
        """True while protect by state holds a leg with a stop rule: its exit is not asked
        for again."""
        return trader.name in self._protecting and self._stops.leg(trader).guarded

    def step(self, now_ns: int) -> None:
        """Act on each leg held, by its state now."""
        del now_ns  # the actions depend on the state alone; the caller bounds the wait
        for trader in self._session.traders:
            if trader.name in self._protecting:
                self._act(trader)

    def settled(self, trader: Trader) -> bool:
        """True when the leg needs nothing more: flat, no stop rule, protected, or nothing
        can be placed for it."""
        state = self.state(trader)
        return state in _SETTLED or (state == NO_COVER and self._gave_up(trader))

    def acting(self, trader: Trader) -> bool:
        """True while protect by state holds the leg and has a step of its own left, with
        no answer of the broker to wait for: an exit to withdraw, a stop to place, a stop
        placed and not answered yet by the order worker, or one accepted whose first word
        may be among the words the worker or the loop has not taken yet."""
        if trader.name not in self._protecting:
            return False
        state = self.state(trader)
        if state == EXIT_UNSENT or (state == NO_COVER and not self._gave_up(trader)):
            return True
        if state != STOP_WAITS:
            return False
        leg = self._stops.leg(trader)
        stop = leg.orders.get(leg.stop) if leg.stop is not None else None
        if stop is not None and leg.stop in trader.working():
            return stop.state is Sent.HANDED or (stop.state is Sent.BOUND and self._words_wait())
        return any(
            kernel_order not in leg.orders and _is_stop(trader, kernel_order)
            for kernel_order in trader.exit_orders()
        )

    def _words_wait(self) -> bool:
        """Whether order words may wait to be taken: the worker is inside a call, holds
        words it has not posted, or the loop's inbox holds messages (as for the deadline
        of a stop, `_stops_deadline`)."""
        session = self._session
        worker = session.parts.worker
        return worker.busy_since_ns is not None or bool(worker.raw_pending or session.inbox_waiting)

    def may_end(self, now_ns: int, until_ns: int) -> bool:
        """Whether a bounded wait due to end at `until_ns` may end at `now_ns`: from
        `until_ns` once no leg has a step of protect by state's own left (`acting`), and
        `PROTECT_FINISH_WAIT_S` after it in any case."""
        if now_ns < until_ns:
            return False
        if now_ns >= until_ns + _FINISH_NS:
            return True
        return not any(self.acting(trader) for trader in self._session.traders)

    def state(self, trader: Trader) -> str:
        """What the leg is now (module docstring)."""
        units = abs(trader.open_units())
        leg = self._stops.leg(trader)
        if units == 0:
            return FLAT
        if not leg.guarded:
            return NO_STOP_RULE
        stop = leg.orders.get(leg.stop) if leg.stop is not None else None
        if stop is not None and leg.stop in trader.working():
            # A cancel the kernel decided in this loop step is handed over at its end.
            if stop.cancel_sent or leg.stop in self._stops.cancelling(trader):
                return CANCEL_WAITS
            return PROTECTED if self._stops.covered(trader) == units else STOP_WAITS
        exits = [kernel_order for kernel_order in trader.exit_orders() if kernel_order != leg.stop]
        unsent = [kernel_order for kernel_order in exits if kernel_order not in leg.orders]
        if any(_is_stop(trader, kernel_order) for kernel_order in unsent):
            # A stop placed in this loop step (by `protect`, or the kernel's re-placement):
            # the commands hand it over at the step's end.
            return STOP_WAITS
        if unsent:
            return EXIT_UNSENT
        watched = [leg.orders[kernel_order] for kernel_order in exits]
        if any(w.state in IN_FLIGHT for w in watched):
            return EXIT_IN_FLIGHT
        return EXIT_WORKING if watched else NO_COVER

    def _act(self, trader: Trader) -> None:
        state = self.state(trader)
        leg = self._stops.leg(trader)
        if state == EXIT_UNSENT:
            self._withdraw(trader, leg)
        elif state == EXIT_IN_FLIGHT:
            self._in_flight_alert(trader, leg)
        elif state == NO_COVER and not self._gave_up(trader):
            outcome = protect(trader)
            self._outcome[trader.name] = outcome
            self._session.event("protect", strategy=trader.name, outcome=outcome)
            if outcome not in PROTECT_PLACED:
                logger.warning("%s: no stop could be placed again (%s)", trader.name, outcome)

    def _gave_up(self, trader: Trader) -> bool:
        return self._outcome.get(trader.name) in _GIVEN_UP

    def _withdraw(self, trader: Trader, leg: LegOrders) -> None:
        """Cancel the exits the kernel made and the session has not sent: they are
        withdrawn when the commands are taken, and never reach the broker."""
        for kernel_order in trader.exit_orders():
            key = (trader.name, kernel_order)
            if kernel_order in leg.orders or key in self._withdrawn:
                continue
            if _is_stop(trader, kernel_order):
                continue
            self._withdrawn.add(key)
            trader.cancel(kernel_order)
            self._session.event("exit_withdrawn", strategy=trader.name, kernel_order=kernel_order)

    def _in_flight_alert(self, trader: Trader, leg: LegOrders) -> None:
        for watched in leg.exits():
            key = (trader.name, watched.kernel_order)
            if watched.state not in IN_FLIGHT or key in self._alerted:
                continue
            self._alerted.add(key)
            self._session.alert(
                AlertKind.ERROR,
                f"{trader.name}: exit order {watched.name()} of {trader.symbol} "
                f"{trader.exchange} in flight, no stop at the broker: check the order book",
            )


def _is_stop(trader: Trader, kernel_order: int) -> bool:
    """Whether a closing order is a stop order (the leg's: the kernel rejects a resting
    closing order beside a leg's stop)."""
    order = order_of(trader, kernel_order)
    return order is not None and order["order_type"] in STOP_TYPES


def protect_and_wait(session: Any, wait_s: float) -> None:
    """The emergency stop's protect by state (16-session-design 3.5): every leg protected
    by state, then the loop turns with no slice released and no strategy called until
    each leg is settled or the wait of `wait_s` seconds may end (`may_end`)."""
    protection = session.stops.protection
    protection.begin(session.traders, session.now())
    until_ns = session.now() + round(wait_s * NANOS_PER_SECOND)
    while not all(protection.settled(trader) for trader in session.traders):
        now_ns = session.now()
        if protection.may_end(now_ns, until_ns):
            return
        session.step(until_ns if now_ns < until_ns else until_ns + _FINISH_NS, protecting=True)


__all__ = [
    "CANCEL_WAITS",
    "EXIT_IN_FLIGHT",
    "EXIT_UNSENT",
    "EXIT_WORKING",
    "FLAT",
    "NO_COVER",
    "NO_STOP_RULE",
    "PROTECTED",
    "PROTECT_FINISH_WAIT_S",
    "STOP_WAITS",
    "Protection",
    "protect_and_wait",
]
