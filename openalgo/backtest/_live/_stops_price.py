"""Triggered but unfilled: the price rule (16-session-design 4.6, decided by price, never
by status words).

For every working leg stop, on each price the session sees (ticks, and quotes when the
feed is silent), against the stop's terms the broker confirmed (the kernel keeps the old
terms until `apply_modify`):

- **Crossed**: a sell stop at or below its trigger, a buy stop at or above. The first
  crossing time is kept, and not cleared if the price comes back (a triggered stop-limit
  rests as a limit at the exchange).
- No fill of the stop within `TRIGGERED_UNFILLED_AFTER_S` of the crossing: the stop is
  repriced to a marketable LIMIT (`reprice_exit`, R7), the last price moved
  `OPTIONS_STOP_LIMIT_BUFFER_PERCENT` against the trader on the tick away from the trader,
  for the order's total units (a partly filled stop too). Still unfilled that long after:
  priced again from the latest price, at most `REPRICE_ATTEMPTS` times. A fill restarts
  the clock.
- A reprice refused (`apply_modify_refused`) or the attempts used up: the position is
  exited at a marketable LIMIT (`request_exit(stop_unfilled, limit_rupees=...)`), with an
  ERROR alert; the owner cancels the stop first, and that LIMIT exit is repriced by this
  same rule while it does not fill (its clock runs from its acceptance).
- An order whose cancel the kernel has decided (its exit owner's, for an exit of the
  position) is neither repriced nor fallen back from: the kernel takes no modify of it,
  and the cancel's word decides. Its clock is kept: a cancel refused lets the rule run on
  at once, an order ended forgets it.

Times are UTC nanoseconds, prices rupees. Used by the loop only.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from .._layout import NANOS_PER_SECOND
from ._margin_reject import HELD_EXIT_WORDS, HELD_FOR_READ
from ._session_trader import Trader
from ._stops_kernel import cancelling, marketable_limit, order_of, reprice, tick_size
from ._types import LOGGER_NAME, AlertKind
from ._venue_requests import STOP_TYPES

logger = logging.getLogger(LOGGER_NAME)

#: Seconds a crossed stop may stay unfilled before it is repriced, and between reprices.
TRIGGERED_UNFILLED_AFTER_S = 2.0
#: Reprices of one order at most before the position is exited at a limit.
REPRICE_ATTEMPTS = 5
#: The kernel's reason for the exit at a limit the rule falls back on (its
#: RISK_EXIT_REASONS).
STOP_UNFILLED = "stop_unfilled"

_WAIT_NS = round(TRIGGERED_UNFILLED_AFTER_S * NANOS_PER_SECOND)


@dataclass
class _Clock:
    """A crossed order of strategy `strategy`: first crossed at `first_ns`, its next
    reprice due at `due_ns`, repriced `attempts` times; `fallback` for the exit at a
    limit the rule fell back on, which is not fallen back from again."""

    strategy: str
    kernel_order: int
    first_ns: int
    due_ns: int
    attempts: int = 0
    fallback: bool = False


class PriceRule:
    """The price rule of one session (module docstring). `session` is the
    `_session_loop.Session` whose strategies it watches."""

    def __init__(self, session: Any) -> None:
        self._session = session
        self._clocks: dict[tuple[str, int], _Clock] = {}
        self._prices: dict[str, float] = {}

    def observe(self, trader: Trader, stop: int | None, price: float, at_ns: int) -> None:
        """A price of `trader`'s instrument, rupees at `at_ns`; `stop` is its leg's stop."""
        self._prices[trader.name] = price
        if stop is None or (trader.name, stop) in self._clocks:
            return
        order = order_of(trader, stop)
        if order is None or order["order_type"] not in STOP_TYPES:
            return
        trigger = order["trigger_rupees"]
        sale = order["side"] == "SELL"
        if (sale and price <= trigger) or (not sale and price >= trigger):
            self._start(trader.name, stop, at_ns)
            self._session.event("stop_crossed", strategy=trader.name, kernel_order=stop)

    def track_exit(self, trader: Trader, kernel_order: int, at_ns: int) -> None:
        """The exit at a limit the rule fell back on was accepted at `at_ns`: its clock
        starts."""
        self._start(trader.name, kernel_order, at_ns).fallback = True

    def watching(self, trader: Trader, kernel_order: int) -> bool:
        """True while the rule watches the order (crossed, or the rule's own exit)."""
        return (trader.name, kernel_order) in self._clocks

    def filled(self, trader: Trader, kernel_order: int, at_ns: int) -> None:
        """A fill of the order at `at_ns` restarts its clock."""
        clock = self._clocks.get((trader.name, kernel_order))
        if clock is not None:
            clock.due_ns = at_ns + _WAIT_NS

    def ended(self, trader: Trader, kernel_order: int) -> None:
        """The order ended: the rule forgets it."""
        self._clocks.pop((trader.name, kernel_order), None)

    def refused(self, trader: Trader, kernel_order: int) -> None:
        """A reprice of the order was refused: exit the position at a limit now."""
        clock = self._clocks.get((trader.name, kernel_order))
        if clock is not None and kernel_order not in cancelling(trader):
            self._fall_back(trader, clock, "its reprice was refused")

    def due(self, now_ns: int) -> None:
        """Reprice each crossed order whose clock ran out; fall back after the last
        attempt."""
        for key, clock in list(self._clocks.items()):
            if clock.due_ns > now_ns:
                continue
            trader = self._session.trader_named(clock.strategy)
            if clock.kernel_order in cancelling(trader):
                continue
            limit = self._limit(trader, clock.kernel_order)
            if limit is None:
                del self._clocks[key]
                continue
            if clock.attempts >= REPRICE_ATTEMPTS:
                self._fall_back(trader, clock, f"{clock.attempts} reprices did not fill it")
                continue
            sent = reprice(trader, clock.kernel_order, limit)
            if sent is None:
                del self._clocks[key]
                continue
            clock.attempts += 1
            clock.due_ns = now_ns + _WAIT_NS
            self._session.event(
                "repriced",
                strategy=trader.name,
                kernel_order=clock.kernel_order,
                limit=limit,
                attempt=clock.attempts,
                sent_now=sent,
            )

    def _start(self, strategy: str, kernel_order: int, at_ns: int) -> _Clock:
        clock = _Clock(strategy, kernel_order, at_ns, at_ns + _WAIT_NS)
        self._clocks[(strategy, kernel_order)] = clock
        return clock

    def _limit(self, trader: Trader, kernel_order: int) -> float | None:
        """The marketable limit of the order from the latest price; None when it no longer
        works or no price has come."""
        order = order_of(trader, kernel_order)
        price = self._prices.get(trader.name)
        if order is None or price is None or kernel_order not in trader.working():
            return None
        return marketable_limit(price, str(order["side"]), tick_size(trader))

    def exit_limit(self, trader: Trader) -> float | None:
        """The marketable LIMIT that exits `trader`'s position at the latest price it saw,
        rupees; None before a price came."""
        price = self._prices.get(trader.name)
        if price is None:
            return None
        side = "SELL" if trader.open_units() > 0 else "BUY"
        return marketable_limit(price, side, tick_size(trader))

    def _fall_back(self, trader: Trader, clock: _Clock, why: str) -> None:
        """Exit the position at a marketable LIMIT (the clock's last word), once; through
        the margin gate, whose answer the alert says (`_margin_reject`)."""
        self._clocks.pop((clock.strategy, clock.kernel_order), None)
        if clock.fallback or trader.finished:
            logger.warning("%s: o%d does not fill (%s)", trader.name, clock.kernel_order, why)
            return
        limit = self._limit(trader, clock.kernel_order) or self.exit_limit(trader)
        if limit is None:
            return
        session = self._session
        asked = session.margin.request_exit(trader, STOP_UNFILLED, float(limit))
        session.event("stop_unfilled", strategy=trader.name, limit=limit, outcome=asked)
        what = (
            f"the position is exited with a LIMIT at {limit:.2f}, repriced while it does not fill"
        )
        if asked == HELD_FOR_READ:
            what = HELD_EXIT_WORDS
        session.alert(
            AlertKind.ERROR,
            f"{trader.name}: the stop of {trader.symbol} {trader.exchange} was reached and "
            f"did not fill ({why}): {what}",
        )


__all__ = ["REPRICE_ATTEMPTS", "STOP_UNFILLED", "TRIGGERED_UNFILLED_AFTER_S", "PriceRule"]
