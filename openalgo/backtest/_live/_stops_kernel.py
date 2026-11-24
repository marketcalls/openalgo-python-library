"""A leg's kernel calls for the stops supervisor (16-session-design 2.10, R5 and R7; the
kernel's third unit, rust/oa_py/src/backtest/engine/legs.rs): tick mode, the stop placed
again (`protect`), a stop repriced to a LIMIT (`reprice_exit`), the exit at a limit, and
the price a marketable LIMIT is sent at.

A refusal the session expects (the kernel session has finished, it is paused at a bar
close, the order ended meanwhile) is logged at debug level and reported to the caller;
any other error of the kernel is raised, which stops the session (16-session-design 3.5).
Prices are rupees on the instrument's tick, times UTC nanoseconds. Used by the loop only.
"""

from __future__ import annotations

import logging
import math
from typing import Any

from .._engine import call
from ..errors import VenueRefusalError
from ._plan_catch_up import INSTRUMENT
from ._session_trader import Trader
from ._types import LOGGER_NAME

logger = logging.getLogger(LOGGER_NAME)

#: Percent a marketable LIMIT is priced beyond the last price, against the trader: the
#: kernel's `OPTIONS_STOP_LIMIT_BUFFER_PERCENT` (rust/oa_bt/src/exec/segment.rs, the
#: limit of an option's stop beyond its trigger), which the binding does not export;
#: 16-session-design 4.6 and open decision 8 use the one buffer for both.
OPTIONS_STOP_LIMIT_BUFFER_PERCENT = 5.0
#: What `protect` answers when it placed a stop or will once the orders in its way end
#: (the binding's `PROTECT_OUTCOMES`).
PROTECT_PLACED = frozenset({"placed", "waits_for_cancels"})
#: Refusals of a price that mean the kernel session cannot take one now.
_PRICE_RACES = frozenset({"session_over", "at_bar_close"})
#: Refusals of an order's call that mean the order ended meanwhile.
_ORDER_RACES = frozenset({"order_ended", "not_pending"})


def push_price(trader: Trader, price: float, at_ns: int) -> bool:
    """Hand `trader`'s kernel session the instrument's last price, rupees at `at_ns`
    (tick mode: trail, breakeven, targets, the mark and the loss limits, R5). False when
    the session takes no price now (finished, or past its close)."""
    if trader.finished:
        return False
    try:
        call(trader.engine.push_tick, INSTRUMENT, float(price), int(at_ns))
    except VenueRefusalError as error:
        if error.kind not in _PRICE_RACES:
            raise
        logger.debug("%s: a price not taken (%s)", trader.name, error.kind)
        return False
    return True


def protect(trader: Trader) -> str | None:
    """Place the leg's stop again at its level for what the position's exits leave (R7),
    in the strategy's product; what the kernel did, one of the binding's
    `PROTECT_OUTCOMES`, or None once the kernel session finished."""
    if trader.finished:
        return None
    return str(call(trader.engine.protect, INSTRUMENT, trader.product))


def reprice(trader: Trader, kernel_order: int, limit_rupees: float) -> bool | None:
    """Modify a leg's stop the price passed (or the engine's exit at a limit) to a LIMIT
    at `limit_rupees` for the order's total units (R7): True when the modify went now,
    False when it waits for the answer to the one sent before, None when the order ended
    meanwhile or the kernel session finished."""
    if trader.finished:
        return None
    try:
        return bool(call(trader.engine.reprice_exit, kernel_order, float(limit_rupees)))
    except VenueRefusalError as error:
        if error.kind not in _ORDER_RACES:
            raise
        logger.debug("%s: o%d not repriced (%s)", trader.name, kernel_order, error.kind)
        return None


def order_of(trader: Trader, kernel_order: int) -> dict[str, Any] | None:
    """The kernel's order as a dict (its confirmed terms: units, filled units, order type,
    limit and trigger in rupees, side), or None for one it does not know."""
    found = call(trader.engine.order, kernel_order)
    return None if found is None else dict(found)


def cancelling(trader: Trader) -> frozenset[int]:
    """The kernel orders whose cancel the kernel has decided and the venue not answered:
    those of its saved state's desk (`live_state`, R14), which lists a cancel from the
    moment it is decided, before its command is taken."""
    if trader.finished:
        return frozenset()
    desk = call(trader.engine.live_state)["desk"]
    return frozenset(int(kernel_order) for kernel_order in desk["cancelling"])


def tick_size(trader: Trader) -> float:
    """The tick of `trader`'s instrument, rupees (the preflight's facts)."""
    return float(trader.part.setup.instruments[INSTRUMENT].tick_size)


def marketable_limit(price: float, side: str, tick: float) -> float:
    """A LIMIT for `side` ("BUY" or "SELL") that trades now: `price` moved
    `OPTIONS_STOP_LIMIT_BUFFER_PERCENT` against the trader (below for a sale), on the tick
    away from the trader, never below one tick. Rupees."""
    buffer = OPTIONS_STOP_LIMIT_BUFFER_PERCENT / 100.0
    steps = price * (1.0 + buffer if side == "BUY" else 1.0 - buffer) / tick
    # The noise of float arithmetic must not move a price that is on the tick already.
    steps = round(steps, 6)
    whole = math.ceil(steps) if side == "BUY" else math.floor(steps)
    places = max(0, -int(math.floor(math.log10(tick))) + 2)
    return round(max(whole, 1) * tick, places)


__all__ = [
    "OPTIONS_STOP_LIMIT_BUFFER_PERCENT",
    "PROTECT_PLACED",
    "cancelling",
    "marketable_limit",
    "order_of",
    "protect",
    "push_price",
    "reprice",
    "tick_size",
]
