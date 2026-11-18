"""The day summary's slippage (16-session-design 1.4; decided item 11 of 10.0): what the
broker's MARKET fills cost beyond the fills a backtest of the session's own bars models for
the same orders.

For each fill of a MARKET order a strategy's kernel session placed (an entry, an exit, the
square-off, the session's own exits; not a stop, which is a trigger order, and not a
LIMIT): the live price less the price a backtest fills the same order at, signed against
the trader (a buy that paid more, or a sale that received less, is positive), times the
units and the instrument's multiplier, rupees. The backtest's price is the open of the
first base bar at or after the order's decision, as a backtest fills a MARKET order
(SPEC-35: a minute with no trade has no bar, so it is the next one), worse by the slippage
the run models, priced by the kernel's own rule (`Engine.market_fill_price`).

The decision's instant depends on what made the order:

- **At a bar close, a timer or an open the session reaches**: the strategy's orders (an
  entry, an add, its exit), the square-off and the time exit. The kernel's `decided_at`,
  the instant a backtest decides the same order at. A live session runs a slice once its
  bar has closed, so it sends such an order a moment after that instant, and an order that
  waited for a stop's cancellation later still; it is set against the bar a backtest
  fills it in all the same.
- **On an event while the kernel waits inside a minute**: the exit a leg's target makes on
  the tick that reached it, and every risk exit but the time exit (the daily loss limit or
  the drawdown halt on a tick, an unprotected or rejected stop, the kill switch, the
  account's loss limit, Ctrl+C, an exit asked for again after a rejection). The kernel
  stamps such an order with the open of the slice in progress, which opened before the
  event, so the instant is the one the session placed it at (its intent's `created_ns`).
  The first bar at or after it is the next minute's: while that minute is in progress,
  its trades so far, and when the session ends before its first trade (a Ctrl+C, say),
  none. A live session tests the daily loss limit and the drawdown halt on every tick, so
  their exits are taken as made on the tick that crossed the limit; one made at a bar
  close (a price only a broker candle carried) or made again at an open (after an exit
  that did not fill) is set against the bar opening after the session placed it too.

The bars are the session's own (the series'). For a released slice that is the bar the
kernel ran (its tick bar, or the broker's candle of a minute the catch-up released), else
the broker's candle read for it since: the minute a restart began in only seeds the bar
builder, so the kernel ran it without a bar and its candle came later, and a minute whose
ticks did not arrive is the same. For a slice not released yet it is its trades so far
(`TickBarBuilder.open_so_far`), so the exit of a session that ends inside the minute of its
decision (the square-off) is compared too. A released slice with no bar had no trade and
is passed over, as a backtest passes a missing bar, unless it is still marked missing: no
bar of it is known (the feed was dead, or a restart began in it, and no candle has come),
so it may hold the trade the broker filled at.

A fill whose bar the session cannot give (its minute still marked missing, no trade of its
minute before the session ended, or its slice closed and waits to be released behind the
catch-up), or whose order its kernel session does not know, is not compared, and is
counted (`not_compared`); the fills compared are counted too (`compared`).

Times are UTC nanoseconds, prices rupees, quantities units.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .._engine import call
from ._plan_catch_up import INSTRUMENT
from ._types import OrderIntent, Purpose

#: The price type of the orders whose fills are compared.
MARKET = "MARKET"
#: The sign of a fill's slippage by its side: a buy that paid more is worse.
_AGAINST_THE_TRADER = {"BUY": 1, "SELL": -1}
#: The kernel's exit reason (`ExitReason` codes) of the exit a leg's target makes on a
#: tick, and of the time exit, the one risk exit made at an open.
_LEG_TARGET, _TIME_EXIT = "leg_target", "time_exit"


@dataclass(frozen=True)
class Slippage:
    """A strategy's slippage (module docstring): `rupees`, positive when the live fills
    cost more than the backtest's; `not_compared`, the MARKET fills with no bar to set
    them against; and `compared`, those set against one."""

    rupees: float = 0.0
    not_compared: int = 0
    compared: int = 0


def strategy_slippage(session: Any, trader: Any) -> Slippage:
    """The slippage of `trader`'s MARKET fills in this start of `session` (module
    docstring), rupees rounded to the paisa. Its kernel session must not have finished
    (`finish` takes it). Raises what the kernel raises for an order or a price it cannot
    read."""
    placed = {
        venue.order_id: (kernel_order, venue.intent)
        for kernel_order, venue in trader.venue.placed()
        if venue.order_id and venue.intent.price_type == MARKET
    }
    multiplier = trader.part.setup.instruments[INSTRUMENT].multiplier
    rupees, not_compared, compared = 0.0, 0, 0
    for fill in session.fills:
        found = placed.get(fill.order_id) if fill.strategy == trader.name else None
        if found is None:
            continue
        decided_at = _decided_at(trader, *found)
        bar = None if decided_at is None else _bar_at_or_after(session, trader, decided_at)
        if bar is None:
            not_compared += 1
            continue
        modelled = call(trader.engine.market_fill_price, INSTRUMENT, fill.action, *bar)
        worse = _AGAINST_THE_TRADER[fill.action] * (fill.price - modelled)
        rupees += worse * fill.units * multiplier
        compared += 1
    return Slippage(round(rupees, 2), not_compared, compared)


def _decided_at(trader: Any, kernel_order: int, intent: OrderIntent) -> int | None:
    """The instant kernel order `kernel_order`, placed with `intent`, was decided at
    (module docstring), or None when `trader`'s kernel session does not know it."""
    order = call(trader.engine.order, kernel_order)
    if order is None:
        return None
    exit_reason = call(trader.engine.order_exit_reason, kernel_order)
    if _made_on_an_event(intent.purpose, exit_reason):
        return intent.created_ns
    return order["decided_at"]


def _made_on_an_event(purpose: Purpose, exit_reason: str | None) -> bool:
    """True for an order the engine makes on an event while its kernel session waits
    inside a minute (module docstring): a leg's target exit, a risk exit but the time
    exit. A strategy's exit shares the target exit's purpose and is told from it by the
    exit reason the kernel records."""
    if purpose is Purpose.RISK_EXIT:
        return exit_reason != _TIME_EXIT
    return purpose is Purpose.EXIT and exit_reason == _LEG_TARGET


def _bar_at_or_after(session: Any, trader: Any, decided_at_ns: int) -> tuple[int, float] | None:
    """(open instant, open in rupees) of the first base bar of `trader`'s instrument that
    opens at or after `decided_at_ns` (module docstring), or None when the session cannot
    give it."""
    series, released = trader.part.series, trader.part.released
    builder = session.builders.get(trader.exchange)
    for index, (opened, closed) in enumerate(series.slices):
        if opened < decided_at_ns:
            continue
        if index < released:
            bar = series.tick_bar(opened) or series.bar(opened)
            if bar is not None:
                return opened, bar.open
            if series.missing_slices(opened, closed):
                return None
            continue
        so_far = None if builder is None else builder.open_so_far(trader.symbol, opened)
        return None if so_far is None else (opened, so_far)
    return None


__all__ = ["MARKET", "Slippage", "strategy_slippage"]
