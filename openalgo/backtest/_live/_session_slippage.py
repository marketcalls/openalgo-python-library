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
the run models, priced by the kernel's own rule (`Engine.market_fill_price`). The bars are
the session's own (the series'). For a released slice that is the bar the kernel ran (its
tick bar, or the broker's candle of a minute the catch-up released), else the broker's
candle read for it since: the minute a restart began in only seeds the bar builder, so the
kernel ran it without a bar and its candle came later, and a minute whose ticks did not
arrive is the same. For a slice not released yet it is its trades so far
(`TickBarBuilder.open_so_far`), so the exit of a session that ends inside the minute of its
decision (Ctrl+C, the square-off) is compared too. A released slice with no bar had no
trade and is passed over, as a backtest passes a missing bar, unless it is still marked
missing: no bar of it is known (the feed was dead, or a restart began in it, and no candle
has come), so it may hold the trade the broker filled at.

A fill whose bar the session cannot give (its minute still marked missing, no trade of its
minute before the session ended, or its slice closed and waits to be released behind the
catch-up), or whose order its kernel session does not know, is not compared, and is
counted (`not_compared`).

Times are UTC nanoseconds, prices rupees, quantities units.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .._engine import call
from ._plan_catch_up import INSTRUMENT

#: The price type of the orders whose fills are compared.
MARKET = "MARKET"
#: The sign of a fill's slippage by its side: a buy that paid more is worse.
_AGAINST_THE_TRADER = {"BUY": 1, "SELL": -1}


@dataclass(frozen=True)
class Slippage:
    """A strategy's slippage (module docstring): `rupees`, positive when the live fills
    cost more than the backtest's, and `not_compared`, the MARKET fills with no bar to
    set them against."""

    rupees: float = 0.0
    not_compared: int = 0


def strategy_slippage(session: Any, trader: Any) -> Slippage:
    """The slippage of `trader`'s MARKET fills in this start of `session` (module
    docstring), rupees rounded to the paisa. Its kernel session must not have finished
    (`finish` takes it). Raises what the kernel raises for an order or a price it cannot
    read."""
    placed = {
        venue.order_id: kernel_order
        for kernel_order, venue in trader.venue.placed()
        if venue.order_id and venue.intent.price_type == MARKET
    }
    multiplier = trader.part.setup.instruments[INSTRUMENT].multiplier
    rupees, not_compared = 0.0, 0
    for fill in session.fills:
        kernel_order = placed.get(fill.order_id) if fill.strategy == trader.name else None
        if kernel_order is None:
            continue
        order = call(trader.engine.order, kernel_order)
        bar = None if order is None else _bar_at_or_after(session, trader, order["decided_at"])
        if bar is None:
            not_compared += 1
            continue
        modelled = call(trader.engine.market_fill_price, INSTRUMENT, fill.action, *bar)
        worse = _AGAINST_THE_TRADER[fill.action] * (fill.price - modelled)
        rupees += worse * fill.units * multiplier
    return Slippage(round(rupees, 2), not_compared)


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
