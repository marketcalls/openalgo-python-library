"""`self.rebalance(weights)`: trade a portfolio to target weights at one decision
(11-portfolio-rebalancing 2.1; the subset built here is described below).

    class EqualWeight(bt.Strategy):
        symbols, exchange, interval, product = ("RELIANCE", "TCS"), "NSE", "D", "CNC"

        def init(self):
            self.quarter = None

        def on_bar(self, bars):
            quarter = (self.now.year, (self.now.month - 1) // 3)
            if quarter != self.quarter:          # the first session of each quarter
                self.quarter = quarter
                self.rebalance({"RELIANCE": 50, "TCS": 50})

How one rebalance is computed:

1. The portfolio value V is the equity at the decision (cash plus every holding at its
   last close, holdings outside the weights included), taken once.
2. Each traded symbol's target is `weight / 100 x V x (1 - cash_buffer / 100)` rupees;
   a symbol of the strategy missing from the weights has weight 0 and is sold to flat.
   The target is converted at the decision price and rounded towards zero to the venue
   rule, as `target(value=...)` does, and one MARKET DAY order trades the difference.
3. The orders are sent sales first (largest first), then buys (largest first), ties by
   symbol, and fill at the next base-bar open: every sale before every buy, whatever
   order the symbols were registered in, so the buys can use the sales' proceeds as far
   as the dated sale-credit rule allows (all of them from 2024-10-07, 80 percent on the
   day from 2020-12-01 to 2024-10-06).
4. The cash buffer is what pays the charges and absorbs a move between the decision
   close and the next open. A buy the cash does not cover at its fill is rejected with
   reject code "margin" and shows in `result.orders`; the next rebalance corrects it.

Not built (11-portfolio-rebalancing): scaling every buy to the cash left at the fill,
no-trade rules (minimum trade, minimum weight change, turnover cap), weight tables and
weight functions, drift bands, the rebalance log, `bt.rebalance`. Scheduling stays in
the strategy: call `rebalance` from `on_bar` when the period changes, or from a
`self.every("quarter", fn)` timer.
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Mapping
from typing import Any

from ._orders import Order
from .errors import ConfigError

PERCENT = 100.0
# Percent of equity a rebalance leaves in cash by default: delivery charges on the
# trades (well under a tenth of a percent of the portfolio for a quarterly rebalance)
# and a move of the portfolio between the decision close and the next open.
DEFAULT_CASH_BUFFER = 2.0
# Weights may add up to 100 percent; this much more is binary rounding of the inputs.
SUM_TOLERANCE = 1e-9


def _number(value: Any) -> bool:
    real = isinstance(value, numbers.Real) and not isinstance(value, bool)
    return real and math.isfinite(value)


def target_weights(weights: Any, symbols: tuple[str, ...]) -> dict[str, float]:
    """Every symbol of the strategy with its weight in percent of equity, 0 for one the
    weights leave out. Raises ConfigError for weights that are not a mapping, a symbol
    the strategy does not trade, a weight that is not a number from 0 to 100, or
    weights that add up to more than 100."""
    if not isinstance(weights, Mapping):
        raise ConfigError(
            "rebalance takes weights as a dict of percent of equity by symbol, such as "
            f"{{'RELIANCE': 25, 'TCS': 25}}, got {weights!r}"
        )
    unknown = sorted(str(symbol) for symbol in weights if symbol not in symbols)
    if unknown:
        raise ConfigError(f"rebalance: {unknown} not traded by this strategy: {list(symbols)}")
    for symbol, weight in weights.items():
        if not (_number(weight) and 0 <= weight <= PERCENT):
            raise ConfigError(
                f"rebalance: the weight of {symbol} is a percent of equity from 0 to 100, "
                f"got {weight!r}"
            )
    total = float(sum(weights.values()))
    if total > PERCENT + SUM_TOLERANCE:
        raise ConfigError(
            f"rebalance: the weights add up to {total:g} percent of equity; they may add "
            "up to 100 at most (the rest stays cash)"
        )
    return {symbol: float(weights.get(symbol, 0.0)) for symbol in symbols}


def cash_buffer_fraction(cash_buffer: Any) -> float:
    """The cash buffer as a fraction of equity. Raises ConfigError unless it is a
    percent from 0 up to (not including) 100."""
    if not (_number(cash_buffer) and 0 <= cash_buffer < PERCENT):
        raise ConfigError(
            f"rebalance: cash_buffer is a percent of equity from 0 to below 100, got "
            f"{cash_buffer!r}"
        )
    return float(cash_buffer) / PERCENT


def trade_order(changes: dict[str, float]) -> list[str]:
    """The symbols in the order their trades are sent: sales (a negative change in
    rupees) largest first, then buys largest first, ties by symbol."""
    sales = sorted((s for s, c in changes.items() if c < 0), key=lambda s: (changes[s], s))
    buys = sorted((s for s, c in changes.items() if c >= 0), key=lambda s: (-changes[s], s))
    return sales + buys


def rebalance(ctx: Any, weights: Any, cash_buffer: Any, leverage: float | None) -> dict[str, Order]:
    """Send the orders of one rebalance (see the module docstring) through the run
    context `ctx`; returns the orders sent by symbol (a symbol already at its target
    has none, and so has one whose weight rounds to no share while none is held, which
    the context records as skipped). Raises ConfigError as `target_weights` and
    `cash_buffer_fraction`."""
    symbols = tuple(ctx.names)
    wanted = target_weights(weights, symbols)
    invested = 1.0 - cash_buffer_fraction(cash_buffer)
    portfolio_value = ctx.account()[0]
    targets = {s: w / PERCENT * portfolio_value * invested for s, w in wanted.items()}
    changes = {s: targets[s] - _held_value(ctx, s) for s in symbols}
    sent = {}
    for symbol in trade_order(changes):
        request = {
            "instrument": ctx.instrument_index(symbol),
            "product": ctx.product,
            "kind": "rupees",
            "amount": targets[symbol],
            "leverage": leverage,
        }
        order = ctx.submit_rebalance(request)
        if order is not None:
            sent[symbol] = order
    return sent


def _held_value(ctx: Any, symbol: str) -> float:
    """The value of a symbol's position at its last mark, rupees (0 when flat): its
    cost plus its open P&L."""
    position = ctx.position(symbol)
    if position.qty == 0:
        return 0.0
    multiplier = ctx.instruments[ctx.symbol_of(ctx.instrument_index(symbol))].multiplier
    return position.qty * position.avg_price * multiplier + position.unrealized


__all__ = [
    "DEFAULT_CASH_BUFFER",
    "cash_buffer_fraction",
    "rebalance",
    "target_weights",
    "trade_order",
]
