"""Random entries (design 17, section 2.4): a strategy that enters at random strategy bars
with the run's own trade count, direction mix and holding times, and exits each trade
after the same number of strategy bars.

The trades (`EntryPlan.trades`): each closed trade of the run gives one random trade
with its symbol, side, quantity in units, product and holding time in strategy bars
(`bars_held`; a trade held less than one bar is held one, since a decided exit fills at
the next open at the earliest). The random runs therefore hold the run's directions and
holding times trade for trade; only where the trades go is drawn.

Where they go: each symbol's trades are placed in a random order at random places among
its strategy bars of the run that do not overlap. A trade decided at the close of bar i
fills at the next open and is exited by a decision at the close of bar i + h, and the
next trade may be decided at that same close. With U bars whose decision can fill (every
strategy bar of the run but the last) and holding times h_1 to h_n, the slack
S = U - 1 - (h_1 + ... + h_n) is cut at n points drawn uniformly from 0 to S and sorted,
which leaves the n + 1 gaps before, between and after the trades. Trades that do not
fit (S below 0) cannot be placed, and the check says so.

The random strategy is the run's own strategy class with its decisions replaced: it
keeps the class's settings (product, sizing caps, risk rules, entry window, square-off),
so the engine applies the same rules to its trades, which may refuse an entry (outside
the entry window, for margin) or close a trade early (the square-off, a risk rule) as it
would the strategy's; the table of the check shows the random runs' trade counts.

Draws: run r reads 2 draws a trade, symbols in registration order (draws r x 2 x trades
onwards): the order of each symbol's trades (one key each), then its cut points.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .._arrays import MADE_BY
from ..strategy import CALLBACKS, Strategy
from . import _random

#: The least strategy bars a random trade is held (module docstring).
LEAST_HOLD = 1


@dataclass(frozen=True)
class Trade:
    """One trade to place: its symbol, side ("BUY" opens a long, "SELL" a short), units,
    product and holding time in strategy bars."""

    symbol: str
    side: str
    units: float
    product: str
    hold: int


@dataclass(frozen=True, eq=False)
class EntryPlan:
    """What random entries are drawn from: the run's trades, and the UTC nanosecond close
    of each strategy bar of the run, by symbol."""

    trades: tuple[Trade, ...]
    closes: Mapping[str, np.ndarray]

    def slack(self, symbol: str) -> int:
        """The bars `symbol`'s trades leave free (module docstring); below 0 they do not
        fit."""
        holds = sum(trade.hold for trade in self.trades if trade.symbol == symbol)
        return len(self.closes[symbol]) - 2 - holds

    def unplaced(self) -> list[str]:
        """The symbols whose trades do not fit among their strategy bars."""
        symbols = dict.fromkeys(trade.symbol for trade in self.trades)
        return [symbol for symbol in symbols if self.slack(symbol) < 0]


def entry_plan(trade_list: Any, closes: Mapping[str, np.ndarray]) -> EntryPlan:
    """The `EntryPlan` of a run's `trade_list` (`Result.trade_list()`) and its strategy
    bars' closes by symbol."""
    trades = tuple(
        Trade(
            str(row.symbol),
            "BUY" if row.direction == "long" else "SELL",
            float(row.quantity),
            str(row.product),
            max(LEAST_HOLD, int(row.bars_held)),
        )
        for row in trade_list.itertuples()
    )
    return EntryPlan(trades, closes)


@dataclass(frozen=True)
class Action:
    """One decision of the random strategy: enter (`side`, `units`, `product`) or exit
    (`side` None) on `symbol`."""

    symbol: str
    side: str | None = None
    units: float = 0.0
    product: str = ""


def schedule(plan: EntryPlan, seed: int, run: int) -> dict[int, tuple[Action, ...]]:
    """Run `run` of the stream `seed`: the decisions of the random strategy by the UTC
    nanosecond close they are made at (module docstring), exits before entries."""
    draws = _random.block(seed, run, 1, 2 * len(plan.trades))[0]
    exits: dict[int, list[Action]] = {}
    entries: dict[int, list[Action]] = {}
    used = 0
    for symbol in dict.fromkeys(trade.symbol for trade in plan.trades):
        mine = [trade for trade in plan.trades if trade.symbol == symbol]
        count, slack, closes = len(mine), plan.slack(symbol), plan.closes[symbol]
        keys, cuts = draws[used : used + count], draws[used + count : used + 2 * count]
        used += 2 * count
        order = np.argsort(keys, kind="stable")
        gaps = np.diff(np.concatenate(([0], np.sort(_random.below(cuts, slack + 1)))))
        bar = 0
        for gap, position in zip(gaps, order, strict=True):
            trade = mine[position]
            bar += int(gap)
            entry = Action(symbol, trade.side, trade.units, trade.product)
            entries.setdefault(int(closes[bar]), []).append(entry)
            bar += trade.hold
            exits.setdefault(int(closes[bar]), []).append(Action(symbol))
    times = sorted(set(exits) | set(entries))
    return {when: (*exits.get(when, ()), *entries.get(when, ())) for when in times}


def _decide(strategy: Strategy, plan: Mapping[int, Sequence[Action]], bar: Any) -> None:
    some = bar if not isinstance(bar, dict) else next(iter(bar.values()))
    for action in plan.get(int(some.ts_close), ()):
        if action.side is None:
            strategy.exit(action.symbol)
        elif action.side == "BUY":
            strategy.buy(qty=action.units, symbol=action.symbol, product=action.product)
        else:
            strategy.sell(qty=action.units, symbol=action.symbol, product=action.product)


def random_strategy(strategy: type, plan: Mapping[int, Sequence[Action]]) -> type:
    """The run's `strategy` class with its decisions replaced by `plan` (module
    docstring): no indicators, timers or callbacks of its own, and every class setting
    kept."""

    def init(self: Strategy) -> None:
        """No indicators, timers or conditions: the decisions are drawn."""

    def on_bar(self: Strategy, bar: Any) -> None:
        _decide(self, plan, bar)

    attributes: dict[str, Any] = {"init": init, "on_bar": on_bar, MADE_BY: None}
    for name in (*CALLBACKS, "on_stop"):
        attributes[name] = getattr(Strategy, name)
    attributes["__doc__"] = f"Random entries with the trades of {strategy.__name__}."
    return type(f"RandomEntries{strategy.__name__}", (strategy,), attributes)


__all__ = ["Action", "EntryPlan", "Trade", "entry_plan", "random_strategy", "schedule"]
