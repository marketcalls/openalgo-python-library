"""Random entries (design 17, section 2.4): a strategy that enters at random strategy bars
with the run's own trade count, direction mix and holding times, and exits each trade
after the same number of strategy bars.

The trades (`EntryPlan.trades`): each closed trade of the run gives one random trade
with its symbol, side, quantity in units, product and holding time in strategy bars
(`bars_held`; a trade held less than one bar is held one, since a decided exit fills at
the next open at the earliest), in the order the run entered them. The random runs
therefore hold the run's directions and holding times trade for trade; where the trades
go is drawn, and so is which of the run's trades of a direction takes each place.

The order: each symbol's places keep the run's sequence of directions, and the trades of
each direction are put in a random order among the places of that direction. Keeping the
sequence keeps the waits it needs (below), so trades that fitted the run fit here.

Where they go: each symbol's trades are placed at random places among its strategy bars
of the run that do not overlap. A trade decided at the close of bar i fills at the next
open and is exited by a decision at the close of bar i + h. The next trade may be
decided at that same close when it goes the other way (the exit and the opposite entry
fill together, a reversal); one in the same direction waits one bar, until the position
is flat again, because an entry decided while the position is still open would add to
it, and the engine refuses an add below the average price (averaging down). With U bars
whose decision can fill (every strategy bar of the run but the last), holding times h_1
to h_n and w neighbours of one direction, the slack S = U - 1 - (h_1 + ... + h_n) - w is
cut at n points drawn uniformly from 0 to S and sorted, which leaves the n + 1 gaps
before, between and after the trades. Trades that do not fit (S below 0) cannot be
placed: the check leaves random entries out and says so, and `schedule` refuses them.

The random strategy is the run's own strategy class with its decisions replaced: it
keeps the class's settings (product, sizing caps, risk rules, entry window, square-off),
so the engine applies the same rules to its trades, which may refuse an entry (outside
the entry window, for margin) or close a trade early (the square-off, a risk rule) as it
would the strategy's; the table of the check shows the random runs' trade counts.

Draws: run r reads 2 draws a trade, symbols in registration order (draws r x 2 x trades
onwards): the order of each symbol's trades within their direction (one key each), then
its cut points.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .._arrays import MADE_BY
from ..errors import ConfigError
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
    """What random entries are drawn from: the run's trades in the order it entered
    them, and the UTC nanosecond close of each strategy bar of the run, by symbol."""

    trades: tuple[Trade, ...]
    closes: Mapping[str, np.ndarray]

    def of(self, symbol: str) -> list[Trade]:
        """The trades of `symbol`, in the order the run entered them."""
        return [trade for trade in self.trades if trade.symbol == symbol]

    def slack(self, symbol: str) -> int:
        """The bars `symbol`'s trades leave free (module docstring); below 0 they do not
        fit."""
        mine = self.of(symbol)
        holds = sum(trade.hold for trade in mine)
        return len(self.closes[symbol]) - 2 - holds - _waits(mine)

    def unplaced(self) -> list[str]:
        """The symbols whose trades do not fit among their strategy bars."""
        symbols = dict.fromkeys(trade.symbol for trade in self.trades)
        return [symbol for symbol in symbols if self.slack(symbol) < 0]


def _waits(trades: Sequence[Trade]) -> int:
    """The bars a sequence of trades waits between neighbours: one for each trade that
    goes the same way as the one before it (module docstring)."""
    pairs = zip(trades[:-1], trades[1:], strict=True)
    return sum(after.side == before.side for before, after in pairs)


def entry_plan(trade_list: Any, closes: Mapping[str, np.ndarray]) -> EntryPlan:
    """The `EntryPlan` of a run's `trade_list` (`Result.trade_list()`, put in the order
    of entry) and its strategy bars' closes by symbol."""
    entered = trade_list.sort_values("entry_time", kind="stable")
    trades = tuple(
        Trade(
            str(row.symbol),
            "BUY" if row.direction == "long" else "SELL",
            float(row.quantity),
            str(row.product),
            max(LEAST_HOLD, int(row.bars_held)),
        )
        for row in entered.itertuples()
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


def _drawn(trades: Sequence[Trade], keys: np.ndarray) -> list[Trade]:
    """`trades` with those of each direction put in the order of their `keys` among the
    places of that direction (module docstring)."""
    sides = np.array([trade.side for trade in trades])
    source = np.arange(len(trades))
    for side in dict.fromkeys(sides):
        places = np.flatnonzero(sides == side)
        source[places] = places[np.argsort(keys[places], kind="stable")]
    return [trades[position] for position in source]


def schedule(plan: EntryPlan, seed: int, run: int) -> dict[int, tuple[Action, ...]]:
    """Run `run` of the stream `seed`: the decisions of the random strategy by the UTC
    nanosecond close they are made at (module docstring), exits before entries. Raises
    ConfigError for a plan whose trades do not fit (`EntryPlan.unplaced`)."""
    unplaced = plan.unplaced()
    if unplaced:
        symbol = unplaced[0]
        raise ConfigError(
            f"Random entries cannot be placed: the trades of {symbol} do not fit among its "
            f"{len(plan.closes[symbol])} strategy bars of the run ({-plan.slack(symbol)} "
            "short); leave out the random entries of a plan whose unplaced() names a symbol."
        )
    draws = _random.block(seed, run, 1, 2 * len(plan.trades))[0]
    exits: dict[int, list[Action]] = {}
    entries: dict[int, list[Action]] = {}
    used = 0
    for symbol in dict.fromkeys(trade.symbol for trade in plan.trades):
        mine = plan.of(symbol)
        count, slack, closes = len(mine), plan.slack(symbol), plan.closes[symbol]
        keys, cuts = draws[used : used + count], draws[used + count : used + 2 * count]
        used += 2 * count
        placed = _drawn(mine, keys)
        gaps = np.diff(np.concatenate(([0], np.sort(_random.below(cuts, slack + 1)))))
        bar = 0
        for k, (gap, trade) in enumerate(zip(gaps, placed, strict=True)):
            bar += int(gap)
            entry = Action(symbol, trade.side, trade.units, trade.product)
            entries.setdefault(int(closes[bar]), []).append(entry)
            bar += trade.hold
            exits.setdefault(int(closes[bar]), []).append(Action(symbol))
            bar += _waits(placed[k : k + 2])
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
