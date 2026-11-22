"""The price a backtest of the session's own bars fills each of its orders at (16-session-design
1.4; decided item 11 of 10.0; 18-multi-strategy 12.3): what the day summary's slippage and
the fill comparison (`_fill_comparison`) set each live fill against. Each price is the
kernel's own rule for the bar (`Engine.market_fill_price`: a reference price worse by the
slippage the run models for the instrument on the bar's trade date, rounded to the tick
away from the trader), so the rule has one source.

- **A MARKET order** (`market_price`): the open of the first base bar at or after the
  order's decision, as a backtest fills a MARKET order (SPEC-35: a minute with no trade has
  no bar, so it is the next one). The decision's instant depends on what made the order.
  At a bar close, a timer or an open the session reaches (the strategy's orders, the
  square-off, the time exit) it is the kernel's `decided_at`, the instant a backtest
  decides the same order at: a live session runs a slice once its bar has closed, so it
  sends such an order a moment after, and an order that waited for a stop's cancellation
  later still. On an event while the kernel waits inside a minute (every risk exit but the
  time exit: the daily loss limit or the drawdown halt on a tick, an unprotected or
  rejected stop, the kill switch, the account's limits, Ctrl+C, an exit asked for again)
  the kernel stamps the order with the open of the slice in progress, which opened before
  the event, so the instant is the one the session placed it at (its intent's
  `created_ns`), and the bar is the next minute's.
- **The engine's exit at a leg's target** (`target_price`), made on the tick that reached
  the target: at the target's level in the bar that tick came in, or at that bar's open
  when it opened at or beyond the level, as the kernel fills a leg's level. It is told
  from a strategy's exit by the exit reason the kernel records (`LEG_TARGET`), which a
  strategy's exit cannot take (`_strategy_orders.ENGINE_LEG_EXIT_REASONS`); its level is
  learned from the kernel's live state when the exit is placed (`TargetLevels`).
- **A stop** (`stop_price`), an SL-M (cash equity, futures) or the stop-limit of an
  option: in the first bar whose low (a sell's stop) or high (a buy's) reaches its
  trigger, at the trigger, or at that bar's open when it opened at or through the
  trigger, as the kernel fills a stop on the same bars. The bars looked at run from the
  one in which the broker began holding the trigger (`StopTriggers`: the stop's placing,
  or the modify that last moved it) to the one in which its fill was heard, which may be
  later than the trade that filled it (a fill heard through a poll, or a lagging
  stream). A stop filled where no bar reached its trigger is not priced. An option's
  stop-limit is held at its limit, and one whose bar opened beyond its limit is not
  priced (the kernel's limit then rests or exits at a later open).

The bars are the session's own (the series'). For a released slice that is the bar the
kernel ran (its tick bar, or the broker's candle of a minute the catch-up released), else
the broker's candle read for it since. For a slice not released yet it is its trades so far
(`TickBarBuilder.open_so_far`): only its open is known then, so a stop not reached in a
released bar is set in it. A released slice with no bar had no trade and is passed over,
as a backtest passes a missing bar, unless it is still marked missing (no bar of it is
known: the feed was dead, or a restart began in it, and no candle has come), so it may
hold the trade the broker filled at. A price the session cannot give is None, with why
(`Unpriced`).

Times are UTC nanoseconds, prices rupees, quantities units.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .._engine import call
from .._strategy_orders import LEG_TARGET
from ._plan_catch_up import INSTRUMENT
from ._types import OrderIntent, Purpose

#: The sign of a fill's slippage by its side: a buy that paid more is worse.
AGAINST_THE_TRADER = {"BUY": 1, "SELL": -1}
#: The kernel's exit reason (an `ExitReason` code) of the time exit, the one risk exit
#: made at an open; a leg's target exit, made on a tick, records `LEG_TARGET`.
_TIME_EXIT = "time_exit"
#: Why a price is missing (`Unpriced.reason`).
NO_BAR = "no bar of its minute"
NOT_KNOWN = "an order its kernel session does not know"
NO_LEVEL = "its target's level was not kept (placed before a restart)"
BEYOND_LIMIT = "its bar opened beyond the stop-limit's limit"
NOT_REACHED = "no bar from its trigger's placing to its fill reached the trigger"


@dataclass(frozen=True)
class Slippage:
    """A strategy's slippage (`_fill_comparison.slippage_of`): its MARKET fills' `rupees`
    against the backtest's (positive when live did worse), and how many of them were not
    compared and compared; its stops' fills likewise."""

    rupees: float = 0.0
    not_compared: int = 0
    compared: int = 0
    stop_rupees: float = 0.0
    stop_not_compared: int = 0
    stop_compared: int = 0


@dataclass(frozen=True)
class Priced:
    """A backtest's fill price of an order (rupees) in the base bar opening at `bar_ns`."""

    bar_ns: int
    price: float


@dataclass(frozen=True)
class Unpriced:
    """No backtest price for an order, and why (`NO_BAR` and the others)."""

    reason: str


class TargetLevels:
    """The level of each target exit a strategy's kernel session placed, rupees, by
    strategy and kernel order: learned from its live state whenever the session saves it
    (`learn`), which it does before handing the exit to the order worker (design rule 4),
    while the leg still holds the target."""

    def __init__(self) -> None:
        self._levels: dict[tuple[str, int], float] = {}

    def learn(self, strategy: str, live_state: Mapping[str, Any]) -> None:
        """Keep the level of each target exit `live_state` (`Engine.live_state()`) shows
        working: its desk's `target_exits` name the order and the target, its legs' the
        target's level."""
        exits = live_state.get("desk", {}).get("target_exits", [])
        if not exits:
            return
        levels = {}
        for leg in live_state.get("legs", []):
            if leg.get("instrument") == INSTRUMENT:
                levels |= {t["index"]: t["level_rupees"] for t in leg.get("targets", [])}
        for target_exit in exits:
            level = levels.get(target_exit.get("target"))
            if isinstance(target_exit.get("order"), int) and level is not None:
                self._levels[(strategy, target_exit["order"])] = float(level)

    def level(self, strategy: str, kernel_order: int) -> float | None:
        """The level of the target exit `kernel_order` of `strategy`, None when unknown."""
        return self._levels.get((strategy, kernel_order))


class StopTriggers:
    """When the broker began holding each order's trigger, by strategy and kernel order:
    the trigger in rupees and the instant (UTC nanoseconds) of each modify the venue map
    applied (`modified`, after the broker accepted it), so a stop's fill is looked for
    from the bar its present trigger has held since (`held_since`)."""

    def __init__(self) -> None:
        self._modified: dict[tuple[str, int], list[tuple[float, int]]] = {}

    def modified(self, strategy: str, kernel_order: int, trigger: float, at_ns: int) -> None:
        """Note that `strategy`'s order `kernel_order` holds `trigger` from `at_ns`."""
        self._modified.setdefault((strategy, kernel_order), []).append((trigger, at_ns))

    def held_since(
        self, strategy: str, kernel_order: int, intent: OrderIntent, trigger: float, start_ns: int
    ) -> int:
        """The instant from which order `kernel_order` of `strategy`, placed as `intent`,
        has held `trigger`, its present one: its placing, or the modify that moved it
        there (a later modify of its units alone keeps it). A trigger moved where this
        start did not see it (before a restart, or by a modify its order's news confirmed)
        is held from the last move this start knows of, at the earliest from its
        beginning (`start_ns`)."""
        held = [(intent.trigger_price, intent.created_ns)]
        held += self._modified.get((strategy, kernel_order), [])
        if held[-1][0] != trigger:
            return max(held[-1][1], start_ns)
        since = held[-1][1]
        for kept, at_ns in reversed(held):
            if kept != trigger:
                break
            since = at_ns
        return since


def is_target_exit(trader: Any, kernel_order: int, intent: OrderIntent) -> bool:
    """True for the engine's exit at a leg's target (module docstring)."""
    if intent.purpose is not Purpose.EXIT:
        return False
    return call(trader.engine.order_exit_reason, kernel_order) == LEG_TARGET


def market_price(
    session: Any, trader: Any, kernel_order: int, intent: OrderIntent
) -> Priced | Unpriced:
    """A backtest's fill of MARKET order `kernel_order` of `trader`, placed as `intent`, at
    the open of the bar opening at its decision (module docstring)."""
    order = call(trader.engine.order, kernel_order)
    if order is None:
        return Unpriced(NOT_KNOWN)
    exit_reason = call(trader.engine.order_exit_reason, kernel_order)
    made_on_event = _made_on_an_event(intent.purpose, exit_reason)
    decided_at = intent.created_ns if made_on_event else order["decided_at"]
    bar = _bar_at_or_after(session, trader, decided_at)
    if bar is None:
        return Unpriced(NO_BAR)
    price = call(trader.engine.market_fill_price, INSTRUMENT, intent.action, *bar)
    return Priced(bar[0], price)


def target_price(
    session: Any, trader: Any, *, side: str, level: float, at_ns: int
) -> Priced | Unpriced:
    """A backtest's fill of the engine's exit of `side` at a leg's target `level` in the
    bar holding `at_ns`, the instant of the tick that reached it, or at its open when it
    opened at or beyond the level (module docstring)."""
    bar = _bar_holding(session, trader, at_ns)
    if bar is None:
        return Unpriced(NO_BAR)
    return _at_level(trader, side, level, *bar, target=True)


def stop_price(
    session: Any,
    trader: Any,
    *,
    side: str,
    trigger: float,
    since_ns: int,
    heard_ns: int,
    limit: float | None = None,
) -> Priced | Unpriced:
    """A backtest's fill of a stop of `side` at `trigger`, which the broker held from
    `since_ns` and whose fill the session heard at `heard_ns`, held at `limit` for an
    option's stop-limit: in the first bar from the one holding `since_ns` to the one
    holding `heard_ns` that reaches the trigger (module docstring)."""
    series, released = trader.part.series, trader.part.released
    # A sell's stop lies below the price and a bar's low reaches it; a buy's lies above.
    below = side == "SELL"
    for index, (opened, closed) in enumerate(series.slices):
        if closed <= min(since_ns, heard_ns):
            continue
        if opened > heard_ns:
            break
        if index >= released:
            so_far = _so_far(session, trader, opened)
            if so_far is None:
                return Unpriced(NO_BAR)
            return _at_level(trader, side, trigger, *so_far, target=False, limit=limit)
        bar = series.bar(opened)
        if bar is None:
            if series.missing_slices(opened, closed):
                return Unpriced(NO_BAR)
            continue
        if (bar.low <= trigger) if below else (bar.high >= trigger):
            return _at_level(trader, side, trigger, opened, bar.open, target=False, limit=limit)
    return Unpriced(NOT_REACHED)


def _at_level(
    trader: Any,
    side: str,
    level: float,
    opened: int,
    open_price: float,
    *,
    target: bool,
    limit: float | None = None,
) -> Priced | Unpriced:
    """The kernel's fill of a closing order of `side` at a leg's `level` (a `target`, else
    a stop's trigger) in the bar opening at `opened` at `open_price`: at the level, or at
    the open when it opened at or beyond it, with the run's slippage; held at `limit` for
    an option's stop-limit, and not priced when the bar opened beyond that limit."""
    if limit is not None and AGAINST_THE_TRADER[side] * (open_price - limit) > 0:
        return Unpriced(BEYOND_LIMIT)
    # A long's target and a short's stop lie above the price: an open at or above them
    # has reached them. A long's stop and a short's target lie below it.
    above = (side == "SELL") == target
    reached = open_price >= level if above else open_price <= level
    reference = open_price if reached else level
    price = call(trader.engine.market_fill_price, INSTRUMENT, side, opened, reference)
    if limit is not None:
        price = min(price, limit) if side == "BUY" else max(price, limit)
    return Priced(opened, price)


def _made_on_an_event(purpose: Purpose, exit_reason: str | None) -> bool:
    """True for an order the engine makes on an event while its kernel session waits
    inside a minute (module docstring): a risk exit but the time exit. (The engine's
    target exit, made on a tick too, is priced at its level, `target_price`.)"""
    return purpose is Purpose.RISK_EXIT and exit_reason != _TIME_EXIT


def _bar_at_or_after(session: Any, trader: Any, decided_at_ns: int) -> tuple[int, float] | None:
    """(open instant, open in rupees) of the first base bar of `trader`'s instrument that
    opens at or after `decided_at_ns` (module docstring), or None when the session cannot
    give it."""
    series, released = trader.part.series, trader.part.released
    for index, (opened, closed) in enumerate(series.slices):
        if opened < decided_at_ns:
            continue
        if index < released:
            bar = series.bar(opened)
            if bar is not None:
                return opened, bar.open
            if series.missing_slices(opened, closed):
                return None
            continue
        return _so_far(session, trader, opened)
    return None


def _bar_holding(session: Any, trader: Any, at_ns: int) -> tuple[int, float] | None:
    """(open instant, open in rupees) of the base bar of `trader`'s instrument whose slice
    holds `at_ns`, or None when the session cannot give it (module docstring)."""
    series, released = trader.part.series, trader.part.released
    for index, (opened, closed) in enumerate(series.slices):
        if not opened <= at_ns < closed:
            continue
        if index < released:
            bar = series.bar(opened)
            return None if bar is None else (opened, bar.open)
        return _so_far(session, trader, opened)
    return None


def _so_far(session: Any, trader: Any, opened: int) -> tuple[int, float] | None:
    """(open instant, open so far) of the slice opening at `opened`, not released yet."""
    builder = session.builders.get(trader.exchange)
    so_far = None if builder is None else builder.open_so_far(trader.symbol, opened)
    return None if so_far is None else (opened, so_far)


__all__ = [
    "AGAINST_THE_TRADER",
    "BEYOND_LIMIT",
    "NOT_KNOWN",
    "NOT_REACHED",
    "NO_BAR",
    "NO_LEVEL",
    "Priced",
    "Slippage",
    "StopTriggers",
    "TargetLevels",
    "Unpriced",
    "is_target_exit",
    "market_price",
    "stop_price",
    "target_price",
]
