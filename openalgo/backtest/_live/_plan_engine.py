"""The kernel session of one strategy, its grid check, and the strategy bound to it
(16-session-design 2.10 K1 and 3.1 step 11; unit 7). Part of `plan`.

- `session_engine` builds the strategy's kernel session (`Engine.session`) from its setup:
  the config a backtest of the day runs with, its instruments with OpenAlgo's facts, and
  no bars (the catch-up adds today's). Its venue is external: a broker fills its orders.
- `check_grid` checks that the session's slice grid is the grid the bar builders
  (`bars.TickBarBuilder`, from `calendar.union_grid([exchange], day, day, base="60s")`)
  and the live series use. The kernel builds its grid from the calendar's union grid
  with the config's days and base interval, so a config of another day or base would
  put the kernel's slices and the session's apart.
- `bind_strategy` makes the strategy instance and its live run context, runs `init`,
  refuses a history too short for an indicator right after it (`LiveRunContext.
  check_history`), refuses conditions, and merges `self.state` saved by an earlier start
  of the day, the saved keys winning over what `init` set (3.6).

Times are UTC nanoseconds.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from .._calendar_convert import DateLike, from_day, native_attr
from .._causality_checks import IndicatorChecks
from .._context import RunData
from .._engine import bar_arrays, call, instrument_dict
from .._layout import NANOS_PER_SECOND
from .._manifest import strategy_source_hash
from .._positional_exit import positional_exits
from .._warm_up_history import empty_base
from ..calendar import NO_BAR, Calendar
from ..errors import ConfigError, LiveError
from ..strategy import Strategy
from ._plan_setup import StrategySetup
from ._plan_strategies import StrategySpec, refuse_conditions
from .bars import TickBarBuilder
from .series import LiveRunContext, LiveSeries

Slices = tuple[tuple[int, int], ...]


def grid_slices(
    calendar: Calendar, exchange: str, first: DateLike, last: DateLike, base_s: int
) -> Slices:
    """(open, close) of the exchange's base slices of `base_s` seconds from the IST date
    `first` to `last`, in order (UTC nanoseconds): the rows of the calendar's union grid
    with a bar of the exchange, the grid the kernel builds from the same calendar."""
    grid: pd.DataFrame = calendar.union_grid([exchange], first, last, base=f"{base_s}s")
    closes = grid[f"close_{grid.attrs['exchanges'][0]}"].to_numpy()
    has_bar = closes != NO_BAR
    opens = grid["ts"].to_numpy()[has_bar].tolist()
    return tuple(zip(opens, closes[has_bar].tolist(), strict=True))


def check_grid(
    config: Mapping[str, Any], exchange: str, day: DateLike, calendar: Calendar, series: LiveSeries
) -> None:
    """Raise LiveError unless the slices of a kernel session built with `config` (its
    first and last day and base interval) on `exchange` are the slices the bar builders
    use on the IST date `day` (module docstring), and the live series' are the same."""
    base_s = int(config["base_nanos"]) // NANOS_PER_SECOND
    first, last = from_day(config["first_day"]), from_day(config["last_day"])
    kernel = grid_slices(calendar, exchange, first, last, base_s)
    builders = TickBarBuilder(exchange, day, watching_from_ns=0, calendar=calendar).slices
    if kernel != builders:
        raise LiveError(
            f"the kernel session's {exchange} grid ({len(kernel)} slices of {base_s} s from "
            f"{first} to {last}) is not the grid the session's bars are built on ({len(builders)} "
            f"1-minute slices of {day}): a session is planned for one trading day on a "
            "1-minute base"
        )
    if series.slices != builders:
        raise LiveError(
            f"the live series of {exchange}:{series.symbol} holds other slices than the bars "
            "are built on: build it for the session's day on a 1-minute base"
        )


def session_engine(setup: StrategySetup, calendar: Calendar) -> Any:
    """The kernel session of one strategy (`Engine.session`), with no bars yet, each
    instrument with its positional exit day as `bt.backtest` gives it (`_positional_exit`:
    a stock future, an option or an MCX delivery contract is exited at the square-off of
    that day, `expiry_exit`; 20-positional-live 9). Raises ConfigError for a config or
    instrument the kernel refuses, BacktestError for a kernel defect, NotImplementedError
    for what the kernel does not run."""
    exits = positional_exits(setup.instruments, calendar).days
    pairs = zip(setup.instruments, exits, strict=True)
    specs = [instrument_dict(instrument, day) for instrument, day in pairs]
    arrays = [bar_arrays(empty_base()) for _ in setup.instruments]
    return call(native_attr("Engine").session, calendar._cal, setup.config, specs, arrays)


def bind_strategy(
    spec: StrategySpec,
    setup: StrategySetup,
    engine: Any,
    series: LiveSeries,
    calendar: Calendar,
    saved_state: Mapping[str, Any] | None,
) -> tuple[Strategy, LiveRunContext]:
    """(the strategy instance, its live run context over `engine` and `series`), `init`
    run (module docstring). Raises ConfigError for a history too short for an indicator
    `init` registered and for a saved state that is not a dict, NotImplementedError for
    conditions, and what `init` raises."""
    strategy = spec.strategy
    plan = setup.plan
    checks = IndicatorChecks(
        strategy_source_hash(strategy) or strategy.__qualname__, measured=spec.measured
    )
    run = RunData(
        plan.exchange, plan.interval, plan.base, {}, calendar, checks.for_run(), setup.history
    )
    context = LiveRunContext(
        engine, list(setup.instruments), plan.product, strategy.size, run, {spec.symbol: series}
    )
    instance = strategy(**(spec.params or {}))
    instance._ctx = context
    instance.init()
    context.check_history()
    refuse_conditions(instance)
    if saved_state is not None:
        if not isinstance(saved_state, Mapping):
            raise ConfigError(
                f"{spec.name}: the saved state is the strategy's self.state, a dict, not "
                f"{type(saved_state).__name__}"
            )
        instance.state.update(saved_state)
    return instance, context


__all__ = ["Slices", "bind_strategy", "check_grid", "grid_slices", "session_engine"]
