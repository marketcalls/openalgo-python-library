"""The session plan of a live session (16-session-design 1.5, 3.1 steps 1 and 11, 3.6;
unit 7): each strategy's settings, its warm-up from broker history, its kernel session
and its catch-up to the present.

Two calls, at two steps of the start:

- `specify` (3.1 step 1) checks the arguments of `bt.live` (`_plan_args`) and each
  strategy (`_plan_strategies`) before anything is called: what 2.3.0 does not run live
  is refused saying what to do, each strategy's settings are resolved as `bt.backtest`
  resolves them for today, and `init` runs on the engine's reference bars to learn the
  history its indicators need. The `SessionSpec` it returns names what the preflight
  checks (instruments, intervals, the capital and the daily loss limits).
- `build` (3.1 step 11) builds each strategy's setup as a backtest of today builds its
  own, but for the live terms (`_plan_setup`: OpenAlgo's instrument facts, an external
  venue, the session's whole-day grid with no bars yet), its kernel session
  (`Engine.session`, its grid checked against the bar builders', `_plan_engine`), its
  live series holding the history, the strategy instance and its live run context, and
  runs `init` (the history checked right after it, `self.state` saved by an earlier
  start merged in). Then the catch-up.

**The catch-up** brings each kernel session and live series from the open to the
present, slice by slice, from reads of today's broker candles. A read releases the
slices settled at it (closed `CANDLE_SETTLE_S` or more before it began), oldest first:
each with its candle, as a minute without a trade when the read holds a candle of a
later slice, or marked missing (`LiveSeries.mark_missing`: no decision on a bar holding
it, 2.5 C) when the read began `CANDLE_LAST_READ_S` or more after its close; any other
slice waits, with the later ones, for a later read, since the broker's latest candle
often lags its close (`_plan_release`). A read with no candle of today thus marks the
slices it gives up on missing. Each marking is a warning, and a candle read later for a
slice marked missing is taken by its series (the kernel keeps the slice it ran, as with
the session's candle checks, 2.5 A step 10). The pauses reveal the bars and deliver the
kernel's events, but decide nothing: no `on_bar`, no timer (08-live-trading section 5
rule 7: missed bars refresh indicators only), and a catch-up that made an order raises
rather than send it.

`build` catches each strategy up to its setup's read, then reads it once more for the
slices that settled meanwhile. `SessionPlan.catch_up` does the same later, between the
build and the start of watching the feed (3.1 steps 11 to 14), reading each strategy at
most once, so a call ends however slow the reads; a strategy's `catch_up` takes candles
read elsewhere (the data worker's). What is left after the last call before watching is
for the session's candle checks: each strategy's `missing` slices, and those still
`uncaught` when watching begins, which the catch-up releases from those checks' candles
(`before` the slice watching began in); from that slice on, the session loop releases
each slice as it closes (`StrategyPlan.release_next`, in `_plan_catch_up`, the module
of a strategy's part of the plan). A restart builds the plan again from the same
arguments: a fresh engine is caught up from the open, and the kernel's saved state is
restored after the catch-up (3.6, K4).

Times are UTC nanoseconds, prices rupees, quantities units (never lots). Not
thread-safe: the main thread builds the plan, and the session loop owns it afterwards.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd

from .._calendar_convert import interval_nanos
from .._products import CNC, OVERNIGHT_PRODUCTS
from ..calendar import Calendar, default_calendar
from ..strategy import Strategy
from ._bars_candle import candle_bar
from ._plan_args import LiveArguments, live_arguments
from ._plan_catch_up import INSTRUMENT, StrategyPlan
from ._plan_engine import bind_strategy, check_grid, session_engine
from ._plan_release import CANDLE_LAST_READ_S, CANDLE_SETTLE_S, missing_note
from ._plan_setup import StrategySetup, read_today, strategy_setup
from ._plan_strategies import StrategySpec, ist_day, refuse_shared_positional, strategy_spec
from ._preflight_plan import InstrumentFacts
from ._types import LiveBar, Now
from .series import DEFAULT_BASE, LiveSeries


@dataclass(frozen=True)
class SessionSpec:
    """A live session as `specify` resolved it: the checked `arguments`, the IST trading
    `day`, and each strategy's spec in call order (`_plan_strategies.StrategySpec`)."""

    arguments: LiveArguments
    day: dt.date
    strategies: tuple[StrategySpec, ...]

    @property
    def instruments(self) -> tuple[tuple[str, str], ...]:
        """The (symbol, exchange) the strategies trade, each once, in call order, as the
        preflight takes them (several strategies may trade one: 18-multi-strategy 4.1)."""
        return tuple(dict.fromkeys((spec.symbol, spec.exchange) for spec in self.strategies))

    @property
    def intervals(self) -> tuple[str, ...]:
        """Every strategy's bar interval, each once, in call order."""
        return tuple(dict.fromkeys(spec.interval for spec in self.strategies))

    @property
    def product(self) -> str:
        """The product the strategies trade, as the preflight checks it (`product_check`):
        their one product, or each named once in call order and joined with commas, which
        the check refuses."""
        return ", ".join(dict.fromkeys(spec.product for spec in self.strategies))

    @property
    def reads_holdings(self) -> bool:
        """True when a strategy trades CNC: from the day after a buy its units are in the
        holdings, so the order worker reads them with every book read (`_broker_units`)."""
        return any(spec.product == CNC for spec in self.strategies)

    @property
    def gtt_legs(self) -> tuple[tuple[str, str, str], ...]:
        """The (symbol, exchange, product) of the CNC and NRML strategies' legs, in call
        order: the legs whose stop is a GTT (20-positional-live 2)."""
        return tuple(
            (spec.symbol, spec.exchange, spec.product)
            for spec in self.strategies
            if spec.product in OVERNIGHT_PRODUCTS
        )

    @property
    def positional(self) -> tuple[tuple[str, str, str, str], ...]:
        """The (name, symbol, exchange, product) of the CNC and NRML strategies, in call
        order: those a stop GTT refusal kept from an earlier day refuses at start
        (`_preflight_carry.refuse_kept`)."""
        return tuple(
            (spec.name, spec.symbol, spec.exchange, spec.product)
            for spec in self.strategies
            if spec.product in OVERNIGHT_PRODUCTS
        )

    @property
    def capital(self) -> float:
        """The strategies' capital together, rupees."""
        return sum(spec.capital for spec in self.strategies)

    @property
    def max_daily_loss(self) -> float:
        """The strategies' daily loss limits together, rupees."""
        return sum(spec.max_daily_loss for spec in self.strategies)

    def account_name(self, broker: str) -> str:
        """The account name the session keeps its journal, logs and locks under: the
        `account=` given, else the broker name `ping` reported (`broker`)."""
        return self.arguments.account_name(broker)


def specify(
    strategy: type[Strategy] | Sequence[type[Strategy]], *, clock: Now, **arguments: Any
) -> SessionSpec:
    """Check a `bt.live` call (its strategy and keyword `arguments`) and resolve each
    strategy for the IST day of the present `clock` gives (module docstring).

    Raises ConfigError for unusable arguments and NotImplementedError for what a live
    session does not run (`_plan_args.live_arguments`, `_plan_strategies.strategy_spec`),
    and what a strategy's `init` raises on the engine's reference bars."""
    checked = live_arguments(strategy, **arguments)
    present_ns = clock()
    day = ist_day(present_ns)
    calendar = default_calendar()
    specs = tuple(
        strategy_spec(cls, name, checked.backtest_arguments(name, day, present_ns), calendar)
        for cls, name in zip(checked.strategies, checked.names, strict=True)
    )
    refuse_shared_positional(specs)
    return SessionSpec(checked, day, specs)


class SessionPlan:
    """The plan of one session's IST `day` (module docstring): each strategy's
    `StrategyPlan` in call order, and the warnings found while building it."""

    def __init__(
        self, day: dt.date, strategies: tuple[StrategyPlan, ...], warnings: Sequence[str]
    ) -> None:
        self.day = day
        self.strategies = strategies
        self._warnings = list(warnings)

    @property
    def warnings(self) -> tuple[str, ...]:
        """The setups' warnings, then those of the reads of today's candles (the data
        layer's, and each marking of slices missing), in order."""
        return tuple(self._warnings)

    def catch_up(self, client: Any, clock: Now) -> None:
        """Catch every strategy up to the present (module docstring): each one with
        slices settled and not caught up at a fresh read of `clock`, or slices marked
        missing, has today's candles read once through `client` (the session's data
        client, on the thread that owns it) and is caught up to the read. Slices that
        settle during the reads, or wait for a later read, are left to the next call.
        Raises the data layer's errors, and what `StrategyPlan.catch_up` raises."""
        for plan in self.strategies:
            read_ns = clock()
            if plan.uncaught(read_ns) or plan.missing:
                frame, notes = read_today(plan.spec, client, plan.calendar, read_ns)
                self._add(notes)
                self._add(_caught_up(plan, frame, read_ns))

    def _add(self, notes: Iterable[str]) -> None:
        self._warnings += [note for note in notes if note not in self._warnings]


def candles(frame: pd.DataFrame, spec: StrategySpec, calendar: Calendar) -> list[LiveBar]:
    """Today's 1-minute bars of a strategy's symbol (the bars layout) as broker candles
    (`bars.candle_bar`)."""
    interval_ns = interval_nanos(DEFAULT_BASE)
    return [
        candle_bar(spec.symbol, spec.exchange, row, interval_ns, calendar=calendar)
        for row in frame.to_dict("records")
    ]


def _caught_up(plan: StrategyPlan, frame: pd.DataFrame, read_ns: int) -> tuple[str, ...]:
    """Catch `plan` up to a read of today's candles begun at `read_ns` (UTC nanoseconds)
    that gave `frame` (the bars layout, empty for a read with no candle of today), and
    return its warnings: one naming the slices it marked missing, if any. Raises what
    `StrategyPlan.catch_up` raises."""
    marked = plan.catch_up(candles(frame, plan.spec, plan.calendar), read_ns)
    if not marked:
        return ()
    spec = plan.spec
    return (missing_note(spec.name, spec.symbol, spec.exchange, marked, none_today=frame.empty),)


def _strategy_plan(
    spec: StrategySpec, setup: StrategySetup, calendar: Calendar, saved: Any
) -> StrategyPlan:
    engine = session_engine(setup, calendar)
    kept = setup.history.symbols.get(spec.symbol)
    history = None if kept is None else kept.base
    series = LiveSeries(
        spec.symbol, spec.exchange, spec.plan.start, history=history, calendar=calendar
    )
    check_grid(setup.config, spec.exchange, spec.plan.start, calendar, series)
    strategy, context = bind_strategy(spec, setup, engine, series, calendar, saved)
    return StrategyPlan(spec, setup, engine, series, strategy, context, calendar)


def build(
    spec: SessionSpec,
    client: Any,
    *,
    facts: Sequence[InstrumentFacts],
    clock: Now,
    user_states: Mapping[str, Mapping[str, Any]] | None = None,
) -> SessionPlan:
    """The plan of the session `spec`, read through `client` (the session's data
    client), with the preflight's instrument `facts` (its report's), at the present
    `clock` gives, caught up to it (module docstring). `user_states` are the strategies'
    `self.state` saved by an earlier start of the day, by strategy name.

    Raises ConfigError for facts that do not cover an instrument, a history too short for
    an indicator and unusable settings; LiveError when the kernel's grid is not the bar
    builders' or the catch-up made an order or met the session's end; ServerError when
    the OpenAlgo server cannot answer history; the data layer's and the kernel's errors;
    and what a strategy's `init` raises."""
    present_ns = clock()
    calendar = default_calendar()
    saved = user_states or {}
    plans, warnings = [], []
    for each in spec.strategies:
        setup = strategy_setup(each, client, facts, present_ns, calendar)
        plans.append(_strategy_plan(each, setup, calendar, saved.get(each.name)))
        warnings += setup.warnings
    for plan in plans:
        warnings += _caught_up(plan, plan.setup.today, plan.setup.read_ns)
    session = SessionPlan(spec.day, tuple(plans), tuple(dict.fromkeys(warnings)))
    session.catch_up(client, clock)
    return session


__all__ = [
    "CANDLE_LAST_READ_S",
    "CANDLE_SETTLE_S",
    "INSTRUMENT",
    "SessionPlan",
    "SessionSpec",
    "StrategyPlan",
    "build",
    "candles",
    "specify",
]
