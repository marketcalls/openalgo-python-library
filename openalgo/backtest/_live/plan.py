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
present: each slice of today whose broker candle was read at or after its close plus
`CANDLE_SETTLE_S` is given its candle (or none, for a minute without a trade), released,
and run. Its pauses reveal the bars and deliver the kernel's events, but decide nothing:
no `on_bar`, no timer (08-live-trading section 5 rule 7: missed bars refresh indicators
only), and a catch-up that made an order raises rather than send it. A slice that closes
while the plan is built is caught up too: `build` reads again while a slice completed
during the reads; `SessionPlan.catch_up` does the same later, for the slices that close
between the build and the start of watching the feed (3.1 steps 11 to 14), and a
strategy's `catch_up` takes candles read elsewhere (the data worker's). No slice is
released before a read of its candle was made, so none is skipped. A restart builds the
plan again from the same arguments: a fresh engine is caught up from the open, and the
kernel's saved state is restored after the catch-up (3.6, K4).

Times are UTC nanoseconds, prices rupees, quantities units (never lots). Not
thread-safe: the main thread builds the plan, and the session loop owns it afterwards.
"""

from __future__ import annotations

import bisect
import datetime as dt
import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd

from .._calendar_convert import format_ist, interval_nanos
from .._driver import deliver
from .._engine import call
from .._layout import NANOS_PER_SECOND
from .._warm_up import _ready_at
from ..calendar import Calendar, default_calendar
from ..errors import ConfigError, LiveError
from ..strategy import Strategy
from ._bars_candle import CANDLE_SOURCE, candle_bar
from ._plan_args import LiveArguments, live_arguments
from ._plan_engine import bind_strategy, check_grid, session_engine
from ._plan_setup import StrategySetup, read_today, strategy_setup
from ._plan_strategies import StrategySpec, ist_day, refuse_shared_legs, strategy_spec
from ._preflight_plan import InstrumentFacts
from ._types import LOGGER_NAME, LiveBar, Now
from .data_worker import CANDLE_READ_OFFSETS_S
from .series import DEFAULT_BASE, LiveRunContext, LiveSeries

#: Seconds after a slice's close from which a read of the broker's candles holds the
#: slice's candle: the first read of the candle checks (08-live-trading section 5 rule
#: 2), so the catch-up and the checks trust a candle from the same instant.
CANDLE_SETTLE_S = CANDLE_READ_OFFSETS_S[0]
#: Each strategy trades one instrument, registered first (1.1).
INSTRUMENT = 0

_SETTLE_NS = round(CANDLE_SETTLE_S * NANOS_PER_SECOND)

logger = logging.getLogger(LOGGER_NAME)


@dataclass(frozen=True)
class SessionSpec:
    """A live session as `specify` resolved it: the checked `arguments`, the IST trading
    `day`, and each strategy's spec in call order (`_plan_strategies.StrategySpec`)."""

    arguments: LiveArguments
    day: dt.date
    strategies: tuple[StrategySpec, ...]

    @property
    def instruments(self) -> tuple[tuple[str, str], ...]:
        """The (symbol, exchange) of each strategy, as the preflight takes them."""
        return tuple((spec.symbol, spec.exchange) for spec in self.strategies)

    @property
    def intervals(self) -> tuple[str, ...]:
        """Every strategy's bar interval, each once, in call order."""
        return tuple(dict.fromkeys(spec.interval for spec in self.strategies))

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
    session does not run (`_plan_args.live_arguments`, `_plan_strategies.strategy_spec`;
    two strategies on one symbol), and what a strategy's `init` raises on the engine's
    reference bars."""
    checked = live_arguments(strategy, **arguments)
    present_ns = clock()
    day = ist_day(present_ns)
    calendar = default_calendar()
    specs = tuple(
        strategy_spec(cls, name, checked.backtest_arguments(name, day, present_ns), calendar)
        for cls, name in zip(checked.strategies, checked.names, strict=True)
    )
    refuse_shared_legs(specs)
    return SessionSpec(checked, day, specs)


class StrategyPlan:
    """One strategy's part of the plan (module docstring): its `spec`, its `setup`, its
    kernel session (`engine`), its live `series`, the strategy instance and its live run
    `context`. `released` counts the slices of the day caught up and released to the
    kernel; the session loop releases the later ones."""

    def __init__(
        self,
        spec: StrategySpec,
        setup: StrategySetup,
        engine: Any,
        series: LiveSeries,
        strategy: Strategy,
        context: LiveRunContext,
        calendar: Calendar,
    ) -> None:
        self.spec, self.setup, self.engine = spec, setup, engine
        self.series, self.strategy, self.context = series, strategy, context
        self.calendar = calendar
        self._slices = series.slices
        self._closes = [close for _, close in self._slices]
        self._released = 0

    @property
    def released(self) -> int:
        """How many slices of the day, from the open, are caught up and released."""
        return self._released

    def uncaught(self, read_ns: int) -> tuple[int, ...]:
        """The opens of the slices a read of the broker's candles begun at `read_ns`
        (UTC nanoseconds) holds, those closed by `read_ns` less `CANDLE_SETTLE_S`, that
        are not caught up yet, oldest first."""
        end = bisect.bisect_right(self._closes, read_ns - _SETTLE_NS)
        return tuple(opened for opened, _ in self._slices[self._released : end])

    def catch_up(self, bars: Iterable[LiveBar], read_ns: int) -> None:
        """Catch up every slice a read of today's candles begun at `read_ns` (UTC
        nanoseconds; the start of the read, not its end, since a slice that settled
        during it may be missing from it) holds (`uncaught`): each in order is given its
        candle among `bars` (the strategy's broker candles of today, `source` "candle";
        a slice without one ran no trade), released and run, deciding nothing (module
        docstring). Candles of slices caught up already, or not held by the read, are
        left alone.

        Raises ConfigError for a bar that is not a candle of the strategy's instrument,
        and LiveError when the catch-up made an order or the exchange's session ended."""
        due = self.uncaught(read_ns)
        if not due:
            return
        wanted = set(due)
        by_open: dict[int, LiveBar] = {}
        for bar in bars:
            if bar.source != CANDLE_SOURCE:
                raise ConfigError(f"the catch-up takes broker candles, not {bar.source!r} bars")
            if bar.ts_open_ns in wanted:
                by_open[bar.ts_open_ns] = bar
        for opened in due:
            bar = by_open.get(opened)
            if bar is not None:
                self.series.take(bar)
                prices = (bar.open, bar.high, bar.low, bar.close, bar.volume)
                call(self.engine.add_bar, INSTRUMENT, (bar.ts_open_ns, *prices))
            self._released += 1
            call(self.engine.release, self._released)
            self._drive()
        self._refuse_orders()

    def indicators_ready(self, decided_at_ns: int) -> bool:
        """True when every indicator need of the strategy is met by the bars closed at
        the decision `decided_at_ns` (D25: the need-th bar of its series, history
        counted, or for an anchored need its whole periods), as the backtest's warm-up
        rule finds it (`_warm_up`): the session decides only then (2.5 C)."""
        return all(_ready_at(need, self.context) <= decided_at_ns for need in self.spec.needs)

    def _drive(self) -> None:
        """Run the kernel's pauses until it waits, revealing bars and delivering events,
        deciding nothing (no `on_bar`, no timer)."""
        while True:
            kind, decided_at, rows, events, _timers = call(self.engine.advance)
            if kind == "waiting":
                return
            self.context.reveal(decided_at, rows)
            if events:
                deliver(self.strategy, self.context, events)
            if kind == "finished":
                raise LiveError(
                    f"{self.spec.exchange}'s session of {self.spec.plan.start} ended before "
                    f"the plan of {self.spec.name} was built: nothing is left to trade today"
                )
            if kind == "timers":
                logger.info(
                    "%s: the timers due at %s were missed before the session started and are "
                    "skipped",
                    self.spec.name,
                    format_ist(decided_at),
                )

    def _refuse_orders(self) -> None:
        """Raise LiveError when the catch-up made an order: a catch-up sends nothing, and
        an order from a callback run on a past bar is not a decision of now."""
        commands = call(self.engine.take_commands)
        if commands:
            raise LiveError(
                f"{self.spec.name}: {len(commands)} orders were made while the session caught "
                "up on today's past bars, and a catch-up sends nothing: place orders in "
                "on_bar, which runs once the session trades"
            )


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
        """The setups' warnings, then any a later read of today's candles gave."""
        return tuple(self._warnings)

    def catch_up(self, client: Any, clock: Now) -> None:
        """Catch every strategy up to the present: read today's candles of each strategy
        with uncaught slices through `client` (the session's data client, on the thread
        that owns it) and catch up to the read, again while a slice completed during the
        reads (module docstring). Raises the data layer's errors, and what `catch_up`
        raises."""
        while True:
            behind = [plan for plan in self.strategies if plan.uncaught(clock())]
            if not behind:
                return
            for plan in behind:
                read_ns = clock()
                frame, notes = read_today(plan.spec, client, plan.calendar, read_ns)
                self._warnings += [note for note in notes if note not in self._warnings]
                plan.catch_up(candles(frame, plan.spec, plan.calendar), read_ns)


def candles(frame: pd.DataFrame, spec: StrategySpec, calendar: Calendar) -> list[LiveBar]:
    """Today's 1-minute bars of a strategy's symbol (the bars layout) as broker candles
    (`bars.candle_bar`)."""
    interval_ns = interval_nanos(DEFAULT_BASE)
    return [
        candle_bar(spec.symbol, spec.exchange, row, interval_ns, calendar=calendar)
        for row in frame.to_dict("records")
    ]


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
        warnings += [note for note in setup.warnings if note not in warnings]
    session = SessionPlan(spec.day, tuple(plans), warnings)
    for plan in plans:
        plan.catch_up(candles(plan.setup.today, plan.spec, calendar), plan.setup.read_ns)
    session.catch_up(client, clock)
    return session


__all__ = [
    "CANDLE_SETTLE_S",
    "SessionPlan",
    "SessionSpec",
    "StrategyPlan",
    "build",
    "candles",
    "specify",
]
