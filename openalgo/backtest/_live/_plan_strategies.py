"""What a live session runs of each strategy class, checked before anything is called
(16-session-design 1.1 and 3.1 step 1; decision 5 of its section 10.0; unit 7). Part of
`plan`.

`strategy_spec` refuses, with NotImplementedError saying what to do, what 2.3.0 does not
run live: `bt.from_signals` and `bt.from_arrays` strategies, strategies on several
symbols or a universe, futures series (an `underlying` with an `expiry`), daily or longer
intervals, CNC and NRML (they need GTT orders live), and condition strategies
(`entry_when`, `exit_when`, `short_when`, `cover_when`), whose arrays are computed ahead
over the whole series (decision 5): their rules go in `on_bar`. It then resolves the
strategy's settings for today as `bt.backtest` resolves them (`_settings.resolve`: a
class attribute wins over the run argument of the same name, each `max_daily_loss` in
rupees) and learns what history its indicators need by running `init` on the engine's
reference bars, as a backtest does before it loads data (`_warm_up_probe`): that run is
also the strategy layer's causality and window-stability check, and the one that finds
a condition strategy, since conditions are declared in `init`.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .._arrays import made_by
from .._causality_checks import IndicatorChecks
from .._conditions import has_conditions
from .._instruments import resolve_instruments
from .._layout import parse_interval
from .._manifest import strategy_source_hash
from .._margin_check import check_margin
from .._messages import not_supported
from .._series import ist_time
from .._settings import RunPlan, resolve
from .._timeframes import _check_interval
from .._warm_up import SeriesNeed
from .._warm_up_measure import WarmUp
from .._warm_up_probe import ProbeContext, history_needs, probe_setup
from ..calendar import Calendar
from ..errors import ConfigError
from ..reference import Instrument
from ..strategy import Strategy
from .series import DEFAULT_BASE

#: The product of every live order in this release (16-session-design 1.1, Tier 1).
LIVE_PRODUCT = "MIS"
#: What a condition strategy declares in `init`, as messages name it.
CONDITIONS = "entry_when, exit_when, short_when, cover_when"


@dataclass(frozen=True)
class StrategySpec:
    """One strategy of a live session as resolved at the start (module docstring).

    `strategy` is the class and `name` its class name; `symbol` and `exchange` what it
    trades, `interval` its bar interval. `capital` and `max_daily_loss` are rupees (the
    limit resolved as `bt.backtest` resolves it, the class attribute winning). `plan` is
    its settings for today (`_settings.RunPlan`, the present of the start), `arguments`
    the arguments of the backtest of today they were resolved from, `instruments` its
    instruments from the reference tables, `needs` the bars its indicators need before
    today and `measured` the warm-ups measured to learn them, by indicator label.
    """

    strategy: type[Strategy]
    name: str
    symbol: str
    exchange: str
    interval: str
    capital: float
    max_daily_loss: float
    plan: RunPlan
    arguments: Mapping[str, Any]
    instruments: tuple[Instrument, ...]
    needs: tuple[SeriesNeed, ...]
    measured: Mapping[str, WarmUp]

    @property
    def params(self) -> Mapping[str, Any] | None:
        """The strategy's parameter values (None for its defaults)."""
        return self.arguments["params"]


def refuse_unsupported(strategy: type[Strategy]) -> None:
    """Raise NotImplementedError for a strategy class 2.3.0 does not run live (module
    docstring; condition strategies are found by `probe`), ConfigError for one whose
    interval cannot be built from 1-minute bars."""
    maker = made_by(strategy)
    if maker is not None:
        raise not_supported(
            f"a {maker} strategy in a live session",
            "write the strategy as a bt.Strategy class that decides in on_bar",
        )
    for declared, what in (("symbols", "several symbols"), ("universe", "a universe")):
        if getattr(strategy, declared):
            raise not_supported(
                f"a strategy on {what} ({declared}=) in a live session",
                "give each symbol a strategy of its own (symbol=) and pass the list to bt.live",
            )
    if strategy.underlying is not None:
        raise not_supported(
            "a futures series (underlying with expiry) in a live session",
            "name the contract as the strategy's symbol, for example NIFTY26NOVFUT on NFO",
        )
    if strategy.interval and parse_interval(strategy.interval)[0] != "intraday":
        raise not_supported(
            f"a {strategy.interval} strategy interval in a live session",
            "trade an intraday interval, such as '5m'",
        )
    if strategy.product is not None and strategy.product != LIVE_PRODUCT:
        raise not_supported(
            f"product {strategy.product} in a live session (positions held overnight need "
            "GTT orders)",
            f"trade {LIVE_PRODUCT}: remove product from the class, or set it to '{LIVE_PRODUCT}'",
        )
    if strategy.interval:
        _check_interval(strategy.interval, DEFAULT_BASE)


def probe(
    strategy: type[Strategy], plan: RunPlan, instruments: list[Instrument], params: Any
) -> tuple[tuple[SeriesNeed, ...], dict[str, WarmUp]]:
    """(the bars before today each series' indicators need, the warm-ups measured to
    learn them): `init` run on the engine's reference bars, as `bt.backtest` runs it
    before loading data (`_warm_up_probe.combination_needs`), on a strategy instance of
    its own. Raises NotImplementedError for a condition strategy, and what `init`
    raises."""
    checks = IndicatorChecks(strategy_source_hash(strategy) or strategy.__qualname__)
    setup = probe_setup(strategy, plan, instruments, None)
    context = ProbeContext(setup, checks)
    instance = strategy(**(params or {}))
    instance._ctx = context
    instance.init()
    refuse_conditions(instance)
    return history_needs(strategy, context.run.checks.outcomes, setup, checks), checks.measured()


def refuse_conditions(instance: Strategy) -> None:
    """Raise NotImplementedError when the strategy instance declared conditions in `init`
    (module docstring)."""
    if has_conditions(instance):
        raise not_supported(
            f"a condition strategy ({CONDITIONS}) in a live session",
            "write its rules in on_bar: buy, sell or exit there when the condition holds",
        )


def strategy_spec(
    strategy: type[Strategy], name: str, arguments: Mapping[str, Any], calendar: Calendar
) -> StrategySpec:
    """The spec of `strategy` (class name `name`) from the arguments of the backtest of
    today (`_plan_args.LiveArguments.backtest_arguments`), with `calendar` (module
    docstring).

    Raises NotImplementedError for what a live session does not run, ConfigError for
    unusable settings and for a strategy with no daily loss limit (mandatory live: from
    `max_daily_loss=` or the class attribute), the reference tables' errors for a contract
    they do not cover, and what the strategy's `init` raises on the reference bars."""
    refuse_unsupported(strategy)
    plan = resolve(strategy, dict(arguments))
    limit = plan.risk["max_daily_loss"]
    if limit is None:
        raise ConfigError(
            f"{name}: max_daily_loss is mandatory in a live session; pass max_daily_loss= "
            "(rupees, or bt.rupees(...)) to bt.live, or set it on the class"
        )
    instruments, _, _ = resolve_instruments(plan, None, calendar)
    check_margin(instruments, plan.product, (plan.start, plan.end))
    needs, measured = probe(strategy, plan, instruments, arguments["params"])
    return StrategySpec(
        strategy=strategy,
        name=name,
        symbol=plan.symbols[0],
        exchange=plan.exchange,
        interval=plan.interval,
        capital=plan.capital,
        max_daily_loss=limit,
        plan=plan,
        arguments=dict(arguments),
        instruments=tuple(instruments),
        needs=needs,
        measured=measured,
    )


def refuse_shared_legs(specs: tuple[StrategySpec, ...]) -> None:
    """Raise NotImplementedError when two strategies trade one symbol on one exchange:
    each leg (a symbol and product) has one strategy, whose orders and broker-held stop
    the session matches to it."""
    seen: dict[tuple[str, str], str] = {}
    for spec in specs:
        key = (spec.symbol, spec.exchange)
        if key in seen:
            raise not_supported(
                f"two strategies on {spec.exchange}:{spec.symbol} ({seen[key]} and "
                f"{spec.name}) in one session",
                "run one strategy per symbol",
            )
        seen[key] = spec.name


def ist_day(present_ns: int) -> dt.date:
    """The IST date of `present_ns` (UTC nanoseconds): the session's trading day."""
    return ist_time(present_ns).date()


__all__ = [
    "CONDITIONS",
    "LIVE_PRODUCT",
    "StrategySpec",
    "ist_day",
    "probe",
    "refuse_conditions",
    "refuse_shared_legs",
    "refuse_unsupported",
    "strategy_spec",
]
