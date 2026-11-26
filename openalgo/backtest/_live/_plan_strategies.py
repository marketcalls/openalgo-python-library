"""What a live session runs of each strategy class, checked before anything is called
(16-session-design 1.1 and 3.1 step 1; decision 5 of its section 10.0; unit 7). Part of
`plan`.

`strategy_spec` refuses, with NotImplementedError saying what to do, what a live session
does not run: `bt.from_signals` and `bt.from_arrays` strategies, strategies on several
symbols or a universe, futures series (an `underlying` with an `expiry`), a daily or longer
interval in MIS (`refuse_daily_mis`; in CNC and NRML such a strategy decides before the
open, `_plan_daily`), and condition strategies
(`entry_when`, `exit_when`, `short_when`, `cover_when`), whose arrays are computed ahead
over the whole series (decision 5): their rules go in `on_bar`. CNC and NRML run, their
positions carried overnight with a GTT stop (20-positional-live 1); two of them on one
symbol and product in one session are refused (`refuse_shared_positional`, decision
11.6). A strategy on a currency
derivatives exchange (CDS, BCD) is refused with ConfigError: currency derivatives are
backtest-only in 2.4.1, as no supervised sandbox session has traded them. It then resolves the
strategy's settings for today as `bt.backtest` resolves them (`_settings.resolve`: a
class attribute wins over the run argument of the same name, except the daily loss
limit: live it is the `max_daily_loss=` of the call, in rupees, and a class attribute
must agree with it) and learns what history its indicators need by running `init` on the engine's
reference bars, as a backtest does before it loads data (`_warm_up_probe`): that run is
also the strategy layer's causality and window-stability check, and the one that finds
a condition strategy, since conditions are declared in `init`. Its version, the id of its
code and parameters every order carries, is read then too (`_version_id`; 18-multi-strategy
12.1). A strategy that names a contract whose expiry is before today is refused with the
reference tables' ConfigError and one sentence more (`_instruments_today`): a position
carried on it was settled by the exchange, and a start on the contract traded now records
the settlement (`_carry_contracts`; the lead's ruling 2 of review round 4 of unit N5a).
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .._arrays import made_by
from .._causality_checks import IndicatorChecks
from .._conditions import has_conditions
from .._instruments import resolve_instruments
from .._manifest import strategy_source_hash
from .._margin_check import check_margin
from .._messages import not_supported
from .._positional_exit import MCX, PHYSICAL, positional_exits
from .._products import MIS, OVERNIGHT_PRODUCTS
from .._risk_settings import GTT_STOPS, rupees_amount
from .._series import ist_time
from .._settings import RunPlan, resolve
from .._timeframes import _check_interval
from .._warm_up import SeriesNeed
from .._warm_up_measure import WarmUp
from .._warm_up_probe import ProbeContext, history_needs, probe_setup
from ..calendar import Calendar
from ..errors import ConfigError, ReferenceDataError
from ..reference import Instrument, default_tables
from ..reference._store import run_advice
from ..reference.symbols import SymbolError, parse_symbol
from ..report._format import indian_digits
from ..strategy import Strategy
from ._plan_daily import decides_before_open
from ._version_id import StrategyVersion, version_of
from .series import DEFAULT_BASE

#: What a condition strategy declares in `init`, as messages name it.
CONDITIONS = "entry_when, exit_when, short_when, cover_when"
#: The currency derivatives exchanges, backtest-only in 2.4.1 (`refuse_currency`).
CURRENCY_EXCHANGES = ("CDS", "BCD")
#: The sentence the refusal of a strategy naming an expired contract ends with
#: (`_instruments_today`), true whether or not a position is carried on it.
EXPIRED_CONTRACT_SETTLED = (
    "If a strategy held a position on it from an earlier day, the exchange has settled it: "
    "start the strategy on the contract traded now, and the start records the settlement"
)
#: The refusal of a strategy on a stock future or option, with what a run can do instead.
STOCK_DERIVATIVES_NOT_LIVE = (
    "{symbol}: stock futures and stock options cannot be traded in bt.live yet. A backtest "
    "can trade one: pass it with instrument=Instrument(symbol=..., exchange=..., "
    'kind=..., underlying=..., expiry=..., lot_size=..., tick_size=..., settlement="physical") '
    "(Instrument from openalgo.backtest.reference); a stock future or option settles by "
    "delivery, so the engine exits it at the square-off of its last trading day"
)


@dataclass(frozen=True)
class StrategySpec:
    """One strategy of a live session as resolved at the start (module docstring).

    `strategy` is the class and `name` its class name; `symbol` and `exchange` what it
    trades, `interval` its bar interval. `capital` and `max_daily_loss` are rupees (the
    limit given to `bt.live`, which a class attribute must agree with). `plan` is
    its settings for today (`_settings.RunPlan`, the present of the start), `arguments`
    the arguments of the backtest of today they were resolved from, `instruments` its
    instruments from the reference tables, `needs` the bars its indicators need before
    today and `measured` the warm-ups measured to learn them, by indicator label.
    `version` is its version: the id of its code and parameters (`_version_id`).
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
    version: StrategyVersion

    @property
    def params(self) -> Mapping[str, Any] | None:
        """The strategy's parameter values (None for its defaults)."""
        return self.arguments["params"]

    @property
    def product(self) -> str:
        """The product the strategy trades ("MIS", "CNC" or "NRML"), as its settings
        resolved it: every order of the strategy carries it, and its leg's units at the
        broker are read by it (`_broker_units`)."""
        return self.plan.product

    @property
    def positional_stops(self) -> str:
        """How the strategy's CNC or NRML stops are held overnight, as its settings
        resolved `positional_stops`: "gtt" or "rearm_daily" (20-positional-live 6)."""
        return self.plan.risk["positional_stops"]

    @property
    def stop_gtt(self) -> bool:
        """Whether the strategy's leg stop rests at the broker as a SINGLE GTT: a CNC or
        NRML strategy with `positional_stops="gtt"`. An MIS strategy's stop, and a
        `"rearm_daily"` one's, is a day stop order."""
        return self.product in OVERNIGHT_PRODUCTS and self.positional_stops == GTT_STOPS


def refuse_unsupported(strategy: type[Strategy]) -> None:
    """Raise NotImplementedError for a strategy class a live session does not run (module
    docstring; condition strategies are found by `probe`), ConfigError for one on a
    currency derivatives exchange (`refuse_currency`) or whose interval cannot be built
    from 1-minute bars."""
    refuse_currency(strategy)
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
            "name the contract as the strategy's symbol in OpenAlgo's form "
            "[underlying][DDMMMYY]FUT, for example NIFTY29DEC26FUT on NFO",
        )
    if strategy.interval:
        _check_interval(strategy.interval, DEFAULT_BASE)


def refuse_daily_mis(name: str, plan: RunPlan) -> None:
    """Raise NotImplementedError for a daily or longer strategy in MIS (the lead's ruling
    N5b-1): it decides before the open on the session that closed and its orders go at the
    next open, while an MIS position must be squared off the day it is entered, so it could
    never be held from one decision to the next, as its backtest on daily bars holds it."""
    if plan.product == MIS and decides_before_open(plan.interval):
        raise not_supported(
            f"{name}: a {plan.interval} strategy in MIS in a live session (it decides before "
            "the open, and an MIS position is squared off the same day)",
            "trade it in CNC or NRML to hold its positions from one decision to the next, or "
            "give it an intraday interval, such as '5m'",
        )


def refuse_shared_positional(specs: Sequence[StrategySpec]) -> None:
    """Raise NotImplementedError when two strategies of a session hold CNC or NRML on one
    symbol and product (20-positional-live decision 11.6): each carried leg has one stop
    GTT and one day-close record of its own, which two strategies' positions on one
    broker line would share. An MIS and a CNC or NRML strategy on one symbol are allowed
    (different products, two broker lines)."""
    sharing: dict[tuple[str, str, str], list[str]] = {}
    for spec in specs:
        if spec.product in OVERNIGHT_PRODUCTS:
            sharing.setdefault((spec.exchange, spec.symbol, spec.product), []).append(spec.name)
    for (exchange, symbol, product), names in sharing.items():
        if len(names) > 1:
            raise not_supported(
                "two CNC or NRML strategies on one symbol and product in a live session "
                f"({' and '.join(names)} on {exchange}:{symbol} in {product})",
                "run one of them in the session, or trade the others in MIS (an MIS and a "
                "CNC or NRML strategy may trade one symbol)",
            )


def refuse_no_tender(
    name: str, product: str, instruments: Sequence[Instrument], calendar: Calendar
) -> None:
    """Raise ConfigError for a CNC or NRML strategy on an MCX contract settled by delivery
    whose tender period the reference table `mcx_contracts` does not give
    (20-positional-live 9; the lead's ruling N5b-3): the session cannot tell the day to exit
    the position before MCX marks it for delivery, where a backtest settles it at expiry
    with a warning. The message names the contract and the table, and what runs instead: no
    run takes reference tables of the user's yet (the exit day is read from the built-in
    ones), so it cannot point to a user file. An MIS strategy on it is squared off each day
    and runs."""
    if product not in OVERNIGHT_PRODUCTS:
        return
    for instrument, day in zip(
        instruments, positional_exits(instruments, calendar).days, strict=True
    ):
        delivery = instrument.exchange == MCX and instrument.settlement == PHYSICAL
        if delivery and instrument.kind == "future" and day is None:
            raise ConfigError(
                f"{name}: {instrument.symbol} is settled by delivery, and the reference "
                "table mcx_contracts gives no tender period for it, so a live session cannot "
                "tell the day to exit a CNC or NRML position on it before MCX marks it for "
                "delivery. Trade it in MIS, or trade a contract the table gives a tender "
                "period for"
            )


def refuse_currency(strategy: type[Strategy]) -> None:
    """Raise ConfigError for a strategy on CDS or BCD: currency derivatives are backtest-only
    in 2.4.1, since no supervised sandbox session has traded them yet (unit C of 2.4.1)."""
    if strategy.exchange in CURRENCY_EXCHANGES:
        raise ConfigError(
            f"{strategy.__name__}: {strategy.exchange} currency derivatives are backtest-only "
            "in this release (no supervised sandbox session has traded them yet); backtest "
            "them with bt.backtest, or trade NSE, BSE, NFO, BFO or MCX live"
        )


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
    `max_daily_loss=` of the call; a class attribute must agree with it), the reference
    tables' errors for a contract
    they do not cover, and what the strategy's `init` raises on the reference bars."""
    refuse_unsupported(strategy)
    plan = resolve(strategy, dict(arguments))
    limit = _live_daily_loss_limit(name, plan.risk["max_daily_loss"], arguments)
    refuse_daily_mis(name, plan)
    instruments = _instruments_today(plan, calendar)
    refuse_no_tender(name, plan.product, instruments, calendar)
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
        version=version_of(strategy, arguments["params"]),
    )


def _instruments_today(plan: RunPlan, calendar: Calendar) -> list[Instrument]:
    """The instruments of `plan` on its day (`resolve_instruments`). Raises what that
    raises; a ConfigError while the plan names a contract whose expiry is before its day
    ends with `EXPIRED_CONTRACT_SETTLED` (module docstring); a ReferenceDataError says
    what a run can do: for a stock future or option, that `bt.live` does not trade one yet
    (`STOCK_DERIVATIVES_NOT_LIVE`), else the reference store's advice for a run
    (`reference._store.run_advice`)."""
    try:
        instruments, _, _ = resolve_instruments(plan, None, calendar)
    except ConfigError as error:
        if not _names_expired(plan):
            raise
        raise ConfigError(f"{error}. {EXPIRED_CONTRACT_SETTLED}") from error
    except ReferenceDataError as error:
        stock = _stock_derivative(plan)
        if stock is not None:
            raise ReferenceDataError(STOCK_DERIVATIVES_NOT_LIVE.format(symbol=stock)) from error
        raise ReferenceDataError(run_advice(str(error))) from error
    return instruments


def _stock_derivative(plan: RunPlan) -> str | None:
    """The first symbol of `plan` that is a future or an option on a stock (an underlying
    the reference tables list as no index on its exchange), or None."""
    for symbol in plan.symbols:
        try:
            parsed = parse_symbol(symbol, plan.exchange)
        except SymbolError:
            continue
        if parsed.underlying is None:
            continue
        try:
            kind = default_tables().underlying_type(parsed.underlying, plan.exchange, plan.start)
        except ReferenceDataError:
            kind = None
        if kind != "index":
            return symbol
    return None


def _names_expired(plan: RunPlan) -> bool:
    """Whether a symbol of `plan` is a contract whose expiry (as its symbol writes it) is
    before the plan's day; a symbol that does not parse names none."""
    for symbol in plan.symbols:
        try:
            expiry = parse_symbol(symbol, plan.exchange).expiry
        except SymbolError:
            continue
        if expiry is not None and expiry < plan.start:
            return True
    return False


def ist_day(present_ns: int) -> dt.date:
    """The IST date of `present_ns` (UTC nanoseconds): the session's trading day."""
    return ist_time(present_ns).date()


def _live_daily_loss_limit(
    name: str, resolved: float | None, arguments: Mapping[str, Any]
) -> float:
    """The strategy's daily loss limit for the session, in rupees: the `max_daily_loss=` of
    the `bt.live` call. A live risk limit is the one given when the session starts, never
    one taken silently from the class (maintainer, 2026-11-16: capital and max_daily_loss
    are mandatory live). `resolved` is the backtest's resolution, in which a class
    attribute wins; when the class sets a different limit, the session refuses to start.

    Raises ConfigError for a limit missing from the call and for a class limit that differs
    from it."""
    given = rupees_amount(arguments.get("max_daily_loss"), "max_daily_loss")
    if given is None:
        raise ConfigError(
            f"{name}: max_daily_loss is mandatory in a live session; pass max_daily_loss= "
            "(rupees, or bt.rupees(...)) to bt.live: a limit set only on the class is not "
            "taken as the live limit"
        )
    if resolved != given:
        raise ConfigError(
            f"{name}: max_daily_loss of Rs {indian_digits(given)} differs from the class "
            f"attribute max_daily_loss of Rs {indian_digits(resolved)}: give bt.live the same "
            "value, or remove the class attribute, so the live limit is the one you pass"
        )
    return given


__all__ = [
    "CONDITIONS",
    "StrategySpec",
    "ist_day",
    "probe",
    "refuse_conditions",
    "refuse_daily_mis",
    "refuse_no_tender",
    "refuse_shared_positional",
    "refuse_unsupported",
    "strategy_spec",
]
