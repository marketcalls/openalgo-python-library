"""The run functions: `bt.backtest` (and the signatures of the ones still to come).

How a backtest runs (the Python half of the design in `rust/oa_bt/src/engine/mod.rs`):

1. **Resolve** (`_settings.resolve`): the strategy's class attributes, the run arguments
   and the documented defaults become one `RunPlan` in kernel units. A strategy class
   attribute wins over a run argument of the same name.
2. **Inputs**: the calendar (`calendar.default_calendar()` unless given), the instruments
   (`instrument=`, or the generic cash-equity instrument for own NSE or BSE data, stated
   in the warnings), and the bars at the base interval from the data layer
   (`data.load_bars`: OpenAlgo through `client`, or the user's `data=`).
3. **Engine** (`_engine.build_engine`): the native `Engine` validates everything and
   refuses, with NotImplementedError naming the feature, any setting it cannot honour
   yet. Nothing is ignored.
4. **Drive** (`_driver.drive`): the strategy instance is bound to a `RunContext`, `init`
   runs, then the loop `advance` -> callbacks -> `on_bar` -> orders runs to the end of
   the grid, one Python call per strategy-bar close. Condition and array strategies
   (`entry_when`, `bt.from_arrays`) will run the same kernel through a Rust driver with
   no Python call per bar.
5. **Result** (`result.Result`): the kernel's result columns as DataFrames, with the
   Assumptions block, the warnings and the manifest.

Python declares and decides; Rust executes: fills, positions, the ledger, the event
stream and every market rule live in the kernel, so `bt.live` (Phase 2) will run the same
rules with a live clock. Sandbox or live is chosen inside OpenAlgo, never here.
"""

from __future__ import annotations

from typing import Any

from . import calendar as calendar_module
from ._context import RunContext
from ._driver import drive
from ._engine import build_engine, call
from ._market import market_settings
from ._settings import assumptions, resolve
from .data import load_bars
from .errors import ConfigError
from .reference import Instrument
from .result import Result
from .strategy import Strategy

CASH_EXCHANGES = ("NSE", "BSE")
PHASE_1_WORK = "is not implemented yet (Phase 1 work in progress)"


def backtest(
    strategy: type[Strategy],
    client: Any = None,
    *,
    capital: float,
    start: Any = None,
    end: Any = None,
    data: Any = None,
    instrument: Instrument | dict[str, Instrument] | None = None,
    calendar: Any = None,
    source: str = "auto",
    tz: str | None = None,
    base: str | None = None,
    fill: str = "next_open",
    quantity_mode: str = "fractional",
    costs: str | None = "zerodha",
    slippage: Any = "default",
    product: str | None = None,
    entry_from: str | None = None,
    entry_until: str | None = None,
    square_off: str | None = None,
    max_daily_loss: float | None = None,
    risk_basis: str = "gross",
    after_risk_exit: str = "lock_session",
    max_positions: int | None = None,
    drawdown_halt: float | None = None,
    cooldown: int | None = None,
    round_levels: bool = True,
    strict_config: bool = True,
    roll: Any = None,
    params: dict[str, Any] | None = None,
) -> Result:
    """Backtest one strategy class and return its `Result`.

    strategy: a `bt.Strategy` subclass (a list of strategies is Phase 3).
    client: an `openalgo.api` client for OpenAlgo data; not needed with `data=`.
    capital: rupees allocated to the strategy (required).
    start, end: IST dates (inclusive); a datetime also bounds the run to slices opening
        at or after `start` and at or before `end`. Taken from `data=` when omitted.
    data: own bars (a DataFrame, or a dict of them by symbol) at the base interval.
    instrument: an `Instrument`, or a dict of them by symbol; without it own NSE or BSE
        data is a generic cash-equity instrument (lot 1, tick 0.05) and a warning says so.
    calendar: a session calendar; the built-in one by default.
    base: the base interval ("1m", "D", ...); default "D" for daily strategies and the
        strategy interval for intraday ones (with a warning until the 1-minute coverage
        probe exists).
    fill, quantity_mode: "next_open"; "fractional" or "venue".
    costs: "zerodha" (the default) or another preset name, a preset from bt.costs, a
        bt.costs.CostModel, or None (no charges, with a warning on every report); see
        `_market`.
    slippage: "default" (the dated per-class table), 0, bt.ticks(n), bt.pct(p),
        bt.slippage.bps(b) or bt.slippage.table(...); see `openalgo.backtest.slippage`.
    product, entry_from, entry_until, square_off: product default by interval and
        exchange; IST "HH:MM" times for MIS (calendar defaults when omitted).
    max_daily_loss, risk_basis, after_risk_exit, max_positions, drawdown_halt, cooldown,
        round_levels, strict_config: risk settings (05-risk-management); strategy class
        attributes of the same names win.
    roll: futures roll settings. params: parameter values for the strategy.

    Raises ConfigError for unusable arguments, the data layer's errors for data, and
    NotImplementedError for any setting or feature the kernel does not run yet.
    """
    if isinstance(strategy, list | tuple):
        raise NotImplementedError("several strategies in one run are planned for Phase 3")
    if not (isinstance(strategy, type) and issubclass(strategy, Strategy)):
        raise ConfigError("backtest takes a bt.Strategy subclass")
    return _run(strategy, client, dict(locals()))


def _run(strategy: type[Strategy], client: Any, args: dict[str, Any]) -> Result:
    """Steps 1 to 5 of the module docstring for one strategy."""
    plan = resolve(strategy, args)
    session_calendar = _calendar(args["calendar"], plan)
    instruments, notes = _instruments(plan, args["instrument"], client)
    frames = [_bars(plan, symbol, client, session_calendar, args) for symbol in plan.symbols]
    market = market_settings(plan, instruments, args)
    config = plan.native_config(deliver_events=strategy.wants_events()) | market.config
    engine = build_engine(session_calendar._cal, config, instruments, frames)
    context = RunContext(engine, instruments, plan.product, strategy.size)
    instance = strategy(**(args["params"] or {}))
    instance._ctx = context
    instance.init()
    drive(instance, engine, context)
    output = call(engine.finish)
    context.close(output["orders"])
    return Result(
        output,
        list(plan.symbols),
        assumptions(plan, session_calendar) | market.assumptions,
        [*plan.warnings, *notes, *market.warnings],
        context.skipped,
        {
            "plan": plan,
            "strategy": strategy.__name__,
            "instruments": [i.to_dict() for i in instruments],
            "cost_schedules": market.cost_schedules,
        },
    )


def _bars(plan: Any, symbol: str, client: Any, calendar: Any, args: dict[str, Any]) -> Any:
    """The base-interval bars of one symbol from the data layer."""
    return load_bars(
        symbol,
        plan.exchange,
        plan.base,
        plan.start,
        plan.end,
        source=args["source"],
        client=client,
        data=_data_for(args["data"], symbol),
        calendar=calendar,
        tz=args["tz"],
    ).bars


def _calendar(calendar: Any, plan: Any) -> calendar_module.Calendar:
    if calendar is None:
        return calendar_module.default_calendar()
    if isinstance(calendar, calendar_module.Calendar):
        return calendar
    return calendar_module.Calendar.from_sessions(calendar, plan.exchange, plan.start, plan.end)


def _instruments(plan: Any, given: Any, client: Any) -> tuple[list[Instrument], list[str]]:
    by_symbol = {given.symbol: given} if isinstance(given, Instrument) else dict(given or {})
    instruments, notes = [], []
    for symbol in plan.symbols:
        if symbol in by_symbol:
            instruments.append(by_symbol[symbol])
        elif client is not None:
            raise NotImplementedError(f"instrument lookup through client.symbol() {PHASE_1_WORK}")
        elif plan.exchange in CASH_EXCHANGES:
            instruments.append(Instrument.cash_equity(symbol, plan.exchange))
            notes.append(f"{symbol}: generic cash-equity instrument assumed (lot 1, tick 0.05)")
        else:
            raise ConfigError(f"pass instrument= for {symbol} on {plan.exchange}")
    return instruments, notes


def _data_for(data: Any, symbol: str) -> Any:
    """The own data of one symbol: `data` itself, or its entry in a dict by symbol."""
    if not isinstance(data, dict):
        return data
    if symbol not in data:
        raise ConfigError(f"data= has no bars for {symbol}")
    return data[symbol]


def live(
    strategy: type[Strategy], client: Any, *, capital: float, max_daily_loss: float, **settings: Any
) -> Any:
    """Trade a strategy through OpenAlgo in the mode set inside OpenAlgo (sandbox or
    live). `capital` and `max_daily_loss` are mandatory risk limits."""
    raise NotImplementedError("bt.live is planned for Phase 2")


def optimize(strategy: type[Strategy], client: Any = None, **settings: Any) -> Any:
    """Sweep the strategy's parameters; returns an OptimizeResult."""
    raise NotImplementedError(f"bt.optimize {PHASE_1_WORK}")


def walk_forward(strategy: type[Strategy], client: Any = None, **settings: Any) -> Any:
    """Walk-forward optimisation; returns a WalkForwardResult."""
    raise NotImplementedError("bt.walk_forward is planned for Phase 4")


def rebalance(*args: Any, **settings: Any) -> Any:
    """Portfolio rebalancing backtests; returns a RebalanceResult."""
    raise NotImplementedError("bt.rebalance is planned for Phase 3")


def from_signals(func: Any, symbol: str, exchange: str, interval: str, **settings: Any) -> Any:
    """A strategy from a function that returns entries and exits."""
    raise NotImplementedError(f"bt.from_signals {PHASE_1_WORK}")


def from_arrays(entries: Any, exits: Any, **settings: Any) -> Any:
    """A strategy from boolean entry and exit arrays."""
    raise NotImplementedError(f"bt.from_arrays {PHASE_1_WORK}")


__all__ = [
    "backtest",
    "from_arrays",
    "from_signals",
    "live",
    "optimize",
    "rebalance",
    "walk_forward",
]
