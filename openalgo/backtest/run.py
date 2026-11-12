"""The run functions: `bt.backtest` (and the signatures of the ones still to come).

How a backtest runs (the Python half of the design in `rust/oa_bt/src/engine/mod.rs`):

1. **Resolve** (`_settings.resolve`): the strategy's class attributes, the run arguments
   and the documented defaults become one `RunPlan` in kernel units. A strategy class
   attribute wins over a run argument of the same name.
2. **Inputs** (`_pipeline.prepare`): the calendar (`calendar.default_calendar()` unless
   given), the base interval and its bars from the data layer (`_base_interval`: the
   1-minute coverage probe; `data.load_bars`: OpenAlgo through `client`, or the user's
   `data=`), and the instruments (`_instruments`: `instrument=`, the generic cash-equity
   instrument, or the reference tables), each stated in the Assumptions block.
3. **Engine** (`_engine.build_engine`): the native `Engine` validates everything and
   refuses, with NotImplementedError naming the feature, any setting it cannot honour
   yet. Nothing is ignored.
4. **Drive** (`_pipeline.execute`, `_driver.drive`): the strategy instance is bound to a
   `RunContext`, `init` runs (indicators, timers), then the loop `advance` -> timers or
   callbacks -> `on_bar` -> orders runs to the end of the grid, one Python call per
   strategy-bar close. Condition and array strategies
   (`entry_when`, `bt.from_arrays`) will run the same kernel through a Rust driver with
   no Python call per bar.
5. **Result** (`result.Result`): the kernel's result columns as DataFrames, with the
   Assumptions block, the warnings, the manifest (`_manifest`) and `rerun` (`_rerun`).

Python declares and decides; Rust executes: fills, positions, the ledger, the event
stream and every market rule live in the kernel, so `bt.live` (Phase 2) will run the same
rules with a live clock. Sandbox or live is chosen inside OpenAlgo, never here.
"""

from __future__ import annotations

from typing import Any

from ._pipeline import execute, prepare
from ._rerun import RunRecord
from .errors import ConfigError
from .reference import Instrument
from .result import Result
from .strategy import Strategy

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
    instrument: an `Instrument`, or a dict of them by symbol; without it NSE or BSE cash
        equity is a generic cash-equity instrument (lot 1, tick 0.05) and futures and
        options come from the reference tables (`_instruments`; the Assumptions say so).
    calendar: a session calendar; the built-in one by default.
    base: the base interval ("1m", "D", ...); default "D" for daily strategies, and for
        intraday ones 1 minute when 1-minute data covers the run for every instrument,
        else the strategy interval (`_base_interval`; the Assumptions say which and why).
    fill, quantity_mode, costs, slippage: "next_open"; "fractional" or "venue";
        "zerodha" or None (no charges, with a warning); "default" (per segment), 0,
        bt.ticks(n) or bt.pct(p).
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
    """Steps 1 to 5 of the module docstring for one strategy (`_pipeline`)."""
    prepared = prepare(strategy, client, args)
    record = RunRecord(strategy, client, {k: v for k, v in args.items() if k != "client"})
    return execute(strategy, prepared, args, record=record)


def live(
    strategy: type[Strategy], client: Any, *, capital: float, max_daily_loss: float, **settings: Any
) -> Any:
    """Trade a strategy through OpenAlgo in the mode set inside OpenAlgo (sandbox or
    live). `capital` and `max_daily_loss` are mandatory risk limits."""
    raise NotImplementedError("bt.live is planned for Phase 2")


def optimize(strategy: type[Strategy], client: Any = None, **settings: Any) -> Any:
    """Sweep the strategy's parameters over a grid and return an `OptimizeResult`
    (`.table`, `.best`, `.select(**params)`, `.heatmap(x, y)`, `.path`). Takes the run
    arguments of `bt.backtest`, ranges named after parameters (`fast=range(5, 30, 5)`;
    `bt.param` ranges otherwise), `objective=`, `maximize=`, `workers=` and `progress=`.
    See `_optimize.optimize` for the rules and errors."""
    from ._optimize import optimize as sweep

    return sweep(strategy, client, **settings)


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
