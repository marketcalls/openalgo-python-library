"""The run functions: `bt.backtest` (and the signatures of the ones still to come).

How a backtest runs (the Python half of the design in `rust/oa_bt/src/engine/mod.rs`):

1. **Resolve** (`_settings.resolve`): the strategy's class attributes, the run arguments
   and the documented defaults become one `RunPlan` in kernel units. A strategy class
   attribute wins over a run argument of the same name.
2. **Inputs** (`_pipeline.prepare`): the calendar (`calendar.default_calendar()` unless
   given), the base interval and its bars from the data layer (`_base_interval`: the
   1-minute coverage probe; `data.load_bars`: OpenAlgo through `client`, or the user's
   `data=`), the instruments (`_instruments`: `instrument=`, the generic cash-equity
   instrument, or the reference tables) and the market settings (`_market`: charges,
   slippage, per-instrument facts), each stated in the Assumptions block.
3. **Engine** (`_engine.build_engine`): the native `Engine` validates everything and
   refuses, with NotImplementedError naming the feature, any setting it cannot honour
   yet. Nothing is ignored.
4. **Drive** (`_pipeline.execute`, `_driver.drive`): the strategy instance is bound to a
   `RunContext`, `init` runs (indicators, timers), then the loop `advance` -> timers or
   callbacks -> `on_bar` -> orders runs to the end of the grid, one Python call per
   strategy-bar close. Condition and array strategies (`entry_when`, `bt.from_arrays`,
   `bt.from_signals`) instead run the same kernel through the Rust signal driver with
   no Python call per bar (`_conditions.drive_conditions`).
5. **Result** (`result.Result`): the kernel's result columns as DataFrames, with the
   Assumptions block, the warnings, the manifest (`_manifest`) and `rerun` (`_rerun`).

Python declares and decides; Rust executes: fills, positions, the ledger, the event
stream and every market rule live in the kernel, so `bt.live`, when it is supported, will
run the same rules with a live clock. Sandbox or live is chosen inside OpenAlgo, never here.
"""

from __future__ import annotations

from typing import Any

from ._messages import SEVERAL_STRATEGIES, not_supported
from ._pipeline import execute, prepare
from ._rerun import RunRecord
from .errors import ConfigError
from .reference import Instrument
from .result import Result
from .strategy import Strategy
from .units import Distance


def backtest(
    strategy: type[Strategy],
    client: Any = None,
    *,
    capital: float,
    start: Any = None,
    end: Any = None,
    data: Any = None,
    benchmark: Any = None,
    instrument: Instrument | dict[str, Instrument] | None = None,
    calendar: Any = None,
    source: str = "auto",
    tz: str | None = None,
    corporate_actions: Any = None,
    base: str | None = None,
    fill: str = "next_open",
    quantity_mode: str = "fractional",
    costs: str | None = "zerodha",
    slippage: Any = "default",
    margin: Any = None,
    product: str | None = None,
    entry_from: str | None = None,
    entry_until: str | None = None,
    square_off: str | None = None,
    max_daily_loss: float | Distance | None = None,
    risk_basis: str = "gross",
    after_risk_exit: str = "lock_session",
    max_positions: int | None = None,
    drawdown_halt: float | Distance | None = None,
    cooldown: int | None = None,
    round_levels: bool = True,
    strict_config: bool = True,
    roll: int | None = 1,
    params: dict[str, Any] | None = None,
    now: Any = None,
) -> Result:
    """Backtest one strategy class and return its `Result`.

    strategy: a `bt.Strategy` subclass (a list of strategies is not supported in this
        version).
    client: an `openalgo.api` client for OpenAlgo data; not needed with `data=`.
    capital: rupees allocated to the strategy (required).
    start, end: IST dates (inclusive); a datetime also bounds the run to slices opening
        at or after `start` and at or before `end`. Taken from `data=` when omitted.
    data: own bars (a DataFrame, a CSV, Parquet or Feather file path, or a dict of them
        by symbol) at the base interval.
    benchmark: what the reports compare the strategy with (alpha, beta, information
        ratio, the comparison tables and charts); it never trades. An index symbol
        ("NIFTY", on NSE_INDEX for an NSE or NFO strategy and BSE_INDEX for BSE or BFO),
        a (symbol, exchange) pair on NSE_INDEX or BSE_INDEX, whose daily closes come
        from the run's own source (a dict `data=` holding it, else `client`), or a
        pandas Series of closes (a DataFrame with a close column). None (the default):
        no benchmark. Missing benchmark data is a warning (`_benchmark`).
    instrument: an `Instrument`, or a dict of them by symbol; without it NSE or BSE cash
        equity is a generic cash-equity instrument (lot 1, tick 0.05) and futures and
        options come from the reference tables (`_instruments`; the Assumptions say so).
    calendar: a session calendar; the built-in one by default.
    corporate_actions: splits, bonuses and checked dates of NSE and BSE shares (a list of
        `bt.data.CorporateAction`, or a dict of lists by symbol). A gap in the bars that
        looks like an unadjusted split or bonus stops the run unless declared; a
        declared one the bars are not adjusted for is a warning, and holding across it
        is not built yet (`_run_corporate_actions`, `data.splits`).
    base: the base interval ("1m", "D", ...); default "D" for daily strategies, and for
        intraday ones 1 minute when 1-minute data covers the run for every instrument,
        else the strategy interval (`_base_interval`; the Assumptions say which and why).
    fill, quantity_mode: "next_open"; "fractional" or "venue".
    costs: "zerodha" (the default) or another preset name, a preset from bt.costs, a
        bt.costs.CostModel, or None (no charges, with a warning on every report); see
        `_market`.
    slippage: "default" (the dated per-class table), 0, bt.ticks(n), bt.pct(p),
        bt.slippage.bps(b) or bt.slippage.table(...); see `openalgo.backtest.slippage`.
    margin: the margin a futures position blocks. None (the default): 20 percent of the
        contract value (5x) for every futures contract, any underlying and date. One
        value for every futures contract, or a dict by underlying symbol ({"NIFTY":
        bt.pct(12), "CRUDEOIL": bt.rupees(450_000)}) whose missing names keep the
        default; a value is bt.pct(x), x percent of the contract value, or
        bt.rupees(x), x rupees per lot of the contract traded (the figure a broker's
        margin calculator shows). A backtest assumption only: in live trading the
        broker computes margin. See `_futures_margin`.
    product, entry_from, entry_until, square_off: product default by interval and
        exchange; IST "HH:MM" times for MIS (calendar defaults when omitted).
    max_daily_loss, risk_basis, after_risk_exit, max_positions, drawdown_halt, cooldown,
        round_levels, strict_config: risk settings; strategy class attributes of the
        same names win. `max_daily_loss` is rupees (or bt.rupees), `drawdown_halt`
        percent from peak equity (bt.pct(x)), `cooldown` base bars.
    roll: sessions before expiry at which a futures position rolls to the next contract
        (1, the session before the last trading day; 0 rolls on it), or None to hold it
        to expiry and settle it there in cash (see `Result.settlements`). A futures
        strategy names its underlying (`underlying`, or underlyings in `symbols`) and
        the contract rank in `expiry` ("current_month" or "next_month"); see `_futures`.
    params: parameter values for the strategy.
    now: the present (UTC nanoseconds, a datetime or an ISO string, naive is IST; the
        clock when None). The data load drops bars that have not closed by then.

    The run ends at the earliest of `end`, the close of the last base bar any
    instrument has, and `now`; equity, trades and metrics stop there, a position still
    open is reported open, and a warning names the end used when it is before `end`.

    Raises ConfigError for unusable arguments, the data layer's errors for data, and
    NotImplementedError for any setting or feature the kernel does not run yet.
    """
    if isinstance(strategy, list | tuple):
        raise not_supported(*SEVERAL_STRATEGIES)
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
    live). `capital` and `max_daily_loss` are mandatory risk limits. It takes no
    `margin=`: in live trading the broker computes margin. Not supported in this
    version: raises NotImplementedError."""
    raise not_supported("bt.live", "test the strategy with bt.backtest")


def optimize(strategy: type[Strategy], client: Any = None, **settings: Any) -> Any:
    """Sweep the strategy's parameters over a grid and return an `OptimizeResult`
    (`.table`, `.best`, `.select(**params)`, `.heatmap(x, y)`, `.path`). Takes the run
    arguments of `bt.backtest`, ranges named after parameters (`fast=range(5, 30, 5)`;
    `bt.param` ranges otherwise), `objective=`, `maximize=`, `workers=` and `progress=`.
    See `_optimize.optimize` for the rules and errors."""
    from ._optimize import optimize as sweep

    return sweep(strategy, client, **settings)


def walk_forward(strategy: type[Strategy], client: Any = None, **settings: Any) -> Any:
    """Walk-forward optimisation. Not supported in this version: raises
    NotImplementedError."""
    raise not_supported(
        "bt.walk_forward",
        "run bt.optimize on each in-sample window and bt.backtest with the chosen "
        "parameters on the window after it",
    )


def rebalance(*args: Any, **settings: Any) -> Any:
    """Portfolio rebalancing backtests. Not supported in this version: raises
    NotImplementedError."""
    raise not_supported(
        "bt.rebalance",
        "call self.rebalance(weights) in on_bar of a strategy on several symbols",
    )


def from_signals(func: Any, symbol: str, exchange: str, interval: str, **settings: Any) -> Any:
    """A strategy from a function of the strategy bars that returns entries and exits,
    for `bt.backtest`; read by the rules of conditions (`_arrays.from_signals`)."""
    from ._arrays import from_signals as build

    return build(func, symbol, exchange, interval, **settings)


def from_arrays(entries: Any, exits: Any = None, *extra: Any, **settings: Any) -> Any:
    """A strategy from boolean entry and exit arrays, one value per strategy bar, for
    `bt.backtest`; read by the rules of conditions (`_arrays.from_arrays`). Everything
    after `exits` is a keyword, and `symbol`, `exchange` and `interval` are required:
    `bt.from_arrays(entries, exits, symbol="SBIN", exchange="NSE", interval="5m")`.
    Raises ConfigError, with that example, when one is missing or more values are given
    by position (`symbol=`, `exchange=`, `interval=`, `shorts=` and `covers=` are
    keywords)."""
    from ._arrays import from_arrays as build

    return build(entries, exits, *extra, **settings)


__all__ = [
    "backtest",
    "from_arrays",
    "from_signals",
    "live",
    "optimize",
    "rebalance",
    "walk_forward",
]
