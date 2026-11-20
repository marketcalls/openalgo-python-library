"""The run functions: `bt.backtest`, `bt.optimize` and the strategy builders; `bt.live` and
`bt.record` are re-exported from `_live_verbs`, `bt.robustness` and `bt.walk_forward`
from `_robust_verbs`.

How a backtest runs (the Python half of the design in `rust/oa_bt/src/engine/mod.rs`):

1. **Resolve** (`_settings.resolve`): the strategy's class attributes, the run arguments
   and the documented defaults become one `RunPlan` in kernel units. A strategy class
   attribute wins over a run argument of the same name.
2. **Inputs** (`_pipeline.prepare`): the calendar (`calendar.default_calendar()` unless
   given), the base interval and its bars from the data layer (`_base_interval`: the
   1-minute coverage probe; `data.load_bars`: OpenAlgo through `client`, or the user's
   `data=`), the history before `start` from the same source (`_warm_up_history`), the
   instruments (`_instruments`: `instrument=`, the generic cash-equity instrument, or
   the reference tables) and the market settings (`_market`: charges, slippage,
   per-instrument facts), each stated in the Assumptions block.
3. **Engine** (`_engine.build_engine`): the native `Engine` validates everything and
   refuses, with NotImplementedError naming the feature, any setting it cannot honour
   yet. Nothing is ignored.
4. **Drive** (`_pipeline.execute`, `_driver.drive`): the strategy instance is bound to a
   `RunContext`, `init` runs (indicators, timers), then the loop `advance` -> timers or
   callbacks -> `on_bar` -> orders runs to the end of the grid, one Python call per
   strategy-bar close once the strategy's indicators are ready (`_warm_up`). Condition
   and array strategies (`entry_when`, `bt.from_arrays`,
   `bt.from_signals`) instead run the same kernel through the Rust signal driver with
   no Python call per bar (`_conditions.drive_conditions`).
5. **Result** (`result.Result`): the kernel's result columns as DataFrames, with the
   Assumptions block, the warnings, the manifest (`_manifest`) and `rerun` (`_rerun`).

A list of strategies is a portfolio run on one account (`_portfolio_run`): each strategy
goes through steps 1 to 3 as it would alone (every strategy's settings checked before any
strategy's data is loaded, `_run_check`), the native fund advances their engines
together over the account (`_portfolio_driver`), and step 5 gives each strategy's
`Result` and the account's `PortfolioResult`. A single strategy never reaches the fund.

Python declares and decides; Rust executes: fills, positions, the ledger, the event
stream and every market rule live in the kernel, so `bt.live` runs the same rules with a
live clock, on an external venue (OpenAlgo and the broker fill the orders). Sandbox or
live is chosen inside OpenAlgo, never here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from ._live_verbs import live, record
from ._messages import not_supported, one_strategy_only
from ._pipeline import execute, prepare
from ._portfolio_run import run_portfolio
from ._rejection_notice import announce_unplaced
from ._rerun import RunRecord
from ._robust_verbs import robustness, walk_forward
from ._strategy_args import SHARED_POOL
from ._warm_up_notice import announce_wait
from .errors import ConfigError
from .reference import Instrument
from .result import Result
from .strategy import Strategy
from .units import Distance

if TYPE_CHECKING:
    from .portfolio_result import PortfolioResult


def backtest(
    strategy: type[Strategy] | Sequence[type[Strategy]],
    client: Any = None,
    *,
    capital: float | Mapping[str, float],
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
    delay_entries: int = 0,
    delay_exits: int = 0,
    quantity_mode: str = "venue",
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
    pool: str = SHARED_POOL,
    account_capital: float | Distance | None = None,
    account_max_daily_loss: float | Distance | None = None,
) -> Result | PortfolioResult:
    """Backtest one strategy class and return its `Result`; or several, given as a list,
    on one account, and return their `PortfolioResult`.

    strategy: a `bt.Strategy` subclass, or a list of them (a portfolio run, below; a list
        of one is a portfolio of one).
    client: an `openalgo.api` client for OpenAlgo data; not needed with `data=`.
    capital: rupees allocated to the strategy (required).
    start, end: IST dates (inclusive); a datetime, or a string with a time of day
        ("2026-03-10 11:00", naive in IST), also bounds the run to slices opening at or
        after `start` and at or before `end`. Taken from `data=` when omitted.
        The sessions before `start` are history: as many bars as the strategy's `self.I`
        indicators need (each one's warm-up, measured on the engine's reference bars)
        are loaded from the run's source and the indicators are computed over them, and
        the strategy decides from `start` on, once its indicators have closed their
        warm-up (`_warm_up`; the Assumptions block and the warnings say what was waited
        for).
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
    fill: where a decided order fills: "next_open" (the default), or, to test how a
        strategy stands worse fill prices, another price of the next base bar: its close
        ("next_close"), its midpoint (high + low) / 2 ("next_mid") or the average of its
        open, high, low and close ("next_ohlc4"), or that price averaged with the bar
        after it ("next_close_2", "next_mid_2", "next_ohlc4_2"; the next bar of the same
        session on an intraday base, the next session's on a daily one). A model fills
        the strategy's orders inside the bar, after the bar's stops and targets, with the
        run's slippage; the engine's own exits and the fills at a level keep their prices.
    delay_entries, delay_exits: strategy bars a decided order that opens or adds, and an
        order that reduces, closes or reverses the position it meets, is held before it
        can fill (0, the default, fills at the next base-bar open); a reversal is an exit,
        so under `delay_entries` alone a strategy always in the market holds only its
        entries from flat. With 1, an order decided at a bar's close fills at the open
        after the next bar's close, and one decided inside a bar (a timer) waits for that
        bar's close first. A cancellation or modification of an order is held as long,
        the order working on its old terms until it arrives. An intraday order whose
        delay ends at its session's close, or any order whose delay ends after the run's
        last bar, is rejected delay_past_close. The legs' stops and targets and the
        engine's own exits are never held. A held order is a working order (in
        `self.orders`), not a position: a strategy that decides its entries by its
        position alone (`self.position.is_flat`) sees none while its entry is held and
        enters again, so decide by `self.orders` too (a condition strategy counts a held
        entry as pending and never does).
    quantity_mode: "venue" (the default: whole shares of cash equity and whole lots of
        derivatives, as an exchange accepts) or "fractional" (part shares of cash
        equity, for research; derivatives stay in whole lots).
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
    pool, account_capital, account_max_daily_loss: the account of a list of strategies
        (below); a run of one strategy leaves them at their defaults.

    The run ends at the earliest of `end`, the close of the last base bar any
    instrument has, and `now`; equity, trades and metrics stop there, a position still
    open is reported open, and a warning names the end used when it is before `end`.
    Empty slots at the end of the last session do not end it early when the session
    before ended its bars as early (15:15 IST for a stock with a closing auction).

    A run in which orders were rejected or skipped for a reason it did not ask for ends
    with one OrdersNotPlacedWarning: how many and the most common reason, the details
    being in `result.warnings` (`_rejection_notice`). Orders refused by the entry window
    or another rule of the run working as it intends are told in `result.warnings` only.
    A run in which the strategy waited inside the run for its indicators ends with one
    WarmUpWarning: the first decision and the strategy bars each symbol waited
    (`_warm_up_notice`).

    A list of strategies runs them together on one account and returns a
    `PortfolioResult`: each strategy's own `Result` (`.strategy(name)`, measured on its own
    capital) and the account's figures and tables. Strategies are named by class name,
    which must differ (one logic twice is two subclasses, or one class and `params` by
    name). Each strategy runs as it would alone (its own exchange, calendar, base interval
    and rules); they meet only through the account, and at one step of one instant the
    first in the list acts first. Two strategies may trade one symbol, on either side.
    These take one value for every strategy, or a dict by strategy class name that names
    every strategy: `capital`, `product`, `base`, `max_daily_loss`, `after_risk_exit`,
    `max_positions`, `drawdown_halt`, `cooldown`, `entry_from`, `entry_until` and
    `square_off`; a class attribute of the same name still wins. `params` is a dict by
    strategy class name (one left out keeps its defaults); a list of one strategy takes
    that strategy's own values, as a class does. Every other argument is the run's: one
    value. Of those keyed by what is traded, each strategy is given the entries of what it
    trades: `data` by symbol, `instrument` by symbol on the strategy's exchange (a
    strategy on a futures underlying the contracts of that underlying, never the share of
    its name), `corporate_actions` by symbol to the strategies on NSE and BSE shares only,
    and `margin` by underlying; an instrument or corporate action no strategy takes is
    refused, a `margin=` no strategy uses is a warning of the run, one table of bars is
    refused when the strategies trade more than one series between them (a share and the
    futures of its name are two), and one list of corporate actions unless they trade
    exactly one share. Every strategy's settings are checked before any data is loaded.

    pool: "shared" (the default): one account; an opening fill needs its margin within the
        strategy's own free cash and within the account's free cash (else it is refused
        `account_margin`), and the default sizer sizes from the lesser of the strategy's
        equity and the account's free cash. "separate": fixed allocations, each strategy
        limited by its own capital only; the account is their sum and no strategy's
        margin can block another's.
    account_capital: rupees, the account's funds (a number or bt.rupees(x)); None (the
        default) for the sum of the capitals, with which a shared account refuses an order
        only when other strategies' open losses leave too little free cash. Give it to
        model a smaller account (three strategies of 5 lakh on 10 lakh of funds). With
        pool="separate" it is None or that sum.
    account_max_daily_loss: rupees (or bt.rupees(x)), the account's daily loss limit: the
        strategies' daily loss measures (on `risk_basis`) of the IST session day added
        up. At or past it every strategy's working entries are cancelled, its positions
        exit as its own daily loss exits do (exit reason "account_loss_limit"), and its
        entries are refused (`account_loss_gate`) until the next session day.

    Raises ConfigError for unusable arguments (a list that names a strategy twice, a dict
    by name that names a strategy the call does not run or leaves one out, account
    settings for one strategy), the data layer's errors for data, and NotImplementedError
    for any setting or feature the kernel does not run yet.
    """
    arguments = dict(locals())
    if isinstance(strategy, list | tuple):
        return run_portfolio(strategy, client, arguments)
    if not (isinstance(strategy, type) and issubclass(strategy, Strategy)):
        raise ConfigError("backtest takes a bt.Strategy subclass, or a list of them")
    return _run(strategy, client, arguments)


def _run(strategy: type[Strategy], client: Any, args: dict[str, Any]) -> Result:
    """Steps 1 to 5 of the module docstring for one strategy (`_pipeline`)."""
    prepared = prepare(strategy, client, args)
    record = RunRecord(strategy, client, {k: v for k, v in args.items() if k != "client"})
    result = execute(strategy, prepared, args, record=record)
    announce_unplaced(result)
    announce_wait(result)
    return result


def optimize(strategy: type[Strategy], client: Any = None, **settings: Any) -> Any:
    """Sweep the strategy's parameters over a grid and return an `OptimizeResult`
    (`.table`, `.best`, `.select(**params)`, `.heatmap(x, y)`, `.path`; and from the grid
    it ran, `.neighbourhood` with a spike warning in `.warnings`, `.deflated_sharpe`,
    `.overfitting_probability`, `.overfitting` and `.tearsheet(path)`, its heatmap page).
    Takes the run arguments of `bt.backtest`, ranges named after parameters
    (`fast=range(5, 30, 5)`; `bt.param` ranges otherwise), `objective=`, `maximize=`,
    `workers=` and `progress=`. See `_optimize.optimize` for the rules and errors."""
    from ._optimize import optimize as sweep

    if isinstance(strategy, list | tuple):
        raise one_strategy_only("bt.optimize")
    return sweep(strategy, client, **settings)


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
    "record",
    "robustness",
    "walk_forward",
]
