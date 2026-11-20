"""The list form of `bt.backtest`: several strategies on one account (a portfolio run).

1. **Settings** (`portfolio_plan`): the strategy classes, named by class name (each name
   once); each strategy's `bt.backtest` arguments, the per-strategy settings resolved by
   name (`_strategy_args`) and the run's other arguments shared; the account (pool,
   funds, daily loss limit). Every argument is checked before any data is loaded.
2. **Each strategy alone**: prepared (`_pipeline.prepare`: its settings, data, history and
   market) and begun (`_pipeline.begin`: its engine, its `init`), in call order, exactly as
   `bt.backtest` would run it alone.
3. **The fund** (`_portfolio_driver.drive_fund`) advances the engines together over the
   account; each strategy decides at its own pauses.
4. **Results**: each strategy's `Result` (`_pipeline.conclude`) with a line saying the
   account it ran in, and the `PortfolioResult` of the account. A strategy whose orders
   were not placed, or that waited for its indicators, is told in one warning each,
   named by the strategy.

A single strategy never comes here: `bt.backtest(Strategy, ...)` runs the code it ran
before portfolio runs existed.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ._market import PACKAGE_DIR
from ._messages import counted
from ._pipeline import begin, conclude, prepare
from ._portfolio_driver import drive_fund
from ._portfolio_notes import rupees_text, spoken_list
from ._reject_reasons import ACCOUNT_LOSS_GATE, ACCOUNT_MARGIN
from ._rejection_notice import unplaced_notice
from ._rerun import PortfolioMember
from ._strategy_args import (
    ACCOUNT_SETTINGS,
    EVERY_STRATEGY,
    MANDATORY,
    PER_STRATEGY,
    Account,
    account_settings,
    by_name,
    class_names,
    params_by_name,
    repeated,
    strategy_classes,
)
from ._warm_up_notice import wait_notice
from .errors import ConfigError, OrdersNotPlacedWarning, WarmUpWarning
from .portfolio_result import PortfolioResult
from .result import Result
from .strategy import Strategy

#: The call a dict by strategy name is checked against, as its messages name it.
BACKTEST = "this backtest"


@dataclass(frozen=True)
class PortfolioPlan:
    """The checked settings of a portfolio run: the strategy classes and their names in
    call order, each strategy's `bt.backtest` arguments by name, and the account."""

    strategies: tuple[type[Strategy], ...]
    names: tuple[str, ...]
    arguments: Mapping[str, dict[str, Any]]
    account: Account


def _capital(name: str, value: Any) -> float:
    """A strategy's capital in rupees. Raises ConfigError naming the strategy unless it is
    a positive number."""
    if isinstance(value, bool) or not isinstance(value, int | float) or not value > 0:
        raise ConfigError(f"{name}: capital must be a positive number of rupees, not {value!r}")
    return float(value)


def _refuse_one_frame(strategies: tuple[type[Strategy], ...], data: Any) -> None:
    """Raise ConfigError for own bars given as one table (not a dict by symbol) to
    strategies that trade more than one symbol between them: the table would be read as
    the bars of every symbol."""
    if data is None or isinstance(data, dict):
        return
    symbols = sorted({symbol for each in strategies for symbol in each.traded_symbols()})
    if len(symbols) > 1:
        raise ConfigError(
            f"the strategies trade {', '.join(symbols)}: give data= as a dict of bars by "
            "symbol, which feeds every strategy that trades the symbol"
        )


def portfolio_plan(strategy: Any, args: Mapping[str, Any]) -> PortfolioPlan:
    """The plan of `bt.backtest` given the list `strategy` and the call's arguments `args`
    (module docstring, step 1). Raises ConfigError for a list that is empty, holds
    something other than a `bt.Strategy` subclass or names a strategy twice, for a dict by
    name that names a strategy the call does not run or leaves one out (`params` may: that
    strategy keeps its defaults), for a capital that is not a positive number, for an
    unusable account (`_strategy_args.account_settings`) and for own bars given as one
    table to strategies on several symbols."""
    strategies = strategy_classes(strategy, "bt.backtest")
    names = class_names(strategies)
    twice = repeated(names)
    if twice:
        raise ConfigError(
            f"bt.backtest runs two strategies named {', '.join(twice)}: results are kept by "
            "strategy class name, so give each its own (a subclass, such as class "
            "SbinFast(EmaCross) with its own parameters, or one class and params by name)"
        )
    settings = {
        setting: by_name(
            args[setting],
            names,
            setting,
            owner=BACKTEST,
            required=MANDATORY if setting == "capital" else EVERY_STRATEGY,
        )
        for setting in PER_STRATEGY
        if setting != "params"
    }
    params = params_by_name(args["params"], names, owner=BACKTEST)
    capitals = [_capital(name, settings["capital"][name]) for name in names]
    account = account_settings(args, capitals)
    _refuse_one_frame(strategies, args["data"])
    arguments = {
        name: {
            **args,
            **{setting: values[name] for setting, values in settings.items()},
            "strategy": cls,
            "params": params[name],
            **ACCOUNT_SETTINGS,
        }
        for cls, name in zip(strategies, names, strict=True)
    }
    return PortfolioPlan(strategies, names, arguments, account)


def _refusals(records: Mapping[str, Any], index: int, reason: str) -> int:
    """How many of strategy `index`'s orders the account refused for `reason`."""
    refused = records["refused"]
    pairs = zip(refused["strategy"], refused["reason"], strict=True)
    return sum(1 for strategy, code in pairs if strategy == index and code == reason)


def member_line(plan: PortfolioPlan, index: int, records: Mapping[str, Any]) -> str:
    """The Assumptions line of strategy `index`'s own result: the account it ran in, the
    strategies it shared it with, and the orders the account refused it."""
    account = plan.account
    others = [name for position, name in enumerate(plan.names) if position != index]
    beside = f" with {spoken_list(others)}" if others else " alone"
    if account.pool == "separate":
        line = (
            f"ran in separate allocations (pool='separate'){beside}: the account is the sum "
            "of the capitals, and no other strategy's margin limited it"
        )
    else:
        refused = _refusals(records, index, ACCOUNT_MARGIN)
        line = (
            f"ran in a shared account of {rupees_text(records['account_capital'])}{beside}; "
            f"{counted(refused, 'order')} refused for the account's margin"
        )
    if account.account_max_daily_loss is not None:
        gated = _refusals(records, index, ACCOUNT_LOSS_GATE)
        line += (
            f"; the account's daily loss limit {rupees_text(account.account_max_daily_loss)} "
            f"refused {counted(gated, 'entry', 'entries')}"
        )
    return line


def announce(results: Mapping[str, Result]) -> None:
    """Issue, per strategy and named by it, the warnings `bt.backtest` ends a run of one
    strategy with: orders not placed for a reason the run did not ask for
    (OrdersNotPlacedWarning) and a wait inside the run for the indicators
    (WarmUpWarning)."""
    for name, result in results.items():
        kinds = [instrument["kind"] for instrument in result.manifest["instruments"]]
        notices = (
            (unplaced_notice(result.orders, result.skipped, kinds), OrdersNotPlacedWarning),
            (wait_notice(result.manifest["arguments"]["warm_up"]), WarmUpWarning),
        )
        for notice, category in notices:
            if notice is not None:
                warnings.warn(f"{name}: {notice}", category, skip_file_prefixes=(PACKAGE_DIR,))


def run_portfolio(strategy: Any, client: Any, args: Mapping[str, Any]) -> PortfolioResult:
    """Run the list of strategies `strategy` on one account with the arguments of the
    `bt.backtest` call `args` (module docstring) and return its `PortfolioResult`. Raises
    ConfigError (`portfolio_plan`, and an account the fund refuses), the data layer's
    errors, NotImplementedError for anything a strategy alone would refuse, and the
    kernel's and the strategies' errors, a kernel error starting with its strategy's
    name."""
    plan = portfolio_plan(strategy, args)
    runs = [
        prepare(cls, client, plan.arguments[name])
        for cls, name in zip(plan.strategies, plan.names, strict=True)
    ]
    members = [
        begin(cls, prepared, plan.arguments[name])
        for cls, name, prepared in zip(plan.strategies, plan.names, runs, strict=True)
    ]
    records = drive_fund(members, plan.names, plan.account)
    results = {
        name: conclude(
            cls,
            prepared,
            plan.arguments[name],
            member,
            PortfolioMember(name),
            assumptions={"account": member_line(plan, index, records)},
        )
        for index, (cls, name, prepared, member) in enumerate(
            zip(plan.strategies, plan.names, runs, members, strict=True)
        )
    }
    announce(results)
    benchmark = runs[0].benchmark
    closes = None if benchmark is None else benchmark.closes
    return PortfolioResult(plan.account, results, records, benchmark=closes)


__all__ = ["PortfolioPlan", "announce", "member_line", "portfolio_plan", "run_portfolio"]
