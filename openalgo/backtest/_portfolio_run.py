"""The list form of `bt.backtest`: several strategies on one account (a portfolio run).

1. **Settings** (`portfolio_plan`): the strategy classes, named by class name (each name
   once); each strategy's `bt.backtest` arguments, the per-strategy settings resolved by
   name (`_strategy_args`), the run's settings keyed by symbol cut to the entries of what
   the strategy trades (`_portfolio_symbols`) and the run's other arguments shared; the
   account (pool, funds, daily loss limit).
2. **Each strategy alone**, in call order, exactly as `bt.backtest` would run it alone:
   checked (`_run_check.check_run`: its settings, instruments and its indicators' needs)
   before any strategy's data is loaded, given its part of `margin=`
   (`_portfolio_symbols.own_margins`), then loaded (`_pipeline.load`: its data, history
   and market) and begun (`_pipeline.begin`: its engine, its `init`).
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

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ._messages import counted
from ._pipeline import begin, conclude, load
from ._portfolio_driver import drive_fund
from ._portfolio_notes import rupees_text, spoken_list
from ._portfolio_symbols import (
    own_corporate_actions,
    own_instruments,
    own_margins,
    refuse_one_frame,
)
from ._reject_reasons import (
    ACCOUNT_LOSS_GATE,
    ACCOUNT_MARGIN,
    ACCOUNT_PROFIT_GATE,
    ACCOUNT_TRADE_LIMIT,
)
from ._rejection_notice import announce_unplaced
from ._rerun import PortfolioMember
from ._run_check import check_run
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
from ._warm_up_notice import announce_wait
from .errors import ConfigError
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


def portfolio_plan(strategy: Any, args: Mapping[str, Any]) -> PortfolioPlan:
    """The plan of `bt.backtest` given the list `strategy` and the call's arguments `args`
    (module docstring, step 1). Raises ConfigError for a list that is empty, holds
    something other than a `bt.Strategy` subclass or names a strategy twice, for a dict by
    name that names a strategy the call does not run or leaves one out (`params` may: that
    strategy keeps its defaults), for a capital that is not a positive number, for an
    unusable account (`_strategy_args.account_settings`), and for a setting keyed by
    what is traded that no strategy can take, or that names no symbol while the
    strategies trade several (`_portfolio_symbols`)."""
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
    refuse_one_frame(strategies, args["data"])
    instruments = own_instruments(strategies, args["instrument"])
    actions = own_corporate_actions(strategies, args["corporate_actions"])
    arguments = {
        name: {
            **args,
            **{setting: values[name] for setting, values in settings.items()},
            "strategy": cls,
            "params": params[name],
            "instrument": instruments[index],
            "corporate_actions": actions[index],
            **ACCOUNT_SETTINGS,
        }
        for index, (cls, name) in enumerate(zip(strategies, names, strict=True))
    }
    return PortfolioPlan(strategies, names, arguments, account)


def _refusals(records: Mapping[str, Any], index: int, reason: str) -> int:
    """How many of strategy `index`'s orders the account refused for `reason`."""
    refused = records["refused"]
    pairs = zip(refused["strategy"], refused["reason"], strict=True)
    return sum(1 for strategy, code in pairs if strategy == index and code == reason)


def member_line(plan: PortfolioPlan, index: int, records: Mapping[str, Any]) -> str:
    """The Assumptions line of strategy `index`'s own result: the account it ran in, the
    strategies it shared it with, and the orders the account refused it (for its margin,
    and for each of its daily rules that is set)."""
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
    rules = (
        (account.account_max_daily_loss, "daily loss limit", ACCOUNT_LOSS_GATE),
        (account.account_daily_profit_target, "daily profit target", ACCOUNT_PROFIT_GATE),
    )
    for amount, what, code in rules:
        if amount is not None:
            gated = _refusals(records, index, code)
            line += (
                f"; the account's {what} {rupees_text(amount)} refused "
                f"{counted(gated, 'entry', 'entries')}"
            )
    cap = account.account_max_trades_per_day
    if cap is not None:
        capped = _refusals(records, index, ACCOUNT_TRADE_LIMIT)
        line += (
            f"; the account's cap of {counted(cap, 'entry', 'entries')} a day refused "
            f"{counted(capped, 'entry', 'entries')}"
        )
    return line


def announce(results: Mapping[str, Result]) -> None:
    """Issue, per strategy and named by it, the warnings `bt.backtest` ends a run of one
    strategy with: orders not placed for a reason the run did not ask for
    (OrdersNotPlacedWarning) and a wait inside the run for the indicators
    (WarmUpWarning)."""
    for name, result in results.items():
        announce_unplaced(result, strategy=name)
        announce_wait(result, strategy=name)


def run_portfolio(strategy: Any, client: Any, args: Mapping[str, Any]) -> PortfolioResult:
    """Run the list of strategies `strategy` on one account with the arguments of the
    `bt.backtest` call `args` (module docstring) and return its `PortfolioResult`. Raises
    ConfigError (`portfolio_plan`, and an account the fund refuses), the data layer's
    errors, NotImplementedError for anything a strategy alone would refuse, and the
    kernel's and the strategies' errors, a kernel error starting with its strategy's
    name. Every strategy's settings are checked before any strategy's data is loaded."""
    plan = portfolio_plan(strategy, args)
    checked = [
        check_run(cls, plan.arguments[name])
        for cls, name in zip(plan.strategies, plan.names, strict=True)
    ]
    checked, notes = own_margins(checked, args["margin"])
    runs = [
        load(each, client, plan.arguments[name])
        for each, name in zip(checked, plan.names, strict=True)
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
    return PortfolioResult(plan.account, results, records, benchmark=closes, run_warnings=notes)


__all__ = ["PortfolioPlan", "announce", "member_line", "portfolio_plan", "run_portfolio"]
