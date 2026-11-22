"""Per-strategy arguments by strategy class name (`_strategy_args`; 18-multi-strategy 1.1,
1.3, decision M5): one set of rules for `bt.backtest` given a list and for `bt.live`.

`SHARED` is one table of every per-strategy setting both verbs take (capital,
max_daily_loss, params, the risk settings after_risk_exit, max_positions, drawdown_halt,
cooldown, entry_from, entry_until and square_off, and the daily limits
max_trades_per_day, daily_profit_target and max_order_value of 18-multi-strategy 2.5a):
each row runs through `bt.backtest`'s list resolution (`_portfolio_run.portfolio_plan`,
before any data is loaded) and through `bt.live`'s argument checks
(`_live._plan_args.live_arguments`, before any call to OpenAlgo, read as the arguments of
each strategy's backtest of the day, which the session plan resolves its settings from),
and both give the row's values or refuse it with the row's words. `BACKTEST_ONLY` holds
the rows of the settings only a backtest takes (`product` and `base`, which a live
session refuses), and of the backtest's rule that a dict by name names every strategy:
`bt.live` checks a `max_daily_loss` dict that leaves a strategy out later, in its plan,
where it is mandatory for each strategy. Expected values are the rows' own inputs.
"""

from __future__ import annotations

import datetime as dt
import inspect
import re

import pytest

import openalgo.backtest as bt
from openalgo.backtest._live._plan_args import NOT_LIVE, live_arguments
from openalgo.backtest._portfolio_run import portfolio_plan
from openalgo.backtest._strategy_args import PER_STRATEGY, refuse_account_settings
from openalgo.backtest.errors import ConfigError


class Alpha(bt.Strategy):
    symbol, exchange, interval = "SBIN", "NSE", "1m"
    fast = 3


class Beta(bt.Strategy):
    symbol, exchange, interval = "SBIN", "NSE", "1m"
    fast = 3


NAMES = ("Alpha", "Beta")
CAPITAL = 100_000
#: The day and the present (UTC nanoseconds) of a live session's backtest arguments.
DAY = dt.date(2026, 11, 16)
PRESENT_NS = 1_763_284_500_000_000_000


def backtest_values(setting: str, given: dict) -> dict:
    """Each strategy's value of `setting` as `bt.backtest([Alpha, Beta], **given)` resolves
    it, from the arguments the call would pass on."""
    call = inspect.signature(bt.backtest).bind([Alpha, Beta], None, **given)
    call.apply_defaults()
    plan = portfolio_plan([Alpha, Beta], dict(call.arguments))
    return {name: plan.arguments[name][setting] for name in NAMES}


def live_values(setting: str, given: dict) -> dict:
    """Each strategy's value of `setting` as `bt.live([Alpha, Beta], **given)` checks it:
    in the arguments of the strategy's backtest of the day, which the session plan
    resolves its settings from."""
    checked = live_arguments([Alpha, Beta], **given)
    return {name: checked.backtest_arguments(name, DAY, PRESENT_NS)[setting] for name in NAMES}


# (row, setting, the call's arguments, each strategy's value or the refusal's words)
SHARED = [
    ("capital one value", "capital", {"capital": 50_000}, {"Alpha": 50_000, "Beta": 50_000}),
    (
        "capital by name",
        "capital",
        {"capital": {"Alpha": 50_000, "Beta": 70_000}},
        {"Alpha": 50_000, "Beta": 70_000},
    ),
    (
        "capital leaves a strategy out",
        "capital",
        {"capital": {"Alpha": 50_000}},
        r"capital is mandatory for every strategy, and the dict gives none for ['Beta']",
    ),
    (
        "capital names a strategy the call does not run",
        "capital",
        {"capital": {"Alpha": 50_000, "Beta": 50_000, "Gamma": 50_000}},
        r"capital names ['Gamma'], which this * does not run: key it by strategy class name",
    ),
    (
        "max_daily_loss one value",
        "max_daily_loss",
        {"capital": CAPITAL, "max_daily_loss": 2_000},
        {"Alpha": 2_000, "Beta": 2_000},
    ),
    (
        "max_daily_loss by name",
        "max_daily_loss",
        {"capital": CAPITAL, "max_daily_loss": {"Alpha": 2_000, "Beta": 3_000}},
        {"Alpha": 2_000, "Beta": 3_000},
    ),
    (
        "max_daily_loss names a strategy the call does not run",
        "max_daily_loss",
        {"capital": CAPITAL, "max_daily_loss": {"Alpha": 1, "Beta": 1, "Delta": 1}},
        r"max_daily_loss names ['Delta'], which this * does not run",
    ),
    (
        "params by name, one left on its defaults",
        "params",
        {"capital": CAPITAL, "params": {"Alpha": {"fast": 5}}},
        {"Alpha": {"fast": 5}, "Beta": None},
    ),
    (
        "params of one strategy given to two",
        "params",
        {"capital": CAPITAL, "params": {"fast": 5}},
        r"params names ['fast'], which this * does not run",
    ),
    (
        "params not a dict",
        "params",
        {"capital": CAPITAL, "params": 5},
        r"params takes a dict of parameter values, not 5",
    ),
    (
        "max_trades_per_day one value",
        "max_trades_per_day",
        {"capital": CAPITAL, "max_trades_per_day": 3},
        {"Alpha": 3, "Beta": 3},
    ),
    (
        "daily_profit_target by name",
        "daily_profit_target",
        {"capital": CAPITAL, "daily_profit_target": {"Alpha": 1_000, "Beta": 2_500}},
        {"Alpha": 1_000, "Beta": 2_500},
    ),
    (
        "max_order_value leaves a strategy out",
        "max_order_value",
        {"capital": CAPITAL, "max_order_value": {"Alpha": 50_000}},
        r"max_order_value as a dict must name every strategy of the run",
    ),
    (
        "max_trades_per_day names a strategy the call does not run",
        "max_trades_per_day",
        {"capital": CAPITAL, "max_trades_per_day": {"Alpha": 1, "Beta": 1, "Gamma": 1}},
        r"max_trades_per_day names ['Gamma'], which this * does not run",
    ),
    # after_risk_exit takes "lock_session" only in this version (its value is checked when
    # each strategy's settings are resolved, after these rules).
    (
        "after_risk_exit one value",
        "after_risk_exit",
        {"capital": CAPITAL, "after_risk_exit": "lock_session"},
        {"Alpha": "lock_session", "Beta": "lock_session"},
    ),
    (
        "after_risk_exit by name",
        "after_risk_exit",
        {"capital": CAPITAL, "after_risk_exit": {"Alpha": "lock_session", "Beta": "lock_session"}},
        {"Alpha": "lock_session", "Beta": "lock_session"},
    ),
    (
        "after_risk_exit leaves a strategy out",
        "after_risk_exit",
        {"capital": CAPITAL, "after_risk_exit": {"Alpha": "lock_session"}},
        "after_risk_exit as a dict must name every strategy of the run",
    ),
    (
        "max_positions by name",
        "max_positions",
        {"capital": CAPITAL, "max_positions": {"Alpha": 1, "Beta": 2}},
        {"Alpha": 1, "Beta": 2},
    ),
    (
        "drawdown_halt by name",
        "drawdown_halt",
        {"capital": CAPITAL, "drawdown_halt": {"Alpha": 10, "Beta": 15}},
        {"Alpha": 10, "Beta": 15},
    ),
    (
        "cooldown one value",
        "cooldown",
        {"capital": CAPITAL, "cooldown": 3},
        {"Alpha": 3, "Beta": 3},
    ),
    (
        "cooldown by name",
        "cooldown",
        {"capital": CAPITAL, "cooldown": {"Alpha": 3, "Beta": 5}},
        {"Alpha": 3, "Beta": 5},
    ),
    (
        "cooldown leaves a strategy out",
        "cooldown",
        {"capital": CAPITAL, "cooldown": {"Alpha": 3}},
        "cooldown as a dict must name every strategy of the run (give one value for all of "
        "them, or a value for each), and the dict gives none for ['Beta']",
    ),
    (
        "entry_from by name",
        "entry_from",
        {"capital": CAPITAL, "entry_from": {"Alpha": "09:30", "Beta": "10:00"}},
        {"Alpha": "09:30", "Beta": "10:00"},
    ),
    (
        "entry_until one value",
        "entry_until",
        {"capital": CAPITAL, "entry_until": "14:30"},
        {"Alpha": "14:30", "Beta": "14:30"},
    ),
    (
        "square_off by name",
        "square_off",
        {"capital": CAPITAL, "square_off": {"Alpha": "14:45", "Beta": "15:00"}},
        {"Alpha": "14:45", "Beta": "15:00"},
    ),
    (
        "square_off names a strategy the call does not run",
        "square_off",
        {"capital": CAPITAL, "square_off": {"Alpha": "15:00", "Beta": "15:00", "Omega": "15:00"}},
        r"square_off names ['Omega'], which this * does not run",
    ),
]


def _pattern(words: str) -> str:
    """The refusal's words as a pattern: literal, except `*` (the call's own noun)."""
    return ".*".join(re.escape(part) for part in words.split("*"))


@pytest.mark.parametrize("resolve", [backtest_values, live_values], ids=["backtest", "live"])
@pytest.mark.parametrize(
    ("setting", "given", "expected"),
    [row[1:] for row in SHARED],
    ids=[row[0] for row in SHARED],
)
def test_both_verbs_resolve_a_setting_by_strategy_name_alike(resolve, setting, given, expected):
    if isinstance(expected, str):
        with pytest.raises(ConfigError, match=_pattern(expected)):
            resolve(setting, given)
    else:
        assert resolve(setting, given) == expected


#: The per-strategy settings that are off for a strategy given None (no cap, no halt, no
#: cooldown, no daily limit): a dict that leaves a strategy out says how to leave it so.
OFF_WHEN_NONE = (
    "max_positions",
    "drawdown_halt",
    "cooldown",
    "max_trades_per_day",
    "daily_profit_target",
    "max_order_value",
)


@pytest.mark.parametrize("resolve", [backtest_values, live_values], ids=["backtest", "live"])
@pytest.mark.parametrize("setting", OFF_WHEN_NONE)
def test_a_dict_leaving_out_a_strategy_says_none_leaves_it_without_the_setting(resolve, setting):
    with pytest.raises(ConfigError) as refused:
        resolve(setting, {"capital": CAPITAL, setting: {"Alpha": 2}})
    assert str(refused.value) == (
        f"{setting} as a dict must name every strategy of the run (give one value for all of "
        f"them, or a value for each), and the dict gives none for ['Beta']: give None for a "
        f"strategy that has no {setting}"
    )


@pytest.mark.parametrize("resolve", [backtest_values, live_values], ids=["backtest", "live"])
def test_a_capital_dict_leaving_out_a_strategy_keeps_its_words(resolve):
    with pytest.raises(ConfigError) as refused:
        resolve("capital", {"capital": {"Alpha": 50_000}})
    assert str(refused.value) == (
        "capital is mandatory for every strategy, and the dict gives none for ['Beta']"
    )


@pytest.mark.parametrize(
    ("setting", "value"),
    [
        ("max_daily_loss", 2_000),
        ("entry_from", "09:30"),
        ("square_off", "15:00"),
        ("after_risk_exit", "lock_session"),
    ],
    ids=["max_daily_loss", "entry_from (None is the calendar's time)", "square_off", "after"],
)
def test_a_backtest_dict_of_a_setting_none_does_not_turn_off_keeps_its_words(setting, value):
    # max_daily_loss is mandatory live; None for the others is a default, not "none".
    with pytest.raises(ConfigError) as refused:
        backtest_values(setting, {"capital": CAPITAL, setting: {"Alpha": value}})
    assert str(refused.value) == (
        f"{setting} as a dict must name every strategy of the run (give one value for all of "
        "them, or a value for each), and the dict gives none for ['Beta']"
    )


@pytest.mark.parametrize("resolve", [backtest_values, live_values], ids=["backtest", "live"])
@pytest.mark.parametrize("setting", OFF_WHEN_NONE)
def test_none_in_a_dict_is_that_strategys_value(resolve, setting):
    # A run with None for one strategy's trade cap is in test_daily_limits.py.
    assert resolve(setting, {"capital": CAPITAL, setting: {"Alpha": None, "Beta": 2}}) == {
        "Alpha": None,
        "Beta": 2,
    }


def test_the_shared_table_holds_every_per_strategy_setting_but_those_refused_live():
    # product and base are refused live (bt.live takes no product= and no base=).
    shared = {row[1] for row in SHARED}
    assert shared == set(PER_STRATEGY) - {"product", "base"}
    assert {"product", "base"} <= set(NOT_LIVE)


BACKTEST_ONLY = [
    (
        "max_daily_loss leaves a strategy out",
        "max_daily_loss",
        {"max_daily_loss": {"Alpha": 2_000}},
        "max_daily_loss as a dict must name every strategy of the run",
    ),
    ("product by name", "product", {"product": {"Alpha": "MIS", "Beta": "CNC"}}, None),
    ("base by name", "base", {"base": {"Alpha": "1m", "Beta": "5m"}}, None),
]


@pytest.mark.parametrize(
    ("setting", "given", "expected"),
    [row[1:] for row in BACKTEST_ONLY],
    ids=[row[0] for row in BACKTEST_ONLY],
)
def test_a_backtest_resolves_its_other_settings_by_strategy_name(setting, given, expected):
    arguments = {"capital": CAPITAL, **given}
    if isinstance(expected, str):
        with pytest.raises(ConfigError, match=re.escape(expected)):
            backtest_values(setting, arguments)
        return
    # None: the value given by name is each strategy's.
    expected = expected or given[setting]
    assert backtest_values(setting, arguments) == expected


@pytest.mark.parametrize("verb", [bt.backtest, bt.live], ids=["backtest", "live"])
@pytest.mark.parametrize(
    ("strategies", "words"),
    [
        ([], "takes a bt.Strategy subclass, or a list of them"),
        ([Alpha, 5], "takes bt.Strategy subclasses, not 5: pass the class itself"),
    ],
    ids=["empty list", "not a class"],
)
def test_both_verbs_refuse_a_list_that_is_not_of_strategy_classes(verb, strategies, words):
    with pytest.raises(ConfigError, match=re.escape(words)):
        verb(strategies, None, capital=CAPITAL, max_daily_loss=1_000)


def test_a_backtest_refuses_two_strategies_of_one_name():
    with pytest.raises(ConfigError, match=r"bt.backtest runs two strategies named Alpha"):
        bt.backtest([Alpha, Alpha], capital=CAPITAL)


def test_a_live_session_still_refuses_two_strategies_of_one_name_as_not_supported():
    with pytest.raises(NotImplementedError, match=r"two strategies with the same name \(Alpha\)"):
        live_arguments([Alpha, Alpha], capital=CAPITAL)


@pytest.mark.parametrize(
    ("given", "named"),
    [
        ({"pool": "separate"}, "pool="),
        ({"account_capital": 500_000}, "account_capital="),
        ({"account_max_daily_loss": 5_000}, "account_max_daily_loss="),
        ({"pool": "separate", "account_capital": 1.0}, "pool=, account_capital="),
    ],
)
def test_a_run_of_one_strategy_refuses_the_account_settings_of_a_list(given, named):
    with pytest.raises(ConfigError, match=re.escape(f"{named} set the account of a list")):
        refuse_account_settings(given)


def test_a_run_of_one_strategy_takes_the_account_settings_at_their_defaults():
    refuse_account_settings({"pool": "shared", "account_capital": None})
    refuse_account_settings({})


def test_a_list_of_one_strategy_takes_its_own_parameter_values_in_both_verbs():
    # As a class does: a list of one is that strategy's run, so params are its own values.
    call = inspect.signature(bt.backtest).bind([Alpha], None, capital=CAPITAL, params={"fast": 5})
    call.apply_defaults()
    plan = portfolio_plan([Alpha], dict(call.arguments))
    assert plan.arguments["Alpha"]["params"] == {"fast": 5}
    checked = live_arguments([Alpha], capital=CAPITAL, params={"fast": 5})
    assert dict(checked.params) == {"Alpha": {"fast": 5}}
