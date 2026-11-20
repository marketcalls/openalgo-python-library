"""Per-strategy arguments by strategy class name (`_strategy_args`; 18-multi-strategy 1.1,
1.3, decision M5): one set of rules for `bt.backtest` given a list and for `bt.live`.

`SHARED` is one table of the settings both verbs take by name today (capital,
max_daily_loss, params): each row runs through `bt.backtest`'s list resolution
(`_portfolio_run.portfolio_plan`, before any data is loaded) and through `bt.live`'s
argument checks (`_live._plan_args.live_arguments`, before any call to OpenAlgo), and both
give the row's values or refuse it with the row's words. `BACKTEST_ONLY` holds the rows
of the settings only a backtest takes by name in this version, and of the backtest's
rule that a dict by name names every strategy: `bt.live` checks a `max_daily_loss` dict
that leaves a strategy out later, in its plan, where it is mandatory for each strategy.
Expected values are the rows' own inputs.
"""

from __future__ import annotations

import inspect
import re

import pytest

import openalgo.backtest as bt
from openalgo.backtest._live._plan_args import live_arguments
from openalgo.backtest._portfolio_run import portfolio_plan
from openalgo.backtest._strategy_args import refuse_account_settings
from openalgo.backtest.errors import ConfigError


class Alpha(bt.Strategy):
    symbol, exchange, interval = "SBIN", "NSE", "1m"
    fast = 3


class Beta(bt.Strategy):
    symbol, exchange, interval = "SBIN", "NSE", "1m"
    fast = 3


NAMES = ("Alpha", "Beta")
CAPITAL = 100_000


def backtest_values(setting: str, given: dict) -> dict:
    """Each strategy's value of `setting` as `bt.backtest([Alpha, Beta], **given)` resolves
    it, from the arguments the call would pass on."""
    call = inspect.signature(bt.backtest).bind([Alpha, Beta], None, **given)
    call.apply_defaults()
    plan = portfolio_plan([Alpha, Beta], dict(call.arguments))
    return {name: plan.arguments[name][setting] for name in NAMES}


def live_values(setting: str, given: dict) -> dict:
    """Each strategy's value of `setting` as `bt.live([Alpha, Beta], **given)` checks it."""
    checked = live_arguments([Alpha, Beta], **given)
    return dict(getattr(checked, setting))


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


BACKTEST_ONLY = [
    ("cooldown one value", "cooldown", {"cooldown": 3}, {"Alpha": 3, "Beta": 3}),
    (
        "cooldown by name",
        "cooldown",
        {"cooldown": {"Alpha": 3, "Beta": 5}},
        {"Alpha": 3, "Beta": 5},
    ),
    (
        "cooldown leaves a strategy out",
        "cooldown",
        {"cooldown": {"Alpha": 3}},
        "cooldown as a dict must name every strategy of the run (give one value for all of "
        "them, or a value for each), and the dict gives none for ['Beta']",
    ),
    (
        "max_daily_loss leaves a strategy out",
        "max_daily_loss",
        {"max_daily_loss": {"Alpha": 2_000}},
        "max_daily_loss as a dict must name every strategy of the run",
    ),
    ("product by name", "product", {"product": {"Alpha": "MIS", "Beta": "CNC"}}, None),
    ("base by name", "base", {"base": {"Alpha": "1m", "Beta": "5m"}}, None),
    (
        "square_off names a strategy the call does not run",
        "square_off",
        {"square_off": {"Alpha": "15:00", "Beta": "15:00", "Omega": "15:00"}},
        "square_off names ['Omega'], which this backtest does not run",
    ),
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
