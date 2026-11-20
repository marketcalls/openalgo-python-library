"""What a live session refuses before anything is called, and says what to do instead
(`_live/plan.py` `specify`, `_plan_args.py`, `_plan_strategies.py`; 16-session-design 1.1,
1.2, 3.1 step 1 and decisions 3, 5 and 9 of its section 10.0, unit 7).

`specify` runs on a clock at 10:15 IST of Monday 2026-11-16 and contacts nothing: each
strategy's `init` runs on the engine's reference bars only.
"""

from __future__ import annotations

import inspect
import re

import pytest
from plan_samples import DAY, LIMITS, EmaTrend, ist, specify

import openalgo.backtest as bt
from openalgo import ta
from openalgo.backtest import run
from openalgo.backtest._live import plan as plan_module
from openalgo.backtest._live._plan_args import MODE_ARGUMENTS, NOT_LIVE
from openalgo.backtest.errors import ConfigError
from openalgo.backtest.reference import default_tables


class Crossing(bt.Strategy):
    """A condition strategy: its rules are declared in init."""

    symbol, exchange, interval = "SBIN", "NSE", "5m"

    def init(self):
        fast, slow = self.I(ta.ema, self.close, 10), self.I(ta.ema, self.close, 20)
        self.entry_when(fast.crossed_above(slow))
        self.exit_when(fast.crossed_below(slow))


def subclass(name: str = "Variant", base: type = EmaTrend, **attributes) -> type:
    """A strategy class derived from `base` with class attributes changed."""
    return type(name, (base,), attributes)


def test_a_condition_strategy_is_refused_pointing_to_on_bar():
    with pytest.raises(
        NotImplementedError, match="condition strategy .* write its rules in on_bar"
    ):
        specify(Crossing)


@pytest.mark.parametrize("name", sorted(MODE_ARGUMENTS))
def test_an_argument_that_chooses_the_mode_is_refused_naming_the_analyzer_switch(name):
    with pytest.raises(
        ConfigError, match=f"takes no {name}=: sandbox or live is chosen inside OpenAlgo"
    ):
        specify(**{name: True})


# The bt.backtest arguments 16-session-design 1.2 lists as having no meaning live, with
# tz and corporate_actions, which describe data of the user's own.
NOT_LIVE_NAMES = (
    "data", "start", "end", "now", "source", "base", "fill", "delay_entries",
    "delay_exits", "slippage", "margin",
    "quantity_mode", "instrument", "product", "roll", "benchmark", "calendar", "tz",
    "corporate_actions", "pool", "account_capital",
)  # fmt: skip


@pytest.mark.parametrize("name", NOT_LIVE_NAMES)
def test_a_backtest_argument_with_no_meaning_live_is_refused_naming_it(name):
    with pytest.raises(ConfigError, match=f"bt.live takes no {name}=: .*; remove {name}="):
        specify(**{name: None})


def test_every_backtest_argument_refused_live_says_why():
    assert sorted(NOT_LIVE) == sorted(NOT_LIVE_NAMES)
    with pytest.raises(ConfigError, match="margin=: the broker computes the margin"):
        specify(margin=bt.pct(20))


def test_an_unknown_argument_is_refused_listing_the_arguments_bt_live_takes():
    with pytest.raises(ConfigError, match="bt.live has no argument capitol=; it takes .*capital"):
        specify(capitol=1_000_000)


def test_bars_from_broker_candles_are_not_built_yet():
    with pytest.raises(NotImplementedError, match="bars='candles'.*use bars='websocket'"):
        specify(bars="candles")


def test_bars_from_the_feeds_ticks_are_the_bars_a_session_runs_on():
    assert specify(bars="websocket").strategies[0].name == "EmaTrend"


@pytest.mark.parametrize(
    ("attributes", "words"),
    [
        ({"symbol": None, "symbols": ("SBIN", "INFY")}, "several symbols .* strategy of its own"),
        ({"symbol": None, "universe": "NIFTY50"}, "a universe .* strategy of its own"),
        (
            {"symbol": None, "underlying": "NIFTY", "expiry": "current_month", "exchange": "NFO"},
            "futures series .* name the contract as the strategy's symbol",
        ),
        ({"interval": "D"}, "a D strategy interval .* trade an intraday interval"),
        ({"product": "CNC"}, "product CNC .* GTT orders.*trade MIS"),
        ({"product": "NRML"}, "product NRML .* GTT orders.*trade MIS"),
    ],
)
def test_a_strategy_a_live_session_does_not_run_is_refused_saying_what_to_do(attributes, words):
    with pytest.raises(NotImplementedError, match=words):
        specify(subclass(**attributes))


def test_the_contract_a_futures_series_refusal_gives_as_an_example_is_a_valid_symbol():
    series = subclass(symbol=None, underlying="NIFTY", expiry="current_month", exchange="NFO")
    with pytest.raises(NotImplementedError) as refused:
        specify(series)
    example = re.search(r"for example (\S+) on NFO", str(refused.value)).group(1)
    assert default_tables().instrument(example, "NFO", DAY).kind == "future"


def test_a_strategy_from_arrays_is_refused_pointing_to_a_strategy_class():
    arrays = bt.from_arrays([True], [False], symbol="SBIN", exchange="NSE", interval="5m")
    with pytest.raises(NotImplementedError, match="bt.from_arrays strategy .* decides in on_bar"):
        specify(arrays)


def test_an_interval_that_1_minute_slices_cannot_build_is_refused():
    with pytest.raises(ConfigError, match="30s bars cannot be built from the run's 1m base bars"):
        specify(subclass(interval="30s"))


def test_two_strategies_with_the_same_name_are_refused():
    first, second = subclass("Twin"), subclass("Twin", symbol="INFY")
    with pytest.raises(NotImplementedError, match=r"two strategies with the same name \(Twin\)"):
        specify([first, second])


def test_two_strategies_on_one_symbol_are_refused():
    with pytest.raises(
        NotImplementedError, match="two strategies on NSE:SBIN .* one strategy per symbol"
    ):
        specify([EmaTrend, subclass("Other")])


def test_several_strategies_run_as_a_list_each_on_its_own_symbol():
    spec = specify([EmaTrend, subclass("OnInfy", symbol="INFY")])
    assert spec.instruments == (("SBIN", "NSE"), ("INFY", "NSE"))
    assert [each.name for each in spec.strategies] == ["EmaTrend", "OnInfy"]


def test_capital_and_the_daily_loss_limit_may_be_given_by_strategy_name():
    other = subclass("OnInfy", symbol="INFY")
    spec = specify(
        [EmaTrend, other],
        capital={"EmaTrend": 300_000, "OnInfy": 200_000},
        max_daily_loss={"EmaTrend": 3_000, "OnInfy": bt.rupees(2_000)},
    )
    assert [(each.capital, each.max_daily_loss) for each in spec.strategies] == [
        (300_000, 3_000),
        (200_000, 2_000),
    ]
    assert (spec.capital, spec.max_daily_loss) == (500_000, 5_000)


def test_a_capital_dict_that_leaves_a_strategy_out_is_refused():
    with pytest.raises(ConfigError, match=r"capital is mandatory for every strategy.*\['OnInfy'\]"):
        specify([EmaTrend, subclass("OnInfy", symbol="INFY")], capital={"EmaTrend": 300_000})


def test_a_dict_keyed_by_a_strategy_the_session_does_not_run_is_refused():
    with pytest.raises(ConfigError, match=r"max_daily_loss names \['Other'\]"):
        specify(max_daily_loss={"Other": 1_000})


def test_a_strategy_with_no_daily_loss_limit_is_refused():
    with pytest.raises(
        ConfigError, match="EmaTrend: max_daily_loss is mandatory in a live session"
    ):
        specify(max_daily_loss={})


def test_a_daily_loss_limit_set_only_on_the_class_is_refused_live():
    with pytest.raises(ConfigError, match="Variant: max_daily_loss is mandatory in a live session"):
        specify(subclass(max_daily_loss=1_500), max_daily_loss={})


def test_a_class_limit_that_differs_from_the_argument_is_refused():
    with pytest.raises(ConfigError, match="differs from the class attribute"):
        specify(subclass(max_daily_loss=1_500), max_daily_loss=5_000)


def test_a_class_limit_equal_to_the_argument_is_the_limit():
    spec = specify(subclass(max_daily_loss=1_500), max_daily_loss=bt.rupees(1_500))
    assert spec.strategies[0].max_daily_loss == 1_500


def test_the_daily_loss_limit_takes_rupees_only():
    with pytest.raises(ConfigError, match="max_daily_loss takes a positive amount of rupees"):
        specify(max_daily_loss=bt.pct(2))


def test_with_several_strategies_params_are_given_by_strategy_name():
    class Tuned(EmaTrend):
        length = 10

        def init(self):
            self.fast = self.I(ta.ema, self.close, self.length)

    other = subclass("OnInfy", base=Tuned, symbol="INFY")
    spec = specify([Tuned, other], params={"OnInfy": {"length": 20}})
    assert [each.params for each in spec.strategies] == [None, {"length": 20}]


@pytest.mark.parametrize("on_stop", ["close", None, "Flatten"])
def test_on_stop_takes_flatten_or_keep(on_stop):
    with pytest.raises(ConfigError, match="on_stop takes 'flatten' .* or 'keep'"):
        specify(on_stop=on_stop)


def test_the_account_defaults_to_the_broker_name_ping_reports():
    assert specify().account_name("zerodha") == "zerodha"


def test_an_account_given_names_the_journal_logs_and_locks():
    assert specify(account="zerodha-family").account_name("zerodha") == "zerodha-family"


def test_the_arguments_shared_with_bt_backtest_take_its_defaults():
    plan = specify().strategies[0].plan
    assert (plan.costs, plan.fill, plan.quantity_mode, plan.risk["risk_basis"]) == (
        "zerodha",
        "next_open",
        "venue",
        "gross",
    )


def test_the_preflight_is_asked_for_every_strategys_instrument_interval_and_limits():
    spec = specify(capital=LIMITS["capital"])
    assert (spec.instruments, spec.intervals) == ((("SBIN", "NSE"),), ("5m",))
    assert (spec.capital, spec.max_daily_loss) == (1_000_000, 5_000)


def specify_without(name: str):
    """`plan.specify` of EmaTrend with the session's limits but `name` left out."""
    given = {key: value for key, value in LIMITS.items() if key != name}
    return plan_module.specify(EmaTrend, clock=lambda: ist(10, 15), **given)


def test_without_capital_bt_live_is_refused_naming_the_argument():
    with pytest.raises(ConfigError, match="bt.live needs capital=: the rupees allocated"):
        specify_without("capital")


def test_without_a_daily_loss_limit_on_the_call_or_the_class_bt_live_is_refused():
    with pytest.raises(
        ConfigError, match="EmaTrend: max_daily_loss is mandatory in a live session; pass"
    ):
        specify_without("max_daily_loss")


def test_a_class_limit_with_no_argument_at_all_is_refused():
    with pytest.raises(ConfigError, match="a limit set only on the class is not taken"):
        plan_module.specify(
            subclass(max_daily_loss=1_500), clock=lambda: ist(10, 15), capital=LIMITS["capital"]
        )


def test_a_default_shared_with_bt_backtest_follows_bt_backtests_own(monkeypatch):
    signature = inspect.signature(run.backtest)
    changed = signature.replace(
        parameters=[
            parameter.replace(default="net") if parameter.name == "risk_basis" else parameter
            for parameter in signature.parameters.values()
        ]
    )

    def backtest(*args, **kwargs):
        raise AssertionError("a live session's plan never runs a backtest")

    backtest.__signature__ = changed
    monkeypatch.setattr(run, "backtest", backtest)
    assert specify().strategies[0].plan.risk["risk_basis"] == "net"
