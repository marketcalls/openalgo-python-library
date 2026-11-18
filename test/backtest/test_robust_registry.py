"""The seventeen checks and the request of a robustness report: the checks in the order of
design 17 section 2, the nine this version runs (the four trade-list checks and the five
bar checks), the error for one it does not (saying what it needs, in plain words), and
every setting checked before work starts. The checks load only when they are used."""

import subprocess
import sys

import pytest
from robust_runs import FREE, Flip, bars, steps_for

import openalgo.backtest as bt
from openalgo.backtest._robust._registry import BUILT, CHECKS, QUICK
from openalgo.backtest._robust._request import build_request
from openalgo.backtest.errors import ConfigError

# Design 17, section 2, in order.
DESIGN_ORDER = (
    "out_of_sample",
    "random_out_of_sample",
    "walk_forward",
    "vs_random",
    "permutation",
    "noise",
    "shifted",
    "other_markets",
    "stress",
    "reshuffle",
    "resample",
    "random_exits",
    "variance",
    "delayed",
    "liquidity",
    "parameters",
    "noise_optimization",
)
ENGINE = "late orders, other fill prices and shifted bar boundaries in the engine"
SEARCHES = "walk-forward and the parameter searches"
# What each check this version does not run needs (section 6), in plain words.
NEEDS = {
    "out_of_sample": SEARCHES,
    "walk_forward": SEARCHES,
    "shifted": ENGINE,
    "random_exits": ENGINE,
    "delayed": ENGINE,
    "liquidity": ENGINE,
    "parameters": SEARCHES,
    "noise_optimization": SEARCHES,
}
TRADE_LIST = ("random_out_of_sample", "reshuffle", "resample", "variance")
THIS_VERSION = (
    "random_out_of_sample",
    "vs_random",
    "permutation",
    "noise",
    "other_markets",
    "stress",
    "reshuffle",
    "resample",
    "variance",
)


def request(**changes):
    settings = {
        "checks": "all",
        "objective": "sharpe",
        "runs": 1_000,
        "seed": 0,
        "workers": None,
        "others": None,
        "check_settings": {},
        "ranges": {},
    }
    return build_request(**{**settings, **changes})


def robustness(**settings):
    return bt.robustness(Flip, data=bars(steps_for([1.0, -1.0], 10)), **FREE, **settings)


def test_the_registry_lists_the_seventeen_checks_in_the_design_order():
    assert tuple(check.name for check in CHECKS) == DESIGN_ORDER
    assert [check.number for check in CHECKS] == list(range(1, 18))


def test_this_version_runs_the_trade_list_and_the_bar_checks():
    assert BUILT == THIS_VERSION
    assert QUICK == TRADE_LIST


@pytest.mark.parametrize("name", list(NEEDS))
def test_a_check_this_version_does_not_run_says_what_it_needs(name):
    with pytest.raises(NotImplementedError) as raised:
        request(checks=[name])
    message = str(raised.value)
    assert "not supported in this version" in message
    assert f"it needs {NEEDS[name]}" in message
    assert "checks='all'" in message


def test_all_lists_every_check_and_runs_the_built_ones():
    asked = request()
    assert tuple(check.name for check in asked.listed) == DESIGN_ORDER
    assert tuple(check.name for check in asked.to_run) == THIS_VERSION


def test_quick_runs_the_trade_list_checks():
    assert tuple(check.name for check in request(checks="quick").listed) == TRADE_LIST


def test_a_list_of_checks_runs_them_in_report_order():
    asked = request(checks=["variance", "resample"])
    assert [check.name for check in asked.to_run] == ["resample", "variance"]


@pytest.mark.parametrize("checks", [["bogus"], [], "some", 3, [3]])
def test_checks_that_name_no_check_are_refused(checks):
    with pytest.raises(ConfigError):
        request(checks=checks)


def test_a_checks_dict_applies_over_its_defaults():
    asked = request(check_settings={"variance": {"points": 10}})
    assert asked.settings["variance"] == {
        "runs": None,
        "min_trades": 30,
        "ruin": 0.5,
        "points": 10.0,
        "pass_share": 0.8,
    }


def test_an_unknown_key_of_a_checks_dict_names_the_keys_it_takes():
    with pytest.raises(ConfigError) as raised:
        request(check_settings={"variance": {"point": 10}})
    assert str(raised.value) == (
        "variance= has no setting ['point']; it takes ['runs', 'min_trades', 'ruin', "
        "'points', 'pass_share']"
    )


@pytest.mark.parametrize(
    ("check", "settings"),
    [
        ("variance", {"points": 150}),
        ("variance", {"points": 0}),
        ("resample", {"pass_share": 1.5}),
        ("resample", {"runs": 0}),
        ("reshuffle", {"max_drawdown": -0.1}),
        ("random_out_of_sample", {"oos": 1.0}),
        ("random_out_of_sample", {"min_trades": True}),
        ("resample", {"ruin": None}),
    ],
)
def test_a_setting_out_of_its_range_is_refused_with_what_it_takes(check, settings):
    with pytest.raises(ConfigError, match="takes"):
        request(check_settings={check: settings})


def test_none_keeps_a_setting_whose_default_is_none():
    asked = request(check_settings={"reshuffle": {"max_drawdown": None, "runs": None}})
    assert asked.settings["reshuffle"]["max_drawdown"] is None


def test_settings_for_a_check_that_does_not_run_are_refused():
    with pytest.raises(ConfigError, match="which checks= does not run"):
        request(checks=["resample"], check_settings={"variance": {"points": 10}})


def test_settings_for_a_check_this_version_does_not_run_say_what_it_needs():
    with pytest.raises(NotImplementedError, match=r"it needs late orders"):
        request(check_settings={"delayed": {"runs": 500}})


def test_others_are_symbols_or_symbol_and_exchange_pairs():
    asked = request(others=["ICICIBANK", ("HDFCBANK", "BSE")])
    assert asked.others == (("ICICIBANK", None), ("HDFCBANK", "BSE"))


@pytest.mark.parametrize("others", ["ICICIBANK", [], [3], [("HDFCBANK",)], [("A", "")]])
def test_others_that_name_no_symbol_are_refused(others):
    with pytest.raises(ConfigError, match="others takes"):
        request(others=others)


def test_others_without_the_other_markets_check_are_refused():
    with pytest.raises(ConfigError, match="which checks= does not run"):
        request(checks="quick", others=["ICICIBANK"])


def test_parameter_ranges_are_not_searched_in_this_version():
    with pytest.raises(NotImplementedError, match="they need walk-forward and the parameter"):
        request(ranges={"fast": range(5, 30, 5)})


@pytest.mark.parametrize(
    ("name", "value"),
    [("runs", 0), ("runs", 2.5), ("seed", -1), ("seed", 2**63), ("workers", 0), ("runs", True)],
)
def test_runs_seed_and_workers_take_whole_numbers(name, value):
    with pytest.raises(ConfigError, match=f"{name} takes a whole number"):
        request(**{name: value})


def test_an_objective_that_is_not_a_metric_is_refused():
    with pytest.raises(ConfigError, match="objective"):
        request(objective="happiness")


def test_bt_robustness_refuses_a_keyword_that_is_nothing_it_takes():
    with pytest.raises(ConfigError, match="neither a check"):
        robustness(colour="blue")


def test_bt_robustness_refuses_a_settings_value_that_is_not_a_dict():
    with pytest.raises(ConfigError, match="takes a dict of its settings"):
        robustness(variance=10)


def test_bt_robustness_takes_a_strategy_class():
    with pytest.raises(ConfigError, match="bt.Strategy subclass"):
        bt.robustness(Flip(), data=bars(steps_for([1.0], 4)), **FREE)


def test_importing_the_backtest_package_does_not_load_the_checks():
    # The checks, their generators and their charts load when a report needs them.
    code = (
        "import sys, openalgo.backtest as bt; bt.Strategy; bt.metrics; "
        "loaded = [m for m in sys.modules if m.startswith('openalgo.backtest._robust')]; "
        "assert not loaded, loaded"
    )
    run = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
