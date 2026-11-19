"""The intervals of every strategy of a session in one preflight (16-session-design 3.1
step 7): each interval named is checked once, with the 1-minute candles the live bars are
checked against."""

from __future__ import annotations

import pytest
from preflight_fakes import NOW_NS, FakeOpenAlgo, make_plan

from openalgo.backtest._live.preflight import run_preflight
from openalgo.backtest.errors import ConfigError, PreflightError

OFFERED = "1m, 3m, 5m, 15m, 30m, 60m, D"


def interval_message(**changes) -> str:
    """The message of the interval check of `make_plan(**changes)`, passed or failed."""
    try:
        report = run_preflight(FakeOpenAlgo(), make_plan(**changes), now=lambda: NOW_NS)
    except PreflightError as error:
        (failure,) = error.failures
        return failure
    (check,) = [check for check in report.checks if check.name == "interval"]
    return check.message


def test_every_strategys_interval_is_checked_with_the_1m_candles():
    message = interval_message(interval=None, intervals=["5m", "15m"])

    assert message == (
        "the broker offers 5m and 15m and 1m candles (the strategies' intervals, and 1m to "
        "check the live bars against)"
    )


def test_an_interval_named_twice_is_checked_once():
    message = interval_message(interval="5m", intervals=["5m", "15m"])

    assert message.startswith("the broker offers 5m and 15m and 1m candles")


def test_a_strategys_interval_the_broker_does_not_offer_fails_naming_it():
    message = interval_message(interval=None, intervals=["5m", "2m"])

    assert message == (
        f"the broker offers no 2m candles (it offers {OFFERED}): choose one of those as each "
        "strategy's interval"
    )


def test_each_interval_written_wrong_is_reported():
    message = interval_message(interval=None, intervals=["5min", ""])

    assert message.startswith("unknown interval '5min'")
    assert "; interval must be a string such as '5m' or 'D', got ''" in message


def test_a_plan_keeps_its_intervals_as_a_tuple_and_names_each_once():
    plan = make_plan(interval="5m", intervals=["15m", "5m"])

    assert (plan.intervals, plan.bar_intervals) == (("15m", "5m"), ("5m", "15m"))


def test_a_plan_that_names_no_interval_is_refused():
    with pytest.raises(ConfigError, match="name the bar interval to check"):
        make_plan(interval=None)


@pytest.mark.parametrize("intervals", ["5m", None, {"5m"}], ids=["text", "none", "set"])
def test_intervals_that_are_not_a_list_are_refused(intervals):
    with pytest.raises(ConfigError, match="intervals is a list"):
        make_plan(intervals=intervals)
