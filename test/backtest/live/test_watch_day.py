"""A live session's day with the risk watch running, on the fake OpenAlgo
(16-session-design 3.2, 3.4 and 7.2 rows F87 and F88, units 8 and 11): a broker not
logged in before the open, and two strategies on two exchanges squared off each at its
own time.

- F87: the session starts at 08:58:00 IST with the broker's login gone (its login
  expired, or only its funds answered as zero while `ping` works); the user logs in at
  09:05; SBIN trades every 10 s from then. The preflight runs again each minute until
  the login is back, the alert test is sent once, an ERROR alert goes at 09:00, and the
  session trades from the open (09:15); the feed is never taken for dead before it.
- F88: SBIN on NSE (square-off 15:00) and a NIFTY call on NFO (square-off 15:10), each
  held by its own strategy from its first decision; the session ends once both are
  flat.
"""

from __future__ import annotations

import logging

import pytest
from fake_openalgo import rest as fake_rest
from fake_openalgo.fixtures import *  # noqa: F403
from plan_samples import candles
from session_samples import NS, TRADE_EVERY_S, BuyAndHold, Hold, ist, rig
from watch_samples import OPTION, OPTION_LOT, AlertTimes, HoldOption, calls_between, events, run

from openalgo.backtest._live._types import LOGGER_NAME

LOGIN_AT = ist(9, 5)


@pytest.fixture
def alert_times(make_fake_openalgo):
    """The fake's time of every alert, kept by a handler on the live logger."""
    handlers: list[AlertTimes] = []

    def attach(world) -> AlertTimes:
        handler = AlertTimes(world.fake.clock.now)
        logging.getLogger(LOGGER_NAME).addHandler(handler)
        handlers.append(handler)
        return handler

    yield attach
    for handler in handlers:
        logging.getLogger(LOGGER_NAME).removeHandler(handler)


def before_the_open(make_fake_openalgo, monkeypatch, tmp_path):
    """The F87 world: started at 08:58:00 with no trade yet; at 09:05 SBIN starts trading
    every 10 s; stopped at 09:17."""
    world = rig(make_fake_openalgo, monkeypatch, tmp_path, start=(8, 58, 0), trades_until=None)

    def trade_from_now():
        world.script.trade()
        world.script.trades(LOGIN_AT + TRADE_EVERY_S * NS, ist(9, 30))

    world.script.at(LOGIN_AT, trade_from_now)
    world.stop_at(ist(9, 17))
    return world


@pytest.fixture
def expired(make_fake_openalgo, monkeypatch, tmp_path, caplog):
    """F87: the broker's login expired; it is restored at 09:05."""
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    world = before_the_open(make_fake_openalgo, monkeypatch, tmp_path)
    world.fake.faults.expire_session()
    world.script.at(LOGIN_AT, world.fake.faults.restore_session)
    return world


@pytest.fixture
def zero_funds(make_fake_openalgo, monkeypatch, tmp_path, caplog):
    """F87's variant: until 09:05 the broker answers its funds as zero, `ping` and the
    rest working."""
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    world = before_the_open(make_fake_openalgo, monkeypatch, tmp_path)
    funds = fake_rest.ROUTES["funds"]

    def zero_until_login(venue, payload):
        answer, status = funds(venue, payload)
        if venue.clock.now() < LOGIN_AT:
            answer = {**answer, "data": {**answer["data"], "availablecash": "0.00"}}
        return answer, status

    monkeypatch.setitem(fake_rest.ROUTES, "funds", zero_until_login)
    return world


@pytest.mark.parametrize("login", ["expired", "zero_funds"])
def test_F87_before_the_open_the_preflight_runs_again_each_minute_until_the_login(login, request):
    world = request.getfixturevalue(login)
    run(world, BuyAndHold)
    checks = [call.ts_ns for call in calls_between(world, "funds", ist(8, 58), LOGIN_AT + NS)]
    assert checks == [ist(8, 58) + minute * 60 * NS for minute in range(8)]


@pytest.mark.parametrize("login", ["expired", "zero_funds"])
def test_F87_the_alert_test_is_sent_once(login, request, caplog):
    world = request.getfixturevalue(login)
    run(world, BuyAndHold)
    tests = [record for record in caplog.records if record.getMessage().startswith("test alert")]
    assert len(tests) == 1


@pytest.mark.parametrize("login", ["expired", "zero_funds"])
def test_F87_a_broker_still_not_logged_in_at_09_00_is_alerted_then(login, request, alert_times):
    world = request.getfixturevalue(login)
    times = alert_times(world)
    run(world, BuyAndHold)
    assert times.of("error", "not logged in") == [ist(9, 0)]


@pytest.mark.parametrize("login", ["expired", "zero_funds"])
def test_F87_once_logged_in_the_session_trades_from_the_open(login, request):
    world = request.getfixturevalue(login)
    outcome = run(world, BuyAndHold)
    assert outcome.stopped_by == "ctrl_c"
    assert [order.action for order in world.orders()] == ["BUY", "SELL"]
    assert all(order.placed_ns > ist(9, 15) for order in world.orders())


@pytest.mark.parametrize("login", ["expired", "zero_funds"])
def test_F87_the_quiet_market_before_the_open_is_no_dead_feed(login, request, caplog):
    world = request.getfixturevalue(login)
    outcome = run(world, BuyAndHold)
    assert events(outcome, "feed_dead") == []


# -- F88: NSE and NFO square-off times ---------------------------------------------------------


@pytest.fixture
def two_exchanges(make_fake_openalgo, monkeypatch, tmp_path, caplog):
    """SBIN (NSE) and the NIFTY call (NFO) trade every 10 s from 14:40:05; SBIN until
    15:02, the call at 120.00 until 15:12."""
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    world = rig(make_fake_openalgo, monkeypatch, tmp_path)
    fake = world.fake
    fake.add_instrument(
        OPTION,
        "NFO",
        lot_size=OPTION_LOT,
        freeze_qty=1800,
        instrument_type="CE",
        expiry="01-DEC-26",
        strike=26000.0,
    )
    fake.seed_candles(OPTION, "NFO", candles(today_until=(14, 40)))

    def trade_option():
        fake.trade(OPTION, "NFO", 120.0, volume=float(OPTION_LOT))

    trade_option()
    for instant in range(ist(14, 40, 15), ist(15, 12), TRADE_EVERY_S * NS):
        world.script.at(instant, trade_option)
    return world


def square_off(world, symbol: str) -> int:
    """When the sale of `symbol` was placed."""
    [sell] = [
        order for order in world.orders() if order.symbol == symbol and order.action == "SELL"
    ]
    return sell.placed_ns


def test_F88_sbin_is_squared_off_at_15_00(two_exchanges):
    run(two_exchanges, [Hold, HoldOption])
    assert ist(15, 0) <= square_off(two_exchanges, "SBIN") < ist(15, 0, 5)


def test_F88_the_option_is_squared_off_at_15_10(two_exchanges):
    run(two_exchanges, [Hold, HoldOption])
    assert ist(15, 10) <= square_off(two_exchanges, OPTION) < ist(15, 10, 5)


def test_F88_the_session_ends_once_both_are_flat(two_exchanges):
    outcome = run(two_exchanges, [Hold, HoldOption])
    assert outcome.stopped_by == "session_end"
    assert ist(15, 10) <= outcome.ended_ns < ist(15, 10, 10)
    assert (
        two_exchanges.fake.position("SBIN", "NSE"),
        two_exchanges.fake.position(OPTION, "NFO"),
    ) == (
        0,
        0,
    )


OPTIONS_ONLY = (
    "every instrument of the session is an option: a quiet market cannot be told from a "
    "dead feed, so a long quiet spell locks entries and asks for REST quotes"
)


def test_a_session_of_options_only_warns_that_a_quiet_market_looks_like_a_dead_feed(
    two_exchanges,
):
    two_exchanges.stop_at(ist(14, 43))
    outcome = run(two_exchanges, HoldOption)
    assert f"Warning: {OPTIONS_ONLY}" in two_exchanges.lines
    assert OPTIONS_ONLY in outcome.warnings


def test_a_session_with_an_equity_gets_no_options_only_warning(two_exchanges):
    two_exchanges.stop_at(ist(14, 43))
    outcome = run(two_exchanges, [BuyAndHold, HoldOption])
    assert OPTIONS_ONLY not in outcome.warnings


def test_F88_sbin_falling_silent_after_its_square_off_is_no_dead_feed(two_exchanges, caplog):
    outcome = run(two_exchanges, [Hold, HoldOption])
    assert events(outcome, "feed_dead") == []
