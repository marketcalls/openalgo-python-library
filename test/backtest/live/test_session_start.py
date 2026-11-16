"""The start of a live session, on the fake OpenAlgo (`_live/session.py`,
`_live/_session_start.py`; 16-session-design 1.2, 3.1 and 3.2, unit 8): what is refused
before anything is sent, the banner, a restart of the day, and the wait for the broker's
login before the open.
"""

from __future__ import annotations

import logging
import threading

import pytest
from fake_openalgo.fixtures import *  # noqa: F403
from session_samples import BuyAndHold, StopFlag, ist, rig

from openalgo.backtest._live._types import LOGGER_NAME
from openalgo.backtest._live.session import run_session
from openalgo.backtest.errors import ConfigError, LiveError


@pytest.fixture
def world(make_fake_openalgo, monkeypatch, tmp_path, caplog):
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    return rig(make_fake_openalgo, monkeypatch, tmp_path)


def test_an_argument_that_chooses_the_mode_is_refused_before_any_call(world):
    with pytest.raises(ConfigError, match="mode"):
        world.run(BuyAndHold, mode="live")
    assert world.fake.log.calls() == []


def test_with_signal_handlers_a_session_off_the_main_thread_is_refused(world):
    caught: list[BaseException] = []

    def start() -> None:
        try:
            run_session(BuyAndHold, world.fake.client(), capital=1, max_daily_loss=1)
        except BaseException as error:
            caught.append(error)

    thread = threading.Thread(target=start)
    thread.start()
    thread.join(30)
    assert isinstance(caught[0], LiveError) and "main thread" in str(caught[0])
    assert world.fake.log.calls() == []


def test_the_banner_names_the_mode_first_and_the_square_off(world):
    world.stop_at(ist(14, 40, 20))
    world.run(BuyAndHold)
    assert world.lines[0] == "OpenAlgo is in LIVE mode: orders go to zerodha"
    assert any("square-off 15:00 IST" in line for line in world.lines)
    assert any("entries until 14:50 IST" in line for line in world.lines)


def test_a_restart_of_the_day_whose_journal_holds_orders_is_refused(world):
    world.stop_at(ist(14, 43))
    world.run(BuyAndHold)
    world.stop, world.feed = StopFlag(), None
    placed = len(world.fake.log.calls("placeorder"))
    with pytest.raises(LiveError, match="restart"):
        world.run(BuyAndHold)
    assert len(world.fake.log.calls("placeorder")) == placed


def test_a_restart_after_a_start_that_sent_nothing_is_a_fresh_start(world):
    world.stop_at(ist(14, 40, 20))
    world.run(BuyAndHold)
    world.stop, world.feed = StopFlag(), None
    world.stop_at(ist(14, 45))
    outcome = world.run(BuyAndHold)
    assert outcome.restarts == 1
    assert [order.action for order in world.orders()] == ["BUY", "SELL"]


def test_before_the_open_a_failed_login_is_checked_again_each_minute(
    make_fake_openalgo, monkeypatch, tmp_path, caplog
):
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    world = rig(make_fake_openalgo, monkeypatch, tmp_path, start=(8, 58, 0), trades_until=None)
    world.fake.faults.expire_session()

    def log_in_and_trade():
        world.fake.faults.restore_session()
        world.script.trade()
        world.script.trades(ist(9, 5, 10), ist(9, 30))

    world.script.at(ist(9, 5), log_in_and_trade)
    world.stop_at(ist(9, 17))
    outcome = world.run(BuyAndHold)
    tests = [record for record in caplog.records if record.getMessage().startswith("test alert")]
    assert len(tests) == 1
    errors = [r.getMessage() for r in caplog.records if r.getMessage().startswith("alert error")]
    assert any("not logged in" in text for text in errors)
    assert outcome.stopped_by == "ctrl_c"
    assert [order.action for order in world.orders()] == ["BUY", "SELL"]
