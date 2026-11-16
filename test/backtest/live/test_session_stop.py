"""How a live session stops, on the fake OpenAlgo (`_live/session.py`,
`_live/_session_stop.py`; 16-session-design 3.3 to 3.5 and 7.2 rows F57 (without its
variant), F58, F62 and F86, unit 8, run without leg rules): Ctrl+C with `on_stop`
"flatten" or "keep", Ctrl+C during the countdown, a second Ctrl+C, a strategy's
exception, a feed that never connects, a journal that fails, an event stream that cannot
be written, and an error of the loop itself.

The strategy is `BuyAndHold` (buys 10 SBIN at its first decision, 14:42:01 IST, sells
three bars later) unless a test names another. "The lock is released" is checked by
taking the account's lock again after the session.
"""

from __future__ import annotations

import logging

import pytest
from fake_openalgo import BROKER, HOST
from fake_openalgo.fixtures import *  # noqa: F403
from session_samples import NS, BuyAndHold, Hold, Raises, alerts, ist, rig

import openalgo.backtest as bt
from openalgo.backtest._live import _session_trader
from openalgo.backtest._live._alerts_logs import SessionLogs
from openalgo.backtest._live._types import LOGGER_NAME, Mode
from openalgo.backtest._live.journal import Journal
from openalgo.backtest._live.preflight import AccountLock
from openalgo.backtest.errors import JournalError, LiveError


@pytest.fixture
def world(make_fake_openalgo, monkeypatch, tmp_path, caplog):
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    return rig(make_fake_openalgo, monkeypatch, tmp_path)


def lock_is_free(world) -> bool:
    """Whether the account's lock can be taken again (the session released it)."""
    with AccountLock(BROKER, server_url=HOST, folder=world.folder / "live"):
        return True


def actions(world) -> list[tuple[str, int]]:
    return [(order.action, order.quantity) for order in world.orders()]


class RestingEntry(bt.Strategy):
    """Rests a LIMIT buy far below the market at its first decision."""

    symbol, exchange, interval = "SBIN", "NSE", "1m"

    def on_bar(self, bar):
        if not self.orders:
            self.buy(10, price=400.0)


def test_F57_ctrl_c_mid_position_exits_and_stops_flat(world):
    world.stop_at(ist(14, 43, 30))
    outcome = world.run(Hold)
    assert outcome.stopped_by == "ctrl_c"
    assert actions(world) == [("BUY", 10), ("SELL", 10)]
    assert world.fake.position("SBIN", "NSE") == 0
    assert outcome.open_positions == ()
    assert ist(14, 43, 30) <= world.orders()[-1].placed_ns <= ist(14, 43, 33)
    assert outcome.ended_ns < ist(14, 44)


def test_ctrl_c_with_on_stop_keep_leaves_the_position_open(world):
    world.stop_at(ist(14, 43, 30))
    outcome = world.run(Hold, on_stop="keep")
    assert outcome.stopped_by == "ctrl_c"
    assert actions(world) == [("BUY", 10)]
    assert [(p.symbol, p.units) for p in outcome.open_positions] == [("SBIN", 10.0)]
    assert any("SBIN NSE 10 units" in line for line in world.lines)


def test_ctrl_c_cancels_a_working_entry(world):
    world.stop_at(ist(14, 43, 30))
    outcome = world.run(RestingEntry)
    assert [order.status for order in world.orders()] == ["cancelled"]
    assert outcome.stopped_by == "ctrl_c"
    assert world.fake.log.calls("cancelorder")


def test_a_second_ctrl_c_stops_with_a_line_per_leg_left_open(world, caplog):
    def refuse_exits():
        for _ in range(10):
            world.fake.reject_next("RMS: blocked", price_type="MARKET")

    world.script.at(ist(14, 43), refuse_exits)
    world.stop_at(ist(14, 43, 30))
    world.stop_at(ist(14, 45, 30))
    outcome = world.run(Hold)
    assert outcome.stopped_by == "ctrl_c"
    assert [(p.symbol, p.units) for p in outcome.open_positions] == [("SBIN", 10.0)]
    assert any("close it by hand" in line for line in world.lines)
    assert any("safe stop" in text for text in alerts(caplog, "error"))
    assert any("failed 5 times" in text for text in alerts(caplog, "error"))


def test_F58_ctrl_c_in_the_countdown_sends_nothing_and_stops_within_a_second(world):
    asked = ist(14, 40, 8)
    world.stop_at(asked)
    outcome = world.run(BuyAndHold)
    assert outcome.stopped_by == "ctrl_c"
    assert world.orders() == []
    assert world.fake.log.calls("placeorder") == []
    assert outcome.ended_ns - asked <= NS
    assert lock_is_free(world)


def test_F62_a_strategys_exception_stops_safely_then_is_raised(world, caplog):
    with pytest.raises(RuntimeError, match="the strategy's own error"):
        world.run(Raises)
    assert actions(world) == [("BUY", 10), ("SELL", 10)]
    assert world.fake.position("SBIN", "NSE") == 0
    assert any("Raises raised RuntimeError" in text for text in alerts(caplog, "error"))
    assert lock_is_free(world)
    journal = Journal.open(
        world.folder / "live" / BROKER / "journal.sqlite",
        account=BROKER,
        broker=BROKER,
        mode=Mode.LIVE,
        now=world.fake.clock.now,
    )
    with journal:
        assert journal.user_state("Raises").state == {"bought": True}


def test_F86_a_feed_that_never_connects_stops_the_start_with_nothing_sent(world):
    world.fake.refuse_feed()
    started = world.fake.clock.now()
    with pytest.raises(LiveError, match="feed"):
        world.run(BuyAndHold)
    assert world.fake.log.calls("placeorder") == []
    assert world.fake.clock.now() - started >= 30 * NS
    assert lock_is_free(world)


def test_a_journal_failure_stops_sending_and_is_raised(world, monkeypatch, caplog):
    def fail(self, intent):
        raise JournalError("cannot record the intent: disk I/O error")

    monkeypatch.setattr(Journal, "record_intent", fail)
    with pytest.raises(JournalError, match="disk I/O"):
        world.run(BuyAndHold)
    assert world.fake.log.calls("placeorder") == []
    assert any("JournalError" in text for text in alerts(caplog, "error"))
    assert lock_is_free(world)


def test_an_event_stream_that_cannot_be_written_is_alerted_once_and_not_fatal(
    world, monkeypatch, caplog
):
    def fail(self, kind, **fields):
        raise LiveError("cannot write the session event stream: disk full")

    monkeypatch.setattr(SessionLogs, "event", fail)
    outcome = world.run(BuyAndHold)
    assert outcome.stopped_by == "session_end"
    assert actions(world) == [("BUY", 10), ("SELL", 10)]
    stream = [text for text in alerts(caplog, "error") if "event stream" in text]
    assert len(stream) == 1
    assert any("event stream" in warning for warning in outcome.warnings)


def test_an_error_of_the_loop_locks_entries_alerts_and_is_raised(world, monkeypatch, caplog):
    original = _session_trader.Trader.release_due

    def fail_later(self):
        if self.part.engine.position(0, "MIS") is not None:
            raise RuntimeError("a defect of the loop")
        return original(self)

    monkeypatch.setattr(_session_trader.Trader, "release_due", fail_later)
    with pytest.raises(RuntimeError, match="a defect of the loop"):
        world.run(Hold)
    assert actions(world) == [("BUY", 10)]
    assert any("SBIN NSE 10 units" in text for text in alerts(caplog, "error"))
    assert lock_is_free(world)
