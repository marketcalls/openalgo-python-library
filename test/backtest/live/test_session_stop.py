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
from openalgo.backtest._live import _session_stop, _session_trader
from openalgo.backtest._live._alerts_logs import SessionLogs
from openalgo.backtest._live._session_exits import EXIT_ATTEMPTS
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
    (gave_up,) = [text for text in alerts(caplog, "error") if "failed 5 times" in text]
    # With no stop rule no stop can be placed again: it is closed by hand, once the
    # session has ended, since until then the engine may send its exit again.
    assert gave_up.endswith(
        "no stop at the broker; to close it by hand, end the session first (Ctrl+C, and a "
        "second Ctrl+C if it does not end): while the session runs, the engine may send "
        "its exit or its stop again"
    )


def test_two_ctrl_c_before_the_loop_looks_act_as_the_first_then_the_second(world):
    # Decided for unit 10: how fast the keys were pressed must not change the outcome, so
    # the flatten's exit is sent and the second Ctrl+C's bounded wait sees it fill.
    world.stop_at(ist(14, 43, 30))
    world.stop_at(ist(14, 43, 30))
    outcome = world.run(Hold)
    assert outcome.stopped_by == "ctrl_c"
    assert actions(world) == [("BUY", 10), ("SELL", 10)]
    assert world.fake.position("SBIN", "NSE") == 0
    assert outcome.open_positions == ()
    assert outcome.ended_ns < ist(14, 43, 30) + _session_stop.SAFE_STOP_STATE_WAIT_S * NS


def test_a_second_ctrl_c_waits_for_the_exit_in_flight_and_stops_flat(world):
    # The safe stop's SELL goes at about 14:43:31 and fills at once; its update is held
    # until 14:43:33, after the second Ctrl+C.
    world.script.at(ist(14, 43, 29), world.fake.faults.hold_order_updates)
    world.stop_at(ist(14, 43, 30))
    world.stop_at(ist(14, 43, 31) + NS // 2)
    world.script.at(ist(14, 43, 33), world.fake.faults.release_order_updates)
    outcome = world.run(Hold)
    assert actions(world) == [("BUY", 10), ("SELL", 10)]
    assert world.fake.position("SBIN", "NSE") == 0
    assert outcome.stopped_by == "ctrl_c"
    assert outcome.open_positions == ()
    assert not any("close it by hand" in line for line in world.lines)
    assert outcome.ended_ns < ist(14, 43, 31) + _session_stop.SAFE_STOP_STATE_WAIT_S * NS + NS


def test_a_second_ctrl_c_names_the_exit_still_in_flight_at_its_bound(world):
    # The safe stop's SELL never reaches the broker and its answer is lost: the session
    # cannot tell for the broker's window whether it was placed.
    world.script.at(ist(14, 43, 29), lambda: world.fake.faults.bad_gateway_page("placeorder"))
    world.stop_at(ist(14, 43, 30))
    world.stop_at(ist(14, 43, 33))
    outcome = world.run(Hold)
    assert outcome.stopped_by == "ctrl_c"
    bound = ist(14, 43, 33) + _session_stop.SAFE_STOP_STATE_WAIT_S * NS
    assert bound <= outcome.ended_ns < bound + 2 * NS
    [left] = outcome.open_positions
    assert (left.symbol, left.units) == ("SBIN", 10.0)
    assert "in flight" in left.note and "check the order book" in left.note
    [line] = [line for line in world.lines if "SBIN NSE 10 units" in line]
    assert "exit order" in line and "in flight" in line
    assert "no stop at the broker" not in line


def test_ctrl_c_asks_again_for_an_exit_that_failed_every_attempt(world):
    # BuyAndHold's exit at 14:45:01 is refused five times: the retry schedule gives up.
    def refuse_five_exits():
        for _ in range(EXIT_ATTEMPTS):
            world.fake.reject_next("RMS: blocked", price_type="MARKET")

    world.script.at(ist(14, 44, 30), refuse_five_exits)
    world.stop_at(ist(14, 47))
    outcome = world.run(BuyAndHold)
    sells = [order for order in world.orders() if order.action == "SELL"]
    assert [order.status for order in sells] == ["rejected"] * EXIT_ATTEMPTS + ["complete"]
    assert ist(14, 47) <= sells[-1].placed_ns <= ist(14, 47, 3)
    assert world.fake.position("SBIN", "NSE") == 0
    assert outcome.stopped_by == "ctrl_c"


class RestsASaleOfIt(bt.Strategy):
    """Buys 10 SBIN at its first decision (14:42:01); at its second (14:43:01) rests a
    LIMIT sale of the 10 at 600: an order on the closing side, which an exit of the
    position cancels first and waits for. (A resting sale of 20 with `reverse=True` is
    refused in a live session, 2.4.0 item P3-2.)"""

    symbol, exchange, interval = "SBIN", "NSE", "1m"

    def init(self):
        self.calls = 0

    def on_bar(self, bar):
        self.calls += 1
        if self.calls == 1:
            self.buy(10)
        elif self.calls == 2:
            self.sell(10, price=600.0)


def test_a_reminder_names_the_cancel_the_safe_stops_exit_waits_for(world, caplog):
    # The sale's answer is lost and it never reached the broker: its cancel cannot be
    # sent until the broker's window has passed, so the exit waits past the first reminder.
    world.script.at(ist(14, 42, 30), lambda: world.fake.faults.bad_gateway_page("placeorder"))
    world.stop_at(ist(14, 43, 30))
    outcome = world.run(RestsASaleOfIt)
    reminders = [text for text in alerts(caplog, "error") if "safe stop:" in text]
    assert reminders
    assert all("its exit waits for the cancel of order o2" in text for text in reminders)
    assert world.fake.position("SBIN", "NSE") == 0
    assert outcome.stopped_by == "ctrl_c"


def test_a_reminder_after_the_close_ended_the_engine_session_says_so(world, caplog):
    def refuse_exits():
        for _ in range(100):
            world.fake.reject_next("RMS: blocked", price_type="MARKET")

    world.script.at(ist(14, 59, 50), refuse_exits)
    world.stop_at(ist(15, 29))
    outcome = world.run(Hold)
    reminders = [text for text in alerts(caplog, "error") if "safe stop:" in text]
    after_the_close = [text for text in reminders if "the exchange's close ended" in text]
    assert after_the_close
    assert not any("answered None" in text for text in reminders)
    assert outcome.stopped_by == "ctrl_c"
    assert [(p.symbol, p.units) for p in outcome.open_positions] == [("SBIN", 10.0)]


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


@pytest.mark.parametrize("on_stop", ["flatten", "keep"])
def test_ctrl_c_noticed_at_the_entry_decision_sends_no_entry(world, on_stop):
    # Hold decides its entry at 14:42:01, in the loop step that notices the stop.
    world.stop_at(ist(14, 42, 1))
    outcome = world.run(Hold, on_stop=on_stop)
    assert world.fake.log.calls("placeorder") == []
    assert outcome.stopped_by == "ctrl_c"
    assert outcome.open_positions == ()
    assert list(outcome.outputs["Hold"]["orders"]["status"]) == ["Canceled"]


def test_an_entry_that_fills_after_the_safe_stop_began_is_exited(world, caplog):
    # The entry decided at 14:42:01 goes out before the loop notices the stop.
    world.stop_at(ist(14, 42, 1) + NS // 2)
    outcome = world.run(Hold)
    assert actions(world) == [("BUY", 10), ("SELL", 10)]
    assert world.fake.position("SBIN", "NSE") == 0
    assert outcome.stopped_by == "ctrl_c"
    assert world.orders()[-1].placed_ns < ist(14, 42, 10)
    assert not [text for text in alerts(caplog, "error") if "safe stop:" in text]


class RaisesAsItsEntryFills(bt.Strategy):
    """Rests a LIMIT buy of 10 SBIN at 400 at its first decision; at its second the
    order fills while `on_bar` runs (`fill_now`), and `on_bar` raises."""

    symbol, exchange, interval = "SBIN", "NSE", "1m"
    fill_now = None

    def init(self):
        self.calls = 0

    def on_bar(self, bar):
        self.calls += 1
        if self.calls == 1:
            self.buy(10, price=400.0)
            return
        type(self).fill_now()
        raise RuntimeError("the strategy's own error")


def test_a_fill_that_arrives_while_on_bar_raises_reaches_the_kernel_and_is_exited(world, caplog):
    def fill_the_resting_entry():
        world.fake.fill(world.orders()[0].order_id, price=400.0)

    RaisesAsItsEntryFills.fill_now = fill_the_resting_entry
    with pytest.raises(RuntimeError, match="the strategy's own error"):
        world.run(RaisesAsItsEntryFills)
    assert actions(world) == [("BUY", 10), ("SELL", 10)]
    assert world.fake.position("SBIN", "NSE") == 0
    assert alerts(caplog, "reconcile_mismatch") == []


def test_one_ctrl_c_during_a_flatten_a_strategy_error_began_does_not_end_it(world):
    # Raises raises at 14:43:01; the flatten's first SELL is refused and asked for again.
    world.script.at(
        ist(14, 43), lambda: world.fake.reject_next("RMS: blocked", price_type="MARKET")
    )
    world.stop_at(ist(14, 43, 2) + NS // 2)
    with pytest.raises(RuntimeError, match="the strategy's own error"):
        world.run(Raises)
    assert actions(world) == [("BUY", 10), ("SELL", 10), ("SELL", 10)]
    assert world.fake.position("SBIN", "NSE") == 0
    assert not any("close it by hand" in line for line in world.lines)


def test_a_failure_as_the_safe_stop_begins_closes_the_session_and_raises_the_strategys_error(
    world, monkeypatch, caplog
):
    def fail(self, now_ns):
        raise LiveError("a defect of the safe stop")

    monkeypatch.setattr(_session_stop.Control, "_begin", fail)
    with pytest.raises(RuntimeError, match="the strategy's own error"):
        world.run(Raises)
    assert any("a defect of the safe stop" in text for text in alerts(caplog, "error"))
    assert lock_is_free(world)
