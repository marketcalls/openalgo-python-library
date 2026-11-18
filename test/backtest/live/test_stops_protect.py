"""Protect by state, on the fake OpenAlgo (`_live/_stops_protect.py`, `_live/_session_stop.py`,
`_live/session.py`; 16-session-design 3.3, 3.5 and 7.2 rows F57 (its variant), F74, F77
and F80): when a session must end with a leg whose stop the broker holds, it never leaves
the leg with neither a stop nor an exit while one can be sent, never has a stop and an
exit working at once, finishes placing a stop it began within its bounded wait, and
never reports a stop whose cancel has no answer as confirmed.

`StopHold` buys 10 SBIN at 14:42:01 IST with a stop at 497.00, `TargetHold` with a target
at 502.00 too, which SBIN reaches at 14:44:05 (`stops_samples`).
"""

from __future__ import annotations

import logging

import pytest
from fake_openalgo.fixtures import *  # noqa: F403
from session_samples import NS, alerts, ist
from session_trace import BOOKS_AGREE
from stops_samples import (
    StopHold,
    TargetHold,
    at_price,
    check_invariants,
    stop_world,
)

from openalgo.backtest._live import _session_trader, _stops_price, _stops_protect
from openalgo.backtest._live._session_exits import EXIT_ATTEMPTS
from openalgo.backtest._live._session_stop import (
    SAFE_STOP_FLATTEN_S,
    SAFE_STOP_STATE_WAIT_S,
    UNPROTECTED_REMINDER_EVERY_S,
)
from openalgo.backtest._live._stops_report import CHECK_FIRST, STOP_PROTECTS
from openalgo.backtest._live._types import LOGGER_NAME


@pytest.fixture
def world(make_fake_openalgo, monkeypatch, tmp_path, caplog):
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    return stop_world(make_fake_openalgo, monkeypatch, tmp_path)


def kinds(world) -> list[tuple]:
    """Each live order at the fake: action, price type, status."""
    return [(o.action, o.price_type, o.status) for o in world.orders()]


def test_F57_ctrl_c_mid_position_cancels_the_stop_then_exits_flat(world):
    world.stop_at(ist(14, 43, 30))
    outcome = world.run(StopHold)

    assert kinds(world) == [
        ("BUY", "MARKET", "complete"),
        ("SELL", "SL-M", "cancelled"),
        ("SELL", "MARKET", "complete"),
    ]
    assert outcome.stopped_by == "ctrl_c"
    assert outcome.open_positions == ()
    check_invariants(world.timeline, strategy="StopHold")


def test_F57_a_second_ctrl_c_right_after_the_stops_cancel_protects_instead_of_exiting(world):
    world.stop_at(ist(14, 43, 30))
    # The first Ctrl+C's book read is compared at 14:43:31, which lets its cancel of the
    # stop go (the exit guard): the cancel is sent at 14:43:32; its word is taken, and the
    # exit made, at 14:43:33.
    world.stop_at(ist(14, 43, 32) + NS // 2)
    outcome = world.run(StopHold)

    assert kinds(world) == [
        ("BUY", "MARKET", "complete"),
        ("SELL", "SL-M", "cancelled"),
        ("SELL", "SL-M", "trigger pending"),
    ]
    assert outcome.stopped_by == "ctrl_c"
    (left,) = outcome.open_positions
    assert (left.units, left.stop_confirmed) == (10.0, True)
    assert left.stop_order_id == world.orders()[-1].order_id
    assert outcome.ended_ns < ist(14, 43, 32) + SAFE_STOP_STATE_WAIT_S * NS + NS
    check_invariants(world.timeline, strategy="StopHold")


def test_two_ctrl_c_at_once_on_a_leg_with_a_stop_leave_it_protected_with_no_exit(world):
    # The flatten's cancel of the stop waits for its book read (the exit guard), and the
    # second Ctrl+C's protect by state keeps it back: the stop is never cancelled.
    world.stop_at(ist(14, 43, 30))
    world.stop_at(ist(14, 43, 30))
    outcome = world.run(StopHold)

    assert kinds(world) == [("BUY", "MARKET", "complete"), ("SELL", "SL-M", "trigger pending")]
    (left,) = outcome.open_positions
    assert (left.stop_order_id, left.stop_confirmed) == (world.orders()[-1].order_id, True)
    check_invariants(world.timeline, strategy="StopHold")


def test_ctrl_c_with_keep_leaves_the_position_with_its_confirmed_stop(world):
    world.stop_at(ist(14, 43, 30))
    outcome = world.run(StopHold, on_stop="keep")

    assert kinds(world)[1] == ("SELL", "SL-M", "trigger pending")
    assert outcome.open_positions[0].stop_confirmed
    assert outcome.ended_ns < ist(14, 43, 32)
    assert any("confirmed at the broker" in line for line in world.lines)
    check_invariants(world.timeline, strategy="StopHold")


def test_the_safe_stops_timeout_leaves_a_leg_whose_exits_fail_with_its_stop(world, caplog):
    def refuse_exits():
        for _ in range(EXIT_ATTEMPTS + 5):
            world.fake.reject_next("RMS: blocked", price_type="MARKET")

    world.script.at(ist(14, 43, 20), refuse_exits)
    world.stop_at(ist(14, 43, 30))
    outcome = world.run(StopHold)

    exits = [o for o in world.orders() if o.price_type == "MARKET" and o.action == "SELL"]
    assert [o.status for o in exits] == ["rejected"] * EXIT_ATTEMPTS
    assert kinds(world)[-1] == ("SELL", "SL-M", "trigger pending")
    assert outcome.stopped_by == "ctrl_c"
    assert outcome.ended_ns >= ist(14, 43, 30) + SAFE_STOP_FLATTEN_S * NS
    assert outcome.open_positions[0].stop_confirmed
    check_invariants(world.timeline, strategy="StopHold")


def test_F74_a_safe_stop_whose_exit_is_unanswered_places_no_stop_and_reminds_each_minute(
    world, caplog
):
    def exit_unanswered_and_openalgo_silent():
        world.fake.faults.timeout("placeorder", reached_broker=True)
        for _ in range(150):
            world.fake.faults.gateway_error("orderbook", status=504)

    world.script.at(ist(14, 43, 32), exit_unanswered_and_openalgo_silent)
    world.stop_at(ist(14, 43, 30))
    outcome = world.run(StopHold)

    assert kinds(world) == [
        ("BUY", "MARKET", "complete"),
        ("SELL", "SL-M", "cancelled"),
        ("SELL", "MARKET", "complete"),
    ]
    assert any("in flight, no stop at the broker" in t for t in alerts(caplog, "error"))
    reminders = [m.ts_ns for m in world.fake.telegram_sent if "safe stop:" in m.message]
    assert len(reminders) >= 2
    gaps = [b - a for a, b in zip(reminders, reminders[1:], strict=False)]
    assert all(gap >= UNPROTECTED_REMINDER_EVERY_S * NS for gap in gaps)
    assert any("not flat yet" in line for line in world.lines)
    assert world.fake.position("SBIN", "NSE") == 0
    assert outcome.stopped_by == "ctrl_c"
    assert outcome.open_positions == ()
    check_invariants(world.timeline, strategy="StopHold")


def raise_once_after_the_stop_is_cancelled(monkeypatch) -> None:
    """The next kernel call of the loop after the stop's cancel was applied raises."""
    original = _session_trader.Trader.drive
    raised = []

    def drive(self, *, deciding):
        stop = self.engine.order(2)
        if not raised and stop is not None and stop["status"] == "Canceled":
            raised.append(True)
            raise RuntimeError("a defect of the loop")
        return original(self, deciding=deciding)

    monkeypatch.setattr(_session_trader.Trader, "drive", drive)


def test_F77_an_error_mid_cancel_then_exit_drops_the_exit_and_places_the_stop_again(
    world, monkeypatch, caplog
):
    at_price(world, ist(14, 44), 502.0)
    raise_once_after_the_stop_is_cancelled(monkeypatch)

    with pytest.raises(RuntimeError, match="a defect of the loop"):
        world.run(TargetHold)

    assert kinds(world) == [
        ("BUY", "MARKET", "complete"),
        ("SELL", "SL-M", "cancelled"),
        ("SELL", "SL-M", "trigger pending"),
    ]
    stop = world.orders()[-1]
    stopped = [t for t in alerts(caplog, "error") if "the session stops on an error" in t]
    assert len(stopped) == 1 and f"stop {stop.order_id} confirmed at the broker" in stopped[0]
    check_invariants(world.timeline, strategy="TargetHold")


def test_F77_when_the_kernels_protect_fails_too_the_alert_lists_the_leg_without_a_stop(
    world, monkeypatch, caplog
):
    world.trace.leave_unprotected("TargetHold", "F77 variant")
    at_price(world, ist(14, 44), 502.0)
    raise_once_after_the_stop_is_cancelled(monkeypatch)

    def fails(trader):
        raise RuntimeError("the kernel's protect failed")

    monkeypatch.setattr(_stops_protect, "protect", fails)
    with pytest.raises(RuntimeError, match="a defect of the loop"):
        world.run(TargetHold)

    assert [kind for _, kind, _ in kinds(world)] == ["MARKET", "SL-M"]
    stopped = [t for t in alerts(caplog, "error") if "the session stops on an error" in t]
    # No stop and nothing working: the one case the alert says to close by hand.
    assert len(stopped) == 1
    assert "10 units: no stop at the broker, close it by hand" in stopped[0]
    check_invariants(world.timeline, strategy="TargetHold", unprotected_at_end=True)


def test_F80_an_ambiguous_stop_placement_leaves_the_leg_marked_and_sends_nothing_more(
    world, caplog
):
    world.trace.leave_unprotected("StopHold", "F80")
    other = world.fake.client()

    def the_same_stop_by_another_client():
        other.placeorder(
            strategy="StopHold",
            symbol="SBIN",
            action="SELL",
            exchange="NSE",
            price_type="SL-M",
            product="MIS",
            quantity=10,
            trigger_price=497.0,
        )

    def the_stops_answer_is_lost():
        # The other client's stop comes first, just after the session decided its own.
        the_same_stop_by_another_client()
        world.fake.faults.bad_gateway_page("placeorder", reached_broker=True)

    world.script.at(ist(14, 42, 3) + NS // 2, the_stops_answer_is_lost)
    # Past NSE's square-off at 15:00, which must not be sent either.
    world.stop_at(ist(15, 1))
    world.stop_at(ist(15, 2))
    outcome = world.run(StopHold)

    sent = [c for c in world.fake.log.calls("placeorder") if c.ts_ns > ist(14, 42, 4)]
    assert sent == []
    assert [o for o in world.orders() if o.placed_ns >= ist(15, 0)] == []
    assert world.fake.log.calls("cancelorder") == []
    ambiguous = [t for t in alerts(caplog, "error") if "cannot be told apart" in t]
    assert len(ambiguous) == 1
    (left,) = outcome.open_positions
    assert "ambiguous stop order" in left.note and not left.stop_confirmed
    assert "ambiguous order" in [w.reason for w in outcome.unprotected]


def test_two_ctrl_c_with_the_stops_cancel_word_lost_place_the_stop_again_before_the_end(world):
    # The flatten cancels the stop once its book read is compared, at 14:43:36; the
    # second Ctrl+C follows, and the broker's word of the cancel is lost on the stream,
    # so only a read of the order tells the session that the stop is gone.
    second_ns = ist(14, 43, 36) + NS // 2
    world.script.at(ist(14, 43, 35), world.fake.faults.drop_next_order_update)
    world.stop_at(ist(14, 43, 35))
    world.stop_at(second_ns)
    outcome = world.run(StopHold)

    assert kinds(world) == [
        ("BUY", "MARKET", "complete"),
        ("SELL", "SL-M", "cancelled"),
        ("SELL", "SL-M", "trigger pending"),
    ]
    (left,) = outcome.open_positions
    assert (left.stop_order_id, left.stop_confirmed) == (world.orders()[-1].order_id, True)
    # The loop's step that ends the wait still waits its turn (1 s here) before the close,
    # whose final book read takes a few of the loop's shortest waits.
    assert outcome.ended_ns <= second_ns + SAFE_STOP_STATE_WAIT_S * NS + NS + NS // 10
    check_invariants(world.timeline, strategy="StopHold")


def test_a_stop_whose_cancel_has_no_word_at_the_end_is_not_reported_confirmed(world):
    # A second Ctrl+C ends the session before the cancel's word (3.3 step 4): the leg is
    # left with no confirmed stop and its line says so, which the invariants take.
    world.trace.left_unprotected_by_decision(
        "StopHold", "3.3 step 4: a second Ctrl+C before the cancel's word"
    )

    def cancel_word_lost_and_reads_failing():
        world.fake.faults.drop_next_order_update()
        for _ in range(20):
            world.fake.faults.gateway_error("orderstatus", status=503)

    # The second Ctrl+C follows the flatten's cancel of the stop, sent once its book read
    # is compared at 14:43:36.
    world.script.at(ist(14, 43, 35), cancel_word_lost_and_reads_failing)
    world.stop_at(ist(14, 43, 35))
    world.stop_at(ist(14, 43, 36) + NS // 2)
    outcome = world.run(StopHold)

    cancelled = world.orders()[1]
    assert (cancelled.price_type, cancelled.status) == ("SL-M", "cancelled")
    (left,) = outcome.open_positions
    assert not left.stop_confirmed
    assert f"stop {cancelled.order_id} being cancelled" in left.note
    assert "confirmed at the broker" not in left.note
    assert "check the order book" in left.note


def test_a_stop_placed_again_in_the_last_step_of_the_second_ctrl_c_wait_reaches_the_broker(
    world,
):
    # The flatten's cancel of the stop goes once its book read is compared at 14:43:31,
    # and is accepted at 14:43:32, when the second Ctrl+C is seen; its word lags on the
    # stream until 14:43:35.5 and the reads of the order fail, so the stop is placed again
    # in the step at 14:43:37 that ends the second Ctrl+C's wait of 5 seconds: it must
    # still be sent, and its first word taken.
    def stream_lags_and_reads_fail():
        world.fake.faults.hold_order_updates()
        for _ in range(5):
            world.fake.faults.gateway_error("orderstatus", status=503)

    world.script.at(ist(14, 43, 29), stream_lags_and_reads_fail)
    world.script.at(ist(14, 43, 35) + NS // 2, world.fake.faults.release_order_updates)
    world.stop_at(ist(14, 43, 30))
    world.stop_at(ist(14, 43, 31) + NS // 2)
    outcome = world.run(StopHold)

    assert kinds(world) == [
        ("BUY", "MARKET", "complete"),
        ("SELL", "SL-M", "cancelled"),
        ("SELL", "SL-M", "trigger pending"),
    ]
    (left,) = outcome.open_positions
    assert (left.stop_order_id, left.stop_confirmed) == (world.orders()[-1].order_id, True)
    check_invariants(world.timeline, strategy="StopHold")


def test_a_supervisor_part_that_fails_does_not_keep_the_emergency_stop_from_protecting(
    world, monkeypatch, caplog
):
    at_price(world, ist(14, 44), 502.0)
    original = _stops_price.PriceRule.due

    def due(self, now_ns):
        (trader,) = self._session.traders
        stop = trader.engine.order(2)
        if stop is not None and stop["status"] == "Canceled":
            raise RuntimeError("a defect of the price rule")
        return original(self, now_ns)

    monkeypatch.setattr(_stops_price.PriceRule, "due", due)
    with pytest.raises(RuntimeError, match="a defect of the price rule"):
        world.run(TargetHold)

    assert kinds(world) == [
        ("BUY", "MARKET", "complete"),
        ("SELL", "SL-M", "cancelled"),
        ("SELL", "SL-M", "trigger pending"),
    ]
    stopped = [t for t in alerts(caplog, "error") if "the session stops on an error" in t]
    assert len(stopped) == 1
    assert f"stop {world.orders()[-1].order_id} confirmed at the broker" in stopped[0]
    check_invariants(world.timeline, strategy="TargetHold")


#: The tick at which SBIN reaches `TargetHold`'s target: the engine cancels the stop.
TARGET_TICK = ist(14, 44, 5)


def test_keep_while_the_stops_cancel_word_is_lost_places_the_stop_again(world):
    at_price(world, ist(14, 44), 502.0)
    world.script.at(TARGET_TICK, world.fake.faults.drop_next_order_update)
    world.stop_at(TARGET_TICK + NS // 2)
    outcome = world.run(TargetHold, on_stop="keep")

    assert kinds(world) == [
        ("BUY", "MARKET", "complete"),
        ("SELL", "SL-M", "cancelled"),
        ("SELL", "SL-M", "trigger pending"),
    ]
    (left,) = outcome.open_positions
    assert (left.stop_order_id, left.stop_confirmed) == (world.orders()[-1].order_id, True)
    check_invariants(world.timeline, strategy="TargetHold")


def test_the_emergency_stop_while_the_stops_cancel_word_is_lost_places_the_stop_again(
    world, monkeypatch, caplog
):
    at_price(world, ist(14, 44), 502.0)
    world.script.at(TARGET_TICK, world.fake.faults.drop_next_order_update)
    original = _session_trader.Trader.drive
    raised = []

    def drive(self, *, deciding):
        cancelling = self.engine.live_state()["desk"]["cancelling"]
        if not raised and 2 in cancelling:
            raised.append(True)
            raise RuntimeError("a defect of the loop")
        return original(self, deciding=deciding)

    monkeypatch.setattr(_session_trader.Trader, "drive", drive)
    with pytest.raises(RuntimeError, match="a defect of the loop"):
        world.run(TargetHold)

    assert kinds(world) == [
        ("BUY", "MARKET", "complete"),
        ("SELL", "SL-M", "cancelled"),
        ("SELL", "SL-M", "trigger pending"),
    ]
    stopped = [t for t in alerts(caplog, "error") if "the session stops on an error" in t]
    assert len(stopped) == 1
    assert f"stop {world.orders()[-1].order_id} confirmed at the broker" in stopped[0]
    check_invariants(world.timeline, strategy="TargetHold")


def raise_once_at(monkeypatch, world, at_ns: int) -> None:
    """The loop's first kernel call at or after `at_ns` raises."""
    original = _session_trader.Trader.drive
    raised = []

    def drive(self, *, deciding):
        if not raised and world.fake.clock.now() >= at_ns:
            raised.append(True)
            raise RuntimeError("a defect of the loop")
        return original(self, deciding=deciding)

    monkeypatch.setattr(_session_trader.Trader, "drive", drive)


def stopped_alert(caplog) -> str:
    """The emergency stop's one ERROR alert."""
    (text,) = [t for t in alerts(caplog, "error") if "the session stops on an error" in t]
    return text


def test_the_emergency_alert_says_a_confirmed_stop_protects_the_leg_until_it_is_cancelled(
    world, monkeypatch, caplog
):
    raise_once_at(monkeypatch, world, ist(14, 43, 30))
    with pytest.raises(RuntimeError, match="a defect of the loop"):
        world.run(StopHold)

    stop = world.orders()[-1]
    assert (stop.price_type, stop.status) == ("SL-M", "trigger pending")
    text = stopped_alert(caplog)
    # A sale by hand beside the working stop would leave the stop to sell 10 more.
    assert f"stop {stop.order_id} confirmed at the broker" in text
    assert STOP_PROTECTS in text
    assert "close it by hand" not in text.lower().replace(STOP_PROTECTS, "")
    check_invariants(world.timeline, strategy="StopHold")


def test_the_emergency_alert_names_an_exit_at_work_and_the_order_book_comes_first(
    world, monkeypatch, caplog
):
    # The exit's fill never reaches the session, which reads no books after an error
    # (3.5): the engine ends long against the fake's flat, and the ERROR alert names the
    # exit in flight instead of a reconciliation.
    world.trace.waive(BOOKS_AGREE, "3.5: an exit filled unheard, and no read after an error")
    at_price(world, ist(14, 44), 502.0)

    def exit_words_held_and_reads_failing():
        world.fake.faults.hold_order_updates()
        for _ in range(30):
            world.fake.faults.gateway_error("orderstatus", status=503)
            world.fake.faults.gateway_error("tradebook", status=503)

    # The target's sale is sent and fills at the broker; the session never hears of it.
    world.script.at(TARGET_TICK + 3 * NS - NS // 2, exit_words_held_and_reads_failing)
    raise_once_at(monkeypatch, world, TARGET_TICK + 5 * NS)
    with pytest.raises(RuntimeError, match="a defect of the loop"):
        world.run(TargetHold)

    sale = next(o for o in world.orders() if (o.action, o.price_type) == ("SELL", "MARKET"))
    assert sale.status == "complete" and world.fake.position("SBIN", "NSE") == 0
    text = stopped_alert(caplog)
    # Selling by hand what the alert lists would sell 10 short.
    assert f"exit order {sale.order_id} in flight" in text
    assert text.endswith(CHECK_FIRST)
    assert "close it by hand" not in text.lower().replace(CHECK_FIRST, "")
