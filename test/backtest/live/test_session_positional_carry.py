"""What a live session says of the CNC legs it carries past the close, on the fake OpenAlgo
(20-positional-live 2 "Day 1, the close", 4.1, 7.3; the lead's rulings N3b-4 and N3b-5,
review findings 3, 6 c and 7 of unit N3b).

Each carried leg is named in a line at the session's end with its stop GTT's trigger ID
and whether a read of the GTT book after the close confirmed it. A stop GTT that ends in
the last seconds before the close (cancelled by hand) is found once the engine's session
has closed and can place nothing more: the leg is then never reported as protected, in
the lines, the result or the day-close record, and one ERROR alert says to place the stop
by hand. A leg whose stop GTT the read confirmed is no ERROR at the end, also beside an MIS
leg left open at the close.

The market and the strategies are `positional_samples`': SBIN trades every 10 s at 500.00
until 15:40 IST; the entries fill at 500.00 at 14:42:01; NSE squares MIS off at 15:00 and
closes at 15:30.
"""

from __future__ import annotations

import logging

import pytest
from fake_openalgo import BROKER
from fake_openalgo.fixtures import *  # noqa: F403
from positional_samples import BOOKS, CncHold, MisHold, calls, gtt_world
from session_samples import alerts, ist
from test_session_positional_close import next_day

from openalgo.backtest._live._session_carry import UNCONFIRMED_KEY
from openalgo.backtest._live._types import LOGGER_NAME
from openalgo.backtest._live.journal import JOURNAL_FILE_NAME, Journal

#: NSE's close on the module's day, UTC nanoseconds.
CLOSE = ist(15, 30)
#: Seconds before the close the stop GTT is cancelled by hand.
LAST_SECONDS = ist(15, 29, 55)
#: Where `HoldAndRest`'s GTT buy rests, below every price of the market.
REST_PRICE = 490.0


@pytest.fixture(autouse=True)
def _logs(caplog):
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)


def to_the_close(make_fake_openalgo, monkeypatch, tmp_path, **options):
    return gtt_world(make_fake_openalgo, monkeypatch, tmp_path, trades_until=(15, 40), **options)


def record_of(world, strategy):
    path = world.folder / "live" / BROKER / JOURNAL_FILE_NAME
    with Journal.open(
        path, account=BROKER, broker=BROKER, mode=world.fake.mode, now=world.fake.clock.now
    ) as journal:
        return journal.day_closes()[strategy]


def carried_lines(world, strategy):
    return [line for line in world.lines if line.startswith(f"{strategy} SBIN NSE 10 units:")]


def test_each_carried_leg_is_named_at_the_end_with_its_confirmed_stop_gtt(
    make_fake_openalgo, monkeypatch, tmp_path, caplog
):
    world = to_the_close(make_fake_openalgo, monkeypatch, tmp_path)

    outcome = world.run(CncHold)

    assert outcome.stopped_by == "session_end"
    trigger_id = world.fake.gtts()[0].trigger_id
    (line,) = carried_lines(world, "CncHold")
    assert f"stop GTT {trigger_id} confirmed at the broker" in line
    (position,) = outcome.open_positions
    assert (position.stop_order_id, position.stop_confirmed) == (trigger_id, True)
    assert not [a for a in alerts(caplog, "error") if "CncHold" in a]


@BOOKS
def test_a_stop_gtt_ended_in_the_last_seconds_is_never_reported_as_protecting_the_leg(
    make_fake_openalgo, monkeypatch, tmp_path, caplog, history
):
    world = to_the_close(make_fake_openalgo, monkeypatch, tmp_path, history=history)
    world.trace.waive("protected", "the stop GTT is cancelled by hand before the close")
    world.trace.left_unprotected_by_decision("CncHold", "cancelled by hand at the close")

    def cancel_by_hand():
        world.fake.set_gtt_status(world.fake.gtts()[0].trigger_id, "cancelled")

    world.script.at(LAST_SECONDS, cancel_by_hand)

    outcome = world.run(CncHold)

    assert outcome.stopped_by == "session_end"
    trigger_id = world.fake.gtts()[0].trigger_id
    (position,) = outcome.open_positions
    assert position.stop_confirmed is False
    assert f"stop GTT {trigger_id} not confirmed at the broker after the close" in position.note
    (line,) = carried_lines(world, "CncHold")
    assert "not confirmed at the broker after the close" in line
    (alert,) = [a for a in alerts(caplog, "error") if "CncHold" in a]
    assert "carried overnight" in alert
    # Nothing of this version places it after the close: the alert asks for it by hand.
    assert alert.endswith("place its stop GTT in OpenAlgo by hand before the next open")
    record = record_of(world, "CncHold")
    (unconfirmed,) = record.carried[UNCONFIRMED_KEY]
    assert (unconfirmed["gtt"], unconfirmed["trigger_id"]) == (1, trigger_id)
    restored = next_day(record.carried).restore_carried_state(record.carried)
    assert list(restored["gtts"]) == [1]


@pytest.mark.parametrize(
    ("history", "may_have_fired"),
    [(True, False), (False, True)],
    ids=["history rows: cancelled", "active only: gone"],
)
def test_a_stop_gtt_gone_after_the_close_may_have_fired_so_the_alert_checks_the_books_first(
    make_fake_openalgo, monkeypatch, tmp_path, caplog, history, may_have_fired
):
    # A book of active GTTs only shows a GTT that fired and one cancelled by hand alike:
    # gone. A stop GTT placed by hand beside a child that sold the units would sell again.
    world = to_the_close(make_fake_openalgo, monkeypatch, tmp_path, history=history)
    world.trace.waive("protected", "the stop GTT is cancelled by hand before the close")
    world.trace.left_unprotected_by_decision("CncHold", "cancelled by hand at the close")

    def cancel_by_hand():
        world.fake.set_gtt_status(world.fake.gtts()[0].trigger_id, "cancelled")

    world.script.at(LAST_SECONDS, cancel_by_hand)

    world.run(CncHold)

    (alert,) = [a for a in alerts(caplog, "error") if "CncHold" in a]
    checked_first = (
        "it may have fired: check the order book and the position first, then place its "
        "stop GTT in OpenAlgo by hand before the next open"
    )
    assert alert.endswith(checked_first) is may_have_fired
    assert alert.endswith("place its stop GTT in OpenAlgo by hand before the next open")
    assert "in_flight" not in alert and "ended_outside" not in alert


class HoldAndRest(CncHold):
    """CncHold that also rests a GTT buy of 5 SBIN at 490.00 at its second decision."""

    def on_bar(self, bar):
        self.calls += 1
        if self.calls == 1:
            self.buy(self.qty, **type(self).rules)
        elif self.calls == 2:
            self.buy(qty=5, price=REST_PRICE, validity="gtt")


@BOOKS
def test_a_gtt_order_ended_in_the_last_seconds_is_named_and_leaves_the_stop_confirmed(
    make_fake_openalgo, monkeypatch, tmp_path, caplog, history
):
    # Found ended once the engine's session has closed: the GTT order is named in a REJECT
    # alert of its own; the leg's stop GTT, which rests, is still the leg's confirmed stop.
    world = to_the_close(make_fake_openalgo, monkeypatch, tmp_path, history=history)

    def cancel_by_hand():
        (order,) = [g for g in world.fake.gtts() if g.action == "BUY"]
        world.fake.set_gtt_status(order.trigger_id, "cancelled")

    world.script.at(LAST_SECONDS, cancel_by_hand)

    outcome = world.run(HoldAndRest)

    assert outcome.stopped_by == "session_end"
    (order,) = [g for g in world.fake.gtts() if g.action == "BUY"]
    (named,) = [a for a in alerts(caplog, "reject") if "HoldAndRest" in a]
    assert f"GTT order {order.trigger_id} of SBIN NSE " in named
    assert named.endswith("after the session's close: the session follows it no more")
    # In plain words, not the follower's event names.
    assert "in_flight" not in named and "ended_outside" not in named
    assert not [a for a in alerts(caplog, "error") if "HoldAndRest" in a]
    (position,) = outcome.open_positions
    assert position.stop_confirmed is True
    assert record_of(world, "HoldAndRest").carried[UNCONFIRMED_KEY] == []


def test_a_carried_leg_beside_an_mis_leg_left_open_at_the_close_is_no_error(
    make_fake_openalgo, monkeypatch, tmp_path, caplog
):
    world = to_the_close(make_fake_openalgo, monkeypatch, tmp_path)
    world.trace.waive("protected", "the MIS square-off is rejected every time")
    world.trace.waive("books agree", "the MIS square-off is rejected every time")
    world.trace.left_unprotected_by_decision("MisHold", "its square-off is rejected")

    def reject_market_orders():
        for _ in range(500):
            world.fake.reject_next("RMS: blocked", price_type="MARKET")

    world.script.at(ist(14, 59, 50), reject_market_orders)

    outcome = world.run([MisHold, CncHold])

    assert outcome.stopped_by == "session_end"
    errors = alerts(caplog, "error")
    assert [a for a in errors if "MisHold SBIN NSE 10 units" in a]
    assert not [a for a in errors if "CncHold SBIN NSE 10 units" in a]
    # The close's read of the GTT book is made though an MIS leg is still open.
    assert [c for c in calls(world, "gttorderbook") if c.ts_ns >= CLOSE]
    (line,) = carried_lines(world, "CncHold")
    assert "confirmed at the broker" in line
