"""A strategy whose session of an earlier day ended without its day-close record, at the next
day's start, on the fake OpenAlgo (20-positional-live 1 and 4.2 step 2; the lead's ruling
N5a-1): an MIS strategy is never refused for it (the broker squares MIS off, nothing is
carried); a CNC strategy whose crash left a position, a working entry or a stop GTT is
refused while the broker may still hold any of them, naming the units the books show, which
may be the user's own (review round 2 of unit N5a), and starts flat once the operator has
closed the position and the GTT by hand.

The days and the strategies are `carry_samples`' and `positional_samples`': on Monday
`MisHold` (MIS) or `CncKept` (CNC) buys 10 SBIN at 500.00 at 14:42:01 IST with a stop 3
points below (an SL-M for MIS, a SINGLE stop GTT at 497.00 for CNC). Monday's session is
stopped by a Ctrl+C at 14:46 that keeps the leg, or crashes then. Tuesday's session starts
at 14:40:05 IST and is stopped at 14:46.
"""

from __future__ import annotations

import datetime as dt
import logging

import pytest
from carry_samples import DAY1, DAY2, DAY3, CncKept, held, journal_of, on, to_day
from fake_openalgo import BROKER
from fake_openalgo.fixtures import *  # noqa: F403
from positional_samples import BOOKS, MODES, STOP_AT, MisHold, calls, gtt_world, gtts
from restart_samples import crash_at, crashing, restart, run_to_crash
from session_samples import StopFlag, alerts

from openalgo.backtest._live._carry_unrecorded import CHECK_BY_HAND
from openalgo.backtest._live._journal_gtt_rows import GttState, GttTerms
from openalgo.backtest._live._types import LOGGER_NAME, Mode, Purpose
from openalgo.backtest.errors import PreflightError

#: Tuesday's session start, and the calls that send anything to the broker.
DAY2_START = on(DAY2, 14, 40, 5)
SENDS = (
    "placeorder",
    "modifyorder",
    "cancelorder",
    "placegttorder",
    "modifygttorder",
    "cancelgttorder",
)


@pytest.fixture(autouse=True)
def _logs(caplog):
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)


def square_off(world):
    """What the broker does to an MIS leg left open at its close: the stop order
    cancelled, the units sold at market."""
    client = world.fake.client()
    for order in world.fake.orders():
        if order.status == "trigger pending":
            client.cancelorder(order_id=order.order_id, strategy=order.strategy)
    client.placeorder(
        strategy="square-off",
        symbol="SBIN",
        action="SELL",
        exchange="NSE",
        price_type="MARKET",
        product="MIS",
        quantity=10,
    )


@MODES
def test_an_mis_strategy_kept_by_ctrl_c_starts_as_usual_the_next_day(
    make_fake_openalgo, monkeypatch, tmp_path, analyzer
):
    world = gtt_world(
        make_fake_openalgo, monkeypatch, tmp_path, analyzer=analyzer, trades_until=(15, 32)
    )
    world.trace.waive("the sum rule", "the broker's square-off of Monday is another day's")
    world.stop_at(STOP_AT)
    monday = world.run(MisHold, on_stop="keep")
    assert monday.stopped_by == "ctrl_c"
    square_off(world)
    assert world.fake.position("SBIN", "NSE") == 0
    world.day, world.stop = DAY1, StopFlag()
    to_day(world, DAY2)
    world.stop_at(on(DAY2, 14, 46))

    tuesday = world.run(MisHold)

    assert tuesday.stopped_by == "ctrl_c"


def crashed_monday(make_fake_openalgo, monkeypatch, tmp_path, **options):
    """A world whose Monday session of `CncKept` crashed at 14:46 holding its 10 CNC units
    and its resting stop GTT, moved to Tuesday's session start."""
    world = gtt_world(make_fake_openalgo, monkeypatch, tmp_path, trades_until=(15, 32), **options)
    world.trace.waive("protected", "the session crashed: its windows end with it")
    crashing(monkeypatch)
    crash_at(world, on(DAY1, 14, 46))
    run_to_crash(world, CncKept)
    world.day, world.stop, world.feed = DAY1, StopFlag(), None
    to_day(world, DAY2)
    return world


def sell_by_hand(world):
    """Another client of the account sells the 10 SBIN CNC units at market."""
    world.fake.client().placeorder(
        strategy="manual",
        symbol="SBIN",
        action="SELL",
        exchange="NSE",
        price_type="MARKET",
        product="CNC",
        quantity=10,
    )


def test_a_crash_before_the_close_record_refuses_the_next_start_while_the_leg_is_held(
    make_fake_openalgo, monkeypatch, tmp_path
):
    world = crashed_monday(make_fake_openalgo, monkeypatch, tmp_path)
    (trigger_id,) = [gtt.trigger_id for gtt in world.fake.gtts()]

    with pytest.raises(PreflightError) as refused:
        world.run(CncKept)

    assert refused.value.failures == (
        f"CncKept: its session of {DAY1} stopped without its day-close record (the process "
        "ended before the close) while holding a position, an order or a GTT, which this version "
        "continues only by a restart of bt.live that same day, before the close; "
        f"its stop GTT {trigger_id} of {DAY1} on "
        f"NSE:SBIN (CNC) has not ended and no day-close record carries it: {CHECK_BY_HAND}"
        f"{own_units_words(10)}",
    )
    assert [c for c in calls(world, *SENDS) if c.ts_ns >= DAY2_START] == []
    assert gtts(world) == [("active", 10, 497.0, "SELL")]


def own_units_words(units):
    """What a refusal adds when the books show `units` SBIN CNC held, worked by hand."""
    return (
        f". The books show {units} SBIN NSE (CNC) held: if any of them are units of your own, "
        "the start cannot tell them from the strategy's and never finds the leg closed, so "
        "once the strategy's units are sold and its GTTs cancelled, start it under another "
        "account name (account=)"
    )


@MODES
@BOOKS
def test_a_crash_settled_by_hand_beside_the_users_own_units_is_refused_naming_them(
    make_fake_openalgo, monkeypatch, tmp_path, analyzer, history
):
    # The operator cancels the stop GTT and sells the strategy's 10 units by hand; the
    # account still holds 50 SBIN of the user's own, which the books cannot tell from the
    # strategy's: the refusal names them and the way out that remains.
    world = crashed_monday(
        make_fake_openalgo, monkeypatch, tmp_path, analyzer=analyzer, history=history
    )
    (trigger_id,) = [gtt.trigger_id for gtt in world.fake.gtts()]
    world.fake.client().cancelgttorder(trigger_id=trigger_id)
    sell_by_hand(world)
    # 60 held for delivery, less the 10 sold today: 50 of the user's own.
    world.fake.set_holding("SBIN", "NSE", 60, average_price=480.0)
    assert held(world) == 50
    by_hand = len(calls(world, *SENDS))

    with pytest.raises(PreflightError) as refused:
        world.run(CncKept)

    assert refused.value.failures == (
        f"CncKept: its session of {DAY1} stopped without its day-close record (the process "
        "ended before the close) while holding a position, an order or a GTT, which this version "
        "continues only by a restart of bt.live that same day, before the close; "
        f"its stop GTT {trigger_id} of {DAY1} on "
        f"NSE:SBIN (CNC) has not ended and no day-close record carries it: {CHECK_BY_HAND}"
        f"{own_units_words(50)}",
    )
    assert calls(world, *SENDS)[by_hand:] == []


@pytest.mark.parametrize("settled", ["units sold", "gtt cancelled"])
def test_a_crash_settled_only_in_part_is_still_refused(
    make_fake_openalgo, monkeypatch, tmp_path, settled
):
    world = crashed_monday(make_fake_openalgo, monkeypatch, tmp_path, history=False)
    (trigger_id,) = [gtt.trigger_id for gtt in world.fake.gtts()]
    if settled == "units sold":
        sell_by_hand(world)
    else:
        world.fake.set_gtt_status(trigger_id, "cancelled")
    by_hand = len(calls(world, *SENDS))

    with pytest.raises(PreflightError):
        world.run(CncKept)

    assert calls(world, *SENDS)[by_hand:] == []


@MODES
@BOOKS
def test_a_crash_settled_by_hand_starts_flat_the_next_day_and_every_day_after(
    make_fake_openalgo, monkeypatch, tmp_path, caplog, analyzer, history
):
    # The operator cancels the stop GTT and sells the 10 units by hand before Tuesday's
    # start: the books show both closed, so the start ends the journal's GTT and begins
    # flat; Wednesday's start is no longer judged by Monday's save.
    world = crashed_monday(
        make_fake_openalgo, monkeypatch, tmp_path, history=history, analyzer=analyzer
    )
    world.trace.waive("books agree", "Monday's leg was closed by hand before Tuesday")
    (trigger_id,) = [gtt.trigger_id for gtt in world.fake.gtts()]
    world.fake.client().cancelgttorder(trigger_id=trigger_id)
    sell_by_hand(world)
    assert held(world) == 0
    by_hand = len(calls(world, *SENDS))
    world.stop_at(on(DAY2, 14, 46))

    tuesday = world.run(CncKept)

    assert tuesday.stopped_by == "ctrl_c"
    assert (
        f"CncKept: what it held since {DAY1} with no day-close record is closed at the "
        f"broker (none of SBIN NSE (CNC) held, none of its GTTs resting): it starts flat on "
        f"{DAY2}"
    ) in world.lines
    with journal_of(world) as journal:
        assert journal.open_gtts() == []
    # Its self.state kept the entry: it buys nothing more, and nothing is sent.
    assert calls(world, *SENDS)[by_hand:] == []
    assert not alerts(caplog, "error")
    world.stop = StopFlag()
    to_day(world, DAY3)
    world.stop_at(on(DAY3, 14, 46))

    assert world.run(CncKept).stopped_by == "ctrl_c"


class CncLimitEntry(CncKept):
    """CncKept whose entry is a DAY LIMIT buy at 495.00, below SBIN's 500.00."""

    def on_bar(self, bar):
        if not self.state.get("bought"):
            self.state["bought"] = True
            self.buy(self.qty, price=495.0, **type(self).rules)


@MODES
def test_a_crash_with_a_cnc_entry_working_refuses_the_next_start_once_it_filled(
    make_fake_openalgo, monkeypatch, tmp_path, analyzer
):
    # Monday's session crashes at 14:46 with its LIMIT entry working; SBIN trades at 494.00
    # after the crash, which fills it with no session running: the save holds no position,
    # yet the broker holds the strategy's 10 units on Tuesday.
    world = gtt_world(
        make_fake_openalgo, monkeypatch, tmp_path, analyzer=analyzer, trades_until=(14, 47)
    )
    world.trace.waive("protected", "the session crashed: its windows end with it")
    crashing(monkeypatch)
    crash_at(world, on(DAY1, 14, 46))
    run_to_crash(world, CncLimitEntry)
    assert held(world) == 0
    world.fake.trade("SBIN", "NSE", 494.0, volume=10.0)
    assert held(world) == 10
    world.day, world.stop, world.feed = DAY1, StopFlag(), None
    to_day(world, DAY2)
    sent = len(calls(world, *SENDS))

    with pytest.raises(PreflightError) as refused:
        world.run(CncLimitEntry)

    assert refused.value.failures == (
        f"CncLimitEntry: its session of {DAY1} stopped without its day-close record (the "
        "process ended before the close) while holding a position, an order or a GTT, which this "
        "version continues only by a restart of bt.live that same day, before the close: "
        f"{CHECK_BY_HAND}"
        f"{own_units_words(10)}",
    )
    assert calls(world, *SENDS)[sent:] == []


def an_earlier_days_gtt_order(world, trigger_id):
    """Record in the journal a GTT order of CncKept of three days before Monday, bound to
    `trigger_id`, that no day-close record carries; its journal `seq`."""
    with journal_of(world) as journal:
        journal.start_session(DAY1 - dt.timedelta(days=3), Mode.LIVE, BROKER)
        terms = GttTerms("BUY", 490.0, 490.0, 5)
        seq = journal.record_gtt(
            "CncKept", 1, 1, "SBIN", "NSE", "CNC", terms, purpose=Purpose.ENTRY
        )
        journal.bind_gtt(seq, trigger_id)
    return seq


def test_a_gtt_of_an_earlier_day_no_record_carries_refuses_the_start_while_it_rests(
    make_fake_openalgo, monkeypatch, tmp_path
):
    world = gtt_world(make_fake_openalgo, monkeypatch, tmp_path)
    placed = world.fake.client().placegttorder(
        symbol="SBIN",
        action="BUY",
        exchange="NSE",
        product="CNC",
        quantity=5,
        price=490.0,
        triggerprice_sl=490.0,
    )
    trigger_id = placed["trigger_id"]
    an_earlier_days_gtt_order(world, trigger_id)

    with pytest.raises(PreflightError) as refused:
        world.run(CncKept)

    earlier = DAY1 - dt.timedelta(days=3)
    assert refused.value.failures == (
        f"CncKept: its GTT order {trigger_id} of {earlier} on NSE:SBIN (CNC) has not ended "
        f"and no day-close record carries it: {CHECK_BY_HAND}",
    )
    assert len(calls(world, "placegttorder")) == 1
    assert calls(world, "placeorder") == []


def test_a_gtt_of_an_earlier_day_gone_from_the_broker_is_ended_and_the_start_goes_on(
    make_fake_openalgo, monkeypatch, tmp_path
):
    world = gtt_world(make_fake_openalgo, monkeypatch, tmp_path)
    world.stop_at(on(DAY1, 14, 46))
    seq = an_earlier_days_gtt_order(world, "261113000001")

    assert world.run(CncKept).stopped_by == "ctrl_c"

    with journal_of(world) as journal:
        assert journal.gtt(seq).state is GttState.ENDED


def test_a_restart_the_day_a_crash_was_settled_is_not_judged_by_the_earlier_save_again(
    make_fake_openalgo, monkeypatch, tmp_path, caplog
):
    # Tuesday's first start found Monday's leg closed and began flat; later that day the
    # account holds 10 SBIN of the user's own (bought by hand), and the session restarts:
    # it resumes Tuesday's save, and Monday's save is not set against the books again.
    world = crashed_monday(make_fake_openalgo, monkeypatch, tmp_path)
    world.trace.waive("books agree", "Monday's leg was closed by hand; the user's own units")
    (trigger_id,) = [gtt.trigger_id for gtt in world.fake.gtts()]
    world.fake.client().cancelgttorder(trigger_id=trigger_id)
    sell_by_hand(world)
    world.stop_at(on(DAY2, 14, 46))
    world.run(CncKept)
    world.fake.client().placeorder(
        strategy="manual",
        symbol="SBIN",
        action="BUY",
        exchange="NSE",
        price_type="MARKET",
        product="CNC",
        quantity=10,
    )
    by_hand = len(calls(world, *SENDS))

    outcome = restart(world, CncKept, stop_ns=on(DAY2, 14, 55))

    assert outcome.stopped_by == "ctrl_c"
    assert calls(world, *SENDS)[by_hand:] == []
