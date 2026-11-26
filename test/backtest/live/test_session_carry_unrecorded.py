"""A strategy whose session of an earlier day ended without its day-close record, at the next
day's start, on the fake OpenAlgo (20-positional-live 1 and 4.2 step 2; the lead's rulings
N5a-1 and N5c-2): an MIS strategy is never refused for it (the broker squares MIS off,
nothing is carried); a CNC strategy whose crash left a position or a stop GTT is continued
from its last save as from a record, by the record's rules against the broker's books
(`test_session_carry_crash`), and starts flat once the operator has closed the position and
the GTT by hand where the books tell that from a GTT that fired; one whose entry still
worked at the crash is refused when the books show units its save does not hold, which may
be the user's own.

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
from positional_samples import BOOKS, MODES, STOP_AT, MisHold, calls, gtt_world
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
def test_a_crash_settled_by_hand_beside_the_users_own_units_is_continued_selling_nothing(
    make_fake_openalgo, monkeypatch, tmp_path, analyzer, history
):
    # The operator cancels the stop GTT and sells the strategy's 10 units by hand; the
    # account still holds 50 SBIN of the user's own, which the books cannot tell from the
    # strategy's (7.2): the start continues the leg as a record's, and sends no order. A
    # book of GTT history shows the GTT cancelled (ended outside the session): it is placed
    # again, for the strategy's 10 units only. A broker's book of active GTTs only shows it
    # gone, as one that fired: the sale by hand, of its shape, is taken as its child (3.2).
    world = crashed_monday(
        make_fake_openalgo, monkeypatch, tmp_path, analyzer=analyzer, history=history
    )
    world.trace.waive("books agree", "the strategy's units sold by hand; 50 of the user's own")
    world.trace.waive("isolation", "a sale by hand of a fired GTT's shape is taken as its child")
    (trigger_id,) = [gtt.trigger_id for gtt in world.fake.gtts()]
    world.fake.client().cancelgttorder(trigger_id=trigger_id)
    sell_by_hand(world)
    world.fake.set_holding("SBIN", "NSE", 60, average_price=480.0)
    assert held(world) == 50
    by_hand = len(calls(world, *SENDS))
    world.stop_at(on(DAY2, 14, 46))

    assert world.run(CncKept).stopped_by == "ctrl_c"

    sent = [(c.what, c.detail["request"].get("quantity")) for c in calls(world, *SENDS)[by_hand:]]
    gone = not history and not analyzer
    assert sent == ([] if gone else [("placegttorder", 10)])
    assert held(world) == 50


def test_a_crash_with_its_units_sold_by_hand_and_its_gtt_resting_is_continued_selling_nothing(
    make_fake_openalgo, monkeypatch, tmp_path, caplog
):
    # Today's trade book shows the sale by hand: the start takes it as the leg's end (other
    # orders' trades), as for a record: nothing is sold or bought and no GTT is placed.
    world = crashed_monday(make_fake_openalgo, monkeypatch, tmp_path, history=False)
    world.trace.waive("books agree", "the strategy's units sold by hand before the start")
    world.trace.waive("isolation", "the sale by hand is the other order whose trade closed it")
    sell_by_hand(world)
    by_hand = len(calls(world, *SENDS))
    world.stop_at(on(DAY2, 14, 46))

    assert world.run(CncKept).stopped_by == "ctrl_c"

    assert held(world) == 0
    later = [c for c in calls(world, *SENDS)[by_hand:] if c.what in ("placeorder", "placegttorder")]
    assert later == []


def test_a_crash_whose_stop_gtt_was_cancelled_by_hand_is_continued_and_protected_again(
    make_fake_openalgo, monkeypatch, tmp_path, caplog
):
    world = crashed_monday(make_fake_openalgo, monkeypatch, tmp_path, history=False)
    (trigger_id,) = [gtt.trigger_id for gtt in world.fake.gtts()]
    world.fake.set_gtt_status(trigger_id, "cancelled")
    by_hand = len(calls(world, *SENDS))
    world.stop_at(on(DAY2, 14, 46))

    assert world.run(CncKept).stopped_by == "ctrl_c"

    (placed,) = calls(world, *SENDS)[by_hand:]
    assert (placed.what, placed.detail["request"]["quantity"]) == ("placegttorder", 10)
    assert held(world) == 10
    assert not alerts(caplog, "error")


@MODES
@BOOKS
def test_a_crash_settled_by_hand_starts_flat_the_next_day_and_every_day_after(
    make_fake_openalgo, monkeypatch, tmp_path, caplog, analyzer, history
):
    # The operator cancels the stop GTT and sells the 10 units by hand before Tuesday's
    # start: the books show both closed. A book that keeps the GTT's history shows it
    # cancelled, so the start ends the journal's GTT and begins flat; Wednesday's start is
    # no longer judged by Monday's save. A broker's book of active GTTs only shows it gone,
    # which 3.3 takes as fired when the leg's units fell by its units: the leg is continued
    # as a record's and the sale by hand, of its shape, is taken as its child.
    world = crashed_monday(
        make_fake_openalgo, monkeypatch, tmp_path, history=history, analyzer=analyzer
    )
    world.trace.waive("books agree", "Monday's leg was closed by hand before Tuesday")
    world.trace.waive("isolation", "a sale by hand of a fired GTT's shape is taken as its child")
    (trigger_id,) = [gtt.trigger_id for gtt in world.fake.gtts()]
    world.fake.client().cancelgttorder(trigger_id=trigger_id)
    sell_by_hand(world)
    assert held(world) == 0
    by_hand = len(calls(world, *SENDS))
    world.stop_at(on(DAY2, 14, 46))

    tuesday = world.run(CncKept)

    assert tuesday.stopped_by == "ctrl_c"
    settled = (
        f"CncKept: what it held since {DAY1} with no day-close record is closed at the "
        f"broker (none of SBIN NSE (CNC) held, none of its GTTs resting): it starts flat on "
        f"{DAY2}"
    )
    gone = not history and not analyzer
    assert (settled in world.lines) is not gone
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
    # the journal no fill, yet the broker holds 10 units on Tuesday, which the start cannot
    # tell from units of the user's own.
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
    (entry,) = [o for o in world.fake.orders(world.fake.mode) if o.action == "BUY"]
    world.day, world.stop, world.feed = DAY1, StopFlag(), None
    to_day(world, DAY2)
    sent = len(calls(world, *SENDS))

    with pytest.raises(PreflightError) as refused:
        world.run(CncLimitEntry)

    assert refused.value.failures == (
        f"CncLimitEntry: its entry order {entry.order_id} (BUY 10 SBIN) still worked when its "
        f"session of {DAY1} stopped without its day-close record, and the books show 10 units "
        "where its last save and the journal's updates hold 0: the start cannot tell what that "
        f"order filled after the stop from units of your own: {CHECK_BY_HAND}"
        f"{own_units_words(10)}",
    )
    assert calls(world, *SENDS)[sent:] == []


def test_a_crash_with_a_cnc_entry_working_that_never_filled_starts_flat(
    make_fake_openalgo, monkeypatch, tmp_path
):
    # The LIMIT entry still worked at the crash and never filled: the books show no unit,
    # so the rebuilt record holds nothing and the strategy starts flat.
    world = gtt_world(make_fake_openalgo, monkeypatch, tmp_path, trades_until=(14, 47))
    world.trace.waive("protected", "the session crashed: its windows end with it")
    crashing(monkeypatch)
    crash_at(world, on(DAY1, 14, 46))
    run_to_crash(world, CncLimitEntry)
    # The exchange's close cancels the entry the crashed session left working.
    (entry,) = [o for o in world.fake.orders(world.fake.mode) if o.status == "open"]
    world.fake.client().cancelorder(order_id=entry.order_id, strategy=entry.strategy)
    world.day, world.stop, world.feed = DAY1, StopFlag(), None
    to_day(world, DAY2)
    world.stop_at(on(DAY2, 14, 46))
    sent = len(calls(world, *SENDS))

    assert world.run(CncLimitEntry).stopped_by == "ctrl_c"

    assert calls(world, *SENDS)[sent:] == []
    assert held(world) == 0


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
