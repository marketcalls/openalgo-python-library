"""A day-close record that a later session outdated, at a later start, on the fake OpenAlgo
(20-positional-live 4.2; the lead's ruling N5a-1, review round 2 of unit N5a): a session that
saved after a strategy's record and ended without a record of its own (a crash) changed
what the strategy holds, so the record is never continued. A later save that holds nothing
starts the strategy flat; one that holds a position is judged against the broker's books as
any save no record carries. The close writes the strategy's record of the day, flat.

The days and the strategies are `carry_samples`': on Monday `CncKept` or `CncKeptTrail`
buys 10 SBIN CNC at 500.00 at 14:42:01 IST with a stop GTT at 497.00 and carries it at
NSE's close. `CncKeptTrail` trails its stop 3 points below the best price and takes its
target at 506.00. A later day's session starts at 14:40:05 IST.
"""

from __future__ import annotations

import logging

import pytest
from carry_samples import (
    DAY1,
    DAY2,
    DAY3,
    DAY4,
    CncKept,
    CncKeptTrail,
    carried_world,
    held,
    journal_of,
    on,
    to_day,
)
from fake_openalgo.fixtures import *  # noqa: F403
from positional_samples import BOOKS, MODES, calls, gtt_world, gtts
from restart_samples import crash_at, crashing, restart, run_to_crash
from session_samples import StopFlag, alerts
from stops_samples import at_price

from openalgo.backtest._live._carry_unrecorded import CHECK_BY_HAND
from openalgo.backtest._live._types import LOGGER_NAME
from openalgo.backtest.errors import PreflightError

#: The calls that send anything to the broker.
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


def _buys_once_more(self, bar):
    if not self.state.get("again"):
        self.state["again"] = True
        self.buy(self.qty, **type(self).rules)


#: `CncKeptTrail` under its own name (so its journal rows and records are the same) that
#: buys 10 once more at its first decision of a later day, the entry kept in `self.state`.
BuysAgain = type("CncKeptTrail", (CncKeptTrail,), {"on_bar": _buys_once_more})


def next_start(world, day):
    """End the fake's day and move to `day`'s session start, stopped at 14:46."""
    world.stop, world.feed = StopFlag(), None
    to_day(world, day)
    world.stop_at(on(day, 14, 46))


def set_aside(name, record_day, saved_day, today):
    """The banner's line of a record a later session outdated, worked by hand."""
    return (
        f"{name}: its day-close record of {record_day} is not continued, since its session "
        f"of {saved_day} changed what it held and ended without a record of its own, and "
        f"nothing of that session is left to carry: it starts flat on {today}"
    )


def refusal_binding(trigger_id, *, record_day):
    """The refusal of `CncKept` whose session of Tuesday crashed before its close holding 10
    SBIN CNC and Monday's stop GTT `trigger_id`, which its outdated day-close record of
    `record_day` binds (the lead's ruling 1 of review round 4 of unit N5a), worked by hand."""
    return (
        f"CncKept: its session of {DAY2} stopped without its day-close record (the process "
        "ended before the close) while holding a position, an order or a GTT, which this version "
        "continues only by a restart of bt.live that same day, before the close; its stop GTT "
        f"{trigger_id} of {DAY1} on NSE:SBIN (CNC) has not ended: its day-close record of "
        f"{record_day} binds it, and a later session of {DAY2} outdates that record: "
        f"{CHECK_BY_HAND}. The books show 10 SBIN NSE (CNC) held: if any of them are units of "
        "your own, the start cannot tell them from the strategy's and never finds the leg "
        "closed, so once the strategy's units are sold and its GTTs cancelled, start it under "
        "another account name (account=)"
    )


def flat_record(world, name):
    """(the day of `name`'s latest day-close record, the units of its open positions)."""
    with journal_of(world) as journal:
        record = journal.day_closes()[name]
    units = [p["quantity_units"] for p in record.carried["positions"] if p["quantity_units"]]
    return record.day, units


def exited_on_tuesday(world, monkeypatch, *, own_units=0):
    """Tuesday's session continues Monday's carry of `CncKeptTrail`, sells its 10 units on
    its target (SBIN at 503.00 from 14:41, then 507.00 from 14:44) and crashes at 14:50,
    before its close; the account holds `own_units` SBIN of the user's own."""
    world.trace.waive("protected", "the session crashed: its windows end with it")
    crashing(monkeypatch)
    world.stop, world.feed = StopFlag(), None
    to_day(world, DAY2)
    if own_units:
        world.fake.set_holding("SBIN", "NSE", own_units, average_price=480.0)
    at_price(world, on(DAY2, 14, 41), 503.0)
    at_price(world, on(DAY2, 14, 44), 507.0)
    crash_at(world, on(DAY2, 14, 50))
    run_to_crash(world, CncKeptTrail)


@MODES
@BOOKS
def test_a_continued_day_that_exited_and_crashed_is_not_continued_again(
    make_fake_openalgo, monkeypatch, tmp_path, caplog, analyzer, history
):
    world = carried_world(
        make_fake_openalgo, monkeypatch, tmp_path, CncKeptTrail, analyzer=analyzer, history=history
    )
    exited_on_tuesday(world, monkeypatch)
    assert held(world) == 0
    sent = len(calls(world, *SENDS))
    next_start(world, DAY3)
    caplog.clear()

    wednesday = world.run(CncKeptTrail)

    assert wednesday.stopped_by == "ctrl_c"
    assert set_aside("CncKeptTrail", DAY1, DAY2, DAY3) in world.lines
    assert calls(world, *SENDS)[sent:] == []
    assert not alerts(caplog, "error")
    assert flat_record(world, "CncKeptTrail") == (DAY3, [])


@MODES
@BOOKS
def test_an_outdated_record_beside_the_users_own_units_places_nothing(
    make_fake_openalgo, monkeypatch, tmp_path, caplog, analyzer, history
):
    # The account holds 30 SBIN of the user's own, so a check of the strategy's 10 units
    # one-sided (20-positional-live 7.2) would pass: Monday's record must not be continued.
    world = carried_world(
        make_fake_openalgo, monkeypatch, tmp_path, CncKeptTrail, analyzer=analyzer, history=history
    )
    exited_on_tuesday(world, monkeypatch, own_units=30)
    sent = len(calls(world, *SENDS))
    next_start(world, DAY3)
    caplog.clear()

    wednesday = world.run(CncKeptTrail)

    assert wednesday.stopped_by == "ctrl_c"
    assert calls(world, *SENDS)[sent:] == []
    assert [g for g in gtts(world) if g[0] == "active"] == []
    assert not alerts(caplog, "error")


@MODES
@BOOKS
def test_a_crash_on_a_continued_day_settled_by_hand_starts_flat_on_every_later_day(
    make_fake_openalgo, monkeypatch, tmp_path, caplog, analyzer, history
):
    world = carried_world(
        make_fake_openalgo, monkeypatch, tmp_path, CncKept, analyzer=analyzer, history=history
    )
    world.trace.waive("protected", "the session crashed: its windows end with it")
    world.trace.waive("books agree", "Tuesday's leg was closed by hand before Wednesday")
    (trigger_id,) = [gtt.trigger_id for gtt in world.fake.gtts()]
    crashing(monkeypatch)
    world.stop, world.feed = StopFlag(), None
    to_day(world, DAY2)
    crash_at(world, on(DAY2, 14, 46))
    run_to_crash(world, CncKept)
    assert held(world) == 10
    next_start(world, DAY3)

    with pytest.raises(PreflightError) as refused:
        world.run(CncKept)

    assert refused.value.failures == (refusal_binding(trigger_id, record_day=DAY1),)
    client = world.fake.client()
    client.cancelgttorder(trigger_id=trigger_id)
    client.placeorder(
        strategy="manual",
        symbol="SBIN",
        action="SELL",
        exchange="NSE",
        price_type="MARKET",
        product="CNC",
        quantity=10,
    )
    by_hand = len(calls(world, *SENDS))
    world.stop, world.feed = StopFlag(), None
    world.stop_at(on(DAY3, 14, 46))
    caplog.clear()

    assert world.run(CncKept).stopped_by == "ctrl_c"
    assert flat_record(world, "CncKept") == (DAY3, [])
    next_start(world, DAY4)
    assert world.run(CncKept).stopped_by == "ctrl_c"

    assert calls(world, *SENDS)[by_hand:] == []
    assert not alerts(caplog, "error")


@MODES
@BOOKS
def test_a_restart_that_exited_after_a_kept_ctrl_c_and_crashed_is_not_continued(
    make_fake_openalgo, monkeypatch, tmp_path, caplog, analyzer, history
):
    # Monday's Ctrl+C at 14:46 keeps the leg and writes Monday's record; a restart that
    # day sells the 10 units on its target and crashes before the close: the save it made
    # after the record outdates it, though both are of Monday.
    world = gtt_world(
        make_fake_openalgo,
        monkeypatch,
        tmp_path,
        analyzer=analyzer,
        history=history,
        trades_until=(15, 32),
    )
    world.trace.waive("protected", "the session crashed: its windows end with it")
    world.stop_at(on(DAY1, 14, 46))
    assert world.run(CncKeptTrail).stopped_by == "ctrl_c"
    assert flat_record(world, "CncKeptTrail") == (DAY1, [10.0])
    crashing(monkeypatch)
    world.stop, world.feed = StopFlag(), None
    at_price(world, on(DAY1, 14, 48), 503.0)
    at_price(world, on(DAY1, 14, 50), 507.0)
    crash_at(world, on(DAY1, 14, 55))
    run_to_crash(world, CncKeptTrail)
    assert held(world) == 0
    sent = len(calls(world, *SENDS))
    world.day = DAY1
    next_start(world, DAY2)
    caplog.clear()

    tuesday = world.run(CncKeptTrail)

    assert tuesday.stopped_by == "ctrl_c"
    assert set_aside("CncKeptTrail", DAY1, DAY1, DAY2) in world.lines
    assert calls(world, *SENDS)[sent:] == []
    assert not alerts(caplog, "error")


@MODES
@BOOKS
def test_a_restart_the_day_a_record_was_set_aside_names_the_position_it_resumes(
    make_fake_openalgo, monkeypatch, tmp_path, caplog, analyzer, history
):
    # Wednesday's first start sets Monday's record aside, buys 10 again at 500.00 with its
    # stop GTT at 497.00 and crashes at 14:46; the restart resumes that position, so its
    # banner never says the strategy starts flat (review round 3 of unit N5a).
    world = carried_world(
        make_fake_openalgo, monkeypatch, tmp_path, CncKeptTrail, analyzer=analyzer, history=history
    )
    world.trace.waive("books agree", "a restart: the earlier start's reads are its own")
    exited_on_tuesday(world, monkeypatch)
    world.stop, world.feed = StopFlag(), None
    to_day(world, DAY3)
    world.script.price = lambda now_ns: 500.0
    crash_at(world, on(DAY3, 14, 46))
    run_to_crash(world, BuysAgain)
    assert set_aside("CncKeptTrail", DAY1, DAY2, DAY3) in world.lines
    assert held(world) == 10
    (trigger_id,) = [g.trigger_id for g in world.fake.gtts(world.fake.mode) if g.status == "active"]
    first, sent = len(world.lines), len(calls(world, *SENDS))
    caplog.clear()

    outcome = restart(world, BuysAgain, stop_ns=on(DAY3, 14, 50))

    assert outcome.stopped_by == "ctrl_c"
    lines = world.lines[first:]
    assert [line for line in lines if "starts flat" in line] == []
    assert (
        f"CncKeptTrail: resumed 10 SBIN NSE, stop GTT {trigger_id} confirmed at the broker, "
        "trigger 497.00 for 10 units"
    ) in lines
    assert calls(world, *SENDS)[sent:] == []
    assert not alerts(caplog, "error")


@MODES
def test_a_restart_that_crashed_after_a_kept_ctrl_c_is_refused_naming_the_record_of_its_gtt(
    make_fake_openalgo, monkeypatch, tmp_path, caplog, analyzer
):
    # Tuesday continues Monday's carry and is stopped at 14:46, keeping the leg: Tuesday's
    # record binds Monday's stop GTT. A restart that day crashes at 14:50 with nothing
    # changed, so its save outdates that record: Wednesday's start is refused, naming the
    # record that binds the GTT and the restart that would have continued it.
    world = carried_world(make_fake_openalgo, monkeypatch, tmp_path, CncKept, analyzer=analyzer)
    world.trace.waive("protected", "the session crashed: its windows end with it")
    (trigger_id,) = [gtt.trigger_id for gtt in world.fake.gtts()]
    next_start(world, DAY2)
    assert world.run(CncKept).stopped_by == "ctrl_c"
    assert flat_record(world, "CncKept") == (DAY2, [10.0])
    crashing(monkeypatch)
    world.stop, world.feed = StopFlag(), None
    crash_at(world, on(DAY2, 14, 50))
    run_to_crash(world, CncKept)
    sent = len(calls(world, *SENDS))
    next_start(world, DAY3)

    with pytest.raises(PreflightError) as refused:
        world.run(CncKept)

    assert refused.value.failures == (refusal_binding(trigger_id, record_day=DAY2),)
    assert calls(world, *SENDS)[sent:] == []
