"""A carried stop GTT whose fate a session ended before telling, on the fake OpenAlgo
(20-positional-live 3.3, 4.1 and 4.2; the lead's ruling N5a-1 (3) and review round 1 of
unit N5a): the close names it not confirmed, with the books to check first, and its
day-close record gives no reference of the leg's units the broker's books do not bear out,
so the next start never takes the GTT as ended outside the session and never places a stop
GTT for units the account does not hold.

The days and the strategies are `carry_samples`': on Monday `CncKept` buys 10 SBIN CNC at
500.00 with its stop GTT (sell 10 at 497.00 or below, the child's limit 492.00), carried at
the close. On Tuesday no session runs: SBIN trades at 495.00 at 10:00 and 10:01, which
fires the GTT and its child sells the 10 units. Wednesday's session starts at 14:40:05 IST
and is stopped by a Ctrl+C at 14:40:40, before the GTT follower's 60 s child search is
over. Thursday's session starts at 14:40:05 IST and is stopped at 14:50.
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
    carried_world,
    held,
    journal_of,
    missed_day,
    on,
    to_day,
)
from fake_openalgo.fixtures import *  # noqa: F403
from positional_samples import BOOKS, MODES, calls
from session_samples import StopFlag, alerts

from openalgo.backtest._live._carry_start import CONFIRMED_KEY, REFERENCES_KEY
from openalgo.backtest._live._types import LOGGER_NAME

#: Tuesday's trades through the stop GTT's trigger.
THROUGH_THE_STOP = (((10, 0), 495.0), ((10, 1), 495.0))
#: The calls that send anything to the broker.
SENDS = ("placeorder", "modifyorder", "cancelorder", "placegttorder", "modifygttorder")


@pytest.fixture(autouse=True)
def _logs(caplog):
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)


def wednesday_stopped_early(world):
    """Monday's leg carried, Tuesday missed with the GTT fired, Wednesday's session
    stopped by a Ctrl+C at 14:40:40."""
    world.trace.waive("books agree", "the GTT fired on Tuesday: its fall is in flight")
    world.trace.waive("protected", "the stop GTT fired on a day no session ran")
    missed_day(world, DAY2, THROUGH_THE_STOP)
    assert held(world) == 0
    to_day(world, DAY3)
    world.stop_at(on(DAY3, 14, 40, 40))
    return world.run(CncKept)


@MODES
@BOOKS
def test_a_close_before_the_follower_resolved_a_gone_stop_gtt_names_it_not_confirmed(
    make_fake_openalgo, monkeypatch, tmp_path, caplog, analyzer, history
):
    world = carried_world(
        make_fake_openalgo, monkeypatch, tmp_path, history=history, analyzer=analyzer
    )
    (trigger_id,) = [gtt.trigger_id for gtt in world.fake.gtts()]

    wednesday = wednesday_stopped_early(world)

    assert wednesday.stopped_by == "ctrl_c"
    (error,) = [a for a in alerts(caplog, "error") if "carried overnight" in a]
    assert f"its stop GTT {trigger_id} is not confirmed" in error
    assert "it may have fired: check the order book and the position first" in error
    (line,) = wednesday.summary.carried
    assert f"stop GTT {trigger_id} not confirmed at the broker after the close" in line
    assert line.endswith(
        "check the order book and the position, then place its stop GTT in OpenAlgo by hand "
        "before the next open"
    )
    (position,) = wednesday.open_positions
    assert position.stop_confirmed is False
    with journal_of(world) as journal:
        record = journal.day_closes()["CncKept"]
    assert record.day == DAY3
    # The broker's 0 units are not the leg's 10: no reference the next start may use.
    assert [ref["settled"] for ref in record.carried[REFERENCES_KEY]] == [False]
    assert record.carried[CONFIRMED_KEY] == []


@MODES
@BOOKS
def test_the_next_start_never_places_a_stop_gtt_for_units_the_account_does_not_hold(
    make_fake_openalgo, monkeypatch, tmp_path, caplog, analyzer, history
):
    world = carried_world(
        make_fake_openalgo, monkeypatch, tmp_path, history=history, analyzer=analyzer
    )
    wednesday_stopped_early(world)
    world.stop = StopFlag()
    to_day(world, DAY4)
    thursday_start = on(DAY4, 14, 40, 5)
    world.stop_at(on(DAY4, 14, 50))

    world.run(CncKept)

    assert [c for c in calls(world, *SENDS) if c.ts_ns >= thursday_start] == []
    assert held(world) == 0
    assert [g.status for g in world.fake.gtts(world.fake.mode) if g.status == "active"] == []


def wednesday_from_a_wrong_reference(world):
    """Monday's record written with a settled reference of 0 units (as no session of this
    version writes one for a leg of 10) and its stop GTT confirmed: measured from it, the
    GTT that fired on Tuesday looks like one that ended outside the session. Wednesday's
    session runs to 14:50, past the follower's child search."""
    world.trace.waive("books agree", "the GTT fired on Tuesday: the broker holds none")
    world.trace.waive("protected", "the stop GTT fired on a day no session ran")
    with journal_of(world) as journal:
        record = journal.day_closes()["CncKept"]
        carried = dict(record.carried)
        carried[REFERENCES_KEY] = [dict(carried[REFERENCES_KEY][0], units=0, settled=True)]
        journal.put_day_close("CncKept", carried, record.user_state, record.version_id, day=DAY1)
    missed_day(world, DAY2, THROUGH_THE_STOP)
    to_day(world, DAY3)
    world.stop_at(on(DAY3, 14, 50))
    return world.run(CncKept)


@MODES
def test_a_stop_gtt_gone_while_the_broker_holds_less_than_the_leg_is_never_placed_again(
    make_fake_openalgo, monkeypatch, tmp_path, caplog, analyzer
):
    world = carried_world(
        make_fake_openalgo, monkeypatch, tmp_path, history=False, analyzer=analyzer
    )
    (trigger_id,) = [gtt.trigger_id for gtt in world.fake.gtts()]

    wednesday_from_a_wrong_reference(world)

    assert [c for c in calls(world, *SENDS) if c.ts_ns >= on(DAY3, 14, 40, 5)] == []
    (locked,) = [a for a in alerts(caplog, "error") if "Its entries are locked" in a]
    assert f"stop GTT {trigger_id}" in locked


def test_its_lock_says_the_broker_holds_less_than_the_leg_and_it_may_have_fired(
    make_fake_openalgo, monkeypatch, tmp_path, caplog
):
    # The broker's book of active GTTs drops a fired GTT, so it is found gone (the sandbox's
    # book shows it triggered, which the follower never takes as ended).
    world = carried_world(make_fake_openalgo, monkeypatch, tmp_path, history=False)
    (trigger_id,) = [gtt.trigger_id for gtt in world.fake.gtts()]

    wednesday_from_a_wrong_reference(world)

    (locked,) = [a for a in alerts(caplog, "error") if "Its entries are locked" in a]
    assert locked.startswith(
        f"LIVE (CncKept): CncKept: stop GTT {trigger_id} ended at the broker with no child "
        "found while OpenAlgo's books show 0 of the leg's 10 units: it may have fired, so it "
        "is not placed again. Its entries are locked and nothing is exited"
    )
