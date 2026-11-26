"""A carried stop GTT that fired on a trading day no session ran, found by the next start, on
the fake OpenAlgo (20-positional-live 4.3; the lead's ruling N5a-2): the trade book covers
only today, so the GTT's exit is recorded at its child's limit, flagged `price_unverified`
and `missed_day`, counted in today's figures, with a WARN alert; with no valid reference of
the leg's units from the day-close record it is a mismatch (entries locked, nothing sent),
and a position sold by hand on a missed day, its stop GTT still resting, is a mismatch too.
A child missed during a session is flagged `price_unverified` on its fill and its trade.

The days are `carry_samples`': on Monday `CncKept` buys 10 SBIN CNC at 500.00 with its stop
GTT (sell 10 at 497.00 or below, the child's limit 492.00), carried at the close. On Tuesday
no session runs: SBIN trades at 495.00 at 10:00 and 10:01, which fires the GTT and fills its
child. Wednesday's session starts at 14:40:05 IST and is stopped at 14:50.
"""

from __future__ import annotations

import logging

import pytest
from carry_samples import (
    DAY1,
    DAY2,
    DAY3,
    CncKept,
    carried_world,
    held,
    journal_of,
    missed_day,
    on,
    to_day,
)
from fake_openalgo.fixtures import *  # noqa: F403
from positional_samples import BOOKS, GTT_LIMIT, MODES, STOP_AT, calls, gtt_world
from session_samples import alerts
from stops_samples import at_price

from openalgo.backtest._live._carry_start import REFERENCES_KEY
from openalgo.backtest._live._result_build import session_result
from openalgo.backtest._live._types import LOGGER_NAME
from openalgo.backtest._live._venue_children import missed_child_id

#: The calls that send anything to the broker.
SENDS = ("placeorder", "modifyorder", "cancelorder", "placegttorder", "modifygttorder")
#: Tuesday's trades through the stop GTT's trigger.
THROUGH_THE_STOP = (((10, 0), 495.0), ((10, 1), 495.0))
DAY3_START = on(DAY3, 14, 40, 5)
BOTH_FLAGS = ("price_unverified", "missed_day")


@pytest.fixture(autouse=True)
def _logs(caplog):
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)


def wednesday(world):
    """Wednesday's session, stopped at 14:50."""
    to_day(world, DAY3)
    world.stop_at(on(DAY3, 14, 50))
    return world.run(CncKept)


def sent_on_wednesday(world):
    return [c.what for c in calls(world, *SENDS, "cancelgttorder") if c.ts_ns >= DAY3_START]


@MODES
@BOOKS
def test_a_stop_gtt_fired_on_a_missed_day_is_recorded_at_its_limit_with_both_flags(
    make_fake_openalgo, monkeypatch, tmp_path, caplog, analyzer, history
):
    world = carried_world(
        make_fake_openalgo, monkeypatch, tmp_path, history=history, analyzer=analyzer
    )
    (trigger_id,) = [gtt.trigger_id for gtt in world.fake.gtts()]
    missed_day(world, DAY2, THROUGH_THE_STOP)
    assert held(world) == 0

    outcome = wednesday(world)

    assert sent_on_wednesday(world) == []
    (fill,) = outcome.fills
    assert (fill.action, fill.units, fill.price, fill.flags) == ("SELL", 10, GTT_LIMIT, BOTH_FLAGS)
    mode = world.fake.mode.name
    assert alerts(caplog, "warn") == [
        f"{mode} (CncKept): CncKept: stop GTT {trigger_id} of SBIN NSE fired on a trading day "
        f"no session ran: its 10 units are recorded sold at the child's limit, {GTT_LIMIT:.2f}, "
        "flagged price_unverified, missed_day; the contract note holds the true price"
    ]
    assert alerts(caplog, "exit") == [
        f"{mode} (CncKept): SELL 10 SBIN NSE at {GTT_LIMIT:.2f} (exit: stop level, "
        f"order {missed_child_id(trigger_id)}, flagged price_unverified, missed_day)"
    ]
    # Counted in today's figures: the trade closes on Wednesday.
    assert outcome.summary.closed_trades == 1
    (trade,) = session_result(outcome).results["CncKept"].trades.itertuples()
    assert (trade.entry_time.date(), trade.exit_time.date()) == (DAY1, DAY3)
    assert set(BOTH_FLAGS) <= set(trade.flags)
    assert (
        f"CncKept: no session ran on {DAY2}, since its day-close record of {DAY1}: a stop GTT "
        "that fired then is recorded at its child's limit, flagged price_unverified and "
        "missed_day, and targets reached then were missed"
    ) in world.lines
    assert not alerts(caplog, "error")


def test_a_stop_gtt_gone_on_a_missed_day_with_no_settled_reference_is_a_mismatch(
    make_fake_openalgo, monkeypatch, tmp_path, caplog
):
    world = carried_world(make_fake_openalgo, monkeypatch, tmp_path, history=False)
    world.trace.waive("books agree", "the GTT's firing cannot be told: nothing is adopted")
    # The close's read of the leg's units counts only when the session's own fills had
    # settled: written here as one that had not, the record gives no reference.
    with journal_of(world) as journal:
        record = journal.day_closes()["CncKept"]
        carried = dict(record.carried)
        carried[REFERENCES_KEY] = [dict(carried[REFERENCES_KEY][0], settled=False)]
        journal.put_day_close("CncKept", carried, record.user_state, record.version_id, day=DAY1)
    (trigger_id,) = [gtt.trigger_id for gtt in world.fake.gtts()]
    missed_day(world, DAY2, THROUGH_THE_STOP)

    outcome = wednesday(world)

    assert sent_on_wednesday(world) == []
    assert outcome.fills == ()
    (locked,) = [a for a in alerts(caplog, "error") if f"stop GTT {trigger_id} (mismatch)" in a]
    assert "whether it fired cannot be told" in locked
    assert "Its entries are locked and nothing is exited" in locked
    assert not alerts(caplog, "warn")


def test_a_cnc_position_sold_by_hand_on_a_missed_day_is_a_mismatch_with_nothing_sent(
    make_fake_openalgo, monkeypatch, tmp_path, caplog
):
    world = carried_world(make_fake_openalgo, monkeypatch, tmp_path)
    world.trace.waive("books agree", "a sale by hand on a missed day: nothing is adopted")
    missed_day(world, DAY2, ())
    world.fake.client().placeorder(
        strategy="manual",
        symbol="SBIN",
        action="SELL",
        exchange="NSE",
        price_type="MARKET",
        product="CNC",
        quantity=10,
    )
    assert held(world) == 0

    outcome = wednesday(world)

    assert sent_on_wednesday(world) == []
    assert outcome.fills == ()
    mismatches = alerts(caplog, "reconcile_mismatch")
    assert mismatches and all("CncKept" in a for a in mismatches)
    assert any(lock.reason == "reconciliation" for lock in outcome.locks)


def test_a_child_missed_during_a_session_is_flagged_price_unverified_on_its_fill_and_trade(
    make_fake_openalgo, monkeypatch, tmp_path, caplog
):
    world = gtt_world(make_fake_openalgo, monkeypatch, tmp_path)
    world.stop_at(STOP_AT + 60 * 10**9)

    def hide_the_child():
        (child,) = [o for o in world.fake.orders() if o.price_type == "LIMIT"]
        world.fake.hide_order(child.order_id, reads=500)
        world.fake.hide_trades(child.order_id, reads=500)

    # The first trade at 495.00 (14:44:05) fires the GTT; its child is hidden at once.
    at_price(world, on(DAY1, 14, 44), 495.0)
    world.script.at(on(DAY1, 14, 44, 5), hide_the_child)

    outcome = world.run(CncKept)

    sells = [fill for fill in outcome.fills if fill.action == "SELL"]
    assert [(f.units, f.price, f.flags) for f in sells] == [(10, GTT_LIMIT, ("price_unverified",))]
    (trade,) = session_result(outcome).results["CncKept"].trades.itertuples()
    assert "price_unverified" in trade.flags and "missed_day" not in trade.flags
    (warned,) = alerts(caplog, "warn")
    assert "fired and its child order was not found" in warned
