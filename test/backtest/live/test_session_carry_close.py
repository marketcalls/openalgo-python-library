"""What a live session says of the CNC legs it carries, on the fake OpenAlgo (20-positional-
live 2 "Day 1, the close", 4.1; the lead's ruling N5a-4): one SESSION alert at every close
and every Ctrl+C that keeps positions lists each carried leg with its units, entry, stop
level and its stop GTT's trigger ID and status, and the daily summary has the same lines;
a stop GTT the close did not confirm is named once, with the advice to check the books,
then place it by hand; a finding about a carried GTT of a strategy that does not run in
the session is an ERROR alert; the result's trades show a trade carried from an earlier
day with its entry day.

The days and the strategies are `carry_samples`': on Monday `CncKept` buys 10 SBIN CNC at
500.00 at 14:42:01 IST with its stop GTT at 497.00, and the leg is carried at NSE's close
(15:30) or kept by a Ctrl+C at 14:46.
"""

from __future__ import annotations

import logging

import pytest
from carry_samples import DAY1, DAY2, CncKept, carried_world, on, to_day
from fake_openalgo.fixtures import *  # noqa: F403
from positional_samples import STOP_AT, gtt_world
from session_samples import alerts

import openalgo.backtest as bt
from openalgo.backtest._live._result_build import session_result
from openalgo.backtest._live._types import LOGGER_NAME

#: NSE's close on Monday.
CLOSE = on(DAY1, 15, 30)


@pytest.fixture(autouse=True)
def _logs(caplog):
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)


def carried_line(trigger_id):
    """The carried leg's line of the module docstring, its stop GTT confirmed."""
    return (
        f"CncKept SBIN NSE 10 units entered on {DAY1} at 500.00, stop 497.00: stop GTT "
        f"{trigger_id} confirmed at the broker, trigger 497.00 for 10 units"
    )


def test_the_close_alerts_each_carried_leg_and_the_daily_summary_names_it(
    make_fake_openalgo, monkeypatch, tmp_path, caplog
):
    world = carried_world(make_fake_openalgo, monkeypatch, tmp_path)
    (trigger_id,) = [gtt.trigger_id for gtt in world.fake.gtts()]

    assert world.first.stopped_by == "session_end"
    carried = [a for a in alerts(caplog, "session") if "Carried overnight" in a]
    assert carried == [
        "LIVE (CncKept): Carried overnight, each with its stop at the broker:\n"
        + carried_line(trigger_id)
    ]
    assert world.first.summary.carried == (carried_line(trigger_id),)
    (summary,) = alerts(caplog, "daily_summary")
    assert f"Carried     : {carried_line(trigger_id)}" in summary.splitlines()


def test_a_ctrl_c_that_keeps_the_leg_alerts_it_as_carried(
    make_fake_openalgo, monkeypatch, tmp_path, caplog
):
    world = gtt_world(make_fake_openalgo, monkeypatch, tmp_path)
    world.stop_at(STOP_AT)

    outcome = world.run(CncKept)

    assert outcome.stopped_by == "ctrl_c"
    (trigger_id,) = [gtt.trigger_id for gtt in world.fake.gtts()]
    carried = [a for a in alerts(caplog, "session") if "Carried overnight" in a]
    assert carried == [
        "LIVE (CncKept): Carried overnight, each with its stop at the broker:\n"
        + carried_line(trigger_id)
    ]


def test_a_stop_gtt_the_close_did_not_confirm_is_named_once_with_the_books_checked_first(
    make_fake_openalgo, monkeypatch, tmp_path, caplog
):
    world = gtt_world(
        make_fake_openalgo, monkeypatch, tmp_path, history=False, trades_until=(15, 32)
    )
    world.trace.waive("protected", "the stop GTT is cancelled by hand before the close")
    world.trace.left_unprotected_by_decision("CncKept", "cancelled by hand at the close")
    world.script.at(
        on(DAY1, 15, 29, 55),
        lambda: world.fake.set_gtt_status(world.fake.gtts()[0].trigger_id, "cancelled"),
    )

    outcome = world.run(CncKept)

    trigger_id = world.fake.gtts()[0].trigger_id
    not_confirmed = (
        f"stop GTT {trigger_id} not confirmed at the broker after the close (it is no longer "
        "resting after the close: gone from the GTT book)"
    )
    advice = (
        "check the order book and the position, then place its stop GTT in OpenAlgo by hand "
        "before the next open"
    )
    assert f"CncKept SBIN NSE 10 units: {not_confirmed}; {advice}" in world.lines
    assert outcome.summary.carried == (
        f"CncKept SBIN NSE 10 units entered on {DAY1} at 500.00, stop 497.00: "
        f"{not_confirmed}: {advice}",
    )
    (error,) = [a for a in alerts(caplog, "error") if "carried overnight" in a]
    assert error == (
        f"LIVE (CncKept): CncKept: SBIN NSE 10 units carried overnight, and its stop GTT "
        f"{trigger_id} is not confirmed (it is no longer resting after the close: gone from "
        "the GTT book); it may have fired: check the order book and the position first, then "
        "place its stop GTT in OpenAlgo by hand before the next open"
    )


class Idle(bt.Strategy):
    """An MIS strategy on SBIN that never trades."""

    symbol, exchange, interval = "SBIN", "NSE", "1m"

    def on_bar(self, bar):
        return None


def test_a_finding_about_a_carried_gtt_of_a_strategy_not_in_the_session_is_an_error_alert(
    make_fake_openalgo, monkeypatch, tmp_path, caplog
):
    world = carried_world(make_fake_openalgo, monkeypatch, tmp_path, history=True)
    world.trace.waive("books agree", "CncKept's CNC units: no strategy of Tuesday holds them")
    (trigger_id,) = [gtt.trigger_id for gtt in world.fake.gtts()]
    to_day(world, DAY2)
    world.fake.set_gtt_status(trigger_id, "cancelled")
    world.stop_at(on(DAY2, 14, 46))

    world.run(Idle)

    assert (
        f"LIVE (Idle): CncKept: GTT {trigger_id} of SBIN NSE ended at the broker, and "
        "CncKept does not run in this session: no engine follows it, so check the GTT and "
        "order books in OpenAlgo"
    ) in alerts(caplog, "error")


def test_the_results_trades_show_a_trade_carried_from_an_earlier_day_with_its_entry_day(
    make_fake_openalgo, monkeypatch, tmp_path
):
    world = carried_world(make_fake_openalgo, monkeypatch, tmp_path)
    to_day(world, DAY2)
    world.script.price = lambda now_ns: 496.0 if now_ns >= on(DAY2, 14, 42) else 500.0

    outcome = world.run(CncKept)

    trades = session_result(outcome).trades
    assert list(trades[["strategy", "entry_day"]].itertuples(index=False, name=None)) == [
        ("CncKept", DAY1)
    ]
    assert trades["exit_time"].iloc[0].date() == DAY2
