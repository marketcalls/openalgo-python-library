"""A restarted leg's stop, on the fake OpenAlgo (`_live/_session_restore.py`,
`_live/_restart_books.py`; 16-session-design 3.1 step 10, 3.6 and 5.5, unit 12): a stop
the restart cannot find a record of is placed again, never exited at market, and the
start's alert of a missing stop says what the restart will do about it.

The scenarios are `restart_samples`': `StopWhenFlat` buys 10 SBIN at 14:42:01 IST with an
SL-M stop at 497.00. A crash drops the session where it stands; a restart is a new session
on the same journal. Expected values come from the scenario and the fake's books.
"""

from __future__ import annotations

import logging

import pytest
from fake_openalgo.fixtures import *  # noqa: F403
from restart_samples import StopWhenFlat, crash_at, crashing, restart, run_to_crash
from session_samples import alerts, ist
from stops_samples import events, stop_world

from openalgo.backtest._live._types import LOGGER_NAME, Purpose
from openalgo.backtest._live._venue_saved import RestoredOrders

#: The crash, while the strategy holds 10 SBIN with its stop working.
CRASHED = ist(14, 42, 30)


@pytest.fixture
def world(make_fake_openalgo, monkeypatch, tmp_path, caplog):
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    world = stop_world(make_fake_openalgo, monkeypatch, tmp_path)
    crashing(monkeypatch)
    return world


def placed_after(world, instant_ns: int) -> list[tuple[str, int, str]]:
    return [
        (o.action, o.quantity, o.price_type) for o in world.orders() if o.placed_ns > instant_ns
    ]


def test_a_restored_stop_with_no_record_is_placed_again_and_no_exit_is_sent(world, monkeypatch):
    # The venue map's saved records lack the stop (3.6's "never recorded"): the saved
    # state's role says it is the leg's stop, so it is placed again (R3).
    crash_at(world, CRASHED)
    run_to_crash(world, StopWhenFlat)
    restore = RestoredOrders.restore

    def without_the_stop(self, saved, pairs):
        kept = {
            old: record
            for old, record in saved.items()
            if record.intent.purpose is not Purpose.STOP
        }
        return restore(self, kept, pairs)

    monkeypatch.setattr(RestoredOrders, "restore", without_the_stop)
    outcome = restart(world, StopWhenFlat, stop_ns=ist(14, 45), on_stop="keep")
    assert placed_after(world, CRASHED) == [("SELL", 10, "SL-M")]
    assert world.fake.position("SBIN", "NSE") == 10
    assert len(events(outcome, "not_sent_before_restart")) == 1


def test_a_missing_stop_of_a_leg_the_books_show_fewer_units_of_is_not_promised_again(world, caplog):
    # The entry fills 4 units, then 6 at 14:42:05; the position book stops showing fills
    # in between, so it shows 4 of the engine's 10 with no trade to explain the rest.
    # While the session is down another client cancels the stop.
    world.fake.split_next_fill(4)
    world.script.at(ist(14, 42, 3), world.fake.lag_positionbook)
    crash_at(world, CRASHED)
    run_to_crash(world, StopWhenFlat)
    other = world.fake.client()
    stop = next(o for o in world.orders() if o.price_type == "SL-M")
    other.cancelorder(order_id=stop.order_id, strategy="StopWhenFlat")
    restart(world, StopWhenFlat, stop_ns=ist(14, 45), on_stop="keep")
    [text] = [t for t in alerts(caplog, "error") if "no working stop at the broker" in t]
    assert "OpenAlgo's books show 4 units, which no trade explains, so no stop is placed" in text
    assert "placed again" not in text
    assert [o for o in world.orders() if o.placed_ns > CRASHED] == []
