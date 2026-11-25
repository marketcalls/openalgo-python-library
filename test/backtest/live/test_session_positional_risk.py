"""The risk rules over a CNC leg's stop GTT in a live session, on the fake OpenAlgo
(20-positional-live 10.3 and 10.4; 16-session-design 5.4 and 6.2; the lead's ruling
N3b-6): the daily loss limit, and the account's limit through the kill switch, exit a CNC
leg with its stop GTT cancelled first (the exit sent only once the cancel is confirmed);
a cancel refused because the GTT fired leaves the exit to the GTT's child, with no second
exit; and a change of the analyzer switch locks the session and alerts with each position
and its stop GTT's status, after which nothing is sent and no GTT call is made.

The market, the strategies and the books are `positional_samples`': the entry fills at
500.00 at 14:42:01 IST, the stop GTT sells 10 at 497.00 or below (child limit 492.00).
"""

from __future__ import annotations

import logging

import pytest
from fake_openalgo.fixtures import *  # noqa: F403
from positional_samples import BOOKS, STOP_AT, CncHold, CncTarget, calls, gtt_world, sells
from session_samples import alerts, ist
from stops_samples import at_price

from openalgo.backtest._live._types import LOGGER_NAME
from openalgo.gtt import GTTAPI

#: A loss of 20.00 rupees on 10 units: SBIN at 498.00, above the stop's 497.00.
LOSS_PRICE, LIMIT = 498.0, 15.0
MOVE_AT = ist(14, 43, 30)
GTT_CALLS = ("placegttorder", "modifygttorder", "cancelgttorder", "gttorderbook")


@pytest.fixture(autouse=True)
def _logs(caplog):
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)


def cancel_then_exit(world):
    """The GTT calls and order placements in order, after the entry."""
    order = [entry.what for entry in calls(world, "cancelgttorder", "placeorder")]
    return order[1:]


@BOOKS
def test_a_daily_loss_breach_cancels_the_stop_gtt_before_its_exit(
    make_fake_openalgo, monkeypatch, tmp_path, history
):
    world = gtt_world(make_fake_openalgo, monkeypatch, tmp_path, history=history)
    at_price(world, MOVE_AT, LOSS_PRICE)
    world.stop_at(STOP_AT)

    outcome = world.run(CncHold, max_daily_loss=LIMIT)

    assert cancel_then_exit(world) == ["cancelgttorder", "placeorder"]
    assert sells(world) == [("MARKET", 10, "complete")]
    assert world.fake.position("SBIN", "NSE") == 0
    assert list(outcome.outputs["CncHold"]["trades"]["exit_reason"]) == ["daily_loss_limit"]


def test_the_account_limit_exits_a_cnc_leg_with_its_stop_gtt_cancelled_first(
    make_fake_openalgo, monkeypatch, tmp_path
):
    world = gtt_world(make_fake_openalgo, monkeypatch, tmp_path)
    at_price(world, MOVE_AT, LOSS_PRICE)
    world.stop_at(STOP_AT)

    world.run(CncHold, account_max_daily_loss=LIMIT)

    assert cancel_then_exit(world) == ["cancelgttorder", "placeorder"]
    assert sells(world) == [("MARKET", 10, "complete")]
    assert world.fake.position("SBIN", "NSE") == 0


def test_a_stop_gtt_cancel_refused_as_it_fired_leaves_the_exit_to_its_child(
    make_fake_openalgo, monkeypatch, tmp_path
):
    world = gtt_world(make_fake_openalgo, monkeypatch, tmp_path)
    at_price(world, MOVE_AT, 503.0)
    cancel = GTTAPI.cancelgttorder

    def fired_first(client, **fields):
        # SBIN gaps through the stop's trigger and the child's limit as the target's
        # cancel is sent: the GTT fires, its child rests, and the cancel is refused.
        world.fake.trade("SBIN", "NSE", 490.0)
        return cancel(client, **fields)

    monkeypatch.setattr(GTTAPI, "cancelgttorder", fired_first)
    world.stop_at(STOP_AT)

    world.run(CncTarget)

    # Only the child: the target's exit never goes beside it.
    assert [(price_type, units) for price_type, units, _ in sells(world)] == [("LIMIT", 10)]
    assert world.fake.position("SBIN", "NSE") == 0


def test_a_mode_switch_locks_the_session_alerts_the_stop_gtt_and_sends_nothing_more(
    make_fake_openalgo, monkeypatch, tmp_path, caplog
):
    world = gtt_world(make_fake_openalgo, monkeypatch, tmp_path)
    flip = ist(14, 43)
    world.script.at(flip, lambda: world.fake.set_analyzer(True))
    at_price(world, ist(14, 44), 503.0)
    world.stop_at(STOP_AT)

    outcome = world.run(CncTarget)

    assert outcome.stopped_by == "mode_breach"
    (alert,) = [a for a in alerts(caplog, "error") if "CncTarget SBIN NSE 10 units" in a]
    assert "stop GTT 261116000001 confirmed at the broker, trigger 497.00 for 10 units" in alert
    after = [entry.what for entry in world.fake.log.calls() if entry.ts_ns > flip]
    assert not set(after) & set(GTT_CALLS)
    assert "placeorder" not in after
