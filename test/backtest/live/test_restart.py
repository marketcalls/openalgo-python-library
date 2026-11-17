"""A crash and a restart of a live session, on the fake OpenAlgo (`_live/_restart_books.py`,
`_live/_session_restore.py`, `_live/_session_saved.py`; 16-session-design 3.1 steps 9 to
12, 3.6, 5.5, 6.2 and 7.2 rows F50 to F56, F65, F73, F75, F76 and F93, unit 12).

The scenarios are `restart_samples`': `StopWhenFlat` buys 10 SBIN at 14:42:01 IST with an
SL-M stop at 497.00, placed and confirmed by about 14:42:05. A crash drops the session
where it stands; a restart is a new session on the same journal. Expected values come
from the scenario and the fake's books.
"""

from __future__ import annotations

import logging

import pytest
from fake_openalgo.fixtures import *  # noqa: F403
from restart_samples import (
    Crash,
    StopWhenFlat,
    TrailsWhenFlat,
    crash_at,
    crashing,
    restart,
    run_to_crash,
    stops_working,
)
from session_samples import Hold, Rig, Script, alerts, ist
from stops_samples import at_price, check_invariants, events, stop_world
from test_stops_exits import TwoTrades, journal_stop, rising

from openalgo.backtest._live._types import LOGGER_NAME
from openalgo.backtest._live.orders import OrderPath
from openalgo.backtest.errors import JournalError, LiveError

#: The crash, while the strategy holds 10 SBIN with its stop working.
CRASHED = ist(14, 42, 30)
#: A price below the stop's trigger (497.00), rupees.
FALL_PRICE = 496.0


@pytest.fixture
def world(make_fake_openalgo, monkeypatch, tmp_path, caplog):
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    world = stop_world(make_fake_openalgo, monkeypatch, tmp_path)
    crashing(monkeypatch)
    return world


@pytest.fixture
def crashed(world):
    """The session crashed at 14:42:30 holding 10 SBIN, its stop working."""
    crash_at(world, CRASHED)
    run_to_crash(world, StopWhenFlat)
    return world


def stop_of(world):
    return next(o for o in world.orders() if o.price_type == "SL-M")


def placed_after(world, instant_ns: int) -> list:
    return [o for o in world.orders() if o.placed_ns > instant_ns]


# ----- F50: a restart in the middle of a position --------------------------------------


def test_F50_a_restart_resumes_the_position_with_its_stop_and_sends_nothing_new(crashed):
    outcome = restart(crashed, StopWhenFlat, stop_ns=ist(14, 45), on_stop="keep")
    assert placed_after(crashed, CRASHED) == []
    (left,) = outcome.open_positions
    assert (left.units, left.stop_order_id, left.stop_confirmed) == (
        10,
        stop_of(crashed).order_id,
        True,
    )
    assert outcome.restarts == 1


def test_F50_the_banner_says_it_is_a_restart_and_names_the_resumed_leg(crashed):
    restart(crashed, StopWhenFlat, stop_ns=ist(14, 45), on_stop="keep")
    first = crashed.lines.index("Restarted: start 2 of the day")
    assert crashed.lines[first - 1] == "OpenAlgo is in LIVE mode: orders go to zerodha"
    assert crashed.lines[first + 1].startswith("StopWhenFlat: resumed 10 SBIN NSE, stop ")
    assert "confirmed" in crashed.lines[first + 1]


def test_F50_the_restarted_engine_trades_on_from_where_it_was(crashed):
    # The price falls through the stop after the restart: the stop's fill, at the trade
    # that went through it, closes the leg.
    crashed.script.price = lambda now_ns: FALL_PRICE if now_ns >= ist(14, 44) else 500.0
    outcome = restart(crashed, StopWhenFlat, stop_ns=ist(14, 46), on_stop="keep")
    assert (stop_of(crashed).status, stop_of(crashed).average_price) == ("complete", FALL_PRICE)
    fills = [e for e in events(outcome, "fill") if e["order_id"] == stop_of(crashed).order_id]
    assert [(e["units"], e["price"]) for e in fills] == [(10, FALL_PRICE)]
    check_invariants(crashed.timeline, strategy="StopWhenFlat", unprotected_at_end=True)


def test_F50_trailing_resumes_after_the_restart(world):
    crash_at(world, CRASHED)
    run_to_crash(world, TrailsWhenFlat)
    rising(world, start=ist(14, 43), step=0.5, until=ist(14, 46))
    restart(world, TrailsWhenFlat, stop_ns=ist(14, 48), on_stop="keep")
    moved = [c for c in world.fake.log.calls("modifyorder") if c.ts_ns > CRASHED]
    assert moved
    (stop,) = stops_working(world)
    assert stop.trigger_price > 497.0
    assert placed_after(world, CRASHED) == []


# ----- F51 and F93: the stop filled while the session was down --------------------------


def test_F51_a_stop_that_filled_while_down_is_attributed_to_the_stop(crashed, caplog):
    crashed.fake.trade("SBIN", "NSE", FALL_PRICE, volume=10.0)
    outcome = restart(crashed, StopWhenFlat, stop_ns=ist(14, 45), on_stop="keep")
    assert stop_of(crashed).status == "complete"
    assert events(outcome, "adopted") == []
    assert outcome.reconciliations == ()
    assert alerts(caplog, "reconcile_mismatch") == []
    # No exit: the restarted strategy is flat and buys again at its next decision.
    sells = [
        o for o in placed_after(crashed, CRASHED) if o.action == "SELL" and o.price_type == "MARKET"
    ]
    assert sells == []


def test_F93_a_stop_fill_journaled_after_the_last_save_is_in_the_journal_position(world):
    # The stop fills at 14:42:35 and its word is journaled; the next save of the engine
    # would come with the 14:42 slice's release at 14:43:01, after the crash.
    world.script.at(
        ist(14, 42, 35), lambda: world.fake.trade("SBIN", "NSE", FALL_PRICE, volume=10.0)
    )
    crash_at(world, ist(14, 42, 50))
    run_to_crash(world, StopWhenFlat)
    outcome = restart(world, StopWhenFlat, stop_ns=ist(14, 44), on_stop="keep")
    assert events(outcome, "adopted") == []
    assert outcome.reconciliations == ()
    assert [o for o in placed_after(world, ist(14, 42, 50)) if o.action == "SELL"] == []
    check_invariants(world.timeline, strategy="StopWhenFlat", unprotected_at_end=True)


# ----- F52, F75, F76: the stop missing at the restart -------------------------------------


@pytest.fixture
def stop_cancelled(crashed):
    """While the session is down another client cancels its stop."""
    other = crashed.fake.client()
    other.cancelorder(order_id=stop_of(crashed).order_id, strategy="StopWhenFlat")
    return crashed


def test_F52_a_missing_stop_is_alerted_at_the_start_and_placed_again(stop_cancelled, caplog):
    outcome = restart(stop_cancelled, StopWhenFlat, stop_ns=ist(14, 45), on_stop="keep")
    assert any("no working stop at the broker" in text for text in alerts(caplog, "error"))
    (stop,) = stops_working(stop_cancelled)
    assert stop.quantity == 10
    (left,) = outcome.open_positions
    assert (left.stop_order_id, left.stop_confirmed) == (stop.order_id, True)


def test_F52_the_stop_is_working_before_the_banner(stop_cancelled):
    restart(stop_cancelled, StopWhenFlat, stop_ns=ist(14, 45), on_stop="keep")
    (stop,) = stops_working(stop_cancelled)
    banner = next(
        m.ts_ns for m in stop_cancelled.fake.telegram_sent if "Restarted: start 2" in m.message
    )
    assert stop.placed_ns < banner


def test_F75_a_stop_asked_for_in_the_countdown_flattens_after_the_stop_is_placed(
    stop_cancelled,
):
    # The banner is shown after the stop's confirmation; the countdown follows.
    outcome = restart(stop_cancelled, StopWhenFlat, stop_ns=ist(14, 42, 45))
    stop = [o for o in stop_cancelled.orders() if o.price_type == "SL-M"][-1]
    assert stop.placed_ns < ist(14, 42, 45)
    assert stop_cancelled.fake.position("SBIN", "NSE") == 0
    assert outcome.stopped_by == "ctrl_c"
    assert outcome.open_positions == ()


def test_F76_a_feed_that_refuses_to_connect_raises_with_the_stop_placed_again(stop_cancelled):
    stop_cancelled.fake.refuse_feed()
    with pytest.raises(LiveError, match="feed"):
        restart(stop_cancelled, StopWhenFlat)
    (stop,) = stops_working(stop_cancelled)
    assert stop.quantity == 10


# ----- F53: a broker position the journal does not hold -------------------------------


def test_F53_a_broker_position_on_a_traded_symbol_refuses_a_fresh_start(world):
    other = world.fake.client()
    other.placeorder(
        strategy="manual",
        symbol="SBIN",
        action="BUY",
        exchange="NSE",
        price_type="MARKET",
        product="MIS",
        quantity=10,
    )
    with pytest.raises(LiveError, match="10 units of SBIN NSE"):
        world.run(StopWhenFlat)
    assert [o.strategy for o in world.orders()] == ["manual"]


# ----- F54: an unbound intent at the restart -------------------------------------------


def test_F54_an_unbound_entry_is_settled_before_anything_is_sent(world):
    world.script.at(
        ist(14, 42), lambda: world.fake.faults.timeout("placeorder", reached_broker=True)
    )
    crash_at(world, ist(14, 42, 3))
    run_to_crash(world, StopWhenFlat)
    outcome = restart(world, StopWhenFlat, stop_ns=ist(14, 45), on_stop="keep")
    resolved = next(c.ts_ns for c in world.fake.log.calls("orderbook") if c.ts_ns > ist(14, 42, 3))
    sent = [c.ts_ns for c in world.fake.log.calls("placeorder") if c.ts_ns > ist(14, 42, 3)]
    assert sent and min(sent) >= resolved
    assert [o.action for o in world.orders()][:1] == ["BUY"]
    assert len([o for o in world.orders() if o.action == "BUY"]) == 1
    (left,) = outcome.open_positions
    assert left.stop_confirmed


# ----- F55: a restart past the square-off -----------------------------------------------


def test_F55_a_restart_past_the_square_off_flattens_in_the_start_phase(world):
    crash_at(world, ist(14, 58))
    run_to_crash(world, StopWhenFlat)
    world.script.sleep((ist(15, 1) - world.fake.clock.now()) / 10**9)
    outcome = restart(world, StopWhenFlat)
    assert world.fake.position("SBIN", "NSE") == 0
    assert [e["outcome"] for e in events(outcome, "restart_square_off")] == ["waits_for_cancels"]
    assert outcome.stopped_by == "session_end"


# ----- F56: the journal of another broker ----------------------------------------------


def test_F56_a_journal_bound_to_another_broker_refuses_the_start(
    world, make_fake_openalgo, tmp_path
):
    world.stop_at(ist(14, 40, 20))
    world.run(StopWhenFlat)
    upstox = make_fake_openalgo(start_ns=world.fake.clock.now(), broker="upstox")
    upstox.add_instrument("SBIN", "NSE")
    upstox.trade("SBIN", "NSE", 500.0, volume=10.0)
    other = Rig(upstox, Script(upstox), tmp_path)
    with pytest.raises(JournalError, match="zerodha"):
        other.run(StopWhenFlat, account="zerodha")
    assert upstox.orders() == []


# ----- F65: a semi-auto intent across a restart ------------------------------------------


def test_F65_a_queued_entry_is_rejected_at_the_restart_and_its_approval_is_a_mismatch(
    world, caplog
):
    world.script.at(ist(14, 42), lambda: world.fake.set_semi_auto(True))
    crash_at(world, ist(14, 42, 10))
    run_to_crash(world, Hold)
    world.fake.set_semi_auto(False)
    # The Action Center's approval places the order at the broker after the restart.
    approval = world.fake.client()
    world.script.at(
        ist(14, 43),
        lambda: approval.placeorder(
            strategy="Hold",
            symbol="SBIN",
            action="BUY",
            exchange="NSE",
            price_type="MARKET",
            product="MIS",
            quantity=10,
        ),
    )
    outcome = restart(world, Hold, stop_ns=ist(14, 45), on_stop="keep")
    errors = alerts(caplog, "error")
    assert any("Action Center" in text for text in errors)
    assert [row.kind for row in outcome.reconciliations][:1] == ["more"]
    assert [e["reason"] for e in events(outcome, "lock")] == ["reconciliation"]
    assert [o.strategy for o in world.orders()] == ["Hold"]


# ----- F73: the crash between the save and the stop's send -----------------------------


def test_F73_a_stop_saved_and_never_sent_is_placed_again_once(world, monkeypatch):
    send = OrderPath.send

    def crash_on_the_stop(path, intent):
        if intent.purpose.value == "stop":
            monkeypatch.setattr(OrderPath, "send", send)
            raise Crash("the stop's send")
        return send(path, intent)

    monkeypatch.setattr(OrderPath, "send", crash_on_the_stop)
    run_to_crash(world, StopWhenFlat)
    crashed_ns = world.fake.clock.now()
    outcome = restart(world, StopWhenFlat, stop_ns=ist(14, 45), on_stop="keep")
    (stop,) = stops_working(world)
    assert stop.placed_ns >= crashed_ns
    assert [o.action for o in world.orders()] == ["BUY", "SELL"]
    assert [e["kernel_order"] for e in events(outcome, "not_sent_before_restart")]
    assert any(w.reason for w in outcome.unprotected)


# ----- the day-long locks ---------------------------------------------------------------


def test_a_reconciliation_lock_is_set_again_at_the_restart(world):
    other = world.fake.client()
    world.script.at(
        ist(14, 43),
        lambda: other.placeorder(
            strategy="manual",
            symbol="SBIN",
            action="SELL",
            exchange="NSE",
            price_type="MARKET",
            product="MIS",
            quantity=6,
        ),
    )
    crash_at(world, ist(14, 44))
    run_to_crash(world, StopWhenFlat)
    outcome = restart(world, StopWhenFlat, stop_ns=ist(14, 46), on_stop="keep")
    restarted = [e for e in events(outcome, "lock") if e["ts_ns"] >= ist(14, 44)]
    assert [e["reason"] for e in restarted] == ["reconciliation"]
    (left,) = outcome.open_positions
    assert left.units == 4


# ----- F69's restart variant: the day's second trade crashes and restarts ---------------


def test_F69_a_restart_during_the_second_trade_keeps_its_stop_as_the_legs(world):
    # Trade 1 ends at its target at 14:43; trade 2 buys again at 14:45:01 with its stop;
    # the session crashes during trade 2 and restarts.
    at_price(world, ist(14, 43), 502.0)
    at_price(world, ist(14, 44), 500.0)
    crash_at(world, ist(14, 45, 30))
    run_to_crash(world, TwoTrades)
    outcome = restart(world, TwoTrades, stop_ns=ist(14, 47), on_stop="keep")
    stops = [(o.order_id, o.status) for o in world.orders() if o.price_type == "SL-M"]
    assert [status for _, status in stops] == ["cancelled", "trigger pending"]
    assert journal_stop(world, outcome, "TwoTrades") == stops[-1][0]
    (left,) = outcome.open_positions
    assert (left.stop_order_id, left.stop_confirmed) == (stops[-1][0], True)
