"""A live session's close with CNC legs, on the fake OpenAlgo (20-positional-live 2 "Day 1,
the close", 4.1, 7.2 and 7.3; the lead's rulings N3b-4 and N3b-5): MIS legs are squared off
and CNC legs carried with their stop GTTs, confirmed by a read of the GTT book after the
close; the session then ends `session_end` and writes one day-close record per strategy
that carries a leg, which the kernel's `restore_carried_state` takes in the next day's
session. A CNC strategy beside the user's own holdings of its symbol is checked
one-sided: the user's units are reported once and never sold.

The market and the strategies are `positional_samples`': SBIN trades every 10 s at 500.00
until 15:32 IST unless a test moves it; the entries fill at 500.00 at 14:42:01; NSE squares
MIS off at 15:00 and closes at 15:30.
"""

from __future__ import annotations

import datetime as dt
import logging

import numpy as np
import pytest
from fake_openalgo import BROKER
from fake_openalgo.fixtures import *  # noqa: F403
from positional_samples import STOP_AT, CncHold, CncTarget, MisHold, calls, gtt_world, gtts
from session_samples import CAPITAL, DAY, alerts, ist
from stops_samples import at_price

from openalgo.backtest._calendar_convert import native_attr, to_day
from openalgo.backtest._live._session_carry import REFERENCES_KEY
from openalgo.backtest._live._types import LOGGER_NAME, Mode
from openalgo.backtest._live.journal import JOURNAL_FILE_NAME, Journal
from openalgo.backtest.calendar import default_calendar

#: NSE's close on the module's day, UTC nanoseconds.
CLOSE = ist(15, 30)


@pytest.fixture(autouse=True)
def _logs(caplog):
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)


def to_the_close(make_fake_openalgo, monkeypatch, tmp_path, **options):
    """A world whose SBIN trades until 15:32 IST, past NSE's close."""
    return gtt_world(make_fake_openalgo, monkeypatch, tmp_path, trades_until=(15, 32), **options)


def day_closes(world):
    """The day-close records of the fake's mode's ledger, by strategy."""
    path = world.folder / "live" / BROKER / JOURNAL_FILE_NAME
    with Journal.open(
        path, account=BROKER, broker=BROKER, mode=world.fake.mode, now=world.fake.clock.now
    ) as journal:
        return journal.day_closes()


def test_an_mis_and_a_cnc_strategy_at_the_close_square_off_mis_and_carry_cnc(
    make_fake_openalgo, monkeypatch, tmp_path, caplog
):
    world = to_the_close(make_fake_openalgo, monkeypatch, tmp_path)

    outcome = world.run([MisHold, CncHold])

    assert outcome.stopped_by == "session_end"
    # The MIS leg's 10 sold at the square-off; the CNC leg's 10 carried with its GTT.
    assert world.fake.position("SBIN", "NSE") == 10
    assert [(status, units) for status, units, *_ in gtts(world)] == [("active", 10)]
    reads_after_close = [c for c in calls(world, "gttorderbook") if c.ts_ns >= CLOSE]
    assert reads_after_close
    errors = alerts(caplog, "error")
    assert not [a for a in errors if "CncHold" in a]
    assert not [a for a in errors if "still not flat after square-off" in a]


def test_a_cnc_leg_carried_at_the_close_writes_one_record_the_next_days_kernel_restores(
    make_fake_openalgo, monkeypatch, tmp_path
):
    world = to_the_close(make_fake_openalgo, monkeypatch, tmp_path)

    world.run([MisHold, CncHold])

    records = day_closes(world)
    assert sorted(records) == ["CncHold"]
    record = records["CncHold"]
    assert (record.day, record.mode) == (DAY, Mode.LIVE)
    (reference,) = record.carried[REFERENCES_KEY]
    assert (reference["symbol"], reference["product"], reference["units"]) == ("SBIN", "CNC", 10)
    assert reference["read_ns"] >= CLOSE
    restored = next_day(record.carried).restore_carried_state(record.carried)
    assert list(restored["gtts"]) == [1]


@pytest.mark.parametrize("analyzer", [False, True], ids=["broker", "sandbox"])
def test_ctrl_c_by_default_keeps_the_cnc_leg_flattens_the_mis_one_and_writes_the_record(
    make_fake_openalgo, monkeypatch, tmp_path, analyzer
):
    world = gtt_world(make_fake_openalgo, monkeypatch, tmp_path, analyzer=analyzer)
    world.stop_at(STOP_AT)

    outcome = world.run([MisHold, CncHold])

    assert outcome.stopped_by == "ctrl_c"
    assert world.fake.position("SBIN", "NSE") == 10
    assert sorted(day_closes(world)) == ["CncHold"]


def test_a_session_with_no_carried_leg_writes_no_record(make_fake_openalgo, monkeypatch, tmp_path):
    world = gtt_world(make_fake_openalgo, monkeypatch, tmp_path)
    world.stop_at(STOP_AT)

    world.run(MisHold)

    assert day_closes(world) == {}


def test_a_cnc_strategy_beside_the_users_own_holdings_sells_only_its_own_units(
    make_fake_openalgo, monkeypatch, tmp_path, caplog
):
    world = gtt_world(make_fake_openalgo, monkeypatch, tmp_path)
    world.trace.waive("books agree", "the user's own 50 CNC units are no difference (7.2)")
    own = world.fake.client()
    own.placeorder(
        strategy="manual",
        symbol="SBIN",
        action="BUY",
        exchange="NSE",
        price_type="MARKET",
        product="CNC",
        quantity=50,
    )
    at_price(world, ist(14, 43, 30), 503.0)

    run = world.run
    world.stop_at(STOP_AT)
    run(CncTarget)

    assert world.fake.position("SBIN", "NSE") == 50
    session_sells = [o.quantity for o in world.orders() if o.action == "SELL"]
    assert session_sells == [10]
    (held,) = [a for a in alerts(caplog, "session") if "beyond the strategy's" in a]
    assert "holds 50 units of SBIN NSE (CNC) beyond the strategy's" in held
    assert alerts(caplog, "reconcile_mismatch") == []


def test_cnc_units_sold_by_hand_below_the_strategys_are_a_mismatch_with_no_exit(
    make_fake_openalgo, monkeypatch, tmp_path, caplog
):
    world = gtt_world(make_fake_openalgo, monkeypatch, tmp_path)
    world.trace.waive("protected", "the sale by hand leaves the stop GTT larger than the rest")
    other = world.fake.client()
    sale = {
        "strategy": "manual",
        "symbol": "SBIN",
        "action": "SELL",
        "exchange": "NSE",
        "price_type": "MARKET",
        "product": "CNC",
        "quantity": 5,
    }
    world.script.at(ist(14, 43), lambda: other.placeorder(**sale))
    world.stop_at(STOP_AT)

    outcome = world.run(CncHold)

    assert alerts(caplog, "reconcile_mismatch")
    locks = {(lock.strategy, lock.reason) for lock in outcome.locks}
    assert ("CncHold", "reconciliation") in locks
    assert [o.quantity for o in world.orders() if o.action == "SELL"] == [5]


#: SBIN as the session's kernel holds it (OpenAlgo's facts of the fake's SBIN).
SBIN = {
    "symbol": "SBIN",
    "exchange": "NSE",
    "kind": "EQ",
    "lot_size": 1,
    "tick_size": 0.05,
    "multiplier": 1.0,
    "freeze_qty": None,
    "last_trading_day": None,
}


def next_day(carried):
    """The kernel session of the trading day after the module's day for the strategy whose
    carried state is `carried` (SBIN on NSE): its capital, 1-minute slices from the open,
    the catch-up's first bar released."""
    assert carried["instruments"] == [{"symbol": "SBIN", "exchange": "NSE"}]
    day = dt.date(2026, 11, 17)
    config = {
        "first_day": to_day(day),
        "last_day": to_day(day),
        "base_nanos": 60 * 10**9,
        "strategy_interval": 60 * 10**9,
        "fill": "next_open",
        "quantity_mode": "venue",
        "capital": float(CAPITAL),
        "costs": None,
        "slippage": ("zero", 0.0, 0),
        "risk": {
            "risk_basis": "gross",
            "after_risk_exit": "lock_session",
            "round_levels": True,
            "strict_config": True,
        },
        "deliver_events": False,
    }
    bars = (np.array([], dtype=np.int64), *(np.array([], dtype=np.float64) for _ in range(5)))
    engine = native_attr("Engine").session(default_calendar()._cal, config, [SBIN], [bars])
    opens = default_calendar().session("NSE", day)[0]
    assert engine.advance()[0] == "waiting"
    engine.add_bar(0, (opens, 500.0, 500.0, 500.0, 500.0, 1_000.0))
    engine.release(1)
    engine.advance()
    return engine
