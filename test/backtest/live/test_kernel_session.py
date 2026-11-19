"""A live session's controls through the binding (`Engine.session`;
rust/oa_py/src/backtest/engine/session.rs): every "place" command names its order's role,
the session asks for exits, cancels while it waits, locks entries, caps the engine
default's sizing equity, reads the daily loss measure and stops where it stands; and it
rejects at submission an entry or add while entries are locked ("live_lock"), while
another order of the symbol works ("live_in_flight"), or beyond the freeze quantity
("freeze_qty"); a plain sale beyond what the orders closing the position leave is such an
entry, and a modification keeps an order's role.

No server is involved: the test plays the venue. The session is the NSE day 2025-07-01
on 1-minute slices from 09:15 IST, one instrument (SBIN), a 1-minute strategy, a capital
of 10,00,000 and bars flat at 500.00.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

import numpy as np
import pytest

from openalgo.backtest._calendar_convert import native_attr, to_day
from openalgo.backtest._reject_reasons import REASONS
from openalgo.backtest.calendar import default_calendar
from openalgo.backtest.errors import VenueRefusalError

DAY = dt.date(2025, 7, 1)
MINUTE_NS = 60 * 1_000_000_000
# 09:15 IST is 03:45 UTC.
OPEN_NS = int(dt.datetime(2025, 7, 1, 3, 45, tzinfo=dt.UTC).timestamp()) * 10**9
NO_BARS = (np.array([], dtype=np.int64), *(np.array([], dtype=np.float64) for _ in range(5)))


def sbin(freeze_qty: int | None = None) -> dict[str, Any]:
    return {
        "symbol": "SBIN",
        "exchange": "NSE",
        "kind": "EQ",
        "lot_size": 1,
        "tick_size": 0.05,
        "multiplier": 1.0,
        "freeze_qty": freeze_qty,
        "last_trading_day": None,
    }


def session_config() -> dict[str, Any]:
    """A cost-free, slippage-free 1-minute run on 1-minute bars of DAY."""
    return {
        "first_day": to_day(DAY),
        "last_day": to_day(DAY),
        "base_nanos": MINUTE_NS,
        "strategy_interval": MINUTE_NS,
        "fill": "next_open",
        "quantity_mode": "venue",
        "capital": 1_000_000.0,
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


def bar(minute: int) -> tuple[int, float, float, float, float, float]:
    """The bar of minute `minute` after 09:15: flat at 500.00."""
    return (OPEN_NS + minute * MINUTE_NS, 500.0, 500.0, 500.0, 500.0, 1_000.0)


def new_session(freeze_qty: int | None = None) -> Any:
    engine_class = native_attr("Engine")
    calendar = default_calendar()._cal
    return engine_class.session(calendar, session_config(), [sbin(freeze_qty)], [NO_BARS])


def new_backtest() -> Any:
    """A backtest of the bar of minute 0."""
    ts_open, *prices = bar(0)
    arrays = (
        np.array([ts_open], dtype=np.int64),
        *(np.array([price], dtype=np.float64) for price in prices),
    )
    return native_attr("Engine")(default_calendar()._cal, session_config(), [sbin()], [arrays])


def release_minute(engine: Any, minute: int) -> tuple:
    """Hand the session the bar of `minute`, release it and return its next pause."""
    assert engine.advance()[0] == "waiting"
    engine.add_bar(0, bar(minute))
    engine.release(minute + 1)
    return engine.advance()


def order(side: str, units: float, **fields: Any) -> dict[str, Any]:
    return {
        "instrument": 0,
        "side": side,
        "quantity": {"kind": "units", "value": units},
        "order_type": "MARKET",
        "validity": "DAY",
        "product": "MIS",
        "reduce_only": False,
        "reverse": False,
        **fields,
    }


EXIT = order("SELL", 1.0, quantity={"kind": "fraction", "value": 1.0}, reduce_only=True)
STOP = order("SELL", 100.0, order_type="SL-M", trigger=450.0, reduce_only=True)


def holding(units: float, freeze_qty: int | None = None) -> Any:
    """A session holding `units` of SBIN in MIS, bought at the close of minute 0 and filled
    at 500.00 in minute 1, paused at the close of minute 1 with no command left to take."""
    engine = new_session(freeze_qty)
    assert release_minute(engine, 0)[0] == "bar_close"
    outcome, order_id, _ = engine.submit(order("BUY", units))
    assert outcome == "submitted"
    assert engine.advance()[0] == "waiting"
    engine.apply_fill(order_id, units, 500.0, OPEN_NS + MINUTE_NS)
    engine.add_bar(0, bar(1))
    engine.release(2)
    assert engine.advance()[0] == "bar_close"
    engine.take_commands()
    return engine


def flat() -> Any:
    """A flat session paused at the close of minute 0."""
    engine = new_session()
    assert release_minute(engine, 0)[0] == "bar_close"
    return engine


def places(engine: Any) -> list[tuple[int, str, float, str, str | None]]:
    """(order id, side, units, role, reason) of every "place" command since the last take."""
    return [
        (c["order_id"], c["side"], c["quantity_units"], c["role"], c["reason"])
        for c in engine.take_commands()
        if c["command"] == "place"
    ]


@pytest.mark.parametrize(
    ("held", "request_", "role"),
    [
        (0.0, order("BUY", 10.0), "entry"),
        (100.0, order("BUY", 10.0), "add"),
        (100.0, order("SELL", 40.0), "exit"),
        (100.0, EXIT, "exit"),
        (100.0, STOP, "exit"),
        (100.0, order("SELL", 160.0, reverse=True), "entry"),
    ],
    ids=["entry", "add", "plain sale", "exit()", "stop", "reversal"],
)
def test_every_place_command_names_its_orders_role(held, request_, role):
    engine = holding(held) if held else flat()
    outcome, order_id, _ = engine.submit(request_)
    assert outcome == "submitted"
    assert [(i, r, reason) for i, _, _, r, reason in places(engine)] == [(order_id, role, None)]


@pytest.mark.parametrize(
    ("reason", "role", "risk_reason"),
    [
        ("square_off", "square_off", None),
        ("operator_stop", "risk_exit", "operator_stop"),
        ("kill_switch", "risk_exit", "kill_switch"),
        ("account_loss", "risk_exit", "account_loss"),
        ("unprotected", "risk_exit", "unprotected"),
        ("stop_rejected", "risk_exit", "stop_rejected"),
        ("stop_unfilled", "risk_exit", "stop_unfilled"),
    ],
)
def test_an_exit_the_session_asks_for_sells_the_whole_position_with_its_reason(
    reason, role, risk_reason
):
    engine = holding(100.0)
    assert engine.advance()[0] == "waiting"
    assert engine.request_exit(0, "MIS", reason) == "sent"
    assert places(engine) == [(2, "SELL", 100.0, role, risk_reason)]


def test_an_exit_asked_for_says_what_it_did():
    engine = holding(100.0)
    assert engine.submit(STOP)[0] == "submitted"
    assert engine.advance()[0] == "waiting"
    engine.take_commands()
    assert engine.request_exit(0, "MIS", "operator_stop") == "waits_for_cancels"
    assert engine.request_exit(0, "MIS", "kill_switch") == "already_exiting"


def test_an_exit_asked_for_a_flat_position_sends_nothing():
    engine = flat()
    assert engine.advance()[0] == "waiting"
    assert engine.request_exit(0, "MIS", "operator_stop") == "flat"
    assert engine.take_commands() == []


def test_an_unknown_exit_reason_is_an_input_error():
    engine = holding(100.0)
    with pytest.raises(native_attr("NativeEngineInputError"), match="unknown exit reason"):
        engine.request_exit(0, "MIS", "daily_loss_limit")


@pytest.mark.parametrize(
    "control",
    [
        lambda engine: engine.lock_entries("feed_dead"),
        lambda engine: engine.unlock_entries("feed_dead"),
        lambda engine: engine.cap_sizing_equity(100_000.0),
        lambda engine: engine.request_exit(0, "MIS", "operator_stop"),
        lambda engine: engine.stop_session(),
    ],
    ids=["lock_entries", "unlock_entries", "cap_sizing_equity", "request_exit", "stop_session"],
)
def test_a_session_control_is_refused_in_a_backtest(control):
    with pytest.raises(native_attr("NativeEngineError"), match="a session's controls") as raised:
        control(new_backtest())
    assert not isinstance(raised.value, native_attr("NativeEngineInputError"))


def test_a_cancellation_is_sent_while_waiting_and_again_after_the_venue_refused_it():
    engine = flat()
    outcome, order_id, _ = engine.submit(order("BUY", 10.0, order_type="LIMIT", price=450.0))
    assert outcome == "submitted"
    assert engine.advance()[0] == "waiting"
    engine.take_commands()
    cancel = {"command": "cancel", "order_id": order_id}
    assert engine.cancel(order_id) == "Submitted"
    assert engine.take_commands() == [cancel]
    engine.apply_cancel_refused(order_id)
    engine.cancel(order_id)
    assert engine.take_commands() == [cancel]


@pytest.mark.parametrize(
    ("held", "request_", "code"),
    [
        (0.0, order("BUY", 10.0), "live_lock"),
        (100.0, order("BUY", 10.0), "live_lock"),
        (100.0, order("SELL", 40.0), None),
        (100.0, EXIT, None),
        (100.0, STOP, None),
    ],
    ids=["entry", "add", "plain sale", "exit()", "stop"],
)
def test_while_entries_are_locked_entries_and_adds_are_rejected_and_exits_are_not(
    held, request_, code
):
    engine = holding(held) if held else flat()
    assert engine.lock_entries("feed_dead") is True
    outcome, _, reject_code = engine.submit(request_)
    assert (outcome, reject_code) == (("rejected", code) if code else ("submitted", None))


def test_entry_locks_are_reasons_in_a_set():
    engine = flat()
    assert engine.lock_entries("mode_breach") is True
    assert engine.lock_entries("mode_breach") is False
    assert engine.lock_entries("feed_dead") is True
    assert engine.entry_locks() == ["feed_dead", "mode_breach"]
    assert engine.unlock_entries("feed_dead") is True
    assert engine.unlock_entries("feed_dead") is False
    assert engine.submit(order("BUY", 10.0))[2] == "live_lock"
    assert engine.unlock_entries("mode_breach") is True
    assert engine.submit(order("BUY", 10.0))[0] == "submitted"


@pytest.mark.parametrize("reason", [" ", "feed_dead ", " feed_dead"])
def test_a_blank_lock_reason_or_one_with_spaces_around_it_is_an_input_error(reason):
    with pytest.raises(native_attr("NativeEngineInputError"), match="entry_lock"):
        new_session().lock_entries(reason)


def test_an_entry_while_another_order_of_the_symbol_works_is_rejected_live_in_flight():
    engine = flat()
    assert engine.submit(order("BUY", 10.0, order_type="LIMIT", price=450.0))[0] == "submitted"
    assert engine.submit(order("BUY", 10.0))[2] == "live_in_flight"


def test_a_plain_sale_beyond_what_the_orders_closing_the_long_leave_is_held_back():
    engine = holding(100.0)
    assert engine.submit(order("SELL", 100.0))[0] == "submitted"
    assert engine.submit(order("SELL", 100.0))[2] == "live_in_flight"


def test_a_modification_that_would_change_an_orders_role_is_an_input_error():
    engine = holding(100.0)
    outcome, order_id, _ = engine.submit(order("SELL", 100.0, order_type="LIMIT", price=520.0))
    assert outcome == "submitted"
    reversal = order("SELL", 250.0, order_type="LIMIT", price=520.0, reverse=True)
    with pytest.raises(native_attr("NativeEngineInputError"), match="keeps what the order is for"):
        engine.modify(order_id, reversal)


def test_an_add_beyond_the_freeze_quantity_is_rejected_freeze_qty():
    engine = holding(60.0, freeze_qty=100)
    assert engine.submit(order("BUY", 41.0))[2] == "freeze_qty"
    assert engine.submit(order("BUY", 40.0))[0] == "submitted"


DEFAULT_SIZED = order("BUY", 0.0, quantity={"kind": "sizer", "sizer": "default"})


@pytest.mark.parametrize(
    ("cap", "units"),
    # 2 percent of the budget is kept back at 500.00: floor(9,80,000 / 500) = 1,960 and
    # floor(98,000 / 500) = 196.
    [(None, 1_960.0), (100_000.0, 196.0), (5_000_000.0, 1_960.0)],
)
def test_the_engine_default_sizes_from_the_lesser_of_the_equity_and_the_cap(cap, units):
    engine = flat()
    if cap is not None:
        assert engine.cap_sizing_equity(cap) is None
    outcome, order_id, _ = engine.submit(DEFAULT_SIZED)
    assert outcome == "submitted"
    assert engine.order(order_id)["quantity_units"] == units


@pytest.mark.parametrize("rupees", [-1.0, float("nan"), float("inf")])
def test_a_sizing_cap_that_is_not_rupees_of_zero_or_more_is_an_input_error(rupees):
    with pytest.raises(native_attr("NativeEngineInputError"), match="sizing_equity_cap"):
        new_session().cap_sizing_equity(rupees)


def test_the_daily_loss_measure_is_the_strategys_pnl_of_the_day():
    # Bought 100 at 500.00; filled 20 more at 500.50 and marked at 500.00: -10.00.
    engine = holding(100.0)
    assert engine.daily_loss_measure() == 0.0
    outcome, order_id, _ = engine.submit(order("BUY", 20.0))
    assert outcome == "submitted"
    assert engine.advance()[0] == "waiting"
    engine.apply_fill(order_id, 20.0, 500.5, OPEN_NS + 2 * MINUTE_NS)
    assert release_minute(engine, 2)[0] == "bar_close"
    assert engine.daily_loss_measure() == -10.0


def test_a_stopped_session_reports_its_working_orders_and_positions_as_they_stand():
    engine = holding(100.0)
    outcome, _, _ = engine.submit(order("SELL", 100.0, order_type="LIMIT", price=600.0))
    assert outcome == "submitted"
    assert engine.advance()[0] == "waiting"
    engine.take_commands()
    kind, decided_at, bars, events, timers = engine.stop_session()
    assert (kind, decided_at, bars, timers) == ("finished", None, [], [])
    assert isinstance(events, list)
    assert engine.take_commands() == []
    result = engine.finish()
    assert result["orders"]["status"] == ["Filled", "Submitted"]
    position = result["final_positions"][0]
    assert (position["product"], position["quantity_units"]) == ("MIS", 100.0)


def test_a_stopped_session_takes_no_more_results():
    engine = holding(100.0)
    engine.stop_session()
    with pytest.raises(VenueRefusalError) as raised:
        engine.apply_fill(1, 1.0, 500.0, OPEN_NS + 2 * MINUTE_NS)
    assert raised.value.kind == "session_over"


def test_the_role_and_reason_vocabularies_are_the_kernels():
    assert native_attr("ORDER_ROLES") == (
        "entry",
        "add",
        "leg_stop",
        "exit",
        "risk_exit",
        "square_off",
    )
    assert native_attr("RISK_EXIT_REASONS") == (
        "daily_loss",
        "drawdown",
        "time_exit",
        "unprotected",
        "stop_rejected",
        "stop_unfilled",
        "kill_switch",
        "account_loss",
        "operator_stop",
    )
    assert native_attr("EXIT_REQUESTS") == (
        "sent",
        "waits_for_cancels",
        "already_exiting",
        "flat",
        "market_closed",
    )


def test_the_live_gates_say_what_happened_and_what_to_do():
    for code in ("live_lock", "live_in_flight"):
        meaning, advice = REASONS[code]
        assert "live session" in meaning and advice
    assert "live session" in REASONS["freeze_qty"][0]
