"""Words a trader reads in a live session, made to say what is true (the lead's ruling N5c-4,
from the docs unit N6a's run of every live page on the fake): an EXIT alert names the exit
reason its trade records (a), the unprotected window between an entry's fill and its stop
GTT's confirmation is named for what it is (c), a stop GTT being modified says so (d), the
example for a stock future names its settlement (e), and the preflight's MIS square-off
words appear only with an MIS strategy (i).

Expected words are written out here, from the rulings; none is produced by the code under
test.
"""

from __future__ import annotations

import datetime as dt
from types import SimpleNamespace

import pandas as pd
import pytest
from carry_samples import CncKept
from fake_openalgo.fixtures import *  # noqa: F403
from plan_samples import specify
from positional_samples import STOP_AT, gtt_world
from preflight_fakes import ist_ns

import openalgo.backtest as bt
from openalgo.backtest._live._exit_reasons import alert_label
from openalgo.backtest._live._plan_stock import stock_derivative
from openalgo.backtest._live._preflight_local import trading_day_checks
from openalgo.backtest._live._result_build import session_result
from openalgo.backtest._live._stops_cover import (
    STOP_ENDED,
    STOP_GTT_UNCONFIRMED,
    STOP_UNCONFIRMED,
    reason_of,
)
from openalgo.backtest._live._stops_gtt import gtt_words
from openalgo.backtest._live._stops_orders import LegOrders
from openalgo.backtest._live._types import Purpose
from openalgo.backtest.calendar import default_calendar
from openalgo.backtest.errors import ReferenceDataError

# (a) ------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("purpose", "code", "label"),
    [
        (Purpose.RISK_EXIT, "expiry_exit", "expiry_exit"),
        (Purpose.RISK_EXIT, "leg_sl", "leg_sl"),
        (Purpose.RISK_EXIT, "account_loss_limit", "account_loss_limit"),
        # The session's own risk exits the kernel records as manual keep their purpose: their
        # words say why (Ctrl+C, a stop not confirmed, the kill switch).
        (Purpose.RISK_EXIT, "manual", "risk_exit"),
        (Purpose.EXIT, "leg_sl", "exit"),
        (Purpose.ENTRY, None, "entry"),
    ],
)
def test_an_alert_names_the_exit_reason_an_engine_risk_exit_records(purpose, code, label):
    assert alert_label(purpose, code) == label


# (c) ------------------------------------------------------------------------------------------


def test_a_gtt_legs_window_before_its_stop_gtt_is_confirmed_is_named_for_that():
    leg = LegOrders(guarded=True)

    assert reason_of(leg, held_cancel=False, retries=False, gtt_asked=True) == STOP_GTT_UNCONFIRMED
    assert STOP_GTT_UNCONFIRMED == "stop GTT not confirmed yet"


@pytest.mark.parametrize("analyzer", [False, True], ids=["broker", "sandbox"])
def test_a_cnc_entrys_window_before_its_stop_gtt_is_confirmed_is_named_so_on_the_fake(
    make_fake_openalgo, monkeypatch, tmp_path, analyzer
):
    # CncKept buys 10 SBIN CNC at 14:42:01 IST, filled at 14:42:02; the order worker places
    # its stop GTT and the next read of the GTT book, at 14:42:04, shows it resting.
    world = gtt_world(
        make_fake_openalgo, monkeypatch, tmp_path, trades_until=(15, 32), analyzer=analyzer
    )
    world.stop_at(STOP_AT)

    outcome = world.run(CncKept)

    windows = session_result(outcome).unprotected
    assert windows[["units", "reason", "seconds"]].to_dict("records") == [
        {"units": 10, "reason": "stop GTT not confirmed yet", "seconds": 2.0}
    ]


def test_a_leg_with_no_stop_order_and_no_stop_gtt_asked_keeps_its_stop_ended():
    assert reason_of(LegOrders(guarded=True), held_cancel=False, retries=False) == STOP_ENDED
    assert STOP_UNCONFIRMED != STOP_ENDED


# (d) ------------------------------------------------------------------------------------------


def gtt_trader(view):
    """A strategy whose kernel holds the stop GTT `view` (an `Engine.gtts` dict), confirmed
    resting by the GTT map as 261116000001."""
    known = SimpleNamespace(
        trigger_id="261116000001", unconfirmed="", unsettled=False, cancel_sent=False
    )
    return SimpleNamespace(
        product="CNC",
        engine=SimpleNamespace(gtts=lambda instrument, product: [view]),
        gtts=SimpleNamespace(gtt=lambda gtt_id: known),
    )


def stop_view(state, pending=None):
    return {
        "gtt_id": 1,
        "purpose": "stop",
        "state": state,
        "trigger_price": 497.0,
        "quantity_units": 10.0,
        "pending": pending,
    }


def test_a_stop_gtt_being_modified_names_the_trigger_it_is_modified_to():
    pending = {"trigger_price": 500.0, "quantity_units": 10.0}

    words = gtt_words(gtt_trader(stop_view("ModifyPending", pending)))

    assert words == (
        "stop GTT 261116000001 confirmed at the broker, trigger 497.00 for 10 units, being "
        "modified to trigger 500.00 for 10 units"
    )


def test_a_stop_gtt_resting_names_its_trigger_only():
    words = gtt_words(gtt_trader(stop_view("Active")))

    assert words == "stop GTT 261116000001 confirmed at the broker, trigger 497.00 for 10 units"


# (e) ------------------------------------------------------------------------------------------


class StockFuture(bt.Strategy):
    """A strategy on SBIN's futures on NFO, which the reference tables give no expiry rule."""

    underlying, exchange, interval, product = "SBIN", "NFO", "D", "NRML"

    def on_bar(self, bar):
        pass


def test_the_example_for_a_stock_future_names_its_physical_settlement_and_why():
    days = pd.date_range("2026-01-01", periods=60, freq="B", tz="Asia/Kolkata")
    bars = pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000}, index=days
    )

    with pytest.raises(ReferenceDataError) as refused:
        bt.backtest(StockFuture, data=bars, capital=100000, start="2026-02-02", end="2026-03-20")

    words = str(refused.value)
    assert (
        'Instrument(symbol="SBIN18NOV26FUT", exchange="NFO", kind="future", underlying="SBIN", '
        "expiry=datetime.date(2026, 11, 18), lot_size=..., tick_size=..., "
        'settlement="physical")'
    ) in words
    assert words.endswith(
        ': a stock future settles by delivery, and settlement="physical" has the engine exit '
        "it at the square-off of its last trading day instead of settling it in cash at expiry"
    )


# (i) ------------------------------------------------------------------------------------------


def test_after_the_mis_square_off_a_session_without_mis_gets_no_mis_words():
    # 2026-11-16 15:12 IST: past NSE's MIS square-off at 15:00, before its 15:30 close.
    at_ns = ist_ns(2026, 11, 16, 15, 12)

    (check,) = trading_day_checks(["NSE"], at_ns, default_calendar(), mis=False)

    assert check.message == "NSE trades today, 2026-11-16, from 09:15 to 15:30 IST"


def test_after_the_mis_square_off_a_session_with_mis_is_warned():
    at_ns = ist_ns(2026, 11, 16, 15, 12)

    (check,) = trading_day_checks(["NSE"], at_ns, default_calendar(), mis=True)

    assert "the MIS square-off time, 15:00 IST, has passed" in check.message


# (f) ------------------------------------------------------------------------------------------


class StockOptionHold(bt.Strategy):
    """A strategy on an SBIN option on NFO, which bt.live does not trade yet."""

    symbol, exchange, interval, product = "SBIN24NOV26600CE", "NFO", "1m", "NRML"

    def on_bar(self, bar):
        pass


def test_a_stock_derivative_in_bt_live_is_refused_with_advice_a_run_can_follow():
    with pytest.raises(ReferenceDataError) as refused:
        specify(StockOptionHold)

    words = str(refused.value)
    assert "stock futures and stock options cannot be traded in bt.live yet" in words
    assert 'settlement="physical"' in words
    assert "load_tables" not in words
    assert "write_symbol_master_snapshot" not in words


@pytest.mark.parametrize(
    ("symbol", "exchange", "stock"),
    [
        ("SBIN24NOV26600CE", "NFO", True),
        ("SBIN24NOV26FUT", "NFO", True),
        ("NIFTY29DEC26FUT", "NFO", False),
        ("GOLDM05DEC26FUT", "MCX", False),
        ("CRUDEOIL18DEC26FUT", "MCX", False),
        ("USDINR29DEC26FUT", "CDS", False),
    ],
)
def test_only_a_stock_future_or_option_on_nfo_or_bfo_is_named_a_stock_derivative(
    symbol, exchange, stock
):
    # A commodity or currency contract is never a stock derivative, so a reference error
    # of one keeps its own words (review round 1 of unit N5c, finding 5).
    plan = SimpleNamespace(symbols=[symbol], exchange=exchange, start=dt.date(2026, 11, 27))

    assert (stock_derivative(plan) == symbol) is stock
