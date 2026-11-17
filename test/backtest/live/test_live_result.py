"""What the public `bt.live` returns, on the fake OpenAlgo (`openalgo/backtest/session_result.py`,
`_live/_result_build.py`, `_live/_result_frames.py`, `run.live`; 16-session-design 1.4,
unit 13): a `SessionResult` with every field and table, each strategy's `bt.Result` built
by the backtest's own result builder, and the result's warnings and open positions.

The market and the script are `session_samples`': the session starts at 14:40:05 IST on
Monday 2026-11-16, SBIN trades at 500.00 every 10 s, NSE's MIS entries close at 14:50 and
its square-off is at 15:00. `BuyAndHold` buys 10 SBIN at 14:42:01 and sells them three bars
later; `Hold` buys and never sells; `StopHold` buys with a stop 3 points below. Expected
values come from the fake's books and the scenario.
"""

from __future__ import annotations

import datetime as dt
import logging

import pandas as pd
import pytest
from fake_openalgo.fixtures import *  # noqa: F403
from live_samples import live, seam
from session_samples import BuyAndHold, Hold, alerts, ist, rig
from stops_samples import StopHold

import openalgo.backtest as bt
from openalgo.backtest._live._alerts_summary import SessionSummary
from openalgo.backtest._live._types import LOGGER_NAME
from openalgo.backtest.errors import BacktestError

DAY = dt.date(2026, 11, 16)
IST = "Asia/Kolkata"
#: The columns of each table of the result, in order.
COLUMNS = {
    "orders": [
        "strategy",
        "kernel_order",
        "intent_id",
        "order_id",
        "purpose",
        "symbol",
        "exchange",
        "action",
        "price_type",
        "quantity",
        "price",
        "trigger_price",
        "created",
        "created_ns",
        "reached",
        "status",
        "filled_units",
        "average_price",
        "reject_code",
    ],
    "fills": [
        "time",
        "at_ns",
        "strategy",
        "symbol",
        "exchange",
        "order_id",
        "action",
        "units",
        "price",
        "purpose",
        "source",
        "estimated",
        "derived_from_average",
    ],
    "drift": [
        "symbol",
        "exchange",
        "time",
        "ts_open_ns",
        "ts_close_ns",
        "candle_ts_open_ns",
        "candle_ts_close_ns",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "ticks",
    ],
    "unprotected": [
        "strategy",
        "symbol",
        "exchange",
        "from",
        "from_ns",
        "to",
        "to_ns",
        "units",
        "reason",
        "seconds",
    ],
    "locks": ["strategy", "reason", "set", "set_ns", "lifted", "lifted_ns"],
    "reconciliations": [
        "time",
        "at_ns",
        "strategy",
        "kind",
        "engine_units",
        "broker_units",
        "action",
    ],
    "skipped_decisions": ["strategy", "decided_at", "decided_at_ns", "reason"],
    "open_positions": [
        "strategy",
        "symbol",
        "exchange",
        "units",
        "stop_order_id",
        "stop_confirmed",
        "note",
    ],
}


@pytest.fixture
def world(make_fake_openalgo, monkeypatch, tmp_path, caplog):
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    made = rig(make_fake_openalgo, monkeypatch, tmp_path)
    seam(made, monkeypatch)
    return made


@pytest.fixture
def day(world):
    """`BuyAndHold`'s whole day through `bt.live`: the result."""
    return live(world, BuyAndHold)


# ----- the fields ------------------------------------------------------------------------


def test_bt_live_returns_a_session_result(day):
    assert isinstance(day, bt.SessionResult)


def test_the_result_names_the_session(day):
    assert (day.mode, day.broker, day.account, day.day) == ("live", "zerodha", "zerodha", DAY)
    assert (day.stopped_by, day.restarts, day.strategies) == ("session_end", 0, ("BuyAndHold",))
    assert day.mode_breach is None


def test_the_start_and_the_end_are_ist_datetimes(day):
    assert day.started == pd.Timestamp(2026, 11, 16, 14, 40, 5, tz=IST)
    assert pd.Timestamp(2026, 11, 16, 15, 0, tz=IST) <= day.ended
    assert str(day.ended.tz) == "IST"


def test_the_summary_holds_the_days_figures(day):
    assert isinstance(day.summary, SessionSummary)
    assert (day.summary.closed_trades, day.summary.flat, day.flat) == (1, True, True)


def test_the_paths_name_the_journal_the_log_and_the_event_stream(day):
    assert set(day.paths) == {"journal", "log", "events"}
    assert all(dt.date(2026, 11, 16).isoformat() in day.paths[k] for k in ("log", "events"))


@pytest.mark.parametrize("table", list(COLUMNS))
def test_each_table_is_a_dataframe_with_its_columns(day, table):
    frame = getattr(day, table)
    assert isinstance(frame, pd.DataFrame)
    assert list(frame.columns) == COLUMNS[table]


def test_the_result_is_frozen(day):
    with pytest.raises(AttributeError):
        day.mode = "sandbox"


# ----- each strategy's result -------------------------------------------------------------


def test_each_strategy_has_a_backtest_result(day):
    assert list(day.results) == ["BuyAndHold"]
    assert isinstance(day.results["BuyAndHold"], bt.Result)


def test_a_strategys_result_holds_the_fills_the_broker_made(world, day):
    fills = day.results["BuyAndHold"].fills
    fake = [(f.action, f.quantity, f.price) for f in world.fake.fills()]
    found = zip(fills["side"], fills["quantity_units"], fills["price_rupees"], strict=True)
    assert list(found) == [(action, float(units), price) for action, units, price in fake]


def test_a_strategys_result_has_its_trades_and_metrics(day):
    result = day.results["BuyAndHold"]
    assert len(result.trades) == 1
    assert result.metrics["trades"] == 1


def test_a_strategys_result_draws_its_tearsheet(day):
    html = day.results["BuyAndHold"].tearsheet()
    assert html.lstrip().lower().startswith("<!doctype html>") and "BuyAndHold" in html


def test_a_strategys_result_states_its_live_assumptions(day):
    assumptions = day.results["BuyAndHold"].assumptions
    assert "built from the feed's ticks on 1-minute slices" in assumptions["base_rule"]
    assert "NSE:SBIN lot 1, tick 0.05" in assumptions["instruments"]


def test_a_strategys_result_cannot_rerun(day):
    with pytest.raises(BacktestError, match="rerun"):
        day.results["BuyAndHold"].rerun()


# ----- the tables ------------------------------------------------------------------------


def test_the_orders_are_those_the_broker_holds(world, day):
    fake = {o.order_id: o for o in world.fake.orders()}
    orders = day.orders
    assert set(orders["order_id"]) == set(fake)
    assert list(orders["purpose"]) == ["entry", "exit"]
    assert list(orders["status"]) == ["Filled", "Filled"]
    assert list(orders["intent_id"]) == [
        "20261116-live-1-BuyAndHold-o1",
        "20261116-live-1-BuyAndHold-o2",
    ]
    assert list(orders["reached"]) == ["ended", "ended"]


def test_the_fills_are_those_the_broker_made_heard_on_the_stream(world, day):
    fills = day.fills
    fake = [(f.order_id, f.action, f.quantity, f.price) for f in world.fake.fills()]
    found = zip(fills["order_id"], fills["action"], fills["units"], fills["price"], strict=True)
    assert list(found) == fake
    assert set(fills["source"]) == {"stream"}
    assert not fills["estimated"].any() and not fills["derived_from_average"].any()


def test_the_start_locks_entries_until_trading_begins(day):
    starting = day.locks[day.locks["reason"] == "starting"]
    [(set_, lifted)] = zip(starting["set"], starting["lifted"], strict=True)
    assert set_ < lifted < pd.Timestamp(2026, 11, 16, 14, 41, tz=IST)


def test_a_lock_that_stands_at_the_end_has_no_lifting(world):
    world.stop_at(ist(14, 44))
    result = live(world, BuyAndHold)
    stop = result.locks[result.locks["reason"] == "operator_stop"]
    assert stop["lifted"].isna().all() and stop["lifted_ns"].isna().all()


def test_a_position_left_open_is_in_the_open_positions(world):
    world.stop_at(ist(14, 44))
    result = live(world, Hold, on_stop="keep")
    [row] = result.open_positions.to_dict("records")
    assert (row["strategy"], row["symbol"], row["units"]) == ("Hold", "SBIN", 10.0)
    assert (row["stop_order_id"], row["stop_confirmed"]) == ("", False)
    assert "close it by hand" in row["note"]


def test_a_stops_unconfirmed_time_is_in_the_unprotected_windows(world):
    world.stop_at(ist(14, 44))
    result = live(world, StopHold, on_stop="keep")
    windows = result.unprotected
    assert set(windows["strategy"]) == {"StopHold"}
    assert (windows["seconds"] > 0).all() and windows["to"].notna().all()


def test_a_difference_with_the_brokers_books_is_in_the_reconciliations(world):
    other = world.fake.client()
    world.script.at(
        ist(14, 43),
        lambda: other.placeorder(
            strategy="manual",
            symbol="SBIN",
            action="BUY",
            exchange="NSE",
            price_type="MARKET",
            product="MIS",
            quantity=10,
        ),
    )
    world.stop_at(ist(14, 46))
    result = live(world, Hold, on_stop="keep")
    rows = result.reconciliations
    assert list(rows["kind"][:2]) == ["more", "more"]
    assert list(rows["broker_units"][:2]) == [20, 20]


def test_the_candles_checked_against_the_tick_bars_are_in_the_drift(day):
    assert len(day.drift) > 0
    assert set(day.drift["symbol"]) == {"SBIN"}


# ----- the warnings and the summary's text ---------------------------------------------------


def test_the_first_warning_names_the_mode_and_the_broker(day):
    assert day.warnings[0] == "LIVE mode: OpenAlgo sent these orders to zerodha, the broker"


def test_a_sandbox_sessions_first_warning_says_nothing_reached_the_broker(
    make_fake_openalgo, monkeypatch, tmp_path
):
    world = rig(make_fake_openalgo, monkeypatch, tmp_path, analyzer=True)
    seam(world, monkeypatch)
    world.stop_at(ist(14, 41))
    result = live(world, BuyAndHold)
    assert result.mode == "sandbox"
    assert result.warnings[0].startswith("SANDBOX mode: OpenAlgo simulated these fills")


def test_the_warnings_name_the_time_without_a_confirmed_stop(world):
    world.stop_at(ist(14, 44))
    result = live(world, StopHold, on_stop="keep")
    assert any("no confirmed stop at the broker for" in w for w in result.warnings)


def test_the_warnings_name_the_positions_left_open(world):
    world.stop_at(ist(14, 44))
    result = live(world, Hold, on_stop="keep")
    assert any("still open when the session ended (Hold 10 SBIN)" in w for w in result.warnings)


def test_the_summary_text_is_the_daily_summary_the_alert_sent(world, caplog):
    result = live(world, BuyAndHold)
    [sent] = alerts(caplog, "daily_summary")
    assert sent == f"LIVE (BuyAndHold): {result.summary_text()}"


def test_the_repr_names_the_session(day):
    assert repr(day).startswith("SessionResult(LIVE 2026-11-16, broker zerodha, stopped by")
