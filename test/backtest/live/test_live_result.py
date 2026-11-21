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
import typing
from collections.abc import Mapping

import pandas as pd
import pytest
from fake_openalgo.fixtures import *  # noqa: F403
from live_samples import live, seam
from session_samples import CAPITAL, BuyAndHold, Hold, StopFlag, alerts, ist, rig
from stops_samples import StopHold, at_price
from test_session_close_late import LATE_SUMMARY, EntersThenStops, TwinOrder
from test_session_figures import DAY_ENDS, TwoRoundTrips

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
        "version_id",
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


def test_the_paths_name_the_journal_the_log_the_event_stream_and_the_decisions_file(day):
    assert set(day.paths) == {"journal", "log", "events", "decisions"}
    named = ("log", "events", "decisions")
    assert all(dt.date(2026, 11, 16).isoformat() in day.paths[k] for k in named)


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


# ----- a leg the close names, unconfirmed ---------------------------------------------------


@pytest.fixture
def late(world):
    """`test_session_close_late`'s scenario through `bt.live`: OpenAlgo holds 20 SBIN that
    the engine does not, seen only by the close's late read."""
    EntersThenStops.flag = world.stop
    world.fake.faults.bad_gateway_page("placeorder", reached_broker=True)
    twin = TwinOrder(world.fake.client(), world.fake.client())
    made = world.clients()
    world.clients = lambda: type(made)(**{**made.__dict__, "orders": twin})
    result = live(world, EntersThenStops, on_stop="keep")
    assert world.fake.position("SBIN", "NSE") == 20
    return result


def test_a_leg_the_close_names_leaves_the_result_not_confirmed_flat(late):
    assert late.summary.differing == (LATE_SUMMARY,)
    # The entry the book cannot tell apart is the only open position (unit 14, F60).
    assert late.open_positions["units"].tolist() == [10]
    assert not late.flat


def test_the_warnings_name_each_leg_the_close_names(late):
    assert any(LATE_SUMMARY in warning for warning in late.warnings)


def test_the_repr_counts_the_legs_the_close_names(late):
    assert repr(late).endswith("1 positions open, 1 leg not confirmed at the close)")


# ----- a restarted start's result -----------------------------------------------------------


def test_a_restarted_starts_result_counts_its_own_money_only(world):
    at_price(world, ist(14, 44, 30), 505.0)
    at_price(world, ist(14, 48, 30), 510.0)
    world.stop_at(ist(14, 46, 30))
    first = live(world, TwoRoundTrips)
    world.stop = StopFlag()
    world.stop_at(DAY_ENDS)
    second = live(world, TwoRoundTrips)
    result = second.results["TwoRoundTrips"]
    assert len(result.trades) == 1
    assert result.metrics["start_equity"] == CAPITAL
    # The summary's charges are each fill's, rounded to the paisa; the result's are its
    # contract notes', rounded once per note.
    before_rounding = result.metrics["net_profit"] + note_rounding(result)
    assert before_rounding == pytest.approx(second.summary.net_pnl - first.summary.net_pnl)
    assert result.gross_pnl["realized"] == pytest.approx(result.trades["gross_pnl_rupees"].sum())


def note_rounding(result: bt.Result) -> float:
    """Rupees the contract notes of `result` changed its charges by, rounding their lines
    once per note: its charges less the day summary's (each fill's rounded to the paisa)."""
    return float(result.contract_notes["adjustment"].sum())


class BuyEarlySellLate(bt.Strategy):
    """Buys 10 SBIN on the 14:42 bar and sells them on the 14:49 bar."""

    symbol, exchange, interval = "SBIN", "NSE", "1m"

    def on_bar(self, bar):
        if bar.ts_close == ist(14, 42) and not self.position.qty:
            self.buy(10)
        elif bar.ts_close == ist(14, 49) and self.position.qty:
            self.exit()


def bought_at_500_sold_at_510(world, *, ctrl_c_at: int | None) -> bt.SessionResult:
    """`BuyEarlySellLate` on `world` with SBIN at 505.00 from 14:44:30 and 510.00 from
    14:48:30; with `ctrl_c_at`, stopped there with on_stop="keep" and started again on the
    same journal. The result of the last start, which stops at 14:51."""
    at_price(world, ist(14, 44, 30), 505.0)
    at_price(world, ist(14, 48, 30), 510.0)
    if ctrl_c_at is not None:
        world.stop_at(ctrl_c_at)
        live(world, BuyEarlySellLate, on_stop="keep")
        world.stop = StopFlag()
    world.stop_at(ist(14, 51))
    return live(world, BuyEarlySellLate)


@pytest.fixture
def unbroken(make_fake_openalgo, monkeypatch, tmp_path_factory):
    """The day of `bought_at_500_sold_at_510` in one start, on a fake of its own."""
    folder = tmp_path_factory.mktemp("unbroken")
    made = rig(make_fake_openalgo, monkeypatch, folder, trades_until=(14, 55))
    seam(made, monkeypatch)
    return bought_at_500_sold_at_510(made, ctrl_c_at=None)


def test_a_position_carried_across_a_restart_brings_its_entry_charges_to_the_result(
    unbroken, make_fake_openalgo, monkeypatch, tmp_path
):
    # The unit 13 follow-up item 2: the restarted start closes the trade, so its result
    # counts the whole trade, the charges of its entry before the restart included. Each
    # start settles the contract notes of its own fills (the entry's before the restart
    # are carried as the account paid them), so the results compare before the notes'
    # rounding, which is what the day summary counts.
    world = rig(make_fake_openalgo, monkeypatch, tmp_path, trades_until=(14, 55))
    seam(world, monkeypatch)
    restarted = bought_at_500_sold_at_510(world, ctrl_c_at=ist(14, 46, 30))
    result = restarted.results["BuyEarlySellLate"]
    whole = unbroken.results["BuyEarlySellLate"]
    before_rounding = result.metrics["net_profit"] + note_rounding(result)
    assert before_rounding == pytest.approx(whole.metrics["net_profit"] + note_rounding(whole))
    assert before_rounding == pytest.approx(restarted.summary.net_pnl)
    assert result.gross_pnl["realized"] == pytest.approx(result.trades["gross_pnl_rupees"].sum())


def test_a_day_with_a_position_carried_across_a_restart_sums_as_the_unbroken_day(
    unbroken, make_fake_openalgo, monkeypatch, tmp_path
):
    world = rig(make_fake_openalgo, monkeypatch, tmp_path, trades_until=(14, 55))
    seam(world, monkeypatch)
    restarted = bought_at_500_sold_at_510(world, ctrl_c_at=ist(14, 46, 30))
    figures = ("closed_trades", "gross_pnl", "charges", "net_pnl")
    assert [getattr(restarted.summary, name) for name in figures] == pytest.approx(
        [getattr(unbroken.summary, name) for name in figures]
    )


def both_starts(world) -> tuple[bt.SessionResult, bt.SessionResult]:
    """`bought_at_500_sold_at_510` with Ctrl+C at 14:46:30: the first start's result, which
    keeps the position open, and the restarted one's, which sells it."""
    at_price(world, ist(14, 44, 30), 505.0)
    at_price(world, ist(14, 48, 30), 510.0)
    world.stop_at(ist(14, 46, 30))
    first = live(world, BuyEarlySellLate, on_stop="keep")
    world.stop = StopFlag()
    world.stop_at(ist(14, 51))
    return first, live(world, BuyEarlySellLate)


def test_the_trade_a_restart_closes_carries_the_charges_its_entry_paid_before_it(world):
    # Unit 14 round 1, finding 5: the restarted engine's charge book holds its own start's
    # fills only, so its trades table must bring the entry's charges, which the first
    # start's day figures hold (the position was its only fill).
    first, restarted = both_starts(world)
    result = restarted.results["BuyEarlySellLate"]
    (trade,) = result.trades.to_dict("records")
    (sale,) = result.fills.to_dict("records")
    assert first.summary.charges > 0
    assert trade["costs_rupees"] == pytest.approx(sale["costs_rupees"] + first.summary.charges)
    assert trade["net_pnl_rupees"] == pytest.approx(
        trade["gross_pnl_rupees"] - trade["costs_rupees"]
    )


@pytest.mark.parametrize("restart", [False, True], ids=["unbroken", "restarted"])
def test_a_live_results_net_profit_is_the_sum_of_its_trades_net(world, restart):
    # The kernel follow-up item 1: the stop settles the day's contract notes and records the
    # slice in progress, so the equity holds them (94.62 before, against the trades' 95.05
    # unbroken and 94.89 restarted).
    if restart:
        _, last = both_starts(world)
    else:
        last = bought_at_500_sold_at_510(world, ctrl_c_at=None)
    result = last.results["BuyEarlySellLate"]
    assert result.metrics["net_profit"] == pytest.approx(result.trades["net_pnl_rupees"].sum())


def test_a_first_starts_result_is_left_as_its_engine_made_it(world):
    at_price(world, ist(14, 44, 30), 505.0)
    world.stop_at(ist(14, 46, 30))
    result = live(world, TwoRoundTrips)
    figures = result.results["TwoRoundTrips"]
    before_rounding = figures.metrics["net_profit"] + note_rounding(figures)
    assert before_rounding == pytest.approx(result.summary.net_pnl)


# ----- the live assumptions -----------------------------------------------------------------


def test_a_live_results_assumptions_say_openalgo_priced_the_fills(day):
    assumptions = day.results["BuyAndHold"].assumptions
    assert "OpenAlgo" in assumptions["fill"] and "next_open" not in assumptions["fill"]
    assert assumptions["slippage"].startswith("not modelled")
    assert "OpenAlgo" in assumptions["risk_exit_fill"]


def test_a_live_results_base_rule_models_no_fill_at_a_bar_open(day):
    base_rule = day.results["BuyAndHold"].assumptions["base_rule"]
    assert base_rule.startswith("base 1m (a live session's bars are built from the feed's ticks")
    assert not any(words in base_rule for words in ("fill at", "fills at", "bar opens"))


# ----- the type hints ----------------------------------------------------------------------


def test_the_summarys_type_has_a_public_name(day):
    from openalgo.backtest.session_result import SessionSummary as Public

    assert type(day.summary) is Public


def test_the_results_type_hints_resolve_once_its_summary_type_is_loaded():
    from openalgo.backtest.session_result import SessionSummary as Public

    hints = typing.get_type_hints(bt.SessionResult)
    assert hints["summary"] is Public
    assert hints["results"] == Mapping[str, bt.Result]
