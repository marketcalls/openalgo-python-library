"""Live fills against a backtest of the same bars (`SessionResult.fill_comparison`,
`_live/_fill_comparison.py`, `_live/_session_slippage.py`; 18-multi-strategy 12.3, unit M11).

The market and the scripts are `session_samples`' and `stops_samples`': the session starts
at 14:40:05 IST on Monday 2026-11-16 and SBIN (tick 0.05) trades 10 units every 10 s, five
seconds past each tenth second, at 500.00 unless a test moves it. A minute's bar opens at
its first trade. The fake fills a MARKET order, and a stop once the price reaches its
trigger, at the last trade's price. The run's slippage is the default for large and mid
cap equity, 2 bps with a floor of one tick, rounded to the tick away from the trader.

By hand, each row below: a buy at 500.00 decided at 14:42:00 against the 14:42 bar's open
500.00 plus 2 bps, 500.10: 0.10 better a unit, Rs -1.00 for 10.
"""

from __future__ import annotations

import logging

import pandas as pd
import pytest
from fake_openalgo.fixtures import *  # noqa: F403
from live_samples import live, seam
from session_samples import NS, BuyAndHold, Hold, StopFlag, ist, rig
from stops_samples import StopHold, at_price, stop_world
from test_live_result import BuyEarlySellLate
from test_reconcile_hand_sales import before_any_read, play
from watch_samples import HoldWithTarget

import openalgo.backtest as bt
from openalgo.backtest._live._fill_comparison import HAND_ORDER, LIMIT_ORDER
from openalgo.backtest._live._session_slippage import NO_BAR
from openalgo.backtest._live._types import LOGGER_NAME
from openalgo.backtest._live._version_id import version_of

IST = "Asia/Kolkata"
COLUMNS = [
    "strategy",
    "version_id",
    "symbol",
    "exchange",
    "order_id",
    "purpose",
    "price_type",
    "side",
    "units",
    "live_time",
    "live_ns",
    "live_price",
    "backtest_time",
    "backtest_ns",
    "backtest_price",
    "difference",
    "rupees",
    "compared",
    "reason",
]


@pytest.fixture
def world(make_fake_openalgo, monkeypatch, tmp_path, caplog):
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    made = rig(make_fake_openalgo, monkeypatch, tmp_path)
    seam(made, monkeypatch)
    return made


@pytest.fixture
def stops(make_fake_openalgo, monkeypatch, tmp_path, caplog):
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    made = stop_world(make_fake_openalgo, monkeypatch, tmp_path)
    seam(made, monkeypatch)
    return made


def minute(hour: int, minute_: int) -> pd.Timestamp:
    return pd.Timestamp(2026, 11, 16, hour, minute_, tz=IST)


def row(result, purpose: str) -> pd.Series:
    [found] = [r for _, r in result.fill_comparison.iterrows() if r["purpose"] == purpose]
    return found


# ----- MARKET orders -------------------------------------------------------------------------


@pytest.fixture
def round_trip(world):
    """`Hold`'s day: the buy above, and the square-off's sale decided at 15:00 that fills at
    505.00, with a trade half a second into 15:00. That bar opens at 505.00; 2 bps of it is
    0.101, so the backtest sells at 504.899, rounded down to the tick, 504.85: the live
    sale did 0.15 better a unit, Rs -1.50."""
    at_price(world, ist(14, 44), 505.0)
    world.script.at(ist(15, 0) + NS // 2, world.script.trade)
    return live(world, Hold)


def test_the_frame_has_its_columns(round_trip):
    assert list(round_trip.fill_comparison.columns) == COLUMNS


def test_a_market_entry_is_set_against_the_open_of_the_bar_opening_at_its_decision(round_trip):
    buy = row(round_trip, "entry")
    assert (buy["side"], buy["units"], buy["price_type"]) == ("BUY", 10, "MARKET")
    assert (buy["live_price"], buy["backtest_price"]) == (500.0, pytest.approx(500.10))
    assert buy["backtest_time"] == minute(14, 42)
    assert (buy["difference"], buy["rupees"]) == (pytest.approx(-0.10), -1.00)
    assert (buy["compared"], buy["reason"]) == (True, "")


def test_the_square_off_is_set_against_the_bar_opening_at_its_decision(round_trip):
    sale = row(round_trip, "square_off")
    assert (sale["live_price"], sale["backtest_price"]) == (505.0, pytest.approx(504.85))
    assert sale["backtest_time"] == minute(15, 0)
    assert (sale["difference"], sale["rupees"]) == (pytest.approx(-0.15), -1.50)


def test_the_compared_market_rows_add_up_to_the_summarys_slippage(round_trip):
    frame = round_trip.fill_comparison
    market = frame[(frame["price_type"] == "MARKET") & frame["compared"]]
    assert market["rupees"].sum() == pytest.approx(round_trip.summary.slippage) == -2.50


def test_every_row_carries_its_strategys_version_id(round_trip):
    assert set(round_trip.fill_comparison["version_id"]) == {version_of(Hold, None).version_id}
    assert set(round_trip.fill_comparison["strategy"]) == {"Hold"}


def test_a_strategys_exit_is_timed_at_its_bar(world):
    # BuyAndHold sells itself at its fourth decision, the close of 14:45: the bar opening
    # then, at 500.00, less 2 bps, 499.90; the live sale at 500.00 did 0.10 better.
    world.stop_at(ist(14, 47, 30))
    result = live(world, BuyAndHold)
    sale = row(result, "exit")
    assert sale["backtest_time"] == minute(14, 45)
    assert (sale["backtest_price"], sale["rupees"]) == (pytest.approx(499.90), -1.00)


def test_a_fill_with_no_bar_yet_is_not_compared(world):
    # Ctrl+C at 14:46:30 sells inside the 14:46 minute; the bar opening after it has no
    # trade before the session ends.
    world.stop_at(ist(14, 46, 30))
    result = live(world, Hold)
    sale = row(result, "risk_exit")
    assert (sale["compared"], sale["reason"]) == (False, NO_BAR)
    assert pd.isna(sale["backtest_price"]) and pd.isna(sale["rupees"])
    assert result.summary.slippage_not_compared == 1


# ----- the engine's target exit ----------------------------------------------------------------


def test_the_engines_target_exit_is_set_against_its_level(world):
    # HoldWithTarget's target is 505.00, 5 points above its fill. SBIN trades at 506.00 from
    # 14:44:30: the 14:44:35 trade reaches the target and the engine's sale fills at 506.00.
    # The 14:44 bar opened at 500.00, below the level, so the backtest sells at the level
    # less 2 bps, 504.899, rounded down to 504.85: the live sale did 1.15 better a unit,
    # Rs -11.50. (Set against the next bar's open, 506.00 less 2 bps, 505.85, it read -1.50.)
    at_price(world, ist(14, 44, 30), 506.0)
    world.stop_at(ist(14, 47, 30))
    result = live(world, HoldWithTarget)
    sale = row(result, "target")
    assert (sale["live_price"], sale["backtest_price"]) == (506.0, pytest.approx(504.85))
    assert sale["backtest_time"] == minute(14, 44)
    assert (sale["difference"], sale["rupees"]) == (pytest.approx(-1.15), -11.50)
    assert result.summary.slippage == -1.00 - 11.50


# ----- stops -----------------------------------------------------------------------------------


def test_a_stop_is_set_against_its_trigger(stops):
    # StopHold's SL-M stop is at 497.00. SBIN trades at 496.00 from 14:44:30: the 14:44:35
    # trade triggers it and it fills at 496.00. The 14:44 bar opened at 500.00, above the
    # trigger, so the backtest sells at the trigger less 2 bps, 496.9006, rounded down to
    # 496.90: the live sale did 0.90 worse a unit, Rs 9.00.
    at_price(stops, ist(14, 44, 30), 496.0)
    stops.stop_at(ist(14, 46))
    result = live(stops, StopHold)
    sale = row(result, "stop")
    assert (sale["price_type"], sale["live_price"]) == ("SL-M", 496.0)
    assert sale["backtest_price"] == pytest.approx(496.90)
    assert sale["backtest_time"] == minute(14, 44)
    assert (sale["difference"], sale["rupees"]) == (pytest.approx(0.90), 9.00)
    assert result.summary.stop_slippage == 9.00


def test_a_stop_gapped_through_is_set_against_its_bars_open(stops):
    # SBIN trades at 496.00 from 14:44:00: the 14:44 bar opens at 496.00, through the
    # trigger, and the stop fills at 496.00 on that first trade. The backtest sells at the
    # open less 2 bps, 495.9008, rounded down to 495.90: the live sale did 0.10 better.
    at_price(stops, ist(14, 44), 496.0)
    stops.stop_at(ist(14, 46))
    result = live(stops, StopHold)
    sale = row(result, "stop")
    assert sale["backtest_price"] == pytest.approx(495.90)
    assert (sale["difference"], sale["rupees"]) == (pytest.approx(-0.10), -1.00)


def test_the_stop_rows_are_not_in_the_market_slippage(stops):
    at_price(stops, ist(14, 44, 30), 496.0)
    stops.stop_at(ist(14, 46))
    result = live(stops, StopHold)
    assert (result.summary.slippage, result.summary.stop_slippage) == (-1.00, 9.00)
    assert "Stop fills  : Rs 9.00 worse than the backtest" in result.summary_text()


def test_a_day_without_a_stop_fill_prints_no_stop_line(round_trip):
    assert round_trip.summary.stop_slippage is None
    assert "Stop fills" not in round_trip.summary_text()


# ----- not compared ------------------------------------------------------------------------------


class BuysAtALimit(bt.Strategy):
    """Buys 10 SBIN with a LIMIT at 500.00 at its first decision."""

    symbol, exchange, interval = "SBIN", "NSE", "1m"

    def init(self):
        self.calls = 0

    def on_bar(self, bar):
        self.calls += 1
        if self.calls == 1:
            self.buy(10, price=500.0)


def test_a_limit_orders_fill_is_not_compared(world):
    world.stop_at(ist(14, 44))
    result = live(world, BuysAtALimit)
    buy = row(result, "entry")
    assert (buy["price_type"], buy["compared"], buy["reason"]) == ("LIMIT", False, LIMIT_ORDER)
    assert result.summary.slippage_not_compared == 1


def test_a_sale_by_hand_the_session_adopted_is_not_compared(
    make_fake_openalgo, monkeypatch, tmp_path, caplog
):
    # Another client sells StopHold's 10 at 14:43:00; two Ctrl+C at 14:43:01; the close's
    # final read adopts the sale (test_reconcile_hand_sales).
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    world = stop_world(make_fake_openalgo, monkeypatch, tmp_path)
    world.stop_at(ist(14, 50))
    outcome = play(world, monkeypatch, before_any_read, 10)
    [hand] = [r for r in outcome.fill_comparison if r.purpose == "hand"]
    assert (hand.side, hand.units, hand.compared, hand.reason) == ("SELL", 10, False, HAND_ORDER)


# ----- across a restart --------------------------------------------------------------------------


def test_the_frame_covers_every_start_of_the_day(make_fake_openalgo, monkeypatch, tmp_path):
    # BuyEarlySellLate buys at the close of 14:42 (Rs -1.00); Ctrl+C at 14:46:30 keeps the
    # position; the restarted start sells it at the close of 14:49 with SBIN at 510.00 from
    # 14:48:30: that bar's first trade is 510.00, 2 bps of it 0.102, so the backtest sells
    # at 509.898, rounded down, 509.85, and the live sale did 0.15 better (Rs -1.50).
    world = rig(make_fake_openalgo, monkeypatch, tmp_path, trades_until=(14, 55))
    seam(world, monkeypatch)
    at_price(world, ist(14, 48, 30), 510.0)
    world.stop_at(ist(14, 46, 30))
    live(world, BuyEarlySellLate, on_stop="keep")
    world.stop = StopFlag()
    world.stop_at(ist(14, 51))
    day = live(world, BuyEarlySellLate)
    frame = day.fill_comparison
    assert frame["purpose"].tolist() == ["entry", "exit"]
    assert frame["rupees"].tolist() == [-1.00, -1.50]
    assert frame["rupees"].sum() == pytest.approx(day.summary.slippage)
