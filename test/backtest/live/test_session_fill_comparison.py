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
from types import SimpleNamespace

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
from openalgo.backtest._live._session_slippage import (
    NO_BAR,
    NOT_REACHED,
    Priced,
    StopTriggers,
    Unpriced,
    stop_price,
)
from openalgo.backtest._live._types import LOGGER_NAME, OrderIntent, Purpose
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


def test_a_stop_heard_filled_after_its_minute_is_set_in_the_bar_that_reached_it(stops):
    # SBIN trades at 496.00 from 14:44:57 (and once more at 14:44:58); the stream holds the
    # order updates from 14:44:56 to 14:45:08. The stop fills at 496.00 in the 14:44
    # minute, and the session hears it in the 14:45 minute. A backtest fills it in the bar
    # whose low reached the trigger, 14:44's, which opened at 500.00 above it: at the
    # trigger less 2 bps, 496.90, as when the fill is heard at once (Rs 9.00 worse).
    # (Set in the 14:45 bar, opened at 496.00, it read 495.90, Rs 1.00 better.)
    at_price(stops, ist(14, 44, 57), 496.0)
    stops.script.at(ist(14, 44, 58), stops.script.trade)
    stops.script.at(ist(14, 44, 56), stops.fake.faults.hold_order_updates)
    stops.script.at(ist(14, 45, 8), stops.fake.faults.release_order_updates)
    stops.stop_at(ist(14, 47))
    result = live(stops, StopHold)
    sale = row(result, "stop")
    assert sale["live_time"] >= minute(14, 45)
    assert sale["backtest_time"] == minute(14, 44)
    assert sale["backtest_price"] == pytest.approx(496.90)
    assert (sale["difference"], sale["rupees"]) == (pytest.approx(0.90), 9.00)
    assert result.summary.stop_slippage == 9.00


class Trails(bt.Strategy):
    """Buys 10 SBIN at its first decision with a stop 3 points below that trails 3 points
    behind the best price."""

    symbol, exchange, interval = "SBIN", "NSE", "1m"

    def init(self):
        self.calls = 0

    def on_bar(self, bar):
        self.calls += 1
        if self.calls == 1:
            self.buy(10, stop=bt.points(3), trail=bt.trail_points(3))


def test_a_trailed_stop_is_looked_for_from_the_bar_its_trigger_moved_in(stops):
    # SBIN trades at 510.00 from 14:43: the stop trails to 507.00. It trades at 506.00 from
    # 14:45:30: the 14:45:35 trade fills the stop at 506.00. The bars before the move went
    # below 507.00 (14:42's opened at 500.00), but the stop was at 497.00 then; the first
    # bar from the move whose low reaches 507.00 is 14:45's, which opened at 510.00 above
    # it: the backtest sells at the trigger less 2 bps, 506.8986, rounded down to 506.85,
    # and the live sale did 0.85 worse a unit, Rs 8.50.
    at_price(stops, ist(14, 43), 510.0)
    at_price(stops, ist(14, 45, 30), 506.0)
    stops.stop_at(ist(14, 47))
    result = live(stops, Trails)
    stop = next(o for o in stops.orders() if o.price_type == "SL-M")
    assert stop.trigger_price == 507.0
    sale = row(result, "stop")
    assert (sale["live_price"], sale["backtest_time"]) == (506.0, minute(14, 45))
    assert sale["backtest_price"] == pytest.approx(506.85)
    assert (sale["difference"], sale["rupees"]) == (pytest.approx(0.85), 8.50)


#: A stop placed at 497.00 at instant 100 (`held_since`'s table).
PLACED = OrderIntent(
    "i", "S", "SBIN", "NSE", "SELL", 10, "SL-M", "MIS", Purpose.STOP, 100, trigger_price=497.0
)


@pytest.mark.parametrize(
    ("modifies", "trigger", "since"),
    [
        ((), 497.0, 100),
        (((507.0, 300),), 507.0, 300),
        (((507.0, 300), (507.0, 400)), 507.0, 300),
        (((507.0, 300), (510.0, 500)), 510.0, 500),
        ((), 507.0, 250),
        (((507.0, 300),), 510.0, 300),
    ],
    ids=[
        "never moved: from its placing",
        "moved: from the modify",
        "its units changed after: from the move",
        "moved twice: from the last move",
        "moved before a restart at 250: from the start",
        "moved where this start did not see: from the last move it knows",
    ],
)
def test_a_stop_holds_its_trigger_from_the_move_that_set_it(modifies, trigger, since):
    triggers = StopTriggers()
    for moved_to, at_ns in modifies:
        triggers.modified("S", 2, moved_to, at_ns)
    assert triggers.held_since("S", 2, PLACED, trigger, start_ns=250) == since


class StandInSeries:
    """Four one-minute slices from 0 (instants in seconds, for reading), the bars given as
    {open instant: (open, high, low)}, and the slices marked missing."""

    slices = ((0, 60), (60, 120), (120, 180), (180, 240))

    def __init__(self, bars: dict, missing: tuple = ()) -> None:
        self._bars = {
            at: SimpleNamespace(open=o, high=h, low=lo) for at, (o, h, lo) in bars.items()
        }
        self._missing = missing

    def bar(self, opened):
        return self._bars.get(opened)

    def missing_slices(self, start, end):
        return tuple(at for at in self._missing if start <= at < end)


def stop_at(
    bars, *, side="SELL", trigger=497.0, since=0, heard=130, released=4, so_far=None, missing=()
):
    """`stop_price` on stand-ins whose fill price is the reference itself (no slippage),
    so the result reads the rule; the slice in progress has opened at `so_far`."""
    builder = SimpleNamespace(open_so_far=lambda symbol, opened: so_far)
    session = SimpleNamespace(builders={"NSE": builder})
    trader = SimpleNamespace(
        part=SimpleNamespace(series=StandInSeries(bars, missing), released=released),
        engine=SimpleNamespace(market_fill_price=lambda _i, _side, _at, reference: reference),
        exchange="NSE",
        symbol="SBIN",
    )
    return stop_price(session, trader, side=side, trigger=trigger, since_ns=since, heard_ns=heard)


ABOVE = (500.0, 500.0, 499.0)


@pytest.mark.parametrize(
    ("given", "priced"),
    [
        ({"bars": {0: ABOVE, 60: (500.0, 500.0, 496.0)}}, Priced(60, 497.0)),
        ({"bars": {0: ABOVE, 60: (495.0, 495.0, 494.0)}}, Priced(60, 495.0)),
        ({"bars": {0: ABOVE, 60: ABOVE, 120: ABOVE}}, Unpriced(NOT_REACHED)),
        ({"bars": {0: ABOVE, 120: (500.0, 500.0, 490.0)}}, Priced(120, 497.0)),
        ({"bars": {0: ABOVE, 120: (500.0, 500.0, 490.0)}, "missing": (60,)}, Unpriced(NO_BAR)),
        (
            {"bars": {0: (500.0, 500.0, 490.0), 60: (499.0, 499.0, 496.0)}, "since": 70},
            Priced(60, 497.0),
        ),
        ({"bars": {0: ABOVE, 60: ABOVE}, "released": 2, "so_far": 498.0}, Priced(120, 497.0)),
        (
            {"bars": {0: ABOVE, 60: (501.0, 504.0, 500.0)}, "side": "BUY", "trigger": 503.0},
            Priced(60, 503.0),
        ),
    ],
    ids=[
        "the first bar whose low reaches the trigger, at the trigger",
        "a bar that opened through the trigger, at its open",
        "no bar reached the trigger: not priced",
        "a minute with no trade passed over",
        "a minute marked missing: not priced",
        "a bar before the trigger was held is not looked at",
        "not reached in a released bar: the slice in progress, by its open so far",
        "a buy's stop, by the bar's high",
    ],
)
def test_a_stop_is_priced_in_the_first_bar_that_reaches_its_trigger(given, priced):
    assert stop_at(**given) == priced


def test_the_stop_rows_are_not_in_the_market_slippage(stops):
    at_price(stops, ist(14, 44, 30), 496.0)
    stops.stop_at(ist(14, 46))
    result = live(stops, StopHold)
    assert (result.summary.slippage, result.summary.stop_slippage) == (-1.00, 9.00)
    assert "Stop fills  : Rs 9.00 worse than the backtest" in result.summary_text()


def test_a_day_without_a_stop_fill_prints_no_stop_line(round_trip):
    assert round_trip.summary.stop_slippage is None
    assert "Stop fills" not in round_trip.summary_text()


class SellsOnItsOwnStop(bt.Strategy):
    """Buys 10 SBIN at its first decision and at its second rests its own exit, a sell
    stop order (SL-M) triggered at 497.00, with no leg rules."""

    symbol, exchange, interval = "SBIN", "NSE", "1m"

    def init(self):
        self.calls = 0

    def on_bar(self, bar):
        self.calls += 1
        if self.calls == 1:
            self.buy(10)
        elif self.calls == 2:
            self.sell(10, trigger_price=497.0)


def test_a_strategys_own_stop_order_is_not_a_stop_fill(world):
    # The sale rests from 14:43 at 497.00; SBIN trades at 496.00 from 14:44:30 and the
    # 14:44:35 trade fills it at 496.00. The 14:44 bar opened at 500.00 above the trigger:
    # the backtest sells at 497.00 less 2 bps, 496.9006, rounded down to 496.90. The row
    # is compared, but it is the strategy's own exit, not a leg's stop: no Stop fills
    # line.
    at_price(world, ist(14, 44, 30), 496.0)
    world.stop_at(ist(14, 46))
    result = live(world, SellsOnItsOwnStop)
    sale = row(result, "exit")
    assert (sale["price_type"], sale["live_price"]) == ("SL-M", 496.0)
    assert (sale["backtest_time"], sale["backtest_price"]) == (
        minute(14, 44),
        pytest.approx(496.90),
    )
    assert result.summary.stop_slippage is None
    assert "Stop fills" not in result.summary_text()


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
