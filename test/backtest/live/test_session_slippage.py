"""The day summary's slippage on the fake OpenAlgo (`_live/_session_slippage.py`,
16-session-design 1.4 and decided item 11 of 10.0): each MARKET fill against the price a
backtest fills the same order at, the open of the base bar opening at its decision with
the run's modelled slippage, signed against the trader, times the units. An order the
engine makes on an event while it waits inside a minute (a target's exit, a risk exit,
Ctrl+C) is decided when the session places it, so its bar is the next minute's.

The market and the script are `session_samples`': the session starts at 14:40:05 IST on
Monday 2026-11-16 and SBIN (tick 0.05) trades every 10 s, five seconds past each tenth
second, at 500.00 until 14:44 and at 505.00 from then on. The fake fills a MARKET order at
the last trade's price. `Hold` buys 10 at its first decision, the close of 14:42, and never
sells: the session's own exit sells them. The run's slippage is the default for large and
mid cap equity, 2 bps with a floor of one tick, rounded to the tick away from the trader.

By hand:
- The buy, decided at 14:42:00, filled at 500.00. The bar opening at 14:42 opens at its
  first trade, 500.00 at 14:42:05; 2 bps of it is 0.10, so the backtest buys at 500.10.
  The live buy did 0.10 better a unit: 10 x (500.00 - 500.10) = -1.00.
- The square-off's sale is decided at 15:00 and fills at 505.00. That bar opens at its
  first trade, 505.00 when it comes before the session ends; 2 bps of it is 0.101, so the
  backtest sells at 504.899, rounded down to the tick, 504.85. The live sale did 0.15
  better a unit: -10 x (505.00 - 504.85) = -1.50.

So the day's slippage is -2.50: Rs 2.50 better than the backtest.

- The sale on Ctrl+C at 14:46:30 is placed then, inside the 14:46 minute, whose bar opened
  before it. The first bar opening after it is 14:47's, whose first trade (14:47:05) comes
  after the session has ended flat: the sale is not compared.
"""

from __future__ import annotations

import pytest
from fake_openalgo.fixtures import *  # noqa: F403
from live_samples import live, seam
from restart_samples import crash_at, crashing, restart, run_to_crash
from session_samples import NS, Hold, StopFlag, ist, rig
from stops_samples import at_price
from test_live_result import BuyEarlySellLate
from test_session_figures import DAY_ENDS, TwoRoundTrips, world_of
from watch_samples import HoldWithTarget

from openalgo.backtest._live._alerts_summary import summary_text

#: The day's slippage of the round trip of the module docstring, rupees.
ROUND_TRIP = -2.50


@pytest.fixture
def world(make_fake_openalgo, monkeypatch, tmp_path):
    made = rig(make_fake_openalgo, monkeypatch, tmp_path)
    seam(made, monkeypatch)
    at_price(made, ist(14, 44), 505.0)
    return made


def slippage_line(summary) -> str:
    """The summary text's Slippage line, after its label."""
    [line] = [line for line in summary_text(summary).splitlines() if line.startswith("Slippage")]
    return line.split(":", 1)[1].strip()


def test_a_sale_on_ctrl_c_is_not_set_against_the_minute_it_came_in(world):
    world.stop_at(ist(14, 46, 30))
    summary = live(world, Hold).summary
    assert (summary.slippage, summary.slippage_not_compared) == (-1.00, 1)
    assert slippage_line(summary) == (
        "Rs 1.00 better than the backtest (1 MARKET fill not compared: no bar of its decision)"
    )


def moved_at_14_44_30(make_fake_openalgo, monkeypatch, tmp_path, price: float):
    """SBIN at 500.00, then at `price` from 14:44:30: the 14:44 bar opens at 500.00, the
    14:45 bar at `price`. Stopped at 14:47:30."""
    world = rig(make_fake_openalgo, monkeypatch, tmp_path)
    seam(world, monkeypatch)
    at_price(world, ist(14, 44, 30), price)
    world.stop_at(ist(14, 47, 30))
    return world


def test_a_targets_exit_on_a_tick_is_set_against_the_bar_opening_after_it(
    make_fake_openalgo, monkeypatch, tmp_path
):
    # `HoldWithTarget` buys 10 at 500.00 (-1.00, as above) with a target 5 points above the
    # fill, 505.00. The 14:44:35 trade at 505.00 reaches it, and the engine's MARKET sale,
    # placed then, fills at 505.00. The bar opening after it is 14:45's, at 505.00: the
    # backtest sells at 504.85, so the live sale did 0.15 better a unit (-1.50). Against
    # the 14:44 bar, which opened at 500.00 before the tick, it would read -51.00.
    world = moved_at_14_44_30(make_fake_openalgo, monkeypatch, tmp_path, 505.0)
    summary = live(world, HoldWithTarget).summary
    assert (summary.slippage, summary.slippage_not_compared) == (-1.00 - 1.50, 0)


def test_a_daily_loss_exit_on_a_tick_is_set_against_the_bar_opening_after_it(
    make_fake_openalgo, monkeypatch, tmp_path
):
    # `Hold` buys 10 at 500.00 (-1.00) under a daily loss limit of Rs 40. The 14:44:35
    # trade at 495.00 marks a loss of Rs 50, and the engine's MARKET sale, placed then,
    # fills at 495.00. The 14:45 bar opens at 495.00: 2 bps of it is 0.099, so the backtest
    # sells at 494.901, rounded down to the tick, 494.90, and the live sale did 0.10 better
    # a unit (-1.00). Against the 14:44 bar's 500.00 it would read 49.00 worse.
    world = moved_at_14_44_30(make_fake_openalgo, monkeypatch, tmp_path, 495.0)
    summary = live(world, Hold, max_daily_loss=40).summary
    assert (summary.slippage, summary.slippage_not_compared) == (-1.00 - 1.00, 0)


def test_a_round_trip_ended_by_the_square_off_sets_both_fills_against_the_backtest(world):
    # A trade half a second into 15:00, before the square-off's sale is sent at 15:00:01.
    world.script.at(ist(15, 0) + NS // 2, world.script.trade)
    summary = live(world, Hold).summary
    assert (summary.slippage, summary.slippage_not_compared) == (ROUND_TRIP, 0)


def test_a_fill_whose_bar_had_no_trade_before_the_session_ended_is_not_compared(world):
    # The minute of the square-off's decision has its first trade at 15:00:05, after the
    # session has ended flat: only the buy is compared.
    summary = live(world, Hold).summary
    assert (summary.slippage, summary.slippage_not_compared) == (-1.00, 1)
    assert slippage_line(summary) == (
        "Rs 1.00 better than the backtest (1 MARKET fill not compared: no bar of its decision)"
    )


def test_the_days_slippage_carries_across_a_restart(make_fake_openalgo, monkeypatch, tmp_path):
    # `BuyEarlySellLate` buys at the close of 14:42 (-1.00, as above) and sells at the close
    # of 14:49 with SBIN at 510.00 from 14:48:30: that bar's first trade is 510.00, 2 bps of
    # it 0.102, so the backtest sells at 509.898, rounded down, 509.85, and the live sale at
    # 510.00 did 0.15 better a unit (-1.50). Ctrl+C at 14:46:30 keeps the position; the
    # restarted start sells it. The day adds the first start's saved slippage to its own.
    world = rig(make_fake_openalgo, monkeypatch, tmp_path, trades_until=(14, 55))
    seam(world, monkeypatch)
    at_price(world, ist(14, 48, 30), 510.0)
    world.stop_at(ist(14, 46, 30))
    first = live(world, BuyEarlySellLate, on_stop="keep").summary
    world.stop = StopFlag()
    world.stop_at(ist(14, 51))
    day = live(world, BuyEarlySellLate).summary
    assert first.slippage == -1.00
    assert (day.slippage, day.slippage_not_compared) == (-2.50, 0)


def restarted_in_the_second_entrys_minute(make_fake_openalgo, monkeypatch, tmp_path):
    """`TwoRoundTrips` (test_session_figures) crashed at 14:48:30, after its second entry,
    decided at 14:48:00, filled at 505.00. The restart begins in that minute, which only
    seeds its bar builder: its kernel runs 14:48 without a bar and marks it missing until
    the broker's candle of it is read."""
    world = world_of(make_fake_openalgo, monkeypatch, tmp_path)
    crashing(monkeypatch)
    crash_at(world, ist(14, 48, 30))
    run_to_crash(world, TwoRoundTrips)
    return world


# By hand, with SBIN at 500.00, then 505.00 from 14:44:30 and 510.00 from 14:48:30, and each
# bar opening at its minute's first trade, five seconds in: the first start's round trip
# -2.50 (the buy at 500.00 against 500.10, the sale at 505.00 against 504.85); the restored
# entry at 505.00 against the 14:48 bar's 505.15, -1.50, where the 14:49 bar's open would
# make it 505.00 against 510.15, -51.50. The restart decides first at 14:50, the close of
# the first whole minute it watches, so the position is sold on Ctrl+C at 14:51, placed
# then, when its kernel still runs the 14:50 minute; the bar opening after it has no trade
# before the session ends, and the sale is not compared.


def test_an_entry_restored_by_a_restart_is_set_against_its_minutes_candle(
    make_fake_openalgo, monkeypatch, tmp_path
):
    world = restarted_in_the_second_entrys_minute(make_fake_openalgo, monkeypatch, tmp_path)
    summary = restart(world, TwoRoundTrips, stop_ns=DAY_ENDS).summary
    assert (summary.slippage, summary.slippage_not_compared) == (-2.50 - 1.50, 1)


def test_an_entry_whose_minute_is_still_marked_missing_is_not_compared(
    make_fake_openalgo, monkeypatch, tmp_path
):
    # No candle can be read once the restart has caught up, so 14:48 stays marked missing:
    # its bar is not known, and the restored entry is not compared.
    world = restarted_in_the_second_entrys_minute(make_fake_openalgo, monkeypatch, tmp_path)

    def candles_fail():
        for _ in range(60):
            world.fake.faults.gateway_error("history", status=503)

    world.script.at(ist(14, 48, 50), candles_fail)
    summary = restart(world, TwoRoundTrips, stop_ns=DAY_ENDS).summary
    assert (summary.slippage, summary.slippage_not_compared) == (-2.50, 2)


def dead_feed_in_the_entrys_minute(
    make_fake_openalgo, monkeypatch, tmp_path, *, stop_ns: int = ist(14, 51)
):
    """SBIN at 500.00, then 502.00 from 14:42:30, and the feed dead from 14:42:02 to
    14:42:50. A buy decided at 14:42:00 (`BuyEarlySellLate`'s, `Hold`'s) fills at 500.00;
    the minute makes no tick bar, so the kernel runs it without one and marks it missing
    until the broker's candle of it is read. Stopped at `stop_ns`."""
    world = rig(make_fake_openalgo, monkeypatch, tmp_path, trades_until=(14, 55))
    seam(world, monkeypatch)
    at_price(world, ist(14, 42, 30), 502.0)
    world.script.at(ist(14, 42, 2), world.fake.faults.feed_dead)
    world.script.at(ist(14, 42, 50), world.fake.faults.feed_alive)
    world.stop_at(stop_ns)
    return world


def candles_fail_from_14_42_03(world) -> None:
    """Every read of the broker's candles fails from 14:42:03 for the rest of the day."""

    def candles_fail():
        for _ in range(80):
            world.fake.faults.gateway_error("history", status=503)

    world.script.at(ist(14, 42, 3), candles_fail)


# By hand: the buy at 500.00 against the 14:42 candle, which opens at its first trade,
# 500.00 at 14:42:05: the backtest buys at 500.10, -1.00, where the 14:43 bar's 502.00 would
# make it 500.00 against 502.15, -21.50. The sale, decided at the close of the 14:48 bar,
# 14:49:00, fills at 502.00; the 14:49 bar opens at 502.00, 2 bps of it 0.1004, so the
# backtest sells at 501.8996, rounded down to the tick, 501.85: -1.50.


def test_an_entry_in_a_minute_the_feed_missed_is_set_against_its_candle(
    make_fake_openalgo, monkeypatch, tmp_path
):
    world = dead_feed_in_the_entrys_minute(make_fake_openalgo, monkeypatch, tmp_path)
    summary = live(world, BuyEarlySellLate).summary
    assert (summary.slippage, summary.slippage_not_compared) == (-1.00 - 1.50, 0)


def test_an_entry_in_a_minute_the_feed_missed_whose_candle_never_comes_is_not_compared(
    make_fake_openalgo, monkeypatch, tmp_path
):
    world = dead_feed_in_the_entrys_minute(make_fake_openalgo, monkeypatch, tmp_path)
    candles_fail_from_14_42_03(world)
    summary = live(world, BuyEarlySellLate).summary
    assert (summary.slippage, summary.slippage_not_compared) == (-1.50, 1)


def test_a_day_with_no_market_fill_compared_reads_not_compared(
    make_fake_openalgo, monkeypatch, tmp_path
):
    # `Hold`'s buy in the minute the feed missed, whose candle never comes, and its sale on
    # Ctrl+C at 14:46:30, whose next minute has no trade before the session ends: neither
    # is compared, so the day has no figure to show, only the count.
    stop_ns = ist(14, 46, 30)
    world = dead_feed_in_the_entrys_minute(
        make_fake_openalgo, monkeypatch, tmp_path, stop_ns=stop_ns
    )
    candles_fail_from_14_42_03(world)
    summary = live(world, Hold).summary
    assert (summary.slippage, summary.slippage_not_compared) == (None, 2)
    assert slippage_line(summary) == "not compared (2 MARKET fills: no bar of their decisions)"
