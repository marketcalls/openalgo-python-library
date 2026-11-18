"""The bar generators of the robustness checks (design 17, sections 2.5 and 2.6): permuted
and noisy bars are pure functions of the bars and a seed, every bar they make is a
valid OHLC on the tick grid, several instruments are permuted together, and the same
seed gives the same bars on every platform.

Bars here are built by hand in the canonical layout: `day_bars(opens, highs, lows,
closes)` puts one bar a day at 09:15 IST from 2025-01-01, `minute_bars(days, per_day)`
puts `per_day` one-minute bars from 09:15 IST on each of `days` days.
"""

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from openalgo.backtest._robust._generate_noisy import Noise, noisy, noisy_one, ranges
from openalgo.backtest._robust._generate_permuted import permuted, rebuilt, source_positions
from openalgo.backtest.errors import DataError

IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
TICK = 0.05
MINUTE = 60 * 10**9


def _ns(day: int, minute: int = 0) -> int:
    start = dt.datetime(2025, 1, 1, 9, 15, tzinfo=IST) + dt.timedelta(days=day, minutes=minute)
    return int(pd.Timestamp(start).value)


def frame(ts_open, opens, highs, lows, closes, step=MINUTE) -> pd.DataFrame:
    ts_open = np.asarray(ts_open, dtype=np.int64)
    bars = pd.DataFrame(
        {
            "ts_open": ts_open,
            "ts_close": ts_open + step,
            "open": np.asarray(opens, dtype=float),
            "high": np.asarray(highs, dtype=float),
            "low": np.asarray(lows, dtype=float),
            "close": np.asarray(closes, dtype=float),
            "volume": np.arange(1.0, len(ts_open) + 1.0) * 100.0,
            "oi": np.full(len(ts_open), np.nan),
        }
    )
    bars.attrs.update(symbol="SBIN", exchange="NSE", interval="D")
    return bars


def day_bars(opens, highs, lows, closes) -> pd.DataFrame:
    return frame([_ns(day) for day in range(len(opens))], opens, highs, lows, closes)


def walk_bars(days: int, per_day: int, seed: int = 7) -> pd.DataFrame:
    """A random walk on the 0.05 grid: `per_day` one-minute bars on each of `days` days."""
    rng = np.random.default_rng(seed)
    count = days * per_day
    ts = [_ns(day, minute) for day in range(days) for minute in range(per_day)]
    closes = 500.0 + np.cumsum(np.round(rng.normal(0.0, 1.0, count) / TICK) * TICK)
    opens = (
        np.concatenate(([500.0], closes[:-1])) + np.round(rng.normal(0, 0.3, count) / TICK) * TICK
    )
    highs = np.maximum(opens, closes) + np.round(rng.uniform(0, 1.0, count) / TICK) * TICK
    lows = np.minimum(opens, closes) - np.round(rng.uniform(0, 1.0, count) / TICK) * TICK
    return frame(ts, opens, highs, lows, closes)


def assert_valid_on_grid(bars: pd.DataFrame, tick: float = TICK) -> None:
    prices = bars[["open", "high", "low", "close"]].to_numpy()
    ticks = prices / tick
    assert np.abs(ticks - np.round(ticks)).max() < 1e-6
    assert (bars["high"] >= bars[["open", "close"]].max(axis=1)).all()
    assert (bars["low"] <= bars[["open", "close"]].min(axis=1)).all()
    assert (prices >= tick - 1e-9).all()


# ----- permuted bars -------------------------------------------------------------------


def test_within_a_session_the_bars_after_the_first_are_permuted_among_themselves():
    # Two sessions of three bars: the first bar of all keeps its move, the second
    # session's first bar is the only one of its set, and each session's other bars go
    # in the order of their keys: (9, 1) puts bar 2 before bar 1, (3, 2) bar 5 before 4.
    ts = [_ns(0, 0), _ns(0, 1), _ns(0, 2), _ns(1, 0), _ns(1, 1), _ns(1, 2)]
    keys = np.array([5, 9, 1, 7, 3, 2], dtype=np.uint64)
    assert source_positions(np.array(ts), ts[0], keys).tolist() == [0, 2, 1, 3, 5, 4]


def test_on_daily_bars_the_sessions_first_bars_are_permuted_across_sessions():
    ts = [_ns(day) for day in range(5)]
    keys = np.array([0, 4, 3, 2, 1], dtype=np.uint64)
    assert source_positions(np.array(ts), ts[0], keys).tolist() == [0, 4, 3, 2, 1]


def test_the_history_and_the_run_are_permuted_apart():
    # Six daily bars, the run from the fourth: the keys would put the run's bars first,
    # but each part keeps its own bars.
    ts = [_ns(day) for day in range(6)]
    keys = np.array([9, 8, 7, 1, 3, 2], dtype=np.uint64)
    assert source_positions(np.array(ts), ts[3], keys).tolist() == [0, 2, 1, 3, 5, 4]


def test_a_permuted_bar_is_rebuilt_from_the_moves_by_hand():
    # Bars (open, high, low, close): (100, 102, 99, 101), (103, 104, 102, 102) and
    # (101, 103, 100, 103). With the order [0, 2, 1] the second position takes the third
    # bar's move: open 101 x 101/102 = 100.0098, high x 103/101 = 101.98, low x
    # 100/101 = 99.02, close 101.98, which the 0.05 grid makes 100.00, 102.00, 99.00 and
    # 102.00. The third takes the second's from the close 101.98: open x 103/101 =
    # 104.01, high x 104/103 = 105.02, low and close x 102/103 = 103.00.
    bars = day_bars([100, 103, 101], [102, 104, 103], [99, 102, 100], [101, 102, 103])
    moved = rebuilt(bars, np.array([0, 2, 1]), TICK)
    expected = [[100, 102, 99, 101], [100, 102, 99, 102], [104, 105, 103, 103]]
    np.testing.assert_allclose(moved[["open", "high", "low", "close"]].to_numpy(), expected)
    assert moved["volume"].tolist() == [100.0, 300.0, 200.0]  # volume travels with its bar
    assert moved["ts_open"].tolist() == bars["ts_open"].tolist()  # times stay put


@pytest.mark.parametrize("run", [0, 1, 2, 3])
def test_every_permuted_bar_is_a_valid_bar_on_the_tick_grid(run):
    bars = walk_bars(days=6, per_day=30)
    (moved,) = permuted([bars], ["SBIN"], [TICK], int(bars["ts_open"].iloc[60]), 11, run)
    assert_valid_on_grid(moved)


def test_a_permuted_path_ends_at_the_real_last_close_and_keeps_the_history_close():
    bars = walk_bars(days=6, per_day=30)
    start = int(bars["ts_open"].iloc[60])
    (moved,) = permuted([bars], ["SBIN"], [TICK], start, 3, 0)
    assert moved["close"].iloc[-1] == pytest.approx(bars["close"].iloc[-1], abs=1e-9)
    assert moved["close"].iloc[59] == pytest.approx(bars["close"].iloc[59], abs=1e-9)
    assert not np.allclose(moved["close"], bars["close"])


def test_a_permutation_keeps_the_moves_and_changes_their_order():
    bars = walk_bars(days=4, per_day=20)
    (moved,) = permuted([bars], ["SBIN"], [TICK], int(bars["ts_open"].iloc[0]), 5, 0)
    ranges_before = np.sort((bars["high"] - bars["low"]).round(6).to_numpy())
    ranges_after = np.sort((moved["high"] - moved["low"]).round(6).to_numpy())
    # Rebuilt prices move each range by at most a tick of rounding at each end.
    assert np.abs(ranges_before - ranges_after).max() <= 2 * TICK + 1e-9
    assert sorted(moved["volume"]) == sorted(bars["volume"])


def test_several_instruments_are_permuted_with_one_order():
    first = walk_bars(days=5, per_day=20, seed=1)
    second = walk_bars(days=5, per_day=20, seed=2)
    a, b = permuted([first, second], ["SBIN", "INFY"], [TICK, TICK], 0, 9, 0)
    # The volumes were 100, 200, ... in both: equal after the permutation only when one
    # order moved both.
    assert a["volume"].tolist() == b["volume"].tolist()
    assert a["volume"].tolist() != first["volume"].tolist()


def test_instruments_on_different_times_cannot_be_permuted_together():
    first = walk_bars(days=5, per_day=20)
    with pytest.raises(DataError, match="not at the same times"):
        permuted([first, first.iloc[1:]], ["SBIN", "INFY"], [TICK, TICK], 0, 9, 0)


def test_the_same_seed_and_run_give_the_same_permuted_bars():
    bars = walk_bars(days=5, per_day=20)
    once = permuted([bars], ["SBIN"], [TICK], 0, 4, 2)[0]
    again = permuted([bars], ["SBIN"], [TICK], 0, 4, 2)[0]
    other_run = permuted([bars], ["SBIN"], [TICK], 0, 4, 3)[0]
    other_seed = permuted([bars], ["SBIN"], [TICK], 0, 5, 2)[0]
    pd.testing.assert_frame_equal(once, again, check_exact=True)
    assert not once.equals(other_run) and not once.equals(other_seed)


def test_a_permutation_leaves_the_bars_it_was_given_untouched():
    bars = walk_bars(days=3, per_day=10)
    kept = bars.copy()
    permuted([bars], ["SBIN"], [TICK], 0, 1, 0)
    pd.testing.assert_frame_equal(bars, kept, check_exact=True)


# ----- noisy bars ----------------------------------------------------------------------


def test_bars_of_a_constant_range_have_that_average_true_range_everywhere():
    # Every bar spans 2 rupees around 100 with no gap: every true range is 2.
    bars = day_bars([100] * 20, [101] * 20, [99] * 20, [100] * 20)
    np.testing.assert_allclose(ranges(bars, 14), np.full(20, 2.0))


def test_a_series_shorter_than_the_period_takes_its_mean_range():
    bars = day_bars([100, 100, 100], [101, 102, 103], [99, 99, 99], [100, 100, 100])
    np.testing.assert_allclose(ranges(bars, 14), np.full(3, 3.0))  # (2 + 3 + 4) / 3


def test_a_moved_price_is_within_its_share_of_the_range():
    # 2,000 bars spanning 10 rupees: a move is at most 0.2 x 10 = 2 rupees, and the grid
    # adds at most half a tick.
    count = 2_000
    bars = day_bars([500] * count, [505] * count, [495] * count, [500] * count)
    draws = np.random.default_rng(3).random(8 * count)
    moved = noisy_one(bars, TICK, Noise(), draws, np.full(count, 10.0))
    for name in ("open", "close"):
        assert (moved[name] - bars[name]).abs().max() <= 2.0 + TICK / 2 + 1e-9


def test_each_price_moves_with_its_own_chance():
    # A price moves when its first draw is below its chance; a shift that rounds back to
    # the same tick (under 0.025 of a reach of 2 rupees) is rare, about 1.25 percent.
    count = 4_000
    bars = day_bars([500] * count, [505] * count, [495] * count, [500] * count)
    (moved,) = noisy([bars], [TICK], Noise(), 21, 0, atrs=[np.full(count, 10.0)])
    shares = {name: float((moved[name] != bars[name]).mean()) for name in ("open", "close")}
    assert shares["open"] == pytest.approx(0.40, abs=0.03)
    assert shares["close"] == pytest.approx(0.75, abs=0.03)


def test_no_chance_and_no_reach_leave_the_bars_as_they_were():
    bars = walk_bars(days=3, per_day=20)
    still = Noise(chances=(0.0, 0.0, 0.0, 0.0))
    nothing = Noise(fraction=0.0)
    for noise in (still, nothing):
        (moved,) = noisy([bars], [TICK], noise, 1, 0)
        np.testing.assert_allclose(
            moved[["open", "high", "low", "close"]], bars[["open", "high", "low", "close"]]
        )


@pytest.mark.parametrize("run", [0, 1, 2])
def test_every_noisy_bar_is_a_valid_bar_on_the_tick_grid(run):
    bars = walk_bars(days=6, per_day=30)
    (moved,) = noisy([bars], [TICK], Noise(fraction=0.5), 13, run)
    assert_valid_on_grid(moved)
    assert moved["volume"].tolist() == bars["volume"].tolist()


def test_the_same_seed_and_run_give_the_same_noisy_bars():
    bars = walk_bars(days=4, per_day=20)
    once = noisy([bars], [TICK], Noise(), 4, 2)[0]
    again = noisy([bars], [TICK], Noise(), 4, 2)[0]
    other = noisy([bars], [TICK], Noise(), 4, 3)[0]
    pd.testing.assert_frame_equal(once, again, check_exact=True)
    assert not once.equals(other)


def test_several_instruments_take_their_own_draws():
    first = walk_bars(days=4, per_day=20, seed=1)
    a, b = noisy([first, first.copy()], [TICK, TICK], Noise(), 4, 0)
    assert not a.equals(b)
