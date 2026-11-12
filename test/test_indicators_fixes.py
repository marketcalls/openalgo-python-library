"""
Behaviour of the `openalgo.ta` indicators the backtest engine relies on (gap G-SDK-05).

Each test states one behaviour. Expected values come from the indicator's
published definition or are worked out by hand in the test; they are never
produced by the code under test. One table compares implementations on purpose:
it pins every NumPy fallback to its Rust kernel.

A test that takes the `backend` fixture runs twice: on the compiled Rust core
("rust", skipped when the extension is not installed) and on the pure NumPy
fallback ("numpy").

Run: .venv\\Scripts\\python.exe -m pytest test/test_indicators_fixes.py -q
"""

import inspect
import math
import warnings

import numpy as np
import pandas as pd
import pytest

from openalgo import ta
from openalgo.indicators import _backend, utils
from openalgo.indicators.hybrid import SAR, PivotPoints
from openalgo.indicators.momentum import Fisher
from openalgo.indicators.oscillators import CMO, RVI
from openalgo.indicators.trend import FRAMA
from openalgo.indicators.volatility import RVI as VolatilityRVI
from openalgo.indicators.volume import CMF, MFI, VWAP

NAN = float("nan")
# Two results agree when they differ by less than this, relative and absolute.
TOLERANCE = 1e-9


@pytest.fixture
def numpy_only():
    """Force the NumPy fallback, as on an install without the extension."""
    had_rust = _backend.HAVE_RUST
    _backend.HAVE_RUST = False
    try:
        yield
    finally:
        _backend.HAVE_RUST = had_rust


def assert_series(actual, expected):
    """Same length, NaN in the same places, values within TOLERANCE."""
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    assert actual.shape == expected.shape
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    np.testing.assert_allclose(actual, expected, rtol=TOLERANCE, atol=TOLERANCE)


def with_holes(values, seed):
    """A copy with a NaN warm-up (three bars) and six NaN bars further in."""
    rng = np.random.default_rng(seed)
    holed = np.array(values, dtype=np.float64)
    holed[:3] = np.nan
    holed[rng.integers(10, holed.size, 6)] = np.nan
    return holed


# ===========================================================================
# 1. Relative Vigor Index (ta.rvi)
# ===========================================================================

RVI_PERIOD = 10
SWMA_BARS = 4  # the symmetric weighted average spans four bars
# Bars whose body is half their range: close - open = 1 and high - low = 2.
FLAT_OPEN, FLAT_HIGH, FLAT_LOW, FLAT_CLOSE = 10.0, 12.0, 10.0, 11.0
FLAT_RVI = 0.5


def flat_bars(bars):
    return (
        np.full(bars, FLAT_OPEN),
        np.full(bars, FLAT_HIGH),
        np.full(bars, FLAT_LOW),
        np.full(bars, FLAT_CLOSE),
    )


def swma(series):
    """Symmetric weighted moving average, weights 1-2-2-1 over four bars."""
    return (series.shift(3) + 2 * series.shift(2) + 2 * series.shift(1) + series) / 6.0


def test_rvi_equals_the_published_formula(backend, market):
    """rvi = sum(swma(c-o), n) / sum(swma(h-l), n), signal = swma(rvi)."""
    open_, high, low, close, _ = market(300, seed=7)
    body = swma(pd.Series(close - open_)).rolling(RVI_PERIOD).sum()
    bar_range = swma(pd.Series(high - low)).rolling(RVI_PERIOD).sum()
    expected = body / bar_range
    rvi, signal = ta.rvi(open_, high, low, close, RVI_PERIOD)
    assert_series(rvi, expected)
    assert_series(signal, swma(expected))


def test_rvi_is_nan_while_a_nan_bar_is_in_its_window(backend):
    hole = 20
    open_, high, low, close = flat_bars(45)
    open_[hole] = NAN
    rvi, _ = ta.rvi(open_, high, low, close, RVI_PERIOD)
    # The NaN sits in the swma of bars 20..23 and so in every 10-bar sum that
    # includes one of them: bars 20..32.
    last_affected = hole + SWMA_BARS - 1 + RVI_PERIOD - 1
    expected = np.full(45, FLAT_RVI)
    expected[: RVI_PERIOD + 2] = NAN
    expected[hole : last_affected + 1] = NAN
    assert_series(rvi, expected)


def test_rvi_waits_for_a_nan_warmup_to_leave_its_window(backend):
    warmup = 4
    open_, high, low, close = flat_bars(30)
    open_[:warmup] = NAN
    rvi, _ = ta.rvi(open_, high, low, close, RVI_PERIOD)
    # swma is NaN up to bar 6, so the first clean 10-bar sum ends at bar 16.
    first_value = warmup + SWMA_BARS - 1 + RVI_PERIOD - 1
    expected = np.full(30, FLAT_RVI)
    expected[:first_value] = NAN
    assert_series(rvi, expected)


# ===========================================================================
# 2. Relative Volatility Index (ta.rvi_volatility)
#
# Dorsey's definition: s = population standard deviation of the last
# `stdev_period` closes; s counts as "up" on a bar that closes higher and as
# "down" on one that closes lower; U and D are Wilder averages of the two
# (seeded with a plain mean); RVI = 100 * U / (U + D).
# ===========================================================================

RVI_VOL_FIRST_BAR = 10 + 14 - 2  # stdev_period + rsi_period - 2 at the defaults


def test_rvi_volatility_is_zero_when_no_bar_closes_higher(backend):
    bar = np.arange(40.0)
    falling = 1000.0 - bar * bar
    expected = [NAN] * RVI_VOL_FIRST_BAR + [0.0] * (40 - RVI_VOL_FIRST_BAR)
    assert_series(ta.rvi_volatility(falling), expected)


def test_rvi_volatility_is_one_hundred_when_no_bar_closes_lower(backend):
    bar = np.arange(40.0)
    rising = 1000.0 + 50.0 * bar - bar * bar / 2.0  # each rise is smaller than the last
    expected = [NAN] * RVI_VOL_FIRST_BAR + [100.0] * (40 - RVI_VOL_FIRST_BAR)
    assert_series(ta.rvi_volatility(rising), expected)


def test_rvi_volatility_matches_a_hand_computed_series(backend):
    closes = [1.0, 3.0, 2.0, 6.0, 6.0, 5.0]
    # Two-bar deviations |change| / 2:  1, 0.5, 2, 0, 0.5 from bar 1.
    # Sides (up, down):   bar 1 (1, 0)   bar 2 (0, 0.5)   bar 3 (2, 0)
    #                     bar 4 (0, 0)   bar 5 (0, 0.5)
    # Wilder over 2 bars, seeded on bars 1-2:  U = 0.5, D = 0.25   -> 200/3
    #   bar 3: U = (0.5 + 2)/2 = 1.25,   D = 0.125                 -> 1000/11
    #   bar 4: U = 0.625,                D = 0.0625                -> 1000/11
    #   bar 5: U = 0.3125,               D = (0.0625 + 0.5)/2      -> 1000/19
    expected = [NAN, NAN, 200 / 3, 1000 / 11, 1000 / 11, 1000 / 19]
    assert_series(ta.rvi_volatility(np.array(closes), 2, 2), expected)


def test_rvi_volatility_is_fifty_when_prices_do_not_move(backend):
    flat = np.full(30, 7.0)
    expected = [NAN] * RVI_VOL_FIRST_BAR + [50.0] * (30 - RVI_VOL_FIRST_BAR)
    assert_series(ta.rvi_volatility(flat), expected)


def test_rvi_volatility_keeps_its_averages_across_a_nan_bar(backend):
    closes = [1.0, 3.0, 2.0, 6.0, NAN, 5.0, 7.0, 6.0, 9.0]
    # Bars 4 and 5 have no deviation or no change. U and D stay at their bar-3
    # values (1.25, 0.125) and carry on:
    #   bar 6: up 1     -> U = 1.125,    D = 0.0625    -> 1800/19
    #   bar 7: down 0.5 -> U = 0.5625,   D = 0.28125   -> 200/3
    #   bar 8: up 1.5   -> U = 1.03125,  D = 0.140625  -> 88
    expected = [NAN, NAN, 200 / 3, 1000 / 11, NAN, NAN, 1800 / 19, 200 / 3, 88.0]
    assert_series(ta.rvi_volatility(np.array(closes), 2, 2), expected)


def test_rvi_volatility_keeps_the_index_of_a_series(market):
    close = pd.Series(market(60, seed=3)[3], index=pd.date_range("2025-01-01", periods=60))
    result = ta.rvi_volatility(close)
    assert isinstance(result, pd.Series)
    assert result.index.equals(close.index)


def test_volatility_class_and_ta_function_are_the_same_indicator(market):
    close = market(80, seed=3)[3]
    assert_series(VolatilityRVI().calculate(close, 5, 7), ta.rvi_volatility(close, 5, 7))


# ===========================================================================
# 3. Parabolic SAR (ta.psar)
#
# Expected values are worked out by hand from the order documented on
# `_backend.sar`: project the stop, test the bar against it with strict
# comparisons, then clamp to the two previous bars. Acceleration starts at
# 0.02, steps by 0.02 and is capped at 0.2 unless a test says otherwise.
# ===========================================================================

# Five rising bars, then bar 5 is an outside bar: a new high (15, above the
# extreme point 14) and a low (9.5) below the projected stop 9.93704.
UP_THEN_OUTSIDE_HIGH = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 14.5, 14.2, 14.0]
UP_THEN_OUTSIDE_LOW = [9.0, 10.0, 11.0, 12.0, 13.0, 9.5, 13.5, 13.0, 12.5]
# Bar 1 reverses the initial uptrend, bars 2 to 4 fall, then bar 5 is an
# outside bar: a new low (14, below the extreme point 15) and a high (21.5)
# above the projected stop 19.275424.
DOWN_THEN_OUTSIDE_HIGH = [20.0, 19.0, 18.0, 17.0, 16.0, 21.5, 20.0]
DOWN_THEN_OUTSIDE_LOW = [19.0, 18.0, 17.0, 16.0, 15.0, 14.0, 14.6]
REVERSAL_BAR = 5
LONG, SHORT = 1.0, -1.0


def psar_with_trend(high, low, acceleration=0.02, maximum=0.2):
    sar, trend = ta.psar(np.array(high), np.array(low), acceleration, maximum, return_trend=True)
    return np.asarray(sar), np.asarray(trend)


def test_psar_uptrend_follows_the_hand_computed_path(backend):
    sar, trend = psar_with_trend(UP_THEN_OUTSIDE_HIGH[:5], UP_THEN_OUTSIDE_LOW[:5])
    # bar 1: 9 + 0.02*(10-9) = 9.02, capped at low[0] = 9
    # bar 2: 9 + 0.04*(11-9) = 9.08, capped at low[0] = 9
    # bar 3: 9 + 0.06*(12-9) = 9.18
    # bar 4: 9.18 + 0.08*(13-9.18) = 9.4856
    assert_series(sar, [9.0, 9.0, 9.0, 9.18, 9.4856])
    assert_series(trend, [LONG] * 5)


def test_psar_reversal_to_short_starts_from_the_higher_of_high_and_extreme_point(backend):
    sar, trend = psar_with_trend(UP_THEN_OUTSIDE_HIGH, UP_THEN_OUTSIDE_LOW)
    assert trend[REVERSAL_BAR] == SHORT
    # max(high 15, extreme point 14); the two previous highs are lower
    assert sar[REVERSAL_BAR] == pytest.approx(15.0, abs=TOLERANCE)


def test_psar_stays_short_while_the_high_is_below_the_projected_stop(backend):
    sar, trend = psar_with_trend(UP_THEN_OUTSIDE_HIGH, UP_THEN_OUTSIDE_LOW)
    assert_series(trend[REVERSAL_BAR:], [SHORT] * 4)
    # bar 6: 15 + 0.02*(9.5-15) = 14.89, raised to the high of bar 5
    # bar 7: 14.89 again, raised to the high of bar 5
    # bar 8: 14.89, above the highs of bars 7 and 6, so it stands
    assert_series(sar[REVERSAL_BAR + 1 :], [15.0, 15.0, 14.89])


def test_psar_downtrend_follows_the_hand_computed_path(backend):
    sar, trend = psar_with_trend(DOWN_THEN_OUTSIDE_HIGH[:5], DOWN_THEN_OUTSIDE_LOW[:5])
    # bar 1: 19.02 is above the low of 18: reverse, max(high 19, extreme point 20) = 20
    # bar 2: 20 + 0.02*(18-20) = 19.96, raised to the high of bar 0
    # bar 3: 20 + 0.04*(17-20) = 19.88
    # bar 4: 19.88 + 0.06*(16-19.88) = 19.6472
    assert_series(sar, [19.0, 20.0, 20.0, 19.88, 19.6472])
    assert_series(trend, [LONG] + [SHORT] * 4)


def test_psar_reversal_to_long_starts_from_the_lower_of_low_and_extreme_point(backend):
    sar, trend = psar_with_trend(DOWN_THEN_OUTSIDE_HIGH, DOWN_THEN_OUTSIDE_LOW)
    assert trend[REVERSAL_BAR] == LONG
    # min(low 14, extreme point 15); the two previous lows are higher
    assert sar[REVERSAL_BAR] == pytest.approx(14.0, abs=TOLERANCE)


def test_psar_stays_long_while_the_low_is_above_the_projected_stop(backend):
    sar, trend = psar_with_trend(DOWN_THEN_OUTSIDE_HIGH, DOWN_THEN_OUTSIDE_LOW)
    assert trend[REVERSAL_BAR + 1] == LONG
    # 14 + 0.02*(21.5-14) = 14.15, capped at the low of bar 5
    assert sar[REVERSAL_BAR + 1] == pytest.approx(14.0, abs=TOLERANCE)


def test_psar_reversal_stop_is_clamped_to_the_bar_two_back(backend):
    high = [20.0, 19.5, 19.7, 19.6, 19.0]
    low = [19.0, 10.0, 9.0, 19.0, 9.0]
    sar, trend = psar_with_trend(high, low)
    # bar 1: 19.02 > low 10: short, max(19.5, 20) = 20
    # bar 2: 19.8 is not below high 19.7; raised to the high of bar 0 = 20
    # bar 3: 20 + 0.04*(9-20) = 19.56 < high 19.6: long, min(low 19, extreme 9) = 9
    # bar 4: 9 + 0.02*(19.6-9) = 9.212 > low 9: short, max(19.0, 19.6) = 19.6,
    #        then raised to the high of bar 2 = 19.7
    assert_series(sar, [19.0, 20.0, 20.0, 9.0, 19.7])
    assert_series(trend, [LONG, SHORT, SHORT, LONG, SHORT])


def test_psar_does_not_reverse_when_the_low_only_touches_the_stop(backend):
    # Step 0.25 keeps the arithmetic exact: 8 + 0.25*(12-8) = 9, the low of bar 1.
    sar, trend = psar_with_trend([12.0, 13.0], [8.0, 9.0], acceleration=0.25, maximum=0.5)
    assert_series(sar, [8.0, 8.0])  # 9 capped at the low of bar 0
    assert_series(trend, [LONG, LONG])


def test_psar_does_not_reverse_when_the_high_only_touches_the_stop(backend):
    # bar 1: 9 > low 7: short, stop max(high 11, extreme point 12) = 12
    # bar 2: 12 + 0.25*(7-12) = 10.75, the high of bar 2: no reversal, raised to 12
    sar, trend = psar_with_trend(
        [12.0, 11.0, 10.75], [8.0, 7.0, 9.0], acceleration=0.25, maximum=0.5
    )
    assert_series(sar, [8.0, 12.0, 12.0])
    assert_series(trend, [LONG, SHORT, SHORT])


def test_psar_of_a_constant_series_never_reverses(backend):
    sar, trend = psar_with_trend([5.0] * 6, [5.0] * 6)
    assert_series(sar, [5.0] * 6)
    assert_series(trend, [LONG] * 6)


def test_psar_starts_long_at_the_first_bar_that_has_a_high_and_a_low(backend):
    sar, trend = psar_with_trend(
        [NAN, 11.0, 12.0, 13.0, 9.0, 8.0], [9.0, 10.0, 11.0, 12.0, 7.0, 6.0]
    )
    # Seeded at bar 1: stop 10, extreme point 11.
    # bar 2: 10.02 capped at the low of bar 1 = 10;  bar 3: 10.08 capped at 10
    # bar 4: 10.18 > low 7: short, max(high 9, extreme point 13) = 13
    # bar 5: 13 + 0.02*(7-13) = 12.88, raised to the high of bar 3 = 13
    assert_series(sar, [NAN, 10.0, 10.0, 10.0, 13.0, 13.0])
    assert_series(trend, [NAN, LONG, LONG, LONG, SHORT, SHORT])


def test_psar_skips_a_nan_bar_as_if_it_were_not_there(backend):
    gap = 3
    high = UP_THEN_OUTSIDE_HIGH[:gap] + [NAN] + UP_THEN_OUTSIDE_HIGH[gap:]
    low = UP_THEN_OUTSIDE_LOW[:gap] + [11.5] + UP_THEN_OUTSIDE_LOW[gap:]
    sar, trend = psar_with_trend(high, low)
    # The values of the series without the gap, worked out in the tests above.
    without_gap = [9.0, 9.0, 9.0, 9.18, 9.4856, 15.0, 15.0, 15.0, 14.89]
    assert_series(sar, without_gap[:gap] + [NAN] + without_gap[gap:])
    assert_series(trend, [LONG] * gap + [NAN] + [LONG] * 2 + [SHORT] * 4)


def test_psar_keeps_clear_of_the_two_previous_bars_on_wide_bars(backend, market):
    _, high, low, _, _ = market(600, seed=5, bar_range=2.5)
    sar, trend = psar_with_trend(high, low)
    bars = np.arange(2, len(sar))
    long = bars[trend[bars] == LONG]
    short = bars[trend[bars] == SHORT]
    assert len(long) > 50 and len(short) > 50
    assert (sar[long] <= np.minimum(low[long - 1], low[long - 2])).all()
    assert (sar[short] >= np.maximum(high[short - 1], high[short - 2])).all()


def test_psar_returns_the_sar_values_only_by_default():
    high, low = np.array(UP_THEN_OUTSIDE_HIGH), np.array(UP_THEN_OUTSIDE_LOW)
    result = ta.psar(high, low)
    assert isinstance(result, np.ndarray)
    assert result.shape == high.shape


def test_psar_returns_sar_and_trend_on_request():
    high, low = pd.Series(UP_THEN_OUTSIDE_HIGH), pd.Series(UP_THEN_OUTSIDE_LOW)
    sar, trend = ta.psar(high, low, return_trend=True)
    assert isinstance(sar, pd.Series) and isinstance(trend, pd.Series)
    assert_series(sar, ta.psar(high, low))
    assert set(trend.unique()) <= {LONG, SHORT}


# ===========================================================================
# 4. VWAP (ta.vwap)
# ===========================================================================

VWAP_BARS_PER_DAY = 75
# Thursday and Friday, then Monday to Thursday of the next week, across a month end.
VWAP_DAYS = ["2025-03-27", "2025-03-28", "2025-03-31", "2025-04-01", "2025-04-02", "2025-04-03"]
VWAP_WEEK_OF_DAY = [0, 0, 1, 1, 1, 1]
VWAP_MONTH_OF_DAY = [0, 0, 0, 1, 1, 1]
# The anchors documented for ta.vwap.
DOCUMENTED_ANCHORS = [
    "Session",
    "Week",
    "Month",
    "Quarter",
    "Year",
    "12M",
    "6M",
    "3M",
    "D",
    "4H",
    "1H",
    "30m",
    "15m",
    "5m",
    "1m",
]


@pytest.fixture(scope="module")
def intraday(market):
    """Five-minute bars from 09:15 India time on each of VWAP_DAYS."""
    sessions = [
        pd.date_range(f"{day} 09:15", periods=VWAP_BARS_PER_DAY, freq="5min", tz="Asia/Kolkata")
        for day in VWAP_DAYS
    ]
    index = sessions[0]
    for session in sessions[1:]:
        index = index.append(session)
    open_, high, low, close, volume = market(len(index), seed=3)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=index
    )


def typical_price(frame):
    return ((frame["high"] + frame["low"] + frame["close"]) / 3.0).to_numpy()


def anchored_vwap(price, volume, period_of_bar):
    """sum(price * volume) / sum(volume), both sums restarting when the period changes."""
    out = np.empty(len(price))
    for bar in range(len(price)):
        if bar == 0 or period_of_bar[bar] != period_of_bar[bar - 1]:
            turnover = traded = 0.0
        turnover += price[bar] * volume[bar]
        traded += volume[bar]
        out[bar] = turnover / traded
    return out


def vwap_of(frame, **kwargs):
    return ta.vwap(frame["high"], frame["low"], frame["close"], frame["volume"], **kwargs)


def expected_vwap(frame, period_of_day):
    period_of_bar = np.repeat(period_of_day, VWAP_BARS_PER_DAY)
    return anchored_vwap(typical_price(frame), frame["volume"].to_numpy(), period_of_bar)


def test_vwap_session_anchor_restarts_on_every_trading_day(backend, intraday):
    assert_series(vwap_of(intraday), expected_vwap(intraday, range(len(VWAP_DAYS))))


def test_vwap_first_bar_of_a_session_is_its_own_typical_price(intraday):
    first_bar_of_day_two = VWAP_BARS_PER_DAY
    vwap = vwap_of(intraday).to_numpy()
    assert vwap[first_bar_of_day_two] == pytest.approx(
        typical_price(intraday)[first_bar_of_day_two]
    )


def test_vwap_week_anchor_restarts_on_the_first_bar_of_each_week(backend, intraday):
    assert_series(vwap_of(intraday, anchor="Week"), expected_vwap(intraday, VWAP_WEEK_OF_DAY))


def test_vwap_month_anchor_restarts_on_the_first_bar_of_each_month(backend, intraday):
    assert_series(vwap_of(intraday, anchor="Month"), expected_vwap(intraday, VWAP_MONTH_OF_DAY))


def test_vwap_hour_anchor_is_measured_from_the_first_bar_of_the_day(intraday):
    bars_per_hour = 12
    frame = intraday.iloc[: 2 * VWAP_BARS_PER_DAY]
    # 75 five-minute bars from 09:15: hours start at bars 0, 12, 24, ... of each day.
    hour_of_bar = np.arange(len(frame)) // bars_per_hour
    hour_of_bar[VWAP_BARS_PER_DAY:] = 100 + np.arange(VWAP_BARS_PER_DAY) // bars_per_hour
    expected = anchored_vwap(typical_price(frame), frame["volume"].to_numpy(), hour_of_bar)
    assert_series(vwap_of(frame, anchor="1H"), expected)


def test_vwap_without_an_anchor_is_one_running_average(intraday):
    assert_series(vwap_of(intraday, anchor=None), expected_vwap(intraday, [0] * len(VWAP_DAYS)))


def test_vwap_timestamps_argument_anchors_plain_arrays(intraday):
    arrays = [intraday[column].to_numpy() for column in ("high", "low", "close", "volume")]
    vwap = ta.vwap(*arrays, timestamps=intraday.index)
    assert_series(vwap, expected_vwap(intraday, range(len(VWAP_DAYS))))


def test_vwap_session_starts_argument_overrides_the_anchor(intraday):
    frame = intraday.iloc[:VWAP_BARS_PER_DAY]
    restart_at = 30
    starts = np.zeros(len(frame), dtype=bool)
    starts[restart_at] = True
    period_of_bar = (np.arange(len(frame)) >= restart_at).astype(int)
    expected = anchored_vwap(typical_price(frame), frame["volume"].to_numpy(), period_of_bar)
    assert_series(vwap_of(frame, session_starts=starts), expected)


def test_vwap_session_is_the_calendar_day_of_the_index_time_zone(market):
    # 04:30 to 06:25 India time is 23:00 to 00:55 UTC: one local day, two UTC days.
    index = pd.date_range("2025-04-01 04:30", periods=24, freq="5min", tz="Asia/Kolkata")
    _, high, low, close, volume = market(len(index), seed=9)
    vwap = ta.vwap(
        pd.Series(high, index=index),
        pd.Series(low, index=index),
        pd.Series(close, index=index),
        pd.Series(volume, index=index),
    )
    one_session = anchored_vwap((high + low + close) / 3.0, volume, np.zeros(len(index)))
    assert_series(vwap, one_session)


def test_vwap_class_bands_restart_with_the_anchor(intraday):
    vwap, _, _ = VWAP().calculate_with_bands(
        intraday["high"], intraday["low"], intraday["close"], intraday["volume"]
    )
    assert_series(vwap, expected_vwap(intraday, range(len(VWAP_DAYS))))


# --- which inputs carry bar times: four bars, two per day, typical price 9, 19, 29, 39

FOUR_HIGH = np.array([10.0, 20.0, 30.0, 40.0])
FOUR_LOW = FOUR_HIGH - 2.0
FOUR_CLOSE = FOUR_HIGH - 1.0
FOUR_TIMES = pd.DatetimeIndex(
    ["2025-04-01 09:15", "2025-04-01 09:20", "2025-04-02 09:15", "2025-04-02 09:20"]
)
RESTARTS_EACH_DAY = [9.0, 14.0, 29.0, 34.0]
ONE_RUNNING_AVERAGE = [9.0, 14.0, 19.0, 24.0]


def four_bars(index, volume=(1.0, 1.0, 1.0, 1.0)):
    """The four bars as (high, low, close, volume) Series on `index`."""
    columns = (FOUR_HIGH, FOUR_LOW, FOUR_CLOSE, np.array(volume))
    return [pd.Series(column, index=index) for column in columns]


@pytest.mark.parametrize(
    "index",
    [
        FOUR_TIMES,
        FOUR_TIMES.tz_localize("Asia/Kolkata"),
        FOUR_TIMES.to_period("min"),
        pd.Index([stamp.date() for stamp in FOUR_TIMES]),
        pd.Index(list(FOUR_TIMES.to_pydatetime())),
    ],
    ids=["DatetimeIndex", "tz-aware DatetimeIndex", "PeriodIndex", "date objects", "datetimes"],
)
def test_vwap_reads_bar_times_from_an_index_of_dates(index):
    assert_series(ta.vwap(*four_bars(index)), RESTARTS_EACH_DAY)


@pytest.mark.parametrize(
    "index",
    [pd.RangeIndex(4), pd.Index([str(stamp) for stamp in FOUR_TIMES]), pd.Index([3, 5, 8, 9])],
    ids=["RangeIndex", "strings", "integers"],
)
def test_vwap_treats_any_other_index_as_one_period(index):
    assert_series(ta.vwap(*four_bars(index)), ONE_RUNNING_AVERAGE)


def test_vwap_of_plain_arrays_is_one_period():
    assert_series(ta.vwap(FOUR_HIGH, FOUR_LOW, FOUR_CLOSE, np.ones(4)), ONE_RUNNING_AVERAGE)


def test_vwap_reads_bar_times_from_the_first_series_among_its_inputs():
    _, low, close, volume = four_bars(FOUR_TIMES)
    assert_series(ta.vwap(FOUR_HIGH, low, close, volume), RESTARTS_EACH_DAY)


@pytest.mark.parametrize("anchor", DOCUMENTED_ANCHORS)
def test_vwap_accepts_every_documented_anchor(anchor):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # "1m" on these bars is one bar per period
        result = ta.vwap(*four_bars(FOUR_TIMES), anchor=anchor)
    assert np.isfinite(result).all()


@pytest.mark.parametrize("anchor", ["Fortnight", "Decade", "Century", "2D", "W", "M", "1h", "", 5])
@pytest.mark.parametrize("container", ["arrays", "series", "session_starts"])
def test_vwap_rejects_an_undocumented_anchor_for_every_kind_of_input(anchor, container):
    if container == "arrays":
        args, extra = (FOUR_HIGH, FOUR_LOW, FOUR_CLOSE, np.ones(4)), {}
    elif container == "series":
        args, extra = four_bars(FOUR_TIMES), {}
    else:
        args, extra = four_bars(FOUR_TIMES), {"session_starts": [True, False, True, False]}
    with pytest.raises(ValueError, match="Unknown anchor"):
        ta.vwap(*args, anchor=anchor, **extra)


@pytest.mark.parametrize(
    "index",
    [
        pd.bdate_range("2025-01-01", periods=4),
        pd.date_range("2025-01-03", periods=4, freq="W-FRI"),
        pd.date_range("2025-01-31", periods=4, freq="ME"),
    ],
    ids=["daily", "weekly", "monthly"],
)
def test_vwap_warns_when_every_anchor_period_holds_a_single_bar(index):
    with pytest.warns(UserWarning, match="anchor=None"):
        vwap = ta.vwap(*four_bars(index))
    assert_series(vwap, [9.0, 19.0, 29.0, 39.0])  # each bar's own typical price


@pytest.mark.parametrize("anchor", [None, "Month"])
def test_vwap_on_daily_bars_does_not_warn_with_a_longer_anchor(anchor):
    daily = pd.bdate_range("2025-01-01", periods=4)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        vwap = ta.vwap(*four_bars(daily), anchor=anchor)
    assert_series(vwap, ONE_RUNNING_AVERAGE)


def test_vwap_is_nan_until_the_period_has_volume(backend):
    vwap = ta.vwap(FOUR_HIGH, FOUR_LOW, FOUR_CLOSE, np.array([0.0, 0.0, 5.0, 5.0]))
    assert_series(vwap, [NAN, NAN, 29.0, 34.0])


def test_vwap_of_a_prefix_without_volume_is_nan_not_an_error(backend):
    vwap = ta.vwap(FOUR_HIGH[:2], FOUR_LOW[:2], FOUR_CLOSE[:2], np.zeros(2))
    assert_series(vwap, [NAN, NAN])


def test_vwap_skips_a_bar_with_nan_volume_and_keeps_its_sums(backend):
    vwap = ta.vwap(FOUR_HIGH, FOUR_LOW, FOUR_CLOSE, np.array([1.0, NAN, 1.0, 1.0]))
    assert_series(vwap, [9.0, NAN, (9 + 29) / 2, (9 + 29 + 39) / 3])


def test_vwap_restarts_on_a_session_whose_first_bar_is_nan(backend):
    high, low, close, volume = four_bars(FOUR_TIMES)
    close.iloc[2] = NAN  # first bar of day two
    # Day two holds one usable bar: typical price 39.
    assert_series(ta.vwap(high, low, close, volume), [9.0, 14.0, NAN, 39.0])


# ===========================================================================
# 5. Pivot points (ta.pivot_points)
# ===========================================================================

# Two sessions of three bars. Day one: high 12, low 9, last close 10.5.
PIVOT_INDEX = pd.DatetimeIndex(
    [
        "2025-04-01 09:15",
        "2025-04-01 12:15",
        "2025-04-01 15:15",
        "2025-04-02 09:15",
        "2025-04-02 12:15",
        "2025-04-02 15:15",
    ]
)
PIVOT_HIGH = [10.0, 12.0, 11.0, 13.0, 14.0, 13.5]
PIVOT_LOW = [9.0, 10.0, 10.0, 12.0, 12.5, 12.0]
PIVOT_CLOSE = [9.5, 11.0, 10.5, 12.5, 13.5, 12.5]
# From day one (H 12, L 9, C 10.5): P = 31.5/3, R1 = 2P-L, S1 = 2P-H,
# R2 = P+(H-L), S2 = P-(H-L), R3 = H+2(P-L), S3 = L-2(H-P).
DAY_ONE_LEVELS = [10.5, 12.0, 9.0, 13.5, 7.5, 15.0, 6.0]
PIVOT_LEVEL_NAMES = ["P", "R1", "S1", "R2", "S2", "R3", "S3"]
BARS_PER_PIVOT_DAY = 3


def pivot_series(values):
    return pd.Series(values, index=PIVOT_INDEX)


def daily_pivots(high=PIVOT_HIGH, low=PIVOT_LOW, close=PIVOT_CLOSE):
    return ta.pivot_points(pivot_series(high), pivot_series(low), pivot_series(close), anchor="D")


def test_pivot_points_without_an_anchor_use_the_bars_own_prices():
    pivot, r1, s1, *_ = ta.pivot_points(
        np.array(PIVOT_HIGH), np.array(PIVOT_LOW), np.array(PIVOT_CLOSE)
    )
    # Bar 0: H 10, L 9, C 9.5 -> P 9.5, R1 = 19 - 9 = 10, S1 = 19 - 10 = 9.
    assert (pivot[0], r1[0], s1[0]) == pytest.approx((9.5, 10.0, 9.0))


@pytest.mark.parametrize("level", range(len(DAY_ONE_LEVELS)), ids=PIVOT_LEVEL_NAMES)
def test_pivot_points_with_a_daily_anchor_come_from_the_previous_day(level):
    day_two = daily_pivots()[level].to_numpy()[BARS_PER_PIVOT_DAY:]
    assert_series(day_two, [DAY_ONE_LEVELS[level]] * BARS_PER_PIVOT_DAY)


def test_pivot_points_with_an_anchor_are_nan_in_the_first_period():
    for level in daily_pivots():
        assert level.iloc[:BARS_PER_PIVOT_DAY].isna().all()


def test_pivot_points_with_an_anchor_do_not_depend_on_the_current_period():
    moved = [value + 50.0 for value in PIVOT_HIGH[BARS_PER_PIVOT_DAY:]]
    changed_high = PIVOT_HIGH[:BARS_PER_PIVOT_DAY] + moved
    for before, after in zip(daily_pivots(), daily_pivots(high=changed_high), strict=True):
        assert_series(after, before)


def test_pivot_points_are_nan_after_a_period_with_a_nan_high():
    high = [10.0, NAN, 11.0] + PIVOT_HIGH[BARS_PER_PIVOT_DAY:]
    for level in daily_pivots(high=high):
        assert level.isna().all()


def test_pivot_points_with_a_weekly_anchor_come_from_the_previous_week():
    index = pd.bdate_range("2025-03-31", periods=10)  # Monday to Friday, twice
    high = np.arange(101.0, 111.0)
    low = high - 2.0
    close = high - 0.5
    pivot = ta.pivot_points(
        pd.Series(high, index=index),
        pd.Series(low, index=index),
        pd.Series(close, index=index),
        anchor="Week",
    )[0]
    # Week one: highest high 105, lowest low 99, last close 104.5.
    week_one_pivot = (105.0 + 99.0 + 104.5) / 3.0
    assert_series(pivot, [NAN] * 5 + [week_one_pivot] * 5)


def test_pivot_points_with_an_anchor_accept_timestamps_for_plain_arrays():
    pivot = ta.pivot_points(
        np.array(PIVOT_HIGH),
        np.array(PIVOT_LOW),
        np.array(PIVOT_CLOSE),
        anchor="D",
        timestamps=PIVOT_INDEX,
    )[0]
    assert_series(pivot, [NAN] * BARS_PER_PIVOT_DAY + [DAY_ONE_LEVELS[0]] * BARS_PER_PIVOT_DAY)


def test_pivot_points_with_an_anchor_need_bar_times():
    with pytest.raises(ValueError, match="needs bar times"):
        ta.pivot_points(np.array(PIVOT_HIGH), np.array(PIVOT_LOW), np.array(PIVOT_CLOSE), "D")


@pytest.mark.parametrize("anchor", ["W", "Fortnight", "2D"])
def test_pivot_points_reject_an_undocumented_anchor(anchor):
    with pytest.raises(ValueError, match="Unknown anchor"):
        ta.pivot_points(
            pivot_series(PIVOT_HIGH), pivot_series(PIVOT_LOW), pivot_series(PIVOT_CLOSE), anchor
        )


# ===========================================================================
# 6. DEMA and TEMA with NaN in the input
# ===========================================================================

# One unknown bar inside a short series.
GAPPED = np.array([1.0, 2.0, 3.0, 4.0, 5.0, NAN, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
GAP_BAR = 5
# Period 2, so alpha = 2/3. Each EMA starts at its first value and keeps its
# state across the NaN bar; the second EMA starts at bar 1, the third at bar 2.
#   bar            0    1      2        3         4      5       6
#   EMA            1   5/3   23/9    95/27    365/81    NaN  1499/243
#   EMA of EMA         5/3   61/27   251/81   109/27    NaN  3979/729
#   EMA of that              61/27   685/243  2647/729  NaN  3535/729
# DEMA[6] = 2 * 1499/243 - 3979/729
# TEMA[6] = 3 * 1499/243 - 3 * 3979/729 + 3535/729
DEMA_AT_BAR_6 = 5015 / 729
TEMA_AT_BAR_6 = 5089 / 729


def test_tema_after_a_nan_bar_is_the_state_carrying_value(backend):
    assert ta.tema(GAPPED, 2)[6] == pytest.approx(TEMA_AT_BAR_6, abs=TOLERANCE)


def test_dema_after_a_nan_bar_is_the_state_carrying_value(backend):
    assert ta.dema(GAPPED, 2)[6] == pytest.approx(DEMA_AT_BAR_6, abs=TOLERANCE)


@pytest.mark.parametrize("name", ["dema", "tema"])
def test_value_after_a_nan_bar_does_not_depend_on_later_bars(name, backend):
    function = getattr(ta, name)
    full = function(GAPPED, 2)
    for bars in range(GAP_BAR + 2, len(GAPPED)):
        assert_series(function(GAPPED[:bars], 2), full[:bars])


@pytest.mark.parametrize("name", ["dema", "tema"])
def test_only_the_nan_bar_itself_is_nan_after_the_warmup(name, backend):
    warmup = {"dema": 2, "tema": 3}[name]  # 2 or 3 times (period - 1)
    result = np.asarray(getattr(ta, name)(GAPPED, 2))
    assert np.flatnonzero(np.isnan(result)).tolist() == [*range(warmup), GAP_BAR]


@pytest.mark.parametrize("name", ["dema", "tema"])
def test_a_nan_warmup_shifts_the_result_without_changing_it(name, backend, market):
    close = market(60, seed=4)[3]
    padded = np.concatenate([[NAN] * 4, close])
    function = getattr(ta, name)
    assert_series(function(padded, 5), np.concatenate([[NAN] * 4, function(close, 5)]))


# ===========================================================================
# 7. Rolling extremes (ta.highest, ta.lowest)
# ===========================================================================


@pytest.mark.parametrize(
    "helper, expected",
    [
        ("highest", [NAN, NAN, NAN, NAN, 3.0, NAN, NAN, NAN, 7.0, 8.0]),
        ("lowest", [NAN, NAN, NAN, NAN, 1.0, NAN, NAN, NAN, 5.0, 6.0]),
    ],
)
def test_rolling_extreme_is_nan_for_a_window_holding_a_nan(helper, expected, backend):
    """The same rule as pandas `.rolling(3).max()` and `.min()`."""
    data = [NAN, NAN, 1.0, 2.0, 3.0, NAN, 5.0, 6.0, 7.0, 8.0]
    assert_series(getattr(ta, helper)(data, 3), expected)


@pytest.mark.parametrize("helper", ["highest", "lowest"])
@pytest.mark.parametrize("period", [6, 10**12, 2**63 - 1])
def test_rolling_extreme_with_a_period_longer_than_the_series_is_all_nan(helper, period, backend):
    assert_series(getattr(ta, helper)(np.arange(5.0), period), [NAN] * 5)


# ===========================================================================
# 8. NaN contract of the window and recursive kernels
#
# A window kernel is NaN on bar i exactly when i is in the warm-up or its
# window holds a NaN. A recursive kernel keeps its state across a NaN bar.
# ===========================================================================

# Eight rising bars: the close sits half a point below the high of a two-point
# range, so the money-flow multiplier ((C-L)-(H-C))/(H-L) is 0.5 on every bar.
RISING_HIGH = np.arange(10.0, 18.0)
RISING_LOW = RISING_HIGH - 2.0
RISING_CLOSE = RISING_HIGH - 0.5


def test_cmf_is_nan_while_a_nan_volume_is_in_its_window(backend):
    volume = np.full(8, 100.0)
    volume[3] = NAN
    cmf = ta.cmf(RISING_HIGH, RISING_LOW, RISING_CLOSE, volume, 3)
    assert_series(cmf, [NAN, NAN, 0.5, NAN, NAN, NAN, 0.5, 0.5])


def test_cmf_is_nan_for_a_nan_close_on_a_bar_without_range(backend):
    high, close = RISING_HIGH.copy(), RISING_CLOSE.copy()
    high[3] = RISING_LOW[3]  # no range: the multiplier would be 0 whatever the close
    close[3] = NAN
    cmf = ta.cmf(high, RISING_LOW, close, np.full(8, 100.0), 3)
    assert np.isnan(cmf[3:6]).all()
    assert cmf[6] == pytest.approx(0.5)


def test_mfi_is_nan_while_a_nan_price_is_in_its_window(backend):
    price = np.array([10.0, 11.0, 10.0, NAN, 12.0, 11.0, 13.0])  # high = low = close
    # Flow per bar: +11, -10, unknown, none (no earlier price), -11, +13.
    # Two-bar windows:
    #   bar 1: +11                 -> 100
    #   bar 2: +11, -10            -> 100 - 100/(1 + 11/10)
    #   bars 3 and 4 hold the NaN  -> NaN
    #   bar 5: none, -11           -> 0
    #   bar 6: -11, +13            -> 100 - 100/(1 + 13/11)
    expected = [NAN, 100.0, 100.0 - 100.0 / 2.1, NAN, NAN, 0.0, 100.0 - 100.0 / (1 + 13 / 11)]
    assert_series(ta.mfi(price, price, price, np.ones(7), 2), expected)


def test_mfi_recovers_after_a_nan_volume(backend):
    price = np.arange(10.0, 17.0)  # only rises: 100 wherever the window is known
    volume = np.array([1.0, 1.0, NAN, 1.0, 1.0, 1.0, 1.0])
    expected = [NAN, 100.0, NAN, NAN, 100.0, 100.0, 100.0]
    assert_series(ta.mfi(price, price, price, volume, 2), expected)


def test_cmo_is_nan_while_a_nan_is_among_the_bars_of_its_window(backend):
    data = np.array([10.0, 11.0, 12.0, NAN, 14.0, 15.0, 16.0, 17.0, 18.0])
    # Three changes span four bars; the first window clear of bar 3 ends at bar 7.
    assert_series(ta.cmo(data, 3), [NAN] * 7 + [100.0, 100.0])


def test_fisher_carries_its_state_across_a_nan_bar(backend):
    price = np.array([1.0, 2.0, NAN, 2.0, 1.0, 2.0])  # high = low, so hl2 is the price
    # bar 1: window (1, 2), position 1 -> value 0.33
    value_1 = 0.66 * 0.5
    fisher_1 = 0.5 * math.log((1 + value_1) / (1 - value_1))
    # bars 2 and 3 hold the NaN. bar 4: window (2, 1), position 0
    value_4 = 0.66 * -0.5 + 0.67 * value_1
    fisher_4 = 0.5 * math.log((1 + value_4) / (1 - value_4)) + 0.5 * fisher_1
    # bar 5: window (1, 2), position 1
    value_5 = 0.66 * 0.5 + 0.67 * value_4
    fisher_5 = 0.5 * math.log((1 + value_5) / (1 - value_5)) + 0.5 * fisher_4
    fisher, trigger = ta.fisher(price, price, 2)
    assert_series(fisher, [NAN, fisher_1, NAN, NAN, fisher_4, fisher_5])
    # The trigger is the Fisher value before the bar: 0 at the start.
    assert_series(trigger, [NAN, 0.0, NAN, NAN, fisher_1, fisher_4])


def test_fisher_after_a_nan_warmup_equals_the_series_without_it(backend, market):
    _, high, low, _, _ = market(60, seed=6)
    padded_high = np.concatenate([[NAN] * 3, high])
    padded_low = np.concatenate([[NAN] * 3, low])
    for padded, clean in zip(
        ta.fisher(padded_high, padded_low, 9), ta.fisher(high, low, 9), strict=True
    ):
        assert_series(padded, np.concatenate([[NAN] * 3, clean]))


def test_frama_is_nan_only_while_a_nan_bar_is_in_its_windows(backend):
    period, bars, hole = 4, 20, 10
    low = np.arange(float(bars))
    high = low + 1.0
    high[hole] = NAN
    midpoint = low + 0.5
    # On a straight line the filter is the midpoint, so FRAMA is the midpoint for
    # the first period + 1 bars and its 5-bar mean (two bars back) after that.
    expected = midpoint.copy()
    expected[period + 1 :] = midpoint[period + 1 :] - 2.0
    # The filter is unknown for the 4 bars whose ranges hold the NaN bar, and
    # its 5-bar mean for 4 bars more.
    expected[hole : hole + period + 4] = NAN
    assert_series(ta.frama(high, low, period), expected)


def test_frama_after_a_nan_warmup_equals_the_series_without_it(backend, market):
    _, high, low, _, _ = market(80, seed=6)
    padded = ta.frama(np.concatenate([[NAN] * 3, high]), np.concatenate([[NAN] * 3, low]), 8)
    assert_series(padded, np.concatenate([[NAN] * 3, ta.frama(high, low, 8)]))


# ===========================================================================
# 9. The indicators that run without the extension, by hand
# ===========================================================================

FALLBACK_CALLS = {
    "fisher": lambda o, h, lo, c, v: ta.fisher(h, lo),
    "cmf": lambda o, h, lo, c, v: ta.cmf(h, lo, c, v),
    "mfi": lambda o, h, lo, c, v: ta.mfi(h, lo, c, v),
    "psar": lambda o, h, lo, c, v: ta.psar(h, lo),
    "cmo": lambda o, h, lo, c, v: ta.cmo(c),
    "frama": lambda o, h, lo, c, v: ta.frama(h, lo),
}


@pytest.mark.parametrize("name", sorted(FALLBACK_CALLS))
def test_indicator_runs_without_the_extension(name, numpy_only, market):
    bars = 120
    result = FALLBACK_CALLS[name](*market(bars, seed=2))
    first = result[0] if isinstance(result, tuple) else result
    assert np.asarray(first).shape == (bars,)


def test_cmf_is_one_when_every_bar_closes_at_its_high(backend):
    period, bars = 3, 6
    high = np.arange(11.0, 11.0 + bars)
    volume = np.arange(100.0, 100.0 + bars)
    cmf = ta.cmf(high, high - 2.0, high, volume, period)
    assert_series(cmf, [NAN] * (period - 1) + [1.0] * (bars - period + 1))


def test_mfi_matches_a_hand_computed_window(backend):
    price = np.array([10.0, 11.0, 10.0, 12.0])  # high = low = close, volume 1
    # Flow per bar: +11, -10, +12. Two-bar windows:
    #   bar 1: +11 only                -> 100
    #   bar 2: +11 and -10             -> 100 - 100/(1 + 11/10)
    #   bar 3: -10 and +12             -> 100 - 100/(1 + 12/10)
    expected = [NAN, 100.0, 100.0 - 100.0 / 2.1, 100.0 - 100.0 / 2.2]
    assert_series(ta.mfi(price, price, price, np.ones(4), 2), expected)


def test_cmo_matches_a_hand_computed_window(backend):
    data = np.array([1.0, 2.0, 4.0, 3.0, 3.0])
    # Changes: +1, +2, -1, 0. Two-change windows:
    #   bar 2: up 3, down 0 -> 100
    #   bar 3: up 2, down 1 -> 100 * (2-1)/3
    #   bar 4: up 0, down 1 -> -100
    assert_series(ta.cmo(data, 2), [NAN, NAN, 100.0, 100.0 / 3.0, -100.0])


def test_fisher_matches_the_recursion_worked_by_hand(backend):
    """value = 0.66*(position-0.5) + 0.67*value[1]; fisher = 0.5*ln((1+v)/(1-v)) + 0.5*fisher[1]."""
    price = np.array([1.0, 2.0, 2.0, 1.0])  # high = low, so hl2 is the price
    # bar 1: window (1, 2), position 1 -> value 0.33
    value_1 = 0.66 * 0.5
    fisher_1 = 0.5 * math.log((1 + value_1) / (1 - value_1))
    # bar 2: window (2, 2) has no range -> fisher repeats
    # bar 3: window (2, 1), position 0 -> value -0.33 + 0.67 * 0.33
    value_3 = 0.66 * -0.5 + 0.67 * value_1
    fisher_3 = 0.5 * math.log((1 + value_3) / (1 - value_3)) + 0.5 * fisher_1
    fisher, trigger = ta.fisher(price, price, 2)
    assert_series(fisher, [NAN, fisher_1, fisher_1, fisher_3])
    assert_series(trigger, [NAN, 0.0, fisher_1, fisher_1])


def test_frama_of_a_straight_line_is_the_five_bar_mean_of_the_midpoint(backend):
    """On a line the fractal dimension is 1, so the weight is 1 and the filter is the price."""
    period, bars = 4, 12
    low = np.arange(float(bars))
    high = low + 1.0
    midpoint = low + 0.5
    # First period + 1 bars: the midpoint. After that: the mean of the last
    # five midpoints, which on a line is the midpoint two bars back.
    expected = midpoint.copy()
    expected[period + 1 :] = midpoint[period + 1 :] - 2.0
    assert_series(ta.frama(high, low, period), expected)


@pytest.mark.parametrize("bars", [13, 14, 15])
def test_stochastic_fallback_returns_one_value_per_bar_on_short_input(bars, numpy_only, market):
    """13 bars is below the %K period of 14, 14 and 15 below the %D warm-up."""
    _, high, low, close, _ = market(bars, seed=6)
    k, d = _backend.stochastic(high, low, close, 14, 3, 3)
    assert k.shape == d.shape == (bars,)


def test_stochastic_fallback_equals_the_rust_kernel_on_complete_bars(market):
    if not _backend.HAVE_RUST:
        pytest.skip("the _oaindicators extension is not installed")
    _, high, low, close, _ = market(300, seed=6)
    compiled = _backend.stochastic(high, low, close, 14, 3, 3)
    _backend.HAVE_RUST = False
    try:
        fallback = _backend.stochastic(high, low, close, 14, 3, 3)
    finally:
        _backend.HAVE_RUST = True
    for from_rust, from_numpy in zip(compiled, fallback, strict=True):
        assert_series(from_numpy, from_rust)


# --- one table pins every NumPy fallback to its Rust kernel

# Long enough that windows of 20 to 30 bars over five columns, each with its own
# NaN bars, still leave most bars with a value.
PARITY_BARS = 2000
# A compared output must be mostly numbers: NaN equals NaN proves nothing.
MIN_FINITE_SHARE = 0.5


@pytest.fixture(scope="module")
def parity_inputs(market):
    """Wide bars (many reversals) with NaN holes, and two boolean signals."""
    open_, high, low, close, volume = market(PARITY_BARS, seed=5, bar_range=2.5)
    holed = [with_holes(series, seed) for seed, series in enumerate((open_, high, low, close))]
    return {
        "o": holed[0],
        "h": holed[1],
        "l": holed[2],
        "c": holed[3],
        "v": with_holes(volume, 4),
        "rise": (close > open_).astype(float),
        "fall": (close < open_).astype(float),
        "starts": (np.arange(PARITY_BARS) % 75 == 0).astype(float),
    }


PARITY_CASES = {
    "fisher": lambda d: _backend.fisher((d["h"] + d["l"]) / 2.0, 9),
    "cmf": lambda d: _backend.cmf(d["h"], d["l"], d["c"], d["v"], 20),
    "mfi": lambda d: _backend.mfi(d["h"], d["l"], d["c"], d["v"], 14),
    "cmo": lambda d: _backend.cmo(d["c"], 14),
    "frama": lambda d: _backend.frama(d["h"], d["l"], 26),
    "sar": lambda d: _backend.sar(d["h"], d["l"], 0.02, 0.2),
    "rvi_vigor": lambda d: _backend.rvi_vigor(d["o"], d["h"], d["l"], d["c"], 10),
    "rvi_volatility": lambda d: _backend.rvi_volatility(d["c"], 10, 14),
    "session_vwap": lambda d: _backend.session_vwap(d["c"], d["v"], d["starts"]),
    "dema": lambda d: _backend.dema(d["c"], 9),
    "tema": lambda d: _backend.tema(d["c"], 9),
    "highest": lambda d: _backend.highest(d["c"], 20),
    "lowest": lambda d: _backend.lowest(d["c"], 20),
    "crossover": lambda d: _backend.crossover(d["c"], d["o"]),
    "crossunder": lambda d: _backend.crossunder(d["c"], d["o"]),
    "cross": lambda d: _backend.cross(d["c"], d["o"]),
    "rising": lambda d: _backend.rising(d["c"], 3),
    "falling": lambda d: _backend.falling(d["c"], 3),
    "change": lambda d: _backend.change(d["c"], 3),
    "exrem": lambda d: _backend.exrem(d["rise"], d["fall"]),
    "flip": lambda d: _backend.flip(d["rise"], d["fall"]),
    "valuewhen": lambda d: _backend.valuewhen(d["rise"], d["c"], 2),
}


def assert_compares_real_values(output, name):
    """Fail when an output is almost all NaN, or a signal that never changes."""
    output = np.asarray(output)
    if output.dtype == np.bool_:
        assert output.any() and not output.all(), f"{name}: the signal never changes"
    else:
        share = np.isfinite(output).mean()
        assert share >= MIN_FINITE_SHARE, f"{name}: only {share:.0%} of the bars are numbers"


@pytest.mark.parametrize("name", sorted(PARITY_CASES))
def test_numpy_fallback_equals_the_rust_kernel(name, parity_inputs):
    """One kernel, two implementations, the same output on wide bars with NaN holes."""
    if not _backend.HAVE_RUST:
        pytest.skip("the _oaindicators extension is not installed")
    compiled = PARITY_CASES[name](parity_inputs)
    _backend.HAVE_RUST = False
    try:
        fallback = PARITY_CASES[name](parity_inputs)
    finally:
        _backend.HAVE_RUST = True
    compiled = compiled if isinstance(compiled, tuple) else (compiled,)
    fallback = fallback if isinstance(fallback, tuple) else (fallback,)
    assert len(compiled) == len(fallback)
    for from_rust, from_numpy in zip(compiled, fallback, strict=True):
        assert np.asarray(from_rust).dtype == np.asarray(from_numpy).dtype
        assert_compares_real_values(from_numpy, name)
        assert_series(from_numpy, from_rust)


# --- the kernel entry points on the indicator classes are the shared kernels

CLASS_KERNELS = {
    "SAR": lambda d: (SAR._calculate_sar, _backend.sar, (d["h"], d["l"], 0.02, 0.2)),
    "RVI": lambda d: (RVI._calculate_rvi, _backend.rvi_vigor, (d["o"], d["h"], d["l"], d["c"], 10)),
    "VWAP": lambda d: (
        VWAP._calculate_session_vwap,
        _backend.session_vwap,
        (d["c"], d["v"], d["starts"]),
    ),
    "Fisher": lambda d: (Fisher._calculate_fisher, _backend.fisher, (d["c"], 9)),
    "CMF": lambda d: (CMF._calculate_cmf, _backend.cmf, (d["h"], d["l"], d["c"], d["v"], 20)),
    "MFI": lambda d: (MFI._calculate_mfi, _backend.mfi, (d["h"], d["l"], d["c"], d["v"], 14)),
    "CMO": lambda d: (CMO._calculate_cmo, _backend.cmo, (d["c"], 14)),
    "FRAMA": lambda d: (FRAMA._calculate_frama, _backend.frama, (d["h"], d["l"], 26)),
}


@pytest.mark.parametrize("name", sorted(CLASS_KERNELS))
def test_class_kernel_entry_point_returns_the_shared_kernel(name, parity_inputs, numpy_only):
    on_class, shared, args = CLASS_KERNELS[name](parity_inputs)
    from_class, from_backend = on_class(*args), shared(*args)
    from_class = from_class if isinstance(from_class, tuple) else (from_class,)
    from_backend = from_backend if isinstance(from_backend, tuple) else (from_backend,)
    for class_output, backend_output in zip(from_class, from_backend, strict=True):
        np.testing.assert_array_equal(class_output, backend_output)


# ===========================================================================
# 10. Signal helpers (ta.crossover and friends)
# ===========================================================================

T, F = True, False


def test_crossover_fires_on_a_move_from_at_or_below_to_above(backend):
    a = [1.0, 3.0, 2.0, 2.5, 1.0, 1.0, 5.0]
    b = [2.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0]
    # 1: up through 2. 3: up from equal. 6: up from equal. A touch (2) is no cross.
    np.testing.assert_array_equal(ta.crossover(a, b), [F, T, F, T, F, F, T])


def test_crossunder_fires_on_a_move_from_at_or_above_to_below(backend):
    a = [1.0, 3.0, 2.0, 2.5, 1.0, 1.0, 5.0]
    b = [2.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0]
    np.testing.assert_array_equal(ta.crossunder(a, b), [F, F, F, F, T, F, F])


def test_cross_fires_in_either_direction(backend):
    a = [1.0, 3.0, 2.0, 2.5, 1.0, 1.0, 5.0]
    b = [2.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0]
    np.testing.assert_array_equal(ta.cross(a, b), [F, T, F, T, T, F, T])


def test_a_nan_on_the_bar_or_the_bar_before_blocks_a_crossover(backend):
    a = [1.0, NAN, 3.0, 1.0]
    b = [2.0, 2.0, 2.0, 2.0]
    np.testing.assert_array_equal(ta.crossover(a, b), [F, F, F, F])
    np.testing.assert_array_equal(ta.crossunder(a, b), [F, F, F, T])


def test_rising_and_falling_compare_with_the_bar_length_back(backend):
    data = [1.0, 2.0, NAN, 4.0, 3.0, 3.0]
    np.testing.assert_array_equal(ta.rising(data, 1), [F, T, F, F, F, F])
    np.testing.assert_array_equal(ta.falling(data, 1), [F, F, F, F, T, F])
    np.testing.assert_array_equal(ta.rising(data, 2), [F, F, F, T, F, F])


def test_change_is_the_difference_with_the_bar_length_back(backend):
    assert_series(ta.change([1.0, 4.0, 9.0, 16.0], 2), [NAN, NAN, 8.0, 12.0])


def test_exrem_keeps_the_first_signal_until_the_other_side_fires(backend):
    primary = [T, T, F, T, T, F, T]
    secondary = [F, F, T, F, F, F, T]
    # Bar 0 kept, bar 1 dropped, bar 2 re-arms, bar 3 kept, bar 4 dropped,
    # bar 6 has both set while latched: it re-arms without a signal.
    np.testing.assert_array_equal(ta.exrem(primary, secondary), [T, F, F, T, F, F, F])


def test_exrem_keeps_a_signal_on_a_bar_where_both_sides_fire_while_armed(backend):
    primary = [T, F, T, T]
    secondary = [T, F, T, F]
    # Bar 0: armed, both set: kept, now latched. Bar 2: latched, both set: it
    # re-arms without a signal. Bar 3: armed again: kept.
    np.testing.assert_array_equal(ta.exrem(primary, secondary), [T, F, F, T])


def test_flip_holds_from_a_primary_bar_until_a_secondary_bar(backend):
    primary = [F, T, F, F, F, T, T]
    secondary = [F, F, F, T, F, F, T]
    # On: bar 1. Off: bar 3. On: bar 5. Bar 6 has both set: primary wins.
    np.testing.assert_array_equal(ta.flip(primary, secondary), [F, T, T, F, F, T, T])


def test_valuewhen_returns_the_value_at_the_nth_most_recent_true_bar(backend):
    condition = [F, T, F, T, F, T]
    values = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0]
    assert_series(ta.valuewhen(condition, values, 1), [NAN, 11.0, 11.0, 13.0, 13.0, 15.0])
    assert_series(ta.valuewhen(condition, values, 2), [NAN, NAN, NAN, 11.0, 11.0, 13.0])


HELPER_BARS = 400
# name -> arguments for the helper, given two price series and two boolean signals
HELPER_CALLS = {
    "crossover": lambda a, b, p, s: (a, b),
    "crossunder": lambda a, b, p, s: (a, b),
    "cross": lambda a, b, p, s: (a, b),
    "rising": lambda a, b, p, s: (a, 3),
    "falling": lambda a, b, p, s: (a, 3),
    "change": lambda a, b, p, s: (a, 3),
    "highest": lambda a, b, p, s: (a, 3),
    "lowest": lambda a, b, p, s: (a, 3),
    "exrem": lambda a, b, p, s: (p, s),
    "flip": lambda a, b, p, s: (p, s),
    "valuewhen": lambda a, b, p, s: (p, a, 2),
}


@pytest.fixture(scope="module")
def helper_inputs(market):
    open_, _, _, close, _ = market(HELPER_BARS, seed=8)
    rise, fall = (close > open_).astype(float), (close < open_).astype(float)
    return with_holes(close, 1), with_holes(open_, 2), rise, fall


@pytest.mark.parametrize("name", sorted(HELPER_CALLS))
def test_helper_runs_in_the_rust_core(name, helper_inputs, monkeypatch):
    if not _backend.HAVE_RUST:
        pytest.skip("the _oaindicators extension is not installed")
    called = []
    compiled = _backend._rs

    class Recorder:
        def __getattr__(self, kernel):
            called.append(kernel)
            return getattr(compiled, kernel)

    monkeypatch.setattr(_backend, "_rs", Recorder())
    getattr(ta, name)(*HELPER_CALLS[name](*helper_inputs))
    assert called == [name]


@pytest.mark.parametrize("name", sorted(HELPER_CALLS))
def test_helper_in_utils_is_the_helper_on_ta(name, helper_inputs, backend):
    """`openalgo.indicators.utils` keeps its public names; they are the same kernels."""
    args = HELPER_CALLS[name](*helper_inputs)
    np.testing.assert_array_equal(getattr(utils, name)(*args), getattr(ta, name)(*args))


def test_crossover_reads_a_longer_second_series_only_as_far_as_the_first(backend):
    np.testing.assert_array_equal(ta.crossover([1.0, 3.0], [2.0, 2.0, 2.0]), [F, T])


def test_crossover_rejects_a_second_series_shorter_than_the_first(backend):
    with pytest.raises(IndexError):
        ta.crossover([1.0, 3.0, 2.0], [2.0, 2.0])


def test_helpers_return_numpy_arrays_for_series_input():
    fast = pd.Series([1.0, 3.0, 2.0])
    slow = pd.Series([2.0, 2.0, 2.0])
    result = ta.crossover(fast, slow)
    assert isinstance(result, np.ndarray) and result.dtype == np.bool_


# ===========================================================================
# 11. Backward compatibility of the changed signatures
# ===========================================================================

REQUIRED_ARG = inspect.Parameter.empty
# Parameters of openalgo 2.0.5, in order, with their defaults.
SIGNATURES_2_0_5 = {
    "vwap": [
        ("high", REQUIRED_ARG),
        ("low", REQUIRED_ARG),
        ("close", REQUIRED_ARG),
        ("volume", REQUIRED_ARG),
        ("anchor", "Session"),
        ("source", "hlc3"),
        ("stdev_mult_1", 1.0),
        ("stdev_mult_2", 2.0),
        ("stdev_mult_3", 3.0),
        ("percent_mult_1", 0.236),
        ("percent_mult_2", 0.382),
        ("percent_mult_3", 0.618),
    ],
    "pivot_points": [("high", REQUIRED_ARG), ("low", REQUIRED_ARG), ("close", REQUIRED_ARG)],
    "psar": [
        ("high", REQUIRED_ARG),
        ("low", REQUIRED_ARG),
        ("acceleration", 0.02),
        ("maximum", 0.2),
    ],
    "rvi": [
        ("open_prices", REQUIRED_ARG),
        ("high", REQUIRED_ARG),
        ("low", REQUIRED_ARG),
        ("close", REQUIRED_ARG),
        ("period", 10),
    ],
    "dema": [("data", REQUIRED_ARG), ("period", REQUIRED_ARG)],
    "tema": [("data", REQUIRED_ARG), ("period", REQUIRED_ARG)],
    "crossover": [("series1", REQUIRED_ARG), ("series2", REQUIRED_ARG)],
    "crossunder": [("series1", REQUIRED_ARG), ("series2", REQUIRED_ARG)],
    "cross": [("series1", REQUIRED_ARG), ("series2", REQUIRED_ARG)],
    "highest": [("data", REQUIRED_ARG), ("period", REQUIRED_ARG)],
    "lowest": [("data", REQUIRED_ARG), ("period", REQUIRED_ARG)],
    "change": [("data", REQUIRED_ARG), ("length", 1)],
    "rising": [("data", REQUIRED_ARG), ("length", REQUIRED_ARG)],
    "falling": [("data", REQUIRED_ARG), ("length", REQUIRED_ARG)],
    "exrem": [("primary", REQUIRED_ARG), ("secondary", REQUIRED_ARG)],
    "flip": [("primary", REQUIRED_ARG), ("secondary", REQUIRED_ARG)],
    "valuewhen": [("expr", REQUIRED_ARG), ("array", REQUIRED_ARG), ("n", 1)],
}


@pytest.mark.parametrize("name", sorted(SIGNATURES_2_0_5))
def test_signature_still_starts_with_the_2_0_5_parameters(name):
    released = SIGNATURES_2_0_5[name]
    current = list(inspect.signature(getattr(ta, name)).parameters.values())
    assert [(p.name, p.default) for p in current[: len(released)]] == released


@pytest.mark.parametrize("name", sorted(SIGNATURES_2_0_5))
def test_every_parameter_added_since_2_0_5_is_optional(name):
    added = list(inspect.signature(getattr(ta, name)).parameters.values())
    added = added[len(SIGNATURES_2_0_5[name]) :]
    assert all(p.default is not REQUIRED_ARG for p in added)


def test_pivot_points_class_still_takes_three_positional_series():
    pivot = PivotPoints().calculate(
        np.array(PIVOT_HIGH), np.array(PIVOT_LOW), np.array(PIVOT_CLOSE)
    )[0]
    assert pivot[0] == pytest.approx(9.5)


def test_ulcerindex_is_zero_not_nan_when_rounding_makes_the_mean_of_squares_negative(backend):
    """Closes from a random walk where the rolling mean of squared drawdowns rounds to
    about -3e-16 on bars 9 to 11: every bar there is a new high, so the index is 0.
    """
    close = np.array(
        [
            101.04,
            103.5,
            104.49,
            100.58,
            103.3,
            104.64,
            103.03,
            104.77,
            105.86,
            106.75,
            106.83,
            108.47,
        ]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        ulcer = ta.ulcerindex(close, length=2, smooth_length=3, signal_length=3)
    assert np.isnan(ulcer[:3]).all()
    assert not np.isnan(ulcer[3:]).any()
    assert np.array_equal(ulcer[9:], [0.0, 0.0, 0.0])
