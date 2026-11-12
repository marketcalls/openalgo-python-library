"""The condition grammar of 14-strategy-api 4.2 and its reading into one boolean per
strategy bar (`_views`, `_clock`, `_conditions.condition_values`).

The strategy bars are six 5-minute bars of SBIN on Tuesday 2025-07-01 opening at 09:15
IST, with closes 10, 12, 11, 13, 15, 14 and opens one rupee lower, so every expected
array below is read off those numbers. Views hold the values of the whole run, as they
do before a run starts, when conditions are declared.
"""

import datetime as dt

import numpy as np
import pandas as pd
import pytest

import openalgo.backtest as bt
from openalgo.backtest._conditions import condition_values
from openalgo.backtest._indicators import LookaheadOutput
from openalgo.backtest._series import BarStore, SeriesView
from openalgo.backtest.errors import ConfigError

IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
NANOS_PER_MINUTE = 60_000_000_000
FIRST_OPEN = dt.datetime(2025, 7, 1, 9, 15, tzinfo=IST)
CLOSES = [10.0, 12.0, 11.0, 13.0, 15.0, 14.0]


def store_of(closes: list[float], minutes: int, symbol: str = "SBIN") -> BarStore:
    """Bars of `minutes` from 09:15 IST with these closes; opens one rupee lower."""
    start = int(pd.Timestamp(FIRST_OPEN).value)
    step = minutes * NANOS_PER_MINUTE
    opens = np.array([start + k * step for k in range(len(closes))], dtype=np.int64)
    close = np.array(closes)
    series = {
        "ts_open": opens,
        "ts_close": opens + step,
        "open": close - 1.0,
        "high": close + 0.5,
        "low": close - 1.5,
        "close": close,
        "volume": np.full(len(closes), 100.0),
    }
    return BarStore(symbol, f"{minutes}m", series)


STORE = store_of(CLOSES, 5)
close = SeriesView(STORE, "close")


def values(condition) -> list:
    return condition_values(condition, STORE, "entry_when").tolist()


def numbers(view) -> list:
    return view._values.tolist()


def test_and_or_and_not_combine_conditions():
    rising = close > close.shift(1)
    high = close > 12
    assert values(rising) == [False, True, False, True, True, False]
    assert values(rising & high) == [False, False, False, True, True, False]
    assert values(rising | high) == [False, True, False, True, True, True]
    assert values(~high) == [True, True, True, False, False, False]


def test_shift_reads_earlier_bars_and_fills_the_start():
    assert numbers(close.shift(2))[2:] == [10.0, 12.0, 11.0, 13.0]
    assert np.isnan(numbers(close.shift(2))[:2]).all()
    assert values((close > 12).shift(1)) == [False, False, False, False, True, True]
    assert numbers(close.shift(0)) == CLOSES
    assert np.isnan(numbers(close.shift(9))).all()


def test_rolling_extremes_include_the_current_bar():
    assert numbers(close.rolling_max(3))[2:] == [12.0, 13.0, 15.0, 15.0]
    assert numbers(close.rolling_min(3))[2:] == [10.0, 11.0, 11.0, 13.0]
    assert np.isnan(numbers(close.rolling_max(3))[:2]).all()


def test_a_breakout_over_the_previous_closes_is_a_condition():
    # Above the highest of the three closes before: bar 3 (13 over 12), bar 4 (15 over 13).
    breakout = close > close.rolling_max(3).shift(1)
    assert values(breakout) == [False, False, False, True, True, False]


def test_crossings_need_the_previous_bar_on_the_other_side():
    assert values(close.crossed_above(12.5)) == [False, False, False, True, False, False]
    assert values(close.crossed_below(11.5)) == [False, False, True, False, False, False]


def test_time_between_is_half_open_on_the_bar_close():
    # The bars close at 09:20, 09:25, 09:30, 09:35, 09:40, 09:45.
    assert values(bt.time_between("09:25", "09:40")) == [False, True, True, True, False, False]


def test_weekday_reads_the_ist_date_of_the_bar_close():
    # 2025-07-01 is a Tuesday.
    assert values(bt.weekday("Tue")) == [True] * 6
    assert values(bt.weekday("monday", 2, "Fri")) == [False] * 6
    assert values(bt.weekday(1)) == [True] * 6


def test_a_clock_condition_combines_with_a_view_and_with_another_clock():
    window = bt.time_between("09:25", "09:40")
    late = bt.time_between("09:45", "10:00")
    assert values((close > 12) & window) == [False, False, False, True, False, False]
    assert values(window & (close > 12)) == [False, False, False, True, False, False]
    assert values(window | late) == [False, True, True, True, False, True]
    assert values(~window) == [True, False, False, False, True, True]


def test_a_view_of_another_interval_counts_its_latest_closed_bar():
    # 10-minute bars close at 09:25, 09:35, 09:45 with closes 1, 3, 2. At the 5-minute
    # closes 09:20 to 09:45 the latest 10-minute close is none, 1, 1, 3, 3, 2.
    ten = store_of([1.0, 3.0, 2.0], 10)
    above = SeriesView(ten, "close") > 1.5
    assert values(above) == [False, False, False, True, True, True]


def test_an_array_in_bar_order_is_read_one_value_per_bar():
    flags = [True, False, True, False, True, False]
    assert values(np.array(flags)) == flags
    assert values(flags) == flags


def test_a_series_is_matched_by_bar_open_time():
    index = pd.DatetimeIndex([FIRST_OPEN + dt.timedelta(minutes=5 * k) for k in range(6)])
    shuffled = pd.Series([True, False, False, True, False, True], index=index)[::-1]
    assert values(shuffled) == [True, False, False, True, False, True]


def test_a_naive_series_index_is_ist():
    naive = dt.datetime(2025, 7, 1, 9, 15)
    index = pd.DatetimeIndex([naive + dt.timedelta(minutes=5 * k) for k in range(6)])
    assert values(pd.Series([True, False] * 3, index=index)) == [True, False] * 3


def test_a_series_may_cover_more_bars_than_the_run():
    index = pd.DatetimeIndex([FIRST_OPEN + dt.timedelta(minutes=5 * k) for k in range(-2, 8)])
    series = pd.Series(np.arange(10) % 2 == 0, index=index)
    assert values(series) == [True, False, True, False, True, False]


def test_condition_values_are_contiguous_booleans():
    out = condition_values(close > 12, STORE, "entry_when")
    assert out.dtype == bool and out.flags["C_CONTIGUOUS"]


# ----------------------------------------------------------------- outside the grammar


@pytest.mark.parametrize(
    ("build", "message"),
    [
        (lambda: (close > 1) & close, "combines conditions"),
        (lambda: ~close, "negates a condition"),
        (lambda: close.shift(-1), "later bar"),
        (lambda: close.rolling_max(0), "1 or more"),
        (lambda: (close > 1).rolling_min(2), "takes a series of numbers"),
        (lambda: bt.time_between("14:30", "09:30"), "start before end"),
        (lambda: bt.weekday("Funday"), "day names"),
        (lambda: bt.weekday(), "at least one day"),
    ],
    ids=[
        "and-on-a-number",
        "not-on-a-number",
        "negative-shift",
        "empty-window",
        "window-on-a-condition",
        "inverted-window",
        "unknown-day",
        "no-day",
    ],
)
def test_an_operation_outside_the_grammar_is_refused(build, message):
    with pytest.raises(ConfigError, match=message):
        build()


@pytest.mark.parametrize(
    ("condition", "message"),
    [
        (close + 1, "series of numbers"),
        (True, "outside the condition grammar"),
        ("close > 5", "outside the condition grammar"),
        (np.ones(6), "one boolean per strategy bar"),
        (np.ones(5, dtype=bool), "5 values and the run has 6 strategy bars"),
        (SeriesView(store_of(CLOSES, 5, "INFY"), "close") > 1, "its own symbol SBIN"),
    ],
    ids=["number", "python-bool", "text", "numeric-array", "short-array", "other-symbol"],
)
def test_a_condition_outside_the_grammar_is_refused(condition, message):
    with pytest.raises(ConfigError, match=message):
        condition_values(condition, STORE, "entry_when")


def test_a_look_ahead_output_is_refused():
    lagging = LookaheadOutput("ichimoku", "lagging_span", "drawn 26 bars back", ("base",))
    with pytest.raises(ConfigError, match="looks ahead"):
        condition_values(lagging, STORE, "entry_when")


def test_a_series_missing_a_strategy_bar_is_refused():
    index = pd.DatetimeIndex([FIRST_OPEN + dt.timedelta(minutes=5 * k) for k in range(5)])
    with pytest.raises(ConfigError, match=r"bar opening at 2025-07-01T09:40:00\+05:30"):
        condition_values(pd.Series([True] * 5, index=index), STORE, "entry_when")


def test_a_clock_condition_has_no_truth_value():
    with pytest.raises(TypeError, match="not and, or, not"):
        bool(bt.time_between("09:30", "14:30"))
