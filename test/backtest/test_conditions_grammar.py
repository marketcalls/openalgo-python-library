"""The condition grammar of 14-strategy-api 4.2 and its evaluation into one value per
strategy bar (`_expr`, `_expr_eval`).

The bars are six 5-minute bars of SBIN on Tuesday 2025-07-01 opening at 09:15 IST, with
closes 10, 12, 11, 13, 15, 14 and opens one rupee lower, so every expected array below
is read off those numbers.
"""

import datetime as dt

import numpy as np
import pandas as pd
import pytest

import openalgo.backtest as bt
from openalgo import ta
from openalgo.backtest._expr import as_expr, column, indicator
from openalgo.backtest._expr_eval import StrategyBars, condition_values, evaluate
from openalgo.backtest.errors import ConfigError

IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
NANOS_PER_MINUTE = 60_000_000_000
FIRST_OPEN = dt.datetime(2025, 7, 1, 9, 15, tzinfo=IST)
CLOSES = [10.0, 12.0, 11.0, 13.0, 15.0, 14.0]


def five_minute_bars(closes=CLOSES, first=FIRST_OPEN) -> StrategyBars:
    start = int(pd.Timestamp(first).value)
    opens = np.array([start + 5 * k * NANOS_PER_MINUTE for k in range(len(closes))])
    close = np.array(closes)
    columns = {
        "open": close - 1.0,
        "high": close + 0.5,
        "low": close - 1.5,
        "close": close,
        "volume": np.full(len(closes), 100.0),
    }
    return StrategyBars("SBIN", opens, opens + 5 * NANOS_PER_MINUTE, columns)


BARS = five_minute_bars()
close = column("SBIN", "close")
opening = column("SBIN", "open")


def values(expr) -> list:
    return evaluate(as_expr(expr), BARS, {}).tolist()


def test_a_column_gives_the_strategy_bars_of_its_field():
    assert values(close) == CLOSES


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        (close + 1, [11.0, 13.0, 12.0, 14.0, 16.0, 15.0]),
        (close - opening, [1.0] * 6),
        (2 * close, [20.0, 24.0, 22.0, 26.0, 30.0, 28.0]),
        (close / 2, [5.0, 6.0, 5.5, 6.5, 7.5, 7.0]),
        (30 - close, [20.0, 18.0, 19.0, 17.0, 15.0, 16.0]),
    ],
    ids=["add", "subtract", "multiply", "divide", "reflected"],
)
def test_arithmetic_is_bar_by_bar(expr, expected):
    assert values(expr) == expected


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        (close > 12, [False, False, False, True, True, True]),
        (close >= 12, [False, True, False, True, True, True]),
        (close < 12, [True, False, True, False, False, False]),
        (close <= 12, [True, True, True, False, False, False]),
        (12 < close, [False, False, False, True, True, True]),  # noqa: SIM300
    ],
    ids=[">", ">=", "<", "<=", "reflected"],
)
def test_comparisons_are_bar_by_bar(expr, expected):
    assert values(expr) == expected


def test_and_or_and_not_combine_conditions():
    rising = close > close.shift(1)
    high = close > 12
    assert values(rising) == [False, True, False, True, True, False]
    assert values(rising & high) == [False, False, False, True, True, False]
    assert values(rising | high) == [False, True, False, True, True, True]
    assert values(~high) == [True, True, True, False, False, False]


def test_shift_reads_earlier_bars_and_fills_the_start():
    assert values(close.shift(2))[2:] == [10.0, 12.0, 11.0, 13.0]
    assert np.isnan(values(close.shift(2))[:2]).all()
    assert values((close > 12).shift(1)) == [False, False, False, False, True, True]
    assert values(close.shift(0)) == CLOSES


def test_rolling_extremes_include_the_current_bar():
    assert values(close.rolling_max(3))[2:] == [12.0, 13.0, 15.0, 15.0]
    assert values(close.rolling_min(3))[2:] == [10.0, 11.0, 11.0, 13.0]
    assert np.isnan(values(close.rolling_max(3))[:2]).all()


def test_crossings_need_the_previous_bar_on_the_other_side():
    level = 12.5
    # Closes 10, 12, 11, 13, 15, 14 cross 12.5 upward at bar 3 and never downward.
    assert values(close.crossed_above(level)) == [False, False, False, True, False, False]
    assert values(close.crossed_below(11.5)) == [False, False, True, False, False, False]


def test_comparisons_with_a_warming_indicator_are_false():
    sma = indicator(ta.sma, close, 3)
    # SMA(3) of 10, 12, 11, 13, 15, 14 is NaN, NaN, 11, 12, 13, 14.
    assert values(sma)[2:] == [11.0, 12.0, 13.0, 14.0]
    assert values(close > sma) == [False, False, False, True, True, False]


def test_an_indicator_with_several_outputs_gives_one_operand_each():
    upper, middle, lower = indicator(ta.bbands, close, 3, 2.0)
    expected = ta.bbands(np.array(CLOSES), 3, 2.0)
    for operand, column_values in zip((upper, middle, lower), expected, strict=True):
        np.testing.assert_array_equal(values(operand), column_values)


def test_an_indicator_input_may_be_an_expression():
    spread = indicator(ta.sma, close - opening, 2)
    assert values(spread)[1:] == [1.0] * 5


def test_time_between_is_half_open_on_the_bar_close():
    # The bars close at 09:20, 09:25, 09:30, 09:35, 09:40, 09:45.
    window = bt.time_between("09:25", "09:40")
    assert values(window) == [False, True, True, True, False, False]


def test_weekday_reads_the_ist_date_of_the_bar_close():
    # 2025-07-01 is a Tuesday.
    assert values(bt.weekday("Tue")) == [True] * 6
    assert values(bt.weekday("monday", 2, "Fri")) == [False] * 6
    assert values(bt.weekday(1)) == [True] * 6


def test_an_array_in_bar_order_is_read_one_value_per_bar():
    flags = [True, False, True, False, True, False]
    assert values(np.array(flags)) == flags
    assert values(close > np.arange(6) + 10.5) == [False, True, False, False, True, False]


def test_a_series_is_aligned_by_bar_open_time():
    index = pd.DatetimeIndex([FIRST_OPEN + dt.timedelta(minutes=5 * k) for k in range(6)])
    shuffled = pd.Series([True, False, False, True, False, True], index=index)[::-1]
    assert values(shuffled) == [True, False, False, True, False, True]


def test_a_naive_series_index_is_ist():
    index = pd.DatetimeIndex(
        [dt.datetime(2025, 7, 1, 9, 15) + dt.timedelta(minutes=5 * k) for k in range(6)]
    )
    assert values(pd.Series([1.0, 2, 3, 4, 5, 6], index=index)) == [1.0, 2, 3, 4, 5, 6]


def test_a_series_index_may_cover_more_bars_than_the_run():
    index = pd.DatetimeIndex([FIRST_OPEN + dt.timedelta(minutes=5 * k) for k in range(-2, 8)])
    series = pd.Series(np.arange(10) % 2 == 0, index=index)
    assert values(series) == [True, False, True, False, True, False]


def test_condition_values_are_contiguous_booleans():
    out = condition_values(close > 12, BARS, "entry_when")
    assert out.dtype == bool and out.flags["C_CONTIGUOUS"]


# ----------------------------------------------------------------- outside the grammar


@pytest.mark.parametrize(
    ("build", "message"),
    [
        (lambda: close == 12, "=="),
        (lambda: close != 12, "!="),
        (lambda: (close > 1) and (close > 2), "truth value"),
        (lambda: 1 < close < 2, "truth value"),
        (lambda: (close > 1) & True, "True or False"),
        (lambda: close + "1", "str operand"),
        (lambda: close.shift(-1), "later bar"),
        (lambda: close.rolling_max(0), "1 or more"),
        (lambda: (close > 1) + 1, "needs a number"),
        (lambda: close & (close > 1), "needs a condition"),
        (lambda: close.crossed_above(close > 1), "needs a number"),
        (lambda: indicator(np.mean, close), "only from openalgo.ta"),
        (lambda: bt.time_between("14:30", "09:30"), "start before end"),
        (lambda: bt.weekday("Funday"), "day names"),
        (lambda: bt.weekday(), "at least one day"),
    ],
    ids=[
        "equality",
        "inequality",
        "and-keyword",
        "chained-comparison",
        "boolean-operand",
        "string-operand",
        "negative-shift",
        "empty-window",
        "arithmetic-on-a-condition",
        "and-on-a-number",
        "crossing-a-condition",
        "foreign-indicator",
        "inverted-window",
        "unknown-day",
        "no-day",
    ],
)
def test_anything_outside_the_grammar_is_refused(build, message):
    with pytest.raises(ConfigError, match=message):
        build()


def test_a_number_is_not_a_condition():
    with pytest.raises(ConfigError, match="needs a condition"):
        condition_values(close + 1, BARS, "entry_when")


def test_an_indicator_output_that_looks_ahead_is_refused_when_used():
    lagging = indicator(ta.ichimoku, column("SBIN", "high"), column("SBIN", "low"), close)[4]
    with pytest.raises(ConfigError, match="looks ahead"):
        condition_values(close > lagging, BARS, "entry_when")


def test_another_symbol_is_refused():
    with pytest.raises(ConfigError, match="its own symbol SBIN"):
        condition_values(column("INFY", "close") > 1, BARS, "entry_when")


def test_an_array_of_the_wrong_length_is_refused():
    with pytest.raises(ConfigError, match="5 values and the run has 6 strategy bars"):
        condition_values(np.ones(5, dtype=bool), BARS, "entry_when")


def test_a_series_missing_a_strategy_bar_is_refused():
    index = pd.DatetimeIndex([FIRST_OPEN + dt.timedelta(minutes=5 * k) for k in range(5)])
    with pytest.raises(ConfigError, match=r"bar opening at 2025-07-01T09:40:00\+05:30"):
        condition_values(pd.Series([True] * 5, index=index), BARS, "entry_when")


def test_an_array_of_text_is_refused():
    with pytest.raises(ConfigError, match="booleans or numbers"):
        as_expr(np.array(["a", "b"]))
