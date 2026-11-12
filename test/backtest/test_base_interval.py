"""The base interval rule and the 1-minute coverage probe (03-architecture 4.1, decision
D18), and what the Assumptions block says about it.

Bars: `strategy_samples.minute_bars` (one session, 2025-07-01) and the same bars
aggregated to 15 minutes. A 15-minute strategy bar built from the 1-minute bars of
09:15 to 09:29 opens at 500.00 and closes at 501.45 (minute 14 closes at 500 + 1.40 +
0.05).
"""

import datetime as dt

import pandas as pd
import pytest
from strategy_samples import at, daily_bars, minute_bars, run

import openalgo.backtest as bt
from openalgo.backtest._base_interval import DAILY_MEANING, meaning


class Quarterly(bt.Strategy):
    symbol, exchange, interval = "SBIN", "NSE", "15m"
    product = "CNC"


def quarter_hour_bars(frame: pd.DataFrame) -> pd.DataFrame:
    grouped = frame.resample("15min")
    return pd.DataFrame(
        {
            "open": grouped["open"].first(),
            "high": grouped["high"].max(),
            "low": grouped["low"].min(),
            "close": grouped["close"].last(),
            "volume": grouped["volume"].sum(),
        }
    ).dropna()


def first_bar(strategy=Quarterly, **settings):
    seen = []

    class Watching(strategy):
        def on_bar(self, bar):
            if not seen:
                seen.append((bar.time, bar.open, bar.close))

    result = run(Watching, **settings)
    return result, seen[0]


def test_1_minute_data_covering_the_run_makes_the_base_1_minute():
    result, _ = first_bar()
    assert result.assumptions["base"] == "1m"
    assert result.assumptions["base_rule"] == (
        "base 1m (1-minute data covers the run for every instrument): stops are tested on "
        "1-minute ranges and orders fill at 1-minute bar opens"
    )


def test_data_only_at_the_strategy_interval_makes_it_the_base():
    result, bar = first_bar(data=quarter_hour_bars(minute_bars()))
    assert result.assumptions["base"] == "15m"
    assert result.assumptions["base_rule"].startswith(
        "base 15m (1-minute data does not cover the run (SBIN: "
    )
    assert bar == (at(9, 15), 500.0, 501.45)


def test_1_minute_data_missing_a_session_makes_the_strategy_interval_the_base():
    # Two sessions asked for, 1 July and 2 July; the data has only 1 July.
    result, bar = first_bar(end=dt.date(2025, 7, 2))
    assert result.assumptions["base"] == "15m"
    assert (
        "1-minute data does not cover the run (SBIN: 1-minute bars run from 2025-07-01"
        in (result.assumptions["base_rule"])
    )
    # Own 1-minute data is resampled to the strategy interval.
    assert bar == (at(9, 15), 500.0, 501.45)


def test_an_explicit_base_is_used_as_given():
    result, _ = first_bar(base="1m")
    assert result.assumptions["base_rule"].startswith("base 1m (set by base=)")


def test_a_daily_strategy_has_a_daily_base():
    class Daily(bt.Strategy):
        symbol, exchange, interval = "SBIN", "NSE", "D"

    days = [dt.date(2025, 7, day) for day in (1, 2, 3)]
    result = run(Daily, data=daily_bars(days))
    assert result.assumptions["base_rule"] == (
        f"base D (a daily or longer strategy): {DAILY_MEANING}"
    )


def test_an_intraday_product_states_where_the_square_off_fills():
    class Intraday(Quarterly):
        product = "MIS"

    result, _ = first_bar(Intraday, data=quarter_hour_bars(minute_bars()))
    assert result.assumptions["base_rule"].endswith(
        "stops are tested on 15-minute ranges and orders fill at 15-minute bar opens; a 15:10 "
        "square-off fills at the 15:15 open"
    )


@pytest.mark.parametrize(
    ("base", "square_off", "expected_tail"),
    [
        ("1m", "15:10", "1-minute bar opens; a 15:10 square-off fills at the 15:10 open"),
        ("15m", "15:10", "15-minute bar opens; a 15:10 square-off fills at the 15:15 open"),
        ("1h", "15:10", "1-hour bar opens; a 15:10 square-off fills at the 15:15 open"),
        ("5m", None, "5-minute bar opens"),
    ],
)
def test_the_meaning_of_an_intraday_base(base, square_off, expected_tail):
    assert meaning(base, square_off, "09:15").endswith(expected_tail)


def test_the_meaning_of_a_daily_base():
    assert meaning("D", None, None) == DAILY_MEANING
