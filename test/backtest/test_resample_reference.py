"""The pandas reference of Historify aggregation, checked against values worked by hand.

The reference (`calendar_fixtures/historify.py`) is written from the SQL in the
OpenAlgo server, `database/historify_db.py` at commit d8cc13f24: lines 1044 to 1145
(`_get_aggregated_ohlcv`), 1148 to 1246 (`_get_daily_aggregated_ohlcv`), 1004 to 1014
(market open per exchange) and 2563 (the pre-open filter of the Parquet export). These
tests pin its behaviour with hand-computed expectations, so the comparison in
`test_resample_historify.py` rests on something that was checked independently. They
need no native module.
"""

import datetime as dt

import pandas as pd
import pytest
from calendar_fixtures import historify
from calendar_fixtures.builders import SECONDS_PER_DAY, at_ist

NS = 1_000_000_000
MONDAY = dt.date(2025, 6, 2)


def at(day: dt.date, hour: int, minute: int) -> int:
    """Epoch seconds of an IST clock time."""
    return at_ist(day, hour, minute) // NS


def rows(stamps: list[int], opens: list[float]) -> pd.DataFrame:
    """Historify rows whose high is open + 2, low open - 2, close open + 1, volume 10."""
    frame = pd.DataFrame({"timestamp": stamps, "open": opens})
    frame["high"] = frame["open"] + 2.0
    frame["low"] = frame["open"] - 2.0
    frame["close"] = frame["open"] + 1.0
    frame["volume"] = 10.0
    frame["oi"] = 0.0
    return frame


def daily_stamp(day: dt.date) -> int:
    """00:00 UTC of a date, the Historify stamp of a daily row."""
    return (day - dt.date(1970, 1, 1)).days * SECONDS_PER_DAY


def test_intraday_buckets_start_at_the_market_open_not_at_the_hour():
    stamps = [at(MONDAY, 9, 15), at(MONDAY, 9, 19), at(MONDAY, 9, 20), at(MONDAY, 10, 14)]
    out = historify.intraday(rows(stamps, [1.0, 2.0, 3.0, 4.0]), "NSE", 3600)
    assert out["timestamp"].tolist() == [at(MONDAY, 9, 15)]


def test_intraday_bucket_takes_first_max_min_last_and_sum():
    stamps = [at(MONDAY, 9, 15), at(MONDAY, 9, 16), at(MONDAY, 9, 19)]
    out = historify.intraday(rows(stamps, [10.0, 30.0, 20.0]), "NSE", 300)
    bar = out.iloc[0]
    assert (bar["open"], bar["high"], bar["low"], bar["close"]) == (10.0, 32.0, 8.0, 21.0)
    assert bar["volume"] == 30.0


def test_a_missing_minute_does_not_shift_later_buckets():
    stamps = [at(MONDAY, 9, 16), at(MONDAY, 9, 21), at(MONDAY, 9, 29)]
    out = historify.intraday(rows(stamps, [1.0, 2.0, 3.0]), "NSE", 300)
    expected = [at(MONDAY, 9, 15), at(MONDAY, 9, 20), at(MONDAY, 9, 25)]
    assert out["timestamp"].tolist() == expected


def test_pre_open_rows_floor_into_buckets_before_the_open():
    # 09:07 is 8 minutes before 09:15: FLOOR(-8 / 5) is -2, the 09:05 bucket.
    out = historify.intraday(rows([at(MONDAY, 9, 7)], [1.0]), "NSE", 300)
    assert out["timestamp"].tolist() == [at(MONDAY, 9, 5)]


def test_the_export_filter_drops_pre_open_rows():
    stamps = [at(MONDAY, 9, 7), at(MONDAY, 9, 15)]
    out = historify.intraday(rows(stamps, [1.0, 2.0]), "NSE", 300, drop_pre_open=True)
    assert out["timestamp"].tolist() == [at(MONDAY, 9, 15)]
    assert out["open"].tolist() == [2.0]


def test_mcx_buckets_start_at_0900():
    stamps = [at(MONDAY, 9, 0), at(MONDAY, 9, 59), at(MONDAY, 10, 0)]
    out = historify.intraday(rows(stamps, [1.0, 2.0, 3.0]), "MCX", 3600)
    assert out["timestamp"].tolist() == [at(MONDAY, 9, 0), at(MONDAY, 10, 0)]


def test_each_ist_date_has_its_own_buckets():
    tuesday = MONDAY + dt.timedelta(days=1)
    stamps = [at(MONDAY, 15, 29), at(tuesday, 9, 15)]
    out = historify.intraday(rows(stamps, [1.0, 2.0]), "NSE", 3600)
    assert out["timestamp"].tolist() == [at(MONDAY, 15, 15), at(tuesday, 9, 15)]


@pytest.mark.parametrize(
    ("interval", "day", "period_start"),
    [
        ("W", dt.date(2026, 3, 4), dt.date(2026, 3, 2)),  # a Wednesday: Monday of that week
        ("W", dt.date(2026, 3, 8), dt.date(2026, 3, 2)),  # a Sunday belongs to the same week
        ("W", dt.date(2026, 1, 1), dt.date(2025, 12, 29)),  # a week across a year end
        ("M", dt.date(2024, 2, 29), dt.date(2024, 2, 1)),
        ("Q", dt.date(2025, 12, 31), dt.date(2025, 10, 1)),
        ("Q", dt.date(2025, 4, 1), dt.date(2025, 4, 1)),
        ("Y", dt.date(2025, 7, 15), dt.date(2025, 1, 1)),
    ],
)
def test_period_bars_are_labelled_with_the_first_date_of_the_period(interval, day, period_start):
    out = historify.period(rows([daily_stamp(day)], [1.0]), interval)
    assert out["timestamp"].tolist() == [daily_stamp(period_start)]


def test_a_week_groups_monday_to_sunday():
    days = [dt.date(2025, 6, 6), dt.date(2025, 6, 8), dt.date(2025, 6, 9)]  # Fri, Sun, Mon
    out = historify.period(rows([daily_stamp(day) for day in days], [1.0, 2.0, 3.0]), "W")
    assert out["timestamp"].tolist() == [daily_stamp(MONDAY), daily_stamp(dt.date(2025, 6, 9))]
    assert out["volume"].tolist() == [20.0, 10.0]


def test_open_interest_is_the_last_row_value_as_the_sql_last_takes_it():
    frame = rows([at(MONDAY, 9, 15), at(MONDAY, 9, 16), at(MONDAY, 9, 17)], [1.0, 2.0, 3.0])
    frame["oi"] = [100.0, 120.0, float("nan")]
    out = historify.intraday(frame, "NFO", 300)
    assert pd.isna(out["oi"].item())


def test_open_is_the_first_row_value_even_when_missing():
    frame = rows([at(MONDAY, 9, 15), at(MONDAY, 9, 16)], [1.0, 2.0])
    frame.loc[0, "open"] = float("nan")
    out = historify.intraday(frame, "NFO", 300)
    assert pd.isna(out["open"].item())


def test_max_min_and_sum_skip_missing_values():
    frame = rows([at(MONDAY, 9, 15), at(MONDAY, 9, 16)], [1.0, 2.0])
    frame.loc[1, ["high", "low", "volume"]] = float("nan")
    bar = historify.intraday(frame, "NFO", 300).iloc[0]
    assert (bar["high"], bar["low"], bar["volume"]) == (3.0, -1.0, 10.0)


def test_canonical_bars_convert_to_historify_rows():
    bars = pd.DataFrame(
        {
            "ts_open": [at_ist(MONDAY, 9, 15)],
            "open": [1.0],
            "high": [3.0],
            "low": [0.5],
            "close": [2.0],
            "volume": [10.0],
            "oi": [5.0],
        }
    )
    assert historify.from_canonical(bars)["timestamp"].tolist() == [at(MONDAY, 9, 15)]
    assert historify.from_canonical(bars, daily=True)["timestamp"].tolist() == [daily_stamp(MONDAY)]
