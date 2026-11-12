"""Bars built by resampling: a coarser intraday interval or D from a base interval,
and W, M, Q, Y from daily bars (FR-DATA-09, the data layer's side).

These tests run the engine's own `resample_bars` and session calendar; nothing is
replaced. Two rules are checked on every path: a bar that has not closed at the
load time is never returned, and neither is a bar built from part of its period.

Session times: NSE 09:15 to 15:30 IST on weekdays; 2024-01-01 is a Monday.
"""

import datetime as dt

import pandas as pd
import pytest
from data_fixtures.fakes import (
    FakeCalendar,
    FakeClient,
    at_ist,
    daily_rows,
    days_between,
    historify_shape,
    ist_date_of,
    ist_frame,
    real_calendar,
    server_load,
    truth_rows,
)

from openalgo.backtest.data import load_bars
from openalgo.backtest.errors import DataQualityError

CAL = FakeCalendar()
JAN_1 = dt.date(2024, 1, 1)
JAN_4 = dt.date(2024, 1, 4)
JAN_5 = dt.date(2024, 1, 5)
JAN_8 = dt.date(2024, 1, 8)
JAN_12 = dt.date(2024, 1, 12)
MAR_31 = dt.date(2024, 3, 31)
JUNE_EVENING = at_ist(dt.date(2024, 6, 3), 18)
QUARTER = daily_rows(CAL, "NSE", days_between(JAN_1, MAR_31))


def _daily_store(rows=QUARTER, **options):
    return FakeClient(db={("X", "NSE", "D"): historify_shape(rows)}, **options)


def _minute_store(first, last):
    rows = truth_rows(CAL, "NSE", days_between(first, last))
    return FakeClient(db={("X", "NSE", "1m"): historify_shape(rows)}), rows


def _load(client, interval, first, last, now_ns=JUNE_EVENING, calendar=CAL, **options):
    options.setdefault("cache", False)
    return server_load(
        client, interval, first, last, calendar=calendar, now_ns=now_ns, source="db", **options
    )


def _volume(rows, first, last):
    return sum(row["volume"] for row in rows if first <= ist_date_of(row["timestamp"]) <= last)


# --- W, M, Q, Y are built from complete daily bars ----------------------------------


def test_monthly_bar_is_the_aggregate_of_every_daily_bar_of_its_month():
    result = _load(_daily_store(), "M", JAN_1, MAR_31)
    february = [r for r in QUARTER if ist_date_of(r["timestamp"]).month == 2]
    bar = result.bars.iloc[1]
    assert (bar["ts_open"], bar["ts_close"]) == (
        at_ist(dt.date(2024, 2, 1), 9, 15), at_ist(dt.date(2024, 2, 29), 15, 30),
    )  # fmt: skip
    assert (bar["open"], bar["close"]) == (february[0]["open"], february[-1]["close"])
    assert bar["high"] == max(r["high"] for r in february)
    assert bar["low"] == min(r["low"] for r in february)
    assert bar["volume"] == sum(r["volume"] for r in february)


def test_period_bars_of_a_server_are_built_from_daily_bars_and_say_so():
    client = _daily_store(aggregates_periods=True)
    result = _load(client, "M", JAN_1, MAR_31)
    assert {interval for _, interval, _, _ in client.history_calls()} == {"D"}
    assert (result.meta.interval, result.meta.base_interval) == ("M", "D")
    assert result.bars.attrs["interval"] == "M"


def test_monthly_bars_are_returned_only_for_months_fully_inside_the_range():
    result = _load(_daily_store(), "M", dt.date(2024, 1, 15), dt.date(2024, 3, 15))
    assert result.bars["ts_open"].tolist() == [at_ist(dt.date(2024, 2, 1), 9, 15)]
    february = _volume(QUARTER, dt.date(2024, 2, 1), dt.date(2024, 2, 29))
    assert result.bars["volume"].tolist() == [february]
    assert result.meta.quality.count("incomplete_period_dropped") == 2


@pytest.mark.parametrize(
    "first, last, weeks",
    [
        (JAN_1, dt.date(2024, 1, 10), [(JAN_1, JAN_5)]),  # ends on a Wednesday
        (dt.date(2024, 1, 3), dt.date(2024, 1, 17), [(JAN_8, JAN_12)]),  # Wednesday to Wednesday
        (JAN_1, JAN_12, [(JAN_1, JAN_5), (JAN_8, JAN_12)]),
    ],
    ids=["cut at the end", "cut at both ends", "whole weeks"],
)
def test_weekly_bars_keep_only_weeks_fully_inside_the_range(first, last, weeks):
    result = _load(_daily_store(), "W", first, last)
    assert result.bars["ts_open"].tolist() == [at_ist(monday, 9, 15) for monday, _ in weeks]
    assert result.bars["ts_close"].tolist() == [at_ist(friday, 15, 30) for _, friday in weeks]


def test_period_whose_daily_bars_start_late_is_dropped_and_the_shortfall_announced():
    from_february = [r for r in QUARTER if ist_date_of(r["timestamp"]) >= dt.date(2024, 2, 5)]
    result = _load(_daily_store(from_february), "M", JAN_1, MAR_31)
    assert result.bars["ts_open"].tolist() == [at_ist(dt.date(2024, 3, 1), 9, 15)]
    assert result.meta.short_history is True
    assert result.meta.warnings[0].startswith(
        "Short history: 2024-01-01 to 2024-03-31 was requested but Historify (source='db') "
        "returned 2024-02-05 to 2024-03-29"
    )


def test_week_that_is_still_running_is_dropped_as_partial():
    wednesday_evening = at_ist(dt.date(2024, 1, 17), 18)
    result = _load(_daily_store(), "W", JAN_1, dt.date(2024, 1, 19), now_ns=wednesday_evening)
    assert result.bars["ts_close"].tolist() == [at_ist(JAN_5, 15, 30), at_ist(JAN_12, 15, 30)]


def test_top_up_cannot_freeze_a_month_built_from_half_its_days(tmp_path):
    client = _daily_store()
    cached = {"cache": True, "cache_dir": tmp_path}
    _load(client, "M", dt.date(2024, 1, 15), dt.date(2024, 3, 15), **cached)
    result = _load(client, "M", JAN_1, MAR_31, **cached)
    assert result.bars["volume"].tolist() == [
        _volume(QUARTER, JAN_1, dt.date(2024, 1, 31)),
        _volume(QUARTER, dt.date(2024, 2, 1), dt.date(2024, 2, 29)),
        _volume(QUARTER, dt.date(2024, 3, 1), MAR_31),
    ]


def test_period_request_caches_the_daily_rows_it_was_built_from(tmp_path):
    _load(_daily_store(), "M", JAN_1, MAR_31, cache=True, cache_dir=tmp_path)
    series = tmp_path / "v2" / "openalgo-db" / "exchange=NSE" / "symbol=X"
    assert [path.name for path in series.iterdir()] == ["interval=D"]


def test_weekly_bars_from_own_daily_data_follow_the_same_rules():
    days = [day for day in days_between(JAN_1, dt.date(2024, 1, 17)) if day.weekday() < 5]
    values = {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 10.0}
    daily = pd.DataFrame(values, index=pd.DatetimeIndex(days))
    result = load_bars(
        "X", "NSE", "W", JAN_1, dt.date(2024, 1, 17), data=daily, tz="Asia/Kolkata", base="D",
        calendar=CAL, now=JUNE_EVENING,
    )  # fmt: skip
    assert result.bars["volume"].tolist() == [50.0, 50.0]  # two whole weeks of five days
    assert result.meta.quality.count("incomplete_period_dropped") == 1


# --- intraday bases -----------------------------------------------------------------


def test_coarser_intraday_bars_are_aggregated_from_the_base_on_the_session_grid():
    client, rows = _minute_store(JAN_1, JAN_1)
    result = _load(client, "5m", JAN_1, JAN_1, base="1m")
    first_five = rows[:5]
    bar = result.bars.iloc[0]
    assert (bar["ts_open"], bar["ts_close"]) == (at_ist(JAN_1, 9, 15), at_ist(JAN_1, 9, 20))
    assert (bar["open"], bar["close"]) == (first_five[0]["open"], first_five[-1]["close"])
    assert bar["high"] == max(r["high"] for r in first_five)
    assert bar["volume"] == sum(r["volume"] for r in first_five)
    assert len(result.bars) == 75


def test_resampled_intraday_bar_that_has_not_closed_is_dropped():
    client, _ = _minute_store(JAN_4, JAN_5)
    result = _load(client, "15m", JAN_4, JAN_5, now_ns=at_ist(JAN_5, 10, 7, 30), base="1m")
    assert result.bars["ts_open"].iloc[-1] == at_ist(JAN_5, 9, 45)
    assert result.bars["ts_close"].iloc[-1] == at_ist(JAN_5, 10, 0)


def test_daily_bar_resampled_from_todays_unfinished_session_is_dropped():
    client, _ = _minute_store(JAN_4, JAN_5)
    result = _load(client, "D", JAN_4, JAN_5, now_ns=at_ist(JAN_5, 10, 7, 30), base="1m")
    assert result.bars["ts_close"].tolist() == [at_ist(JAN_4, 15, 30)]
    assert result.meta.normalisation.partial_dropped >= 1


def test_weekly_bars_can_be_built_from_an_intraday_base():
    client, rows = _minute_store(JAN_1, JAN_12)
    result = _load(client, "W", JAN_1, JAN_12, base="1m")
    assert result.bars["ts_close"].tolist() == [at_ist(JAN_5, 15, 30), at_ist(JAN_12, 15, 30)]
    assert result.bars["volume"].tolist() == [
        _volume(rows, JAN_1, JAN_5),
        _volume(rows, JAN_8, JAN_12),
    ]


def test_quality_of_the_base_bars_is_kept_beside_that_of_the_result():
    rows = truth_rows(CAL, "NSE", [JAN_1])
    thin = ist_frame(rows[:100] + rows[110:])  # ten 1-minute bars are missing
    result = load_bars(
        "X", "NSE", "5m", JAN_1, JAN_1, data=thin, base="1m", calendar=CAL, now=JUNE_EVENING
    )
    assert result.meta.base_quality.count("missing_bars") == 10
    assert result.meta.quality.count("missing_bars") == 2  # 10:55 and 11:00 have no base bar
    assert result.meta.quality.interval == "5m"


def test_infinite_high_of_a_base_bar_fails_the_resampled_load():
    rows = truth_rows(CAL, "NSE", [JAN_1])
    rows[7] = dict(rows[7], high=float("inf"))  # the 09:22 bar, inside the 09:20 5m bar
    client = FakeClient(db={("X", "NSE", "1m"): historify_shape(rows)})
    with pytest.raises(DataQualityError, match="1 row has an open, high, low or close"):
        _load(client, "5m", JAN_1, JAN_1, base="1m")
    result = _load(client, "5m", JAN_1, JAN_1, base="1m", strict=False)
    assert [issue.check for issue in result.meta.quality.errors] == ["non_finite_price"]
    assert result.meta.quality.errors[0].examples == ("2024-01-01 09:20 IST",)


def test_resampler_drop_counts_are_in_the_metadata():
    rows = truth_rows(CAL, "NSE", [JAN_1, dt.date(2024, 1, 2)])
    with_preopen = historify_shape(rows, CAL, "NSE", preopen_minutes=8)  # 8 bars a day
    client = FakeClient(db={("X", "NSE", "1m"): with_preopen})
    kept = {"regular_session_only": False}
    result = _load(client, "5m", JAN_1, dt.date(2024, 1, 2), base="1m", provider_options=kept)
    assert result.meta.resample_dropped == {
        "pre_open_bars": 16, "post_close_bars": 0, "closed_day_bars": 0, "empty_buckets": 0,
    }  # fmt: skip
    assert len(result.bars) == 2 * 75


def test_resampling_regular_session_bars_drops_nothing():
    client, _ = _minute_store(JAN_1, JAN_1)
    dropped = _load(client, "5m", JAN_1, JAN_1, base="1m").meta.resample_dropped
    assert set(dropped.values()) == {0}


def test_base_equal_to_the_interval_returns_the_bars_as_loaded():
    client, rows = _minute_store(JAN_1, JAN_1)
    result = _load(client, "1m", JAN_1, JAN_1, base="1m")
    assert (len(result.bars), result.meta.base_interval) == (len(rows), None)


def test_resampling_uses_the_engines_calendar_for_real_holidays():
    republic_day = dt.date(2024, 1, 26)  # a Friday
    calendar = real_calendar(holidays=[republic_day])
    january = daily_rows(calendar, "NSE", days_between(JAN_1, dt.date(2024, 1, 31)))
    result = _load(
        _daily_store(january), "W", dt.date(2024, 1, 22), republic_day, calendar=calendar
    )
    assert result.bars["ts_close"].tolist() == [at_ist(dt.date(2024, 1, 25), 15, 30)]
