"""`resample_bars`: buckets on the bar lattice, in and out of the canonical layout (FR-DATA-09).

A bucket of interval N starts at the exchange's regular open plus whole multiples of N
on every date and is decided by the clock, so missing 1-minute bars and a short last
bucket are handled. Expected values are written out from the bar builder
(`bars_frame`: open counts up by one per row, high is open + 2, low open - 2, close
open + 1, volume 10).
"""

import datetime as dt

import numpy as np
import pandas as pd
import pytest
from calendar_fixtures.builders import (
    at_ist,
    bars_frame,
    bars_per_output,
    daily_frame,
    nse_session,
    opens_between,
    weekdays,
)

from openalgo.backtest._layout import BAR_COLUMNS
from openalgo.backtest.calendar import Calendar
from openalgo.backtest.errors import CalendarError, ConfigError, DataError, TimestampError
from openalgo.backtest.resample import resample_bars

MONDAY = dt.date(2025, 6, 2)
TUESDAY = dt.date(2025, 6, 3)
SATURDAY = dt.date(2025, 6, 7)
EVENING = dt.date(2024, 11, 1)  # a weekday given an 18:00 to 19:00 session in these tests
NO_DROPS = {"pre_open_bars": 0, "post_close_bars": 0, "closed_day_bars": 0, "empty_buckets": 0}


@pytest.fixture
def cal() -> Calendar:
    return Calendar(builtin=False)


@pytest.fixture
def monday_minutes() -> pd.DataFrame:
    return bars_frame(nse_session(MONDAY))


def test_output_is_in_the_canonical_layout(cal, monday_minutes):
    out = resample_bars(monday_minutes, "5m", exchange="NSE", calendar=cal)
    assert tuple(out.columns) == BAR_COLUMNS
    assert isinstance(out.index, pd.RangeIndex)
    assert [str(out[column].dtype) for column in ("ts_open", "ts_close")] == ["int64", "int64"]
    assert all(out[column].dtype == np.float64 for column in BAR_COLUMNS[2:])


def test_output_attrs_carry_the_new_interval_and_the_drop_counts(cal, monday_minutes):
    out = resample_bars(monday_minutes, "5m", exchange="NSE", calendar=cal)
    assert out.attrs == {"symbol": "TEST", "exchange": "NSE", "interval": "5m", "dropped": NO_DROPS}
    assert monday_minutes.attrs["interval"] == "1m"  # the input is not changed


def test_five_minute_buckets_start_at_the_session_open(cal, monday_minutes):
    out = resample_bars(monday_minutes, "5m", exchange="NSE", calendar=cal)
    np.testing.assert_array_equal(out["ts_open"], nse_session(MONDAY, step_minutes=5))
    np.testing.assert_array_equal(out["ts_close"], out["ts_open"] + 5 * 60 * 10**9)


def test_a_bucket_takes_first_max_min_last_and_sum(cal, monday_minutes):
    first = resample_bars(monday_minutes, "5m", exchange="NSE", calendar=cal).iloc[0]
    # Rows 0 to 4 open at 100 to 104.
    assert (first["open"], first["high"], first["low"], first["close"], first["volume"]) == (
        100.0,
        106.0,
        98.0,
        105.0,
        50.0,
    )
    assert np.isnan(first["oi"])


def test_the_last_bucket_of_a_session_is_short_and_closes_at_the_session_close(cal, monday_minutes):
    out = resample_bars(monday_minutes, "1h", exchange="NSE", calendar=cal)
    assert out["ts_open"].tolist() == [at_ist(MONDAY, hour, 15) for hour in range(9, 16)]
    assert bars_per_output(out) == [60, 60, 60, 60, 60, 60, 15]
    assert out["ts_close"].iloc[-1] == at_ist(MONDAY, 15, 30)


def test_missing_bars_do_not_move_the_buckets(cal):
    opens = [at_ist(MONDAY, 9, minute) for minute in (16, 17, 18, 19, 25, 29)]
    out = resample_bars(bars_frame(opens), "5m", exchange="NSE", calendar=cal)
    assert out["ts_open"].tolist() == [at_ist(MONDAY, 9, 15), at_ist(MONDAY, 9, 25)]
    assert bars_per_output(out) == [4, 2]


def test_pre_open_bars_are_dropped_and_counted(cal):
    opens = np.concatenate([[at_ist(MONDAY, 9, 7)], opens_between(MONDAY, (9, 15), (9, 20))])
    out = resample_bars(bars_frame(opens), "5m", exchange="NSE", calendar=cal)
    assert (out["ts_open"].tolist(), out["open"].tolist()) == ([at_ist(MONDAY, 9, 15)], [101.0])
    assert out.attrs["dropped"]["pre_open_bars"] == 1


def test_post_close_bars_are_dropped_and_never_merged(cal):
    opens = np.concatenate(
        [opens_between(MONDAY, (15, 25), (15, 30)), opens_between(MONDAY, (15, 30), (15, 45))]
    )
    out = resample_bars(bars_frame(opens), "1h", exchange="NSE", calendar=cal)
    assert (out["ts_open"].tolist(), bars_per_output(out)) == ([at_ist(MONDAY, 15, 15)], [5])
    assert out.attrs["dropped"]["post_close_bars"] == 15


def test_a_bucket_without_a_known_price_emits_no_bar_and_is_counted(cal):
    bars = bars_frame(opens_between(MONDAY, (9, 15), (9, 25)))
    bars.loc[5:, ["open", "high", "low", "close"]] = np.nan
    out = resample_bars(bars, "5m", exchange="NSE", calendar=cal)
    assert out["ts_open"].tolist() == [at_ist(MONDAY, 9, 15)]
    assert out.attrs["dropped"]["empty_buckets"] == 1


def test_a_special_session_is_bucketed_on_the_regular_lattice(cal):
    cal.add_session("NSE", EVENING, "18:00", "19:00")
    out = resample_bars(
        bars_frame(opens_between(EVENING, (18, 0), (19, 0))), "30m", exchange="NSE", calendar=cal
    )
    # 09:15 plus whole half hours: 17:45 (straddling the 18:00 open), 18:15, 18:45.
    assert out["ts_open"].tolist() == [
        at_ist(EVENING, 17, 45),
        at_ist(EVENING, 18, 15),
        at_ist(EVENING, 18, 45),
    ]
    assert out["ts_close"].tolist() == [
        at_ist(EVENING, 18, 15),
        at_ist(EVENING, 18, 45),
        at_ist(EVENING, 19, 0),
    ]
    assert bars_per_output(out) == [15, 30, 15]


def test_a_session_shorter_than_the_bucket_gives_one_bar_cut_at_the_close(cal):
    cal.add_session("NSE", EVENING, "18:00", "19:00")
    out = resample_bars(
        bars_frame(opens_between(EVENING, (18, 0), (19, 0))), "2h", exchange="NSE", calendar=cal
    )
    assert (out["ts_open"].tolist(), out["ts_close"].tolist()) == (
        [at_ist(EVENING, 17, 15)],
        [at_ist(EVENING, 19, 0)],
    )
    assert bars_per_output(out) == [60]


def test_daily_bars_span_the_session_whether_or_not_every_minute_exists(cal):
    opens = [at_ist(MONDAY, 9, 40), at_ist(MONDAY, 12, 0), at_ist(TUESDAY, 9, 15)]
    out = resample_bars(bars_frame(opens), "D", exchange="NSE", calendar=cal)
    assert out["ts_open"].tolist() == [at_ist(MONDAY, 9, 15), at_ist(TUESDAY, 9, 15)]
    assert out["ts_close"].tolist() == [at_ist(MONDAY, 15, 30), at_ist(TUESDAY, 15, 30)]
    assert bars_per_output(out) == [2, 1]


def test_weekly_bars_from_daily_bars_close_on_the_last_session_of_the_week(cal):
    cal.add_closure("NSE", "2026-04-03")  # Good Friday
    out = resample_bars(
        daily_frame(weekdays(dt.date(2026, 3, 30), dt.date(2026, 4, 2))),
        "W",
        exchange="NSE",
        calendar=cal,
    )
    assert out["ts_open"].tolist() == [at_ist(dt.date(2026, 3, 30), 9, 15)]
    assert out["ts_close"].tolist() == [at_ist(dt.date(2026, 4, 2), 15, 30)]
    assert bars_per_output(out) == [4]


def test_monthly_bars_from_daily_bars_follow_calendar_months(cal):
    out = resample_bars(
        daily_frame(weekdays(dt.date(2025, 1, 1), dt.date(2025, 2, 28))),
        "M",
        exchange="NSE",
        calendar=cal,
    )
    assert out["ts_open"].tolist() == [
        at_ist(dt.date(2025, 1, 1), 9, 15),
        at_ist(dt.date(2025, 2, 3), 9, 15),
    ]
    assert bars_per_output(out) == [23, 20]


def test_minute_bars_resample_to_weekly_through_daily_bars(cal):
    opens = np.concatenate([nse_session(MONDAY), nse_session(TUESDAY)])
    out = resample_bars(bars_frame(opens), "W", exchange="NSE", calendar=cal)
    assert out["ts_open"].tolist() == [at_ist(MONDAY, 9, 15)]
    assert out["ts_close"].tolist() == [at_ist(dt.date(2025, 6, 6), 15, 30)]
    assert (bars_per_output(out), out.attrs["interval"]) == ([750], "W")


def test_daily_to_daily_returns_the_bars_unchanged(cal):
    daily = daily_frame(weekdays(MONDAY, TUESDAY))
    out = resample_bars(daily, "D", exchange="NSE", calendar=cal)
    pd.testing.assert_frame_equal(out, daily, check_flags=False)
    assert out is not daily


def test_daily_to_daily_rejects_unsorted_bars(cal):
    daily = daily_frame([TUESDAY, MONDAY])
    with pytest.raises(DataError, match="not sorted"):
        resample_bars(daily, "D", exchange="NSE", calendar=cal)


def test_daily_to_daily_rejects_a_missing_timestamp(cal):
    daily = daily_frame([MONDAY, TUESDAY])
    daily.loc[1, "ts_open"] = np.iinfo(np.int64).min
    with pytest.raises(TimestampError, match="NaT"):
        resample_bars(daily, "D", exchange="NSE", calendar=cal)


def test_a_missing_intraday_timestamp_is_a_timestamp_error(cal, monday_minutes):
    monday_minutes.loc[0, "ts_open"] = np.iinfo(np.int64).min
    with pytest.raises(TimestampError, match="NaT"):
        resample_bars(monday_minutes, "5m", exchange="NSE", calendar=cal)


def test_the_source_interval_can_be_passed_when_attrs_do_not_have_it(cal, monday_minutes):
    monday_minutes.attrs.pop("interval")
    with pytest.raises(ConfigError, match="source_interval"):
        resample_bars(monday_minutes, "5m", exchange="NSE", calendar=cal)
    assert (
        len(resample_bars(monday_minutes, "5m", exchange="NSE", calendar=cal, source_interval="1m"))
        == 75
    )


@pytest.mark.parametrize(
    ("source", "target"), [("5m", "1m"), ("5m", "7m"), ("D", "1h"), ("W", "M")]
)
def test_a_target_that_cannot_be_built_from_the_source_is_a_config_error(cal, source, target):
    bars = bars_frame(nse_session(MONDAY, step_minutes=5), interval="5m")
    with pytest.raises(ConfigError, match="cannot resample"):
        resample_bars(bars, target, exchange="NSE", calendar=cal, source_interval=source)


def test_an_exchange_that_contradicts_the_bars_is_a_config_error(cal, monday_minutes):
    with pytest.raises(ConfigError, match="does not match"):
        resample_bars(monday_minutes, "5m", exchange="MCX", calendar=cal)


def test_an_unknown_exchange_is_a_calendar_error(cal, monday_minutes):
    with pytest.raises(CalendarError, match="unknown exchange"):
        resample_bars(monday_minutes, "5m", exchange="nse", calendar=cal)


def test_an_unknown_closed_day_policy_is_a_config_error(cal, monday_minutes):
    with pytest.raises(ConfigError, match="must be one of"):
        resample_bars(monday_minutes, "5m", exchange="NSE", calendar=cal, on_closed_day="ignore")


class MiddayCalendar:
    """A `SessionCalendar` that is not the engine's: weekdays trade 10:00 to 12:00 IST."""

    def __init__(self, closed: tuple[dt.date, ...] = ()) -> None:
        self.closed = closed

    def session(self, exchange: str, day: dt.date) -> tuple[int, int] | None:
        if day.weekday() >= 5 or day in self.closed:
            return None
        return at_ist(day, 10, 0), at_ist(day, 12, 0)


def test_any_session_calendar_can_drive_the_resampler():
    out = resample_bars(
        bars_frame(opens_between(MONDAY, (10, 0), (12, 0))),
        "45m",
        exchange="NSE",
        calendar=MiddayCalendar(),
    )
    # The NSE lattice runs from 09:15: 10:00, 10:45, 11:30; the last is cut at 12:00.
    assert out["ts_open"].tolist() == [
        at_ist(MONDAY, 10, 0),
        at_ist(MONDAY, 10, 45),
        at_ist(MONDAY, 11, 30),
    ]
    assert out["ts_close"].iloc[-1] == at_ist(MONDAY, 12, 0)
    assert bars_per_output(out) == [45, 45, 30]


def test_a_session_calendar_decides_which_days_are_closed():
    bars = bars_frame(opens_between(MONDAY, (10, 0), (12, 0)))
    with pytest.raises(CalendarError, match="2025-06-02"):
        resample_bars(bars, "30m", exchange="NSE", calendar=MiddayCalendar(closed=(MONDAY,)))


def test_period_bounds_come_from_a_session_calendar():
    daily = daily_frame([MONDAY, TUESDAY], open_at=(10, 0))
    out = resample_bars(
        daily, "W", exchange="NSE", calendar=MiddayCalendar(closed=(dt.date(2025, 6, 6),))
    )
    assert out["ts_open"].tolist() == [at_ist(MONDAY, 10, 0)]
    assert out["ts_close"].tolist() == [at_ist(dt.date(2025, 6, 5), 12, 0)]  # Thursday


def test_a_calendar_without_a_session_method_is_a_config_error(monday_minutes):
    with pytest.raises(ConfigError, match="calendar must provide session"):
        resample_bars(monday_minutes, "5m", exchange="NSE", calendar=object())


def test_bars_missing_a_canonical_column_are_a_data_error(cal, monday_minutes):
    with pytest.raises(DataError, match="missing the columns \\['oi'\\]"):
        resample_bars(monday_minutes.drop(columns="oi"), "5m", exchange="NSE", calendar=cal)


def test_unsorted_bars_are_a_data_error(cal, monday_minutes):
    reversed_bars = monday_minutes.iloc[::-1].reset_index(drop=True)
    reversed_bars.attrs = dict(monday_minutes.attrs)
    with pytest.raises(DataError, match="not sorted"):
        resample_bars(reversed_bars, "5m", exchange="NSE", calendar=cal)


def test_bars_on_a_closed_day_are_a_calendar_error_that_names_the_fix(cal):
    with pytest.raises(CalendarError) as raised:
        resample_bars(bars_frame(nse_session(SATURDAY)), "5m", exchange="NSE", calendar=cal)
    assert "2025-06-07" in str(raised.value) and "add_session" in str(raised.value)


def test_bars_on_a_closed_day_can_be_dropped_and_counted(cal):
    opens = np.concatenate([nse_session(dt.date(2025, 6, 6)), nse_session(SATURDAY)])
    out = resample_bars(bars_frame(opens), "5m", exchange="NSE", calendar=cal, on_closed_day="drop")
    assert (len(out), out.attrs["dropped"]["closed_day_bars"]) == (75, 375)


def test_the_default_calendar_is_used_when_none_is_given():
    # 2026-03-03 is Holi in the built-in data (NSE/CMTR/71775): NSE is closed.
    with pytest.raises(CalendarError, match="2026-03-03"):
        resample_bars(bars_frame(nse_session(dt.date(2026, 3, 3))), "5m", exchange="NSE")


def test_empty_bars_resample_to_empty_bars(cal):
    out = resample_bars(bars_frame([]), "5m", exchange="NSE", calendar=cal)
    assert (len(out), tuple(out.columns), out.attrs["dropped"]) == (0, BAR_COLUMNS, NO_DROPS)
