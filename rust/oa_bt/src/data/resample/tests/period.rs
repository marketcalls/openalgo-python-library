//! Daily bars to weekly, monthly, quarterly and yearly bars.

use super::Columns;
use crate::data::resample::{resample_period, CalendarUnit, ClosedDayPolicy, ResampleError};
use crate::market::calendar::testing::{calendar, d, ist};
use crate::market::calendar::Coverage;
use crate::model::{Day, Exchange, TsNanos};

/// 00:00 UTC of each date, the raw Historify stamp of a daily bar.
fn utc_midnights(days: &[Day]) -> Vec<TsNanos> {
    days.iter()
        .map(|&day| i64::from(day) * 86_400_000_000_000)
        .collect()
}

#[test]
fn period_ranges_are_calendar_periods() {
    let cases = [
        (
            CalendarUnit::Week,
            d(2025, 6, 4),
            (d(2025, 6, 2), d(2025, 6, 8)),
        ),
        (
            CalendarUnit::Month,
            d(2024, 2, 10),
            (d(2024, 2, 1), d(2024, 2, 29)),
        ),
        (
            CalendarUnit::Quarter,
            d(2025, 11, 15),
            (d(2025, 10, 1), d(2025, 12, 31)),
        ),
        (
            CalendarUnit::Year,
            d(2025, 6, 4),
            (d(2025, 1, 1), d(2025, 12, 31)),
        ),
    ];
    for (unit, day, expected) in cases {
        assert_eq!(unit.range(day), expected, "{unit:?}");
    }
}

#[test]
fn interval_codes_name_the_units() {
    let codes = [
        ("W", CalendarUnit::Week),
        ("M", CalendarUnit::Month),
        ("Q", CalendarUnit::Quarter),
    ];
    for (code, unit) in codes {
        assert_eq!(CalendarUnit::from_code(code), Some(unit));
    }
    assert_eq!(CalendarUnit::from_code("Y"), Some(CalendarUnit::Year));
    assert_eq!(CalendarUnit::from_code("D"), None);
}

#[test]
fn a_week_spans_its_first_and_last_session() {
    let days = [
        d(2025, 6, 2),
        d(2025, 6, 3),
        d(2025, 6, 4),
        d(2025, 6, 5),
        d(2025, 6, 6),
    ];
    let columns = Columns::ramp(utc_midnights(&days));
    let bars = columns.bars();
    let out = resample_period(
        &calendar(),
        Exchange::Nse,
        &bars,
        CalendarUnit::Week,
        ClosedDayPolicy::Error,
    )
    .unwrap()
    .bars;
    assert_eq!(out.ts_open, vec![1_748_835_900_000_000_000]); // 2025-06-02 09:15 IST
    assert_eq!(out.ts_close, vec![1_749_204_000_000_000_000]); // 2025-06-06 15:30 IST
    assert_eq!((out.open[0], out.close[0], out.volume[0]), (0.0, 4.0, 5.0));
}

#[test]
fn a_week_opening_on_a_holiday_opens_with_its_first_session() {
    let mut cal = calendar();
    cal.set_closed(Exchange::Nse, d(2025, 6, 2), Coverage::Sourced)
        .unwrap();
    let columns = Columns::ramp(vec![ist(d(2025, 6, 3), 9, 15), ist(d(2025, 6, 4), 9, 15)]);
    let bars = columns.bars();
    let out = resample_period(
        &cal,
        Exchange::Nse,
        &bars,
        CalendarUnit::Week,
        ClosedDayPolicy::Error,
    )
    .unwrap()
    .bars;
    assert_eq!(out.ts_open, vec![1_748_922_300_000_000_000]); // 2025-06-03 09:15 IST
}

#[test]
fn two_rows_on_one_date_are_an_error() {
    let day = d(2025, 6, 2);
    let columns = Columns::ramp(vec![ist(day, 0, 0), ist(day, 9, 15)]);
    let bars = columns.bars();
    let result = resample_period(
        &calendar(),
        Exchange::Nse,
        &bars,
        CalendarUnit::Month,
        ClosedDayPolicy::Error,
    );
    assert_eq!(result, Err(ResampleError::DuplicateDay { index: 1, day }));
}

#[test]
fn a_row_on_a_closed_day_is_an_error_or_dropped() {
    let columns = Columns::ramp(utc_midnights(&[d(2025, 6, 6), d(2025, 6, 7)]));
    let bars = columns.bars();
    let cal = calendar();
    let strict = resample_period(
        &cal,
        Exchange::Nse,
        &bars,
        CalendarUnit::Week,
        ClosedDayPolicy::Error,
    );
    assert!(
        matches!(strict, Err(ResampleError::ClosedDay { .. })),
        "{strict:?}"
    );
    let dropped = resample_period(
        &cal,
        Exchange::Nse,
        &bars,
        CalendarUnit::Week,
        ClosedDayPolicy::Drop,
    )
    .unwrap();
    assert_eq!(
        (dropped.bars.volume, dropped.dropped.closed_day_bars),
        (vec![1.0], 1)
    );
}
