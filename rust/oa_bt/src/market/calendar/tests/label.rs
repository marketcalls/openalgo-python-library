//! Bars labelled by their close time, read as labelled by their open: an error, and
//! never confused with a 5 hour 30 minute offset.

use crate::market::calendar::testing::{calendar, d, ist, opens, MINUTE};
use crate::market::calendar::{
    check_intraday, hm, validate_intraday, Calendar, Coverage, OffsetCheck, RuleClose, SeriesError,
    ValidateOptions,
};
use crate::model::{Day, Exchange, TsNanos};

const LENIENT: ValidateOptions = ValidateOptions {
    allow_closed_days: true,
};
const SHIFT: i64 = 330 * MINUTE;

/// Close stamps of the bars of interval `minutes` of NSE days 09:15 to 15:30: one
/// interval after each open, and the session close for the last bar.
fn nse_close_stamps(days: &[Day], minutes: i64) -> Vec<TsNanos> {
    let stamp = |day: Day| {
        let mut closes: Vec<TsNanos> = opens(day, (9, 15), (15, 30), minutes * MINUTE)
            .iter()
            .map(|open| (open + minutes * MINUTE).min(ist(day, 15, 30)))
            .collect();
        closes.dedup();
        closes
    };
    days.iter().flat_map(|&day| stamp(day)).collect()
}

fn two_days() -> [Day; 2] {
    [d(2025, 6, 2), d(2025, 6, 3)]
}

#[test]
fn close_stamped_minute_bars_are_an_error() {
    let ts = nse_close_stamps(&two_days(), 1);
    let result = validate_intraday(&calendar(), Exchange::Nse, &ts, MINUTE, LENIENT);
    let Err(SeriesError::CloseLabelled {
        days_at_close,
        days,
        first_at_close,
        ..
    }) = result
    else {
        panic!("expected a close-label error, got {result:?}")
    };
    assert_eq!((days_at_close, days), (2, 2));
    assert_eq!(first_at_close, 1_748_858_400_000_000_000); // 2025-06-02 15:30 IST
}

#[test]
fn close_stamped_hourly_bars_are_an_error() {
    // 10:15, 11:15, ..., 15:15 and 15:30: the last bar is cut at the close.
    let ts = nse_close_stamps(&two_days(), 60);
    let result = validate_intraday(&calendar(), Exchange::Nse, &ts, 60 * MINUTE, LENIENT);
    assert!(
        matches!(result, Err(SeriesError::CloseLabelled { .. })),
        "{result:?}"
    );
}

#[test]
fn close_stamped_bars_of_a_special_session_are_an_error() {
    let mut cal: Calendar = calendar();
    let sunday = d(2026, 11, 8);
    let close = RuleClose::At(hm(15, 30));
    cal.set_session(
        Exchange::Nse,
        sunday,
        hm(14, 20),
        close,
        false,
        Coverage::Sourced,
    )
    .unwrap();
    // Open-labelled 15-minute bars of that session are 14:15 to 15:15.
    let mut ts = opens(sunday, (14, 30), (15, 45), 15 * MINUTE);
    ts.extend(opens(d(2026, 11, 9), (9, 30), (15, 45), 15 * MINUTE));
    let result = validate_intraday(&cal, Exchange::Nse, &ts, 15 * MINUTE, LENIENT);
    assert!(
        matches!(result, Err(SeriesError::CloseLabelled { .. })),
        "{result:?}"
    );
}

#[test]
fn a_close_labelled_series_is_not_an_offset() {
    let ts = nse_close_stamps(&two_days(), 1);
    let check = check_intraday(&calendar(), Exchange::Nse, &ts, MINUTE).unwrap();
    assert_eq!(check.offset, OffsetCheck::Aligned);
    assert!(check.close_labelled);
}

#[test]
fn a_shifted_series_is_never_close_labelled() {
    let cases = [two_days().to_vec(), vec![d(2025, 6, 2)]];
    for days in cases {
        let ts: Vec<TsNanos> = days
            .iter()
            .flat_map(|&day| opens(day, (9, 15), (15, 30), MINUTE))
            .map(|ts| ts + SHIFT)
            .collect();
        let check = check_intraday(&calendar(), Exchange::Nse, &ts, MINUTE).unwrap();
        assert_ne!(check.offset, OffsetCheck::Aligned, "{} days", days.len());
        assert!(!check.close_labelled, "{} days", days.len());
    }
}

#[test]
fn an_open_labelled_series_with_a_bar_at_the_close_is_not_close_labelled() {
    let ts: Vec<TsNanos> = two_days()
        .iter()
        .flat_map(|&day| opens(day, (9, 15), (15, 31), MINUTE))
        .collect();
    let check = check_intraday(&calendar(), Exchange::Nse, &ts, MINUTE).unwrap();
    assert_eq!(check.days_with_bar_at_close, 2);
    assert!(!check.close_labelled);
}

#[test]
fn an_open_labelled_series_that_starts_late_is_not_close_labelled() {
    let ts: Vec<TsNanos> = two_days()
        .iter()
        .flat_map(|&day| opens(day, (9, 30), (15, 30), MINUTE))
        .collect();
    let check = check_intraday(&calendar(), Exchange::Nse, &ts, MINUTE).unwrap();
    assert_eq!(check.days_starting_late, 2);
    assert!(!check.close_labelled);
}
