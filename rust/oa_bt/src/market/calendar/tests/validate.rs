//! Checking and validating an intraday series: what is counted, what is an error and
//! in which order.

use crate::market::calendar::testing::{calendar, d, ist, opens, MINUTE};
use crate::market::calendar::{
    check_intraday, validate_intraday, Coverage, Placement, SeriesError, ValidateOptions,
};
use crate::model::{Exchange, TsNanos};

const LENIENT: ValidateOptions = ValidateOptions {
    allow_closed_days: true,
};

#[test]
fn bars_outside_the_session_are_counted_by_side() {
    let day = d(2025, 6, 2);
    let mut ts = opens(day, (9, 10), (15, 35), MINUTE);
    ts.sort_unstable();
    let check = check_intraday(&calendar(), Exchange::Nse, &ts, MINUTE).unwrap();
    // 09:10 to 09:14 before the open, 15:30 to 15:34 after the close.
    assert_eq!((check.pre_open_bars, check.post_close_bars), (5, 5));
    assert_eq!(check.bars_in_session, 375);
    assert_eq!(
        (check.days, check.session_days, check.missing_bars),
        (1, 1, 0)
    );
}

#[test]
fn each_bar_has_a_placement_in_input_order() {
    let day = d(2025, 6, 2);
    let ts = [
        ist(day, 9, 14),
        ist(day, 9, 15),
        ist(day, 15, 30),
        ist(d(2025, 6, 7), 9, 15),
    ];
    let check = check_intraday(&calendar(), Exchange::Nse, &ts, MINUTE).unwrap();
    let expected = [
        Placement::PreOpen,
        Placement::InSession,
        Placement::PostClose,
        Placement::ClosedDay,
    ];
    assert_eq!(check.placement, expected);
}

#[test]
fn closed_days_missing_days_and_duplicates_are_counted() {
    let mut ts: Vec<TsNanos> = opens(d(2025, 6, 2), (9, 15), (9, 20), MINUTE);
    ts.push(ts[4]); // a repeated bar
    ts.extend(opens(d(2025, 6, 4), (9, 15), (15, 30), MINUTE)); // 2025-06-03 has no bar
    ts.extend(opens(d(2025, 6, 7), (9, 15), (9, 17), MINUTE)); // a Saturday
    let check = check_intraday(&calendar(), Exchange::Nse, &ts, MINUTE).unwrap();
    assert_eq!(check.duplicate_bars, 1);
    // The series runs to Saturday 2025-06-07: 06-03, 06-05 and 06-06 have no bar.
    assert_eq!(check.missing_days, 3);
    assert_eq!((check.closed_days, check.closed_day_bars), (1, 2));
    assert_eq!(check.first_closed_day, Some(d(2025, 6, 7)));
    assert_eq!(check.missing_bars, 375 - 5); // only on the days with data
}

#[test]
fn days_on_template_coverage_and_unverified_hours_are_counted() {
    let mut cal = calendar();
    cal.set_year_coverage(Exchange::Nfo, 2026, Coverage::Sourced);
    let mut ts = opens(d(2025, 6, 2), (9, 15), (9, 20), MINUTE);
    ts.extend(opens(d(2026, 8, 3), (9, 15), (9, 20), MINUTE));
    let check = check_intraday(&cal, Exchange::Nfo, &ts, MINUTE).unwrap();
    // 2025 has no holiday list; the NFO hours before 2026-01-16 are unverified.
    assert_eq!(
        (check.template_only_days, check.unverified_hours_days),
        (1, 1)
    );
}

#[test]
fn unreadable_series_are_errors() {
    let cal = calendar();
    let day = d(2025, 6, 2);
    let unsorted = [ist(day, 9, 16), ist(day, 9, 15)];
    let result = check_intraday(&cal, Exchange::Nse, &unsorted, MINUTE);
    assert_eq!(result, Err(SeriesError::NotSorted { index: 1 }));
    let result = check_intraday(&cal, Exchange::Nse, &[ist(day, 9, 15)], 0);
    assert_eq!(result, Err(SeriesError::InvalidInterval { nanos: 0 }));
    let result = check_intraday(&cal, Exchange::Nse, &[ist(day, 9, 15), i64::MAX], MINUTE);
    assert_eq!(
        result,
        Err(SeriesError::UnsupportedTimestamp {
            index: 1,
            ts: i64::MAX
        })
    );
}

#[test]
fn closed_day_bars_pass_only_when_allowed() {
    let ts = opens(d(2025, 6, 7), (9, 15), (9, 20), MINUTE);
    let cal = calendar();
    let strict = validate_intraday(&cal, Exchange::Nse, &ts, MINUTE, ValidateOptions::default());
    let SeriesError::ClosedDay {
        days,
        bars,
        coverage,
        ..
    } = strict.unwrap_err()
    else {
        panic!("expected a closed-day error")
    };
    assert_eq!((days, bars, coverage), (1, 5, Coverage::Template));
    let allowed = validate_intraday(&cal, Exchange::Nse, &ts, MINUTE, LENIENT).unwrap();
    assert_eq!(allowed.closed_day_bars, 5);
}

#[test]
fn an_off_lattice_error_comes_before_a_closed_day_error() {
    let mut ts = opens(d(2025, 6, 2), (9, 17), (9, 30), 5 * MINUTE);
    ts.extend(opens(d(2025, 6, 7), (9, 15), (9, 20), 5 * MINUTE));
    let result = validate_intraday(
        &calendar(),
        Exchange::Nse,
        &ts,
        5 * MINUTE,
        ValidateOptions::default(),
    );
    assert!(
        matches!(result, Err(SeriesError::OffLattice { bars: 3, .. })),
        "{result:?}"
    );
}

#[test]
fn error_messages_say_what_to_do() {
    let cal = calendar();
    let off = opens(d(2025, 6, 2), (9, 17), (9, 30), 5 * MINUTE);
    let message = validate_intraday(&cal, Exchange::Nse, &off, 5 * MINUTE, LENIENT)
        .unwrap_err()
        .to_string();
    assert!(
        message.contains("regular open (2025-06-02 09:15:00 IST)"),
        "{message}"
    );
    let closed = opens(d(2025, 6, 7), (9, 15), (9, 20), MINUTE);
    let message = validate_intraday(
        &cal,
        Exchange::Nse,
        &closed,
        MINUTE,
        ValidateOptions::default(),
    )
    .unwrap_err()
    .to_string();
    assert!(
        message.contains("no holiday data is loaded for that year"),
        "{message}"
    );
}
