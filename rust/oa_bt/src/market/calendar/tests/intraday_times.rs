//! Default intraday entry and square-off times, from `intraday_defaults.csv`.

use crate::market::calendar::testing::{calendar, d};
use crate::market::calendar::{default_intraday_times, hm, Coverage, RuleClose};
use crate::model::Exchange;

#[test]
fn clock_times_apply_to_regular_sessions() {
    const NSE_15_00: i64 = 1_748_856_600_000_000_000; // 2025-06-02 15:00 IST
    const CDS_16_30: i64 = 1_748_862_000_000_000_000; // 2025-06-02 16:30 IST
    const NSE_14_50: i64 = 1_788_772_800_000_000_000; // 2026-09-07 14:50 IST
                                                      // (exchange, date, entry_until, unverified)
    let cases = [
        (Exchange::Nse, d(2025, 6, 2), NSE_15_00, false),
        (Exchange::Nfo, d(2025, 6, 2), NSE_15_00, false),
        (Exchange::Cds, d(2025, 6, 2), CDS_16_30, false),
        (Exchange::Nse, d(2026, 9, 7), NSE_14_50, false),
    ];
    for (exchange, day, entry_until, unverified) in cases {
        let times = default_intraday_times(&calendar(), exchange, day).unwrap();
        assert_eq!(times.entry_until, entry_until, "{exchange:?} {day}");
        assert!(!times.relative_to_close);
        assert_eq!(times.unverified, unverified, "{exchange:?} {day}");
    }
}

#[test]
fn the_square_off_follows_the_entry_cut_off() {
    let times = default_intraday_times(&calendar(), Exchange::Nse, d(2025, 6, 2)).unwrap();
    assert_eq!(times.square_off, 1_748_857_200_000_000_000); // 2025-06-02 15:10 IST
}

#[test]
fn mcx_times_are_taken_before_that_days_close() {
    let times = default_intraday_times(&calendar(), Exchange::Mcx, d(2025, 6, 2)).unwrap();
    assert_eq!(times.entry_until, 1_748_885_400_000_000_000); // 2025-06-02 23:00 IST
    assert_eq!(times.square_off, 1_748_886_000_000_000_000); // 2025-06-02 23:10 IST
    assert!(times.relative_to_close);
    assert!(times.unverified, "relative to an unverified MCX close");
}

#[test]
fn an_evening_session_takes_its_times_from_its_close() {
    let mut cal = calendar();
    let day = d(2025, 6, 2);
    cal.set_session(
        Exchange::Nse,
        day,
        hm(18, 0),
        RuleClose::At(hm(19, 0)),
        false,
        Coverage::Sourced,
    )
    .unwrap();
    let times = default_intraday_times(&cal, Exchange::Nse, day).unwrap();
    // 30 and 20 minutes before 19:00.
    assert_eq!(times.entry_until, 1_748_869_200_000_000_000); // 2025-06-02 18:30 IST
    assert_eq!(times.square_off, 1_748_869_800_000_000_000); // 2025-06-02 18:40 IST
    assert!(times.relative_to_close);
}

#[test]
fn a_session_shorter_than_the_margins_has_no_entry_window() {
    let mut cal = calendar();
    let day = d(2025, 6, 2);
    cal.set_session(
        Exchange::Nse,
        day,
        hm(9, 15),
        RuleClose::At(hm(9, 25)),
        false,
        Coverage::Sourced,
    )
    .unwrap();
    let times = default_intraday_times(&cal, Exchange::Nse, day).unwrap();
    assert_eq!(times.entry_until, times.session_open);
    assert_eq!(times.square_off, times.session_open);
}

#[test]
fn a_closed_day_has_no_times() {
    assert!(default_intraday_times(&calendar(), Exchange::Nse, d(2025, 6, 7)).is_none());
}
