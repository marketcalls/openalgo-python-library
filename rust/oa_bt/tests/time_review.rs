#![cfg(feature = "time")]
//! Regression tests for the review findings on calendars and resampling, one or more
//! per finding, against the public API of the crate. Each failed before its fix.
//!
//! Inputs are built with plain epoch arithmetic (`ist`), not with the crate's own
//! helpers. Expected instants that pin behaviour are literal UTC nanoseconds with the
//! IST time in a comment.

use oa_bt::data::resample::{resample_intraday, BarsIn, ClosedDayPolicy};
use oa_bt::market::calendar::{
    check_intraday, default_intraday_times, hm, union_grid, validate_intraday, BaseInterval,
    Calendar, CalendarError, Coverage, OffsetCheck, RuleClose, SeriesError, ShiftDirection,
    ValidateOptions,
};
use oa_bt::model::{days_from_civil, Day, Exchange, TsNanos};

const MINUTE: i64 = 60_000_000_000;
const SHIFT_5H30: i64 = 330 * MINUTE;
const LENIENT: ValidateOptions = ValidateOptions {
    allow_closed_days: true,
};

fn d(year: i32, month: u32, day: u32) -> Day {
    days_from_civil(year, month, day)
}

/// The UTC instant of an IST date and clock time.
fn ist(day: Day, hour: i64, minute: i64) -> TsNanos {
    (i64::from(day) * 86_400 + hour * 3_600 + minute * 60 - 19_800) * 1_000_000_000
}

fn every(day: Day, from: (i64, i64), to: (i64, i64), step: i64) -> Vec<TsNanos> {
    let (start, end) = (ist(day, from.0, from.1), ist(day, to.0, to.1));
    (0..)
        .map(|k| start + k * step)
        .take_while(|&ts| ts < end)
        .collect()
}

fn calendar() -> Calendar {
    Calendar::new().expect("the embedded tables parse")
}

/// Flat bars with one price per row, and a volume of 1.
struct Flat {
    ts: Vec<TsNanos>,
    price: Vec<f64>,
    volume: Vec<f64>,
}

impl Flat {
    fn new(ts: Vec<TsNanos>, price: Vec<f64>) -> Flat {
        let volume = vec![1.0; ts.len()];
        Flat { ts, price, volume }
    }

    fn bars(&self) -> BarsIn<'_> {
        let p = &self.price;
        BarsIn {
            ts_open: &self.ts,
            open: p,
            high: p,
            low: p,
            close: p,
            volume: &self.volume,
            oi: p,
        }
    }
}

// Finding 1: cash intraday defaults stay ahead of broker auto square-off once the
// closing auction ends continuous trading at 15:15.
#[test]
fn finding_1_cash_intraday_defaults_move_earlier_from_the_closing_auction_date() {
    let times = default_intraday_times(&calendar(), Exchange::Nse, d(2026, 9, 7)).unwrap();
    assert_eq!(times.entry_until, 1_788_772_800_000_000_000); // 2026-09-07 14:50 IST
    assert_eq!(times.square_off, 1_788_773_400_000_000_000); // 2026-09-07 15:00 IST
    assert!(times.unverified);
}

#[test]
fn finding_1_derivatives_close_at_15_45_from_2026_12_07() {
    let session = calendar().session(Exchange::Nfo, d(2026, 12, 7)).unwrap();
    assert_eq!(session.close, 1_796_638_500_000_000_000); // 2026-12-07 15:45 IST
}

// Finding 2: a date or timestamp outside the supported range never panics.
#[test]
fn finding_2_a_far_date_has_no_session() {
    assert!(calendar().session(Exchange::Nse, d(2300, 1, 1)).is_none());
}

#[test]
fn finding_2_setting_a_far_date_is_a_typed_error() {
    let result = calendar().set_closed(Exchange::Nse, d(2300, 1, 1), Coverage::Sourced);
    assert!(
        matches!(result, Err(CalendarError::UnsupportedDay(_))),
        "{result:?}"
    );
}

#[test]
fn finding_2_a_missing_timestamp_is_an_error() {
    let result = check_intraday(&calendar(), Exchange::Nse, &[i64::MIN, 0], MINUTE);
    assert!(matches!(
        result,
        Err(SeriesError::UnsupportedTimestamp { index: 0, .. })
    ));
}

// Finding 3: bars on the server lattice (regular open plus whole intervals) are valid on
// a special session, and a bucket that straddles the session open belongs to it.
#[test]
fn finding_3_server_lattice_bars_are_valid_on_a_special_session() {
    let mut cal = calendar();
    let day = d(2026, 11, 8);
    cal.set_session(
        Exchange::Nse,
        day,
        hm(14, 20),
        RuleClose::At(hm(15, 30)),
        false,
        Coverage::Sourced,
    )
    .unwrap();
    // The 15-minute lattice from 09:15: 14:15, 14:30, 14:45, 15:00, 15:15.
    let server_bars = every(day, (14, 15), (15, 30), 15 * MINUTE);
    let result = validate_intraday(&cal, Exchange::Nse, &server_bars, 15 * MINUTE, LENIENT);
    assert!(result.is_ok(), "{result:?}");
}

// Finding 4: bars after the session close are never merged into a session bucket.
#[test]
fn finding_4_post_close_bars_are_dropped_and_counted() {
    let day = d(2025, 6, 2);
    let mut ts = every(day, (14, 15), (15, 30), MINUTE);
    let in_session = ts.len();
    ts.extend(every(day, (15, 40), (16, 0), MINUTE));
    let price = (0..ts.len())
        .map(|i| if i < in_session { 100.0 } else { 999.0 })
        .collect();
    let flat = Flat::new(ts, price);
    let out = resample_intraday(
        &calendar(),
        Exchange::Nse,
        &flat.bars(),
        MINUTE,
        60 * MINUTE,
        ClosedDayPolicy::Error,
    )
    .unwrap();
    assert_eq!(out.bars.close, vec![100.0, 100.0]);
    let first_close = 1_748_857_500_000_000_000; // 2025-06-02 15:15 IST
    let last_close = 1_748_858_400_000_000_000; // 2025-06-02 15:30 IST
    assert_eq!(out.bars.ts_close, vec![first_close, last_close]);
    assert_eq!(out.dropped.post_close_bars, 20);
}

// Finding 5: the offset check neither misses a shifted series nor flags a thin one.
fn shifted_mcx(days: &[Day], from: (i64, i64)) -> Vec<TsNanos> {
    let ts = days
        .iter()
        .flat_map(|&day| every(day, from, (23, 30), MINUTE));
    ts.map(|t| t + SHIFT_5H30).collect()
}

#[test]
fn finding_5_two_shifted_mcx_days_are_an_error() {
    let ts = shifted_mcx(&[d(2025, 6, 3), d(2025, 6, 5)], (9, 0));
    let result = validate_intraday(&calendar(), Exchange::Mcx, &ts, MINUTE, LENIENT);
    assert!(
        matches!(result, Err(SeriesError::Offset { .. })),
        "{result:?}"
    );
}

#[test]
fn finding_5_four_shifted_mcx_days_without_the_first_bar_are_an_error() {
    let days = [d(2025, 6, 2), d(2025, 6, 3), d(2025, 6, 4), d(2025, 6, 5)];
    let ts = shifted_mcx(&days, (9, 1));
    let result = validate_intraday(&calendar(), Exchange::Mcx, &ts, MINUTE, LENIENT);
    assert!(
        matches!(result, Err(SeriesError::Offset { .. })),
        "{result:?}"
    );
}

#[test]
fn finding_5_one_shifted_mcx_day_is_inconclusive_not_an_error() {
    let ts = shifted_mcx(&[d(2025, 6, 3)], (9, 0));
    let check = validate_intraday(&calendar(), Exchange::Mcx, &ts, MINUTE, LENIENT).unwrap();
    assert!(
        matches!(check.offset, OffsetCheck::Inconclusive(_)),
        "{:?}",
        check.offset
    );
}

fn early_mcx(days: &[Day], from: (i64, i64)) -> Vec<TsNanos> {
    let ts = days
        .iter()
        .flat_map(|&day| every(day, from, (23, 30), MINUTE));
    ts.map(|t| t - SHIFT_5H30).collect()
}

#[test]
fn finding_5_two_early_mcx_days_are_an_error() {
    let ts = early_mcx(&[d(2025, 6, 3), d(2025, 6, 5)], (9, 0));
    let result = validate_intraday(&calendar(), Exchange::Mcx, &ts, MINUTE, LENIENT);
    let Err(SeriesError::Offset { evidence, .. }) = result else {
        panic!("expected an offset error, got {result:?}")
    };
    assert_eq!(evidence.direction, ShiftDirection::Early5h30);
}

#[test]
fn finding_5_four_early_mcx_days_without_the_first_bar_are_an_error() {
    let days = [d(2025, 6, 2), d(2025, 6, 3), d(2025, 6, 4), d(2025, 6, 5)];
    let ts = early_mcx(&days, (9, 1));
    let result = validate_intraday(&calendar(), Exchange::Mcx, &ts, MINUTE, LENIENT);
    assert!(
        matches!(result, Err(SeriesError::Offset { .. })),
        "{result:?}"
    );
}

#[test]
fn finding_5_one_early_mcx_day_is_inconclusive_not_an_error() {
    let ts = early_mcx(&[d(2025, 6, 3)], (9, 0));
    let check = validate_intraday(&calendar(), Exchange::Mcx, &ts, MINUTE, LENIENT).unwrap();
    assert!(
        matches!(check.offset, OffsetCheck::Inconclusive(_)),
        "{:?}",
        check.offset
    );
}

#[test]
fn finding_5_a_thin_evening_currency_series_is_not_an_error() {
    let ts: Vec<TsNanos> = (0..5)
        .flat_map(|k| every(d(2025, 6, 2) + k, (17, 5), (19, 30), 5 * MINUTE))
        .collect();
    let result = validate_intraday(&calendar(), Exchange::Cds, &ts, MINUTE, LENIENT);
    assert!(result.is_ok(), "{result:?}");
}

#[test]
fn finding_5_derivatives_bars_past_the_close_are_not_an_error() {
    let ts = every(d(2025, 6, 2), (15, 25), (15, 45), MINUTE);
    let result = validate_intraday(&calendar(), Exchange::Nfo, &ts, MINUTE, LENIENT);
    assert!(result.is_ok(), "{result:?}");
}

// Finding 6: the grid carries the close of every exchange's bar.
#[test]
fn finding_6_the_daily_grid_gives_each_exchange_its_own_close() {
    let day = d(2025, 6, 2);
    let exchanges = [Exchange::Mcx, Exchange::Nse];
    let grid = union_grid(&calendar(), &exchanges, day, day, BaseInterval::Daily).unwrap();
    let mcx_open = 1_748_835_000_000_000_000; // 2025-06-02 09:00 IST
    let nse_open = 1_748_835_900_000_000_000; // 2025-06-02 09:15 IST
    assert_eq!(grid.ts, vec![mcx_open, nse_open]);
    // MCX closes at 23:30 (United States daylight saving), NSE at 15:30.
    assert_eq!(grid.bar_close[0][0], 1_748_887_200_000_000_000); // 2025-06-02 23:30 IST
    assert_eq!(grid.bar_close[1][1], 1_748_858_400_000_000_000); // 2025-06-02 15:30 IST
}

// Finding 7: the validator serves the data layer.
#[test]
fn finding_7_bars_off_the_lattice_are_an_error() {
    let ts = every(d(2025, 6, 2), (9, 17), (10, 0), 5 * MINUTE);
    let result = validate_intraday(&calendar(), Exchange::Nse, &ts, 5 * MINUTE, LENIENT);
    assert!(
        matches!(result, Err(SeriesError::OffLattice { .. })),
        "{result:?}"
    );
}

#[test]
fn finding_7_bars_on_a_closed_day_are_an_error_by_default() {
    let ts = every(d(2025, 6, 7), (9, 15), (10, 0), MINUTE); // a Saturday
    let result = validate_intraday(
        &calendar(),
        Exchange::Nse,
        &ts,
        MINUTE,
        ValidateOptions::default(),
    );
    assert!(
        matches!(result, Err(SeriesError::ClosedDay { .. })),
        "{result:?}"
    );
}

#[test]
fn finding_7_a_late_first_bar_is_counted_not_an_error() {
    let ts = every(d(2025, 6, 2), (10, 0), (15, 30), MINUTE);
    let check = validate_intraday(&calendar(), Exchange::Nse, &ts, MINUTE, LENIENT).unwrap();
    assert_eq!((check.days_starting_late, check.missing_bars), (1, 45));
}

// Finding 8: sessions per year needs a window long enough to be a year's sample, and
// says how well the window is covered.
#[test]
fn finding_8_a_one_week_window_cannot_be_annualised() {
    let result = calendar().sessions_per_year(&[Exchange::Nse], d(2025, 6, 2), d(2025, 6, 6));
    assert!(
        matches!(result, Err(CalendarError::WindowTooShort { .. })),
        "{result:?}"
    );
}

#[test]
fn finding_8_a_year_without_a_holiday_list_is_named() {
    let result = calendar().sessions_per_year(&[Exchange::Nse], d(2025, 1, 1), d(2025, 12, 31));
    let result = result.unwrap();
    assert_eq!(result.coverage, Coverage::Template);
    assert_eq!(result.template_only, vec![(Exchange::Nse, 2025)]);
}

// Finding 9, smaller items.
#[test]
fn finding_9_a_bucket_with_no_known_value_emits_no_bar() {
    let ts = every(d(2025, 6, 2), (9, 15), (9, 25), MINUTE);
    let price = (0..10)
        .map(|i| if i < 5 { 100.0 } else { f64::NAN })
        .collect();
    let mut flat = Flat::new(ts, price);
    flat.volume = flat.price.clone();
    let out = resample_intraday(
        &calendar(),
        Exchange::Nse,
        &flat.bars(),
        MINUTE,
        5 * MINUTE,
        ClosedDayPolicy::Error,
    )
    .unwrap();
    assert_eq!(out.bars.ts_open, vec![1_748_835_900_000_000_000]); // 2025-06-02 09:15 IST
    assert_eq!(out.dropped.empty_buckets, 1);
}

#[test]
fn finding_9_new_mcx_hours_reach_an_evening_session_on_a_holiday() {
    let mut cal = calendar();
    let day = d(2025, 6, 2);
    cal.set_session(
        Exchange::Mcx,
        day,
        hm(17, 0),
        RuleClose::Regular,
        false,
        Coverage::Sourced,
    )
    .unwrap();
    cal.set_regular_hours(Exchange::Mcx, None, hm(9, 0), hm(23, 0))
        .unwrap();
    let session = cal.session(Exchange::Mcx, day).unwrap();
    assert_eq!(session.close, 1_748_885_400_000_000_000); // 2025-06-02 23:00 IST
}
