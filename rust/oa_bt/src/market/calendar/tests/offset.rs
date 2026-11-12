//! The 5 hour 30 minute offset check: three outcomes and the rule that decides.
//! The review cases (thin series, one shifted day, two shifted days) are in
//! `tests/time_review.rs`.

use crate::market::calendar::offset::check_offset;
use crate::market::calendar::testing::{calendar, d, opens, MINUTE};
use crate::market::calendar::{OffsetCheck, ShiftDirection};
use crate::model::{Day, Exchange, TsNanos};

const SHIFT: i64 = 330 * MINUTE;

fn nse_days(days: &[Day]) -> Vec<TsNanos> {
    days.iter()
        .flat_map(|&day| opens(day, (9, 15), (15, 30), MINUTE))
        .collect()
}

#[test]
fn a_complete_series_is_aligned() {
    let ts = nse_days(&[d(2025, 6, 2), d(2025, 6, 3)]);
    assert_eq!(
        check_offset(&calendar(), Exchange::Nse, &ts, MINUTE),
        OffsetCheck::Aligned
    );
}

#[test]
fn an_empty_series_is_aligned() {
    assert_eq!(
        check_offset(&calendar(), Exchange::Nse, &[], MINUTE),
        OffsetCheck::Aligned
    );
}

#[test]
fn two_early_days_are_shifted_early() {
    let ts: Vec<TsNanos> = nse_days(&[d(2025, 6, 2), d(2025, 6, 3)])
        .iter()
        .map(|ts| ts - SHIFT)
        .collect();
    let OffsetCheck::Shifted(evidence) = check_offset(&calendar(), Exchange::Nse, &ts, MINUTE)
    else {
        panic!("expected a shift")
    };
    assert_eq!(evidence.direction, ShiftDirection::Early5h30);
    // As given, 09:15 to 10:00 IST of each day is in session (45 bars) and the other
    // 330 bars of the 375 are before the open.
    assert_eq!((evidence.misplaced, evidence.misplaced_if_undone), (660, 0));
    assert_eq!(evidence.covered_days_if_undone, 2);
}

#[test]
fn two_late_days_are_shifted_late() {
    let ts: Vec<TsNanos> = nse_days(&[d(2025, 6, 2), d(2025, 6, 3)])
        .iter()
        .map(|ts| ts + SHIFT)
        .collect();
    let outcome = check_offset(&calendar(), Exchange::Nse, &ts, MINUTE);
    assert!(matches!(outcome, OffsetCheck::Shifted(e) if e.direction == ShiftDirection::Late5h30));
}

#[test]
fn one_late_day_is_inconclusive() {
    let ts: Vec<TsNanos> = nse_days(&[d(2025, 6, 2)])
        .iter()
        .map(|ts| ts + SHIFT)
        .collect();
    let outcome = check_offset(&calendar(), Exchange::Nse, &ts, MINUTE);
    assert!(matches!(outcome, OffsetCheck::Inconclusive(e) if e.covered_days_if_undone == 1));
}

#[test]
fn a_few_bars_after_the_close_are_inconclusive_at_most() {
    let mut ts = nse_days(&[d(2025, 6, 2), d(2025, 6, 3)]);
    ts.extend(opens(d(2025, 6, 3), (15, 30), (15, 40), MINUTE));
    let outcome = check_offset(&calendar(), Exchange::Nse, &ts, MINUTE);
    assert!(!matches!(outcome, OffsetCheck::Shifted(_)), "{outcome:?}");
}

#[test]
fn the_outcome_has_a_code_and_a_reason() {
    let cal = calendar();
    let late: Vec<TsNanos> = nse_days(&[d(2025, 6, 2)])
        .iter()
        .map(|ts| ts + SHIFT)
        .collect();
    let inconclusive = check_offset(&cal, Exchange::Nse, &late, MINUTE);
    assert_eq!(inconclusive.code(), "inconclusive");
    assert!(
        inconclusive
            .reason()
            .contains("may be 5 hours 30 minutes late"),
        "{}",
        inconclusive.reason()
    );
    assert_eq!(OffsetCheck::Aligned.code(), "aligned");
    assert_eq!(OffsetCheck::Aligned.reason(), "");
    assert_eq!(ShiftDirection::Late5h30.code(), "late_5h30");
    assert_eq!(ShiftDirection::Early5h30.code(), "early_5h30");
}
