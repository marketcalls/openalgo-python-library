//! Tests of the metrics kernel, one file per topic.
//!
//! Expected values are worked out by hand in the test, or written as the published
//! formula applied to the literal inputs; none is produced by the code under test.

mod adjusted;
mod benchmark;
mod combine;
mod curve;
mod drawdown;
mod exposure;
mod moments;
mod montecarlo;
mod opening;
mod ratios;
mod rolling;
mod sessions;
mod tails;
mod trade_ratios;
mod trades;
mod ulcer;

use crate::model::{days_from_civil, ts_from_ist, TsNanos, NANOS_PER_DAY, NANOS_PER_HOUR};

/// UTC nanoseconds of an IST date and clock time.
pub(super) fn ist(year: i32, month: u32, day: u32, hour: u32, minute: u32) -> TsNanos {
    ts_from_ist(days_from_civil(year, month, day), hour * 60 + minute)
}

/// The 15:30 IST close of an NSE session in June 2025 (2 June 2025 is a Monday).
pub(super) fn june_close(day: u32) -> TsNanos {
    ist(2025, 6, day, 15, 30)
}

pub(super) fn hours(count: i64) -> TsNanos {
    count * NANOS_PER_HOUR
}

pub(super) fn days(count: i64) -> TsNanos {
    count * NANOS_PER_DAY
}

/// Assert two floats agree to within `1e-12`, absolute or relative.
#[track_caller]
pub(super) fn assert_close(actual: f64, expected: f64) {
    let tolerance = 1e-12 * expected.abs().max(1.0);
    assert!(
        (actual - expected).abs() <= tolerance,
        "expected {expected}, got {actual}"
    );
}

#[track_caller]
pub(super) fn assert_all_close(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len(), "{actual:?} vs {expected:?}");
    for (&a, &e) in actual.iter().zip(expected) {
        assert_close(a, e);
    }
}
