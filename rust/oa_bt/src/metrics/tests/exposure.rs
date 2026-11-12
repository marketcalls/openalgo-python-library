//! Exposure: the share of bars that overlap a held position.

use super::hours;
use crate::metrics::{exposure, MetricsError};
use crate::model::TsNanos;

fn marks(count: i64) -> Vec<TsNanos> {
    (0..count).map(hours).collect()
}

#[test]
fn a_bar_counts_when_the_position_overlaps_it() {
    // Bars (0,1], (1,2], (2,3], (3,4]; held over [1, 3): the middle two bars.
    let result = exposure(&marks(5), &[hours(1)], &[hours(3)]).unwrap();
    assert_eq!(result, 0.5);
}

#[test]
fn overlapping_trades_count_each_bar_once() {
    // Held over [1, 3) and [2, 4): bars (1,2], (2,3], (3,4] of five.
    let result = exposure(&marks(6), &[hours(1), hours(2)], &[hours(3), hours(4)]).unwrap();
    assert_eq!(result, 3.0 / 5.0);
}

#[test]
fn an_open_trade_is_held_to_the_end() {
    let result = exposure(&marks(5), &[hours(2)], &[TsNanos::MAX]).unwrap();
    assert_eq!(result, 0.5);
}

#[test]
fn a_trade_inside_one_bar_counts_that_bar() {
    let entry = hours(1) + hours(1) / 4;
    let exit = hours(1) + hours(1) / 2;
    assert_eq!(exposure(&marks(5), &[entry], &[exit]).unwrap(), 0.25);
}

#[test]
fn no_trade_and_a_zero_length_trade_hold_nothing() {
    assert_eq!(exposure(&marks(5), &[], &[]).unwrap(), 0.0);
    assert_eq!(exposure(&marks(5), &[hours(2)], &[hours(2)]).unwrap(), 0.0);
}

#[test]
fn a_single_point_has_no_bar() {
    assert!(exposure(&marks(1), &[0], &[hours(1)]).unwrap().is_nan());
}

#[test]
fn malformed_input_is_refused() {
    assert_eq!(
        exposure(&marks(3), &[hours(2)], &[hours(1)]),
        Err(MetricsError::ExitBeforeEntry { index: 0 })
    );
    assert!(matches!(
        exposure(&[hours(2), hours(1)], &[], &[]),
        Err(MetricsError::NotIncreasing { index: 1, .. })
    ));
    assert!(matches!(
        exposure(&marks(3), &[0], &[]),
        Err(MetricsError::LengthMismatch { .. })
    ));
}
