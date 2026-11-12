//! Combined equity of strategies on fixed allocations.

use crate::metrics::{combine_equity, EquityCurve, MetricsError};

#[test]
fn the_combined_equity_is_the_sum_on_the_union_of_timestamps() {
    let a = EquityCurve {
        ts: &[1, 3],
        equity: &[100.0, 110.0],
    };
    let b = EquityCurve {
        ts: &[2, 3, 4],
        equity: &[50.0, 40.0, 45.0],
    };
    let (ts, total) = combine_equity(&[a, b], None).unwrap();
    assert_eq!(ts, vec![1, 2, 3, 4]);
    // A: 100, 100, 110, 110. B: 50 (its allocation, before its first mark), 50, 40, 45.
    assert_eq!(total, vec![150.0, 150.0, 150.0, 155.0]);
}

#[test]
fn one_series_is_returned_as_it_is() {
    let a = EquityCurve {
        ts: &[1, 2],
        equity: &[100.0, 90.0],
    };
    assert_eq!(
        combine_equity(&[a], None).unwrap(),
        (vec![1, 2], vec![100.0, 90.0])
    );
}

#[test]
fn nothing_or_an_empty_series_is_refused() {
    assert_eq!(
        combine_equity(&[], None),
        Err(MetricsError::NothingToCombine)
    );
    let empty = EquityCurve {
        ts: &[],
        equity: &[],
    };
    assert_eq!(
        combine_equity(&[empty], None),
        Err(MetricsError::TooFewPoints {
            needed: 1,
            found: 0
        })
    );
}
