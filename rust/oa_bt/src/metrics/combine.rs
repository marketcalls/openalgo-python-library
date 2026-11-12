//! Combined equity of several strategies run on fixed capital allocations.
//!
//! Each strategy trades its own allocation, so the combined account is the sum of the
//! strategies' equities. Their series are put on the union of their timestamps: a
//! strategy's equity at an instant is its last value at or before it (it has not moved
//! since its last mark), and before its first point it is that first value (its
//! allocated capital, untouched).

use crate::model::TsNanos;

use super::curve::EquityCurve;
use super::error::MetricsError;

/// The combined equity of `curves`: (union timestamps, summed equity in rupees).
///
/// Errors: `NothingToCombine` for no series; `TooFewPoints` for an empty series;
/// `LengthMismatch`, `NotIncreasing` and `NotFinite` for a malformed one.
pub fn combine_equity(curves: &[EquityCurve]) -> Result<(Vec<TsNanos>, Vec<f64>), MetricsError> {
    if curves.is_empty() {
        return Err(MetricsError::NothingToCombine);
    }
    for curve in curves {
        curve.check(1)?;
    }
    let mut ts: Vec<TsNanos> = curves.iter().flat_map(|c| c.ts.iter().copied()).collect();
    ts.sort_unstable();
    ts.dedup();
    let mut total = vec![0.0; ts.len()];
    for curve in curves {
        let mut next = 0;
        let mut value = curve.equity.first().copied().unwrap_or(0.0);
        for (slot, &instant) in total.iter_mut().zip(&ts) {
            while let Some(&mark) = curve.ts.get(next) {
                if mark > instant {
                    break;
                }
                value = curve.equity.get(next).copied().unwrap_or(value);
                next += 1;
            }
            *slot += value;
        }
    }
    Ok((ts, total))
}
