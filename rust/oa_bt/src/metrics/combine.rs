//! Combined equity of several strategies run on fixed capital allocations.
//!
//! Each strategy trades its own allocation, so the combined account is the sum of the
//! strategies' equities. Their series are put on the union of their timestamps: a
//! strategy's equity at an instant is its last value at or before it (it has not moved
//! since its last mark). Before its first point it is its opening wealth, the capital
//! allocated to it, when the caller gives one; otherwise its first value, which is then
//! its starting equity (`curve`). The combined opening wealth is the sum of the
//! strategies' opening wealths.

use crate::model::TsNanos;

use super::curve::{check_finite, check_length, EquityCurve};
use super::error::MetricsError;

/// The combined equity of `curves`: (union timestamps, summed equity in rupees).
/// `opening_wealth` gives each curve's capital in rupees, in the order of `curves`, or
/// is `None` when every curve starts at its first point.
///
/// Errors: `NothingToCombine` for no series; `TooFewPoints` for an empty series;
/// `LengthMismatch`, `NotIncreasing` and `NotFinite` for a malformed one;
/// `LengthMismatch` for an opening wealth list that does not have one value per curve,
/// `NotFinite` for a value in it that is NaN or infinite and `InvalidOpeningWealth` for
/// one that is zero or below.
pub fn combine_equity(
    curves: &[EquityCurve],
    opening_wealth: Option<&[f64]>,
) -> Result<(Vec<TsNanos>, Vec<f64>), MetricsError> {
    if curves.is_empty() {
        return Err(MetricsError::NothingToCombine);
    }
    for curve in curves {
        curve.check(1)?;
    }
    if let Some(openings) = opening_wealth {
        check_openings(openings, curves.len())?;
    }
    let mut ts: Vec<TsNanos> = curves.iter().flat_map(|c| c.ts.iter().copied()).collect();
    ts.sort_unstable();
    ts.dedup();
    let mut total = vec![0.0; ts.len()];
    for (number, curve) in curves.iter().enumerate() {
        let opening = opening_wealth.and_then(|openings| openings.get(number).copied());
        let mut next = 0;
        let mut value = opening.or(curve.equity.first().copied()).unwrap_or(0.0);
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

fn check_openings(openings: &[f64], curves: usize) -> Result<(), MetricsError> {
    check_length(openings.len(), curves, "opening wealth")?;
    check_finite(openings, "opening wealth")?;
    match openings.iter().find(|&&value| value <= 0.0) {
        Some(&value) => Err(MetricsError::InvalidOpeningWealth { value }),
        None => Ok(()),
    }
}
