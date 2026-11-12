//! Exposure: the share of the run's bars during which a position was open.
//!
//! Definition: an equity series of points `t_0 < t_1 < ... < t_n` has `n` bars, bar
//! `i` being the interval `(t_{i-1}, t_i]` that ends at its mark. A trade holds a
//! position over `[entry, exit)`. Bar `i` is in the market when some trade overlaps
//! it: `entry < t_i` and `exit > t_{i-1}`. Exposure is the number of bars in the
//! market divided by `n`, a fraction from 0 to 1. A trade with `exit = entry` holds
//! nothing. A trade still open at the end passes `TsNanos::MAX` as its exit.
//!
//! With fewer than two points there is no bar and exposure is NaN.

use crate::model::TsNanos;

use super::curve::{check_increasing, check_length};
use super::error::MetricsError;

/// Holding intervals `[entry, exit)` merged into disjoint, increasing intervals.
fn merged_intervals(entry: &[TsNanos], exit: &[TsNanos]) -> Vec<(TsNanos, TsNanos)> {
    let mut held: Vec<(TsNanos, TsNanos)> = entry
        .iter()
        .zip(exit)
        .map(|(&a, &b)| (a, b))
        .filter(|(a, b)| b > a)
        .collect();
    held.sort_unstable();
    let mut merged: Vec<(TsNanos, TsNanos)> = Vec::with_capacity(held.len());
    for (start, end) in held {
        match merged.last_mut() {
            Some(last) if start <= last.1 => last.1 = last.1.max(end),
            _ => merged.push((start, end)),
        }
    }
    merged
}

/// Fraction of the bars of the equity series stamped `ts` that overlap a trade held
/// from `entry` to `exit` (definition in the module docs).
///
/// Errors: `LengthMismatch` when `entry` and `exit` differ in length; `NotIncreasing`
/// for equity timestamps that do not strictly increase; `ExitBeforeEntry`.
pub fn exposure(ts: &[TsNanos], entry: &[TsNanos], exit: &[TsNanos]) -> Result<f64, MetricsError> {
    check_length(exit.len(), entry.len(), "trade exit times")?;
    check_increasing(ts, "equity timestamps")?;
    if let Some(index) = entry.iter().zip(exit).position(|(a, b)| b < a) {
        return Err(MetricsError::ExitBeforeEntry { index });
    }
    if ts.len() < 2 {
        return Ok(f64::NAN);
    }
    let held = merged_intervals(entry, exit);
    let mut next = 0;
    let mut in_market = 0usize;
    for bar in ts.windows(2) {
        let (start, end) = (bar[0], bar[1]);
        while held.get(next).is_some_and(|&(_, out)| out <= start) {
            next += 1;
        }
        if held.get(next).is_some_and(|&(inside, _)| inside < end) {
            in_market += 1;
        }
    }
    Ok(in_market as f64 / (ts.len() - 1) as f64)
}
