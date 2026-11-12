//! The tails of the session return distribution: value at risk, expected shortfall,
//! tail and outlier ratios, the Omega ratio and the gain-to-pain ratio.
//!
//! Definitions, for session returns `r_1..r_n` (fractions), quantiles `Q(p)` as
//! `moments` defines them and `TAIL_PROBABILITY = 0.05`:
//!
//! - Value at risk (historical, 95 percent): `Q(0.05)`, the session return that the
//!   run fell below on 5 percent of its sessions. A return, so a loss is negative.
//! - Expected shortfall (conditional value at risk, 95 percent): the mean of the returns
//!   at or below the value at risk, the lower CVaR of Rockafellar and Uryasev (2002),
//!   "Conditional value-at-risk for general loss distributions", Journal of Banking and
//!   Finance 26(7), on the empirical distribution. Never above the value at risk.
//! - Tail ratio: `Q(0.95) / |Q(0.05)|`, how far the right tail reaches against the left.
//! - Outlier win ratio: `Q(0.99)` over the mean of the positive returns; outlier loss
//!   ratio: `Q(0.01)` over the mean of the negative returns. How many typical gains (or
//!   losses) the extreme one is.
//! - Omega ratio at a threshold `L` (Keating and Shadwick 2002, "A Universal Performance
//!   Measure", Journal of Performance Measurement 6(3), on the empirical distribution):
//!   `sum(max(r - L, 0)) / sum(max(L - r, 0))`. The threshold is the per-session
//!   risk-free rate.
//! - Gain to pain (Schwager 2012, Hedge Fund Market Wizards, Wiley, appendix on
//!   performance statistics): the sum of the monthly returns over the magnitude of the
//!   sum of the negative ones.
//!
//! Undefined cases: with no return every figure is NaN. The outlier ratios are NaN
//! without a positive (or negative) return. A ratio with a zero denominator is infinite
//! when its numerator is positive and NaN otherwise.

use super::curve::check_finite;
use super::error::MetricsError;
use super::moments::{mean, quantile_sorted, sorted};

/// Probability in each tail for value at risk, expected shortfall and the tail ratio:
/// 95 percent confidence, the level regulators and most risk reports use.
pub const TAIL_PROBABILITY: f64 = 0.05;

/// Probability in each tail for the outlier ratios: the 1st and 99th percentiles.
pub const OUTLIER_PROBABILITY: f64 = 0.01;

/// Tail statistics of session returns (definitions in the module docs).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TailStats {
    /// A session return, a fraction (negative for a loss).
    pub value_at_risk: f64,
    /// A mean session return, a fraction.
    pub expected_shortfall: f64,
    pub tail_ratio: f64,
    pub outlier_win_ratio: f64,
    pub outlier_loss_ratio: f64,
    pub omega: f64,
}

/// `numerator / denominator` for a denominator of zero or more: infinite when it is
/// zero and the numerator positive, NaN when both are zero (the metrics convention).
pub(super) fn ratio_or_infinite(numerator: f64, denominator: f64) -> f64 {
    if denominator > 0.0 {
        numerator / denominator
    } else if numerator > 0.0 {
        f64::INFINITY
    } else {
        f64::NAN
    }
}

/// The Omega ratio of `returns` at `threshold`, a per-session return (module docs).
pub fn omega(returns: &[f64], threshold: f64) -> f64 {
    if returns.is_empty() {
        return f64::NAN;
    }
    let gains: f64 = returns.iter().map(|r| (r - threshold).max(0.0)).sum();
    let shortfalls: f64 = returns.iter().map(|r| (threshold - r).max(0.0)).sum();
    ratio_or_infinite(gains, shortfalls)
}

/// Tail statistics of session returns, with the Omega ratio at `threshold` (module
/// docs).
///
/// Errors: `NotFinite` for a return that is NaN or infinite.
pub fn tail_stats(returns: &[f64], threshold: f64) -> Result<TailStats, MetricsError> {
    check_finite(returns, "session return")?;
    let ordered = sorted(returns);
    let q = |p: f64| quantile_sorted(&ordered, p);
    let var = q(TAIL_PROBABILITY);
    let beyond: Vec<f64> = ordered.iter().copied().filter(|&r| r <= var).collect();
    let gains: Vec<f64> = ordered.iter().copied().filter(|&r| r > 0.0).collect();
    let losses: Vec<f64> = ordered.iter().copied().filter(|&r| r < 0.0).collect();
    let tail_ratio = if ordered.is_empty() {
        f64::NAN
    } else {
        ratio_or_infinite(q(1.0 - TAIL_PROBABILITY), var.abs())
    };
    Ok(TailStats {
        value_at_risk: var,
        expected_shortfall: mean(&beyond),
        tail_ratio,
        outlier_win_ratio: q(1.0 - OUTLIER_PROBABILITY) / mean(&gains),
        outlier_loss_ratio: q(OUTLIER_PROBABILITY) / mean(&losses),
        omega: omega(returns, threshold),
    })
}

/// Gain to pain of monthly returns (fractions; module docs).
///
/// Errors: `NotFinite` for a return that is NaN or infinite.
pub fn gain_to_pain(monthly_returns: &[f64]) -> Result<f64, MetricsError> {
    check_finite(monthly_returns, "monthly return")?;
    if monthly_returns.is_empty() {
        return Ok(f64::NAN);
    }
    let total: f64 = monthly_returns.iter().sum();
    let pain: f64 = monthly_returns.iter().filter(|&&r| r < 0.0).sum();
    Ok(ratio_or_infinite(total, pain.abs()))
}
