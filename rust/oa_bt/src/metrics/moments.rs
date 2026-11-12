//! Moments, the lag-1 autocorrelation and quantiles of a sample of returns.
//!
//! Definitions, for values `x_1..x_n` with mean `m` and central moments
//! `m_k = ((x_1 - m)^k + ... + (x_n - m)^k) / n`:
//!
//! - Moment ratios `g1 = m_3 / m_2^(3/2)` and `g2 = m_4 / m_2^2 - 3`.
//! - Skewness: the adjusted Fisher-Pearson coefficient `G1 = g1 sqrt(n (n - 1)) / (n - 2)`,
//!   NaN below three values (Joanes and Gill 1998, "Comparing measures of sample skewness
//!   and kurtosis", The Statistician 47(1), estimator G1).
//! - Excess kurtosis: `G2 = ((n + 1) g2 + 6) (n - 1) / ((n - 2) (n - 3))`, NaN below four
//!   values (same paper, estimator G2). A normal sample has an excess kurtosis near 0.
//! - Lag-1 autocorrelation: `sum((x_t - m) (x_{t+1} - m), t = 1..n-1) / sum((x_t - m)^2)`,
//!   the sample autocorrelation of Box and Jenkins (Time Series Analysis: Forecasting and
//!   Control, 1976), NaN below three values.
//! - Quantile at probability `p`: linear interpolation between the order statistics
//!   around the position `(n - 1) p`, counting from 0 (Hyndman and Fan 1996, "Sample
//!   quantiles in statistical packages", The American Statistician 50(4), definition 7).
//!
//! A sample whose values are all equal has no variation: its moment ratios, skewness,
//! kurtosis and autocorrelation are NaN. Equality is exact, so that rounding in the mean
//! of equal values is never read as a tiny variance.

use super::error::MetricsError;

/// Mean of `values`; NaN for none.
pub(super) fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

/// True when at least two values differ.
pub(super) fn varies(values: &[f64]) -> bool {
    values.windows(2).any(|pair| pair[0] != pair[1])
}

/// The moment ratios `(g1, g2)` (module docs); NaN below two values or without variation.
pub(super) fn moment_ratios(values: &[f64]) -> (f64, f64) {
    if values.len() < 2 || !varies(values) {
        return (f64::NAN, f64::NAN);
    }
    let centre = mean(values);
    let n = values.len() as f64;
    let (mut m2, mut m3, mut m4) = (0.0, 0.0, 0.0);
    for value in values {
        let d = value - centre;
        m2 += d * d;
        m3 += d * d * d;
        m4 += d * d * d * d;
    }
    let (m2, m3, m4) = (m2 / n, m3 / n, m4 / n);
    (m3 / m2.powf(1.5), m4 / (m2 * m2) - 3.0)
}

/// Sample skewness `G1` (module docs).
pub fn skewness(values: &[f64]) -> f64 {
    let n = values.len() as f64;
    if values.len() < 3 {
        return f64::NAN;
    }
    moment_ratios(values).0 * (n * (n - 1.0)).sqrt() / (n - 2.0)
}

/// Sample excess kurtosis `G2` (module docs).
pub fn excess_kurtosis(values: &[f64]) -> f64 {
    let n = values.len() as f64;
    if values.len() < 4 {
        return f64::NAN;
    }
    let g2 = moment_ratios(values).1;
    ((n + 1.0) * g2 + 6.0) * (n - 1.0) / ((n - 2.0) * (n - 3.0))
}

/// Lag-1 sample autocorrelation (module docs).
pub fn autocorrelation(values: &[f64]) -> f64 {
    if values.len() < 3 || !varies(values) {
        return f64::NAN;
    }
    let centre = mean(values);
    let spread: f64 = values.iter().map(|v| (v - centre) * (v - centre)).sum();
    let lagged: f64 = values
        .windows(2)
        .map(|pair| (pair[0] - centre) * (pair[1] - centre))
        .sum();
    lagged / spread
}

/// A copy of `values` in increasing order (NaN after every number).
pub(super) fn sorted(values: &[f64]) -> Vec<f64> {
    let mut copy = values.to_vec();
    copy.sort_by(f64::total_cmp);
    copy
}

/// Quantile of increasing values at `probability` in `[0, 1]` (module docs); NaN for
/// no value.
pub(super) fn quantile_sorted(sorted: &[f64], probability: f64) -> f64 {
    let Some(last) = sorted.len().checked_sub(1) else {
        return f64::NAN;
    };
    let position = probability.clamp(0.0, 1.0) * last as f64;
    let below = position.floor();
    let lower_index = below as usize;
    let lower = sorted.get(lower_index).copied().unwrap_or(f64::NAN);
    let upper = sorted.get(lower_index + 1).copied().unwrap_or(lower);
    if position == below {
        lower
    } else {
        lower + (position - below) * (upper - lower)
    }
}

/// Quantiles of `values` at each of `probabilities` (fractions from 0 to 1), by the
/// definition in the module docs. NaN values are left out (a figure some samples cannot
/// define); with none left every quantile is NaN.
///
/// Errors: `InvalidProbability` for a probability outside `[0, 1]`.
pub fn percentiles(values: &[f64], probabilities: &[f64]) -> Result<Vec<f64>, MetricsError> {
    if let Some(&value) = probabilities.iter().find(|p| !(0.0..=1.0).contains(*p)) {
        return Err(MetricsError::InvalidProbability { value });
    }
    let known: Vec<f64> = values.iter().copied().filter(|v| !v.is_nan()).collect();
    let ordered = sorted(&known);
    Ok(probabilities
        .iter()
        .map(|&p| quantile_sorted(&ordered, p))
        .collect())
}
