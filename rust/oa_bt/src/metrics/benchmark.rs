//! Benchmark-relative statistics: beta, Jensen's alpha and correlation.
//!
//! Definitions, on the sessions where both the strategy return `s_i` and the
//! benchmark return `b_i` are known (a NaN marks a missing value), with `N` sessions
//! per year and the per-session risk-free rate `rf` (see `ratios`):
//!
//! - Beta: `cov(s, b) / var(b)`, the slope of the least-squares line of `s` on `b`.
//! - Alpha (Jensen's alpha, annualised): `(mean(s - rf) - beta * mean(b - rf)) * N`,
//!   the intercept of that line on excess returns, scaled by simple multiplication.
//! - Correlation: Pearson's `cov(s, b) / (sd(s) * sd(b))`.
//!
//! Undefined cases: with fewer than two paired sessions, or a benchmark that never
//! moves, all three are NaN; a strategy that never moves leaves the correlation NaN.

use super::curve::check_length;
use super::error::MetricsError;
use super::ratios::per_period_rate;

/// Benchmark-relative statistics (definitions in the module docs).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BenchmarkStats {
    /// Annualised Jensen's alpha, a fraction per year.
    pub alpha: f64,
    pub beta: f64,
    pub correlation: f64,
    /// Sessions on which both returns are known.
    pub sessions: usize,
}

/// Beta, alpha and correlation of `strategy` against `benchmark`, two series of
/// session returns (fractions) on the same session dates, NaN where missing.
///
/// Errors: `LengthMismatch` when the series differ in length; `NotFinite` for an
/// infinite return; the errors of `per_period_rate`.
pub fn benchmark_stats(
    strategy: &[f64],
    benchmark: &[f64],
    periods_per_year: f64,
    risk_free_annual: f64,
) -> Result<BenchmarkStats, MetricsError> {
    check_length(benchmark.len(), strategy.len(), "benchmark returns")?;
    let rf = per_period_rate(risk_free_annual, periods_per_year)?;
    let infinite = |values: &[f64]| values.iter().position(|v| v.is_infinite());
    if let Some(index) = infinite(strategy) {
        let what = "strategy return";
        return Err(MetricsError::NotFinite { what, index });
    }
    if let Some(index) = infinite(benchmark) {
        let what = "benchmark return";
        return Err(MetricsError::NotFinite { what, index });
    }
    let pairs: Vec<(f64, f64)> = strategy
        .iter()
        .zip(benchmark)
        .filter(|(s, b)| !s.is_nan() && !b.is_nan())
        .map(|(&s, &b)| (s, b))
        .collect();
    let mut stats = BenchmarkStats {
        alpha: f64::NAN,
        beta: f64::NAN,
        correlation: f64::NAN,
        sessions: pairs.len(),
    };
    if pairs.len() < 2 {
        return Ok(stats);
    }
    let n = pairs.len() as f64;
    let mean_s = pairs.iter().map(|p| p.0).sum::<f64>() / n;
    let mean_b = pairs.iter().map(|p| p.1).sum::<f64>() / n;
    let (mut cov, mut var_s, mut var_b) = (0.0, 0.0, 0.0);
    for &(s, b) in &pairs {
        cov += (s - mean_s) * (b - mean_b);
        var_s += (s - mean_s) * (s - mean_s);
        var_b += (b - mean_b) * (b - mean_b);
    }
    if var_b > 0.0 {
        stats.beta = cov / var_b;
        stats.alpha = ((mean_s - rf) - stats.beta * (mean_b - rf)) * periods_per_year;
        if var_s > 0.0 {
            stats.correlation = cov / (var_s * var_b).sqrt();
        }
    }
    Ok(stats)
}
