//! Benchmark-relative statistics: beta, Jensen's alpha, correlation, R squared, tracking
//! error, information ratio and Treynor ratio.
//!
//! Definitions, on the sessions where both the strategy return `s_i` and the
//! benchmark return `b_i` are known (a NaN marks a missing value), with `N` sessions
//! per year and the per-session risk-free rate `rf` (see `ratios`):
//!
//! - Beta: `cov(s, b) / var(b)`, the slope of the least-squares line of `s` on `b`.
//! - Alpha (Jensen's alpha, annualised): `(mean(s - rf) - beta * mean(b - rf)) * N`,
//!   the intercept of that line on excess returns, scaled by simple multiplication.
//! - Correlation: Pearson's `cov(s, b) / (sd(s) * sd(b))`.
//! - R squared: the square of the correlation, the share of the strategy's variance the
//!   least-squares line on the benchmark explains.
//! - Tracking error: the sample standard deviation of the active return `s - b`, times
//!   `sqrt(N)` (Grinold and Kahn 2000, Active Portfolio Management, 2nd ed., McGraw-Hill).
//! - Information ratio: `mean(s - b) / sd(s - b) * sqrt(N)`, the annual active return
//!   per unit of tracking error (same source).
//! - Treynor ratio (Treynor 1965, "How to Rate Management of Investment Funds", Harvard
//!   Business Review 43(1)): the annual mean excess return `mean(s - rf) * N` over beta.
//!
//! Undefined cases: with fewer than two paired sessions, or a benchmark that never
//! moves, beta, alpha, correlation, R squared and Treynor are NaN; a strategy that never
//! moves leaves the correlation and R squared NaN; tracking error is NaN below two paired
//! sessions and the information ratio also when the active return never varies.
//! Treynor is NaN when beta is zero.

use super::curve::{check_length, check_not_infinite};
use super::error::MetricsError;
use super::moments::varies;
use super::ratios::{per_period_rate, sample_std};

/// Benchmark-relative statistics (definitions in the module docs).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BenchmarkStats {
    /// Annualised Jensen's alpha, a fraction per year.
    pub alpha: f64,
    pub beta: f64,
    pub correlation: f64,
    pub r_squared: f64,
    /// Annualised, a fraction.
    pub tracking_error: f64,
    pub information_ratio: f64,
    pub treynor: f64,
    /// Sessions on which both returns are known.
    pub sessions: usize,
}

/// Beta, alpha, correlation and the active-return figures of `strategy` against
/// `benchmark`, two series of session returns (fractions) on the same session dates,
/// NaN where missing.
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
    check_not_infinite(strategy, "strategy return")?;
    check_not_infinite(benchmark, "benchmark return")?;
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
        r_squared: f64::NAN,
        tracking_error: f64::NAN,
        information_ratio: f64::NAN,
        treynor: f64::NAN,
        sessions: pairs.len(),
    };
    if pairs.len() < 2 {
        return Ok(stats);
    }
    regression(&pairs, rf, periods_per_year, &mut stats);
    active(&pairs, periods_per_year, &mut stats);
    Ok(stats)
}

/// Beta, alpha, correlation, R squared and Treynor of at least two pairs.
fn regression(pairs: &[(f64, f64)], rf: f64, periods_per_year: f64, stats: &mut BenchmarkStats) {
    let strategy: Vec<f64> = pairs.iter().map(|p| p.0).collect();
    let benchmark: Vec<f64> = pairs.iter().map(|p| p.1).collect();
    // A series of equal values never moves, whatever its computed mean rounds to.
    if !varies(&benchmark) {
        return;
    }
    let n = pairs.len() as f64;
    let mean_s = strategy.iter().sum::<f64>() / n;
    let mean_b = benchmark.iter().sum::<f64>() / n;
    let (mut cov, mut var_s, mut var_b) = (0.0, 0.0, 0.0);
    for &(s, b) in pairs {
        cov += (s - mean_s) * (b - mean_b);
        var_s += (s - mean_s) * (s - mean_s);
        var_b += (b - mean_b) * (b - mean_b);
    }
    stats.beta = cov / var_b;
    stats.alpha = ((mean_s - rf) - stats.beta * (mean_b - rf)) * periods_per_year;
    if stats.beta != 0.0 {
        stats.treynor = (mean_s - rf) * periods_per_year / stats.beta;
    }
    if varies(&strategy) {
        stats.correlation = cov / (var_s * var_b).sqrt();
        stats.r_squared = stats.correlation * stats.correlation;
    }
}

/// Tracking error and information ratio of at least two pairs.
fn active(pairs: &[(f64, f64)], periods_per_year: f64, stats: &mut BenchmarkStats) {
    let difference: Vec<f64> = pairs.iter().map(|(s, b)| s - b).collect();
    let deviation = sample_std(&difference);
    let scale = periods_per_year.sqrt();
    stats.tracking_error = deviation * scale;
    if deviation > 0.0 {
        let mean = difference.iter().sum::<f64>() / difference.len() as f64;
        stats.information_ratio = mean / deviation * scale;
    }
}
