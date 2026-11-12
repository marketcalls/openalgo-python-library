//! Drawdown-based measures on session closes: the ulcer index and its performance
//! index, the conditional drawdown and the serenity index; and the statistics of a
//! path of returns (used by the horizon summary and by Monte Carlo).
//!
//! Definitions, for session returns `r_1..r_n` (fractions) in date order:
//!
//! - Close drawdowns: the wealth `W_k = (1 + r_1) ... (1 + r_k)` from `W_0 = 1` at the
//!   opening, and `D_k = W_k / max(1, W_1..W_k) - 1`, zero or negative. Once a session
//!   loses everything (`1 + r <= 0`) the wealth is 0 and the path ends there.
//! - Ulcer index (Martin and McCann 1989):
//!   `sqrt((D_1^2 + ... + D_n^2) / n)`, a fraction; 0 for a path that never falls.
//! - Ulcer performance index (the Martin ratio, same source): `(CAGR - R) / ulcer index`
//!   with `R` the annual risk-free rate.
//! - Conditional drawdown at risk (Chekhlov, Uryasev and Zabarankin 2005, "Drawdown
//!   measure in portfolio optimization", International Journal of Theoretical and
//!   Applied Finance 8(1)), at 95 percent: the mean of the close drawdowns at or below
//!   their 5 percent quantile (`moments`), zero or negative.
//! - Serenity index (excess return over drawdown-based risk):
//!   `(CAGR - R) / (ulcer index x pitfall)`, with the pitfall indicator
//!   `-CDaR / annualised volatility`: the excess return per unit of how long and how
//!   badly the run stayed under water, and of how deep its worst drawdowns were against
//!   its ordinary swings.
//! - Path statistics: total return `W_n - 1`, CAGR over the given elapsed time, the
//!   volatility, Sharpe and Sortino of `ratios`, and the deepest close drawdown.
//!
//! Undefined cases: with no session the ulcer index and the conditional drawdown are
//! NaN. A ratio with a zero denominator is infinite when its numerator is positive and
//! NaN otherwise; any NaN input gives NaN.

use super::curve::check_finite;
use super::error::MetricsError;
use super::moments::{mean, quantile_sorted, sorted};
use super::ratios::{cagr, return_stats};
use super::tails::{ratio_or_infinite, TAIL_PROBABILITY};
use crate::model::TsNanos;

/// Statistics of one path of returns (definitions in the module docs).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PathStats {
    /// Fraction.
    pub total_return: f64,
    /// Fraction per year.
    pub cagr: f64,
    /// Annualised, a fraction.
    pub volatility: f64,
    pub sharpe: f64,
    pub sortino: f64,
    /// The deepest close drawdown, a fraction from 0 down to -1.
    pub max_drawdown: f64,
}

/// The returns up to and including the first that loses everything.
fn until_ruin(returns: &[f64]) -> &[f64] {
    match returns.iter().position(|&r| r <= -1.0) {
        Some(ruin) => returns.get(..=ruin).unwrap_or(returns),
        None => returns,
    }
}

/// The drawdown at each session close (module docs), as many as the path has sessions.
pub fn close_drawdowns(returns: &[f64]) -> Vec<f64> {
    let (mut wealth, mut peak) = (1.0_f64, 1.0_f64);
    until_ruin(returns)
        .iter()
        .map(|r| {
            wealth = (wealth * (1.0 + r)).max(0.0);
            peak = peak.max(wealth);
            wealth / peak - 1.0
        })
        .collect()
}

/// The ulcer index of close drawdowns (module docs).
pub fn ulcer_index(drawdowns: &[f64]) -> f64 {
    let squares: Vec<f64> = drawdowns.iter().map(|d| d * d).collect();
    mean(&squares).sqrt()
}

/// The conditional drawdown at risk of close drawdowns (module docs).
pub fn conditional_drawdown(drawdowns: &[f64]) -> f64 {
    let ordered = sorted(drawdowns);
    let cutoff = quantile_sorted(&ordered, TAIL_PROBABILITY);
    let worst: Vec<f64> = ordered.into_iter().filter(|&d| d <= cutoff).collect();
    mean(&worst)
}

/// The ulcer performance index (module docs). All arguments are annual fractions
/// except the ulcer index, a fraction.
pub fn ulcer_performance_index(cagr: f64, risk_free_annual: f64, ulcer: f64) -> f64 {
    if cagr.is_nan() || ulcer.is_nan() {
        return f64::NAN;
    }
    ratio_or_infinite(cagr - risk_free_annual, ulcer)
}

/// The serenity index (module docs). `cdar` is the conditional drawdown (zero or
/// negative) and `volatility` the annualised volatility.
pub fn serenity_index(
    cagr: f64,
    risk_free_annual: f64,
    ulcer: f64,
    cdar: f64,
    volatility: f64,
) -> f64 {
    let pitfall = -cdar / volatility;
    if [cagr, ulcer, pitfall].iter().any(|v| v.is_nan()) {
        return f64::NAN;
    }
    ratio_or_infinite(cagr - risk_free_annual, ulcer * pitfall)
}

/// Statistics of a path of session returns over `elapsed` nanoseconds (module docs).
/// `periods_per_year` and `risk_free_annual` annualise as in `ratios`.
///
/// Errors: `NotFinite` for a return that is NaN or infinite; those of
/// `ratios::return_stats`.
pub fn path_stats(
    returns: &[f64],
    periods_per_year: f64,
    risk_free_annual: f64,
    elapsed: TsNanos,
) -> Result<PathStats, MetricsError> {
    check_finite(returns, "session return")?;
    let lived = until_ruin(returns);
    let stats = return_stats(lived, periods_per_year, risk_free_annual)?;
    let growth = lived.iter().fold(1.0, |w, r| (w * (1.0 + r)).max(0.0));
    let deepest = close_drawdowns(lived).into_iter().fold(0.0, f64::min);
    Ok(PathStats {
        total_return: growth - 1.0,
        cagr: cagr(growth, elapsed),
        volatility: stats.volatility,
        sharpe: stats.sharpe,
        sortino: stats.sortino,
        max_drawdown: deepest,
    })
}
