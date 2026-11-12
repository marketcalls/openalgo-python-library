//! Corrections of the Sharpe and Sortino ratios of session returns: for autocorrelated
//! returns, for comparison with the Sharpe ratio, and the probability that the true
//! ratio is above zero; with the skewness and kurtosis they rest on.
//!
//! Definitions, for session returns `r_1..r_n`, `N` sessions per year, and Sharpe and
//! Sortino as `ratios` defines them (annualised by `sqrt(N)`):
//!
//! - Autocorrelation penalty (Lo 2002, "The Statistics of Sharpe Ratios", Financial
//!   Analysts Journal 58(4), the annualised ratio under first-order autocorrelation):
//!   with `rho` the lag-1 autocorrelation of the returns (`moments`) and `q = N` rounded
//!   to whole sessions, `P = sqrt(1 + 2 sum((1 - k/q) rho^k, k = 1..q-1))`. Annualising
//!   by `sqrt(q)` assumes independent sessions; `q` sessions with that autocorrelation
//!   have `P^2` times the variance, so the annual ratio is divided by `P`. `P = 1` when
//!   `rho = 0`; NaN when `rho` is NaN (below three returns, or no variation).
//! - Smart Sharpe and smart Sortino: Sharpe and Sortino divided by `P`.
//! - Adjusted Sortino: Sortino divided by `sqrt(2)` (Schwager 2012, Hedge Fund Market
//!   Wizards, Wiley, appendix on performance statistics). For returns spread evenly
//!   about the target the downside deviation is about the standard deviation over
//!   `sqrt(2)`, so the adjusted figure reads on the Sharpe ratio's scale.
//! - Probabilistic ratio (Bailey and Lopez de Prado 2012, "The Sharpe Ratio Efficient
//!   Frontier", Journal of Risk 15(2), the probabilistic Sharpe ratio against a
//!   benchmark ratio of 0): with the per-session ratio `S` (the annual ratio over
//!   `sqrt(N)`), the moment ratios `g1` and `g2` of the returns (`moments`), and
//!   `V = 1 - g1 S + (g2 + 2) / 4 S^2` (the paper's `(kurtosis - 1) / 4` with the
//!   kurtosis `g2 + 3`), it is `Phi(S sqrt(n - 1) / sqrt(V))`, a probability. Applied
//!   to the Sharpe ratio as published, and in the same form to the Sortino and adjusted
//!   Sortino ratios (an extension: the variance `V` is derived for the Sharpe ratio).
//!   NaN below two returns, for a ratio that is not finite, or when `V` is not positive.
//! - Risk-return ratio: mean session return over its sample standard deviation, per
//!   session and with no risk-free rate (the reward-to-variability ratio of Sharpe 1966,
//!   "Mutual Fund Performance", Journal of Business 39(1), at a zero rate). NaN below two
//!   returns or when the deviation is zero.

use std::f64::consts::SQRT_2;

use super::error::MetricsError;
use super::moments::{autocorrelation, excess_kurtosis, mean, moment_ratios, skewness};
use super::normal::normal_cdf;
use super::ratios::{return_stats, sample_std};

/// Corrected Sharpe and Sortino ratios of session returns (definitions in the module
/// docs). Ratios are pure numbers; probabilities are fractions.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AdjustedRatios {
    pub autocorrelation: f64,
    pub autocorrelation_penalty: f64,
    pub smart_sharpe: f64,
    pub smart_sortino: f64,
    pub adjusted_sortino: f64,
    pub probabilistic_sharpe: f64,
    pub probabilistic_sortino: f64,
    pub probabilistic_adjusted_sortino: f64,
    /// Sample skewness `G1` (`moments`).
    pub skew: f64,
    /// Sample excess kurtosis `G2` (`moments`).
    pub kurtosis: f64,
    pub risk_return_ratio: f64,
}

/// The autocorrelation penalty `P` for a lag-1 autocorrelation `rho` over
/// `periods_per_year` sessions (module docs).
pub fn autocorrelation_penalty(rho: f64, periods_per_year: f64) -> f64 {
    if !rho.is_finite() || !periods_per_year.is_finite() || periods_per_year < 1.0 {
        return f64::NAN;
    }
    let q = periods_per_year.round() as usize;
    let (mut power, mut sum) = (1.0, 0.0);
    for k in 1..q {
        power *= rho;
        sum += (1.0 - k as f64 / q as f64) * power;
    }
    let variance_ratio = 1.0 + 2.0 * sum;
    if variance_ratio > 0.0 {
        variance_ratio.sqrt()
    } else {
        f64::NAN
    }
}

/// The probabilistic ratio of a per-session ratio over `sessions` returns with moment
/// ratios `g1` and `g2` (module docs).
pub fn probabilistic_ratio(per_session: f64, sessions: usize, g1: f64, g2: f64) -> f64 {
    let known = per_session.is_finite() && g1.is_finite() && g2.is_finite();
    if sessions < 2 || !known {
        return f64::NAN;
    }
    let variance = 1.0 - g1 * per_session + (g2 + 2.0) / 4.0 * per_session * per_session;
    if variance <= 0.0 {
        return f64::NAN;
    }
    normal_cdf(per_session * ((sessions - 1) as f64).sqrt() / variance.sqrt())
}

/// The corrected ratios of session returns (module docs). `returns` are fractions,
/// `periods_per_year` the sessions per year and `risk_free_annual` the annual rate.
///
/// Errors: those of `ratios::return_stats`.
pub fn adjusted_ratios(
    returns: &[f64],
    periods_per_year: f64,
    risk_free_annual: f64,
) -> Result<AdjustedRatios, MetricsError> {
    let stats = return_stats(returns, periods_per_year, risk_free_annual)?;
    let rho = autocorrelation(returns);
    let penalty = autocorrelation_penalty(rho, periods_per_year);
    let adjusted_sortino = stats.sortino / SQRT_2;
    let (g1, g2) = moment_ratios(returns);
    let per_session = |annual: f64| annual / periods_per_year.sqrt();
    let probability = |annual: f64| probabilistic_ratio(per_session(annual), returns.len(), g1, g2);
    // `sample_std` is exactly 0 for equal values, never a rounding residue.
    let deviation = sample_std(returns);
    Ok(AdjustedRatios {
        autocorrelation: rho,
        autocorrelation_penalty: penalty,
        smart_sharpe: stats.sharpe / penalty,
        smart_sortino: stats.sortino / penalty,
        adjusted_sortino,
        probabilistic_sharpe: probability(stats.sharpe),
        probabilistic_sortino: probability(stats.sortino),
        probabilistic_adjusted_sortino: probability(adjusted_sortino),
        skew: skewness(returns),
        kurtosis: excess_kurtosis(returns),
        risk_return_ratio: if deviation > 0.0 {
            mean(returns) / deviation
        } else {
            f64::NAN
        },
    })
}
