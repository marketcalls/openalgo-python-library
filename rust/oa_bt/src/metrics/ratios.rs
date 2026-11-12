//! Annualised return and the volatility-based ratios of a series of session returns.
//!
//! Definitions, for session returns `r_1..r_n`, `N` sessions per year and an annual
//! risk-free rate `R`:
//!
//! - Per-session risk-free rate `rf = (1 + R)^(1/N) - 1`: the rate that compounds to
//!   `R` over the `N` sessions of a year. It is subtracted from every session return;
//!   subtracting the annual rate itself would charge a year of interest per session.
//! - Excess return `x_i = r_i - rf`; mean excess `m = (x_1 + ... + x_n) / n`.
//! - Volatility: the sample standard deviation of the session returns (divisor
//!   `n - 1`) times `sqrt(N)`.
//! - Sharpe ratio: `m / s * sqrt(N)`, with `s` the sample standard deviation of the
//!   excess returns (equal to that of the returns, `rf` being constant).
//! - Sortino ratio: `m / d * sqrt(N)`, with the target downside deviation
//!   `d = sqrt((min(x_1, 0)^2 + ... + min(x_n, 0)^2) / n)`: every session counts in the
//!   divisor, and only shortfalls below the risk-free rate count in the sum.
//! - CAGR: `G^(1/y) - 1` for the growth `G` (final over initial wealth) and the
//!   elapsed time `y` in years of `DAYS_PER_YEAR` calendar days.
//! - Calmar ratio: CAGR divided by the magnitude of the maximum drawdown, both over
//!   the whole run.
//!
//! Undefined cases: with fewer than two returns volatility, Sharpe and Sortino are NaN.
//! A zero standard deviation makes Sharpe NaN. A zero downside deviation makes Sortino
//! infinite when the mean excess is positive and NaN otherwise. CAGR is NaN for less
//! than `MIN_CAGR_DAYS` of elapsed time, and -1 for a growth of zero or below (ruin).
//! Calmar is NaN when CAGR is, infinite for a positive CAGR without drawdown, and NaN
//! for no growth without drawdown.

use crate::model::{TsNanos, NANOS_PER_DAY};

use super::curve::check_finite;
use super::error::MetricsError;
use super::{DAYS_PER_YEAR, MIN_CAGR_DAYS};

/// Volatility-based statistics of a series of session returns.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ReturnStats {
    /// The per-session risk-free rate subtracted from each return, a fraction.
    pub risk_free_per_period: f64,
    /// Mean excess return per session, a fraction.
    pub mean_excess: f64,
    /// Annualised volatility, a fraction.
    pub volatility: f64,
    /// Annualised Sharpe ratio.
    pub sharpe: f64,
    /// Annualised Sortino ratio.
    pub sortino: f64,
}

fn check_periods(periods_per_year: f64) -> Result<(), MetricsError> {
    if periods_per_year.is_finite() && periods_per_year > 0.0 {
        Ok(())
    } else {
        Err(MetricsError::InvalidPeriodsPerYear {
            value: periods_per_year,
        })
    }
}

/// The per-period rate that compounds to `annual_rate` over `periods_per_year`
/// periods: `(1 + annual_rate)^(1 / periods_per_year) - 1`. Both rates are fractions.
///
/// Errors: `InvalidRiskFreeRate` for a rate that is not finite or is -1 or below;
/// `InvalidPeriodsPerYear` for a count that is not positive and finite.
pub fn per_period_rate(annual_rate: f64, periods_per_year: f64) -> Result<f64, MetricsError> {
    check_periods(periods_per_year)?;
    if !annual_rate.is_finite() || annual_rate <= -1.0 {
        return Err(MetricsError::InvalidRiskFreeRate { value: annual_rate });
    }
    Ok((1.0 + annual_rate).powf(1.0 / periods_per_year) - 1.0)
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

/// Sample standard deviation (divisor `n - 1`); NaN below two values.
pub(super) fn sample_std(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return f64::NAN;
    }
    let centre = mean(values);
    let squares: f64 = values.iter().map(|v| (v - centre) * (v - centre)).sum();
    (squares / (values.len() - 1) as f64).sqrt()
}

/// Volatility, Sharpe and Sortino of session returns (definitions in the module docs).
///
/// `returns` are fractions, `periods_per_year` is the number of sessions per year and
/// `risk_free_annual` the annual risk-free rate as a fraction.
///
/// Errors: `NotFinite` for a return that is NaN or infinite; the errors of
/// `per_period_rate`.
pub fn return_stats(
    returns: &[f64],
    periods_per_year: f64,
    risk_free_annual: f64,
) -> Result<ReturnStats, MetricsError> {
    let rf = per_period_rate(risk_free_annual, periods_per_year)?;
    check_finite(returns, "session return")?;
    let excess: Vec<f64> = returns.iter().map(|r| r - rf).collect();
    let scale = periods_per_year.sqrt();
    let mut stats = ReturnStats {
        risk_free_per_period: rf,
        mean_excess: f64::NAN,
        volatility: f64::NAN,
        sharpe: f64::NAN,
        sortino: f64::NAN,
    };
    if excess.is_empty() {
        return Ok(stats);
    }
    stats.mean_excess = mean(&excess);
    if excess.len() < 2 {
        return Ok(stats);
    }
    stats.volatility = sample_std(returns) * scale;
    let deviation = sample_std(&excess);
    if deviation > 0.0 {
        stats.sharpe = stats.mean_excess / deviation * scale;
    }
    let shortfall: f64 = excess.iter().map(|x| x.min(0.0).powi(2)).sum();
    let downside = (shortfall / excess.len() as f64).sqrt();
    stats.sortino = if downside > 0.0 {
        stats.mean_excess / downside * scale
    } else if stats.mean_excess > 0.0 {
        f64::INFINITY
    } else {
        f64::NAN
    };
    Ok(stats)
}

/// Compound annual growth rate of a growth factor `growth` (final over initial
/// wealth) earned over `elapsed` nanoseconds (definition and undefined cases in the
/// module docs). A non-finite growth gives NaN.
pub fn cagr(growth: f64, elapsed: TsNanos) -> f64 {
    if !growth.is_finite() {
        return f64::NAN;
    }
    if growth <= 0.0 {
        return -1.0;
    }
    let days = elapsed as f64 / NANOS_PER_DAY as f64;
    if days < MIN_CAGR_DAYS {
        return f64::NAN;
    }
    growth.powf(DAYS_PER_YEAR / days) - 1.0
}

/// Calmar ratio: `cagr / |max_drawdown|`, with `max_drawdown` a fraction from 0 down
/// to -1 (undefined cases in the module docs).
pub fn calmar(cagr: f64, max_drawdown: f64) -> f64 {
    let depth = max_drawdown.abs();
    if cagr.is_nan() || depth.is_nan() {
        f64::NAN
    } else if depth > 0.0 {
        cagr / depth
    } else if cagr > 0.0 {
        f64::INFINITY
    } else {
        f64::NAN
    }
}
