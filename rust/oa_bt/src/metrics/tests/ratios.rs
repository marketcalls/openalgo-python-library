//! The per-session risk-free rate, volatility, Sharpe, Sortino, CAGR and Calmar.

use super::{assert_close, days};
use crate::metrics::{cagr, calmar, per_period_rate, return_stats, MetricsError};

#[test]
fn the_annual_rate_is_converted_to_the_rate_that_compounds_to_it() {
    // 1.1 * 1.1 = 1.21: ten percent a session twice a year is 21 percent a year.
    assert_close(per_period_rate(0.21, 2.0).unwrap(), 0.1);
    assert_close(per_period_rate(0.0, 249.0).unwrap(), 0.0);
}

#[test]
fn unusable_rates_and_counts_are_refused() {
    assert_eq!(
        per_period_rate(-1.0, 2.0),
        Err(MetricsError::InvalidRiskFreeRate { value: -1.0 })
    );
    assert!(per_period_rate(f64::NAN, 2.0).is_err());
    assert_eq!(
        per_period_rate(0.05, 0.0),
        Err(MetricsError::InvalidPeriodsPerYear { value: 0.0 })
    );
    assert!(return_stats(&[0.1, f64::NAN], 4.0, 0.0).is_err());
}

#[test]
fn volatility_sharpe_and_sortino_follow_their_definitions() {
    let r = [0.1, -0.1, 2.0 / 9.0];
    let n: f64 = 3.0;
    let mean: f64 = (0.1 - 0.1 + 2.0 / 9.0) / n;
    let variance =
        ((0.1 - mean).powi(2) + (-0.1 - mean).powi(2) + (2.0 / 9.0 - mean).powi(2)) / (n - 1.0);
    // Four sessions a year: annualise by sqrt(4) = 2.
    let stats = return_stats(&r, 4.0, 0.0).unwrap();
    assert_close(stats.mean_excess, mean);
    assert_close(stats.volatility, variance.sqrt() * 2.0);
    assert_close(stats.sharpe, mean / variance.sqrt() * 2.0);
    // Only the -0.1 session falls short of zero; the divisor counts all three.
    assert_close(stats.sortino, mean / (0.01_f64 / n).sqrt() * 2.0);
    assert!((stats.sharpe - 0.9107).abs() < 1e-4, "{}", stats.sharpe);
}

#[test]
fn the_per_session_risk_free_rate_is_subtracted_from_each_return() {
    // Two sessions a year at 21 percent a year: 10 percent a session.
    let stats = return_stats(&[0.3, 0.1], 2.0, 0.21).unwrap();
    assert_close(stats.risk_free_per_period, 0.1);
    // Excess returns 0.2 and 0.0: mean 0.1, sample deviation sqrt(0.02).
    assert_close(stats.sharpe, 0.1 / 0.02_f64.sqrt() * 2.0_f64.sqrt());
    assert_close(stats.sharpe, 1.0);
    // Subtracting the annual 0.21 from each session would give a Sharpe of -0.1.
    assert!((stats.sharpe + 0.1).abs() > 0.5);
}

#[test]
fn a_flat_series_has_zero_volatility_and_no_sharpe_or_sortino() {
    let stats = return_stats(&[0.0, 0.0, 0.0], 249.0, 0.0).unwrap();
    assert_eq!(stats.volatility, 0.0);
    assert!(stats.sharpe.is_nan() && stats.sortino.is_nan());
}

#[test]
fn equal_nonzero_returns_have_no_deviation_whatever_their_mean_rounds_to() {
    // The mean of three 0.01 returns is a rounding unit away from 0.01; the deviation is
    // still exactly zero, so Sharpe stays undefined instead of enormous.
    let stats = return_stats(&[0.01, 0.01, 0.01], 249.0, 0.0).unwrap();
    assert_eq!(stats.volatility, 0.0);
    assert!(stats.sharpe.is_nan());
}

#[test]
fn a_series_with_no_shortfall_has_an_infinite_sortino() {
    let stats = return_stats(&[0.01, 0.02], 249.0, 0.0).unwrap();
    assert_eq!(stats.sortino, f64::INFINITY);
}

#[test]
fn fewer_than_two_returns_leave_the_ratios_undefined() {
    for returns in [&[][..], &[0.01][..]] {
        let stats = return_stats(returns, 249.0, 0.0).unwrap();
        assert!(stats.volatility.is_nan() && stats.sharpe.is_nan() && stats.sortino.is_nan());
    }
    assert!(return_stats(&[], 249.0, 0.0).unwrap().mean_excess.is_nan());
}

#[test]
fn cagr_annualises_over_elapsed_calendar_time() {
    let year = 365.25;
    assert_close(cagr(1.21, days(2) * 365 + days(1) / 2), 0.1);
    assert_close(cagr(1.1, (year * 86_400e9) as i64), 0.1);
    // Half a year at 10 percent compounds to 21 percent a year.
    assert_close(cagr(1.1, (year / 2.0 * 86_400e9) as i64), 0.21);
}

#[test]
fn cagr_is_undefined_below_four_weeks_and_minus_one_at_ruin() {
    assert!(cagr(1.01, days(27)).is_nan());
    assert!(cagr(1.01, days(28)).is_finite());
    assert_eq!(cagr(0.0, days(1)), -1.0);
    assert_eq!(cagr(-0.5, days(400)), -1.0);
    assert!(cagr(f64::NAN, days(400)).is_nan());
}

#[test]
fn calmar_divides_cagr_by_the_depth_of_the_maximum_drawdown() {
    assert_close(calmar(0.1, -0.2), 0.5);
    assert_close(calmar(-0.1, -0.2), -0.5);
    assert_eq!(calmar(0.1, 0.0), f64::INFINITY);
    assert!(calmar(0.0, 0.0).is_nan());
    assert!(calmar(f64::NAN, -0.1).is_nan());
}
