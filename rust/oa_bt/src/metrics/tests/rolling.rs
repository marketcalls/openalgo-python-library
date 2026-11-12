//! Rolling volatility, Sharpe, Sortino, win rate and beta.

use super::assert_close;
use crate::metrics::{rolling_stats, MetricsError};

const RETURNS: [f64; 4] = [0.01, -0.01, 0.02, 0.0];

#[test]
fn each_value_is_the_statistic_of_the_window_ending_there() {
    let rolling = rolling_stats(&RETURNS, None, 2, 4.0, 0.0).unwrap();
    assert!(rolling.volatility[0].is_nan() && rolling.sharpe[0].is_nan());
    // Window 0.01, -0.01: sample deviation sqrt(0.0002), times sqrt(4); mean 0.
    assert_close(rolling.volatility[1], 0.0002_f64.sqrt() * 2.0);
    assert_close(rolling.sharpe[1], 0.0);
    // Window -0.01, 0.02: mean 0.005 over sqrt(0.00045), times 2.
    assert_close(rolling.sharpe[2], 0.005 / 0.000_45_f64.sqrt() * 2.0);
    // Only -0.01 falls short: downside sqrt(0.0001 / 2).
    assert_close(rolling.sortino[2], 0.005 / (0.0001_f64 / 2.0).sqrt() * 2.0);
}

#[test]
fn the_rolling_win_rate_counts_the_windows_sessions() {
    let rolling = rolling_stats(&RETURNS, None, 2, 4.0, 0.0).unwrap();
    assert!(rolling.win_rate[0].is_nan());
    assert_close(rolling.win_rate[1], 0.5);
    // 0.02 won and the flat session is not counted.
    assert_close(rolling.win_rate[3], 1.0);
    assert!(rolling.beta.is_empty());
}

#[test]
fn rolling_beta_regresses_the_window_on_the_benchmark() {
    let benchmark: Vec<f64> = RETURNS.iter().map(|r| r / 2.0).collect();
    let rolling = rolling_stats(&RETURNS, Some(&benchmark), 3, 4.0, 0.0).unwrap();
    assert!(rolling.beta[1].is_nan());
    assert_close(rolling.beta[2], 2.0);
    assert_close(rolling.beta[3], 2.0);
}

#[test]
fn a_window_longer_than_the_run_leaves_every_value_undefined() {
    let rolling = rolling_stats(&RETURNS, None, 10, 4.0, 0.0).unwrap();
    assert_eq!(rolling.volatility.len(), 4);
    assert!(rolling.volatility.iter().all(|v| v.is_nan()));
}

#[test]
fn a_window_below_two_sessions_or_a_mismatched_benchmark_is_refused() {
    assert_eq!(
        rolling_stats(&RETURNS, None, 1, 4.0, 0.0),
        Err(MetricsError::InvalidWindow { sessions: 1 })
    );
    assert!(matches!(
        rolling_stats(&RETURNS, Some(&[0.0]), 2, 4.0, 0.0),
        Err(MetricsError::LengthMismatch { .. })
    ));
}
