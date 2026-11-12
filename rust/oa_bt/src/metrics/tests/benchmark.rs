//! Beta, alpha and correlation against a benchmark.

use super::assert_close;
use crate::metrics::{benchmark_stats, MetricsError};

#[test]
fn twice_the_benchmark_has_beta_two_and_correlation_one() {
    let benchmark = [0.01, -0.02, 0.03];
    let strategy = [0.02, -0.04, 0.06];
    let stats = benchmark_stats(&strategy, &benchmark, 249.0, 0.0).unwrap();
    assert_close(stats.beta, 2.0);
    assert_close(stats.correlation, 1.0);
    assert_close(stats.alpha, 0.0);
    assert_eq!(stats.sessions, 3);
}

#[test]
fn alpha_is_the_annualised_intercept_on_excess_returns() {
    // rf = 0.1 a session (21 percent a year, two sessions a year). With s = 2b:
    // mean(s - rf) - 2 * mean(b - rf) = rf, and two sessions a year make 0.2.
    let benchmark = [0.01, -0.02, 0.03];
    let strategy = [0.02, -0.04, 0.06];
    let stats = benchmark_stats(&strategy, &benchmark, 2.0, 0.21).unwrap();
    assert_close(stats.alpha, 0.2);
}

#[test]
fn a_general_pair_follows_the_least_squares_definitions() {
    let (b, s) = ([0.01, -0.02, 0.03], [0.02, -0.01, 0.01]);
    let (mb, ms) = (0.02 / 3.0, 0.02 / 3.0);
    let cov: f64 = (0..3).map(|i| (s[i] - ms) * (b[i] - mb)).sum();
    let var_b: f64 = b.iter().map(|x| (x - mb).powi(2)).sum();
    let var_s: f64 = s.iter().map(|x| (x - ms).powi(2)).sum();
    let stats = benchmark_stats(&s, &b, 4.0, 0.0).unwrap();
    assert_close(stats.beta, cov / var_b);
    assert_close(stats.correlation, cov / (var_b * var_s).sqrt());
    assert_close(stats.alpha, (ms - cov / var_b * mb) * 4.0);
}

#[test]
fn an_opposite_strategy_has_negative_beta_and_correlation() {
    let stats = benchmark_stats(&[-0.01, 0.02], &[0.01, -0.02], 249.0, 0.0).unwrap();
    assert_close(stats.beta, -1.0);
    assert_close(stats.correlation, -1.0);
}

#[test]
fn sessions_with_a_missing_benchmark_return_are_left_out() {
    let benchmark = [0.01, f64::NAN, -0.02, 0.03];
    let strategy = [0.02, 0.5, -0.04, 0.06];
    let stats = benchmark_stats(&strategy, &benchmark, 249.0, 0.0).unwrap();
    assert_eq!(stats.sessions, 3);
    assert_close(stats.beta, 2.0);
}

#[test]
fn a_benchmark_that_never_moves_leaves_everything_undefined() {
    let stats = benchmark_stats(&[0.01, 0.02], &[0.0, 0.0], 249.0, 0.0).unwrap();
    assert!(stats.beta.is_nan() && stats.alpha.is_nan() && stats.correlation.is_nan());
}

#[test]
fn fewer_than_two_paired_sessions_leave_everything_undefined() {
    let stats = benchmark_stats(&[0.01], &[0.02], 249.0, 0.0).unwrap();
    assert!(stats.beta.is_nan() && stats.correlation.is_nan());
}

#[test]
fn malformed_input_is_refused() {
    assert!(matches!(
        benchmark_stats(&[0.01], &[0.02, 0.0], 249.0, 0.0),
        Err(MetricsError::LengthMismatch { .. })
    ));
    assert!(matches!(
        benchmark_stats(&[0.01, 0.0], &[f64::INFINITY, 0.0], 249.0, 0.0),
        Err(MetricsError::NotFinite { index: 0, .. })
    ));
}
