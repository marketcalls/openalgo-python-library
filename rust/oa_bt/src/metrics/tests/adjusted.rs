//! The autocorrelation penalty, smart and adjusted ratios, and probabilistic ratios.

use super::assert_close;
use crate::metrics::{adjusted_ratios, autocorrelation_penalty, probabilistic_ratio, return_stats};

#[test]
fn the_penalty_sums_the_decaying_autocorrelations_of_a_year() {
    // q = 4, rho = 0.5: 0.75 x 0.5 + 0.5 x 0.25 + 0.25 x 0.125 = 0.53125.
    assert_close(
        autocorrelation_penalty(0.5, 4.0),
        (1.0_f64 + 2.0 * 0.53125).sqrt(),
    );
    // Sessions per year is rounded to whole sessions: 4.4 counts as 4.
    assert_close(autocorrelation_penalty(0.5, 4.4), (2.0625_f64).sqrt());
}

#[test]
fn independent_returns_have_no_penalty() {
    assert_eq!(autocorrelation_penalty(0.0, 249.0), 1.0);
    assert!(autocorrelation_penalty(f64::NAN, 249.0).is_nan());
}

#[test]
fn smart_ratios_divide_by_the_penalty_of_the_returns_own_autocorrelation() {
    // Returns rising by 0.01 a session have a lag-1 autocorrelation of 0.25 (as for
    // 1, 2, 3, 4). With q = 4: 0.75 x 0.25 + 0.5 x 0.0625 + 0.25 x 0.015625.
    let returns = [0.01, 0.02, 0.03, 0.04];
    let penalty = (1.0_f64 + 2.0 * 0.222_656_25).sqrt();
    let plain = return_stats(&returns, 4.0, 0.0).unwrap();
    let adjusted = adjusted_ratios(&returns, 4.0, 0.0).unwrap();
    assert_close(adjusted.autocorrelation, 0.25);
    assert_close(adjusted.autocorrelation_penalty, penalty);
    assert_close(adjusted.smart_sharpe, plain.sharpe / penalty);
    // Mean 0.025 over a sample deviation of sqrt(0.0005 / 3), times sqrt(4).
    assert_close(plain.sharpe, 0.025 / (0.0005_f64 / 3.0).sqrt() * 2.0);
}

#[test]
fn the_adjusted_sortino_is_the_sortino_over_root_two() {
    // Mean 0.04 / 3; downside sqrt(0.0001 / 3); four sessions a year.
    let returns = [0.02, -0.01, 0.03];
    let sortino = 0.04 / 3.0 / (0.0001_f64 / 3.0).sqrt() * 2.0;
    let adjusted = adjusted_ratios(&returns, 4.0, 0.0).unwrap();
    assert_close(adjusted.adjusted_sortino, sortino / 2.0_f64.sqrt());
}

#[test]
fn the_probabilistic_ratio_is_the_normal_probability_of_the_standardised_ratio() {
    // Kurtosis 1 (excess -2) and no skew make the variance term 1: z = 0.2 x sqrt(25) = 1,
    // and the normal table gives 0.841344746068543 at 1.
    assert_close(
        probabilistic_ratio(0.2, 26, 0.0, -2.0),
        0.841_344_746_068_543,
    );
    // A ratio of zero is as likely above zero as below.
    assert_close(probabilistic_ratio(0.0, 50, 0.3, 1.0), 0.5);
}

#[test]
fn the_probabilistic_ratio_is_undefined_without_enough_data() {
    assert!(probabilistic_ratio(0.2, 1, 0.0, 0.0).is_nan());
    assert!(probabilistic_ratio(f64::INFINITY, 30, 0.0, 0.0).is_nan());
    assert!(probabilistic_ratio(0.2, 30, f64::NAN, 0.0).is_nan());
    // 1 - 10 x 0.5 + 0.5 x 0.25 is negative: no variance, no probability.
    assert!(probabilistic_ratio(0.5, 30, 10.0, 0.0).is_nan());
}

#[test]
fn a_two_point_sample_has_no_skew_and_an_excess_kurtosis_of_minus_two() {
    let returns: Vec<f64> = (0..26)
        .map(|i| if i % 2 == 0 { 0.03 } else { -0.01 })
        .collect();
    let adjusted = adjusted_ratios(&returns, 249.0, 0.0).unwrap();
    // The moment ratios are 0 and -2; the reported G2 corrects for the sample size:
    // ((n + 1) (-2) + 6) (n - 1) / ((n - 2) (n - 3)) with n = 26.
    assert!(adjusted.skew.abs() < 1e-9, "{}", adjusted.skew);
    assert_close(
        adjusted.kurtosis,
        (27.0 * -2.0 + 6.0) * 25.0 / (24.0 * 23.0),
    );
}

#[test]
fn the_risk_return_ratio_is_the_unannualised_mean_over_deviation() {
    let adjusted = adjusted_ratios(&[0.02, -0.01, 0.03], 249.0, 0.065).unwrap();
    // No risk-free rate and no annualisation: mean over the sample deviation.
    let mean = 0.04 / 3.0;
    let squares = (0.02_f64 - mean).powi(2) + (-0.01 - mean).powi(2) + (0.03 - mean).powi(2);
    assert_close(adjusted.risk_return_ratio, mean / (squares / 2.0).sqrt());
}

#[test]
fn a_flat_series_has_no_corrected_ratio() {
    let adjusted = adjusted_ratios(&[0.001; 5], 249.0, 0.0).unwrap();
    for value in [
        adjusted.smart_sharpe,
        adjusted.probabilistic_sharpe,
        adjusted.risk_return_ratio,
        adjusted.skew,
    ] {
        assert!(value.is_nan(), "{adjusted:?}");
    }
}
