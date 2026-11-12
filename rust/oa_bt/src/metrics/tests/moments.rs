//! Skewness, kurtosis, autocorrelation, quantiles and the normal distribution function.

use super::assert_close;
use crate::metrics::{
    autocorrelation, excess_kurtosis, normal_cdf, percentiles, skewness, MetricsError,
};

#[test]
fn skewness_and_kurtosis_follow_the_adjusted_estimators() {
    // [0, 0, 0, 1]: mean 0.25, m2 = 0.1875, m3 = 0.09375, m4 = 0.08203125, so
    // g1 = 2 / sqrt(3) and g2 = -2/3; G1 = g1 sqrt(12) / 2 = 2 and
    // G2 = (5 (-2/3) + 6) 3 / (2 x 1) = 4.
    let values = [0.0, 0.0, 0.0, 1.0];
    assert_close(skewness(&values), 2.0);
    assert_close(excess_kurtosis(&values), 4.0);
}

#[test]
fn a_symmetric_sample_has_no_skew() {
    assert_close(skewness(&[-0.02, 0.0, 0.02]), 0.0);
}

#[test]
fn too_few_values_or_no_variation_leave_the_moments_undefined() {
    assert!(skewness(&[0.01, 0.02]).is_nan());
    assert!(excess_kurtosis(&[0.01, 0.02, 0.03]).is_nan());
    assert!(skewness(&[0.01; 5]).is_nan());
    assert!(excess_kurtosis(&[0.01; 5]).is_nan());
}

#[test]
fn the_lag_one_autocorrelation_is_the_sample_estimate() {
    // [1, 2, 3, 4]: deviations -1.5, -0.5, 0.5, 1.5; lagged products 0.75 - 0.25 + 0.75
    // = 1.25 over a sum of squares of 5.
    assert_close(autocorrelation(&[1.0, 2.0, 3.0, 4.0]), 0.25);
    // Alternating signs: -3 over 4.
    assert_close(autocorrelation(&[1.0, -1.0, 1.0, -1.0]), -0.75);
    assert!(autocorrelation(&[1.0, 2.0]).is_nan());
    assert!(autocorrelation(&[0.5; 4]).is_nan());
}

#[test]
fn quantiles_interpolate_between_order_statistics() {
    let values = [5.0, 1.0, 4.0, 2.0, 3.0];
    // Positions (n - 1) p = 0.2, 2 and 3.6 among 1, 2, 3, 4, 5.
    let result = percentiles(&values, &[0.05, 0.5, 0.9, 0.0, 1.0]).unwrap();
    for (actual, expected) in result.into_iter().zip([1.2, 3.0, 4.6, 1.0, 5.0]) {
        assert_close(actual, expected);
    }
}

#[test]
fn quantiles_leave_out_undefined_values() {
    let result = percentiles(&[5.0, f64::NAN, 1.0, 3.0], &[0.5]).unwrap();
    assert_close(result[0], 3.0);
    assert!(percentiles(&[f64::NAN], &[0.5]).unwrap()[0].is_nan());
    assert!(percentiles(&[], &[0.5]).unwrap()[0].is_nan());
}

#[test]
fn a_level_outside_zero_to_one_is_refused() {
    assert_eq!(
        percentiles(&[1.0], &[95.0]),
        Err(MetricsError::InvalidProbability { value: 95.0 })
    );
}

#[test]
fn the_normal_distribution_function_matches_the_table() {
    // Reference values: 0.5 erfc(-x / sqrt(2)) in double precision.
    let table = [
        (0.0, 0.5),
        (1.96, 0.975_002_104_851_779_6),
        (-1.0, 0.158_655_253_931_457_07),
        (-3.0, 0.001_349_898_031_630_095_9),
        (-7.5, 3.190_891_672_910_92e-14),
        (-8.0, 6.220_960_574_271_819e-16),
    ];
    for (x, expected) in table {
        let actual = normal_cdf(x);
        assert!(
            (actual - expected).abs() <= 1e-15_f64.max(expected * 1e-12),
            "{x}: expected {expected}, got {actual}"
        );
    }
    assert_eq!((normal_cdf(-40.0), normal_cdf(40.0)), (0.0, 1.0));
    assert!(normal_cdf(f64::NAN).is_nan());
}
