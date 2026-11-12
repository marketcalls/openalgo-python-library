//! Value at risk, expected shortfall, tail and outlier ratios, Omega and gain to pain.

use super::assert_close;
use crate::metrics::{gain_to_pain, omega, tail_stats};

/// Ten session returns from -5 to +6 percent.
const TEN: [f64; 10] = [0.03, -0.05, 0.06, -0.01, 0.0, 0.01, -0.03, 0.02, 0.05, 0.04];

#[test]
fn value_at_risk_and_expected_shortfall_read_the_left_tail() {
    let tails = tail_stats(&TEN, 0.0).unwrap();
    // Sorted -0.05, -0.03, ...: the 5 percent quantile sits at position 0.45, between
    // -0.05 and -0.03. Only -0.05 is at or below it.
    assert_close(tails.value_at_risk, -0.05 + 0.45 * 0.02);
    assert_close(tails.expected_shortfall, -0.05);
}

#[test]
fn the_tail_ratio_sets_the_right_tail_against_the_left() {
    let tails = tail_stats(&TEN, 0.0).unwrap();
    // The 95 percent quantile sits at position 8.55, between 0.05 and 0.06.
    assert_close(tails.tail_ratio, 0.0555 / 0.041);
}

#[test]
fn outlier_ratios_compare_the_extreme_return_with_the_typical_one() {
    let tails = tail_stats(&TEN, 0.0).unwrap();
    // 99 percent at position 8.91: 0.0591, over the mean gain 0.21 / 6.
    assert_close(tails.outlier_win_ratio, 0.0591 / 0.035);
    // 1 percent at position 0.09: -0.0482, over the mean loss -0.09 / 3.
    assert_close(tails.outlier_loss_ratio, -0.0482 / -0.03);
}

#[test]
fn omega_weighs_gains_above_the_threshold_against_shortfalls_below_it() {
    // At 0: gains 0.21, shortfalls 0.09.
    assert_close(omega(&TEN, 0.0), 0.21 / 0.09);
    // At 0.01: gains 0.01 + 0.02 + 0.03 + 0.04 + 0.05 = 0.15, shortfalls
    // 0.06 + 0.04 + 0.02 + 0.01 = 0.13.
    assert_close(omega(&TEN, 0.01), 0.15 / 0.13);
    assert_eq!(omega(&[0.01, 0.02], 0.0), f64::INFINITY);
    assert!(omega(&[], 0.0).is_nan());
}

#[test]
fn a_flat_series_has_no_tail_ratio_and_no_omega() {
    let tails = tail_stats(&[0.0; 4], 0.0).unwrap();
    assert_eq!((tails.value_at_risk, tails.expected_shortfall), (0.0, 0.0));
    assert!(tails.tail_ratio.is_nan() && tails.omega.is_nan());
    assert!(tails.outlier_win_ratio.is_nan() && tails.outlier_loss_ratio.is_nan());
}

#[test]
fn all_winning_sessions_have_no_outlier_loss_ratio() {
    let tails = tail_stats(&[0.01, 0.02, 0.03], 0.0).unwrap();
    // The 5 percent quantile is a gain: position 0.1 between 0.01 and 0.02.
    assert_close(tails.value_at_risk, 0.011);
    assert!(tails.outlier_loss_ratio.is_nan());
    assert_eq!(tails.omega, f64::INFINITY);
}

#[test]
fn no_session_has_no_tail_figure() {
    let tails = tail_stats(&[], 0.0).unwrap();
    assert!(tails.value_at_risk.is_nan() && tails.expected_shortfall.is_nan());
    assert!(tails.tail_ratio.is_nan());
}

#[test]
fn gain_to_pain_is_the_sum_of_months_over_the_losing_months() {
    // Sum 0.05 over the magnitude of -0.02 - 0.01.
    assert_close(
        gain_to_pain(&[0.05, -0.02, 0.03, -0.01]).unwrap(),
        0.05 / 0.03,
    );
    assert_eq!(gain_to_pain(&[0.01, 0.02]).unwrap(), f64::INFINITY);
    assert!(gain_to_pain(&[0.0, 0.0]).unwrap().is_nan());
    assert!(gain_to_pain(&[]).unwrap().is_nan());
    assert!(gain_to_pain(&[f64::NAN]).is_err());
}
