//! The standard normal distribution function, for the probabilistic ratios.
//!
//! `normal_cdf` is algorithm 5666 of Hart et al. (Computer Approximations, Wiley, 1968)
//! in the double-precision form given by West (2005), "Better approximations to
//! cumulative normal functions", Wilmott Magazine: a ratio of polynomials in |x| below
//! 10 / sqrt(2) and a continued fraction above it, accurate to about 1e-14 everywhere.

use std::f64::consts::PI;

/// Numerator coefficients of the rational part, highest power first (Hart 5666).
const NUMERATOR: [f64; 7] = [
    3.526_249_659_989_11e-2,
    0.700_383_064_443_688,
    6.373_962_203_531_65,
    33.912_866_078_383,
    112.079_291_497_871,
    221.213_596_169_931,
    220.206_867_912_376,
];

/// Denominator coefficients of the rational part, highest power first (Hart 5666).
const DENOMINATOR: [f64; 8] = [
    8.838_834_764_831_84e-2,
    1.755_667_163_182_64,
    16.064_177_579_207,
    86.780_732_202_946_1,
    296.564_248_779_674,
    637.333_633_378_831,
    793.826_512_519_948,
    440.413_735_824_752,
];

/// |x| below which the rational part is used: 10 / sqrt(2).
const RATIONAL_LIMIT: f64 = 7.071_067_811_865_47;

/// |x| beyond which the tail is below the smallest positive normal f64.
const TAIL_IS_ZERO_BEYOND: f64 = 37.0;

fn polynomial(coefficients: &[f64], x: f64) -> f64 {
    coefficients.iter().fold(0.0, |sum, &c| sum * x + c)
}

/// The probability that a standard normal variable is at most `x`. NaN for NaN.
pub fn normal_cdf(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    let z = x.abs();
    let tail = if z > TAIL_IS_ZERO_BEYOND {
        0.0
    } else if z < RATIONAL_LIMIT {
        (-z * z / 2.0).exp() * polynomial(&NUMERATOR, z) / polynomial(&DENOMINATOR, z)
    } else {
        let mut fraction = z + 0.65;
        for depth in [4.0, 3.0, 2.0, 1.0] {
            fraction = z + depth / fraction;
        }
        (-z * z / 2.0).exp() / fraction / (2.0 * PI).sqrt()
    };
    if x > 0.0 {
        1.0 - tail
    } else {
        tail
    }
}
