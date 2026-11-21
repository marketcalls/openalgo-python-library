//! The daily profit targets of 18-multi-strategy 2.5a (rules 5 and 6): when a day's
//! measure has reached its target.
//!
//! The measure is the daily loss limit's (`daily_loss`): the session day's realized P&L
//! plus the open MTM, on the run's `risk_basis`, positive for a profit. The strategy's
//! target is evaluated where and when its daily loss limit is (step 6 on bar marks in a
//! backtest, on tick marks live); the account's on the account's measure, the strategies'
//! measures of the session day summed (`fund::rules`, and a live session's risk watch).

use crate::portfolio::round_to_paise;

/// True when a day's measure has reached the target: a profit of `target_rupees` or more.
/// Both are compared in whole paise, so floating-point noise cannot decide a target that
/// falls exactly on the measure.
pub fn is_target_reached(measure_rupees: f64, target_rupees: f64) -> bool {
    round_to_paise(measure_rupees) >= round_to_paise(target_rupees)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_profit_at_or_past_the_target_reaches_it() {
        assert!(is_target_reached(5_000.0, 5_000.0));
        assert!(is_target_reached(7_250.5, 5_000.0));
        assert!(!is_target_reached(4_999.99, 5_000.0));
        assert!(!is_target_reached(-6_000.0, 5_000.0));
        // (100.10 - 100.00) x 3 is 0.30 rupees, but just under it in binary floating point;
        // it still reaches a 0.30 target.
        assert!(is_target_reached((100.10_f64 - 100.00) * 3.0, 0.30));
    }
}
