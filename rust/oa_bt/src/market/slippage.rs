//! Slippage on fills (04-execution 4.3, decision SPEC-27).
//!
//! Every MARKET fill, entry or exit, is worse than its reference price (the open, or a
//! level) by the slippage for its own side: a buy pays more, a sell receives less. The
//! slipped price is rounded to the tick away from the trader and may leave the bar's
//! range by the slippage.
//!
//! Built: no slippage, a fixed number of ticks, basis points and percent of price, each
//! with a minimum in ticks. Not built yet: the per-segment default table of 04-execution
//! 4.3 (`SlippageSpec::SegmentDefault`), which is the run default once it exists.

use std::fmt;

use crate::exec::error::ExecError;
use crate::exec::price::round_away_from_trader;
use crate::model::Side;

/// Basis points in one unit (1 bps is 0.01 percent).
const BPS_PER_UNIT: f64 = 10_000.0;

/// How much worse than the reference price a fill is.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SlippageSpec {
    /// No slippage (`slippage=0`).
    Zero,
    /// A fixed number of ticks (`bt.ticks(n)`).
    Ticks(u32),
    /// Basis points of the price, at least `min_ticks` ticks.
    Bps { bps: f64, min_ticks: u32 },
    /// Percent of the price (of the premium for options), at least `min_ticks` ticks.
    PercentOfPrice { percent: f64, min_ticks: u32 },
    /// The segment default of 04-execution 4.3.
    SegmentDefault,
}

/// What went wrong while applying slippage.
#[derive(Clone, Debug, PartialEq)]
pub enum SlippageError {
    /// A negative or non-finite slippage amount.
    InvalidSpec { reason: String },
    /// The reference price or the tick cannot be used.
    Price(ExecError),
    /// A slippage model that is not built yet.
    NotBuilt { feature: &'static str },
}

impl fmt::Display for SlippageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SlippageError::InvalidSpec { reason } => write!(f, "slippage: {reason}"),
            SlippageError::Price(inner) => write!(f, "slippage: {inner}"),
            SlippageError::NotBuilt { feature } => {
                write!(
                    f,
                    "{feature} is not implemented yet (Phase 1 work in progress)"
                )
            }
        }
    }
}

impl std::error::Error for SlippageError {}

fn non_negative(what: &str, value: f64) -> Result<f64, SlippageError> {
    if value.is_finite() && value >= 0.0 {
        Ok(value)
    } else {
        Err(SlippageError::InvalidSpec {
            reason: format!("{what} must be zero or positive, got {value}"),
        })
    }
}

/// The amount of slippage in rupees of price, before rounding.
fn amount_rupees(spec: SlippageSpec, price: f64, tick: f64) -> Result<f64, SlippageError> {
    let floor = |ticks: u32| f64::from(ticks) * tick;
    match spec {
        SlippageSpec::Zero => Ok(0.0),
        SlippageSpec::Ticks(ticks) => Ok(floor(ticks)),
        SlippageSpec::Bps { bps, min_ticks } => {
            Ok((price * non_negative("bps", bps)? / BPS_PER_UNIT).max(floor(min_ticks)))
        }
        SlippageSpec::PercentOfPrice { percent, min_ticks } => {
            Ok((price * non_negative("percent", percent)? / 100.0).max(floor(min_ticks)))
        }
        SlippageSpec::SegmentDefault => Err(SlippageError::NotBuilt {
            feature: "the per-segment default slippage table",
        }),
    }
}

/// The fill price after slippage: worse for `side` by the slippage, rounded to the
/// tick away from the trader.
///
/// Units: rupees. Errors: `InvalidSpec` for a negative amount, `Price` for an unusable
/// price or tick, `NotBuilt` for the segment default.
pub fn slipped_price(
    spec: SlippageSpec,
    side: Side,
    reference_rupees: f64,
    tick_rupees: f64,
) -> Result<f64, SlippageError> {
    let amount = amount_rupees(spec, reference_rupees, tick_rupees)?;
    let worse = reference_rupees + side.sign() * amount;
    round_away_from_trader(worse.max(0.0), tick_rupees, side).map_err(SlippageError::Price)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_slippage_keeps_a_price_on_the_grid() {
        assert_eq!(
            slipped_price(SlippageSpec::Zero, Side::Buy, 800.5, 0.05),
            Ok(800.5)
        );
    }

    #[test]
    fn slippage_is_adverse_for_each_side() {
        let two_ticks = SlippageSpec::Ticks(2);
        assert_eq!(slipped_price(two_ticks, Side::Buy, 100.0, 0.05), Ok(100.1));
        assert_eq!(slipped_price(two_ticks, Side::Sell, 100.0, 0.05), Ok(99.9));
    }

    #[test]
    fn bps_have_a_floor_in_ticks_and_round_away_from_the_trader() {
        // 2 bps of 801.00 is 0.1602: the buyer pays 801.1602, rounded up to 801.20.
        let bps = SlippageSpec::Bps {
            bps: 2.0,
            min_ticks: 1,
        };
        assert_eq!(slipped_price(bps, Side::Buy, 801.0, 0.05), Ok(801.2));
        // On a 10.00 price 2 bps is 0.002, below one tick: the floor of 0.05 applies.
        assert_eq!(slipped_price(bps, Side::Sell, 10.0, 0.05), Ok(9.95));
    }

    #[test]
    fn the_segment_default_is_not_built_yet() {
        let result = slipped_price(SlippageSpec::SegmentDefault, Side::Buy, 100.0, 0.05);
        assert!(matches!(result, Err(SlippageError::NotBuilt { .. })));
    }
}
