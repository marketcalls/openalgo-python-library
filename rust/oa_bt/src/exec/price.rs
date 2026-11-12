//! Prices on the tick grid.
//!
//! Two rules, both computed on whole price units so that floating-point noise can never
//! decide a tie:
//!
//! - **Levels** (stops, targets, trails, roll-shifted levels) are rounded to the nearest
//!   tick, and an exact half tick rounds up (decision SPEC-03).
//! - **Slipped fills** are rounded to the tick away from the trader: a buy up, a sell
//!   down, so a slipped price is never better than the slippage allows (SPEC-27).
//!
//! A price unit is a hundredth of a paisa (`PRICE_UNITS_PER_RUPEE`). Every tick in use
//! is a whole number of units: the smallest, 0.0025 rupees on currency derivatives, is
//! 25 units.

use crate::model::Side;

use super::error::ExecError;

/// Price units per rupee. One unit is a hundredth of a paisa.
pub const PRICE_UNITS_PER_RUPEE: f64 = 10_000.0;

/// Largest price, in rupees, the unit arithmetic accepts. It is far above any traded
/// price and far below the `i64` range once multiplied by `PRICE_UNITS_PER_RUPEE`.
pub const MAX_PRICE_RUPEES: f64 = 1.0e12;

/// How far, in units, a tick may sit from a whole number of units and still count as
/// whole: absorbs the binary error of a decimal such as 0.05, nothing more.
const WHOLE_UNIT_TOLERANCE: f64 = 1.0e-6;

fn to_units(what: &'static str, rupees: f64) -> Result<i64, ExecError> {
    if !rupees.is_finite() || !(0.0..=MAX_PRICE_RUPEES).contains(&rupees) {
        return Err(ExecError::InvalidPrice {
            what,
            value: rupees,
        });
    }
    Ok((rupees * PRICE_UNITS_PER_RUPEE).round() as i64)
}

/// The tick size in price units.
///
/// Errors: `InvalidTick` unless the tick is positive, finite and a whole number of
/// price units.
pub fn tick_units(tick_rupees: f64) -> Result<i64, ExecError> {
    let invalid = ExecError::InvalidTick { tick: tick_rupees };
    let units = to_units("tick size", tick_rupees).map_err(|_| invalid.clone())?;
    let whole = (tick_rupees * PRICE_UNITS_PER_RUPEE - units as f64).abs() < WHOLE_UNIT_TOLERANCE;
    if units > 0 && whole {
        Ok(units)
    } else {
        Err(invalid)
    }
}

fn to_rupees(units: i64) -> f64 {
    units as f64 / PRICE_UNITS_PER_RUPEE
}

/// A level rounded to the nearest tick; an exact half tick rounds up (SPEC-03).
///
/// Units: rupees in, rupees out. Errors: `InvalidPrice` for a negative, infinite, NaN
/// or out-of-range price; `InvalidTick` for an unusable tick.
pub fn round_to_tick(price_rupees: f64, tick_rupees: f64) -> Result<f64, ExecError> {
    let price = to_units("price", price_rupees)?;
    let tick = tick_units(tick_rupees)?;
    // floor((2p + t) / 2t) is round-half-up of p / t, exact in integers.
    let ticks = (2 * price + tick) / (2 * tick);
    Ok(to_rupees(ticks * tick))
}

/// A slipped fill price rounded to the tick away from the trader: up for a buy, down
/// for a sell (SPEC-27). A price already on the grid is returned unchanged.
///
/// Units: rupees in, rupees out. Errors: as `round_to_tick`.
pub fn round_away_from_trader(
    price_rupees: f64,
    tick_rupees: f64,
    side: Side,
) -> Result<f64, ExecError> {
    let price = to_units("price", price_rupees)?;
    let tick = tick_units(tick_rupees)?;
    let ticks = match side {
        Side::Buy => (price + tick - 1) / tick,
        Side::Sell => price / tick,
    };
    Ok(to_rupees(ticks * tick))
}

/// True when a price is a whole number of ticks. Errors: as `round_to_tick`.
pub fn is_on_tick(price_rupees: f64, tick_rupees: f64) -> Result<bool, ExecError> {
    Ok(to_units("price", price_rupees)? % tick_units(tick_rupees)? == 0)
}

/// A price in whole price units, for comparisons that binary noise must not flip (a bar
/// that touches a limit of 1410.75 touches it, whatever the last bit says). Only for
/// prices already checked where they entered the run (bars, limits, triggers), which
/// lie far inside the `i64` range.
pub fn whole_units(price_rupees: f64) -> i64 {
    (price_rupees * PRICE_UNITS_PER_RUPEE).round() as i64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn levels_round_to_the_nearest_tick() {
        // 1504.00 x (1 - 2%) = 1473.92 -> 1473.90 on a 0.05 tick (trace-06b).
        assert_eq!(round_to_tick(1473.92, 0.05), Ok(1473.90));
        assert_eq!(round_to_tick(1473.93, 0.05), Ok(1473.95));
    }

    #[test]
    fn an_exact_half_tick_rounds_up() {
        assert_eq!(round_to_tick(100.025, 0.05), Ok(100.05));
        assert_eq!(round_to_tick(0.00125, 0.0025), Ok(0.0025));
    }

    #[test]
    fn slipped_prices_round_away_from_the_trader() {
        assert_eq!(round_away_from_trader(100.01, 0.05, Side::Buy), Ok(100.05));
        assert_eq!(round_away_from_trader(100.04, 0.05, Side::Sell), Ok(100.0));
        assert_eq!(round_away_from_trader(100.05, 0.05, Side::Buy), Ok(100.05));
    }

    #[test]
    fn unusable_prices_and_ticks_are_errors() {
        assert!(round_to_tick(f64::NAN, 0.05).is_err());
        assert!(round_to_tick(-1.0, 0.05).is_err());
        assert!(round_to_tick(100.0, 0.0).is_err());
        assert!(round_to_tick(100.0, 0.000_01).is_err());
        assert_eq!(is_on_tick(800.5, 0.05), Ok(true));
        assert_eq!(is_on_tick(800.52, 0.05), Ok(false));
    }
}
