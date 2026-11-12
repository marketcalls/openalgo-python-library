//! From distances to levels, and price comparisons that floating-point noise cannot
//! flip.
//!
//! A distance becomes points (rupees of price) on the leg it applies to; a level is the
//! reference price plus or minus those points, rounded to the nearest tick unless
//! `round_levels` is off (SPEC-03). Comparisons, excursions and step counts are taken
//! on whole price units, a hundredth of a paisa, the grid `exec::price` rounds on.

use crate::exec::price::{round_to_tick, PRICE_UNITS_PER_RUPEE};
use crate::model::Side;

use super::super::error::RmsError;
use super::super::rules::Distance;
use super::types::LegContext;

/// A price or a distance in whole price units. Inputs are positive finite prices or
/// distances checked where they enter the leg, so the conversion cannot overflow.
pub(super) fn units(rupees: f64) -> i64 {
    (rupees * PRICE_UNITS_PER_RUPEE).round() as i64
}

/// Errors: `InvalidValue` unless `value` is positive and finite.
pub(super) fn positive(what: &'static str, value: f64) -> Result<f64, RmsError> {
    if value.is_finite() && value > 0.0 {
        Ok(value)
    } else {
        Err(RmsError::InvalidValue { what, value })
    }
}

/// The points a distance stands for on a leg whose reference price is
/// `reference_rupees` and which holds `held_units`.
///
/// Percent is of the reference (the fill average, 14-strategy-api 1), ticks are the
/// instrument's, rupees are of the leg's MTM (divided by quantity times multiplier),
/// and an ATR distance is its multiple of the ATR value resolved at the decision.
///
/// Errors: `InvalidRule` when the distance does not come to a positive number of
/// points (an ATR that was never resolved, for example).
pub(super) fn points(
    distance: Distance,
    reference_rupees: f64,
    held_units: f64,
    context: &LegContext,
) -> Result<f64, RmsError> {
    let value = match distance {
        Distance::Points(points) => points,
        Distance::Percent(percent) => reference_rupees * percent / 100.0,
        Distance::Ticks(count) => f64::from(count) * context.tick_rupees,
        Distance::Rupees(rupees) => rupees / (held_units * context.multiplier),
        Distance::Atr {
            multiple,
            atr_rupees,
        } => multiple * atr_rupees,
    };
    if value.is_finite() && value > 0.0 {
        Ok(value)
    } else {
        Err(RmsError::InvalidRule {
            rule: "leg rules",
            reason: format!(
                "{distance:?} comes to {value} points on this leg; a distance must be a \
                 positive number of points"
            ),
        })
    }
}

/// +1 when a higher price favours a leg on `side` (a long), -1 when a lower one does.
pub(super) fn favour(side: Side) -> f64 {
    side.sign()
}

/// A level rounded to the tick unless `round_levels` is off (SPEC-03).
///
/// Errors: `InvalidRule` naming `what` when the level is not a positive price, which
/// happens when a stop or a short's target lies a distance at or beyond the entry
/// price itself.
pub(super) fn round_level(
    level_rupees: f64,
    context: &LegContext,
    what: &'static str,
) -> Result<f64, RmsError> {
    let not_a_price = |level: f64| RmsError::InvalidRule {
        rule: what,
        reason: format!(
            "the level comes to {level} rupees, not a positive price: the distance reaches \
             the entry price; use a smaller distance"
        ),
    };
    if !(level_rupees.is_finite() && level_rupees > 0.0) {
        return Err(not_a_price(level_rupees));
    }
    let placed = if context.round_levels {
        round_to_tick(level_rupees, context.tick_rupees).map_err(|error| RmsError::InvalidRule {
            rule: what,
            reason: error.to_string(),
        })?
    } else {
        level_rupees
    };
    if placed > 0.0 {
        Ok(placed)
    } else {
        Err(not_a_price(placed))
    }
}

/// True when `candidate` is a stop that protects more than `current`: higher for a
/// long, lower for a short. Any stop protects more than none.
pub(super) fn tighter(side: Side, candidate: f64, current: Option<f64>) -> bool {
    match current {
        None => true,
        Some(current) => match side {
            Side::Buy => units(candidate) > units(current),
            Side::Sell => units(candidate) < units(current),
        },
    }
}

/// True when `price` has reached a stop at `level`: at or below it for a long, at or
/// above it for a short (the live module's comparisons).
pub(super) fn stop_reached(side: Side, price: f64, level: f64) -> bool {
    match side {
        Side::Buy => units(price) <= units(level),
        Side::Sell => units(price) >= units(level),
    }
}

/// True when `price` has reached a target at `level`: at or above it for a long, at or
/// below it for a short.
pub(super) fn target_reached(side: Side, price: f64, level: f64) -> bool {
    match side {
        Side::Buy => units(price) >= units(level),
        Side::Sell => units(price) <= units(level),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn context(round_levels: bool) -> LegContext {
        LegContext {
            side: Side::Buy,
            tick_rupees: 0.05,
            multiplier: 1.0,
            round_levels,
            lot_size_units: 1,
            fractional: false,
        }
    }

    #[test]
    fn each_unit_comes_to_points_on_the_leg() {
        let ctx = context(true);
        // 2 percent of 1504.00 is 30.08 points (trace-06b); 3 ticks of 0.05 is 0.15;
        // 500 rupees of MTM on 100 units is 5 points; 2 x an ATR of 3.5 is 7.
        let cases = [
            (Distance::Points(4.0), 4.0),
            (Distance::Percent(2.0), 30.08),
            (Distance::Ticks(3), 0.15),
            (Distance::Rupees(500.0), 5.0),
            (
                Distance::Atr {
                    multiple: 2.0,
                    atr_rupees: 3.5,
                },
                7.0,
            ),
        ];
        for (distance, expected) in cases {
            let got = points(distance, 1504.0, 100.0, &ctx).unwrap_or(f64::NAN);
            assert!((got - expected).abs() < 1e-9, "{distance:?}: {got}");
        }
    }

    #[test]
    fn an_unresolved_atr_is_refused() {
        let unresolved = Distance::Atr {
            multiple: 2.0,
            atr_rupees: f64::NAN,
        };
        assert!(points(unresolved, 100.0, 1.0, &context(true)).is_err());
    }

    #[test]
    fn levels_round_to_the_nearest_tick_only_when_asked() {
        // 1504.00 - 30.08 = 1473.92: 1473.90 on the tick, unchanged without rounding.
        assert_eq!(round_level(1473.92, &context(true), "stop"), Ok(1473.9));
        assert_eq!(round_level(1473.92, &context(false), "stop"), Ok(1473.92));
    }

    #[test]
    fn a_level_at_or_below_zero_is_a_configuration_error() {
        assert!(round_level(-5.0, &context(true), "stop").is_err());
        assert!(round_level(0.01, &context(true), "stop").is_err());
    }

    #[test]
    fn comparisons_include_the_exact_touch() {
        assert!(stop_reached(Side::Buy, 95.0, 95.0));
        assert!(!stop_reached(Side::Buy, 95.05, 95.0));
        assert!(stop_reached(Side::Sell, 105.0, 105.0));
        assert!(target_reached(Side::Buy, 120.0, 120.0));
        assert!(!target_reached(Side::Sell, 80.5, 80.0));
        // 0.1 + 0.2 is 0.30000000000000004 in binary; on price units it is 0.30.
        assert!(stop_reached(Side::Buy, 0.1 + 0.2, 0.3));
    }

    #[test]
    fn a_tighter_stop_is_higher_for_a_long_and_lower_for_a_short() {
        assert!(tighter(Side::Buy, 801.0, Some(800.5)));
        assert!(!tighter(Side::Buy, 800.5, Some(800.5)));
        assert!(tighter(Side::Sell, 93.0, Some(110.0)));
        assert!(tighter(Side::Sell, 93.0, None));
    }
}
