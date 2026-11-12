//! Target levels and quantities (12-position-sizing 3 and 4).
//!
//! - Fractions are of the quantity held when the order with `targets=` filled; a single
//!   `target=` is target 0 with the whole quantity.
//! - Each quantity rounds down to the venue quantity; what is lost stays in the
//!   remainder (SPEC-13). A target whose share rounds to zero is skipped.
//! - An add gives every unfilled target its original fraction of the tranche; the rest
//!   joins the remainder; a filled target is never re-armed (SPEC-08, SPEC-09).
//! - Unfilled levels follow the reference price; a filled target keeps the level it
//!   filled at (SPEC-10).
//! - A partial exit by signal shrinks every working target in proportion to the
//!   position, so working targets never exceed the position.

use super::super::error::RmsError;
use super::super::rules::{LegRules, TargetSpec};
use super::levels::{favour, points, round_level};
use super::types::{LegContext, TargetLevel, TargetState};
use super::LegState;

/// Relative slack below which a quantity counts as a whole number of lots: absorbs the
/// binary error of a product such as 0.29 x 100 (28.999999999999996), nothing more.
const WHOLE_LOT_SLACK: f64 = 1.0e-9;

/// The targets of a rule set with their fractions: `target=` is one target taking the
/// whole quantity, `targets=` keeps its order.
pub(super) fn target_specs(rules: &LegRules) -> Vec<TargetSpec> {
    match rules.target {
        Some(distance) => vec![TargetSpec {
            distance,
            fraction: 1.0,
        }],
        None => rules.targets.clone(),
    }
}

/// `units` rounded down to the venue quantity (whole lots), or kept when fractional
/// quantities are allowed.
pub(super) fn venue_floor(units: f64, context: &LegContext) -> f64 {
    if context.fractional {
        return units;
    }
    let lot = f64::from(context.lot_size_units.max(1));
    let lots = units / lot;
    (lots * (1.0 + WHOLE_LOT_SLACK)).floor() * lot
}

impl LegState {
    /// The level of a target `spec` at the current reference price.
    fn target_level(&self, spec: TargetSpec) -> Result<f64, RmsError> {
        let distance = points(
            spec.distance,
            self.reference_rupees,
            self.held_units,
            &self.context,
        )?;
        let level = self.reference_rupees + favour(self.context.side) * distance;
        round_level(level, &self.context, "target")
    }

    /// The targets of the opening fill, from the quantity it filled.
    pub(super) fn plan_targets(&self) -> Result<Vec<TargetLevel>, RmsError> {
        let mut planned = Vec::new();
        for (index, spec) in target_specs(&self.rules).into_iter().enumerate() {
            planned.push(TargetLevel {
                index: u16::try_from(index).unwrap_or(u16::MAX),
                level_rupees: self.target_level(spec)?,
                quantity_units: venue_floor(spec.fraction * self.held_units, &self.context),
                state: TargetState::Working,
            });
        }
        Ok(planned)
    }

    /// An add: every unfilled target takes its original fraction of the tranche, and
    /// every unfilled level follows the new reference price.
    pub(super) fn share_tranche(&mut self, tranche_units: f64) -> Result<(), RmsError> {
        let specs = target_specs(&self.rules);
        for slot in 0..self.targets.len() {
            let target = self.targets[slot];
            let Some(spec) = specs.get(usize::from(target.index)).copied() else {
                continue;
            };
            if target.state == TargetState::Filled {
                continue;
            }
            let share = venue_floor(spec.fraction * tranche_units, &self.context);
            let level = self.target_level(spec)?;
            let target = &mut self.targets[slot];
            target.quantity_units += share;
            target.level_rupees = level;
        }
        Ok(())
    }

    /// A partial exit by signal from `before_units` to `after_units`: every working
    /// target keeps the same share of the position it had.
    pub(super) fn shrink_targets(&mut self, before_units: f64, after_units: f64) {
        let kept = after_units / before_units;
        let context = self.context;
        for target in self.targets.iter_mut() {
            if target.state == TargetState::Working {
                target.quantity_units = venue_floor(target.quantity_units * kept, &context);
            }
        }
    }

    /// Mark a target filled (its level and units stay as the record of the fill).
    pub(super) fn mark_target_filled(&mut self, index: u16) {
        for target in self.targets.iter_mut() {
            if target.index == index {
                target.state = TargetState::Filled;
            }
        }
    }

    /// Units on targets that can still fill.
    pub(super) fn working_target_units(&self) -> f64 {
        self.targets
            .iter()
            .filter(|target| target.state == TargetState::Working)
            .map(|target| target.quantity_units)
            .sum()
    }

    /// Shift every working target by a roll spread, rounded again.
    pub(super) fn shift_targets(&mut self, spread_rupees: f64) -> Result<(), RmsError> {
        let context = self.context;
        for target in self.targets.iter_mut() {
            if target.state == TargetState::Working {
                target.level_rupees =
                    round_level(target.level_rupees + spread_rupees, &context, "target")?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Side;

    fn context(lot_size_units: u32, fractional: bool) -> LegContext {
        LegContext {
            side: Side::Buy,
            tick_rupees: 0.05,
            multiplier: 1.0,
            round_levels: true,
            lot_size_units,
            fractional,
        }
    }

    #[test]
    fn quantities_round_down_to_whole_lots() {
        // Half of one 75-unit lot rounds to zero (12-position-sizing 3).
        assert_eq!(venue_floor(37.5, &context(75, false)), 0.0);
        assert_eq!(venue_floor(150.0, &context(75, false)), 150.0);
        assert_eq!(venue_floor(33.3, &context(1, false)), 33.0);
    }

    #[test]
    fn binary_noise_does_not_lose_a_share() {
        // 0.29 x 100 is 28.999999999999996 in binary; the target takes 29 shares.
        assert_eq!(venue_floor(0.29 * 100.0, &context(1, false)), 29.0);
    }

    #[test]
    fn fractional_quantities_are_kept() {
        assert_eq!(venue_floor(33.3, &context(1, true)), 33.3);
    }
}
