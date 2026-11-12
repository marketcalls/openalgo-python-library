//! The price-level rules of one leg (one position): its stop, targets, trail and
//! breakeven, and the places the slice sequence touches them.
//!
//! | Call | Step of 03-architecture 4.2 | What it does |
//! |---|---|---|
//! | `LegState::open` | 2, after an entry fill | Levels from the fill average; the favourable extreme seeded from it (SPEC-01) |
//! | `LegState::on_add` | 2, after an add | Stop and unfilled targets recomputed from the new average, in force for step 3 of the same slice (SPEC-08 to SPEC-12) |
//! | `LegState::test_bar` | 3 | Levels against the base bar: a level crossed at the open fills at the open; otherwise the stop beats a target inside the range; every target reached fills, nearest first (SPEC-28, SPEC-33). Legs opened in step 2 are tested (SPEC-02) |
//! | `LegState::on_reduce` | 2 or 3, after a partial exit | Stop quantity shrinks to the position; a filled target is marked; a signal exit shrinks the working targets in proportion |
//! | `LegState::after_target_fill` | 3, after a target fill | `after_first_target="breakeven"` |
//! | `LegState::ratchet` | 4 | Trail and breakeven from this bar's extreme, in force from the next slice; the entry bar's extreme counts (SPEC-01) |
//! | `LegState::shift` | 2, futures roll | Levels, extreme and reference shifted by the roll spread (SPEC-19, SPEC-20) |
//! | `LegState::on_tick` | tick mode | Best price, trail, stop then target on one price: the order of the live module and of the server's 35 risk vectors |
//!
//! Every level refers to the reference price, the position's average price
//! (`reference="avg_price"`, 12-position-sizing 4), shifted by a roll. The stop's
//! quantity is always the position; working targets never exceed it, and the
//! remainder is the position less the working targets. A stop never loosens: every
//! move goes through `tighten_stop`.
//!
//! | Module | Responsibility |
//! |---|---|
//! | `types` | The data a leg reports: levels, targets, trail and breakeven states, hits |
//! | `levels` | Distances to points and levels, tick rounding, comparisons on price units |
//! | `targets` | Target levels and quantities, adds and partial exits |
//! | `ratchet` | Trail and breakeven, `after_first_target` |
//! | `hits` | Testing a bar or a tick |

mod hits;
mod levels;
mod ratchet;
mod targets;
#[cfg(test)]
mod tests;
mod types;

pub use types::{
    BarPrices, BreakevenState, LegContext, LevelHit, LevelInForce, LevelKind, RatchetOutcome,
    TargetLevel, TargetState, TickOutcome, TrailState,
};

use super::error::RmsError;
use super::rules::{Distance, LegRules};
use levels::{favour, points, positive, round_level};
use ratchet::TrailPlan;

/// The levels of one leg.
#[derive(Clone, Debug, PartialEq)]
pub struct LegState {
    /// The rules of the opening order (SPEC-14).
    pub rules: LegRules,
    /// Side, tick, rounding and venue quantity.
    pub context: LegContext,
    /// The reference price of every level: the average price (shifted by a roll).
    pub reference_rupees: f64,
    /// The stop in force.
    pub stop: Option<LevelInForce>,
    /// The targets.
    pub targets: Vec<TargetLevel>,
    /// Units assigned to no working target (they leave by the stop, the trail or a
    /// signal): the position less the working targets.
    pub remainder_units: f64,
    /// The trail, as reported.
    pub trail: Option<TrailState>,
    /// The breakeven move.
    pub breakeven: Option<BreakevenState>,
    /// Units held (the position's absolute quantity).
    held_units: f64,
    /// Best price seen since the entry, rupees (the trail reports it too).
    extreme_rupees: f64,
    /// The trail's distances, fixed at the entry.
    trail_plan: Option<TrailPlan>,
    /// The fixed stop at the current reference: the stepped trail's anchor (SPEC-12).
    anchor_rupees: Option<f64>,
    /// `after_first_target` has acted.
    after_first_target_done: bool,
}

impl LegState {
    /// The leg a fill opened, with levels from its fill average (step 2).
    ///
    /// Errors: `InvalidValue` for an average price or quantity that is not positive;
    /// `InvalidRule` for a distance or a level that cannot be used (a stop at or beyond
    /// zero).
    pub fn open(
        rules: LegRules,
        context: LegContext,
        average_price_rupees: f64,
        quantity_units: f64,
    ) -> Result<LegState, RmsError> {
        let reference = positive("average price", average_price_rupees)?;
        let mut leg = LegState {
            rules,
            context,
            reference_rupees: reference,
            stop: None,
            targets: Vec::new(),
            remainder_units: 0.0,
            trail: None,
            breakeven: None,
            held_units: positive("quantity", quantity_units)?,
            extreme_rupees: reference,
            trail_plan: None,
            anchor_rupees: None,
            after_first_target_done: false,
        };
        leg.anchor_rupees = leg.fixed_stop()?;
        if let Some(anchor) = leg.anchor_rupees {
            leg.tighten_stop(anchor);
        }
        leg.targets = leg.plan_targets()?;
        if let Some(spec) = leg.rules.trail {
            let plan = TrailPlan::resolve(spec, reference, leg.held_units, &context)?;
            leg.trail_plan = Some(plan);
            leg.trail = Some(TrailState {
                armed: false,
                favourable_extreme_rupees: reference,
            });
        }
        if let Some(spec) = leg.rules.breakeven {
            leg.breakeven = Some(leg.breakeven_state(spec, false)?);
        }
        leg.settle();
        Ok(leg)
    }

    /// True when the leg has no rule: it has no level and only follows the quantity.
    pub fn is_plain(&self) -> bool {
        self.rules.is_empty()
    }

    /// Units the leg holds.
    pub fn held_units(&self) -> f64 {
        self.held_units
    }

    /// Check the rules of an add against the position's (SPEC-14): an add may repeat
    /// them, or give none, but may not change them. ATR distances compare by their
    /// multiple, since the ATR value is taken at each decision.
    ///
    /// Errors: `InvalidRule` when the add's rules differ.
    pub fn accepts_add_rules(&self, rules: &LegRules) -> Result<(), RmsError> {
        if rules.is_empty() || rule_shape(rules) == rule_shape(&self.rules) {
            return Ok(());
        }
        Err(RmsError::InvalidRule {
            rule: "leg rules",
            reason: "an add may repeat the position's stop, target, targets, trail, breakeven \
                     and after_first_target, or give none, but may not change them: the \
                     position keeps the rules of the order that opened it"
                .to_string(),
        })
    }

    /// An add filled: the new average price and the units the tranche added. The stop
    /// is the more protective of the current one and the fixed stop recomputed from the
    /// new average; unfilled targets follow the average and take their fractions of
    /// the tranche. In force for step 3 of the same slice (SPEC-11).
    ///
    /// Errors: `InvalidValue` for a price or quantity that is not positive;
    /// `InvalidRule` for a level that cannot be used.
    pub fn on_add(
        &mut self,
        average_price_rupees: f64,
        tranche_units: f64,
    ) -> Result<(), RmsError> {
        self.reference_rupees = positive("average price", average_price_rupees)?;
        self.held_units += positive("tranche quantity", tranche_units)?;
        self.share_tranche(tranche_units)?;
        self.anchor_rupees = self.fixed_stop()?;
        if let Some(anchor) = self.anchor_rupees {
            self.tighten_stop(anchor);
        }
        if let (Some(spec), Some(state)) = (self.rules.breakeven, self.breakeven) {
            let moved = self.breakeven_state(spec, state.armed)?;
            self.breakeven = Some(moved);
            if moved.armed {
                self.tighten_stop(moved.level_rupees);
            }
        }
        self.settle();
        Ok(())
    }

    /// The position was reduced to `remaining_units` (positive): by the target
    /// `filled_target`, or by a signal when `None`.
    ///
    /// Errors: `InvalidValue` for a remaining quantity that is not positive (a closed
    /// position drops its leg instead).
    pub fn on_reduce(
        &mut self,
        remaining_units: f64,
        filled_target: Option<u16>,
    ) -> Result<(), RmsError> {
        let before = self.held_units;
        self.held_units = positive("remaining quantity", remaining_units)?;
        match filled_target {
            Some(index) => self.mark_target_filled(index),
            None => self.shrink_targets(before, self.held_units),
        }
        self.settle();
        Ok(())
    }

    /// The position was rolled: shift every level, the extreme and the reference by
    /// `spread_rupees` (next contract's open minus the near contract's open), rounding
    /// each level again (SPEC-03, SPEC-20).
    ///
    /// Errors: `InvalidRule` for a level that is no longer a positive price.
    pub fn shift(&mut self, spread_rupees: f64) -> Result<(), RmsError> {
        let context = self.context;
        let shift = |level: f64| round_level(level + spread_rupees, &context, "shifted level");
        self.reference_rupees += spread_rupees;
        self.extreme_rupees += spread_rupees;
        if let Some(trail) = self.trail.as_mut() {
            trail.favourable_extreme_rupees = self.extreme_rupees;
        }
        self.anchor_rupees = self.anchor_rupees.map(shift).transpose()?;
        if let Some(stop) = self.stop.as_mut() {
            stop.level_rupees = shift(stop.level_rupees)?;
        }
        if let Some(state) = self.breakeven.as_mut() {
            state.trigger_price_rupees += spread_rupees;
            state.level_rupees = shift(state.level_rupees)?;
        }
        self.shift_targets(spread_rupees)
    }

    /// The fixed stop at the current reference, from the `stop=` rule.
    fn fixed_stop(&self) -> Result<Option<f64>, RmsError> {
        let Some(distance) = self.rules.stop else {
            return Ok(None);
        };
        let points = points(
            distance,
            self.reference_rupees,
            self.held_units,
            &self.context,
        )?;
        let level = self.reference_rupees - favour(self.context.side) * points;
        round_level(level, &self.context, "stop").map(Some)
    }

    /// Keep the quantities consistent: the stop covers the position, the remainder is
    /// the position less the working targets.
    fn settle(&mut self) {
        if let Some(stop) = self.stop.as_mut() {
            stop.quantity_units = self.held_units;
        }
        self.remainder_units = (self.held_units - self.working_target_units()).max(0.0);
    }
}

/// The rules with every resolved ATR value cleared, for comparing an add's rules with
/// the position's.
fn rule_shape(rules: &LegRules) -> LegRules {
    let clear = |distance: Distance| match distance {
        Distance::Atr { multiple, .. } => Distance::Atr {
            multiple,
            atr_rupees: 0.0,
        },
        other => other,
    };
    let mut shape = rules.clone();
    shape.stop = shape.stop.map(clear);
    shape.target = shape.target.map(clear);
    for target in shape.targets.iter_mut() {
        target.distance = clear(target.distance);
    }
    if let Some(trail) = shape.trail.as_mut() {
        trail.trigger = clear(trail.trigger);
        trail.step = trail.step.map(clear);
    }
    if let Some(breakeven) = shape.breakeven.as_mut() {
        breakeven.trigger = clear(breakeven.trigger);
        breakeven.offset = breakeven.offset.map(clear);
    }
    shape
}
