//! Ratchets: the trailing stop and the breakeven move (step 4, or each tick).
//!
//! Both work from the favourable extreme, the best price seen since the entry, seeded
//! from the fill average (SPEC-01). The favourable excursion is the extreme less the
//! reference price (a short: the reference less the extreme), measured in whole price
//! units so a comparison at the boundary cannot be flipped by binary noise.
//!
//! Trail, OpenAlgo's `{x, y}` (05-risk-management 3):
//! - continuous (`y = 0`): arms when the excursion reaches `x`; the candidate stop is
//!   the extreme less `x` (a long). The live module's own state may keep a separate
//!   arming threshold, so the plan holds the two distances apart.
//! - stepped (`y > 0`): for every whole `x` of excursion the stop moves `y` from its
//!   anchor, the fixed stop recomputed from the reference (SPEC-12). Without a stop it
//!   does nothing (the live module; `strict_config` refuses it at submission).
//!
//! A candidate is rounded to the tick, never passes the extreme, and is taken only if
//! it tightens the stop (or creates one). Armed is latched; `leg_trail_armed` is
//! written once. Breakeven arms when the excursion reaches its trigger and moves the
//! stop to the reference plus its offset if that tightens it (SPEC-24); with a trail
//! the more protective level is in force.

use crate::model::Side;

use super::super::error::RmsError;
use super::super::rules::{AfterFirstTarget, BreakevenSpec, TrailSpec};
use super::super::vocabulary::{RiskEvent, RiskEventName, RiskScope};
use super::levels::{favour, points, positive, round_level, tighter, units};
use super::types::{BarPrices, BreakevenState, LegContext, LevelInForce, RatchetOutcome};
use super::LegState;

/// A trail with its distances in points, fixed at the entry (SPEC-25).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum TrailPlan {
    /// Arms at `arm_points` of excursion; the stop then sits `distance_points` behind
    /// the extreme. `bt.trail(x, 0)` has both equal to `x`.
    Continuous {
        arm_points: f64,
        distance_points: f64,
    },
    /// Arms at `trigger_points`; the stop is the anchor plus `step_points` for every
    /// whole `trigger_points` of excursion.
    Stepped {
        trigger_points: f64,
        step_points: f64,
    },
}

impl TrailPlan {
    /// The plan of a trail rule on a leg at its entry.
    ///
    /// Errors: `InvalidRule` when a distance is not a positive number of points.
    pub(super) fn resolve(
        spec: TrailSpec,
        reference_rupees: f64,
        held_units: f64,
        context: &LegContext,
    ) -> Result<TrailPlan, RmsError> {
        let x = points(spec.trigger, reference_rupees, held_units, context)?;
        Ok(match spec.step {
            None => TrailPlan::Continuous {
                arm_points: x,
                distance_points: x,
            },
            Some(step) => TrailPlan::Stepped {
                trigger_points: x,
                step_points: points(step, reference_rupees, held_units, context)?,
            },
        })
    }

    fn arm_points(self) -> f64 {
        match self {
            TrailPlan::Continuous { arm_points, .. } => arm_points,
            TrailPlan::Stepped { trigger_points, .. } => trigger_points,
        }
    }
}

/// A leg risk event with its level.
fn leg_event(name: RiskEventName, level_rupees: Option<f64>) -> RiskEvent {
    RiskEvent {
        name,
        scope: RiskScope::Leg,
        level_rupees,
        target_index: None,
        observed_rupees: None,
    }
}

impl LegState {
    /// Update trail and breakeven from a bar's favourable extreme (step 4). The moved
    /// stop is in force from the next slice, so it is never tested against this bar.
    ///
    /// Errors: `InvalidValue` for a price that is not positive and finite;
    /// `InvalidRule` for a level that is not a positive price.
    pub fn ratchet(&mut self, bar: &BarPrices) -> Result<RatchetOutcome, RmsError> {
        self.ratchet_at(bar.favourable(self.context.side))
    }

    /// Observe `best_rupees` as a candidate extreme, then move the trail and the
    /// breakeven.
    pub(super) fn ratchet_at(&mut self, best_rupees: f64) -> Result<RatchetOutcome, RmsError> {
        let best = positive("price", best_rupees)?;
        self.observe(best);
        let before = self.marks();
        let mut events = Vec::new();
        events.extend(self.ratchet_trail()?);
        events.extend(self.ratchet_breakeven()?);
        Ok(RatchetOutcome {
            levels_moved: self.marks() != before,
            events,
        })
    }

    /// The stop and the armed states: what makes a row of the position history.
    fn marks(&self) -> (Option<i64>, Option<bool>, Option<bool>) {
        (
            self.stop.map(|stop| units(stop.level_rupees)),
            self.trail.map(|trail| trail.armed),
            self.breakeven.map(|breakeven| breakeven.armed),
        )
    }

    /// Take `price` into the favourable extreme.
    fn observe(&mut self, price: f64) {
        let better = match self.context.side {
            Side::Buy => price > self.extreme_rupees,
            Side::Sell => price < self.extreme_rupees,
        };
        if better {
            self.extreme_rupees = price;
        }
        if let Some(trail) = self.trail.as_mut() {
            trail.favourable_extreme_rupees = self.extreme_rupees;
        }
    }

    /// The favourable excursion in price units (negative when the extreme is worse than
    /// the reference, as after an add above it).
    fn excursion_units(&self) -> i64 {
        let gap = units(self.extreme_rupees) - units(self.reference_rupees);
        match self.context.side {
            Side::Buy => gap,
            Side::Sell => -gap,
        }
    }

    /// Set the stop to `level` if that tightens it. True when it moved.
    pub(super) fn tighten_stop(&mut self, level: f64) -> bool {
        let current = self.stop.map(|stop| stop.level_rupees);
        if !tighter(self.context.side, level, current) {
            return false;
        }
        self.stop = Some(LevelInForce {
            level_rupees: level,
            quantity_units: self.held_units,
        });
        true
    }

    fn stop_level(&self) -> Option<f64> {
        self.stop.map(|stop| stop.level_rupees)
    }

    fn ratchet_trail(&mut self) -> Result<Option<RiskEvent>, RmsError> {
        let (Some(plan), Some(trail)) = (self.trail_plan, self.trail) else {
            return Ok(None);
        };
        let excursion = self.excursion_units();
        let reached = excursion > 0 && excursion >= units(plan.arm_points());
        if !trail.armed && !reached {
            return Ok(None);
        }
        if let Some(candidate) = self.trail_candidate(plan, excursion)? {
            self.tighten_stop(candidate);
        }
        if trail.armed {
            return Ok(None);
        }
        if let Some(state) = self.trail.as_mut() {
            state.armed = true;
        }
        Ok(Some(leg_event(
            RiskEventName::LegTrailArmed,
            self.stop_level(),
        )))
    }

    /// The stop the trail asks for, rounded and kept behind the extreme; `None` for a
    /// stepped trail with no anchor or no whole step yet.
    fn trail_candidate(&self, plan: TrailPlan, excursion: i64) -> Result<Option<f64>, RmsError> {
        let toward = favour(self.context.side);
        let raw = match plan {
            TrailPlan::Continuous {
                distance_points, ..
            } => self.extreme_rupees - toward * distance_points,
            TrailPlan::Stepped {
                trigger_points,
                step_points,
            } => {
                let steps = excursion / units(trigger_points).max(1);
                match self.anchor_rupees {
                    Some(anchor) if steps > 0 => anchor + toward * step_points * steps as f64,
                    _ => return Ok(None),
                }
            }
        };
        let rounded = round_level(raw, &self.context, "trailing stop")?;
        let behind_extreme = match self.context.side {
            Side::Buy => rounded.min(self.extreme_rupees),
            Side::Sell => rounded.max(self.extreme_rupees),
        };
        Ok(Some(behind_extreme))
    }

    fn ratchet_breakeven(&mut self) -> Result<Option<RiskEvent>, RmsError> {
        let Some(state) = self.breakeven else {
            return Ok(None);
        };
        let trigger = units(state.trigger_price_rupees) - units(self.reference_rupees);
        let trigger = match self.context.side {
            Side::Buy => trigger,
            Side::Sell => -trigger,
        };
        if !state.armed && self.excursion_units() < trigger {
            return Ok(None);
        }
        self.tighten_stop(state.level_rupees);
        if state.armed {
            return Ok(None);
        }
        if let Some(state) = self.breakeven.as_mut() {
            state.armed = true;
        }
        Ok(Some(leg_event(
            RiskEventName::LegBreakevenArmed,
            self.stop_level(),
        )))
    }

    /// The breakeven state of `spec` at the current reference, keeping `armed`.
    ///
    /// Errors: `InvalidRule` for a distance or a level that cannot be used.
    pub(super) fn breakeven_state(
        &self,
        spec: BreakevenSpec,
        armed: bool,
    ) -> Result<BreakevenState, RmsError> {
        let reference = self.reference_rupees;
        let held = self.held_units;
        let toward = favour(self.context.side);
        let trigger = points(spec.trigger, reference, held, &self.context)?;
        let offset = match spec.offset {
            Some(offset) => points(offset, reference, held, &self.context)?,
            None => 0.0,
        };
        Ok(BreakevenState {
            armed,
            trigger_price_rupees: reference + toward * trigger,
            level_rupees: round_level(reference + toward * offset, &self.context, "breakeven")?,
        })
    }

    /// After a target fill: `after_first_target="breakeven"` moves the stop to the
    /// average price on the first target fill, if that tightens it (or creates one).
    /// Returns `leg_breakeven_armed` with the new stop when it moved. The stop is in
    /// force from the next slice (the leg's levels are not tested again in this one).
    ///
    /// Errors: `InvalidRule` for a level that is not a positive price.
    pub fn after_target_fill(&mut self) -> Result<Option<RiskEvent>, RmsError> {
        if self.rules.after_first_target != AfterFirstTarget::Breakeven
            || self.after_first_target_done
        {
            return Ok(None);
        }
        self.after_first_target_done = true;
        let level = round_level(self.reference_rupees, &self.context, "breakeven")?;
        if !self.tighten_stop(level) {
            return Ok(None);
        }
        Ok(Some(leg_event(
            RiskEventName::LegBreakevenArmed,
            Some(level),
        )))
    }
}
