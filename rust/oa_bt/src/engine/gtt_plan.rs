//! Which GTTs hold a positional leg's levels, and when a GTT is modified.
//!
//! The leg state (`rms::LegState`) stays the one source of a CNC or NRML position's
//! levels: stop, targets, trail and breakeven are computed exactly as for an
//! engine-held MIS leg. This module turns those levels into GTTs (decision GTT-11) and
//! decides when a changed level is sent as a modification (decision GTT-3).

use crate::exec::gtt::child_limit;
use crate::exec::price::{round_to_tick, tick_units, whole_units};
use crate::exec::{Gtt, GttKind, GttLeg, GttLegRole, GttPolicy, GttSlot, TriggerDirection};
use crate::model::Side;
use crate::rms::LegState;

use super::error::KernelError;

/// One GTT a positional leg needs: what it holds, its quantity and its levels (rupees,
/// as the leg has them).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct PlannedGtt {
    pub(super) slot: GttSlot,
    pub(super) quantity_units: f64,
    pub(super) stop_rupees: Option<f64>,
    pub(super) target_rupees: Option<f64>,
}

/// The GTTs a positional leg needs, by decision GTT-11 reading A: one GTT per working
/// target with its own quantity (an OCO with a stop leg for that quantity when the leg
/// has a stop, else a SINGLE target), plus a SINGLE stop GTT for the remainder the
/// targets do not cover. A stop alone is one SINGLE GTT for the whole position; one
/// target for the whole position with a stop is one OCO (04-execution 4.4 item 2).
pub(super) fn protective_plan(leg: &LegState) -> Vec<PlannedGtt> {
    let stop = leg.stop.map(|stop| stop.level_rupees);
    let mut plan: Vec<PlannedGtt> = leg
        .targets
        .iter()
        .filter(|target| target.is_working())
        .map(|target| PlannedGtt {
            slot: GttSlot::Target(target.index),
            quantity_units: target.quantity_units,
            stop_rupees: stop,
            target_rupees: Some(target.level_rupees),
        })
        .collect();
    if stop.is_some() && leg.remainder_units > 0.0 {
        plan.push(PlannedGtt {
            slot: GttSlot::Remainder,
            quantity_units: leg.remainder_units,
            stop_rupees: stop,
            target_rupees: None,
        });
    }
    plan
}

/// How the child orders of a position's GTTs are priced.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct ChildTerms {
    /// The protected position's side: `Buy` for a long.
    pub(super) position_side: Side,
    /// The instrument's tick, rupees.
    pub(super) tick_rupees: f64,
    /// The segment's child buffer, percent (GTT-1).
    pub(super) buffer_percent: f64,
}

impl ChildTerms {
    /// The side of every child order: the opposite of the position.
    pub(super) fn child_side(&self) -> Side {
        match self.position_side {
            Side::Buy => Side::Sell,
            Side::Sell => Side::Buy,
        }
    }

    /// One leg: its trigger on the tick grid (a level is rounded to the nearest tick,
    /// SPEC-03, so a GTT is a valid order even with `round_levels` off), reached from
    /// the side the role implies, and its child limit beyond it.
    fn leg(&self, role: GttLegRole, level_rupees: f64) -> Result<GttLeg, KernelError> {
        let trigger = round_to_tick(level_rupees, self.tick_rupees)?;
        let falls_to_it = (role == GttLegRole::Stop) == (self.position_side == Side::Buy);
        let direction = if falls_to_it {
            TriggerDirection::Falling
        } else {
            TriggerDirection::Rising
        };
        let child = self.child_side();
        Ok(GttLeg {
            role,
            trigger_rupees: trigger,
            direction,
            child_limit_rupees: child_limit(trigger, child, self.buffer_percent, self.tick_rupees)?,
        })
    }

    /// The legs of a planned GTT: the stop, then the target.
    ///
    /// Errors: `Exec` for a level or tick that is not a usable price.
    pub(super) fn legs(&self, plan: &PlannedGtt) -> Result<Vec<GttLeg>, KernelError> {
        let mut legs = Vec::with_capacity(2);
        if let Some(stop) = plan.stop_rupees {
            legs.push(self.leg(GttLegRole::Stop, stop)?);
        }
        if let Some(target) = plan.target_rupees {
            legs.push(self.leg(GttLegRole::Target, target)?);
        }
        Ok(legs)
    }
}

/// SINGLE for one leg, OCO for a stop with a target.
pub(super) fn kind_of(legs: &[GttLeg]) -> GttKind {
    if legs.len() > 1 {
        GttKind::Oco
    } else {
        GttKind::Single
    }
}

/// Whether the GTT in force must be modified to `legs` and `quantity_units` in base
/// bar `slice`, by decision GTT-3 reading A: at most once per base bar, and only when
/// the quantity changes or a trigger moves by at least `min_modify_ticks` ticks (a leg
/// that appears or goes is a move).
///
/// Errors: `Exec` for an unusable tick.
pub(super) fn needs_modification(
    gtt: &Gtt,
    legs: &[GttLeg],
    quantity_units: f64,
    slice: u32,
    policy: &GttPolicy,
    tick_rupees: f64,
) -> Result<bool, KernelError> {
    if gtt.last_modified_slice == Some(slice) {
        return Ok(false);
    }
    if whole_units(gtt.quantity_units) != whole_units(quantity_units) {
        return Ok(true);
    }
    let least_move = i64::from(policy.min_modify_ticks) * tick_units(tick_rupees)?;
    let roles = [GttLegRole::Stop, GttLegRole::Target];
    Ok(roles.into_iter().any(|role| {
        let now = gtt.trigger(role).map(whole_units);
        let wanted = legs
            .iter()
            .find(|leg| leg.role == role)
            .map(|leg| whole_units(leg.trigger_rupees));
        match (now, wanted) {
            (Some(now), Some(wanted)) => (wanted - now).abs() >= least_move,
            (None, None) => false,
            _ => true,
        }
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exec::{GttId, GttPurpose, GttStatus, InstrumentId};
    use crate::model::Product;
    use crate::rms::{Distance, LegContext, LegRules, TargetSpec};

    const LONG_CASH: ChildTerms = ChildTerms {
        position_side: Side::Buy,
        tick_rupees: 0.05,
        buffer_percent: 1.0,
    };

    fn stop_gtt(trigger: f64, quantity: f64, last_modified: Option<u32>) -> Gtt {
        let plan = PlannedGtt {
            slot: GttSlot::Remainder,
            quantity_units: quantity,
            stop_rupees: Some(trigger),
            target_rupees: None,
        };
        Gtt {
            id: GttId(1),
            instrument: InstrumentId(0),
            product: Product::Nrml,
            side: Side::Sell,
            quantity_units: quantity,
            kind: GttKind::Single,
            legs: LONG_CASH.legs(&plan).unwrap_or_default(),
            purpose: GttPurpose::Protect(GttSlot::Remainder),
            status: GttStatus::Active,
            created_on: 0,
            eligible_from: 0,
            last_modified_slice: last_modified,
            fired: None,
        }
    }

    fn stop_legs(trigger: f64) -> Vec<GttLeg> {
        let plan = PlannedGtt {
            slot: GttSlot::Remainder,
            quantity_units: 75.0,
            stop_rupees: Some(trigger),
            target_rupees: None,
        };
        LONG_CASH.legs(&plan).unwrap_or_default()
    }

    /// A long leg of `units` at 1500.00 with `rules`, venue quantities, tick 0.05.
    fn leg(rules: LegRules, units: f64) -> LegState {
        let context = LegContext {
            side: Side::Buy,
            tick_rupees: 0.05,
            multiplier: 1.0,
            round_levels: true,
            lot_size_units: 1,
            fractional: false,
        };
        LegState::open(rules, context, 1500.0, units).unwrap_or_else(|e| panic!("{e}"))
    }

    fn plan_of(rules: LegRules, units: f64) -> Vec<(GttSlot, f64, Option<f64>, Option<f64>)> {
        protective_plan(&leg(rules, units))
            .into_iter()
            .map(|p| (p.slot, p.quantity_units, p.stop_rupees, p.target_rupees))
            .collect()
    }

    #[test]
    fn a_stop_alone_is_planned_as_one_stop_for_the_whole_position() {
        let rules = LegRules {
            stop: Some(Distance::Points(50.0)),
            ..LegRules::default()
        };
        assert_eq!(
            plan_of(rules, 10.0),
            [(GttSlot::Remainder, 10.0, Some(1450.0), None)]
        );
    }

    #[test]
    fn a_stop_with_one_target_is_planned_as_one_oco_for_the_whole_position() {
        let rules = LegRules {
            stop: Some(Distance::Points(50.0)),
            target: Some(Distance::Points(30.0)),
            ..LegRules::default()
        };
        assert_eq!(
            plan_of(rules, 10.0),
            [(GttSlot::Target(0), 10.0, Some(1450.0), Some(1530.0))]
        );
    }

    #[test]
    fn several_targets_are_planned_one_each_with_a_stop_for_the_remainder() {
        // 100 units: 50 to the first target, 30 to the second, 20 left for the stop.
        let rules = LegRules {
            stop: Some(Distance::Points(50.0)),
            targets: vec![
                TargetSpec {
                    distance: Distance::Points(20.0),
                    fraction: 0.5,
                },
                TargetSpec {
                    distance: Distance::Points(40.0),
                    fraction: 0.3,
                },
            ],
            ..LegRules::default()
        };
        assert_eq!(
            plan_of(rules, 100.0),
            [
                (GttSlot::Target(0), 50.0, Some(1450.0), Some(1520.0)),
                (GttSlot::Target(1), 30.0, Some(1450.0), Some(1540.0)),
                (GttSlot::Remainder, 20.0, Some(1450.0), None),
            ]
        );
    }

    #[test]
    fn a_target_without_a_stop_is_planned_as_a_single_target() {
        let rules = LegRules {
            target: Some(Distance::Points(30.0)),
            ..LegRules::default()
        };
        assert_eq!(
            plan_of(rules, 10.0),
            [(GttSlot::Target(0), 10.0, None, Some(1530.0))]
        );
    }

    #[test]
    fn a_long_stop_falls_to_its_trigger_and_its_child_sells_below_it() {
        // Trace 08a: stop 25175.00 on a long future, child 1 percent below: 24923.25.
        let legs = stop_legs(25_175.0);
        assert_eq!(legs.len(), 1);
        assert_eq!(legs[0].direction, TriggerDirection::Falling);
        assert_eq!(legs[0].child_limit_rupees, 24_923.25);
    }

    #[test]
    fn a_short_target_falls_to_its_trigger_and_its_child_buys_above_it() {
        let short = ChildTerms {
            position_side: Side::Sell,
            ..LONG_CASH
        };
        let plan = PlannedGtt {
            slot: GttSlot::Target(0),
            quantity_units: 10.0,
            stop_rupees: Some(110.0),
            target_rupees: Some(90.0),
        };
        let legs = short.legs(&plan).unwrap_or_default();
        let target = legs[1];
        assert_eq!(legs[0].direction, TriggerDirection::Rising);
        assert_eq!(target.direction, TriggerDirection::Falling);
        // 1 percent above 90.00 is 90.90.
        assert_eq!(target.child_limit_rupees, 90.9);
    }

    #[test]
    fn a_trail_that_moves_the_stop_a_tick_modifies_the_gtt() {
        // Trace 08a: the trail moves the stop from 24860.00 to 25080.00.
        let policy = GttPolicy::default();
        let gtt = stop_gtt(24_860.0, 75.0, None);
        let moved = needs_modification(&gtt, &stop_legs(25_080.0), 75.0, 2, &policy, 0.05);
        assert_eq!(moved, Ok(true));
        let one_tick = needs_modification(&gtt, &stop_legs(24_860.05), 75.0, 2, &policy, 0.05);
        assert_eq!(one_tick, Ok(true));
    }

    #[test]
    fn an_unchanged_level_and_quantity_send_no_modification() {
        let policy = GttPolicy::default();
        let gtt = stop_gtt(24_860.0, 75.0, None);
        let same = needs_modification(&gtt, &stop_legs(24_860.0), 75.0, 2, &policy, 0.05);
        assert_eq!(same, Ok(false));
    }

    #[test]
    fn a_gtt_is_modified_at_most_once_per_base_bar() {
        let policy = GttPolicy::default();
        let gtt = stop_gtt(24_860.0, 75.0, Some(2));
        let again = needs_modification(&gtt, &stop_legs(25_080.0), 75.0, 2, &policy, 0.05);
        assert_eq!(again, Ok(false));
    }

    #[test]
    fn a_new_quantity_modifies_the_gtt_whatever_the_levels() {
        let policy = GttPolicy::default();
        let gtt = stop_gtt(24_860.0, 75.0, None);
        let added = needs_modification(&gtt, &stop_legs(24_860.0), 150.0, 2, &policy, 0.05);
        assert_eq!(added, Ok(true));
    }
}
