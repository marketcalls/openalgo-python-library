//! The GTT a positional leg needs, and when it is modified.
//!
//! The leg state (`rms::LegState`) stays the one source of a CNC or NRML position's
//! levels: stop, targets, trail and breakeven are computed exactly as for an engine-held
//! MIS leg. Only the stop is held at the broker, as one SINGLE GTT for all the leg's open
//! units; the targets stay with the engine, tested on every session's bars as an MIS
//! leg's are (design 20-positional-live P6: OpenAlgo's OCO GTT behaves differently from
//! one broker to the next, so none is used). This module turns the leg's stop into that
//! GTT and decides when a changed stop or quantity is sent as a modification (decision
//! GTT-3).

use crate::exec::gtt::child_limit;
use crate::exec::price::{round_to_tick, tick_units, whole_units};
use crate::exec::{Gtt, GttLeg, GttLegRole, GttPolicy, TriggerDirection};
use crate::model::Side;
use crate::rms::LegState;

use super::error::KernelError;

/// The stop GTT a positional leg needs: its quantity and its level (rupees, as the leg
/// has it).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct PlannedStop {
    pub(super) quantity_units: f64,
    pub(super) stop_rupees: f64,
}

/// The stop GTT a positional leg needs: one SINGLE GTT at the leg's stop for all its open
/// units, so a target's partial exit cuts it to the units left (design 20-positional-live
/// P6, 04-execution 4.4 item 2). None for a leg without a stop.
pub(super) fn protective_plan(leg: &LegState) -> Option<PlannedStop> {
    leg.stop.map(|stop| PlannedStop {
        quantity_units: leg.held_units(),
        stop_rupees: stop.level_rupees,
    })
}

/// How the child order of a position's stop GTT is priced.
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
    /// The side of the child order: the opposite of the position.
    pub(super) fn child_side(&self) -> Side {
        match self.position_side {
            Side::Buy => Side::Sell,
            Side::Sell => Side::Buy,
        }
    }

    /// The stop's trigger on the tick grid (a level is rounded to the nearest tick,
    /// SPEC-03, so a GTT is a valid order even with `round_levels` off), reached by a fall
    /// for a long and a rise for a short, and its child limit beyond it.
    ///
    /// Errors: `Exec` for a level or tick that is not a usable price.
    pub(super) fn stop(&self, stop_rupees: f64) -> Result<GttLeg, KernelError> {
        let trigger = round_to_tick(stop_rupees, self.tick_rupees)?;
        let direction = match self.position_side {
            Side::Buy => TriggerDirection::Falling,
            Side::Sell => TriggerDirection::Rising,
        };
        let child = self.child_side();
        Ok(GttLeg {
            role: GttLegRole::Stop,
            trigger_rupees: trigger,
            direction,
            child_limit_rupees: child_limit(trigger, child, self.buffer_percent, self.tick_rupees)?,
        })
    }
}

/// Whether the GTT in force must be modified to `leg` and `quantity_units` in base bar
/// `slice`, by decision GTT-3 reading A: at most once per base bar, and only when the
/// quantity changes or the trigger moves by at least `min_modify_ticks` ticks.
///
/// Errors: `Exec` for an unusable tick.
pub(super) fn needs_modification(
    gtt: &Gtt,
    leg: &GttLeg,
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
    let moved = whole_units(leg.trigger_rupees) - whole_units(gtt.leg.trigger_rupees);
    Ok(moved.abs() >= least_move)
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
        Gtt {
            id: GttId(1),
            instrument: InstrumentId(0),
            product: Product::Nrml,
            side: Side::Sell,
            quantity_units: quantity,
            leg: stop_leg(trigger),
            purpose: GttPurpose::Protect,
            status: GttStatus::Active,
            created_on: 0,
            eligible_from: 0,
            last_modified_slice: last_modified,
        }
    }

    fn stop_leg(trigger: f64) -> GttLeg {
        LONG_CASH.stop(trigger).unwrap_or_else(|e| panic!("{e}"))
    }

    /// A long leg of `units` at 1000.00 with `rules`, venue quantities, tick 0.05.
    fn leg(rules: LegRules, units: f64) -> LegState {
        let context = LegContext {
            side: Side::Buy,
            tick_rupees: 0.05,
            multiplier: 1.0,
            round_levels: true,
            lot_size_units: 1,
            fractional: false,
        };
        LegState::open(rules, context, 1000.0, units).unwrap_or_else(|e| panic!("{e}"))
    }

    fn plan_of(rules: LegRules) -> Option<(f64, f64)> {
        protective_plan(&leg(rules, 100.0)).map(|p| (p.quantity_units, p.stop_rupees))
    }

    fn stop_50() -> LegRules {
        LegRules {
            stop: Some(Distance::Points(50.0)),
            ..LegRules::default()
        }
    }

    #[test]
    fn a_stop_alone_is_planned_as_one_stop_for_the_whole_position() {
        assert_eq!(plan_of(stop_50()), Some((100.0, 950.0)));
    }

    #[test]
    fn a_stop_with_a_target_is_planned_as_the_stop_alone_for_the_whole_position() {
        // 100 shares at 1000.00, stop 950.00, target 1100.00: the target is the engine's.
        let rules = LegRules {
            target: Some(Distance::Points(100.0)),
            ..stop_50()
        };
        assert_eq!(plan_of(rules), Some((100.0, 950.0)));
    }

    #[test]
    fn several_targets_leave_the_stop_for_every_open_unit() {
        let half = |points| TargetSpec {
            distance: Distance::Points(points),
            fraction: 0.5,
        };
        let rules = LegRules {
            targets: vec![half(100.0), half(150.0)],
            ..stop_50()
        };
        assert_eq!(plan_of(rules), Some((100.0, 950.0)));
    }

    #[test]
    fn after_a_target_fills_the_stop_is_planned_for_the_units_left() {
        // Two targets of 50: once the first sells 50, the stop covers the 50 left.
        let half = |points| TargetSpec {
            distance: Distance::Points(points),
            fraction: 0.5,
        };
        let rules = LegRules {
            targets: vec![half(100.0), half(150.0)],
            ..stop_50()
        };
        let mut state = leg(rules, 100.0);
        assert_eq!(state.on_reduce(50.0, Some(0)), Ok(()));
        let plan = protective_plan(&state).map(|p| (p.quantity_units, p.stop_rupees));
        assert_eq!(plan, Some((50.0, 950.0)));
    }

    #[test]
    fn a_target_without_a_stop_needs_no_gtt() {
        let rules = LegRules {
            target: Some(Distance::Points(100.0)),
            ..LegRules::default()
        };
        assert_eq!(plan_of(rules), None);
    }

    #[test]
    fn a_long_stop_falls_to_its_trigger_and_its_child_sells_below_it() {
        // Trace 08a: stop 25175.00 on a long future, child 1 percent below: 24923.25.
        let stop = stop_leg(25_175.0);
        assert_eq!(stop.role, GttLegRole::Stop);
        assert_eq!(stop.direction, TriggerDirection::Falling);
        assert_eq!(stop.child_limit_rupees, 24_923.25);
    }

    #[test]
    fn a_short_stop_rises_to_its_trigger_and_its_child_buys_above_it() {
        let short = ChildTerms {
            position_side: Side::Sell,
            ..LONG_CASH
        };
        let stop = short.stop(1050.0).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(stop.direction, TriggerDirection::Rising);
        // 1 percent above 1050.00 is 1060.50.
        assert_eq!(stop.child_limit_rupees, 1060.5);
    }

    #[test]
    fn a_trail_that_moves_the_stop_a_tick_modifies_the_gtt() {
        // Trace 08a: the trail moves the stop from 24860.00 to 25080.00.
        let policy = GttPolicy::default();
        let gtt = stop_gtt(24_860.0, 75.0, None);
        let moved = needs_modification(&gtt, &stop_leg(25_080.0), 75.0, 2, &policy, 0.05);
        assert_eq!(moved, Ok(true));
        let one_tick = needs_modification(&gtt, &stop_leg(24_860.05), 75.0, 2, &policy, 0.05);
        assert_eq!(one_tick, Ok(true));
    }

    #[test]
    fn an_unchanged_level_and_quantity_send_no_modification() {
        let policy = GttPolicy::default();
        let gtt = stop_gtt(24_860.0, 75.0, None);
        let same = needs_modification(&gtt, &stop_leg(24_860.0), 75.0, 2, &policy, 0.05);
        assert_eq!(same, Ok(false));
    }

    #[test]
    fn a_gtt_is_modified_at_most_once_per_base_bar() {
        let policy = GttPolicy::default();
        let gtt = stop_gtt(24_860.0, 75.0, Some(2));
        let again = needs_modification(&gtt, &stop_leg(25_080.0), 75.0, 2, &policy, 0.05);
        assert_eq!(again, Ok(false));
    }

    #[test]
    fn a_new_quantity_modifies_the_gtt_whatever_the_level() {
        let policy = GttPolicy::default();
        let gtt = stop_gtt(24_860.0, 75.0, None);
        let added = needs_modification(&gtt, &stop_leg(24_860.0), 150.0, 2, &policy, 0.05);
        assert_eq!(added, Ok(true));
        let cut = needs_modification(&gtt, &stop_leg(24_860.0), 25.0, 2, &policy, 0.05);
        assert_eq!(cut, Ok(true));
    }
}
