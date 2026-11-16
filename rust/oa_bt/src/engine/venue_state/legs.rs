//! The legs of a live session's saved state (`venue_state`): each open position's leg
//! state whole, the levels it reports and what it keeps besides (`rms::LegMemory`), so
//! a restored leg ratchets, arms and exits exactly as the saved one would.

use crate::portfolio::PositionKey;
use crate::rms::leg::{BreakevenState, LevelInForce, TargetLevel, TargetState, TrailState};
use crate::rms::{LegContext, LegLevels, LegMemory, LegState, TrailPlan};

use super::error::LiveStateError;
use super::requests::{key_fields, read_key, read_rules, read_side, write_rules};
use super::value::{list, map, Read, StateValue};

fn write_context(context: &LegContext) -> StateValue {
    map([
        ("side", context.side.code().into()),
        ("tick_rupees", context.tick_rupees.into()),
        ("multiplier", context.multiplier.into()),
        ("round_levels", context.round_levels.into()),
        ("lot_size_units", context.lot_size_units.into()),
        ("fractional", context.fractional.into()),
    ])
}

fn read_context(read: &Read<'_>) -> Result<LegContext, LiveStateError> {
    Ok(LegContext {
        side: read_side(read, "side")?,
        tick_rupees: read.f64_of("tick_rupees")?,
        multiplier: read.f64_of("multiplier")?,
        round_levels: read.bool_of("round_levels")?,
        lot_size_units: read.whole_of("lot_size_units")?,
        fractional: read.bool_of("fractional")?,
    })
}

fn write_plan(plan: TrailPlan) -> StateValue {
    match plan {
        TrailPlan::Continuous {
            arm_points,
            distance_points,
        } => map([
            ("kind", "continuous".into()),
            ("arm_points", arm_points.into()),
            ("distance_points", distance_points.into()),
        ]),
        TrailPlan::Stepped {
            trigger_points,
            step_points,
        } => map([
            ("kind", "stepped".into()),
            ("trigger_points", trigger_points.into()),
            ("step_points", step_points.into()),
        ]),
    }
}

fn read_plan(read: &Read<'_>) -> Result<TrailPlan, LiveStateError> {
    let kind = read.field("kind")?;
    Ok(match kind.text()? {
        "continuous" => TrailPlan::Continuous {
            arm_points: read.f64_of("arm_points")?,
            distance_points: read.f64_of("distance_points")?,
        },
        "stepped" => TrailPlan::Stepped {
            trigger_points: read.f64_of("trigger_points")?,
            step_points: read.f64_of("step_points")?,
        },
        other => return Err(kind.malformed(format!("{other:?} is not a trail plan"))),
    })
}

fn write_target(target: &TargetLevel) -> StateValue {
    map([
        ("index", target.index.into()),
        ("level_rupees", target.level_rupees.into()),
        ("quantity_units", target.quantity_units.into()),
        ("filled", (target.state == TargetState::Filled).into()),
    ])
}

fn read_target(read: &Read<'_>) -> Result<TargetLevel, LiveStateError> {
    let state = if read.bool_of("filled")? {
        TargetState::Filled
    } else {
        TargetState::Working
    };
    Ok(TargetLevel {
        index: read.whole_of("index")?,
        level_rupees: read.f64_of("level_rupees")?,
        quantity_units: read.f64_of("quantity_units")?,
        state,
    })
}

/// The leg of the position `key`.
pub(super) fn write_leg(key: PositionKey, leg: &LegState) -> StateValue {
    let [instrument, product] = key_fields(key);
    let memory = leg.memory();
    let stop = leg.stop.map(|stop| {
        map([
            ("level_rupees", stop.level_rupees.into()),
            ("quantity_units", stop.quantity_units.into()),
        ])
    });
    let trail = leg.trail.map(|trail| {
        map([
            ("armed", trail.armed.into()),
            (
                "favourable_extreme_rupees",
                trail.favourable_extreme_rupees.into(),
            ),
        ])
    });
    let breakeven = leg.breakeven.map(|be| {
        map([
            ("armed", be.armed.into()),
            ("trigger_price_rupees", be.trigger_price_rupees.into()),
            ("level_rupees", be.level_rupees.into()),
        ])
    });
    map([
        instrument,
        product,
        ("rules", write_rules(&leg.rules)),
        ("context", write_context(&leg.context)),
        ("reference_rupees", leg.reference_rupees.into()),
        ("stop", stop.into()),
        ("targets", list(&leg.targets, write_target)),
        ("remainder_units", leg.remainder_units.into()),
        ("trail", trail.into()),
        ("breakeven", breakeven.into()),
        ("held_units", memory.held_units.into()),
        ("extreme_rupees", memory.extreme_rupees.into()),
        ("trail_plan", memory.trail_plan.map(write_plan).into()),
        ("anchor_rupees", memory.anchor_rupees.into()),
        (
            "after_first_target_done",
            memory.after_first_target_done.into(),
        ),
    ])
}

/// The leg `write_leg` wrote, with its position key.
///
/// Errors: `Malformed` for a field missing or unusable.
pub(super) fn read_leg(
    read: &Read<'_>,
    instruments: usize,
) -> Result<(PositionKey, LegState), LiveStateError> {
    let stop = match read.maybe("stop")? {
        Some(stop) => Some(LevelInForce {
            level_rupees: stop.f64_of("level_rupees")?,
            quantity_units: stop.f64_of("quantity_units")?,
        }),
        None => None,
    };
    let trail = match read.maybe("trail")? {
        Some(trail) => Some(TrailState {
            armed: trail.bool_of("armed")?,
            favourable_extreme_rupees: trail.f64_of("favourable_extreme_rupees")?,
        }),
        None => None,
    };
    let breakeven = match read.maybe("breakeven")? {
        Some(be) => Some(BreakevenState {
            armed: be.bool_of("armed")?,
            trigger_price_rupees: be.f64_of("trigger_price_rupees")?,
            level_rupees: be.f64_of("level_rupees")?,
        }),
        None => None,
    };
    let levels = LegLevels {
        reference_rupees: read.f64_of("reference_rupees")?,
        stop,
        targets: read
            .items_of("targets")?
            .iter()
            .map(read_target)
            .collect::<Result<_, _>>()?,
        remainder_units: read.f64_of("remainder_units")?,
        trail,
        breakeven,
    };
    let memory = LegMemory {
        held_units: read.f64_of("held_units")?,
        extreme_rupees: read.f64_of("extreme_rupees")?,
        trail_plan: read
            .maybe("trail_plan")?
            .map(|plan| read_plan(&plan))
            .transpose()?,
        anchor_rupees: read.opt_f64_of("anchor_rupees")?,
        after_first_target_done: read.bool_of("after_first_target_done")?,
    };
    let rules = read_rules(&read.field("rules")?)?;
    let context = read_context(&read.field("context")?)?;
    let key = read_key(read, instruments)?;
    Ok((key, LegState::resume(rules, context, levels, memory)))
}
