//! Builders shared by the leg tests.

use crate::model::Side;
use crate::rms::error::RmsError;
use crate::rms::leg::{BarPrices, LegContext, LegState};
use crate::rms::rules::{BreakevenSpec, Distance, LegRules, TargetSpec, TrailSpec};

/// A cash-equity leg on a 0.05 tick, one share per lot, levels rounded.
pub(super) fn context(side: Side) -> LegContext {
    LegContext {
        side,
        tick_rupees: 0.05,
        multiplier: 1.0,
        round_levels: true,
        lot_size_units: 1,
        fractional: false,
    }
}

/// A leg opened at `average` for `units`; a failed open shows as the error.
pub(super) fn open(
    rules: LegRules,
    side: Side,
    average: f64,
    units: f64,
) -> Result<LegState, RmsError> {
    LegState::open(rules, context(side), average, units)
}

pub(super) fn bar(open: f64, high: f64, low: f64, close: f64) -> BarPrices {
    BarPrices {
        open,
        high,
        low,
        close,
    }
}

pub(super) fn points(value: f64) -> Distance {
    Distance::Points(value)
}

pub(super) fn stop_and_target(stop: f64, target: f64) -> LegRules {
    LegRules {
        stop: Some(points(stop)),
        target: Some(points(target)),
        ..LegRules::default()
    }
}

/// `targets=[(points, fraction), ...]` with a stop.
pub(super) fn stop_and_targets(stop: f64, targets: &[(f64, f64)]) -> LegRules {
    LegRules {
        stop: Some(points(stop)),
        targets: targets
            .iter()
            .map(|&(distance, fraction)| TargetSpec {
                distance: points(distance),
                fraction,
            })
            .collect(),
        ..LegRules::default()
    }
}

/// `bt.trail(x, y)` in points; `y = 0` is the continuous form.
pub(super) fn trail(x: f64, y: f64) -> TrailSpec {
    TrailSpec {
        trigger: points(x),
        step: (y > 0.0).then(|| points(y)),
    }
}

/// `bt.breakeven(trigger, offset)` in points.
pub(super) fn breakeven(trigger: f64, offset: f64) -> BreakevenSpec {
    BreakevenSpec {
        trigger: points(trigger),
        offset: (offset > 0.0).then(|| points(offset)),
    }
}

/// The stop level in force, if any.
pub(super) fn stop_level(leg: &LegState) -> Option<f64> {
    leg.stop.map(|stop| stop.level_rupees)
}

/// (level, units, filled) of every target.
pub(super) fn target_rows(leg: &LegState) -> Vec<(f64, f64, bool)> {
    leg.targets
        .iter()
        .map(|t| {
            let filled = t.state == crate::rms::leg::TargetState::Filled;
            (t.level_rupees, t.quantity_units, filled)
        })
        .collect()
}
