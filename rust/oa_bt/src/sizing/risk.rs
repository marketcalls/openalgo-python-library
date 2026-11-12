//! Risk sizers (12-position-sizing 5): fixed risk to the order's stop, and the same with
//! the stop distance taken from the ATR.
//!
//! Fixed risk: the rupees at risk are `equity x risk_percent / 100` less the estimated
//! costs of the round trip; the quantity is that over the loss per unit at the stop
//! (`stop distance x multiplier`), capped at `hard_limit_units`, then split into
//! `units` equal tranches of which the order is the first. Rounding down to the venue
//! rule and the freeze cap follow in `size`.

use super::{positive, SizingContext, SizingError};

/// Units the fixed-risk sizer asks for, before the venue rule.
///
/// Errors: `InvalidSizer` for a non-positive risk percent, `units` of 0, a non-positive
/// hard limit, or a missing or non-positive stop distance.
pub(super) fn fixed_risk_units(
    risk_percent: f64,
    units: u32,
    hard_limit_units: Option<f64>,
    ctx: &SizingContext,
) -> Result<f64, SizingError> {
    let distance = ctx
        .stop_distance_rupees
        .ok_or_else(|| SizingError::InvalidSizer {
            reason: "bt.size.fixed_risk needs a stop: pass stop= with the order".to_string(),
        })?;
    risked_units(risk_percent, units, hard_limit_units, distance, ctx)
}

/// Units the ATR-risk sizer asks for: fixed risk with a stop `multiple x atr` away.
///
/// Errors: as `fixed_risk_units`, and `InvalidSizer` when the ATR is not a positive
/// number (too few bars for its period at the decision).
pub(super) fn atr_risk_units(
    risk_percent: f64,
    multiple: f64,
    atr_rupees: f64,
    ctx: &SizingContext,
) -> Result<f64, SizingError> {
    if !(atr_rupees.is_finite() && atr_rupees > 0.0) {
        return Err(SizingError::InvalidSizer {
            reason: format!(
                "bt.size.atr_risk needs the ATR at the decision, got {atr_rupees}: the \
                 strategy has fewer closed bars than the ATR period"
            ),
        });
    }
    let distance = positive("the ATR multiple", multiple)? * atr_rupees;
    risked_units(risk_percent, 1, None, distance, ctx)
}

fn risked_units(
    risk_percent: f64,
    units: u32,
    hard_limit_units: Option<f64>,
    distance: f64,
    ctx: &SizingContext,
) -> Result<f64, SizingError> {
    let distance = positive("the stop distance", distance)?;
    let percent = positive("risk_pct", risk_percent)?;
    if units == 0 {
        return Err(SizingError::InvalidSizer {
            reason: "units must be at least 1".to_string(),
        });
    }
    let budget = ctx.equity_rupees * percent / 100.0 - ctx.estimated_cost_rupees;
    let loss_per_unit = distance * positive("the multiplier", ctx.multiplier)?;
    let mut quantity = budget.max(0.0) / loss_per_unit;
    if let Some(limit) = hard_limit_units {
        quantity = quantity.min(positive("hard_limit", limit)?);
    }
    Ok(quantity / f64::from(units))
}
