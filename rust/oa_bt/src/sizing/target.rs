//! Target positions (`self.target(...)`, 12-position-sizing 5, FR-SIZE-08): the
//! position a strategy asks to hold on one instrument, in signed units.
//!
//! A target given in units or lots is taken as given, as `buy(qty=)` and `buy(lots=)`
//! are: the order it leads to meets the quantity rule or is rejected for it. A target
//! given as a value or a percent of the sizing equity is converted at the sizing price
//! and rounded towards zero to the venue rule (whole lots; whole shares unless the run
//! is fractional); a percent of a sizing equity below zero is a flat target. The target
//! is rounded, never the order that reaches it, so a trim is not under-sold
//! (11-portfolio-rebalancing 2.1). The engine sends the difference between this and the
//! position (`engine::target`).

use super::{unit_value, venue::round_down_to_venue, SizingContext, SizingError};

/// Percent in a whole.
const PERCENT: f64 = 100.0;

/// The position `target()` asks for: positive is long, negative short, zero flat.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TargetSize {
    /// Units (`target(qty=)`), taken as given.
    Units(f64),
    /// Whole lots (`target(lots=)`).
    Lots(i32),
    /// Rupees of value at the sizing price (`target(value=)`).
    Rupees(f64),
    /// Percent of the sizing equity (`target(percent=)`), percent units.
    PercentOfEquity(f64),
}

fn finite(what: &str, value: f64) -> Result<f64, SizingError> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(SizingError::InvalidSizer {
            reason: format!("target {what} must be a finite number, got {value}"),
        })
    }
}

/// The target position in signed units (see the module documentation).
///
/// Errors: `InvalidSizer` for an amount that is not a finite number, or a value or
/// percent target when the sizing price or the multiplier is not a positive number.
pub fn target_units(target: TargetSize, ctx: &SizingContext) -> Result<f64, SizingError> {
    let rupees = match target {
        TargetSize::Units(units) => return finite("qty", units),
        TargetSize::Lots(lots) => {
            return Ok(f64::from(lots) * f64::from(ctx.lot_size_units.max(1)));
        }
        TargetSize::Rupees(rupees) => finite("value", rupees)?,
        // Clamped as the sizers clamp their basis: the sign of a negative equity must
        // not flip the side the strategy asked for.
        TargetSize::PercentOfEquity(percent) => {
            finite("percent", percent)? / PERCENT * ctx.sizing_equity().max(0.0)
        }
    };
    let units = rupees.abs() / unit_value(ctx)?;
    let rounded = round_down_to_venue(units, ctx.lot_size_units, ctx.fractional);
    Ok(rounded.copysign(rupees))
}
