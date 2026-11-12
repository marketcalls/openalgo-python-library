//! Position sizing: from a sizer and the account to a quantity in units
//! (12-position-sizing 5, 03-architecture 7).
//!
//! A sizer returns a quantity that is rounded down to the venue rule (whole lots; whole
//! shares for cash equity unless the run is fractional) and capped at the instrument's
//! freeze quantity (the largest whole-lot quantity at or below it); the engine then
//! checks margin at the fill bar. A size that rounds to zero is not an error: the order
//! is skipped with `SkipReason::RoundedToZero`, logged and counted in the result.
//!
//! Sizing uses the decision price; the engine re-checks the order at the fill price.
//! Costs in a risk budget, and in the engine default, come from the Indian cost model's
//! estimate for that order, which the engine puts in
//! `SizingContext::estimated_cost_rupees` (zero when the run is cost-free).
//!
//! | Module | Responsibility |
//! |---|---|
//! | `venue` | Rounding down to the venue rule, the freeze-quantity cap |
//! | `risk` | Fixed risk to the stop, and ATR risk |
//! | `default` | The engine default: 1x the capital net of estimated costs, 1 lot for F&O |

mod default;
mod risk;
mod venue;

#[cfg(test)]
mod tests;

use std::fmt;

pub use default::{engine_default_units, VENUE_CASH_BUFFER};
pub use venue::{cap_at_freeze, round_down_to_venue};

/// What a percent sizer takes its percent of.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SizingBasis {
    /// Current equity (compounding).
    #[default]
    Equity,
    /// The capital allocated at the start (no compounding).
    InitialCapital,
    /// Equity not blocked as margin.
    FreeCash,
}

/// A sizer, as passed in `size=` or set as the strategy default.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SizerSpec {
    /// A fixed quantity in units (`bt.size.qty(n)`), rounded down to the venue rule.
    Qty(f64),
    /// A fixed number of lots (`bt.size.lots(n)`).
    Lots(u32),
    /// Rupees of notional (`bt.size.value(rupees)`).
    Value { rupees: f64 },
    /// Percent of a basis (`bt.size.percent(p, basis=...)`), percent units.
    Percent { percent: f64, basis: SizingBasis },
    /// Risk `risk_percent` percent of equity to the stop (`bt.size.fixed_risk(...)`):
    /// the risked quantity is capped at `hard_limit_units` and split into `units` equal
    /// tranches, of which the order is one.
    FixedRisk {
        risk_percent: f64,
        units: u32,
        hard_limit_units: Option<f64>,
    },
    /// Fixed risk with the stop distance taken as `atr_rupees` times `multiple`
    /// (`bt.size.atr_risk(...)`); the ATR is resolved through `openalgo.ta` at the
    /// decision.
    AtrRisk {
        risk_percent: f64,
        multiple: f64,
        atr_rupees: f64,
    },
    /// The engine default of 03-architecture 7 (no `qty`, `lots` or `size`).
    EngineDefault,
}

/// What a sizer can see. Money is rupees, quantities units.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SizingContext {
    /// Current equity of the strategy's sub-account, rupees.
    pub equity_rupees: f64,
    /// Capital allocated at the start, rupees.
    pub initial_capital_rupees: f64,
    /// Equity not blocked as margin, rupees.
    pub free_cash_rupees: f64,
    /// The price the order is sized at, rupees: the decision price, or the expected
    /// fill price when the engine default is resolved at the fill.
    pub decision_price_rupees: f64,
    /// Distance from the decision price to the order's stop, rupees of price.
    pub stop_distance_rupees: Option<f64>,
    /// Units in one lot.
    pub lot_size_units: u32,
    /// Rupees per one-rupee move per unit.
    pub multiplier: f64,
    /// The cost model's estimate for the order, rupees (the entry for the engine
    /// default, the round trip for the risk sizers).
    pub estimated_cost_rupees: f64,
    /// True when fractional quantities are allowed (fractional mode, cash equity).
    pub fractional: bool,
    /// True for a future or an option (the engine default is then one lot).
    pub derivative: bool,
    /// The exchange's largest quantity per order, units, when it sets one.
    pub freeze_quantity_units: Option<u32>,
    /// Instruments the strategy's capital is split across by the engine default (1 or
    /// more; 03-architecture 7: split equally across a universe).
    pub instruments: u32,
    /// Fraction of the engine default's budget kept back as cash (`VENUE_CASH_BUFFER` in
    /// venue mode, 0 when the size is taken at the fill price).
    pub cash_buffer: f64,
}

/// Why a sized order was skipped.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SkipReason {
    /// The quantity rounded down to zero.
    RoundedToZero,
    /// A cap left no room.
    Cap,
    /// Margin left no room.
    Margin,
}

impl SkipReason {
    /// The reason name used in results.
    pub fn code(self) -> &'static str {
        match self {
            SkipReason::RoundedToZero => "rounded_to_zero",
            SkipReason::Cap => "cap",
            SkipReason::Margin => "margin",
        }
    }
}

/// A quantity, or the reason there is none.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SizeOutcome {
    /// Units, positive, on the venue rule.
    Units(f64),
    /// No order.
    Skipped(SkipReason),
}

/// What went wrong while sizing.
#[derive(Clone, Debug, PartialEq)]
pub enum SizingError {
    /// The sizer or the context cannot be used.
    InvalidSizer { reason: String },
    /// A sizer that is not built yet.
    NotBuilt { feature: &'static str },
}

impl fmt::Display for SizingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SizingError::InvalidSizer { reason } => write!(f, "sizer: {reason}"),
            SizingError::NotBuilt { feature } => {
                write!(
                    f,
                    "{feature} is not implemented yet (Phase 1 work in progress)"
                )
            }
        }
    }
}

impl std::error::Error for SizingError {}

/// `value` when it is a positive finite number; otherwise `InvalidSizer` naming `what`.
pub(crate) fn positive(what: &str, value: f64) -> Result<f64, SizingError> {
    if value.is_finite() && value > 0.0 {
        Ok(value)
    } else {
        Err(SizingError::InvalidSizer {
            reason: format!("{what} must be a positive number, got {value}"),
        })
    }
}

/// Rupees one unit is worth at the sizing price.
pub(crate) fn unit_value(ctx: &SizingContext) -> Result<f64, SizingError> {
    Ok(positive("the sizing price", ctx.decision_price_rupees)?
        * positive("the multiplier", ctx.multiplier)?)
}

fn basis_rupees(basis: SizingBasis, ctx: &SizingContext) -> f64 {
    match basis {
        SizingBasis::Equity => ctx.equity_rupees,
        SizingBasis::InitialCapital => ctx.initial_capital_rupees,
        SizingBasis::FreeCash => ctx.free_cash_rupees,
    }
}

/// The quantity a sizer asks for before the venue rule, in units.
fn raw_units(spec: SizerSpec, ctx: &SizingContext) -> Result<f64, SizingError> {
    match spec {
        SizerSpec::Qty(units) => positive("quantity", units),
        SizerSpec::Lots(lots) if lots > 0 => Ok(f64::from(lots) * f64::from(ctx.lot_size_units)),
        SizerSpec::Lots(_) => Err(SizingError::InvalidSizer {
            reason: "lots must be at least 1".to_string(),
        }),
        SizerSpec::Value { rupees } => Ok(positive("value", rupees)? / unit_value(ctx)?),
        SizerSpec::Percent { percent, basis } => {
            let budget = positive("percent", percent)? / 100.0 * basis_rupees(basis, ctx);
            Ok(budget / unit_value(ctx)?)
        }
        SizerSpec::FixedRisk {
            risk_percent,
            units,
            hard_limit_units,
        } => risk::fixed_risk_units(risk_percent, units, hard_limit_units, ctx),
        SizerSpec::AtrRisk {
            risk_percent,
            multiple,
            atr_rupees,
        } => risk::atr_risk_units(risk_percent, multiple, atr_rupees, ctx),
        SizerSpec::EngineDefault => engine_default_units(ctx),
    }
}

/// The quantity a sizer asks for, in units: rounded down to the venue rule and capped
/// at the freeze quantity, or the reason it gives none.
///
/// Errors: `InvalidSizer` for a non-positive amount, sizing price or multiplier, a
/// fixed-risk sizer without a stop distance, or an ATR that is not a positive number.
pub fn size(spec: SizerSpec, ctx: &SizingContext) -> Result<SizeOutcome, SizingError> {
    let raw = raw_units(spec, ctx)?.max(0.0);
    let rounded = round_down_to_venue(raw, ctx.lot_size_units, ctx.fractional);
    let units = cap_at_freeze(rounded, ctx.freeze_quantity_units, ctx.lot_size_units);
    Ok(match (rounded > 0.0, units > 0.0) {
        (_, true) => SizeOutcome::Units(units),
        (true, false) => SizeOutcome::Skipped(SkipReason::Cap),
        (false, false) => SizeOutcome::Skipped(SkipReason::RoundedToZero),
    })
}
