//! Position sizing: from a sizer and the account to a quantity in units
//! (12-position-sizing 5).
//!
//! A sizer returns a quantity that is rounded down to the venue rule (whole lots; whole
//! shares for cash equity unless the run is fractional), then capped and checked
//! against margin by the engine. A size that rounds to zero is not an error: the order
//! is skipped with `SkipReason::RoundedToZero`, logged and counted in the result.
//!
//! Sizing uses the decision price; the engine re-checks the order at the fill price.
//! Costs in a risk budget come from the cost model's estimate for that order, which the
//! engine puts in `SizingContext::estimated_cost_rupees`.
//!
//! Built: fixed quantity, lots, value and percent sizers. Not built yet: fixed risk,
//! ATR risk and the engine default (1x the allocated capital at the fill price, net of
//! estimated costs, 03-architecture 7).

use std::fmt;

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
    /// Percent of a basis (`bt.size.percent(p, basis=...)`).
    Percent { percent: f64, basis: SizingBasis },
    /// Risk a percent of equity to the stop (`bt.size.fixed_risk(...)`).
    FixedRisk {
        risk_percent: f64,
        units: u32,
        hard_limit_units: Option<f64>,
    },
    /// Fixed risk with the stop distance taken as ATR times a multiple.
    AtrRisk {
        risk_percent: f64,
        multiple: f64,
        atr_rupees: f64,
    },
    /// The engine default of 03-architecture 7.
    EngineDefault,
}

/// What a sizer can see.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SizingContext {
    /// Current equity of the strategy's sub-account, rupees.
    pub equity_rupees: f64,
    /// Capital allocated at the start, rupees.
    pub initial_capital_rupees: f64,
    /// Equity not blocked as margin, rupees.
    pub free_cash_rupees: f64,
    /// The price at the decision, rupees.
    pub decision_price_rupees: f64,
    /// Distance from the decision price to the order's stop, rupees of price.
    pub stop_distance_rupees: Option<f64>,
    /// Units in one lot.
    pub lot_size_units: u32,
    /// Rupees per one-rupee move per unit.
    pub multiplier: f64,
    /// The cost model's estimate for the order, rupees.
    pub estimated_cost_rupees: f64,
    /// True when fractional quantities are allowed (fractional mode, cash equity).
    pub fractional: bool,
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

/// Round a quantity down to the venue rule: whole lots, or left as it is when
/// fractional quantities are allowed.
pub fn round_down_to_venue(units: f64, lot_size_units: u32, fractional: bool) -> f64 {
    if fractional {
        return units;
    }
    let lot = f64::from(lot_size_units.max(1));
    (units / lot).floor() * lot
}

fn positive(what: &str, value: f64) -> Result<f64, SizingError> {
    if value.is_finite() && value > 0.0 {
        Ok(value)
    } else {
        Err(SizingError::InvalidSizer {
            reason: format!("{what} must be a positive number, got {value}"),
        })
    }
}

fn basis_rupees(basis: SizingBasis, ctx: &SizingContext) -> f64 {
    match basis {
        SizingBasis::Equity => ctx.equity_rupees,
        SizingBasis::InitialCapital => ctx.initial_capital_rupees,
        SizingBasis::FreeCash => ctx.free_cash_rupees,
    }
}

/// The quantity a sizer asks for, in units.
///
/// Errors: `InvalidSizer` for a non-positive amount or decision price; `NotBuilt` for
/// fixed risk, ATR risk and the engine default.
pub fn size(spec: SizerSpec, ctx: &SizingContext) -> Result<SizeOutcome, SizingError> {
    let raw = match spec {
        SizerSpec::Qty(units) => positive("quantity", units)?,
        SizerSpec::Lots(lots) => f64::from(lots) * f64::from(ctx.lot_size_units),
        SizerSpec::Value { rupees } => {
            let unit_value =
                positive("decision price", ctx.decision_price_rupees)? * ctx.multiplier;
            positive("value", rupees)? / unit_value
        }
        SizerSpec::Percent { percent, basis } => {
            let unit_value =
                positive("decision price", ctx.decision_price_rupees)? * ctx.multiplier;
            positive("percent", percent)? / 100.0 * basis_rupees(basis, ctx) / unit_value
        }
        SizerSpec::FixedRisk { .. } => return Err(not_built("the fixed-risk sizer")),
        SizerSpec::AtrRisk { .. } => return Err(not_built("the ATR-risk sizer")),
        SizerSpec::EngineDefault => return Err(not_built("the engine default size")),
    };
    let units = round_down_to_venue(raw, ctx.lot_size_units, ctx.fractional);
    if units > 0.0 {
        Ok(SizeOutcome::Units(units))
    } else {
        Ok(SizeOutcome::Skipped(SkipReason::RoundedToZero))
    }
}

fn not_built(feature: &'static str) -> SizingError {
    SizingError::NotBuilt { feature }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn context() -> SizingContext {
        SizingContext {
            equity_rupees: 200_000.0,
            initial_capital_rupees: 100_000.0,
            free_cash_rupees: 50_000.0,
            decision_price_rupees: 800.0,
            stop_distance_rupees: None,
            lot_size_units: 1,
            multiplier: 1.0,
            estimated_cost_rupees: 0.0,
            fractional: false,
        }
    }

    #[test]
    fn value_and_percent_sizers_round_down_to_whole_shares() {
        let value = size(SizerSpec::Value { rupees: 10_000.0 }, &context());
        assert_eq!(value, Ok(SizeOutcome::Units(12.0)));
        let half = SizerSpec::Percent {
            percent: 50.0,
            basis: SizingBasis::InitialCapital,
        };
        assert_eq!(size(half, &context()), Ok(SizeOutcome::Units(62.0)));
    }

    #[test]
    fn lots_are_units_of_the_lot_size() {
        let futures = SizingContext {
            lot_size_units: 75,
            ..context()
        };
        assert_eq!(
            size(SizerSpec::Lots(2), &futures),
            Ok(SizeOutcome::Units(150.0))
        );
    }

    #[test]
    fn a_size_that_rounds_to_zero_is_skipped() {
        let tiny = size(SizerSpec::Value { rupees: 100.0 }, &context());
        assert_eq!(tiny, Ok(SizeOutcome::Skipped(SkipReason::RoundedToZero)));
    }

    #[test]
    fn unbuilt_sizers_say_so() {
        let error = size(SizerSpec::EngineDefault, &context());
        assert!(matches!(error, Err(SizingError::NotBuilt { .. })));
    }
}
