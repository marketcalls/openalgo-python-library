//! Margin: what opening exposure blocks from a strategy's free cash (04-execution 5).
//!
//! The engine asks for the margin of the quantity an order opens (an entry or an add;
//! a reduction blocks nothing) at the fill price, and rejects the order with
//! `insufficient_margin` when it is more than the free cash. Zero defaults are
//! forbidden: a position for which no margin rule exists is an error, never free.
//!
//! Built: CNC buys (100 percent of value), MIS equity at 1x, long options (the premium
//! paid). Not built yet: the dated margin table for futures, MIS leverage above 1x
//! (it needs the per-symbol leverage table), and short options (Phase 3).

use std::fmt;

use crate::model::{Day, Exchange, InstrumentKind, Product, Side};

/// One margin question: open `quantity_units` at `price_rupees` on `day`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MarginRequest {
    /// Exchange of the instrument.
    pub exchange: Exchange,
    /// Equity, future or option.
    pub kind: InstrumentKind,
    /// CNC, MIS or NRML.
    pub product: Product,
    /// Side of the exposure being opened.
    pub side: Side,
    /// Units opened, positive.
    pub quantity_units: f64,
    /// Price, rupees.
    pub price_rupees: f64,
    /// Rupees per one-rupee move per unit.
    pub multiplier: f64,
    /// Intraday leverage asked for; 1.0 when none was asked for.
    pub leverage: f64,
    /// IST trade date, for dated tables.
    pub day: Day,
}

/// What went wrong while computing margin.
#[derive(Clone, Debug, PartialEq)]
pub enum MarginError {
    /// The request cannot be priced (non-positive quantity, price or leverage).
    InvalidRequest { reason: String },
    /// A margin rule that is not built yet.
    NotBuilt { feature: &'static str },
}

impl fmt::Display for MarginError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MarginError::InvalidRequest { reason } => write!(f, "margin: {reason}"),
            MarginError::NotBuilt { feature } => {
                write!(
                    f,
                    "{feature} is not implemented yet (Phase 1 work in progress)"
                )
            }
        }
    }
}

impl std::error::Error for MarginError {}

/// The margin rules of a run.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct MarginModel;

impl MarginModel {
    /// Rupees of free cash that opening the request blocks.
    ///
    /// Errors: `InvalidRequest` for a non-positive quantity, price, multiplier or
    /// leverage; `NotBuilt` for futures, MIS leverage above 1x and short options.
    pub fn initial_margin(&self, request: &MarginRequest) -> Result<f64, MarginError> {
        let value = notional(request)?;
        match (request.kind, request.product, request.side) {
            (InstrumentKind::Equity, Product::Mis, _) if request.leverage > 1.0 => Err(not_built(
                "MIS leverage above 1x (the per-symbol leverage table)",
            )),
            (InstrumentKind::Equity, Product::Cnc | Product::Mis, _) => Ok(value),
            (InstrumentKind::Option, _, Side::Buy) => Ok(value),
            (InstrumentKind::Option, _, Side::Sell) => Err(not_built("short-option margin")),
            (InstrumentKind::Future, _, _) => Err(not_built("the dated futures margin table")),
            (kind, product, _) => Err(MarginError::InvalidRequest {
                reason: format!("no margin rule for {} with {}", kind.code(), product.code()),
            }),
        }
    }
}

fn notional(request: &MarginRequest) -> Result<f64, MarginError> {
    let inputs = [
        ("quantity", request.quantity_units),
        ("price", request.price_rupees),
        ("multiplier", request.multiplier),
        ("leverage", request.leverage),
    ];
    for (what, value) in inputs {
        if !value.is_finite() || value <= 0.0 {
            return Err(MarginError::InvalidRequest {
                reason: format!("{what} must be a positive number, got {value}"),
            });
        }
    }
    Ok(request.quantity_units * request.price_rupees * request.multiplier)
}

fn not_built(feature: &'static str) -> MarginError {
    MarginError::NotBuilt { feature }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(kind: InstrumentKind, product: Product, leverage: f64) -> MarginRequest {
        MarginRequest {
            exchange: Exchange::Nse,
            kind,
            product,
            side: Side::Buy,
            quantity_units: 100.0,
            price_rupees: 500.0,
            multiplier: 1.0,
            leverage,
            day: 0,
        }
    }

    #[test]
    fn delivery_and_unleveraged_intraday_block_the_full_value() {
        let model = MarginModel;
        let cnc = request(InstrumentKind::Equity, Product::Cnc, 1.0);
        let mis = request(InstrumentKind::Equity, Product::Mis, 1.0);
        assert_eq!(model.initial_margin(&cnc), Ok(50_000.0));
        assert_eq!(model.initial_margin(&mis), Ok(50_000.0));
    }

    #[test]
    fn rules_without_data_are_not_free() {
        let model = MarginModel;
        let future = request(InstrumentKind::Future, Product::Nrml, 1.0);
        let leveraged = request(InstrumentKind::Equity, Product::Mis, 5.0);
        assert!(matches!(
            model.initial_margin(&future),
            Err(MarginError::NotBuilt { .. })
        ));
        assert!(matches!(
            model.initial_margin(&leveraged),
            Err(MarginError::NotBuilt { .. })
        ));
    }
}
