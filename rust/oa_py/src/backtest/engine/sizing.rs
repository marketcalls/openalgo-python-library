//! Sizing across the boundary: the strategy's `reinvest` and `size_caps` from the risk
//! dict, a `target()` request, and what a strategy's own sizing function sees.
//!
//! The risk dict (`openalgo/backtest/_size_controls.py`) carries `reinvest` (bool) and
//! `size_caps` (None, or a dict with `position` as (kind, amount) with kind "units",
//! "lots", "rupees" or "percent", `leverage` as a number, and `risk` as (kind, amount)
//! with kind "rupees" or "percent"; each may be None). A target dict has `instrument`,
//! `product`, `kind` ("units", "lots", "rupees" or "percent"), `amount` and `leverage`.

use oa_bt::engine::TargetRequest;
use oa_bt::exec::InstrumentId;
use oa_bt::model::{Product, Side};
use oa_bt::sizing::{SizeCaps, SizingContext, SizingControls, TargetSize};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use super::setup::{field, optional, unknown};
use super::NativeEngineInputError;

/// A cap as it crosses: (kind, amount).
type Cap = (String, f64);

fn whole(amount: f64, what: &str) -> PyResult<u32> {
    if amount >= 1.0 && amount.fract() == 0.0 && amount <= f64::from(u32::MAX) {
        Ok(amount as u32)
    } else {
        Err(NativeEngineInputError::new_err(format!(
            "{what} must be a whole number of lots, 1 or more, got {amount}"
        )))
    }
}

fn caps(dict: &Bound<'_, PyDict>) -> PyResult<SizeCaps> {
    let mut caps = SizeCaps {
        leverage: optional(dict, "leverage")?,
        ..SizeCaps::default()
    };
    if let Some((kind, amount)) = optional::<Cap>(dict, "position")? {
        match kind.as_str() {
            "units" => caps.position_units = Some(amount),
            "lots" => caps.position_lots = Some(whole(amount, "the position cap")?),
            "rupees" => caps.position_rupees = Some(amount),
            "percent" => caps.position_percent = Some(amount),
            _ => {
                return Err(unknown(
                    "position cap",
                    &kind,
                    "units, lots, rupees or percent",
                ))
            }
        }
    }
    if let Some((kind, amount)) = optional::<Cap>(dict, "risk")? {
        match kind.as_str() {
            "rupees" => caps.risk_rupees = Some(amount),
            "percent" => caps.risk_percent = Some(amount),
            _ => return Err(unknown("risk cap", &kind, "rupees or percent")),
        }
    }
    Ok(caps)
}

/// The sizing controls of the risk dict; absent keys leave the defaults (reinvesting,
/// no cap).
pub(super) fn controls(risk: &Bound<'_, PyDict>) -> PyResult<SizingControls> {
    let defaults = SizingControls::default();
    let size_caps: Option<Bound<'_, PyDict>> = optional(risk, "size_caps")?;
    Ok(SizingControls {
        reinvest: optional(risk, "reinvest")?.unwrap_or(defaults.reinvest),
        caps: size_caps
            .map(|dict| caps(&dict))
            .transpose()?
            .unwrap_or_default(),
    })
}

/// A `target()` request from its dict.
pub(super) fn target_request(dict: &Bound<'_, PyDict>) -> PyResult<TargetRequest> {
    let product: String = field(dict, "product")?;
    let kind: String = field(dict, "kind")?;
    let size = match kind.as_str() {
        "units" => TargetSize::Units(field(dict, "amount")?),
        "lots" => TargetSize::Lots(field(dict, "amount")?),
        "rupees" => TargetSize::Rupees(field(dict, "amount")?),
        "percent" => TargetSize::PercentOfEquity(field(dict, "amount")?),
        _ => {
            return Err(unknown(
                "target kind",
                &kind,
                "units, lots, rupees or percent",
            ))
        }
    };
    Ok(TargetRequest {
        instrument: InstrumentId(field(dict, "instrument")?),
        product: Product::from_code(&product)
            .ok_or_else(|| unknown("product", &product, "CNC, MIS or NRML"))?,
        size,
        leverage: optional(dict, "leverage")?,
    })
}

/// What a sizer sees for an order on `side`, as the dict a sizing function's `ctx` is
/// built from: rupees and units, the position signed (negative when short).
pub(super) fn inputs<'py>(
    py: Python<'py>,
    ctx: &SizingContext,
    side: Side,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new_bound(py);
    dict.set_item("equity", ctx.sizing_equity())?;
    dict.set_item("capital", ctx.initial_capital_rupees)?;
    dict.set_item("cash", ctx.free_cash_rupees)?;
    dict.set_item("price", ctx.decision_price_rupees)?;
    dict.set_item("stop_distance", ctx.stop_distance_rupees)?;
    dict.set_item("lot_size", ctx.lot_size_units)?;
    dict.set_item("multiplier", ctx.multiplier)?;
    dict.set_item("fractional", ctx.fractional)?;
    dict.set_item("position", ctx.held_on_side_units * side.sign())?;
    Ok(dict)
}
