//! Cost estimates for sizers (12-position-sizing 5: "costs in the risk budget come from
//! the Indian cost model's estimate for that order").
//!
//! An estimate prices the order at the sizing price on the decision's trade date, as
//! the cost model's reported total: the entry alone for the engine default, a buy and a
//! sell of the same quantity on the same day for the risk sizers. It is an estimate:
//! the order fills at another price, and in contract-note rounding the day's note is
//! rounded once more.

use crate::exec::InstrumentSpec;
use crate::market::costs::CostModel;
use crate::market::facts::InstrumentMarket;
use crate::model::{Day, Product, Side};

use super::{fill_segment, ChargeError};

/// Rupees an order of `units` at `price_rupees` is estimated to cost on `day`.
///
/// Errors: `Cost` when the fill cannot be priced. Zero units cost nothing.
#[allow(clippy::too_many_arguments)]
pub fn estimate_entry(
    model: &CostModel,
    spec: &InstrumentSpec,
    facts: Option<&InstrumentMarket>,
    product: Product,
    side: Side,
    units: f64,
    price_rupees: f64,
    day: Day,
) -> Result<f64, ChargeError> {
    if units <= 0.0 {
        return Ok(0.0);
    }
    let segment = fill_segment(spec, facts, product)?;
    let cost = model.fill_cost(segment, side, price_rupees, units, day)?;
    Ok(model.reported_total(&cost))
}

/// Rupees a buy and a sell of `units` at `price_rupees` on `day` are estimated to
/// cost together.
///
/// Errors: `Cost` when the fills cannot be priced. Zero units cost nothing.
pub fn estimate_round_trip(
    model: &CostModel,
    spec: &InstrumentSpec,
    facts: Option<&InstrumentMarket>,
    product: Product,
    units: f64,
    price_rupees: f64,
    day: Day,
) -> Result<f64, ChargeError> {
    if units <= 0.0 {
        return Ok(0.0);
    }
    let segment = fill_segment(spec, facts, product)?;
    let cost = model.round_trip_cost(segment, price_rupees, price_rupees, units, day, day)?;
    Ok(model.reported_total(&cost))
}
