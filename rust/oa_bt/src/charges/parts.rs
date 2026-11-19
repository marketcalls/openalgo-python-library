//! An order filled in parts: costed once on its total (`CostModel::fill_cost` prices one
//! executed order, so a brokerage cap or a flat per-order amount is charged once).
//!
//! A live session's venue reports an order's fills one increment at a time. Each part is
//! charged the cost of the order's fills so far, on their total, less what the earlier
//! parts were charged: the accruals by component, and the rupees posted to the ledger as
//! the total rounded to the paisa less what was posted before. So an order filled in any
//! number of parts posts what it posts filled at once, and its first part is charged
//! exactly as a fill of a backtest is.
//!
//! A CNC fill of cash equity is charged part by part by the same-day rule (`same_day`),
//! which prices each fill against the day's fills on the other side.

use crate::market::costs::{CostBreakdown, CostModel};
use crate::model::Product;
use crate::portfolio::round_to_paise;

use super::book::{is_cash_equity, ChargeBook, FillToCharge};
use super::same_day::difference;
use super::{fill_segment, price_units, ChargeError};

/// What the fills of one order (on one side of its position change) have been charged
/// so far, for costing the next on their total.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct OrderCharges {
    /// Units filled so far.
    pub units: f64,
    /// Their traded value, rupees: units x multiplier x price (`price_units`).
    pub value_rupees: f64,
    /// The cost of those units on their total, as the cost model returns it.
    pub cost: CostBreakdown,
    /// What the ledger was posted for them, rupees, whole paise.
    pub charged_rupees: f64,
}

impl ChargeBook {
    /// Charge one part of an order under `model` (module docs), `so_far` being what the
    /// order's earlier parts were charged; it is updated. Returns the rupees to post to
    /// the ledger, whole paise.
    ///
    /// Errors: `Cost` when the fill cannot be priced.
    pub fn charge_part(
        &mut self,
        model: &CostModel,
        fill: &FillToCharge<'_>,
        so_far: &mut OrderCharges,
    ) -> Result<f64, ChargeError> {
        if fill.key.product == Product::Cnc && is_cash_equity(fill.spec) {
            return self.charge(model, fill);
        }
        let segment = fill_segment(fill.spec, fill.facts, fill.key.product)?;
        let units = so_far.units + fill.units;
        let part_quantity = price_units(fill.spec, fill.units);
        let value_rupees = so_far.value_rupees + part_quantity * fill.price_rupees;
        let (price, quantity) = if so_far.units > 0.0 {
            let quantity = price_units(fill.spec, units);
            (value_rupees / quantity, quantity)
        } else {
            (fill.price_rupees, part_quantity)
        };
        let total = model.fill_cost(segment, fill.side, price, quantity, fill.day)?;
        let posted_total = round_to_paise(model.reported_total(&total));
        let charged = round_to_paise(posted_total - so_far.charged_rupees);
        let part = difference(&total, &so_far.cost);
        *so_far = OrderCharges {
            units,
            value_rupees,
            cost: total,
            charged_rupees: posted_total,
        };
        self.book_charged(model, fill, (segment, part), charged, None)
    }
}
