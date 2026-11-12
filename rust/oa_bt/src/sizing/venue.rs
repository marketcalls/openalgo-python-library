//! The venue rule for sized quantities (03-architecture 7): whole lots (whole shares
//! for cash equity) unless the run is fractional, and never above the freeze quantity.

/// Relative tolerance below which a quantity on a lot boundary counts as on it, so that
/// a computed 150.0000000001 units of a 75-unit lot is two lots, not one.
const LOT_TOLERANCE: f64 = 1.0e-9;

/// Round a quantity in units down to the venue rule: whole lots, or left as it is when
/// fractional quantities are allowed. A lot size of 0 is treated as 1.
pub fn round_down_to_venue(units: f64, lot_size_units: u32, fractional: bool) -> f64 {
    if fractional {
        return units;
    }
    let lot = f64::from(lot_size_units.max(1));
    let lots = units / lot;
    (lots + LOT_TOLERANCE * lots.abs().max(1.0)).floor() * lot
}

/// Cap a sized quantity at the freeze quantity: the largest whole number of lots at or
/// below it. A sizer's quantity is capped rather than denied, because the sizer asked
/// for as much as its rule allows; an explicit `qty=` or `lots=` above the freeze
/// quantity is still denied by the freeze check (reject code `freeze_qty`).
pub fn cap_at_freeze(units: f64, freeze_quantity_units: Option<u32>, lot_size_units: u32) -> f64 {
    match freeze_quantity_units {
        Some(limit) if units > f64::from(limit) => {
            round_down_to_venue(f64::from(limit), lot_size_units, false)
        }
        _ => units,
    }
}
