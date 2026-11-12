//! The initial risk of a trade (12-position-sizing 6, FR-SIZE-09): what the fill that
//! opened a position would lose at the stop its entry set, the denominator of the
//! trade's R-multiple.
//!
//! It is read once, right after the opening fill's leg is built (step 2, or step 3 for
//! a resting order), so a stop that a trail, a breakeven move or an add tightens later
//! does not change it. A position opened with no stop (no `stop=`, only a trail or a
//! breakeven that arm later, or the next contract of a futures roll, which carries the
//! rolled leg over afterwards) has no initial risk, and its trade has no R.

use crate::exec::Fill;
use crate::portfolio::PositionKey;

use super::kernel::Engine;

impl Engine {
    /// Rupees the position `key`, just opened by `fill`, would lose at its stop: units
    /// times the distance from the leg's reference price (the fill average) to the stop,
    /// times the multiplier. `None` when the leg has no stop, or when the stop is not on
    /// the losing side of the price (a level that cannot lose).
    pub(super) fn initial_risk(&self, key: PositionKey, fill: &Fill) -> Option<f64> {
        let (_, leg) = self.legs.iter().find(|(k, _)| *k == key)?;
        let stop = leg.stop?;
        let per_unit = (leg.reference_rupees - stop.level_rupees) * fill.side.sign();
        let rupees = per_unit * fill.quantity_units * leg.context.multiplier;
        (rupees.is_finite() && rupees > 0.0).then_some(rupees)
    }
}
