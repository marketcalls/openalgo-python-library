//! Charges in the kernel (04-execution 6): every fill charged through the cost model
//! (`market::costs`) at the rates of its IST trade date, contract-note rounding at the
//! session close, the DP charge on delivery sells, cost estimates for sizing, and the
//! record of which dated market-table rows a run used.
//!
//! # How a fill is charged
//!
//! The segment comes from the instrument, the fill's product and the instrument's
//! market facts (`Segment::resolve`): an equity fill is delivery or intraday by its
//! product. One fill is one executed order. In `contract_note` rounding (the default)
//! the fill's cost is the unrounded accrual the cost model returns; the ledger posts
//! it rounded to the paisa, and at the session close each day's contract note (one per
//! trade day and note group, as `ContractNoteAggregator` defines it) is rounded and the
//! difference between what the note charges and what was posted is posted as one
//! adjustment, so the ledger's charges equal the notes'. In `calculator` rounding each
//! fill is costed the way the Zerodha calculator costs a one-sided trade and there is
//! no note adjustment.
//!
//! A delivery (CNC) sell from holdings also pays the DP charge, once per scrip per sell
//! day; it is not on the contract note. A CNC buy and sell of the same scrip on one day
//! is charged as intraday by brokers (04-execution 2); that reclassification is not
//! built yet and such a sell stops the run with `NotBuilt` rather than being charged as
//! delivery.
//!
//! | Module | Responsibility |
//! |---|---|
//! | `book` | `ChargeBook`: the run's charge state, per-fill charging, trade numbers |
//! | `notes` | Contract-note settlement at the session close |
//! | `estimate` | Cost estimates for sizers |
//! | `usage` | `TableUse`: dated table rows a run used |

mod book;
mod estimate;
mod notes;
mod usage;

#[cfg(test)]
mod tests;

use std::fmt;

use crate::exec::InstrumentSpec;
use crate::market::costs::{CostError, Segment};
use crate::market::facts::InstrumentMarket;
use crate::model::Product;

pub use book::{ChargeBook, ChargesOutput, FillCharge, FillToCharge};
pub use estimate::{estimate_entry, estimate_round_trip};
pub use notes::SettledNote;
pub use usage::{MarketTable, TableUse};

/// Why a fill could not be charged.
#[derive(Clone, Debug, PartialEq)]
pub enum ChargeError {
    /// The cost model cannot price the fill (no segment, no schedule for the date).
    Cost(CostError),
    /// A charging rule that is not built yet.
    NotBuilt { feature: &'static str },
}

impl fmt::Display for ChargeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ChargeError::Cost(inner) => inner.fmt(f),
            ChargeError::NotBuilt { feature } => write!(
                f,
                "{feature} is not implemented yet (Phase 1 work in progress)"
            ),
        }
    }
}

impl std::error::Error for ChargeError {}

impl From<CostError> for ChargeError {
    fn from(error: CostError) -> ChargeError {
        ChargeError::Cost(error)
    }
}

/// The cost segment of a fill of `spec` in `product`.
///
/// Errors: `Cost` when the exchange, kind and product do not resolve to a segment (a
/// BSE equity without its scrip group, an MCX future without its commodity, currency
/// derivatives).
pub fn fill_segment(
    spec: &InstrumentSpec,
    facts: Option<&InstrumentMarket>,
    product: Product,
) -> Result<Segment, ChargeError> {
    let bse_group = facts.and_then(|f| f.bse_group.as_deref());
    let underlying = facts.and_then(|f| f.underlying.as_deref());
    Ok(Segment::resolve(
        spec.exchange,
        spec.kind,
        product,
        bse_group,
        underlying,
    )?)
}
