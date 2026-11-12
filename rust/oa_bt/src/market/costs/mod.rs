//! Indian trading costs: brokerage presets and statutory charges, versioned by
//! effective date, with contract-note and calculator rounding modes.
//!
//! # Modules
//!
//! | Module | Responsibility |
//! |---|---|
//! | `segment` | The segment table: one `Segment` per row of the rate card |
//! | `resolve` | Instrument (exchange, kind, product, BSE group, underlying) to segment |
//! | `schedule` | `CostSchedule`: one dated version of the rates, and its validation |
//! | `book` | `ScheduleBook`: lookup by trade date, and the trust status of a lookup |
//! | `builtin` | The built-in schedule versions: `schedules.csv` and its parser |
//! | `brokerage` | Brokerage presets (the broker's part of the cost) |
//! | `rounding` | Rounding modes; exact rounding of sums; calculator rounding |
//! | `rules` | The arithmetic for one fill, unrounded and calculator-rounded |
//! | `cost_model` | `CostModel`: per-fill cost, round trip, DP charge, exercise STT |
//! | `batch` | The same per-fill cost over slices |
//! | `note` | Contract-note aggregation and rounding |
//! | `breakdown`, `error` | Result and error types |
//!
//! Units: prices and amounts are rupees, quantities are numbers of price units
//! (never lots), a `_rate` is a fraction (0.001 is 0.1 percent), anything named
//! `percent` takes percent units, and trade dates are IST days.
//!
//! Currency derivatives and MCX options are out of scope and are an error, never a
//! guess. A trade dated before the first schedule version is an error unless the
//! caller opts in to backfill.
//!
//! # Per-fill use (the kernel)
//!
//! ```
//! use oa_bt::market::costs::{ContractNoteAggregator, CostError, CostModel, Segment};
//! use oa_bt::model::{days_from_civil, Side};
//!
//! # fn main() -> Result<(), CostError> {
//! let model = CostModel::builtin()?; // built-in schedules, Zerodha, contract-note rounding
//! let day = days_from_civil(2026, 11, 12);
//! let mut notes = ContractNoteAggregator::new();
//!
//! let buy = model.fill_cost(Segment::NseEqIntraday, Side::Buy, 1000.0, 400.0, day)?;
//! let sell = model.fill_cost(Segment::NseEqIntraday, Side::Sell, 1100.0, 400.0, day)?;
//! notes.add(day, Segment::NseEqIntraday, &buy);
//! notes.add(day, Segment::NseEqIntraday, &sell);
//!
//! // At the session close: what the day's contract note charges.
//! assert_eq!(notes.total_charged(), 200.62);
//! # Ok(())
//! # }
//! ```
//!
//! # Rounding
//!
//! `Rounding::ContractNote` (default) returns unrounded accruals per fill and
//! leaves rounding to `ContractNoteAggregator`, once per trade day and note. The
//! contract-note rule is an assumption (`CONTRACT_NOTE_RULE`). `Rounding::Calculator`
//! reproduces the Zerodha brokerage calculator, to the paisa for equity and equity
//! derivatives and within one paisa for MCX futures, and exists for parity tests.

mod batch;
mod book;
mod breakdown;
mod brokerage;
mod builtin;
mod cost_model;
mod error;
mod note;
mod resolve;
mod rounding;
mod rules;
mod schedule;
mod segment;

#[cfg(test)]
mod tests;

pub use batch::CostColumns;
pub use book::{RateStatus, ScheduleBook, Trust};
pub use breakdown::{CostBreakdown, DpCharge, COMPONENT_NAMES};
pub use brokerage::{Brokerage, BrokerageRule, DpFeeSource};
pub use builtin::builtin_schedules;
pub use cost_model::CostModel;
pub use error::{CostError, NameKind};
pub use note::{contract_notes, ContractNoteAggregator, NoteLine, CONTRACT_NOTE_RULE};
pub use rounding::Rounding;
pub use schedule::{CostSchedule, SegmentRates, SegmentRow, TaxBase, TaxUnit};
pub use segment::{BrokerageClass, NoteGroup, Segment, SEGMENT_COUNT};
