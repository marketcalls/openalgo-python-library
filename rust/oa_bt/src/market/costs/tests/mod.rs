//! Unit tests for the cost engine, one file per topic.
//!
//! Expected figures come from the reference note `indian-trading-costs.md` (its
//! section and row are named in each test) or are worked out by hand in the test's
//! comment. None is produced by running the engine.

mod batch;
mod charges;
mod contract_note;
mod data_file;
mod errors;
mod invariants;
mod mcx;
mod presets;
mod rate_card;
mod schedules;
mod segments;
mod ties;
mod user_schedules;
mod worked_examples;

use crate::model::{days_from_civil, Day, Side};

use super::{
    Brokerage, CostBreakdown, CostError, CostModel, Rounding, ScheduleBook, Segment, SegmentRates,
};

/// One crore rupees: rates quoted "per crore" read directly off a fill this size.
const CRORE: f64 = 10_000_000.0;

/// The day of the reference snapshot, the last day the current rates are verified
/// for.
fn snapshot_day() -> Day {
    days_from_civil(2026, 11, 12)
}

fn builtin_book() -> ScheduleBook {
    ScheduleBook::builtin().unwrap()
}

/// Built-in schedules, Zerodha brokerage, unrounded accruals.
fn accrual_model() -> CostModel {
    CostModel::builtin().unwrap()
}

/// Built-in schedules, Zerodha brokerage, calculator rounding.
fn calculator_model() -> CostModel {
    CostModel::new(builtin_book(), Brokerage::zerodha(), Rounding::Calculator)
}

/// Built-in schedules and unrounded accruals under another brokerage preset.
fn model_with(brokerage: Brokerage) -> CostModel {
    CostModel::new(builtin_book(), brokerage, Rounding::ContractNote)
}

/// A round trip with both legs on one day, which is what most tests cost.
trait SameDay {
    fn round_trip_on(
        &self,
        segment: Segment,
        buy_price: f64,
        sell_price: f64,
        quantity: f64,
        day: Day,
    ) -> Result<CostBreakdown, CostError>;
}

impl SameDay for CostModel {
    fn round_trip_on(
        &self,
        segment: Segment,
        buy_price: f64,
        sell_price: f64,
        quantity: f64,
        day: Day,
    ) -> Result<CostBreakdown, CostError> {
        self.round_trip_cost(segment, buy_price, sell_price, quantity, day, day)
    }
}

/// Unrounded Zerodha cost of one fill worth `value` rupees on the snapshot day.
fn accrued(segment: Segment, side: Side, value: f64) -> CostBreakdown {
    accrual_model()
        .fill_cost_value(segment, side, value, snapshot_day())
        .unwrap()
}

/// The built-in rates of a segment on the snapshot day.
fn current_rates(segment: Segment) -> SegmentRates {
    let book = builtin_book();
    let (index, _) = book.lookup(snapshot_day()).unwrap();
    book.versions()[index].row(segment).unwrap().rates
}

#[track_caller]
fn assert_close(actual: f64, expected: f64) {
    let tolerance = 1e-9 * (1.0 + actual.abs().max(expected.abs()));
    assert!(
        (actual - expected).abs() <= tolerance,
        "{actual} is not {expected}"
    );
}

/// Assert that `actual` is `expected` and is a whole number of paise.
#[track_caller]
fn assert_paise(actual: f64, expected: f64) {
    let paise = |rupees: f64| (rupees * 100.0).round() as i64;
    assert_eq!(
        paise(actual),
        paise(expected),
        "{actual} is not {expected} to the paisa"
    );
    assert!(
        (actual - expected).abs() < 1e-6,
        "{actual} is not rounded to the paisa"
    );
}

/// A small deterministic generator, so the property tests need no dependency.
struct Lcg(u64);

impl Lcg {
    /// The next value, uniform in [0, 1).
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
    }
}
