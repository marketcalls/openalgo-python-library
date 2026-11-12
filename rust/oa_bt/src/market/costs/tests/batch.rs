//! The batch API costs each row exactly as the per-fill API does.

use crate::model::{days_from_civil, Day, Side};

use super::super::{CostColumns, CostError, CostModel, Segment, Trust, SEGMENT_COUNT};
use super::{accrual_model, calculator_model, snapshot_day, Lcg};

/// Owned output columns for a batch of `rows` fills.
struct Columns {
    amounts: [Vec<f64>; 7],
    schedule: Vec<u16>,
    trust: Vec<u8>,
}

impl Columns {
    fn new(rows: usize) -> Columns {
        Columns {
            amounts: std::array::from_fn(|_| vec![f64::NAN; rows]),
            schedule: vec![u16::MAX; rows],
            trust: vec![u8::MAX; rows],
        }
    }

    fn view(&mut self) -> CostColumns<'_> {
        let [brokerage, stt, exchange_txn, sebi_fee, stamp_duty, gst, total] = &mut self.amounts;
        CostColumns {
            brokerage,
            stt,
            exchange_txn,
            sebi_fee,
            stamp_duty,
            gst,
            total,
            schedule: &mut self.schedule,
            trust: &mut self.trust,
        }
    }

    fn row(&self, row: usize) -> [f64; 7] {
        std::array::from_fn(|column| self.amounts[column][row])
    }
}

/// Four valid fills on the snapshot day.
struct Batch {
    segments: [u8; 4],
    sides: [u8; 4],
    prices: [f64; 4],
    quantities: [f64; 4],
    days: [Day; 4],
}

impl Batch {
    fn valid() -> Batch {
        Batch {
            segments: [0, 1, 10, 11],
            sides: [0, 1, 1, 0],
            prices: [100.0; 4],
            quantities: [1.0; 4],
            days: [snapshot_day(); 4],
        }
    }

    fn cost_into(&self, out: &mut Columns) -> Result<(), (usize, CostError)> {
        accrual_model().fill_cost_batch(
            &self.segments,
            &self.sides,
            &self.prices,
            &self.quantities,
            &self.days,
            &mut out.view(),
        )
    }

    fn cost(&self) -> Result<(), (usize, CostError)> {
        self.cost_into(&mut Columns::new(4))
    }
}

fn batch_matches_per_fill(model: &CostModel) {
    let days = [
        days_from_civil(2025, 6, 2),
        days_from_civil(2026, 3, 31),
        snapshot_day(),
    ];
    let mut rng = Lcg(99);
    let rows = 300;
    let segments: Vec<u8> = (0..rows)
        .map(|_| (rng.next() * SEGMENT_COUNT as f64) as u8)
        .collect();
    let sides: Vec<u8> = (0..rows).map(|_| (rng.next() * 2.0) as u8).collect();
    let prices: Vec<f64> = (0..rows).map(|_| rng.next() * 5_000.0).collect();
    let quantities: Vec<f64> = (0..rows).map(|_| (rng.next() * 500.0).floor()).collect();
    let trade_days: Vec<Day> = (0..rows)
        .map(|_| days[(rng.next() * 3.0) as usize])
        .collect();

    let mut out = Columns::new(rows);
    model
        .fill_cost_batch(
            &segments,
            &sides,
            &prices,
            &quantities,
            &trade_days,
            &mut out.view(),
        )
        .unwrap();

    for row in 0..rows {
        let segment = Segment::from_u8(segments[row]).unwrap();
        let side = Side::from_u8(sides[row]).unwrap();
        let cost = model
            .fill_cost(segment, side, prices[row], quantities[row], trade_days[row])
            .unwrap();
        let [b, s, e, f, d, g] = cost.components();
        assert_eq!(
            out.row(row),
            [b, s, e, f, d, g, model.reported_total(&cost)],
            "row {row}"
        );
        let status = model.status(segment, trade_days[row]).unwrap();
        assert_eq!(
            (out.schedule[row] as usize, out.trust[row]),
            (status.schedule, status.trust as u8)
        );
    }
}

#[test]
fn batch_equals_per_fill_in_contract_note_mode() {
    batch_matches_per_fill(&accrual_model());
}

#[test]
fn batch_equals_per_fill_in_calculator_mode() {
    batch_matches_per_fill(&calculator_model());
}

#[test]
fn a_batch_reports_which_version_priced_each_row_and_how_far_it_is_verified() {
    // Old version (never verified), the snapshot day (verified), the day after
    // (past the verified-through date), and BSE stock options (unverified row).
    let mut batch = Batch::valid();
    batch.days = [
        days_from_civil(2025, 6, 2),
        snapshot_day(),
        snapshot_day() + 1,
        snapshot_day(),
    ];
    batch.segments[3] = Segment::BfoOptOther as u8;
    let mut out = Columns::new(4);
    batch.cost_into(&mut out).unwrap();
    assert_eq!(out.schedule, [0, 1, 1, 1]);
    let trust = [
        Trust::Unverified,
        Trust::Verified,
        Trust::PastVerification,
        Trust::Unverified,
    ];
    assert_eq!(out.trust, trust.map(|t| t as u8));
}

#[test]
fn an_empty_batch_succeeds() {
    let mut out = Columns::new(0);
    let result = accrual_model().fill_cost_batch(&[], &[], &[], &[], &[], &mut out.view());
    assert_eq!(result, Ok(()));
}

#[test]
fn an_unknown_segment_id_is_reported_with_its_row() {
    let mut batch = Batch::valid();
    batch.segments[2] = 200;
    assert_eq!(batch.cost(), Err((2, CostError::UnknownSegmentId(200))));
}

#[test]
fn an_unknown_side_is_reported_with_its_row() {
    let mut batch = Batch::valid();
    batch.sides[1] = 2;
    assert_eq!(batch.cost(), Err((1, CostError::UnknownSide(2))));
}

#[test]
fn a_nan_price_is_reported_with_its_row() {
    let mut batch = Batch::valid();
    batch.prices[3] = f64::NAN;
    assert!(matches!(
        batch.cost(),
        Err((3, CostError::InvalidInput { field: "price", .. }))
    ));
}

#[test]
fn a_negative_quantity_is_reported_with_its_row() {
    let mut batch = Batch::valid();
    batch.quantities[1] = -5.0;
    assert_eq!(
        batch.cost(),
        Err((
            1,
            CostError::InvalidInput {
                field: "quantity",
                value: -5.0
            }
        ))
    );
}

#[test]
fn a_traded_value_that_overflows_is_reported_with_its_row() {
    let mut batch = Batch::valid();
    (batch.prices[2], batch.quantities[2]) = (1e200, 1e200);
    assert!(matches!(
        batch.cost(),
        Err((2, CostError::InvalidInput { .. }))
    ));
}

#[test]
fn a_date_before_the_first_version_is_reported_with_its_row() {
    let mut batch = Batch::valid();
    batch.days[3] = days_from_civil(2019, 1, 1);
    assert!(matches!(
        batch.cost(),
        Err((3, CostError::NoScheduleForDay { .. }))
    ));
}

#[test]
fn a_date_outside_the_supported_range_is_reported_with_its_row() {
    let mut batch = Batch::valid();
    batch.days[0] = Day::MAX;
    assert_eq!(
        batch.cost(),
        Err((0, CostError::UnsupportedDay { day: Day::MAX }))
    );
}

#[test]
fn inputs_of_different_lengths_are_rejected() {
    let batch = Batch::valid();
    let mut out = Columns::new(4);
    let result = accrual_model().fill_cost_batch(
        &batch.segments,
        &batch.sides[..3],
        &batch.prices,
        &batch.quantities,
        &batch.days,
        &mut out.view(),
    );
    assert_eq!(
        result,
        Err((
            0,
            CostError::LengthMismatch {
                name: "sides",
                expected: 4,
                got: 3
            }
        ))
    );
}

#[test]
fn output_columns_of_the_wrong_length_are_rejected() {
    let result = Batch::valid().cost_into(&mut Columns::new(3));
    assert_eq!(
        result,
        Err((
            0,
            CostError::LengthMismatch {
                name: "out.brokerage",
                expected: 4,
                got: 3
            }
        ))
    );
}
