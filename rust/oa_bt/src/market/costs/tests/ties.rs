//! Exact ties in contract-note rounding: a line that is exactly half way between two
//! rounding steps goes up, whatever the float that carried it looked like.
//!
//! Expected charges are worked out in integer arithmetic from the rupees-per-crore
//! figures of the reference note (sections 1 and 3), not by the engine.

use crate::model::Side;

use super::super::{ContractNoteAggregator, CostBreakdown, NoteGroup, Segment};
use super::{accrual_model, snapshot_day};

/// What the note of a day holding one fill worth `value` rupees charges.
fn single_fill_note(segment: Segment, side: Side, value: f64) -> CostBreakdown {
    let day = snapshot_day();
    let mut notes = ContractNoteAggregator::new();
    let cost = accrual_model()
        .fill_cost_value(segment, side, value, day)
        .unwrap();
    notes.add(day, segment, &cost);
    notes.note(day, segment.note_group()).unwrap().charged
}

fn stt(cost: &CostBreakdown) -> f64 {
    cost.stt
}

fn stamp_duty(cost: &CostBreakdown) -> f64 {
    cost.stamp_duty
}

fn sebi_fee(cost: &CostBreakdown) -> f64 {
    cost.sebi_fee
}

fn exchange_txn(cost: &CostBreakdown) -> f64 {
    cost.exchange_txn
}

/// One line of the rate card: its name, a segment and side that pay it, the charge in
/// rupees per crore of traded value, and where it appears in a breakdown.
type Line = (&'static str, Segment, Side, u64, fn(&CostBreakdown) -> f64);

/// Lines a note rounds to the rupee.
#[rustfmt::skip]
const TAX_LINES: [Line; 10] = [
    ("delivery STT, buy",   Segment::NseEqDelivery, Side::Buy,  10_000, stt),
    ("delivery STT, sell",  Segment::BseEqDelivery, Side::Sell, 10_000, stt),
    ("intraday STT",        Segment::NseEqIntraday, Side::Sell,  2_500, stt),
    ("futures STT",         Segment::NfoFut,        Side::Sell,  5_000, stt),
    ("options STT",         Segment::NfoOpt,        Side::Sell, 15_000, stt),
    ("commodity CTT",       Segment::McxFut,        Side::Sell,  1_000, stt),
    ("delivery stamp duty", Segment::NseEqDelivery, Side::Buy,   1_500, stamp_duty),
    ("intraday stamp duty", Segment::NseEqIntraday, Side::Buy,     300, stamp_duty),
    ("futures stamp duty",  Segment::NfoFut,        Side::Buy,     200, stamp_duty),
    ("options stamp duty",  Segment::BfoOpt,        Side::Buy,     300, stamp_duty),
];

/// Lines a note rounds to the paisa.
#[rustfmt::skip]
const CHARGE_LINES: [Line; 3] = [
    ("SEBI fee",            Segment::NseEqDelivery, Side::Buy,  10,  sebi_fee),
    ("NSE exchange charge", Segment::NseEqDelivery, Side::Sell, 307, exchange_txn),
    ("BSE exchange charge", Segment::BseEqIntraday, Side::Buy,  375, exchange_txn),
];

/// Paise per rupee times rupees per crore: the charge on Rs V is V x per_crore /
/// `PER_CRORE_IN_PAISE` paise.
const PER_CRORE_IN_PAISE: u64 = 100_000;
const LARGEST_VALUE: u64 = 300_000;
const RUPEE_IN_PAISE: u64 = 100;
const PAISA: u64 = 1;

/// Check every whole-rupee trade value up to Rs 3 lakh at which the line is exactly
/// half way between two steps of `unit_paise`. The expected charge is worked out in
/// integers: (2 x V x rate + step) / (2 x step), rounded down. Returns the number of
/// ties found.
fn assert_ties_round_up(line: &Line, unit_paise: u64) -> u32 {
    let (name, segment, side, per_crore, pick) = *line;
    let step = unit_paise * PER_CRORE_IN_PAISE;
    let mut ties = 0;
    for value in 1..=LARGEST_VALUE {
        let doubled = 2 * value * per_crore;
        if doubled % (2 * step) != step {
            continue;
        }
        let expected_rupees = ((doubled + step) / (2 * step) * unit_paise) as f64 / 100.0;
        let charged = single_fill_note(segment, side, value as f64);
        assert_eq!(pick(&charged), expected_rupees, "{name} on Rs {value}");
        ties += 1;
    }
    ties
}

#[test]
fn a_stamp_duty_of_exactly_one_rupee_fifty_is_charged_two_rupees() {
    // NSE delivery buy of 100 shares at Rs 100: 0.015 percent of Rs 10,000 is Rs 1.50
    // exactly, although the float product is 1.4999999999999998.
    let day = snapshot_day();
    let model = accrual_model();
    let mut notes = ContractNoteAggregator::new();
    let cost = model
        .fill_cost(Segment::NseEqDelivery, Side::Buy, 100.0, 100.0, day)
        .unwrap();
    notes.add(day, Segment::NseEqDelivery, &cost);
    assert_eq!(
        notes
            .note(day, NoteGroup::NseCash)
            .unwrap()
            .charged
            .stamp_duty,
        2.0
    );
}

#[test]
fn delivery_stamp_duty_ties_round_up_at_the_values_that_used_to_fall() {
    // 0.015 percent of 50,000 / 70,000 / 90,000 / 1,50,000 is 7.50 / 10.50 / 13.50 /
    // 22.50, charged as 8 / 11 / 14 / 23.
    let rows = [
        (50_000.0, 8.0),
        (70_000.0, 11.0),
        (90_000.0, 14.0),
        (150_000.0, 23.0),
    ];
    for (value, charged) in rows {
        let note = single_fill_note(Segment::NseEqDelivery, Side::Buy, value);
        assert_eq!(note.stamp_duty, charged, "Rs {value}");
    }
}

#[test]
fn every_tax_tie_up_to_three_lakh_rounds_up_to_the_rupee() {
    for line in &TAX_LINES {
        assert!(
            assert_ties_round_up(line, RUPEE_IN_PAISE) > 0,
            "{} has no tie to test",
            line.0
        );
    }
}

#[test]
fn every_charge_tie_up_to_three_lakh_rounds_up_to_the_paisa() {
    for line in &CHARGE_LINES {
        assert!(
            assert_ties_round_up(line, PAISA) > 0,
            "{} has no tie to test",
            line.0
        );
    }
}

#[test]
fn a_brokerage_of_exactly_one_and_a_half_paise_is_charged_two_paise() {
    // 0.03 percent of an intraday order of Rs 50 is Rs 0.015.
    let note = single_fill_note(Segment::NseEqIntraday, Side::Buy, 50.0);
    assert_eq!(note.brokerage, 0.02);
}

#[test]
fn many_small_fills_sum_exactly_before_rounding() {
    // (fills, value of each, stamp duty charged): 30 delivery buys of Rs 500 owe 7.5
    // paise each, Rs 2.25 in all, charged Rs 2; 20 buys of Rs 5,000 owe 75 paise
    // each, exactly Rs 15; 6 buys of Rs 2,500 owe 37.5 paise each, exactly Rs 2.25,
    // charged Rs 2; 2 buys of Rs 5,000 owe Rs 1.50 in all, charged Rs 2.
    let rows = [
        (30, 500.0, 2.0),
        (20, 5_000.0, 15.0),
        (6, 2_500.0, 2.0),
        (2, 5_000.0, 2.0),
    ];
    let day = snapshot_day();
    let model = accrual_model();
    for (fills, value, charged) in rows {
        let mut notes = ContractNoteAggregator::new();
        for _ in 0..fills {
            let cost = model
                .fill_cost_value(Segment::NseEqDelivery, Side::Buy, value, day)
                .unwrap();
            notes.add(day, Segment::NseEqDelivery, &cost);
        }
        let note = notes.note(day, NoteGroup::NseCash).unwrap();
        assert_eq!(
            note.charged.stamp_duty, charged,
            "{fills} fills of Rs {value}"
        );
    }
}
