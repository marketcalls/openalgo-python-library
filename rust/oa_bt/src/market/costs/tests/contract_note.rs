//! Contract-note rounding: accruals summed exactly per trade day and note, then
//! rounded.

use crate::model::{days_from_civil, Day, Side};

use super::super::{
    contract_notes, ContractNoteAggregator, CostBreakdown, CostError, NoteGroup, Segment,
    CONTRACT_NOTE_RULE,
};
use super::{accrual_model, assert_close, assert_paise, snapshot_day};

/// Accrue one Zerodha fill into `notes`.
fn accrue(
    notes: &mut ContractNoteAggregator,
    day: Day,
    segment: Segment,
    side: Side,
    price: f64,
    quantity: f64,
) {
    let cost = accrual_model()
        .fill_cost(segment, side, price, quantity, day)
        .unwrap();
    notes.add(day, segment, &cost);
}

/// A small option sale: one lot of 75 at a premium of Rs 4 owes STT of Rs 0.45.
fn accrue_small_option_sales(notes: &mut ContractNoteAggregator, day: Day, count: usize) {
    for _ in 0..count {
        accrue(notes, day, Segment::NfoOpt, Side::Sell, 4.0, 75.0);
    }
}

#[test]
fn the_day_note_of_each_worked_example_matches_the_calculator_total() {
    // Section 6: a day holding only the example's buy and sell is charged the
    // calculator's total.
    let rows = [
        (Segment::NseEqIntraday, 1_000.0, 1_100.0, 200.62),
        (Segment::NseEqDelivery, 1_000.0, 1_100.0, 931.42),
        (Segment::NfoFut, 1_000.0, 1_100.0, 294.33),
        (Segment::NfoOpt, 100.0, 110.0, 149.52),
    ];
    let day = snapshot_day();
    for (segment, buy, sell, total) in rows {
        let mut notes = ContractNoteAggregator::new();
        accrue(&mut notes, day, segment, Side::Buy, buy, 400.0);
        accrue(&mut notes, day, segment, Side::Sell, sell, 400.0);
        assert_eq!(notes.total_charged(), total, "{segment:?}");
    }
}

#[test]
fn tax_is_rounded_once_per_day_not_once_per_fill() {
    // Ten small option sales owe 10 x 0.45 = Rs 4.50 of STT, charged as Rs 5.
    // Rounded fill by fill they would owe nothing.
    let day = snapshot_day();
    let mut notes = ContractNoteAggregator::new();
    accrue_small_option_sales(&mut notes, day, 10);
    let note = notes.note(day, NoteGroup::NseFo).unwrap();
    assert_eq!(
        (note.fills, note.accrued.stt, note.charged.stt),
        (10, 4.5, 5.0)
    );
}

#[test]
fn flat_brokerage_is_charged_per_fill_on_the_note() {
    let day = snapshot_day();
    let mut notes = ContractNoteAggregator::new();
    accrue_small_option_sales(&mut notes, day, 10);
    assert_paise(
        notes.note(day, NoteGroup::NseFo).unwrap().charged.brokerage,
        200.0,
    );
}

#[test]
fn delivery_and_intraday_trades_of_one_exchange_share_a_note() {
    let day = snapshot_day();
    let mut notes = ContractNoteAggregator::new();
    accrue(
        &mut notes,
        day,
        Segment::NseEqDelivery,
        Side::Buy,
        512.3,
        37.0,
    );
    accrue(
        &mut notes,
        day,
        Segment::NseEqIntraday,
        Side::Sell,
        98.75,
        211.0,
    );
    assert_eq!(notes.len(), 1);
    assert_eq!(notes.note(day, NoteGroup::NseCash).unwrap().fills, 2);
}

#[test]
fn futures_and_options_of_one_exchange_share_a_note() {
    let day = snapshot_day();
    let mut notes = ContractNoteAggregator::new();
    accrue(&mut notes, day, Segment::NfoFut, Side::Sell, 24_012.5, 75.0);
    accrue(&mut notes, day, Segment::NfoOpt, Side::Buy, 132.4, 150.0);
    assert_eq!(notes.len(), 1);
    assert_eq!(notes.note(day, NoteGroup::NseFo).unwrap().fills, 2);
}

#[test]
fn notes_are_keyed_by_day_and_exchange_segment() {
    let d1 = snapshot_day();
    let d2 = d1 + 1;
    let mut notes = ContractNoteAggregator::new();
    accrue(
        &mut notes,
        d2,
        Segment::NseEqDelivery,
        Side::Sell,
        530.0,
        37.0,
    );
    accrue(&mut notes, d1, Segment::McxFut, Side::Sell, 5_512.0, 100.0);
    accrue(
        &mut notes,
        d1,
        Segment::BfoOptOther,
        Side::Sell,
        310.0,
        20.0,
    );
    accrue(
        &mut notes,
        d1,
        Segment::BseEqDeliveryX,
        Side::Buy,
        12.3,
        1_000.0,
    );
    accrue(
        &mut notes,
        d1,
        Segment::NseEqDelivery,
        Side::Buy,
        512.3,
        37.0,
    );
    let keys: Vec<(Day, NoteGroup)> = notes.notes().map(|note| (note.day, note.group)).collect();
    assert_eq!(
        keys,
        vec![
            (d1, NoteGroup::NseCash),
            (d1, NoteGroup::BseCash),
            (d1, NoteGroup::BseFo),
            (d1, NoteGroup::Mcx),
            (d2, NoteGroup::NseCash),
        ]
    );
}

#[test]
fn the_adjustment_is_charged_minus_accrued() {
    // Three small option sales: STT accrued 1.35, charged Rs 1, so the tax line
    // moves by -0.35. The other lines are already whole paise except exchange
    // (0.31977 charged 0.32) and GST (10.8575586 charged 10.86).
    let day = snapshot_day();
    let mut notes = ContractNoteAggregator::new();
    accrue_small_option_sales(&mut notes, day, 3);
    let note = notes.note(day, NoteGroup::NseFo).unwrap();
    assert_close(note.charged.stt - note.accrued.stt, -0.35);
    assert_close(note.adjustment(), note.charged_total - note.accrued.total());
}

#[test]
fn exercise_stt_joins_the_note_of_the_expiry_day() {
    // STT of Rs 3.75 on an exercised option is charged as Rs 4 on that day's note.
    let day = snapshot_day();
    let mut notes = ContractNoteAggregator::new();
    notes.add_stt(day, Segment::NfoOpt, 3.75);
    let note = notes.note(day, NoteGroup::NseFo).unwrap();
    assert_eq!(
        note.charged,
        CostBreakdown {
            stt: 4.0,
            ..CostBreakdown::ZERO
        }
    );
}

#[test]
fn a_new_aggregator_is_empty() {
    let notes = ContractNoteAggregator::new();
    assert!(notes.is_empty());
    assert_eq!(notes.total_charged(), 0.0);
}

#[test]
fn the_rule_text_says_it_is_an_assumption() {
    assert!(CONTRACT_NOTE_RULE.starts_with("assumed, not verified"));
}

/// Two fills on one day and note, as the columns the batch function takes: STT of
/// 0.45 and 0.30, brokerage of 20 each, nothing else.
fn two_fill_columns() -> ([Day; 2], [u8; 2], [[f64; 2]; 6]) {
    let day = snapshot_day();
    let segments = [Segment::NfoOpt as u8, Segment::NfoFut as u8];
    let columns = [
        [20.0, 20.0],
        [0.45, 0.30],
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
    ];
    ([day, day], segments, columns)
}

fn as_slices(columns: &[[f64; 2]; 6]) -> [&[f64]; 6] {
    [
        &columns[0],
        &columns[1],
        &columns[2],
        &columns[3],
        &columns[4],
        &columns[5],
    ]
}

#[test]
fn the_batch_function_sums_and_rounds_each_note() {
    // STT 0.45 + 0.30 = 0.75, charged as Rs 1; brokerage 40.
    let (days, segments, columns) = two_fill_columns();
    let notes = contract_notes(&days, &segments, as_slices(&columns)).unwrap();
    assert_eq!(notes.len(), 1);
    assert_eq!((notes[0].group, notes[0].fills), (NoteGroup::NseFo, 2));
    assert_eq!(
        notes[0].charged,
        CostBreakdown {
            brokerage: 40.0,
            stt: 1.0,
            ..CostBreakdown::ZERO
        }
    );
    assert_eq!(notes[0].charged_total, 41.0);
}

#[test]
fn the_batch_function_reports_the_row_of_an_unknown_segment() {
    let (days, _, columns) = two_fill_columns();
    let result = contract_notes(&days, &[Segment::NfoOpt as u8, 99], as_slices(&columns));
    assert_eq!(result, Err((1, CostError::UnknownSegmentId(99))));
}

#[test]
fn the_batch_function_rejects_a_negative_amount() {
    let (days, segments, mut columns) = two_fill_columns();
    columns[1][1] = -1.0;
    let result = contract_notes(&days, &segments, as_slices(&columns));
    assert_eq!(
        result,
        Err((
            1,
            CostError::InvalidInput {
                field: "stt",
                value: -1.0
            }
        ))
    );
}

#[test]
fn the_batch_function_rejects_a_date_outside_the_supported_range() {
    let (_, segments, columns) = two_fill_columns();
    let days = [days_from_civil(2026, 11, 12), Day::MAX];
    let result = contract_notes(&days, &segments, as_slices(&columns));
    assert_eq!(
        result,
        Err((1, CostError::UnsupportedDay { day: Day::MAX }))
    );
}

#[test]
fn the_batch_function_rejects_columns_of_different_lengths() {
    let (days, segments, columns) = two_fill_columns();
    let result = contract_notes(&days[..1], &segments, as_slices(&columns));
    assert!(matches!(result, Err((0, CostError::LengthMismatch { .. }))));
}
