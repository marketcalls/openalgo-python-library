//! Sharing what a contract note charges among the fills accrued into it.

use crate::model::Side;

use super::super::{apportion, ContractNoteAggregator, CostBreakdown, NoteLine, Segment};
use super::{accrual_model, assert_close, snapshot_day};

/// The note of `fills` (segment, side, price, quantity), all on the snapshot day, and
/// each fill's unrounded cost.
fn note_of(fills: &[(Segment, Side, f64, f64)]) -> (NoteLine, Vec<CostBreakdown>) {
    let mut notes = ContractNoteAggregator::new();
    let costs: Vec<CostBreakdown> = fills
        .iter()
        .map(|&(segment, side, price, quantity)| {
            let cost = accrual_model()
                .fill_cost(segment, side, price, quantity, snapshot_day())
                .unwrap();
            notes.add(snapshot_day(), segment, &cost);
            cost
        })
        .collect();
    let group = fills[0].0.note_group();
    (notes.note(snapshot_day(), group).unwrap(), costs)
}

fn paise(rupees: f64) -> i64 {
    (rupees * 100.0).round() as i64
}

fn is_whole_paise(rupees: f64) -> bool {
    (rupees * 100.0 - (rupees * 100.0).round()).abs() < 1e-6
}

#[test]
fn the_shares_add_up_to_what_the_note_charges() {
    let (note, costs) = note_of(&[
        (Segment::NseEqIntraday, Side::Buy, 1_000.35, 37.0),
        (Segment::NseEqIntraday, Side::Sell, 1_101.30, 37.0),
        (Segment::NseEqDelivery, Side::Buy, 512.45, 113.0),
        (Segment::NseEqIntraday, Side::Buy, 87.15, 1_000.0),
        (Segment::NseEqIntraday, Side::Sell, 88.40, 1_000.0),
    ]);
    let shares = apportion(&note, &costs);
    assert_eq!(shares.len(), costs.len());
    assert!(
        shares.iter().all(|&share| is_whole_paise(share)),
        "{shares:?}"
    );
    let total: i64 = shares.iter().map(|&share| paise(share)).sum();
    assert_eq!(total, paise(note.charged_total));
}

#[test]
fn a_note_of_one_fill_is_that_fills_charge() {
    let (note, costs) = note_of(&[(Segment::NseEqIntraday, Side::Sell, 1_101.30, 400.0)]);
    assert_eq!(apportion(&note, &costs), vec![note.charged_total]);
}

#[test]
fn a_line_is_shared_in_proportion_to_what_each_fill_accrued_of_it() {
    // Two sales of the same value: the STT line, rounded once on the note, is shared
    // equally, and each sale pays half the note.
    let sale = (Segment::NseEqIntraday, Side::Sell, 1_101.30, 400.0);
    let (note, costs) = note_of(&[sale, sale]);
    let shares = apportion(&note, &costs);
    assert!(
        (paise(shares[0]) - paise(shares[1])).abs() <= 1,
        "{shares:?}"
    );
    assert_close(shares[0] + shares[1], note.charged_total);
}

#[test]
fn a_fill_that_paid_none_of_a_line_takes_none_of_its_rounding() {
    // Intraday STT is on the sale alone: the buy pays its own accrual, plus the
    // rounding of the stamp duty it alone paid, give or take the paise of the lines
    // rounded to the paisa.
    let (note, costs) = note_of(&[
        (Segment::NseEqIntraday, Side::Buy, 1_000.35, 400.0),
        (Segment::NseEqIntraday, Side::Sell, 1_101.30, 400.0),
    ]);
    let shares = apportion(&note, &costs);
    let buy = &costs[0];
    assert_eq!(buy.stt, 0.0);
    let expected = buy.total() + note.charged.stamp_duty - buy.stamp_duty;
    assert!(
        (shares[0] - expected).abs() < 0.03,
        "{} vs {expected}",
        shares[0]
    );
}

#[test]
fn the_paise_left_over_go_to_the_earlier_fill_on_a_tie() {
    // Three identical small option sales: equal exact shares, so the paise that
    // rounding leaves over go to the earliest fills, one each.
    let sale = (Segment::NfoOpt, Side::Sell, 4.0, 75.0);
    let (note, costs) = note_of(&[sale, sale, sale]);
    let shares: Vec<i64> = apportion(&note, &costs).into_iter().map(paise).collect();
    assert!(
        shares.windows(2).all(|pair| pair[0] >= pair[1]),
        "{shares:?}"
    );
    assert!(shares[0] - shares[2] <= 1, "{shares:?}");
    assert_eq!(shares.iter().sum::<i64>(), paise(note.charged_total));
}

#[test]
fn a_note_with_no_fill_given_has_no_shares() {
    let (note, _) = note_of(&[(Segment::NseEqIntraday, Side::Buy, 1_000.0, 400.0)]);
    assert!(apportion(&note, &[]).is_empty());
}
