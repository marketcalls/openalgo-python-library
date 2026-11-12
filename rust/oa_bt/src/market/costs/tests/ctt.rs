//! CTT on MCX futures by commodity class.
//!
//! CTT is 0.01 percent of the sale of a commodity derivative, paid by the seller
//! (Finance Act 2013 section 117), unless its commodity is agricultural (section
//! 116(7)); the agricultural commodities are the 61 of rule 3 of the CTT Rules 2013
//! as substituted by notification 13/2015. So an MCX agri contract on a commodity in
//! rule 3 (cardamom) pays none, one on a commodity outside it (crude palm oil) pays
//! it, and so does a non-agri contract (crude oil). Each class is resolved from its
//! commodity name and costed on a contract note worked by hand.

use crate::model::{days_from_civil, Day, Exchange, InstrumentKind, Product, Side};

use super::super::{builtin_schedules, ContractNoteAggregator, CostBreakdown, NoteGroup, Segment};
use super::{accrual_model, assert_close, assert_paise, snapshot_day, CRORE};

/// The contract note of one day holding a buy and a sale of `quantity` units of an
/// MCX future on `commodity`, costed in contract-note mode.
fn mcx_note(commodity: &str, day: Day, buy: f64, sell: f64, quantity: f64) -> (CostBreakdown, f64) {
    let segment = Segment::resolve(
        Exchange::Mcx,
        InstrumentKind::Future,
        Product::Nrml,
        None,
        Some(commodity),
    )
    .unwrap();
    let model = accrual_model();
    let mut notes = ContractNoteAggregator::new();
    for (side, price) in [(Side::Buy, buy), (Side::Sell, sell)] {
        let cost = model
            .fill_cost(segment, side, price, quantity, day)
            .unwrap();
        notes.add(day, segment, &cost);
    }
    let note = notes.note(day, NoteGroup::Mcx).unwrap();
    (note.charged, note.charged_total)
}

#[track_caller]
fn assert_note(actual: (CostBreakdown, f64), expected: [f64; 7]) {
    let (lines, total) = actual;
    let [brokerage, ctt, exchange, sebi, stamp, gst, expected_total] = expected;
    assert_paise(lines.brokerage, brokerage);
    assert_paise(lines.stt, ctt);
    assert_paise(lines.exchange_txn, exchange);
    assert_paise(lines.sebi_fee, sebi);
    assert_paise(lines.stamp_duty, stamp);
    assert_paise(lines.gst, gst);
    assert_paise(total, expected_total);
}

#[test]
fn a_cardamom_note_has_no_ctt_because_cardamom_is_in_rule_3() {
    // 100 kg bought at 2,380 and sold at 2,415 on 2026-11-12 (IN-2026-04-01):
    //   buy 2,38,000, sell 2,41,500, turnover 4,79,500
    //   brokerage 20 + 20 (0.03 percent of each leg is over Rs 20) = 40
    //   CTT none
    //   exchange 0.0021 percent of turnover = 10.0695 -> 10.07
    //   SEBI Re 1 per crore = 0.04795 -> 0.05
    //   stamp 0.002 percent of the buy = 4.76 -> Rs 5
    //   GST 0.18 x (40 + 10.0695 + 0.04795) = 9.021141 -> 9.02
    //   total 40 + 10.07 + 0.05 + 5 + 9.02 = 64.14
    let note = mcx_note("CARDAMOM", snapshot_day(), 2_380.0, 2_415.0, 100.0);
    assert_note(note, [40.0, 0.0, 10.07, 0.05, 5.0, 9.02, 64.14]);
}

#[test]
fn a_cpo_note_has_ctt_because_crude_palm_oil_is_not_in_rule_3() {
    // 1,000 units (one lot) bought at 1,012.50 and sold at 1,031.70 on 2026-11-12:
    //   buy 10,12,500, sell 10,31,700, turnover 20,44,200
    //   brokerage 20 + 20 = 40
    //   CTT 0.01 percent of the sell = 103.17 -> Rs 103
    //   exchange 0.0021 percent = 42.9282 -> 42.93
    //   SEBI Re 1 per crore (an agri contract) = 0.20442 -> 0.20
    //   stamp 0.002 percent of the buy = 20.25 -> Rs 20
    //   GST 0.18 x (40 + 42.9282 + 0.20442) = 14.9638716 -> 14.96
    //   total 40 + 103 + 42.93 + 0.20 + 20 + 14.96 = 221.09
    let note = mcx_note("CPO", snapshot_day(), 1_012.5, 1_031.7, 1_000.0);
    assert_note(note, [40.0, 103.0, 42.93, 0.20, 20.0, 14.96, 221.09]);
}

#[test]
fn a_cpo_note_of_2022_has_ctt_at_that_year_exchange_rate() {
    // The same trade on 2022-08-01 (IN-2022-07-18): exchange Rs 2.60 per lakh.
    //   CTT 103.17 -> Rs 103; exchange 0.0026 percent = 53.1492 -> 53.15
    //   SEBI 0.20442 -> 0.20; stamp 20.25 -> Rs 20
    //   GST 0.18 x (40 + 53.1492 + 0.20442) = 16.8036516 -> 16.80
    //   total 40 + 103 + 53.15 + 0.20 + 20 + 16.80 = 233.15
    let note = mcx_note(
        "CPO",
        days_from_civil(2022, 8, 1),
        1_012.5,
        1_031.7,
        1_000.0,
    );
    assert_note(note, [40.0, 103.0, 53.15, 0.20, 20.0, 16.80, 233.15]);
}

#[test]
fn a_crudeoil_note_has_ctt_because_crude_oil_is_not_agricultural() {
    // 100 barrels bought at 5,487 and sold at 5,523 on 2026-11-12:
    //   buy 5,48,700, sell 5,52,300, turnover 11,01,000
    //   brokerage 20 + 20 = 40
    //   CTT 0.01 percent of the sell = 55.23 -> Rs 55
    //   exchange 0.0021 percent = 23.121 -> 23.12
    //   SEBI Rs 10 per crore = 1.101 -> 1.10
    //   stamp 0.002 percent of the buy = 10.974 -> Rs 11
    //   GST 0.18 x (40 + 23.121 + 1.101) = 11.55996 -> 11.56
    //   total 40 + 55 + 23.12 + 1.10 + 11 + 11.56 = 141.78
    let note = mcx_note("CRUDEOIL", snapshot_day(), 5_487.0, 5_523.0, 100.0);
    assert_note(note, [40.0, 55.0, 23.12, 1.10, 11.0, 11.56, 141.78]);
}

#[test]
fn every_version_charges_ctt_on_sells_exactly_outside_rule_3() {
    // 0.01 percent of a Rs 1 crore sale is Rs 1,000; a purchase pays none.
    use Segment::*;
    let paying = [McxFut, McxFutAgriCtt, McxFutRbdpmolein];
    let exempt = [McxFutAgri, McxFutCastorseed, McxFutPepper, McxFutKapas];
    let model = accrual_model();
    let ctt = |segment, side, day| {
        model
            .fill_cost_value(segment, side, CRORE, day)
            .unwrap()
            .stt
    };
    for version in builtin_schedules().unwrap() {
        let day = version.valid_from;
        for segment in paying {
            assert_close(ctt(segment, Side::Sell, day), 1_000.0);
            assert_eq!(
                ctt(segment, Side::Buy, day),
                0.0,
                "{} {segment:?}",
                version.id
            );
        }
        for segment in exempt {
            for side in [Side::Buy, Side::Sell] {
                assert_eq!(ctt(segment, side, day), 0.0, "{} {segment:?}", version.id);
            }
        }
    }
}
