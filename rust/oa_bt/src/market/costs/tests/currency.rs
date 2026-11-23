//! NSE currency futures and options (CDS): the reference note's section 2, the
//! maintainer's calculator screenshot of 2026-11-23 (USDINR, one lot of USD 1,000, so
//! 1,000 units of the quoted price) and the dated rows of `schedules.csv`. Every figure
//! is worked by hand in the test's comment.

use crate::model::{days_from_civil, Day, Side};

use super::super::{
    Brokerage, ContractNoteAggregator, CostModel, NoteGroup, Rounding, Segment, Trust,
};
use super::{assert_close, assert_paise, builtin_book, calculator_model, SameDay, CRORE};

/// The day of the screenshot, the day the current currency rows are verified through.
fn screenshot_day() -> Day {
    days_from_civil(2026, 11, 23)
}

/// One USDINR lot: a contract of USD 1,000 quoted in rupees per dollar.
const ONE_LOT_UNITS: f64 = 1_000.0;
const BUY: f64 = 88.33;
const SELL: f64 = 88.3525;

fn zerodha_accruals() -> CostModel {
    CostModel::new(builtin_book(), Brokerage::zerodha(), Rounding::ContractNote)
}

#[test]
fn currency_futures_reproduce_the_calculator_screenshot() {
    // Turnover (88.33 + 88.3525) x 1,000 = 1,76,682.50. Brokerage: each leg is worth
    // about Rs 88,000, 0.03 percent is Rs 26.50, so Rs 20 a leg, 40. Exchange line:
    // 0.00035 percent is 0.6184, shown 0.62, plus IPFT 0.00005 percent 0.0883, shown
    // 0.09: 0.71. SEBI Rs 10 per crore: 0.1767, 0.18. GST leaves the IPFT out, as the
    // calculator does: 18 percent of 40 + 0.62 + 0.18 = 7.344, 7.34. Stamp 0.0001
    // percent of 88,330 is 0.09, rounded to the rupee: 0. Total, without the IPFT the
    // exchange line shows: 40 + 0.62 + 0.18 + 7.34 = 48.14.
    let model = calculator_model();
    let cost = model
        .round_trip_on(Segment::CdsFut, BUY, SELL, ONE_LOT_UNITS, screenshot_day())
        .unwrap();
    assert_paise(cost.brokerage, 40.0);
    assert_paise(cost.exchange_txn, 0.71);
    assert_paise(cost.gst, 7.34);
    assert_paise(cost.sebi_fee, 0.18);
    assert_paise(cost.stamp_duty, 0.0);
    assert_paise(cost.stt, 0.0);
    assert_paise(cost.shown_not_charged, 0.09);
    assert_paise(model.reported_total(&cost), 48.14);
}

#[test]
fn currency_options_reproduce_the_calculator_screenshot() {
    // The same prices as premiums (strike 88.50). Premium turnover 1,76,682.50.
    // Brokerage Rs 20 a leg, 40. Exchange: 0.0311 percent is 54.948, shown 54.95, plus
    // IPFT 0.002 percent 3.534, shown 3.53: 58.48. SEBI 0.18. GST includes the IPFT:
    // 18 percent of 40 + 58.48 + 0.18 = 17.7588, 17.76. Stamp 0. Total 116.42.
    let model = calculator_model();
    let cost = model
        .round_trip_on(Segment::CdsOpt, BUY, SELL, ONE_LOT_UNITS, screenshot_day())
        .unwrap();
    assert_paise(cost.brokerage, 40.0);
    assert_paise(cost.exchange_txn, 58.48);
    assert_paise(cost.gst, 17.76);
    assert_paise(cost.sebi_fee, 0.18);
    assert_paise(cost.stamp_duty, 0.0);
    assert_eq!(cost.shown_not_charged, 0.0);
    assert_paise(model.reported_total(&cost), 116.42);
}

#[test]
fn accruals_charge_what_the_exchange_bills_ipft_and_gst_on_it_included() {
    // A buy of Rs 1 crore. Futures: exchange (0.00035 + 0.00005) percent = 40, SEBI 10,
    // stamp 0.0001 percent = 10, no STT, GST 18 percent of 20 + 40 + 10 = 12.60.
    // Options on Rs 1 crore of premium: exchange (0.0311 + 0.002) percent = 3,310, SEBI
    // 10, stamp 10, GST 18 percent of 20 + 3,310 + 10 = 601.20.
    let rows = [
        (Segment::CdsFut, 40.0, 12.6),
        (Segment::CdsOpt, 3_310.0, 601.2),
    ];
    for (segment, exchange, gst) in rows {
        let cost = zerodha_accruals()
            .fill_cost_value(segment, Side::Buy, CRORE, screenshot_day())
            .unwrap();
        assert_close(cost.brokerage, 20.0);
        assert_close(cost.exchange_txn, exchange);
        assert_close(cost.sebi_fee, 10.0);
        assert_close(cost.stamp_duty, 10.0);
        assert_eq!(cost.stt, 0.0);
        assert_close(cost.gst, gst);
        assert_eq!(cost.shown_not_charged, 0.0);
    }
}

#[test]
fn a_currency_round_trip_on_one_day_is_one_contract_note() {
    // Accrued per fill under the default preset (futures 0.03 percent or Rs 20, so Rs 20
    // a leg). Buy: exchange 88,330 x 0.000004 = 0.35332, SEBI 0.08833, stamp 0.08833,
    // GST 18 percent of 20.44165 = 3.679497. Sell: exchange 0.35341, SEBI 0.0883525,
    // GST 18 percent of 20.4417625 = 3.67951725. The CDS note: brokerage 40.00,
    // exchange 0.70673 = 0.71, SEBI 0.1766825 = 0.18, stamp 0.08833 = Rs 0, GST 7.35901
    // = 7.36: 48.25.
    let model = CostModel::builtin().unwrap();
    let day = screenshot_day();
    let mut notes = ContractNoteAggregator::new();
    for (side, price) in [(Side::Buy, BUY), (Side::Sell, SELL)] {
        let cost = model
            .fill_cost(Segment::CdsFut, side, price, ONE_LOT_UNITS, day)
            .unwrap();
        notes.add(day, Segment::CdsFut, &cost);
    }
    let note = notes.note(day, NoteGroup::NseCds).unwrap();
    assert_eq!(note.group.code(), "CDS");
    assert_paise(note.charged.exchange_txn, 0.71);
    assert_paise(note.charged.gst, 7.36);
    assert_paise(note.charged.stamp_duty, 0.0);
    assert_paise(note.charged_total, 48.25);
}

#[test]
fn currency_charges_follow_the_dated_rows() {
    // Exchange charge with IPFT on Rs 1 crore: 0.0009 + 0.00005 percent = 95 before the
    // uniform charges of 2024-10-01, 40 from that day; options 0.035 + 0.002 percent =
    // 3,700, then 3,310. The SEBI fee was halved to Rs 5 per crore until 2021-03-31.
    let model = zerodha_accruals();
    let rows = [
        (days_from_civil(2024, 9, 30), Segment::CdsFut, 95.0, 10.0),
        (days_from_civil(2024, 10, 1), Segment::CdsFut, 40.0, 10.0),
        (days_from_civil(2024, 9, 30), Segment::CdsOpt, 3_700.0, 10.0),
        (days_from_civil(2024, 10, 1), Segment::CdsOpt, 3_310.0, 10.0),
        (days_from_civil(2021, 1, 15), Segment::CdsFut, 95.0, 5.0),
    ];
    for (day, segment, exchange, sebi) in rows {
        let cost = model
            .fill_cost_value(segment, Side::Sell, CRORE, day)
            .unwrap();
        assert_close(cost.exchange_txn, exchange);
        assert_close(cost.sebi_fee, sebi);
    }
}

#[test]
fn only_the_current_currency_rows_are_verified() {
    let model = zerodha_accruals();
    for segment in [Segment::CdsFut, Segment::CdsOpt] {
        let current = model.status(segment, screenshot_day()).unwrap();
        assert_eq!(current.trust, Trust::Verified, "{segment:?}");
        let older = model.status(segment, days_from_civil(2025, 6, 2)).unwrap();
        assert_eq!(older.trust, Trust::Unverified, "{segment:?}");
    }
}

#[test]
fn an_exercised_currency_option_pays_no_stt() {
    let stt = zerodha_accruals()
        .exercise_stt(Segment::CdsOpt, 1.5, ONE_LOT_UNITS, screenshot_day())
        .unwrap();
    assert_eq!(stt, 0.0);
}
