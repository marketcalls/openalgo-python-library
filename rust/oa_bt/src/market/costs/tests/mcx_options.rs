//! MCX options on non-agri commodities (MCX_OPT): the reference note's section 3, the
//! maintainer's calculator screenshot of 2026-11-23 (COPPER, lot 2,500 units, strike
//! 900) and the dated rows of `schedules.csv`. Every figure is worked by hand in the
//! test's comment.

use crate::model::{days_from_civil, Day, Side};

use super::super::{Brokerage, CostError, CostModel, Rounding, Segment, Trust};
use super::{assert_close, assert_paise, builtin_book, calculator_model, SameDay, CRORE};

fn screenshot_day() -> Day {
    days_from_civil(2026, 11, 23)
}

/// One COPPER lot: 2,500 kg quoted per kg.
const ONE_LOT_UNITS: f64 = 2_500.0;

fn zerodha_accruals() -> CostModel {
    CostModel::new(builtin_book(), Brokerage::zerodha(), Rounding::ContractNote)
}

#[test]
fn a_copper_option_round_trip_reproduces_the_calculator_screenshot() {
    // Premiums 100 and 110 on 2,500 units: turnover 5,25,000. Brokerage Rs 20 a leg, 40.
    // Exchange 0.0418 percent: 219.45. CTT 0.05 percent of the sell premium 2,75,000:
    // 137.50 (to the paisa, as for MCX futures). SEBI Rs 10 per crore: 0.525, 0.53. GST
    // 18 percent of 40 + 219.45 + 0.53 = 46.7964, 46.80. Stamp 0.003 percent of the buy
    // premium 2,50,000: 7.50, rounded to the rupee: 8. Total, the sum of those lines:
    // 452.28.
    let model = calculator_model();
    let cost = model
        .round_trip_on(
            Segment::McxOpt,
            100.0,
            110.0,
            ONE_LOT_UNITS,
            screenshot_day(),
        )
        .unwrap();
    assert_paise(cost.brokerage, 40.0);
    assert_paise(cost.exchange_txn, 219.45);
    assert_paise(cost.stt, 137.5);
    assert_paise(cost.sebi_fee, 0.53);
    assert_paise(cost.gst, 46.8);
    assert_paise(cost.stamp_duty, 8.0);
    assert_paise(model.reported_total(&cost), 452.28);
}

#[test]
fn accruals_per_crore_of_premium() {
    // Sell of Rs 1 crore of premium: CTT 0.05 percent = 5,000, exchange 0.0418 percent
    // = 4,180, SEBI 10, no stamp, GST 18 percent of 20 + 4,180 + 10 = 757.80. Buy: no
    // CTT, stamp 0.003 percent = 300.
    let model = zerodha_accruals();
    let sell = model
        .fill_cost_value(Segment::McxOpt, Side::Sell, CRORE, screenshot_day())
        .unwrap();
    assert_close(sell.stt, 5_000.0);
    assert_close(sell.exchange_txn, 4_180.0);
    assert_close(sell.sebi_fee, 10.0);
    assert_eq!(sell.stamp_duty, 0.0);
    assert_close(sell.gst, 757.8);
    let buy = model
        .fill_cost_value(Segment::McxOpt, Side::Buy, CRORE, screenshot_day())
        .unwrap();
    assert_eq!(buy.stt, 0.0);
    assert_close(buy.stamp_duty, 300.0);
}

#[test]
fn the_exchange_charge_follows_the_dated_rows() {
    // Rs 50 per lakh of premium before 2024-10-01, Rs 41.80 from that day.
    let model = zerodha_accruals();
    let rows = [
        (days_from_civil(2024, 9, 30), 5_000.0),
        (days_from_civil(2024, 10, 1), 4_180.0),
    ];
    for (day, exchange) in rows {
        let cost = model.fill_cost_value(Segment::McxOpt, Side::Buy, CRORE, day);
        assert_close(cost.unwrap().exchange_txn, exchange);
    }
}

#[test]
fn only_the_current_mcx_option_row_is_verified() {
    let model = zerodha_accruals();
    let current = model.status(Segment::McxOpt, screenshot_day()).unwrap();
    assert_eq!(current.trust, Trust::Verified);
    let older = model.status(Segment::McxOpt, days_from_civil(2025, 6, 2));
    assert_eq!(older.unwrap().trust, Trust::Unverified);
}

#[test]
fn the_exercise_of_an_mcx_option_is_refused() {
    let result = zerodha_accruals().exercise_stt(Segment::McxOpt, 5.0, 2_500.0, screenshot_day());
    let error = result.unwrap_err();
    assert!(
        matches!(error, CostError::UnsupportedSegment(_)),
        "{error:?}"
    );
    assert!(error.to_string().contains("MCX option"), "{error}");
}
