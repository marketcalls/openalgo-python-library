//! One executed order charged at two segments' rates: a CNC order of cash equity that is
//! partly intraday and partly delivery pays the brokerage of one order.

use crate::model::Side;

use super::super::{Brokerage, Segment};
use super::{assert_close, model_with, snapshot_day};

const PRICE: f64 = 1_000.0;

/// Brokerage of a CNC buy at Rs 1,000.00 with `intraday` and `delivery` units.
fn split_brokerage(preset: Brokerage, intraday: f64, delivery: f64) -> f64 {
    model_with(preset)
        .split_fill_cost(
            Side::Buy,
            PRICE,
            (Segment::NseEqIntraday, intraday),
            (Segment::NseEqDelivery, delivery),
            snapshot_day(),
        )
        .unwrap()
        .brokerage
}

#[test]
fn a_split_order_pays_at_most_the_larger_per_order_amount_of_its_two_rules() {
    // Worked by hand, 10 units on each side (Rs 10,000 each):
    // - standard: intraday 0.1 percent is 10, delivery 2.5 percent is 250 capped at 20;
    //   together 30, one order pays at most Rs 20.
    // - zerodha: intraday 0.03 percent is 3, delivery free: 3.
    // - flat Rs 20 everywhere: one order, Rs 20 (not 40).
    // - 0.03 percent with no cap: no per-order limit, 3 + 3 = 6.
    let uncapped = Brokerage::percent(0.03, None).unwrap();
    let rows = [
        (Brokerage::standard(), 20.0),
        (Brokerage::zerodha(), 3.0),
        (Brokerage::flat(20.0).unwrap(), 20.0),
        (uncapped, 6.0),
    ];
    for (preset, brokerage) in rows {
        assert_close(split_brokerage(preset, 10.0, 10.0), brokerage);
    }
}

#[test]
fn a_split_order_under_the_cap_pays_both_parts_in_full() {
    // Standard, 2 units intraday and 0.5 delivery (Rs 2,000 and Rs 500): 0.1 percent is
    // 2.00 and 2.5 percent is 12.50; together 14.50, under Rs 20.
    assert_close(split_brokerage(Brokerage::standard(), 2.0, 0.5), 14.5);
}

#[test]
fn an_order_wholly_on_one_side_costs_what_a_fill_costs() {
    // Standard: 2.5 percent of Rs 500 is 12.50 on delivery; 0.1 percent of Rs 5,000 is
    // 5.00 intraday.
    assert_close(split_brokerage(Brokerage::standard(), 0.0, 0.5), 12.5);
    assert_close(split_brokerage(Brokerage::standard(), 5.0, 0.0), 5.0);
}

#[test]
fn gst_falls_on_the_brokerage_the_split_order_pays() {
    // Standard, 200 units on each side at Rs 1,000.00: brokerage 20; exchange 0.00297
    // percent plus IPFT 0.0001 percent of 4,00,000 is 12.28; SEBI 0.40; GST 18 percent of
    // 20 + 12.28 + 0.40 = 5.8824.
    let cost = model_with(Brokerage::standard())
        .split_fill_cost(
            Side::Buy,
            PRICE,
            (Segment::NseEqIntraday, 200.0),
            (Segment::NseEqDelivery, 200.0),
            snapshot_day(),
        )
        .unwrap();
    assert_close(cost.brokerage, 20.0);
    assert_close(cost.gst, 5.8824);
}
