#![cfg(all(feature = "costs", feature = "time"))]
//! A strategy's own GTT orders (`validity="gtt"`, 14-strategy-api 2.4, 04-execution 4.4
//! item 2): resting entries that wait across sessions, their trigger and child, their
//! cancellation and modification, and the product rule.
//!
//! A daily strategy on daily bars of July 2025, CNC, no charges and no slippage. Every
//! order is decided at a day's close; a day the test does not write is the flat bar
//! (1500.00, 1510.00, 1490.00, 1500.00), so the decision price is 1500.00.

mod order_support;

use oa_bt::engine::{RunOutput, Step, Submission};
use oa_bt::exec::{
    FillAt, GttLegRole, GttStatus, OrderId, OrderRequest, OrderStatus, OrderType, RejectCode,
    TriggerDirection, Validity,
};
use oa_bt::model::{Day, Product, Side};
use oa_bt::rms::{Distance, LegRules};
use order_support::{day_open, events_at, gtt_order, july_2025, limit, run_daily, Act, Ohlc};

const FLAT: Ohlc = (1500.0, 1510.0, 1490.0, 1500.0);
const SESSIONS: [u32; 6] = [1, 2, 3, 4, 7, 8];

fn july(written: &[(u32, Ohlc)]) -> Vec<(Day, Ohlc)> {
    SESSIONS
        .iter()
        .map(|&day| {
            let bar = written
                .iter()
                .find(|(d, _)| *d == day)
                .map_or(FLAT, |(_, bar)| *bar);
            (july_2025(day), bar)
        })
        .collect()
}

fn no_change(_: &mut oa_bt::engine::RunConfig) {}

/// A GTT buy of 10 at `price` decided at the close of 2025-07-01.
fn gtt_buy(price: f64) -> Vec<(Day, Act)> {
    let request = gtt_order(Side::Buy, 10.0, limit(price));
    vec![(july_2025(1), Act::Submit(request))]
}

fn status(output: &RunOutput) -> OrderStatus {
    output.orders[0].status
}

#[test]
fn a_gtt_buy_below_the_market_waits_for_a_fall_with_its_child_one_percent_above() {
    let (_, gtts, _) = run_daily(&july(&[]), gtt_buy(1450.0), no_change);
    let leg = gtts[0].legs[0];
    assert_eq!(leg.role, GttLegRole::Order);
    assert_eq!(leg.direction, TriggerDirection::Falling);
    // 1450 x 1.01 = 1464.50.
    assert_eq!(leg.child_limit_rupees, 1464.5);
}

#[test]
fn a_gtt_entry_rests_across_sessions_and_fills_when_its_trigger_is_reached() {
    // 07-02 dips to 1460 only; 07-03 falls to 1440: the trigger 1450 is reached inside
    // the range and the child, marketable there, buys at 1450.
    let bars = july(&[
        (2, (1490.0, 1495.0, 1460.0, 1470.0)),
        (3, (1460.0, 1465.0, 1440.0, 1445.0)),
    ]);
    let (output, gtts, _) = run_daily(&bars, gtt_buy(1450.0), no_change);
    assert_eq!(
        events_at(&output, day_open(july_2025(1))),
        ["Submitted", "GttPlaced g1"]
    );
    let day_3 = day_open(july_2025(3));
    assert_eq!(
        events_at(&output, day_3),
        [
            "GttTriggered g1",
            "Filled",
            "GttChildFilled g1",
            "PositionOpened"
        ]
    );
    let fill = &output.fills[0];
    assert_eq!(
        (
            fill.stamp.slice_ts,
            fill.stamp.step,
            fill.fill.price_rupees,
            fill.fill.at
        ),
        (day_3, Step::Levels, 1450.0, FillAt::Level)
    );
    assert_eq!(fill.fill.order, Some(OrderId(1)));
    assert_eq!(status(&output), OrderStatus::Filled);
    assert_eq!(gtts[0].status, GttStatus::ChildFilled);
}

#[test]
fn a_gtt_breakout_buy_whose_bar_gaps_past_its_child_is_cancelled_unfilled() {
    // Trigger 1550 above the 1500 decision price, child limit 1565.50. 07-03 opens at
    // 1580 and its low 1570 never comes back to the limit.
    let bars = july(&[(3, (1580.0, 1590.0, 1570.0, 1585.0))]);
    let (output, gtts, _) = run_daily(&bars, gtt_buy(1550.0), no_change);
    assert_eq!(gtts[0].legs[0].direction, TriggerDirection::Rising);
    assert_eq!(
        events_at(&output, day_open(july_2025(3))),
        ["GttTriggered g1", "GttChildCanceled g1", "Canceled"]
    );
    assert!(output.fills.is_empty());
    assert_eq!(status(&output), OrderStatus::Canceled);
}

#[test]
fn cancelling_a_gtt_order_cancels_its_gtt() {
    let mut acts = gtt_buy(1450.0);
    acts.push((july_2025(2), Act::Cancel(OrderId(1))));
    let (output, gtts, _) = run_daily(&july(&[]), acts, no_change);
    assert_eq!(
        events_at(&output, day_open(july_2025(2))),
        ["GttCanceled g1", "Canceled"]
    );
    assert_eq!(gtts[0].status, GttStatus::Canceled);
}

#[test]
fn a_modified_gtt_order_takes_its_new_trigger_from_the_next_bar() {
    // Moved from 1450 to 1470 at the close of 07-02 (decision price 1480). 07-03 dips to
    // 1465: the new trigger fires and the order buys at 1470.
    let bars = july(&[
        (2, (1490.0, 1495.0, 1475.0, 1480.0)),
        (3, (1480.0, 1485.0, 1465.0, 1470.0)),
    ]);
    let mut acts = gtt_buy(1450.0);
    let moved = gtt_order(Side::Buy, 10.0, limit(1470.0));
    acts.push((july_2025(2), Act::Modify(OrderId(1), moved)));
    let (output, gtts, script) = run_daily(&bars, acts, no_change);
    assert!(matches!(
        script.modifications[..],
        [Ok(Submission::Submitted(OrderId(1)))]
    ));
    assert_eq!(
        events_at(&output, day_open(july_2025(2))),
        ["Modified", "GttModified g1"]
    );
    assert_eq!(gtts[0].trigger(GttLegRole::Order), Some(1470.0));
    assert_eq!(output.fills[0].fill.price_rupees, 1470.0);
}

#[test]
fn a_gtt_with_a_trigger_and_a_price_uses_the_price_as_its_child_limit() {
    let request = gtt_order(
        Side::Buy,
        10.0,
        OrderType::StopLimit {
            trigger_rupees: 1450.0,
            limit_rupees: 1452.0,
        },
    );
    let acts = vec![(july_2025(1), Act::Submit(request))];
    let (_, gtts, _) = run_daily(&july(&[]), acts, no_change);
    let leg = gtts[0].legs[0];
    assert_eq!(
        (leg.trigger_rupees, leg.child_limit_rupees),
        (1450.0, 1452.0)
    );
}

#[test]
fn a_gtt_entry_with_a_stop_places_its_protective_gtt_when_it_fills() {
    let request = OrderRequest {
        leg_rules: LegRules {
            stop: Some(Distance::Points(50.0)),
            ..LegRules::default()
        },
        ..gtt_order(Side::Buy, 10.0, limit(1450.0))
    };
    let bars = july(&[(3, (1460.0, 1465.0, 1440.0, 1445.0))]);
    let acts = vec![(july_2025(1), Act::Submit(request))];
    let (output, gtts, _) = run_daily(&bars, acts, no_change);
    assert_eq!(
        events_at(&output, day_open(july_2025(3)))
            .last()
            .map(String::as_str),
        Some("GttPlaced g2")
    );
    assert_eq!(gtts[1].trigger(GttLegRole::Stop), Some(1400.0));
    assert_eq!(gtts[1].status, GttStatus::Active);
}

#[test]
fn a_gtt_order_on_mis_is_rejected_product_rule() {
    let request = OrderRequest {
        product: Product::Mis,
        validity: Validity::Gtt,
        ..order_support::order(Side::Buy, 10.0, limit(1450.0), Product::Mis)
    };
    let acts = vec![(july_2025(1), Act::Submit(request))];
    let (output, gtts, script) = run_daily(&july(&[]), acts, no_change);
    assert!(matches!(
        script.submissions[..],
        [Submission::Rejected(_, RejectCode::ProductRule)]
    ));
    assert!(gtts.is_empty());
    assert_eq!(status(&output), OrderStatus::Rejected);
}
