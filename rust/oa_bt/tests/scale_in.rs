#![cfg(all(feature = "costs", feature = "time"))]
//! The scale-in limits through the slice sequence (12-position-sizing 2, decisions
//! SPEC-14 and SPEC-15, reject code `scale_in_limit` of 05-risk-management 7): each
//! limit refuses the add that breaks it at submission, an entry from flat is not an add,
//! adds decided together or still waiting count, and averaging down is judged at the
//! decision price.
//!
//! A one-minute MIS strategy on one-minute bars of SBIN on 2025-07-01, flat at 100.00
//! unless a bar says otherwise. The order decided at the close of bar k fills at the
//! open of bar k + 1.

mod order_support;

use oa_bt::engine::{Engine, KernelError, RunOutput, RunSetup, Submission};
use oa_bt::exec::{OrderId, OrderRequest, OrderType, RejectCode};
use oa_bt::market::calendar::Calendar;
use oa_bt::model::{Product, Side};
use oa_bt::rms::{Distance, LegRules, PositionCap, RiskCap, ScaleLimits, Spacing};
use order_support::*;

/// The decision instant at the close of bar `k`.
fn decision(k: u32) -> i64 {
    minute(k + 1)
}

fn buy(units: f64) -> Act {
    Act::Submit(order(Side::Buy, units, OrderType::Market, Product::Mis))
}

fn sell(units: f64) -> Act {
    Act::Submit(order(Side::Sell, units, OrderType::Market, Product::Mis))
}

fn buy_with_stop(units: f64, points: f64) -> Act {
    let rules = LegRules {
        stop: Some(Distance::Points(points)),
        ..LegRules::default()
    };
    let request = order(Side::Buy, units, OrderType::Market, Product::Mis);
    Act::Submit(OrderRequest {
        leg_rules: rules,
        ..request
    })
}

/// Run 20 one-minute bars, flat at 100.00 but for `bars`, with the scale-in `limits`.
fn run(limits: ScaleLimits, bars: &[(u32, Ohlc)], acts: Vec<(i64, Act)>) -> (RunOutput, Script) {
    let mut config = minute_config();
    config.risk.scale = limits;
    run_script(engine(config, equity(), minute_bars(20, 100.0, bars)), acts)
}

fn outcomes(script: &Script) -> Vec<String> {
    script
        .submissions
        .iter()
        .map(|submission| match submission {
            Submission::Submitted(_) => "submitted".to_string(),
            Submission::Rejected(_, code) => format!("rejected {}", code.code()),
            Submission::Skipped(reason) => format!("skipped {}", reason.code()),
        })
        .collect()
}

/// Units the entry and adds filled (an MIS position is closed by the session end).
fn bought(output: &RunOutput) -> f64 {
    fills(output)
        .iter()
        .filter(|fill| matches!(fill.6, "entry" | "add"))
        .map(|fill| fill.3)
        .sum()
}

const SUBMITTED: &str = "submitted";
const REFUSED: &str = "rejected scale_in_limit";

#[test]
fn max_adds_refuses_the_add_beyond_the_limit() {
    let limits = ScaleLimits {
        max_adds: Some(1),
        ..ScaleLimits::default()
    };
    let acts = vec![
        (decision(0), buy(10.0)),
        (decision(2), buy(10.0)),
        (decision(4), buy(10.0)),
    ];
    let (output, script) = run(limits, &[], acts);
    assert_eq!(outcomes(&script), vec![SUBMITTED, SUBMITTED, REFUSED]);
    assert_eq!(bought(&output), 20.0);
}

#[test]
fn an_add_still_waiting_for_its_fill_counts_towards_max_adds() {
    let limits = ScaleLimits {
        max_adds: Some(1),
        ..ScaleLimits::default()
    };
    let acts = vec![
        (decision(0), buy(10.0)),
        (decision(2), buy(10.0)),
        (decision(2), buy(10.0)),
    ];
    let (_, script) = run(limits, &[], acts);
    assert_eq!(outcomes(&script), vec![SUBMITTED, SUBMITTED, REFUSED]);
}

#[test]
fn an_order_decided_with_the_entry_is_an_add_to_it() {
    let limits = ScaleLimits {
        max_adds: Some(0),
        ..ScaleLimits::default()
    };
    let acts = vec![(decision(0), buy(10.0)), (decision(0), buy(10.0))];
    let (output, script) = run(limits, &[], acts);
    assert_eq!(outcomes(&script), vec![SUBMITTED, REFUSED]);
    assert_eq!(bought(&output), 10.0);
}

#[test]
fn max_position_caps_the_position_after_the_add() {
    // 10 + 10 = 20 units is within 25; 20 + 10 = 30 is not.
    let limits = ScaleLimits {
        max_position: Some(PositionCap::Units(25.0)),
        ..ScaleLimits::default()
    };
    let acts = vec![
        (decision(0), buy(10.0)),
        (decision(2), buy(10.0)),
        (decision(4), buy(10.0)),
    ];
    let (_, script) = run(limits, &[], acts);
    assert_eq!(outcomes(&script), vec![SUBMITTED, SUBMITTED, REFUSED]);
}

#[test]
fn max_position_by_value_is_taken_at_the_decision_price() {
    // 20 units at the decision price 100.00 are 2000.00 of value.
    let cap = |rupees| ScaleLimits {
        max_position: Some(PositionCap::Rupees(rupees)),
        ..ScaleLimits::default()
    };
    let acts = || vec![(decision(0), buy(10.0)), (decision(2), buy(10.0))];
    assert_eq!(
        outcomes(&run(cap(2_000.0), &[], acts()).1),
        vec![SUBMITTED, SUBMITTED]
    );
    assert_eq!(
        outcomes(&run(cap(1_999.0), &[], acts()).1),
        vec![SUBMITTED, REFUSED]
    );
}

#[test]
fn an_entry_from_flat_is_not_held_to_the_scale_in_limits() {
    let limits = ScaleLimits {
        max_position: Some(PositionCap::Units(5.0)),
        max_adds: Some(0),
        ..ScaleLimits::default()
    };
    let (output, script) = run(limits, &[], vec![(decision(0), buy(10.0))]);
    assert_eq!(outcomes(&script), vec![SUBMITTED]);
    assert_eq!(bought(&output), 10.0);
}

#[test]
fn min_spacing_in_bars_counts_the_bars_closed_after_the_last_tranche() {
    // The entry fills in bar 1. With a spacing of 2 bars, adds decided at the closes of
    // bars 1 and 2 are refused and the one at bar 3 is accepted; it fills in bar 4, so
    // the add decided at the close of bar 4 is refused again.
    let limits = ScaleLimits {
        min_spacing: Some(Spacing::Bars(2)),
        ..ScaleLimits::default()
    };
    let acts = (0..5).map(|k| (decision(k), buy(10.0))).collect();
    let (_, script) = run(limits, &[], acts);
    assert_eq!(
        outcomes(&script),
        vec![SUBMITTED, REFUSED, REFUSED, SUBMITTED, REFUSED]
    );
}

#[test]
fn min_spacing_as_a_price_distance_is_measured_from_the_last_tranche() {
    // The entry fills at 100.00 (bar 1 open). Bar 1 closes 100.50: 0.50 from it, short
    // of 1 point. Bar 2 closes 101.00: 1 point, accepted; it fills at 101.00 (bar 3
    // open), and bar 3 closes 101.00, no distance from that tranche.
    let limits = ScaleLimits {
        min_spacing: Some(Spacing::Price(Distance::Points(1.0))),
        ..ScaleLimits::default()
    };
    let bars = [
        (1, (100.0, 100.5, 100.0, 100.5)),
        (2, (100.5, 101.0, 100.5, 101.0)),
        (3, (101.0, 101.0, 101.0, 101.0)),
    ];
    let acts = (0..4).map(|k| (decision(k), buy(10.0))).collect();
    let (_, script) = run(limits, &bars, acts);
    assert_eq!(
        outcomes(&script),
        vec![SUBMITTED, REFUSED, SUBMITTED, REFUSED]
    );
}

/// The entry fills at 100.00 in bar 1, which closes at 99.00 (below the average).
const LOWER_CLOSE: [(u32, Ohlc); 1] = [(1, (100.0, 100.0, 99.0, 99.0))];

#[test]
fn an_add_below_the_average_of_a_long_is_refused_by_default() {
    let acts = vec![(decision(0), buy(10.0)), (decision(1), buy(10.0))];
    let (output, script) = run(ScaleLimits::default(), &LOWER_CLOSE, acts);
    assert_eq!(outcomes(&script), vec![SUBMITTED, REFUSED]);
    assert_eq!(output.orders[1].reject_code, Some(RejectCode::ScaleInLimit));
}

#[test]
fn allow_averaging_down_lets_the_add_below_the_average_through() {
    let limits = ScaleLimits {
        allow_averaging_down: true,
        ..ScaleLimits::default()
    };
    let acts = vec![(decision(0), buy(10.0)), (decision(1), buy(10.0))];
    let (output, script) = run(limits, &LOWER_CLOSE, acts);
    assert_eq!(outcomes(&script), vec![SUBMITTED, SUBMITTED]);
    assert_eq!(bought(&output), 20.0);
}

#[test]
fn an_add_above_the_average_of_a_short_is_refused_by_default() {
    // Sold at 100.00 in bar 1, which closes at 101.00.
    let bars = [(1, (100.0, 101.0, 100.0, 101.0))];
    let acts = vec![(decision(0), sell(10.0)), (decision(1), sell(10.0))];
    let (_, script) = run(ScaleLimits::default(), &bars, acts);
    assert_eq!(outcomes(&script), vec![SUBMITTED, REFUSED]);
}

#[test]
fn averaging_down_is_judged_at_the_decision_price_not_the_fill() {
    // Bar 1 closes at 100.50, above the average 100.00, so the add is accepted; bar 2
    // gaps down and the add fills at 99.00, below the average (SPEC-15).
    let bars = [
        (1, (100.0, 100.5, 100.0, 100.5)),
        (2, (99.0, 99.0, 99.0, 99.0)),
    ];
    let acts = vec![(decision(0), buy(10.0)), (decision(1), buy(10.0))];
    let (output, script) = run(ScaleLimits::default(), &bars, acts);
    assert_eq!(outcomes(&script), vec![SUBMITTED, SUBMITTED]);
    let add = fills(&output)[1];
    assert_eq!((add.4, add.6), (99.0, "add"));
}

/// Bought 10 at 100.00 with a 5-point stop (95.00); bar 1 closes at 104.00. An add of
/// 10 at 104.00 gives an average of 102.00 and a stop recomputed to 97.00 (tighter than
/// 95.00), so the open risk after it is (102.00 - 97.00) x 20 = 100.00.
fn risk_run(cap: f64) -> Script {
    let limits = ScaleLimits {
        max_risk: Some(RiskCap::Rupees(cap)),
        ..ScaleLimits::default()
    };
    let bars = [(1, (100.0, 104.0, 100.0, 104.0))];
    let acts = vec![
        (decision(0), buy_with_stop(10.0, 5.0)),
        (decision(1), buy(10.0)),
    ];
    run(limits, &bars, acts).1
}

#[test]
fn max_risk_caps_the_open_risk_of_the_position_after_the_add() {
    assert_eq!(outcomes(&risk_run(100.0)), vec![SUBMITTED, SUBMITTED]);
    assert_eq!(outcomes(&risk_run(99.0)), vec![SUBMITTED, REFUSED]);
}

#[test]
fn max_risk_refuses_an_add_to_a_position_without_a_stop() {
    let limits = ScaleLimits {
        max_risk: Some(RiskCap::Rupees(1.0e9)),
        ..ScaleLimits::default()
    };
    let acts = vec![(decision(0), buy(10.0)), (decision(2), buy(10.0))];
    let (_, script) = run(limits, &[], acts);
    assert_eq!(outcomes(&script), vec![SUBMITTED, REFUSED]);
}

#[test]
fn a_modified_add_does_not_count_itself_as_waiting() {
    // max_adds 1: the resting limit add at 99.00 is the one add; changing its price to
    // 98.00 is not a second add.
    let limits = ScaleLimits {
        max_adds: Some(1),
        ..ScaleLimits::default()
    };
    let limit_buy = |price| order(Side::Buy, 10.0, limit(price), Product::Mis);
    let acts = vec![
        (decision(0), buy(10.0)),
        (decision(1), Act::Submit(limit_buy(99.0))),
        (decision(2), Act::Modify(OrderId(2), limit_buy(98.0))),
    ];
    let (_, script) = run(limits, &[], acts);
    assert_eq!(outcomes(&script), vec![SUBMITTED, SUBMITTED]);
    assert_eq!(
        script.modifications,
        vec![Ok(Submission::Submitted(OrderId(2)))]
    );
}

#[test]
fn an_unusable_scale_in_limit_is_refused_before_the_run() {
    let mut config = minute_config();
    config.risk.scale.min_spacing = Some(Spacing::Bars(0));
    let calendar = Calendar::new().expect("the embedded calendar");
    let setup = RunSetup {
        config,
        calendar,
        instruments: vec![equity()],
        bars: vec![minute_bars(5, 100.0, &[])],
    };
    assert!(matches!(Engine::new(setup), Err(KernelError::Rms(_))));
}
