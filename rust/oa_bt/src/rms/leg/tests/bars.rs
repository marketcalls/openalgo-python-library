//! Levels against one base bar (step 3): the open first, the stop wins inside the
//! range, every target reached fills nearest first. Prices are those of trace 2.

use crate::exec::fill::FillAt;
use crate::model::Side;
use crate::rms::leg::{LevelHit, LevelKind};

use super::support::{bar, open, stop_and_target, stop_and_targets};

fn hits(side: Side, average: f64, b: crate::rms::leg::BarPrices) -> Vec<LevelHit> {
    // Stop 2 points, target 3 points, 50 units (trace 2).
    open(stop_and_target(2.0, 3.0), side, average, 50.0)
        .map(|leg| leg.test_bar(&b))
        .unwrap_or_default()
}

fn summary(hits: &[LevelHit]) -> Vec<(LevelKind, f64, f64, FillAt, f64)> {
    hits.iter()
        .map(|h| {
            (
                h.kind,
                h.level_rupees,
                h.price_rupees,
                h.at,
                h.quantity_units,
            )
        })
        .collect()
}

#[test]
fn a_long_stop_crossed_at_the_open_fills_at_the_open_and_reports_its_level() {
    // Entry 420.00, stop 418.00; the bar opens at 417.20 (trace-02a 09:22).
    let got = hits(Side::Buy, 420.0, bar(417.2, 417.5, 416.8, 417.0));
    assert_eq!(
        summary(&got),
        vec![(LevelKind::Stop, 418.0, 417.2, FillAt::Open, 50.0)]
    );
}

#[test]
fn a_bar_one_tick_short_of_a_level_does_nothing() {
    // Low 418.05 against the stop 418.00 (trace-02a 09:21).
    assert!(hits(Side::Buy, 420.0, bar(419.6, 419.8, 418.05, 418.1)).is_empty());
}

#[test]
fn a_target_crossed_at_the_open_fills_there_and_the_stop_is_not_tested() {
    // Entry 417.00: target 420.00, stop 415.00; open 420.60, low 414.50 (trace-02a 09:27).
    let got = hits(Side::Buy, 417.0, bar(420.6, 420.7, 414.5, 415.2));
    assert_eq!(
        summary(&got),
        vec![(LevelKind::Target(0), 420.0, 420.6, FillAt::Open, 50.0)]
    );
}

#[test]
fn the_stop_wins_when_the_range_reaches_both_levels() {
    // Entry 415.50: stop 413.50, target 418.50; range 413.40 to 418.60 (trace-02a 09:31).
    let got = hits(Side::Buy, 415.5, bar(415.4, 418.6, 413.4, 416.0));
    assert_eq!(
        summary(&got),
        vec![(LevelKind::Stop, 413.5, 413.5, FillAt::Level, 50.0)]
    );
}

#[test]
fn a_short_mirrors_the_long() {
    // Entry 420.00 short: stop 422.00, opened through at 422.70 (trace-02b 09:22).
    let gap = hits(Side::Sell, 420.0, bar(422.7, 423.1, 422.4, 422.9));
    assert_eq!(
        summary(&gap),
        vec![(LevelKind::Stop, 422.0, 422.7, FillAt::Open, 50.0)]
    );
    // Entry 423.00 short: target 420.00 crossed at the open 419.40 (trace-02b 09:27).
    let target = hits(Side::Sell, 423.0, bar(419.4, 425.5, 419.3, 425.0));
    assert_eq!(
        summary(&target),
        vec![(LevelKind::Target(0), 420.0, 419.4, FillAt::Open, 50.0)]
    );
    // Entry 425.00 short: both levels in the range, the stop 427.00 wins (09:31).
    let both = hits(Side::Sell, 425.0, bar(425.1, 427.1, 421.9, 426.0));
    assert_eq!(
        summary(&both),
        vec![(LevelKind::Stop, 427.0, 427.0, FillAt::Level, 50.0)]
    );
}

#[test]
fn every_target_the_bar_reaches_fills_nearest_first() {
    // Entry 1000.00, 100 units: targets +10 for 50 and +20 for 30 (SPEC-28). A bar from
    // 1005.00 to 1025.00 reaches both; the stop 990.00 is not reached.
    let leg = open(
        stop_and_targets(10.0, &[(20.0, 0.3), (10.0, 0.5)]),
        Side::Buy,
        1000.0,
        100.0,
    );
    let got = leg.map(|leg| leg.test_bar(&bar(1005.0, 1025.0, 1004.0, 1024.0)));
    assert_eq!(
        got.map(|hits| summary(&hits)),
        Ok(vec![
            (LevelKind::Target(1), 1010.0, 1010.0, FillAt::Level, 50.0),
            (LevelKind::Target(0), 1020.0, 1020.0, FillAt::Level, 30.0),
        ])
    );
}

#[test]
fn a_target_crossed_at_the_open_does_not_hold_back_a_further_target() {
    // Targets 1010.00 (50) and 1020.00 (30); the bar opens at 1012.00 and reaches 1021.00.
    let leg = open(
        stop_and_targets(10.0, &[(10.0, 0.5), (20.0, 0.3)]),
        Side::Buy,
        1000.0,
        100.0,
    );
    let got = leg.map(|leg| leg.test_bar(&bar(1012.0, 1021.0, 985.0, 1015.0)));
    assert_eq!(
        got.map(|hits| summary(&hits)),
        Ok(vec![
            (LevelKind::Target(0), 1010.0, 1012.0, FillAt::Open, 50.0),
            (LevelKind::Target(1), 1020.0, 1020.0, FillAt::Level, 30.0),
        ])
    );
}

#[test]
fn a_stop_in_the_range_beats_several_targets() {
    let leg = open(
        stop_and_targets(10.0, &[(10.0, 0.5), (20.0, 0.3)]),
        Side::Buy,
        1000.0,
        100.0,
    );
    let got = leg.map(|leg| leg.test_bar(&bar(1001.0, 1030.0, 989.0, 1000.0)));
    assert_eq!(
        got.map(|hits| summary(&hits)),
        Ok(vec![(LevelKind::Stop, 990.0, 990.0, FillAt::Level, 100.0)])
    );
}

#[test]
fn a_leg_without_levels_never_hits() {
    let leg = open(
        crate::rms::rules::LegRules::default(),
        Side::Buy,
        500.0,
        100.0,
    );
    let got = leg.map(|leg| leg.test_bar(&bar(10.0, 5000.0, 1.0, 20.0)));
    assert_eq!(got, Ok(Vec::new()));
}
