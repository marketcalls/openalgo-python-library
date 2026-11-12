//! Trail and breakeven (step 4). Prices are those of trace 1 unless stated.

use crate::model::Side;
use crate::rms::leg::{LegState, RatchetOutcome};
use crate::rms::rules::{AfterFirstTarget, Distance, LegRules, TrailSpec};
use crate::rms::vocabulary::RiskEventName;

use super::support::{bar, breakeven, open, points, stop_and_targets, stop_level, trail};

fn with_trail(stop: f64, x: f64, y: f64) -> LegRules {
    LegRules {
        stop: Some(points(stop)),
        trail: Some(trail(x, y)),
        ..LegRules::default()
    }
}

/// Ratchet a leg through bars given as (high, low), returning the stop after each.
fn stops_after(leg: &mut LegState, bars: &[(f64, f64)]) -> Vec<Option<f64>> {
    bars.iter()
        .map(|&(high, low)| {
            let _ = leg.ratchet(&bar(low, high, low, high));
            stop_level(leg)
        })
        .collect()
}

fn event_names(outcome: &RatchetOutcome) -> Vec<RiskEventName> {
    outcome.events.iter().map(|e| e.name).collect()
}

#[test]
fn a_continuous_trail_arms_at_x_and_keeps_x_behind_the_best_price() {
    // Entry 800.50, stop 796.50, bt.trail(3, 0): highs 801.50 (excursion 1), 803.50
    // (3, arms at 800.50), 805.00 (802.00) (trace-01a 09:45 to 09:47).
    let Ok(mut leg) = open(with_trail(4.0, 3.0, 0.0), Side::Buy, 800.5, 100.0) else {
        panic!("the leg opens");
    };
    let stops = stops_after(&mut leg, &[(801.5, 800.3), (803.5, 801.2), (805.0, 803.0)]);
    assert_eq!(stops, vec![Some(796.5), Some(800.5), Some(802.0)]);
}

#[test]
fn the_trail_arming_event_is_written_once_with_the_stop_it_set() {
    let Ok(mut leg) = open(with_trail(4.0, 3.0, 0.0), Side::Buy, 800.5, 100.0) else {
        panic!("the leg opens");
    };
    let first = leg
        .ratchet(&bar(801.4, 803.5, 801.2, 803.4))
        .unwrap_or_default();
    let second = leg
        .ratchet(&bar(803.4, 805.0, 803.0, 804.9))
        .unwrap_or_default();
    assert_eq!(event_names(&first), vec![RiskEventName::LegTrailArmed]);
    assert_eq!(first.events[0].level_rupees, Some(800.5));
    assert!(second.events.is_empty() && second.levels_moved);
}

#[test]
fn a_new_extreme_alone_does_not_move_the_levels() {
    // Excursion 1.00 of x = 3: the extreme moves to 801.50, nothing else does.
    let Ok(mut leg) = open(with_trail(4.0, 3.0, 0.0), Side::Buy, 800.5, 100.0) else {
        panic!("the leg opens");
    };
    let outcome = leg.ratchet(&bar(800.4, 801.5, 800.3, 801.4));
    assert_eq!(outcome.map(|o| o.levels_moved), Ok(false));
    assert_eq!(leg.trail.map(|t| t.favourable_extreme_rupees), Some(801.5));
}

#[test]
fn a_stepped_trail_moves_y_for_every_whole_x_from_the_stop() {
    // Entry 801.00, stop 797.00, bt.trail(2, 1): excursions 1.90 (no step), 2.00 (798.00),
    // 4.40 (799.00) (trace-01a 10:01 to 10:03).
    let Ok(mut leg) = open(with_trail(4.0, 2.0, 1.0), Side::Buy, 801.0, 100.0) else {
        panic!("the leg opens");
    };
    let stops = stops_after(&mut leg, &[(802.9, 800.7), (803.0, 802.5), (805.4, 802.8)]);
    assert_eq!(stops, vec![Some(797.0), Some(798.0), Some(799.0)]);
}

#[test]
fn a_stepped_trail_never_passes_the_best_price_seen() {
    // Entry 100.00, stop 99.00, bt.trail(1, 10): an excursion of 5 asks for 149.00 and
    // is held at the extreme 105.00 (the server's vector of the same name).
    let rules = with_trail(1.0, 1.0, 10.0);
    let Ok(mut leg) = LegState::open(rules, super::support::context(Side::Buy), 100.0, 50.0) else {
        panic!("the leg opens");
    };
    let stops = stops_after(&mut leg, &[(105.0, 101.0)]);
    assert_eq!(stops, vec![Some(105.0)]);
}

#[test]
fn a_short_trail_follows_the_lowest_price() {
    // Short at 100.00, stop 110.00, bt.trail(3, 0): a low of 90.00 puts it at 93.00.
    let Ok(mut leg) = open(with_trail(10.0, 3.0, 0.0), Side::Sell, 100.0, 75.0) else {
        panic!("the leg opens");
    };
    let stops = stops_after(&mut leg, &[(101.0, 90.0), (95.0, 92.0)]);
    assert_eq!(stops, vec![Some(93.0), Some(93.0)]);
}

#[test]
fn the_entry_bar_extreme_counts_from_the_fill_average() {
    // SPEC-01, trace-01c: entry 800.50, entry bar high 803.60: the trail arms at 800.60.
    let Ok(mut leg) = open(with_trail(4.0, 3.0, 0.0), Side::Buy, 800.5, 100.0) else {
        panic!("the leg opens");
    };
    let outcome = leg.ratchet(&bar(800.5, 803.6, 800.4, 803.0));
    assert_eq!(outcome.map(|o| o.events.len()), Ok(1));
    assert_eq!(stop_level(&leg), Some(800.6));
}

#[test]
fn a_trail_without_a_stop_creates_the_first_one() {
    let rules = LegRules {
        trail: Some(trail(3.0, 0.0)),
        ..LegRules::default()
    };
    let Ok(mut leg) = open(rules, Side::Buy, 100.0, 50.0) else {
        panic!("the leg opens");
    };
    assert_eq!(stop_level(&leg), None);
    assert_eq!(stops_after(&mut leg, &[(110.0, 100.0)]), vec![Some(107.0)]);
}

#[test]
fn breakeven_arms_at_its_trigger_and_moves_the_stop_to_entry_plus_offset() {
    // Trace-01d: entry 800.50, stop 796.50, bt.breakeven(2, 0.5): 802.40 does not arm,
    // 802.50 arms and moves the stop to 801.00.
    let rules = LegRules {
        stop: Some(points(4.0)),
        breakeven: Some(breakeven(2.0, 0.5)),
        ..LegRules::default()
    };
    let Ok(mut leg) = open(rules, Side::Buy, 800.5, 100.0) else {
        panic!("the leg opens");
    };
    assert_eq!(leg.breakeven.map(|b| b.trigger_price_rupees), Some(802.5));
    let quiet = leg
        .ratchet(&bar(800.6, 802.4, 800.1, 802.0))
        .unwrap_or_default();
    let armed = leg
        .ratchet(&bar(802.0, 802.5, 801.8, 802.2))
        .unwrap_or_default();
    assert!(quiet.events.is_empty());
    assert_eq!(event_names(&armed), vec![RiskEventName::LegBreakevenArmed]);
    assert_eq!(armed.events[0].level_rupees, Some(801.0));
    assert_eq!(stop_level(&leg), Some(801.0));
}

#[test]
fn breakeven_never_loosens_a_tighter_stop() {
    // A trail and a breakeven on one leg: the more protective level is in force (SPEC-24).
    let rules = LegRules {
        stop: Some(points(4.0)),
        trail: Some(trail(1.0, 0.0)),
        breakeven: Some(breakeven(2.0, 0.0)),
        ..LegRules::default()
    };
    let Ok(mut leg) = open(rules, Side::Buy, 800.5, 100.0) else {
        panic!("the leg opens");
    };
    // High 805.00: the trail asks for 804.00 and breakeven for 800.50; 804.00 is in force.
    let _ = leg.ratchet(&bar(800.5, 805.0, 800.4, 804.0));
    assert_eq!(stop_level(&leg), Some(804.0));
    assert_eq!(leg.breakeven.map(|b| b.armed), Some(true));
}

#[test]
fn breakeven_creates_a_stop_when_none_is_set() {
    let rules = LegRules {
        breakeven: Some(breakeven(2.0, 0.0)),
        ..LegRules::default()
    };
    let Ok(mut leg) = open(rules, Side::Sell, 420.0, 50.0) else {
        panic!("the leg opens");
    };
    let _ = leg.ratchet(&bar(419.0, 419.5, 418.0, 418.5));
    assert_eq!(stop_level(&leg), Some(420.0));
}

#[test]
fn after_the_first_target_the_stop_moves_to_the_average_price() {
    let rules = LegRules {
        after_first_target: AfterFirstTarget::Breakeven,
        ..stop_and_targets(10.0, &[(10.0, 0.5), (20.0, 0.3)])
    };
    let Ok(mut leg) = open(rules, Side::Buy, 1000.0, 100.0) else {
        panic!("the leg opens");
    };
    let _ = leg.on_reduce(50.0, Some(0));
    let first = leg.after_target_fill();
    assert_eq!(
        first.map(|e| e.map(|e| (e.name, e.level_rupees))),
        Ok(Some((RiskEventName::LegBreakevenArmed, Some(1000.0))))
    );
    assert_eq!(stop_level(&leg), Some(1000.0));
    // The second target does not move it again.
    let _ = leg.on_reduce(20.0, Some(1));
    assert_eq!(leg.after_target_fill(), Ok(None));
}

#[test]
fn percent_and_tick_trails_are_fixed_in_points_at_the_entry() {
    // bt.trail_pct(0.5) on an entry at 1000.00 is a 5-point trail (SPEC-25).
    let rules = LegRules {
        trail: Some(TrailSpec {
            trigger: Distance::Percent(0.5),
            step: None,
        }),
        ..LegRules::default()
    };
    let Ok(mut leg) = open(rules, Side::Buy, 1000.0, 10.0) else {
        panic!("the leg opens");
    };
    assert_eq!(stops_after(&mut leg, &[(1004.95, 999.0)]), vec![None]);
    assert_eq!(
        stops_after(&mut leg, &[(1012.0, 1005.0)]),
        vec![Some(1007.0)]
    );
}
