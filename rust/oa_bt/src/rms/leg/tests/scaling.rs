//! Opening levels, adds, target fills and partial exits (12-position-sizing 3 and 4).
//! The scale-in sequence is trace 3.

use crate::model::Side;
use crate::rms::error::RmsError;
use crate::rms::leg::{LegContext, LegState};
use crate::rms::rules::{Distance, LegRules, TrailSpec};

use super::support::{
    bar, context, open, points, stop_and_target, stop_and_targets, stop_level, target_rows, trail,
};

fn trace_3_rules() -> LegRules {
    LegRules {
        trail: Some(trail(8.0, 0.0)),
        ..stop_and_targets(10.0, &[(10.0, 0.5), (20.0, 0.3)])
    }
}

fn trace_3_leg() -> LegState {
    match open(trace_3_rules(), Side::Buy, 1000.0, 100.0) {
        Ok(leg) => leg,
        Err(error) => panic!("the trace 3 leg opens: {error}"),
    }
}

#[test]
fn target_fractions_are_of_the_quantity_the_entry_filled() {
    // 100 at 1000.00: 50 at 1010.00, 30 at 1020.00, 20 on the stop and the trail.
    let leg = trace_3_leg();
    assert_eq!(stop_level(&leg), Some(990.0));
    assert_eq!(
        target_rows(&leg),
        vec![(1010.0, 50.0, false), (1020.0, 30.0, false)]
    );
    assert_eq!(leg.remainder_units, 20.0);
}

#[test]
fn an_add_before_any_target_fills_shares_the_tranche_and_follows_the_average() {
    // SPEC-08, SPEC-10: add 100 at 1004.00, average 1002.00.
    let mut leg = trace_3_leg();
    assert_eq!(leg.on_add(1002.0, 100.0), Ok(()));
    assert_eq!(
        leg.stop.map(|s| (s.level_rupees, s.quantity_units)),
        Some((992.0, 200.0))
    );
    assert_eq!(
        target_rows(&leg),
        vec![(1012.0, 100.0, false), (1022.0, 60.0, false)]
    );
    assert_eq!(leg.remainder_units, 40.0);
}

#[test]
fn a_target_fill_keeps_its_level_and_the_stop_shrinks_to_the_position() {
    let mut leg = trace_3_leg();
    let _ = leg.on_add(1002.0, 100.0);
    assert_eq!(leg.on_reduce(100.0, Some(0)), Ok(()));
    assert_eq!(leg.stop.map(|s| s.quantity_units), Some(100.0));
    assert_eq!(
        target_rows(&leg),
        vec![(1012.0, 100.0, true), (1022.0, 60.0, false)]
    );
    assert_eq!(leg.remainder_units, 40.0);
}

#[test]
fn an_add_after_a_target_joins_only_the_unfilled_target_and_never_loosens_the_stop() {
    // SPEC-09: target 0 filled, the trail armed at 1004.50; add 100 at 1012.00,
    // average 1007.00. The recomputed stop 997.00 is looser, so 1004.50 stays.
    let mut leg = trace_3_leg();
    let _ = leg.on_add(1002.0, 100.0);
    let _ = leg.on_reduce(100.0, Some(0));
    let _ = leg.ratchet(&bar(1006.0, 1012.5, 1005.5, 1012.0));
    assert_eq!(stop_level(&leg), Some(1004.5));
    assert_eq!(leg.on_add(1007.0, 100.0), Ok(()));
    assert_eq!(
        leg.stop.map(|s| (s.level_rupees, s.quantity_units)),
        Some((1004.5, 200.0))
    );
    assert_eq!(
        target_rows(&leg),
        vec![(1012.0, 100.0, true), (1027.0, 90.0, false)]
    );
    assert_eq!(leg.remainder_units, 110.0);
    assert_eq!(leg.trail.map(|t| t.armed), Some(true));
}

#[test]
fn averaging_down_does_not_loosen_the_stop() {
    // Long 100 at 1000.00, stop 990.00; add 100 at 980.00: average 990.00, the
    // recomputed stop 980.00 is looser, so 990.00 stays, now for 200.
    let mut leg = trace_3_leg();
    let _ = leg.on_add(990.0, 100.0);
    assert_eq!(
        leg.stop.map(|s| (s.level_rupees, s.quantity_units)),
        Some((990.0, 200.0))
    );
}

#[test]
fn a_partial_exit_by_signal_shrinks_the_working_targets_in_proportion() {
    // 100 held: targets 50 and 30, remainder 20. Exit half: 25 and 15, remainder 10.
    let mut leg = trace_3_leg();
    assert_eq!(leg.on_reduce(50.0, None), Ok(()));
    assert_eq!(
        target_rows(&leg),
        vec![(1010.0, 25.0, false), (1020.0, 15.0, false)]
    );
    assert_eq!(leg.remainder_units, 10.0);
    assert_eq!(leg.stop.map(|s| s.quantity_units), Some(50.0));
}

#[test]
fn a_target_that_rounds_to_zero_lots_is_skipped_and_its_share_stays_in_the_remainder() {
    // One 75-unit lot: half of it is no whole lot (12-position-sizing 3, SPEC-13).
    let lots = LegContext {
        lot_size_units: 75,
        ..context(Side::Buy)
    };
    let rules = stop_and_targets(50.0, &[(100.0, 0.5)]);
    let Ok(leg) = LegState::open(rules, lots, 24_000.0, 75.0) else {
        panic!("the leg opens");
    };
    assert_eq!(target_rows(&leg), vec![(24_100.0, 0.0, false)]);
    assert_eq!(leg.remainder_units, 75.0);
    let through = leg.test_bar(&bar(24_050.0, 24_150.0, 24_040.0, 24_120.0));
    assert_eq!(through, Vec::new());
}

#[test]
fn percent_levels_round_to_the_nearest_tick() {
    // Trace-06b: entry 1504.00, stop 2 percent 1473.92 -> 1473.90, target 3.5 percent
    // 1556.64 -> 1556.65.
    let rules = LegRules {
        stop: Some(Distance::Percent(2.0)),
        target: Some(Distance::Percent(3.5)),
        ..LegRules::default()
    };
    let leg = open(rules, Side::Buy, 1504.0, 10.0);
    assert_eq!(leg.as_ref().map(stop_level), Ok(Some(1473.9)));
    assert_eq!(
        leg.map(|leg| target_rows(&leg)),
        Ok(vec![(1556.65, 10.0, false)])
    );
}

#[test]
fn rupee_distances_are_spread_over_the_quantity() {
    // bt.rupees(500) on 100 units is 5 points: stop 495.00 under an entry at 500.00.
    let rules = LegRules {
        stop: Some(Distance::Rupees(500.0)),
        ..LegRules::default()
    };
    assert_eq!(
        open(rules, Side::Buy, 500.0, 100.0).map(|l| stop_level(&l)),
        Ok(Some(495.0))
    );
}

#[test]
fn a_stop_beyond_zero_is_a_configuration_error() {
    let rules = stop_and_target(10.0, 1.0);
    let error = open(rules, Side::Buy, 8.0, 10.0);
    assert!(matches!(
        error,
        Err(RmsError::InvalidRule { rule: "stop", .. })
    ));
}

#[test]
fn an_add_may_repeat_the_rules_or_give_none_but_not_change_them() {
    let leg = trace_3_leg();
    assert_eq!(leg.accepts_add_rules(&LegRules::default()), Ok(()));
    assert_eq!(leg.accepts_add_rules(&trace_3_rules()), Ok(()));
    let wider = LegRules {
        stop: Some(points(12.0)),
        ..trace_3_rules()
    };
    assert!(matches!(
        leg.accepts_add_rules(&wider),
        Err(RmsError::InvalidRule { .. })
    ));
}

#[test]
fn an_atr_rule_repeated_on_an_add_matches_whatever_the_atr_is_now() {
    let atr = |atr_rupees| LegRules {
        trail: Some(TrailSpec {
            trigger: Distance::Atr {
                multiple: 2.0,
                atr_rupees,
            },
            step: None,
        }),
        ..LegRules::default()
    };
    let Ok(leg) = open(atr(3.5), Side::Buy, 500.0, 10.0) else {
        panic!("the leg opens");
    };
    assert_eq!(leg.accepts_add_rules(&atr(4.25)), Ok(()));
}

#[test]
fn a_roll_shifts_every_level_and_the_extreme() {
    // A long at 24000.00 with a 150-point stop and a 100-point trail, rolled with a
    // spread of 120.00: stop 23850.00 -> 23970.00, reference 24120.00.
    let rules = LegRules {
        stop: Some(points(150.0)),
        trail: Some(trail(100.0, 0.0)),
        ..LegRules::default()
    };
    let Ok(mut leg) = open(rules, Side::Buy, 24_000.0, 75.0) else {
        panic!("the leg opens");
    };
    assert_eq!(leg.shift(120.0), Ok(()));
    assert_eq!(stop_level(&leg), Some(23_970.0));
    assert_eq!(leg.reference_rupees, 24_120.0);
    assert_eq!(
        leg.trail.map(|t| t.favourable_extreme_rupees),
        Some(24_120.0)
    );
}

#[test]
fn the_time_exit_is_not_built_yet() {
    let rules = LegRules {
        max_hold_bars: Some(10),
        ..LegRules::default()
    };
    assert!(matches!(
        open(rules, Side::Buy, 500.0, 10.0),
        Err(RmsError::NotBuilt { .. })
    ));
}
