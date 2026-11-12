#![cfg(all(feature = "costs", feature = "time"))]
//! GTTs holding the stops and targets of CNC positions (04-execution 4.4 and the
//! decisions GTT-1 to GTT-11): which GTTs a position gets, how they fire, how they
//! follow trails and adds, cancel-then-exit, the engine's exit after an unfilled child,
//! slippage on child fills and the 360-day renewal.
//!
//! A daily strategy on daily bars from 2025-07-01, CNC, no charges. Every order is
//! decided at a day's close and fills at the next day's open; a day the test does not
//! write is the flat bar (1500.00, 1510.00, 1490.00, 1500.00).

mod order_support;

use oa_bt::engine::{RunOutput, Step};
use oa_bt::exec::{
    FillAt, FillFlags, Gtt, GttKind, GttLegRole, GttStatus, OrderType, QuantitySpec,
};
use oa_bt::market::slippage::SlippageSpec;
use oa_bt::model::{Day, Product, Side};
use oa_bt::rms::{Distance, LegRules, StrategyRiskConfig, TargetSpec, TrailSpec};
use order_support::{
    day_open, events_at, fills, july_2025, order, positional_buy, run_daily, Act, Ohlc,
};

const FLAT: Ohlc = (1500.0, 1510.0, 1490.0, 1500.0);
/// The July 2025 sessions the tests use: 1 to 4, then 7 to 10.
const SESSIONS: [u32; 8] = [1, 2, 3, 4, 7, 8, 9, 10];

/// The bars of the July sessions, flat unless written.
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

fn points(value: f64) -> Distance {
    Distance::Points(value)
}

fn stop(value: f64) -> LegRules {
    LegRules {
        stop: Some(points(value)),
        ..LegRules::default()
    }
}

/// Buy 10 with `rules` at the close of 2025-07-01 (fills at the 07-02 open, 1500.00).
fn buy_on_the_first(rules: LegRules) -> Vec<(Day, Act)> {
    vec![(july_2025(1), Act::Submit(positional_buy(10.0, rules)))]
}

fn zero_slippage(_: &mut oa_bt::engine::RunConfig) {}

/// (day open, step, side, units, price, where, reason) of every fill.
fn fill_rows(output: &RunOutput) -> Vec<(i64, Step, Side, f64, f64, FillAt, &'static str)> {
    fills(output)
}

/// A GTT's (stop, target) triggers, rupees.
type Triggers = (Option<f64>, Option<f64>);

fn triggers(gtt: &Gtt) -> Triggers {
    (
        gtt.trigger(GttLegRole::Stop),
        gtt.trigger(GttLegRole::Target),
    )
}

// ---------------------------------------------------------------- which GTTs (GTT-11)

#[test]
fn a_stop_alone_is_one_single_gtt_for_the_whole_position() {
    let (_, gtts, _) = run_daily(&july(&[]), buy_on_the_first(stop(50.0)), zero_slippage);
    assert_eq!(gtts.len(), 1);
    let gtt = &gtts[0];
    assert_eq!(
        (gtt.kind, gtt.side, gtt.quantity_units),
        (GttKind::Single, Side::Sell, 10.0)
    );
    assert_eq!(triggers(gtt), (Some(1450.0), None));
    // The child sells 1 percent below the trigger: 1435.50 (GTT-1).
    assert_eq!(gtt.legs[0].child_limit_rupees, 1435.5);
    assert_eq!(gtt.status, GttStatus::Active);
}

#[test]
fn a_stop_with_a_target_is_one_oco_gtt() {
    let rules = LegRules {
        target: Some(points(30.0)),
        ..stop(50.0)
    };
    let (_, gtts, _) = run_daily(&july(&[]), buy_on_the_first(rules), zero_slippage);
    assert_eq!(gtts.len(), 1);
    assert_eq!(gtts[0].kind, GttKind::Oco);
    assert_eq!(triggers(&gtts[0]), (Some(1450.0), Some(1530.0)));
}

/// 100 units, stop 50 points, targets 20 points for half and 40 points for 30 percent.
fn two_targets() -> Vec<(Day, Act)> {
    let rules = LegRules {
        targets: vec![
            TargetSpec {
                distance: points(20.0),
                fraction: 0.5,
            },
            TargetSpec {
                distance: points(40.0),
                fraction: 0.3,
            },
        ],
        ..stop(50.0)
    };
    vec![(july_2025(1), Act::Submit(positional_buy(100.0, rules)))]
}

#[test]
fn several_targets_are_one_oco_each_and_a_single_stop_for_the_remainder() {
    let (_, gtts, _) = run_daily(&july(&[]), two_targets(), zero_slippage);
    let held: Vec<(GttKind, f64, Triggers)> = gtts
        .iter()
        .map(|gtt| (gtt.kind, gtt.quantity_units, triggers(gtt)))
        .collect();
    assert_eq!(
        held,
        [
            (GttKind::Oco, 50.0, (Some(1450.0), Some(1520.0))),
            (GttKind::Oco, 30.0, (Some(1450.0), Some(1540.0))),
            (GttKind::Single, 20.0, (Some(1450.0), None)),
        ]
    );
}

#[test]
fn a_target_gtt_that_fires_leaves_the_others_working_until_the_stop() {
    // 07-03 reaches 1525: the first target (1520) fills 50. 07-04 falls to 1445: the
    // stop legs of the other two GTTs sell 30 and 20 at 1450.
    let bars = july(&[
        (3, (1505.0, 1525.0, 1500.0, 1510.0)),
        (4, (1505.0, 1508.0, 1445.0, 1450.0)),
    ]);
    let (output, gtts, _) = run_daily(&bars, two_targets(), zero_slippage);
    let exits: Vec<(f64, f64, &str, Option<u16>)> = output.fills[1..]
        .iter()
        .map(|r| {
            (
                r.fill.quantity_units,
                r.fill.price_rupees,
                r.fill.reason.code(),
                r.fill.target_index,
            )
        })
        .collect();
    assert_eq!(
        exits,
        [
            (50.0, 1520.0, "leg_target", Some(0)),
            (30.0, 1450.0, "leg_sl", None),
            (20.0, 1450.0, "leg_sl", None),
        ]
    );
    let statuses: Vec<GttStatus> = gtts.iter().map(|gtt| gtt.status).collect();
    assert_eq!(statuses, [GttStatus::ChildFilled; 3]);
}

// ---------------------------------------------------------------- firing (OCO)

#[test]
fn an_oco_target_fires_inside_the_range_and_cancels_its_stop() {
    // Stop 1450, target 1530. 07-03 trades up to 1535: the target leg fires at 1530; its
    // child sells at 1514.70 or better, marketable at 1530.
    let rules = LegRules {
        target: Some(points(30.0)),
        ..stop(50.0)
    };
    let bars = july(&[(3, (1510.0, 1535.0, 1505.0, 1520.0))]);
    let (output, gtts, _) = run_daily(&bars, buy_on_the_first(rules), zero_slippage);
    let day = day_open(july_2025(3));
    assert_eq!(
        fill_rows(&output)[1],
        (
            day,
            Step::Levels,
            Side::Sell,
            10.0,
            1530.0,
            FillAt::Level,
            "leg_target"
        )
    );
    assert_eq!(
        events_at(&output, day),
        [
            "GttTriggered g1",
            "leg_target_hit",
            "GttChildFilled g1",
            "PositionClosed"
        ]
    );
    assert_eq!(gtts[0].fired, Some(GttLegRole::Target));
}

#[test]
fn with_both_triggers_inside_one_bar_the_stop_fires() {
    let rules = LegRules {
        target: Some(points(30.0)),
        ..stop(50.0)
    };
    let bars = july(&[(3, (1500.0, 1535.0, 1445.0, 1500.0))]);
    let (output, gtts, _) = run_daily(&bars, buy_on_the_first(rules), zero_slippage);
    assert_eq!(gtts[0].fired, Some(GttLegRole::Stop));
    assert_eq!(output.fills[1].fill.price_rupees, 1450.0);
}

#[test]
fn a_target_crossed_at_the_open_fires_before_a_stop_inside_the_range() {
    // 07-03 opens at 1540, beyond the target 1530, then falls through the stop 1450:
    // the target fires at the open and its child sells at the open.
    let rules = LegRules {
        target: Some(points(30.0)),
        ..stop(50.0)
    };
    let bars = july(&[(3, (1540.0, 1545.0, 1440.0, 1450.0))]);
    let (output, gtts, _) = run_daily(&bars, buy_on_the_first(rules), zero_slippage);
    assert_eq!(gtts[0].fired, Some(GttLegRole::Target));
    assert_eq!(
        (output.fills[1].fill.price_rupees, output.fills[1].fill.at),
        (1540.0, FillAt::Open)
    );
}

// ---------------------------------------------------------------- modification (GTT-3)

#[test]
fn a_trail_modifies_the_gtt_in_step_4_and_the_new_stop_holds_from_the_next_bar() {
    // Trail 20 points. 07-03 reaches 1530: armed, the stop moves to 1510 in step 4. That
    // bar's own low of 1500 does not fire it; 07-04's low of 1505 does, at 1510.
    let rules = LegRules {
        trail: Some(TrailSpec {
            trigger: points(20.0),
            step: None,
        }),
        ..stop(50.0)
    };
    let bars = july(&[
        (2, (1500.0, 1505.0, 1495.0, 1500.0)),
        (3, (1505.0, 1530.0, 1500.0, 1525.0)),
        (4, (1515.0, 1520.0, 1505.0, 1508.0)),
    ]);
    let (output, gtts, _) = run_daily(&bars, buy_on_the_first(rules), zero_slippage);
    let day_3 = day_open(july_2025(3));
    assert_eq!(
        events_at(&output, day_3),
        ["leg_trail_armed", "GttModified g1"]
    );
    assert_eq!(gtts[0].last_modified_slice, Some(2));
    let exit = &output.fills[1];
    assert_eq!(
        (exit.stamp.slice_ts, exit.fill.price_rupees),
        (day_open(july_2025(4)), 1510.0)
    );
}

#[test]
fn an_add_modifies_the_gtt_to_the_new_quantity_and_stop_in_step_4() {
    // 10 at 1500 (stop 1450), then 10 more at the 07-03 open of 1520: average 1510, the
    // fixed stop from it 1460 is the tighter one. GTT g1 becomes 20 units at 1460.
    let bars = july(&[(3, (1520.0, 1525.0, 1515.0, 1520.0))]);
    let mut acts = buy_on_the_first(stop(50.0));
    acts.push((july_2025(2), Act::Submit(positional_buy(10.0, stop(50.0)))));
    let (output, gtts, _) = run_daily(&bars, acts, zero_slippage);
    let day_3 = day_open(july_2025(3));
    assert_eq!(
        events_at(&output, day_3),
        ["Filled", "PositionChanged", "GttModified g1"]
    );
    assert_eq!(
        (gtts[0].quantity_units, triggers(&gtts[0])),
        (20.0, (Some(1460.0), None))
    );
}

// ---------------------------------------------------------------- cancel-then-exit (GTT-9)

#[test]
fn a_signal_exit_cancels_the_gtt_before_it_fills() {
    let mut acts = buy_on_the_first(stop(50.0));
    acts.push((
        july_2025(2),
        Act::Submit(order_support::exit_all(Side::Buy, Product::Cnc)),
    ));
    let (output, gtts, _) = run_daily(&july(&[]), acts, zero_slippage);
    assert_eq!(
        events_at(&output, day_open(july_2025(3))),
        ["GttCanceled g1", "Filled", "PositionClosed"]
    );
    assert_eq!(gtts[0].status, GttStatus::Canceled);
}

#[test]
fn a_partial_exit_cancels_the_gtt_and_places_one_for_the_rest_in_step_4() {
    let reduce = oa_bt::exec::OrderRequest {
        reduce_only: true,
        quantity: QuantitySpec::Units(4.0),
        ..order(Side::Sell, 4.0, OrderType::Market, Product::Cnc)
    };
    let mut acts = buy_on_the_first(stop(50.0));
    acts.push((july_2025(2), Act::Submit(reduce)));
    let (output, gtts, _) = run_daily(&july(&[]), acts, zero_slippage);
    assert_eq!(
        events_at(&output, day_open(july_2025(3))),
        [
            "GttCanceled g1",
            "Filled",
            "PositionChanged",
            "GttPlaced g2"
        ]
    );
    assert_eq!(
        (gtts[1].quantity_units, triggers(&gtts[1])),
        (6.0, (Some(1450.0), None))
    );
}

#[test]
fn the_daily_loss_exit_cancels_the_gtt_first() {
    // Stop 100 points (1400). 07-03 closes at 1460: a loss of 400 against a limit of
    // 300. The exit fills at that close in step 6, after the GTT is cancelled.
    let bars = july(&[(3, (1480.0, 1485.0, 1455.0, 1460.0))]);
    let (output, _, _) = run_daily(&bars, buy_on_the_first(stop(100.0)), |config| {
        config.risk = StrategyRiskConfig {
            max_daily_loss_rupees: Some(300.0),
            ..StrategyRiskConfig::default()
        };
    });
    let names = events_at(&output, day_open(july_2025(3)));
    assert_eq!(names[0], "daily_loss_limit_hit");
    let cancel = names.iter().position(|n| n == "GttCanceled g1");
    let closed = names.iter().position(|n| n == "PositionClosed");
    assert!(cancel.is_some() && cancel < closed, "{names:?}");
}

// ---------------------------------------------------------------- unfilled child (GTT-4)

#[test]
fn an_unfilled_child_on_a_friday_exits_at_mondays_open_plus_slippage() {
    // Two ticks of slippage: bought at 1500.10, stop 1450.10, child limit 1435.55
    // (1450.10 x 0.99 = 1435.599, down to the tick). Friday 07-04 opens at 1430, beyond
    // both, and its high 1434 never comes back to the limit. The engine exits at Monday
    // 07-07's open 1420 less two ticks, flagged.
    let bars = july(&[
        (4, (1430.0, 1434.0, 1420.0, 1425.0)),
        (7, (1420.0, 1428.0, 1415.0, 1422.0)),
    ]);
    let (output, gtts, _) = run_daily(&bars, buy_on_the_first(stop(50.0)), |config| {
        config.slippage = SlippageSpec::Ticks(2);
    });
    let friday = day_open(july_2025(4));
    assert_eq!(
        events_at(&output, friday),
        [
            "GttTriggered g1",
            "leg_sl_hit",
            "gtt_child_unfilled",
            "GttChildCanceled g1"
        ]
    );
    let exit = &output.fills[1];
    assert_eq!(
        (exit.stamp.slice_ts, exit.stamp.step, exit.fill.price_rupees),
        (day_open(july_2025(7)), Step::OpenFills, 1419.9)
    );
    assert!(exit.fill.flags.contains(FillFlags::GTT_CHILD_UNFILLED));
    assert_eq!(exit.fill.gtt, Some(gtts[0].id));
    assert_eq!(gtts[0].status, GttStatus::ChildCanceled);
}

// ---------------------------------------------------------------- child pricing (GTT-5, GTT-7)

#[test]
fn a_marketable_child_fills_at_its_trigger_less_slippage() {
    // Two ticks of slippage: bought at 1500.10, stop 1450.10, reached inside 07-03's
    // range; the marketable child sells at the trigger less two ticks: 1450.00.
    let bars = july(&[(3, (1480.0, 1485.0, 1445.0, 1460.0))]);
    let (output, _, _) = run_daily(&bars, buy_on_the_first(stop(50.0)), |config| {
        config.slippage = SlippageSpec::Ticks(2);
    });
    assert_eq!(output.fills[1].fill.price_rupees, 1450.0);
}

#[test]
fn a_marketable_child_never_fills_past_its_limit() {
    // Two percent of slippage: bought at 1530.00, stop 1480.00, child limit 1465.20.
    // 07-03 opens at the stop; two percent below 1480 is 1450.40, past the limit, so
    // the child fills at its limit (GTT-5).
    let bars = july(&[(3, (1480.0, 1485.0, 1445.0, 1460.0))]);
    let (output, _, _) = run_daily(&bars, buy_on_the_first(stop(50.0)), |config| {
        config.slippage = SlippageSpec::PercentOfPrice {
            percent: 2.0,
            min_ticks: 0,
        };
    });
    assert_eq!(output.fills[1].fill.price_rupees, 1465.2);
}

#[test]
fn a_child_filled_at_its_limit_after_a_gap_takes_no_slippage() {
    // Two ticks of slippage: bought at 1500.10, stop 1450.10, child limit 1435.55.
    // 07-03 opens at 1430, beyond the limit, and trades back up to 1440 (GTT-7).
    let bars = july(&[(3, (1430.0, 1440.0, 1425.0, 1435.0))]);
    let (output, _, _) = run_daily(&bars, buy_on_the_first(stop(50.0)), |config| {
        config.slippage = SlippageSpec::Ticks(2);
    });
    let exit = output.fills[1].fill;
    assert_eq!((exit.price_rupees, exit.at), (1435.55, FillAt::Level));
}

// ---------------------------------------------------------------- renewal (GTT-2)

#[test]
fn a_gtt_older_than_360_days_is_cancelled_and_placed_again_at_a_session_open() {
    // Bought at the close of 2025-01-01, filled on 2025-01-02, when g1 is created. The
    // first session more than 360 days later is Monday 2025-12-29 (2025-01-02 + 361).
    let calendar = oa_bt::market::calendar::Calendar::new().expect("calendar");
    let first = oa_bt::model::days_from_civil(2025, 1, 1);
    let last = oa_bt::model::days_from_civil(2026, 1, 9);
    let bars: Vec<(Day, Ohlc)> = (first..=last)
        .filter(|day| {
            calendar
                .session(oa_bt::model::Exchange::Nse, *day)
                .is_some()
        })
        .map(|day| (day, FLAT))
        .collect();
    let acts = vec![(first, Act::Submit(positional_buy(10.0, stop(100.0))))];
    let (output, gtts, _) = run_daily(&bars, acts, zero_slippage);
    let renewal = oa_bt::model::days_from_civil(2025, 12, 29);
    assert_eq!(
        events_at(&output, day_open(renewal)),
        ["GttCanceled g1", "GttPlaced g2"]
    );
    assert_eq!(gtts[0].status, GttStatus::Expired);
    assert_eq!(
        (gtts[1].created_on, gtts[1].status),
        (renewal, GttStatus::Active)
    );
    assert_eq!(triggers(&gtts[1]), (Some(1400.0), None));
}
