#![cfg(all(feature = "costs", feature = "time"))]
//! A CNC or NRML position's stop is one SINGLE GTT for all its open units, and its
//! targets are held by the engine exactly as an MIS leg's are (design 20-positional-live
//! P6, 04-execution 4.4 item 2): a target exit fills at the target or at a gapped open, a
//! full one cancels the stop GTT first, a partial one cuts the GTT to the units left, and
//! inside one bar a level crossed at the open comes first, else the stop. A leg with a
//! stop has no level tested in a bar where its stop GTT is not in force (the bar of a
//! partial exit at the open), and breakeven after the first target moves the GTT.
//!
//! A daily strategy on daily bars, no charges, no slippage. The position is opened at
//! the close of 2025-07-01 and fills at the 07-02 open of 1000.00; 07-02 reaches neither
//! level, and the bar under test is 07-03's. A day the test does not write is the flat
//! bar (1000.00, 1005.00, 995.00, 1000.00).

mod order_support;

use oa_bt::engine::{RunOutput, Step};
use oa_bt::exec::{FillAt, Gtt, GttId, GttLegRole, GttStatus, InstrumentSpec, OrderRequest};
use oa_bt::exec::{OrderType, TriggerDirection};
use oa_bt::model::{Day, Product, Side};
use oa_bt::rms::{AfterFirstTarget, Distance, LegRules, TargetSpec};
use order_support::{
    day_open, equity, events_at, fills, future, july_2025, order, run_daily_on, Act, FillRow, Ohlc,
};

const FLAT: Ohlc = (1000.0, 1005.0, 995.0, 1000.0);
/// The entry day: inside both levels of either side.
const ENTRY_DAY: Ohlc = (1000.0, 1010.0, 990.0, 1000.0);
/// 100 shares of the long, one lot of 75 of the short future.
const LONG_UNITS: f64 = 100.0;
const SHORT_UNITS: f64 = 75.0;

fn points(value: f64) -> Distance {
    Distance::Points(value)
}

/// Stop 50 points and target 100 points: 950.00 and 1100.00 for the long, 1050.00 and
/// 900.00 for the short.
fn stop_and_target() -> LegRules {
    LegRules {
        stop: Some(points(50.0)),
        target: Some(points(100.0)),
        ..LegRules::default()
    }
}

/// Stop 50 points, half at 100 points (1100.00) and half at 150 points (1150.00).
fn stop_and_two_targets() -> LegRules {
    let half = |distance| TargetSpec {
        distance: points(distance),
        fraction: 0.5,
    };
    LegRules {
        stop: Some(points(50.0)),
        targets: vec![half(100.0), half(150.0)],
        ..LegRules::default()
    }
}

/// A CNC buy of 100 shares or an NRML sale of one lot, with `rules`.
fn opening(side: Side, rules: LegRules) -> OrderRequest {
    let (units, product) = match side {
        Side::Buy => (LONG_UNITS, Product::Cnc),
        Side::Sell => (SHORT_UNITS, Product::Nrml),
    };
    OrderRequest {
        leg_rules: rules,
        ..order(side, units, OrderType::Market, product)
    }
}

/// The run: the position opened as `side` with `rules`, then `third` on 07-03, `fourth`
/// on 07-04 and a flat bar on 07-07.
fn run(side: Side, rules: LegRules, third: Ohlc, fourth: Ohlc) -> (RunOutput, Vec<Gtt>) {
    run_with(side, rules, (third, fourth), Vec::new())
}

/// `run` with the strategy's `later` orders, each decided at its day's close.
fn run_with(
    side: Side,
    rules: LegRules,
    (third, fourth): (Ohlc, Ohlc),
    later: Vec<(Day, Act)>,
) -> (RunOutput, Vec<Gtt>) {
    let instrument: InstrumentSpec = match side {
        Side::Buy => equity(),
        Side::Sell => future(),
    };
    let bars: Vec<(Day, Ohlc)> = [
        (1, FLAT),
        (2, ENTRY_DAY),
        (3, third),
        (4, fourth),
        (7, FLAT),
    ]
    .iter()
    .map(|&(day, bar)| (july_2025(day), bar))
    .collect();
    let mut acts = vec![(july_2025(1), Act::Submit(opening(side, rules)))];
    acts.extend(later);
    let (output, gtts, _) = run_daily_on(instrument, &bars, acts, |_| {});
    (output, gtts)
}

fn third_day() -> i64 {
    day_open(july_2025(3))
}

/// The exit of the position: its fill row, target index and GTT.
fn exit_of(output: &RunOutput) -> (FillRow, Option<u16>, Option<GttId>) {
    let rows = fills(output);
    assert_eq!(rows.len(), 2, "one entry and one exit: {rows:?}");
    let record = &output.fills[1];
    (rows[1], record.fill.target_index, record.fill.gtt)
}

// ---------------------------------------------------------------- the GTT a position gets

#[test]
fn a_long_with_a_stop_and_a_target_gets_one_single_stop_gtt_for_all_its_units() {
    let (_, gtts) = run(Side::Buy, stop_and_target(), FLAT, FLAT);
    assert_eq!(gtts.len(), 1, "{gtts:?}");
    let gtt = &gtts[0];
    let stop = gtt.leg;
    assert_eq!(
        (stop.role, stop.trigger_rupees, stop.direction),
        (GttLegRole::Stop, 950.0, TriggerDirection::Falling)
    );
    // The child sells 1 percent below the trigger: 950.00 x 0.99 = 940.50.
    assert_eq!(stop.child_limit_rupees, 940.5);
    assert_eq!((gtt.side, gtt.quantity_units), (Side::Sell, LONG_UNITS));
    assert_eq!(gtt.status, GttStatus::Active);
}

#[test]
fn a_short_future_gets_one_single_stop_gtt_above_the_price() {
    let (_, gtts) = run(Side::Sell, stop_and_target(), FLAT, FLAT);
    assert_eq!(gtts.len(), 1, "{gtts:?}");
    let gtt = &gtts[0];
    let stop = gtt.leg;
    assert_eq!(
        (stop.role, stop.trigger_rupees, stop.direction),
        (GttLegRole::Stop, 1050.0, TriggerDirection::Rising)
    );
    // The child buys 1 percent above the trigger: 1050.00 x 1.01 = 1060.50.
    assert_eq!(stop.child_limit_rupees, 1060.5);
    assert_eq!((gtt.side, gtt.quantity_units), (Side::Buy, SHORT_UNITS));
}

#[test]
fn several_targets_still_get_one_stop_gtt_for_all_the_units() {
    let (_, gtts) = run(Side::Buy, stop_and_two_targets(), FLAT, FLAT);
    assert_eq!(gtts.len(), 1, "{gtts:?}");
    assert_eq!(
        (gtts[0].trigger(GttLegRole::Stop), gtts[0].quantity_units),
        (Some(950.0), LONG_UNITS)
    );
}

// ---------------------------------------------------------------- a target reached

#[test]
fn a_long_target_reached_inside_the_bar_exits_at_the_target_after_cancelling_the_gtt() {
    let (output, gtts) = run(
        Side::Buy,
        stop_and_target(),
        (1020.0, 1105.0, 1015.0, 1090.0),
        FLAT,
    );
    let row = (
        third_day(),
        Step::Levels,
        Side::Sell,
        LONG_UNITS,
        1100.0,
        FillAt::Level,
        "leg_target",
    );
    assert_eq!(exit_of(&output), (row, Some(0), None));
    assert_eq!(
        events_at(&output, third_day()),
        ["leg_target_hit", "GttCanceled g1", "PositionClosed"]
    );
    assert_eq!(gtts[0].status, GttStatus::Canceled);
}

#[test]
fn a_long_gapping_open_beyond_the_target_exits_at_the_open() {
    let (output, _) = run(
        Side::Buy,
        stop_and_target(),
        (1120.0, 1130.0, 1110.0, 1125.0),
        FLAT,
    );
    let row = (
        third_day(),
        Step::Levels,
        Side::Sell,
        LONG_UNITS,
        1120.0,
        FillAt::Open,
        "leg_target",
    );
    assert_eq!(exit_of(&output), (row, Some(0), None));
    assert_eq!(
        events_at(&output, third_day()),
        ["leg_target_hit", "GttCanceled g1", "PositionClosed"]
    );
}

#[test]
fn a_short_target_reached_inside_the_bar_exits_at_the_target_after_cancelling_the_gtt() {
    let (output, gtts) = run(
        Side::Sell,
        stop_and_target(),
        (980.0, 985.0, 895.0, 910.0),
        FLAT,
    );
    let row = (
        third_day(),
        Step::Levels,
        Side::Buy,
        SHORT_UNITS,
        900.0,
        FillAt::Level,
        "leg_target",
    );
    assert_eq!(exit_of(&output), (row, Some(0), None));
    assert_eq!(
        events_at(&output, third_day()),
        ["leg_target_hit", "GttCanceled g1", "PositionClosed"]
    );
    assert_eq!(gtts[0].status, GttStatus::Canceled);
}

#[test]
fn a_short_gapping_open_beyond_the_target_exits_at_the_open() {
    let (output, _) = run(
        Side::Sell,
        stop_and_target(),
        (880.0, 890.0, 870.0, 885.0),
        FLAT,
    );
    let row = (
        third_day(),
        Step::Levels,
        Side::Buy,
        SHORT_UNITS,
        880.0,
        FillAt::Open,
        "leg_target",
    );
    assert_eq!(exit_of(&output), (row, Some(0), None));
}

// ---------------------------------------------------------------- the stop and a target in one bar

#[test]
fn a_long_bar_reaching_both_levels_from_inside_exits_at_the_stop_by_its_gtt() {
    // Opens at 1000.00 between the levels, trades to 940.00 and 1110.00: the stop first.
    let (output, gtts) = run(
        Side::Buy,
        stop_and_target(),
        (1000.0, 1110.0, 940.0, 1000.0),
        FLAT,
    );
    let row = (
        third_day(),
        Step::Levels,
        Side::Sell,
        LONG_UNITS,
        950.0,
        FillAt::Level,
        "leg_sl",
    );
    assert_eq!(exit_of(&output), (row, None, Some(gtts[0].id)));
    assert_eq!(
        events_at(&output, third_day()),
        [
            "GttTriggered g1",
            "leg_sl_hit",
            "GttChildFilled g1",
            "PositionClosed"
        ]
    );
}

#[test]
fn a_short_bar_reaching_both_levels_from_inside_exits_at_the_stop_by_its_gtt() {
    let (output, gtts) = run(
        Side::Sell,
        stop_and_target(),
        (1000.0, 1060.0, 890.0, 1000.0),
        FLAT,
    );
    let row = (
        third_day(),
        Step::Levels,
        Side::Buy,
        SHORT_UNITS,
        1050.0,
        FillAt::Level,
        "leg_sl",
    );
    assert_eq!(exit_of(&output), (row, None, Some(gtts[0].id)));
}

#[test]
fn a_long_stop_crossed_at_the_open_fires_its_gtt_before_a_target_inside_the_bar() {
    // Opens at 940.00, beyond the stop 950.00 and its child's limit 940.50, then trades
    // up to 1105.00 through the target: the GTT fires at the open and its child fills at
    // its limit when the price comes back to it (GTT-7); the target is not tested.
    let (output, gtts) = run(
        Side::Buy,
        stop_and_target(),
        (940.0, 1105.0, 930.0, 1000.0),
        FLAT,
    );
    let row = (
        third_day(),
        Step::Levels,
        Side::Sell,
        LONG_UNITS,
        940.5,
        FillAt::Level,
        "leg_sl",
    );
    assert_eq!(exit_of(&output), (row, None, Some(gtts[0].id)));
}

#[test]
fn a_long_target_crossed_at_the_open_wins_over_the_stop_inside_the_bar() {
    // Opens at 1120.00 beyond the target, then falls through the stop to 930.00: the
    // target exits at the open, and the stop is not tested against the rest of the bar.
    let (output, gtts) = run(
        Side::Buy,
        stop_and_target(),
        (1120.0, 1125.0, 930.0, 940.0),
        FLAT,
    );
    let row = (
        third_day(),
        Step::Levels,
        Side::Sell,
        LONG_UNITS,
        1120.0,
        FillAt::Open,
        "leg_target",
    );
    assert_eq!(exit_of(&output), (row, Some(0), None));
    assert_eq!(gtts[0].status, GttStatus::Canceled);
}

// ---------------------------------------------------------------- a partial target exit

#[test]
fn a_first_target_exit_cuts_the_stop_gtt_to_the_units_left_in_step_4() {
    // 07-03 reaches 1105.00: the first target sells 50 at 1100.00 and the stop GTT g1 is
    // modified to the 50 shares left, at the same trigger.
    let (output, gtts) = run(
        Side::Buy,
        stop_and_two_targets(),
        (1020.0, 1105.0, 1015.0, 1090.0),
        FLAT,
    );
    assert_eq!(
        events_at(&output, third_day()),
        ["leg_target_hit", "PositionChanged", "GttModified g1"]
    );
    assert_eq!(gtts.len(), 1, "{gtts:?}");
    assert_eq!(
        (
            gtts[0].status,
            gtts[0].quantity_units,
            gtts[0].trigger(GttLegRole::Stop)
        ),
        (GttStatus::Active, 50.0, Some(950.0))
    );
    let target = &output.fills[1];
    assert_eq!(
        (
            target.stamp.step,
            target.fill.quantity_units,
            target.fill.price_rupees,
            target.fill.target_index,
            target.fill.gtt
        ),
        (Step::Levels, 50.0, 1100.0, Some(0), None)
    );
}

#[test]
fn after_a_first_target_the_stop_gtt_sells_the_units_left_in_one_order() {
    // 07-04 falls to 940.00 through the stop: g1 sells the 50 shares left at 950.00.
    let (output, gtts) = run(
        Side::Buy,
        stop_and_two_targets(),
        (1020.0, 1105.0, 1015.0, 1090.0),
        (1080.0, 1085.0, 940.0, 945.0),
    );
    let exits: Vec<(f64, f64, &str, Option<GttId>)> = output.fills[1..]
        .iter()
        .map(|r| {
            (
                r.fill.quantity_units,
                r.fill.price_rupees,
                r.fill.reason.code(),
                r.fill.gtt,
            )
        })
        .collect();
    assert_eq!(
        exits,
        [
            (50.0, 1100.0, "leg_target", None),
            (50.0, 950.0, "leg_sl", Some(gtts[0].id)),
        ]
    );
}

/// Stop 50 points and one target, 100 points (1100.00), for half the position.
fn stop_and_a_target_for_half() -> LegRules {
    LegRules {
        stop: Some(points(50.0)),
        targets: vec![TargetSpec {
            distance: points(100.0),
            fraction: 0.5,
        }],
        ..LegRules::default()
    }
}

#[test]
fn with_a_target_for_half_the_position_the_stop_gtt_sells_every_unit_in_one_order() {
    // g1 holds the stop of all 100 shares, the target's half included. 07-03 falls to
    // 940.00 through the stop and never reaches the target: g1 sells the 100 at 950.00,
    // one exit fill.
    let (output, gtts) = run(
        Side::Buy,
        stop_and_a_target_for_half(),
        (1000.0, 1005.0, 940.0, 945.0),
        FLAT,
    );
    assert_eq!(gtts.len(), 1, "{gtts:?}");
    let row = (
        third_day(),
        Step::Levels,
        Side::Sell,
        LONG_UNITS,
        950.0,
        FillAt::Level,
        "leg_sl",
    );
    assert_eq!(exit_of(&output), (row, None, Some(gtts[0].id)));
}

// ---------------------------------------------------------------- the bar of a partial exit at the open

/// Through both levels of the long from inside: the stop at 950.00 and the target at
/// 1100.00.
const BOTH_LEVELS: Ohlc = (1000.0, 1110.0, 940.0, 1000.0);

/// The strategy sells 50 of the 100 shares at the close of 07-02; the sale fills at the
/// 07-03 open of 1000.00.
fn sell_half_on_the_second() -> Vec<(Day, Act)> {
    let reduce = OrderRequest {
        reduce_only: true,
        ..order(Side::Sell, 50.0, OrderType::Market, Product::Cnc)
    };
    vec![(july_2025(2), Act::Submit(reduce))]
}

#[test]
fn a_partial_exit_at_the_open_leaves_no_level_of_a_leg_with_a_stop_tested_in_that_bar() {
    // The sale cancels g1 first (cancel-then-exit, GTT-9) and g2 holds the stop of the
    // 50 left from the next base bar, so no stop is held in 07-03. Its targets are not
    // tested alone there either: a bar through both levels would otherwise exit at the
    // target, where the stop comes first for an MIS leg.
    let (output, gtts) = run_with(
        Side::Buy,
        stop_and_target(),
        (BOTH_LEVELS, FLAT),
        sell_half_on_the_second(),
    );
    let sale = (
        third_day(),
        Step::OpenFills,
        Side::Sell,
        50.0,
        1000.0,
        FillAt::Open,
        "signal",
    );
    assert_eq!(fills(&output)[1..], [sale]);
    assert_eq!(
        events_at(&output, third_day()),
        [
            "GttCanceled g1",
            "Filled",
            "PositionChanged",
            "GttPlaced g2"
        ]
    );
    assert_eq!(
        (
            gtts[1].status,
            gtts[1].quantity_units,
            gtts[1].trigger(GttLegRole::Stop)
        ),
        (GttStatus::Active, 50.0, Some(950.0))
    );
}

#[test]
fn a_leg_without_a_stop_has_its_target_tested_in_the_bar_of_a_partial_exit() {
    // No stop, so no GTT: the target is the engine's alone, tested in every bar as an MIS
    // leg's. 07-03 reaches 1110.00 and the target sells the 50 left at 1100.00.
    let rules = LegRules {
        target: Some(points(100.0)),
        ..LegRules::default()
    };
    let (output, gtts) = run_with(
        Side::Buy,
        rules,
        (BOTH_LEVELS, FLAT),
        sell_half_on_the_second(),
    );
    assert!(gtts.is_empty(), "{gtts:?}");
    let target = (
        third_day(),
        Step::Levels,
        Side::Sell,
        50.0,
        1100.0,
        FillAt::Level,
        "leg_target",
    );
    assert_eq!(fills(&output)[2..], [target]);
}

// ---------------------------------------------------------------- breakeven after the first target

/// `stop_and_two_targets` with `after_first_target="breakeven"`.
fn two_targets_then_breakeven() -> LegRules {
    LegRules {
        after_first_target: AfterFirstTarget::Breakeven,
        ..stop_and_two_targets()
    }
}

/// 07-03 reaches the first target; 07-04 opens at 1080.00 and falls to 990.00.
fn breakeven_run() -> (RunOutput, Vec<Gtt>) {
    run(
        Side::Buy,
        two_targets_then_breakeven(),
        (1020.0, 1105.0, 1015.0, 1090.0),
        (1080.0, 1085.0, 990.0, 995.0),
    )
}

#[test]
fn breakeven_after_the_first_target_moves_the_stop_gtt_to_the_entry_for_the_units_left() {
    // The first target sells 50 at 1100.00 and moves the stop to the entry price 1000.00;
    // step 4 modifies g1 once, for both its quantity and its trigger.
    let (output, gtts) = breakeven_run();
    assert_eq!(
        events_at(&output, third_day()),
        [
            "leg_target_hit",
            "PositionChanged",
            "leg_breakeven_armed",
            "GttModified g1"
        ]
    );
    assert_eq!(gtts.len(), 1, "{gtts:?}");
    assert_eq!(
        (gtts[0].quantity_units, gtts[0].trigger(GttLegRole::Stop)),
        (50.0, Some(1000.0))
    );
    // The child sells 1 percent below the moved trigger: 1000.00 x 0.99 = 990.00.
    assert_eq!(gtts[0].leg.child_limit_rupees, 990.0);
}

#[test]
fn after_breakeven_the_stop_gtt_sells_the_units_left_at_the_entry() {
    // 07-04 falls through the moved trigger 1000.00: g1 fires inside the bar and sells
    // the 50 left at 1000.00.
    let (output, gtts) = breakeven_run();
    let fourth_day = day_open(july_2025(4));
    assert_eq!(
        events_at(&output, fourth_day),
        [
            "GttTriggered g1",
            "leg_sl_hit",
            "GttChildFilled g1",
            "PositionClosed"
        ]
    );
    let exit = (
        fourth_day,
        Step::Levels,
        Side::Sell,
        50.0,
        1000.0,
        FillAt::Level,
        "leg_sl",
    );
    assert_eq!(fills(&output)[2..], [exit]);
    assert_eq!(
        (output.fills[2].fill.gtt, gtts[0].status),
        (Some(gtts[0].id), GttStatus::ChildFilled)
    );
}
