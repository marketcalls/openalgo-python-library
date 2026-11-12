#![cfg(all(feature = "costs", feature = "time"))]
//! The daily loss limit through the slice sequence (05-risk-management 5, decisions
//! SPEC-16, SPEC-17 and SPEC-18, lead rulings on the gate): the measure on the closes,
//! the breach in step 6, REDUCING until the next session, the exits, and the
//! `daily_loss_gate` rejection at placement.
//!
//! The daily-base cases replay the daily-loss sequence of trace 8b on its near
//! contract alone. Trace 8b rolls to the next contract on 2025-07-30; at zero cost and
//! zero slippage the roll is P&L-neutral and both contracts close 95.00 apart on that
//! day, so the near contract alone breaches at the same close with the same measure,
//! -16125.00. Futures margin is not built yet, so the near contract's bars are traded
//! as a cash-equity stand-in (lot 1, 75 units, CNC); the leg stop and trail of 8b are
//! left out because neither is reached (lows above 24710.00, excursion below 200).

mod risk_support;

use oa_bt::engine::{BarColumns, Event, RunOutput, Step, StrategyInterval};
use oa_bt::exec::{FillAt, InstrumentId};
use oa_bt::market::calendar::BaseInterval;
use oa_bt::model::{Day, Side};
use oa_bt::rms::{AfterRiskExit, RiskScope, StrategyRiskConfig, TradingState};
use risk_support::*;

/// The near contract of the daily cases.
const NEAR: InstrumentId = InstrumentId(0);
/// The traded instrument of the intraday cases.
const A: InstrumentId = InstrumentId(0);
/// A second intraday instrument.
const OTHER: InstrumentId = InstrumentId(1);
/// Capital that covers the CNC margin of 75 units near 25000.00.
const CAPITAL_8B: f64 = 2_500_000.0;
/// One lot of the NIFTY contract of trace 8b, in units.
const LOT_8B: f64 = 75.0;

/// The near contract of trace 8b, 2025-07-25 to 2025-07-31: (day, open, high, low, close).
const NEAR_ROWS: [(u32, [f64; 4]); 5] = [
    (25, [24_950.0, 25_040.0, 24_900.0, 25_000.0]),
    (28, [25_010.0, 25_010.0, 24_930.0, 24_960.0]),
    (29, [24_950.0, 25_090.0, 24_900.0, 24_940.0]),
    (30, [24_920.0, 24_965.0, 24_785.0, 24_795.0]),
    (31, [24_805.0, 24_855.0, 24_765.0, 24_835.0]),
];

fn limit(rupees: f64) -> StrategyRiskConfig {
    StrategyRiskConfig {
        max_daily_loss_rupees: Some(rupees),
        ..StrategyRiskConfig::default()
    }
}

/// Trace 8b's decisions on the near contract: buy one lot at the 2025-07-25 close, and
/// again at the 2025-07-30 and 2025-07-31 closes.
fn run_8b(risk: StrategyRiskConfig) -> (RunOutput, Script) {
    let mut bars = BarColumns::default();
    for (day, prices) in NEAR_ROWS {
        push_ohlc(&mut bars, at(july_2025(day), 9, 15), prices);
    }
    let days = (july_2025(25), july_2025(31));
    let daily = config(
        days,
        BaseInterval::Daily,
        StrategyInterval::Daily,
        CAPITAL_8B,
        risk,
    );
    let kernel = engine(daily, vec![equity("NEAR")], vec![bars]).expect("a valid run");
    let actions = [25, 30, 31].map(|day| (day_close(july_2025(day)), Action::Buy(NEAR, LOT_8B)));
    run_script(kernel, actions.to_vec())
}

fn slice(day: u32) -> i64 {
    at(july_2025(day), 9, 15)
}

#[test]
fn trace_8b_breaches_at_the_roll_day_close_and_exits_at_that_close() {
    let (output, _) = run_8b(limit(10_000.0));
    assert_eq!(
        fills(&output),
        vec![
            (
                slice(28),
                Step::OpenFills,
                Side::Buy,
                LOT_8B,
                25_010.0,
                FillAt::Open,
                "entry",
                true,
                false
            ),
            (
                slice(30),
                Step::AggregateRules,
                Side::Sell,
                LOT_8B,
                24_795.0,
                FillAt::Close,
                "daily_loss_limit",
                false,
                true
            ),
        ]
    );
    // (24795.00 - 25010.00) x 75
    assert_eq!(output.gross_pnl.realized_rupees, -16_125.0);
    assert_eq!(output.gross_pnl.unrealized_rupees, 0.0);
}

#[test]
fn trace_8b_writes_the_breach_the_lock_and_the_gate_rejection_in_order() {
    let (output, _) = run_8b(limit(10_000.0));
    let event = |day, step, name: &str| (slice(day), step, name.to_string());
    assert_eq!(
        events(&output),
        vec![
            event(25, 7, "Submitted"),
            event(28, 2, "Filled"),
            event(28, 2, "PositionOpened"),
            event(30, 6, "daily_loss_limit_hit"),
            event(30, 6, "state ACTIVE -> REDUCING"),
            event(30, 6, "PositionClosed"),
            event(30, 7, "Rejected daily_loss_gate"),
            event(31, 1, "state REDUCING -> ACTIVE"),
            event(31, 7, "Submitted"),
        ]
    );
}

#[test]
fn the_breach_event_reports_minus_the_limit_and_the_run_mark_to_market() {
    let (output, _) = run_8b(limit(10_000.0));
    let breach = output.events.iter().find_map(|event| match event {
        Event::Risk {
            instrument, event, ..
        } => Some((*instrument, *event)),
        _ => None,
    });
    let (instrument, event) = breach.expect("a daily loss event");
    assert_eq!(instrument, None);
    assert_eq!(event.scope, RiskScope::Session);
    assert_eq!(event.level_rupees, Some(-10_000.0));
    assert_eq!(event.observed_rupees, Some(-16_125.0));
}

#[test]
fn reducing_lasts_from_the_breach_to_the_next_session() {
    let (_, script) = run_8b(limit(10_000.0));
    let states: Vec<TradingState> = script.states.iter().map(|(_, state)| *state).collect();
    use TradingState::{Active, Reducing};
    // Decisions at the closes of 2025-07-25, 28, 29, 30 and 31.
    assert_eq!(states, vec![Active, Active, Active, Reducing, Active]);
    assert_eq!(
        outcomes(&script),
        vec!["submitted", "rejected daily_loss_gate", "submitted"]
    );
}

#[test]
fn a_loss_exactly_at_the_limit_breaches() {
    let (output, _) = run_8b(limit(16_125.0));
    let exit = output.fills.last().map(|record| record.fill.reason.code());
    assert_eq!(exit, Some("daily_loss_limit"));
    assert_eq!(output.gross_pnl.realized_rupees, -16_125.0);
}

#[test]
fn a_loss_one_paisa_short_of_the_limit_does_not_breach() {
    // 2025-07-30 measures -16125.00 and 2025-07-31 -13125.00: neither reaches -16125.01.
    let (output, script) = run_8b(limit(16_125.01));
    assert_eq!(output.fills.len(), 1);
    assert!(script
        .states
        .iter()
        .all(|(_, s)| *s == TradingState::Active));
}

#[test]
fn the_measure_is_the_runs_mtm_since_its_first_fill_not_the_days_change() {
    // On 2025-07-30 the run is at -16125.00 since its first fill, but the day's own
    // change is (24795.00 - 24940.00) x 75 = -10875.00: a 12000.00 limit breaches only
    // on the run's measure (SPEC-16).
    let (output, _) = run_8b(limit(12_000.0));
    let observed = output.events.iter().find_map(|event| match event {
        Event::Risk { event, .. } => event.observed_rupees,
        _ => None,
    });
    assert_eq!(observed, Some(-16_125.0));
}

#[test]
fn after_risk_exit_does_not_release_the_daily_loss_lock() {
    let risk = StrategyRiskConfig {
        after_risk_exit: AfterRiskExit::AllowReentry,
        ..limit(10_000.0)
    };
    let (_, script) = run_8b(risk);
    assert_eq!(
        outcomes(&script),
        vec!["submitted", "rejected daily_loss_gate", "submitted"]
    );
}

// ----- intraday: a 15-minute strategy on a 15-minute base, CNC, limit 1000.00 -----

fn day1() -> Day {
    july_2025(1)
}

fn day2() -> Day {
    july_2025(2)
}

/// Run on 2025-07-01 and 2025-07-02 with bars from `price(day_index, k)`.
fn run_intraday(
    price: impl Fn(usize, usize) -> Option<(f64, f64)>,
    actions: Vec<(i64, Action)>,
    risk: StrategyRiskConfig,
) -> (RunOutput, Script) {
    let bars = sessions_15m(&[day1(), day2()], price);
    let kernel = engine(
        intraday_config((day1(), day2()), risk),
        vec![equity("SBIN")],
        vec![bars],
    )
    .expect("a valid run");
    run_script(kernel, actions)
}

#[test]
fn an_intraday_breach_exits_at_the_next_open_and_locks_entries_until_the_next_session() {
    // Bought 100 at 100.00 (bar 1 open). Bar 4 closes at 89.00: (89 - 100) x 100 =
    // -1100.00 breaches. The exit fills at bar 5's open, 90.00. Entries decided at the
    // closes of bars 4 and 23 are refused; the first decision of the next day is not.
    let price = |day: usize, k: usize| {
        Some(match (day, k) {
            (0, 0..=3) => (100.0, 100.0),
            (0, 4) => (100.0, 89.0),
            _ => (90.0, 90.0),
        })
    };
    let buy = Action::Buy(A, 100.0);
    let actions = vec![
        (bar_close(day1(), 0), buy),
        (bar_close(day1(), 4), buy),
        (bar_close(day1(), 23), buy),
        (bar_close(day2(), 0), buy),
    ];
    let (output, script) = run_intraday(price, actions, limit(1_000.0));
    let event = |day, k, step, name: &str| (bar_open(day, k), step, name.to_string());
    assert_eq!(
        events(&output),
        vec![
            event(day1(), 0, 7, "Submitted"),
            event(day1(), 1, 2, "Filled"),
            event(day1(), 1, 2, "PositionOpened"),
            event(day1(), 4, 6, "daily_loss_limit_hit"),
            event(day1(), 4, 6, "state ACTIVE -> REDUCING"),
            event(day1(), 4, 7, "Rejected daily_loss_gate"),
            event(day1(), 5, 2, "PositionClosed"),
            event(day1(), 23, 7, "Rejected daily_loss_gate"),
            event(day2(), 0, 1, "state REDUCING -> ACTIVE"),
            event(day2(), 0, 7, "Submitted"),
            event(day2(), 1, 2, "Filled"),
            event(day2(), 1, 2, "PositionOpened"),
        ]
    );
    let exit = fills(&output)[1];
    let expected = (
        bar_open(day1(), 5),
        Step::OpenFills,
        Side::Sell,
        100.0,
        90.0,
        FillAt::Open,
        "daily_loss_limit",
        false,
        false,
    );
    assert_eq!(exit, expected);
    assert_eq!(script.submissions.len(), 4);
}

#[test]
fn a_breach_on_the_last_bar_of_a_session_exits_at_that_close() {
    // Bar 24 (15:15) closes the session at 85.00: (85 - 100) x 100 = -1500.00. There is
    // no later bar, so the exit fills in step 6 at that close, flagged; the next
    // session starts ACTIVE.
    let price = |day: usize, k: usize| {
        Some(match (day, k) {
            (0, 24) => (100.0, 85.0),
            (0, _) => (100.0, 100.0),
            _ => (85.0, 85.0),
        })
    };
    let buy = Action::Buy(A, 100.0);
    let actions = vec![(bar_close(day1(), 0), buy), (bar_close(day2(), 0), buy)];
    let (output, script) = run_intraday(price, actions, limit(1_000.0));
    let exit = fills(&output)[1];
    let last_bar = bar_open(day1(), 24);
    let expected = (
        last_bar,
        Step::AggregateRules,
        Side::Sell,
        100.0,
        85.0,
        FillAt::Close,
        "daily_loss_limit",
        false,
        true,
    );
    assert_eq!(exit, expected);
    assert_eq!(outcomes(&script), vec!["submitted", "submitted"]);
    let boundary = (
        bar_open(day2(), 0),
        1,
        "state REDUCING -> ACTIVE".to_string(),
    );
    assert!(events(&output).contains(&boundary));
}

#[test]
fn an_exit_whose_instrument_has_no_later_bar_fills_at_the_session_close() {
    // Bar 20 (14:15) closes at 89.00 and breaches; bars 21 to 24 are missing, so the
    // queued exit fills in step 8 of the last slice at the 89.00 close, flagged.
    let price = |day: usize, k: usize| match (day, k) {
        (0, 20) => Some((100.0, 89.0)),
        (0, 21..) => None,
        _ => Some((100.0, 100.0)),
    };
    let buy = Action::Buy(A, 100.0);
    let (output, _) = run_intraday(price, vec![(bar_close(day1(), 0), buy)], limit(1_000.0));
    let exit = fills(&output)[1];
    let expected = (
        bar_open(day1(), 24),
        Step::SessionClose,
        Side::Sell,
        100.0,
        89.0,
        FillAt::Close,
        "daily_loss_limit",
        false,
        true,
    );
    assert_eq!(exit, expected);
}

#[test]
fn a_loss_realized_by_a_run_that_ended_this_session_breaches_while_flat() {
    // Bought 100 at 100.00; the strategy's exit fills at bar 3's open, 88.00: -1200.00
    // realized by a run that ended this session. The strategy is flat, yet REDUCING.
    let price = |_: usize, k: usize| Some(if k < 3 { (100.0, 100.0) } else { (88.0, 88.0) });
    let actions = vec![
        (bar_close(day1(), 0), Action::Buy(A, 100.0)),
        (bar_close(day1(), 2), Action::Exit(A)),
        (bar_close(day1(), 3), Action::Buy(A, 100.0)),
    ];
    let (output, script) = run_intraday(price, actions, limit(1_000.0));
    assert_eq!(
        outcomes(&script),
        vec!["submitted", "submitted", "rejected daily_loss_gate"]
    );
    let breach = (bar_open(day1(), 3), 6, "daily_loss_limit_hit".to_string());
    assert!(events(&output).contains(&breach));
    assert_eq!(output.fills.len(), 2);
}

#[test]
fn an_entry_still_waiting_for_a_bar_is_cancelled_when_entries_lock() {
    // A is bought and breaches at bar 3's close (89.00). An entry on OTHER, decided at
    // bar 1's close, waits because OTHER has no bar from bar 2 to bar 6 (SPEC-06): it is
    // cancelled in step 6 of the breach, so it can never open a position while REDUCING.
    let a = sessions_15m(&[day1()], |_, k| {
        Some(match k {
            0..=2 => (100.0, 100.0),
            3 => (100.0, 89.0),
            _ => (90.0, 90.0),
        })
    });
    let other = sessions_15m(&[day1()], |_, k| {
        (!(2..=6).contains(&k)).then_some((50.0, 50.0))
    });
    let kernel = engine(
        intraday_config((day1(), day1()), limit(1_000.0)),
        vec![equity("A"), equity("OTHER")],
        vec![a, other],
    )
    .expect("a valid run");
    let actions = vec![
        (bar_close(day1(), 0), Action::Buy(A, 100.0)),
        (bar_close(day1(), 1), Action::Buy(OTHER, 100.0)),
    ];
    let (output, _) = run_script(kernel, actions);
    let step_6: Vec<String> = events(&output)
        .into_iter()
        .filter(|(ts, step, _)| *ts == bar_open(day1(), 3) && *step == 6)
        .map(|(_, _, name)| name)
        .collect();
    assert_eq!(
        step_6,
        vec![
            "daily_loss_limit_hit",
            "state ACTIVE -> REDUCING",
            "Canceled"
        ]
    );
    assert!(output
        .fills
        .iter()
        .all(|record| record.fill.instrument == A));
}
