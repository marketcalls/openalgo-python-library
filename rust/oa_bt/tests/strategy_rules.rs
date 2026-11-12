#![cfg(all(feature = "costs", feature = "time"))]
//! The strategy-level rules of Phase 1 through the slice sequence (05-risk-management 3
//! and 4, FR-RMS-04, lead rulings): the drawdown halt that latches HALTED, the cooldown
//! after an exit, the maximum number of positions, the time exit, and the checks on
//! their settings.

mod risk_support;

use oa_bt::engine::{BarColumns, Event, KernelError, RunOutput, Step, StrategyInterval};
use oa_bt::exec::{FillAt, InstrumentId};
use oa_bt::market::calendar::BaseInterval;
use oa_bt::model::{Day, Side};
use oa_bt::rms::{RiskScope, RmsError, ScaleLimits, StrategyRiskConfig};
use risk_support::*;

const A: InstrumentId = InstrumentId(0);
const B: InstrumentId = InstrumentId(1);
const CAPITAL: f64 = 1_000_000.0;

fn day(n: u32) -> Day {
    july_2025(n)
}

// ----- drawdown halt: a daily strategy on a daily base, 1000 units bought at 100.00 -----

/// Daily bars 2025-07-01 to 2025-07-04 closing 100.00, 95.00, 90.00, 90.00; the buy at
/// the 2025-07-01 close fills at the 2025-07-02 open, 100.00. Equity at the closes:
/// 1000000.00, 995000.00, 990000.00. Entries are tried at the 2025-07-03 and 2025-07-04
/// closes.
fn run_daily(risk: StrategyRiskConfig) -> (RunOutput, Script) {
    let mut bars = BarColumns::default();
    for (n, open, close) in [
        (1, 100.0, 100.0),
        (2, 100.0, 95.0),
        (3, 95.0, 90.0),
        (4, 90.0, 90.0),
    ] {
        push_bar(&mut bars, at(day(n), 9, 15), open, close);
    }
    let daily = config(
        (day(1), day(4)),
        BaseInterval::Daily,
        StrategyInterval::Daily,
        CAPITAL,
        risk,
    );
    let kernel = engine(daily, vec![equity("SBIN")], vec![bars]).expect("a valid run");
    let actions = [1, 3, 4].map(|n| (day_close(day(n)), Action::Buy(A, 1_000.0)));
    run_script(kernel, actions.to_vec())
}

fn drawdown(percent: f64) -> StrategyRiskConfig {
    StrategyRiskConfig {
        drawdown_halt_percent: Some(percent),
        ..StrategyRiskConfig::default()
    }
}

#[test]
fn a_drawdown_exactly_at_the_limit_halts_and_exits_at_that_close() {
    // Peak 1000000.00, limit 1 percent: the floor is 990000.00, reached on 2025-07-03.
    let (output, _) = run_daily(drawdown(1.0));
    let exit = fills(&output)[1];
    let expected = (
        at(day(3), 9, 15),
        Step::AggregateRules,
        Side::Sell,
        1_000.0,
        90.0,
        FillAt::Close,
        "drawdown_halt",
        false,
        true,
    );
    assert_eq!(exit, expected);
    let halt = output.events.iter().find_map(|event| match event {
        Event::Risk { event, .. } => Some(*event),
        _ => None,
    });
    let halt = halt.expect("a risk_halt event");
    assert_eq!(halt.name.code(), "risk_halt");
    assert_eq!(halt.scope, RiskScope::Strategy);
    assert_eq!(halt.level_rupees, Some(990_000.0));
    assert_eq!(halt.observed_rupees, Some(990_000.0));
}

#[test]
fn the_drawdown_halt_latches_across_sessions() {
    let (output, script) = run_daily(drawdown(1.0));
    assert_eq!(
        outcomes(&script),
        vec!["submitted", "rejected halted", "rejected halted"]
    );
    let changes: Vec<String> = events(&output)
        .into_iter()
        .map(|(_, _, name)| name)
        .filter(|name| name.starts_with("state"))
        .collect();
    assert_eq!(changes, vec!["state ACTIVE -> HALTED"]);
}

#[test]
fn a_drawdown_short_of_the_limit_does_not_halt() {
    // Equity falls to 990000.00, exactly 1 percent below the peak; with a 1.01 percent
    // limit the floor is 989900.00. The add decided on 2025-07-03 fills at 90.00, and on
    // 2025-07-04 equity is 1000000.00 + (90.00 - 95.00) x 2000 = 990000.00 again. The
    // add is decided at 90.00, below the average 100.00, so it averages down, which
    // this strategy allows (12-position-sizing 2, SPEC-15).
    let risk = StrategyRiskConfig {
        scale: ScaleLimits {
            allow_averaging_down: true,
            ..ScaleLimits::default()
        },
        ..drawdown(1.01)
    };
    let (output, script) = run_daily(risk);
    let reasons: Vec<&str> = fills(&output).iter().map(|fill| fill.6).collect();
    assert_eq!(reasons, vec!["entry", "add"]);
    assert_eq!(
        outcomes(&script),
        vec!["submitted", "submitted", "submitted"]
    );
}

#[test]
fn a_daily_loss_and_a_drawdown_on_one_close_keep_the_daily_loss_reason_and_halt() {
    // On 2025-07-03 the run is at (90 - 100) x 1000 = -10000.00, exactly the daily
    // limit, and equity is exactly at the 1 percent floor.
    let risk = StrategyRiskConfig {
        max_daily_loss_rupees: Some(10_000.0),
        ..drawdown(1.0)
    };
    let (output, script) = run_daily(risk);
    let slice_3: Vec<String> = events(&output)
        .into_iter()
        .filter(|(ts, step, _)| *ts == at(day(3), 9, 15) && *step == 6)
        .map(|(_, _, name)| name)
        .collect();
    assert_eq!(
        slice_3,
        vec![
            "daily_loss_limit_hit",
            "state ACTIVE -> REDUCING",
            "risk_halt",
            "state REDUCING -> HALTED",
            "PositionClosed",
        ]
    );
    assert_eq!(fills(&output)[1].6, "daily_loss_limit");
    assert_eq!(
        outcomes(&script),
        vec!["submitted", "rejected halted", "rejected halted"]
    );
}

// ----- cooldown, maximum positions, time exit: 15-minute strategy and base, CNC -----

/// One session, 2025-07-01, every bar of every instrument at (open, close) =
/// `price(k)`.
fn run_session(
    instruments: usize,
    price: impl Fn(usize) -> (f64, f64) + Copy,
    actions: Vec<(i64, Action)>,
    risk: StrategyRiskConfig,
) -> (RunOutput, Script) {
    let names = ["A", "B"];
    let specs = names[..instruments]
        .iter()
        .map(|name| equity(name))
        .collect();
    let bars = (0..instruments)
        .map(|_| sessions_15m(&[day(1)], |_, k| Some(price(k))))
        .collect();
    let kernel = engine(intraday_config((day(1), day(1)), risk), specs, bars).expect("a valid run");
    run_script(kernel, actions)
}

fn decision(k: usize) -> i64 {
    bar_close(day(1), k)
}

#[test]
fn entries_inside_the_cooldown_are_rejected_until_its_bars_have_closed() {
    // The exit fills in slice 3. With a cooldown of 2 bars, entries decided at the
    // closes of slices 3 and 4 are refused; the one at the close of slice 5 is not.
    let risk = StrategyRiskConfig {
        cooldown_bars: Some(2),
        ..StrategyRiskConfig::default()
    };
    let actions = vec![
        (decision(0), Action::Buy(A, 10.0)),
        (decision(2), Action::Exit(A)),
        (decision(3), Action::Buy(A, 10.0)),
        (decision(4), Action::Buy(A, 10.0)),
        (decision(5), Action::Buy(A, 10.0)),
    ];
    let (_, script) = run_session(1, |_| (100.0, 100.0), actions, risk);
    assert_eq!(
        outcomes(&script),
        vec![
            "submitted",
            "submitted",
            "rejected cooldown",
            "rejected cooldown",
            "submitted"
        ]
    );
}

#[test]
fn the_position_cap_counts_an_entry_still_waiting_for_its_fill() {
    // max_positions 1: A and B are bought at the same decision; B is refused because
    // A's entry is already working. Once A has exited, B is accepted.
    let risk = StrategyRiskConfig {
        max_positions: Some(1),
        ..StrategyRiskConfig::default()
    };
    let actions = vec![
        (decision(0), Action::Buy(A, 10.0)),
        (decision(0), Action::Buy(B, 10.0)),
        (decision(1), Action::Exit(A)),
        (decision(2), Action::Buy(B, 10.0)),
    ];
    let (_, script) = run_session(2, |_| (100.0, 100.0), actions, risk);
    assert_eq!(
        outcomes(&script),
        vec![
            "submitted",
            "rejected max_positions",
            "submitted",
            "submitted"
        ]
    );
}

#[test]
fn an_add_is_not_held_back_by_the_position_cap() {
    let risk = StrategyRiskConfig {
        max_positions: Some(1),
        ..StrategyRiskConfig::default()
    };
    let actions = vec![
        (decision(0), Action::Buy(A, 10.0)),
        (decision(1), Action::Buy(A, 10.0)),
    ];
    let (output, script) = run_session(1, |_| (100.0, 100.0), actions, risk);
    assert_eq!(outcomes(&script), vec!["submitted", "submitted"]);
    assert_eq!(output.final_positions[0].quantity_units, 20.0);
}

#[test]
fn a_position_held_max_hold_bars_exits_at_the_next_open() {
    // Bar k opens at 100 + k. Bought in slice 1 at 101.00; with max_hold 3 it is held in
    // slices 1, 2 and 3 and exits at the slice 4 open, 104.00, with no order.
    let risk = StrategyRiskConfig {
        max_hold_bars: Some(3),
        ..StrategyRiskConfig::default()
    };
    let price = |k: usize| {
        let open = 100.0 + k as f64;
        (open, open)
    };
    let (output, _) = run_session(1, price, vec![(decision(0), Action::Buy(A, 10.0))], risk);
    let expected_exit = (
        bar_open(day(1), 4),
        Step::OpenFills,
        Side::Sell,
        10.0,
        104.0,
        FillAt::Open,
        "time_exit",
        false,
        false,
    );
    assert_eq!(fills(&output)[1], expected_exit);
    // (104.00 - 101.00) x 10
    assert_eq!(output.gross_pnl.realized_rupees, 30.0);
    let closed = (bar_open(day(1), 4), 2, "PositionClosed".to_string());
    assert_eq!(events(&output).last(), Some(&closed));
}

// ----- settings -----

fn refused(risk: StrategyRiskConfig) -> Option<&'static str> {
    let bars = sessions_15m(&[day(1)], |_, _| Some((100.0, 100.0)));
    match engine(
        intraday_config((day(1), day(1)), risk),
        vec![equity("A")],
        vec![bars],
    ) {
        Err(KernelError::Rms(RmsError::InvalidRule { rule, .. })) => Some(rule),
        _ => None,
    }
}

#[test]
fn unusable_risk_settings_are_refused_before_the_run() {
    let base = StrategyRiskConfig::default;
    let cases = [
        (
            StrategyRiskConfig {
                max_daily_loss_rupees: Some(-5.0),
                ..base()
            },
            "max_daily_loss",
        ),
        (
            StrategyRiskConfig {
                drawdown_halt_percent: Some(100.0),
                ..base()
            },
            "drawdown_halt",
        ),
        (
            StrategyRiskConfig {
                cooldown_bars: Some(0),
                ..base()
            },
            "cooldown",
        ),
        (
            StrategyRiskConfig {
                max_hold_bars: Some(0),
                ..base()
            },
            "max_hold",
        ),
    ];
    for (risk, rule) in cases {
        assert_eq!(refused(risk), Some(rule));
    }
}
