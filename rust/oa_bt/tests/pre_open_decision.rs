#![cfg(all(feature = "costs", feature = "time"))]
//! The pre-open decision of a daily or longer strategy in a live session (design
//! 20-positional-live section 8, decision 11.1; unit N4 item N4-3): before the open, the
//! session decides the latest completed bar of the strategy's interval that the live layer
//! gives (the broker's candle of the session that closed), once; its orders wait for the
//! open; a decision after the open sends them at once, flagged `late_decision`.
//!
//! SBIN on NSE, a daily strategy on 1-minute slices, no charges, no slippage. Daily bars:
//! Tue 2025-07-01 and Wed 07-02 close at 1,000.00 and 800.00. A MARKET buy sized for
//! 1,00,000 rupees of value (125 units at 800.00) is decided on the bar of 07-02.

use oa_bt::data::resample::CalendarUnit;
use oa_bt::engine::{
    run, BarColumns, CostConfig, Engine, KernelError, OrderRole, Pause, PreOpen, RunConfig,
    RunSetup, StrategyBar, StrategyInterval, Submission, VenueCommand,
};
use oa_bt::exec::{FillFlags, OrderId, OrderRequest, OrderType, QuantitySpec};
use oa_bt::market::calendar::{BaseInterval, Calendar};
use oa_bt::model::{days_from_civil, ts_from_ist, Day, Product, Side};
use oa_bt::sizing::SizerSpec;

mod venue_support;

use venue_support::{config, sbin, session, Act, Script, NSE_OPEN_MINUTE, SBIN};

const CLOSE_MINUTE: u32 = 15 * 60 + 30;
const SECOND: i64 = 1_000_000_000;
const BUY: OrderId = OrderId(1);

fn july(day: i32) -> Day {
    days_from_civil(2025, 7, day as u32)
}

fn open_of(day: Day) -> i64 {
    ts_from_ist(day, NSE_OPEN_MINUTE)
}

fn close_of(day: Day) -> i64 {
    ts_from_ist(day, CLOSE_MINUTE)
}

/// The broker's daily candle of `day`, closing at `close`.
fn daily(day: Day, close: f64) -> StrategyBar {
    StrategyBar {
        instrument: SBIN,
        ts_open: open_of(day),
        ts_close: close_of(day),
        open: 1000.0,
        high: 1010.0,
        low: 790.0,
        close,
        volume: 1_000.0,
    }
}

fn daily_session(day: Day) -> Engine {
    session(RunConfig {
        first_day: day,
        last_day: day,
        strategy_interval: StrategyInterval::Daily,
        ..config()
    })
}

/// A MARKET buy of SBIN in CNC for 1,00,000 rupees of value.
fn buy() -> OrderRequest {
    OrderRequest {
        product: Product::Cnc,
        ..OrderRequest::market(
            SBIN,
            Side::Buy,
            QuantitySpec::Sizer(SizerSpec::Value { rupees: 100_000.0 }),
        )
    }
}

fn decided(outcome: PreOpen) -> (Pause, i64, bool) {
    match outcome {
        PreOpen::Decided {
            pause,
            bar_close,
            late,
        } => (pause, bar_close, late),
        PreOpen::NotDecided => panic!("a bar was to be decided"),
    }
}

fn places(commands: &[VenueCommand]) -> Vec<(OrderId, Side, f64, OrderType, OrderRole)> {
    commands
        .iter()
        .filter_map(|command| match *command {
            VenueCommand::Place {
                order,
                side,
                units,
                order_type,
                role,
                ..
            } => Some((order, side, units, order_type, role)),
            _ => None,
        })
        .collect()
}

const ENTRY: (OrderId, Side, f64, OrderType, OrderRole) =
    (BUY, Side::Buy, 125.0, OrderType::Market, OrderRole::Entry);

#[test]
fn a_start_before_the_open_decides_the_previous_sessions_bar() {
    let mut kernel = daily_session(july(3));
    let bars = [daily(july(1), 1000.0), daily(july(2), 800.0)];
    let at = open_of(july(3)) - 45 * 60 * SECOND;
    let (pause, bar_close, late) = decided(kernel.pre_open_decision(&bars, None, at).unwrap());
    assert_eq!((bar_close, late), (close_of(july(2)), false));
    let Pause::BarClose {
        decided_at, bars, ..
    } = pause
    else {
        panic!("a bar-close pause");
    };
    assert_eq!(decided_at, close_of(july(2)));
    assert_eq!(bars, vec![daily(july(2), 800.0)]);
}

#[test]
fn its_market_entry_waits_for_the_open_and_goes_as_the_session_reaches_it() {
    let mut kernel = daily_session(july(3));
    let at = open_of(july(3)) - 45 * 60 * SECOND;
    let bars = [daily(july(2), 800.0)];
    kernel.pre_open_decision(&bars, None, at).unwrap();
    assert_eq!(kernel.submit(buy()), Ok(Submission::Submitted(BUY)));
    assert_eq!(kernel.take_commands(), Vec::new());
    assert!(matches!(
        kernel.advance(),
        Ok(Pause::Waiting { next_slice: 0 })
    ));
    kernel
        .push_tick(SBIN, 801.0, open_of(july(3)) - SECOND)
        .expect("a pre-open tick");
    assert_eq!(
        kernel.take_commands(),
        Vec::new(),
        "nothing before the open"
    );
    kernel
        .push_tick(SBIN, 801.0, open_of(july(3)) + SECOND / 5)
        .expect("the open's tick");
    assert_eq!(places(&kernel.take_commands()), vec![ENTRY]);
    assert_eq!(kernel.order_flags(BUY), FillFlags::NONE);
}

#[test]
fn the_decision_and_its_order_equal_a_backtest_on_the_same_daily_bars() {
    // The backtest decides at the close of 07-02 and fills at the 07-03 open.
    let mut columns = BarColumns::default();
    for (day, close) in [(1, 1000.0), (2, 800.0), (3, 805.0)] {
        columns.ts_open.push(open_of(july(day)));
        columns.open.push(1000.0);
        columns.high.push(1010.0);
        columns.low.push(790.0);
        columns.close.push(close);
        columns.volume.push(1_000.0);
    }
    let backtest_config = RunConfig {
        first_day: july(1),
        last_day: july(3),
        base: BaseInterval::Daily,
        strategy_interval: StrategyInterval::Daily,
        costs: CostConfig::Off,
        ..config()
    };
    let setup = RunSetup {
        config: backtest_config,
        calendar: Calendar::new().expect("the embedded calendar tables parse"),
        instruments: vec![sbin()],
        bars: vec![columns],
    };
    let mut backtest = Engine::new(setup).expect("a valid backtest");
    let mut script = Script::new(vec![(close_of(july(2)), Act::Submit(buy()))]);
    run(&mut backtest, &mut script).expect("the backtest completes");
    let tested = backtest.order(BUY).expect("the backtest's order").clone();
    let mut kernel = daily_session(july(3));
    let at = open_of(july(3)) - 45 * 60 * SECOND;
    kernel
        .pre_open_decision(&[daily(july(2), 800.0)], None, at)
        .unwrap();
    kernel.submit(buy()).expect("the same order");
    let live = kernel.order(BUY).expect("the session's order").clone();
    assert_eq!(live.decided_at, tested.decided_at);
    assert_eq!(live.quantity_units, tested.quantity_units);
    assert_eq!(live.request.side, tested.request.side);
    assert_eq!(live.request.order_type, tested.request.order_type);
    let fill = &backtest.finish().unwrap().fills[0];
    assert_eq!(
        fill.stamp.slice_ts,
        open_of(july(3)),
        "the backtest fills at the open"
    );
}

#[test]
fn a_second_start_the_same_morning_does_not_decide_the_bar_again() {
    let mut kernel = daily_session(july(3));
    let at = open_of(july(3)) - 30 * 60 * SECOND;
    let last = Some(close_of(july(2)));
    let outcome = kernel.pre_open_decision(&[daily(july(2), 800.0)], last, at);
    assert_eq!(outcome, Ok(PreOpen::NotDecided));
    assert!(matches!(
        kernel.submit(buy()),
        Err(KernelError::NotAtDecisionPoint)
    ));
    assert_eq!(
        kernel.last_close(SBIN),
        Some(800.0),
        "the bar is fed all the same"
    );
}

#[test]
fn a_start_after_the_open_sends_at_once_flagged_late_decision() {
    let mut kernel = daily_session(july(3));
    let at = open_of(july(3)) + 5 * 60 * SECOND;
    let (_, _, late) = decided(
        kernel
            .pre_open_decision(&[daily(july(2), 800.0)], None, at)
            .unwrap(),
    );
    assert!(late);
    kernel.submit(buy()).expect("submitted");
    assert_eq!(places(&kernel.take_commands()), vec![ENTRY]);
    assert!(kernel.order_flags(BUY).contains(FillFlags::LATE_DECISION));
    assert!(matches!(kernel.advance(), Ok(Pause::Waiting { .. })));
    kernel.apply_fill(BUY, 125.0, 801.0, at).expect("the fill");
    let flags = kernel.order_flags(BUY);
    assert_eq!(
        flags,
        FillFlags::NONE,
        "the order has ended; its fill keeps the flag"
    );
    kernel.stop_session().expect("stopped");
    let output = kernel.finish().expect("finished");
    assert!(output.fills[0]
        .fill
        .flags
        .contains(FillFlags::LATE_DECISION));
}

#[test]
fn of_two_missed_days_only_the_latest_bar_is_decided() {
    let mut kernel = daily_session(july(4));
    let bars = [daily(july(2), 800.0), daily(july(3), 820.0)];
    let at = open_of(july(4)) - 60 * SECOND;
    let last = Some(close_of(july(1)));
    let (pause, bar_close, _) = decided(kernel.pre_open_decision(&bars, last, at).unwrap());
    assert_eq!(bar_close, close_of(july(3)));
    let Pause::BarClose { bars: decided, .. } = pause else {
        panic!("a bar-close pause");
    };
    assert_eq!(decided, vec![daily(july(3), 820.0)]);
    assert_eq!(kernel.last_close(SBIN), Some(820.0));
}

#[test]
fn a_daily_session_makes_no_decision_of_its_own() {
    let mut kernel = daily_session(july(3));
    let mut decisions = 0;
    loop {
        match kernel.advance().expect("a slice") {
            Pause::Waiting { next_slice } if next_slice < 375 => kernel.release(next_slice + 1),
            Pause::Finished { .. } => break,
            Pause::BarClose { .. } => decisions += 1,
            _ => {}
        }
    }
    assert_eq!(decisions, 0);
}

fn weekly_session(day: Day) -> Engine {
    session(RunConfig {
        first_day: day,
        last_day: day,
        strategy_interval: StrategyInterval::Period(CalendarUnit::Week),
        ..config()
    })
}

/// The broker's weekly candle of the week of Mon 2025-06-30, closing on Fri 07-04.
fn week_bar() -> StrategyBar {
    StrategyBar {
        ts_open: open_of(days_from_civil(2025, 6, 30)),
        ts_close: close_of(july(4)),
        ..daily(july(4), 810.0)
    }
}

#[test]
fn a_weekly_strategy_decides_only_at_the_start_after_the_weeks_last_session() {
    // Friday morning: the week's bar has not closed.
    let mut friday = weekly_session(july(4));
    let at = open_of(july(4)) - 60 * SECOND;
    let outcome = friday.pre_open_decision(&[week_bar()], None, at);
    assert_eq!(outcome, Ok(PreOpen::NotDecided));
    // Monday morning: decided.
    let mut monday = weekly_session(july(7));
    let at = open_of(july(7)) - 60 * SECOND;
    let (_, bar_close, _) = decided(monday.pre_open_decision(&[week_bar()], None, at).unwrap());
    assert_eq!(bar_close, close_of(july(4)));
    // Tuesday morning: decided already.
    let mut tuesday = weekly_session(july(8));
    let at = open_of(july(8)) - 60 * SECOND;
    let last = Some(close_of(july(4)));
    let outcome = tuesday.pre_open_decision(&[week_bar()], last, at);
    assert_eq!(outcome, Ok(PreOpen::NotDecided));
}

#[test]
fn a_bar_that_is_not_of_the_strategys_interval_is_refused() {
    let mut weekly = weekly_session(july(7));
    let at = open_of(july(7)) - 60 * SECOND;
    let refused = weekly.pre_open_decision(&[daily(july(4), 810.0)], None, at);
    assert!(matches!(
        refused,
        Err(KernelError::InvalidConfig {
            field: "pre_open_decision",
            ..
        })
    ));
}

#[test]
fn an_intraday_strategy_and_a_second_call_are_refused() {
    let mut intraday = session(config());
    let at = open_of(july(1)) - 60 * SECOND;
    let refused = intraday.pre_open_decision(&[], None, at);
    assert!(matches!(refused, Err(KernelError::InvalidConfig { .. })));
    let mut kernel = daily_session(july(3));
    kernel.pre_open_decision(&[], None, at).unwrap();
    let again = kernel.pre_open_decision(&[], None, at);
    assert!(matches!(again, Err(KernelError::InvalidConfig { .. })));
}

#[test]
fn an_order_waiting_for_the_open_survives_a_restart_and_goes_at_the_open() {
    let mut kernel = daily_session(july(3));
    let at = open_of(july(3)) - 45 * 60 * SECOND;
    kernel
        .pre_open_decision(&[daily(july(2), 800.0)], None, at)
        .unwrap();
    kernel.submit(buy()).expect("submitted");
    assert!(matches!(kernel.advance(), Ok(Pause::Waiting { .. })));
    let saved = kernel.live_state();
    let mut restarted = daily_session(july(3));
    let ids = restarted.restore_live_state(&saved).expect("restored");
    assert_eq!(ids, vec![(BUY, BUY)]);
    assert_eq!(
        restarted.take_commands(),
        Vec::new(),
        "nothing before the open"
    );
    restarted
        .push_tick(SBIN, 801.0, open_of(july(3)) + SECOND / 5)
        .expect("the open's tick");
    assert_eq!(places(&restarted.take_commands()), vec![ENTRY]);
}
