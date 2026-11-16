#![cfg(all(feature = "costs", feature = "time"))]
//! The engine's own orders in a live session (`engine::venue_exits`): the exits a
//! backtest fills with no order are sent as orders where the backtest queues or fills
//! them, and the entries a backtest would refuse at their fill bar are withdrawn.
//!
//! The test day is 2025-07-01: MIS entries stop at 15:00 (slice 345) and the square-off
//! is at 15:10 (slice 355). The bars rise 0.10 a minute from 500.00
//! (`venue_support::rising`).

mod venue_support;

use oa_bt::engine::{Engine, RunConfig, RunOutput, Step, VenueCommand};
use oa_bt::exec::{FillFlags, OrderId, OrderStatus, OrderType};
use oa_bt::model::{Product, Side};
use oa_bt::rms::{ExitReason, StrategyRiskConfig};
use venue_support::{
    at_close, bars, charged_config, config, echo_until, events, exit, limit, market, minute, order,
    run_backtest, run_echo, session, slice_open, Act, Echo, Ohlc, Script,
};

/// The slice that opens at the square-off, 15:10.
const SQUARE_OFF_SLICE: usize = 355;
/// The slice that opens at the end of MIS entries, 15:00.
const ENTRY_UNTIL_SLICE: usize = 345;

/// An echoed session of `acts` on `bars`, run until it waits before `slice`; the
/// commands made before that wait are left in the engine.
fn echoed_until(
    config: RunConfig,
    columns: &oa_bt::engine::BarColumns,
    acts: Vec<(i64, Act)>,
    slice: usize,
) -> (Engine, Echo) {
    let mut kernel = session(config);
    let mut echo = Echo::default();
    echo_until(
        &mut kernel,
        columns,
        &mut Script::new(acts),
        &mut echo,
        slice,
    );
    (kernel, echo)
}

fn buy_100_at_the_open() -> Vec<(i64, Act)> {
    vec![at_close(0, Act::Submit(market(Side::Buy, 100.0)))]
}

/// The (order, side, units) of the orders placed by `commands`.
fn placed(commands: &[VenueCommand]) -> Vec<(OrderId, Side, f64)> {
    commands
        .iter()
        .filter_map(|command| match *command {
            VenueCommand::Place {
                order, side, units, ..
            } => Some((order, side, units)),
            _ => None,
        })
        .collect()
}

#[test]
fn the_square_off_is_sent_when_the_session_reaches_the_square_off_open() {
    let columns = bars(&[], &[]);
    let (mut kernel, _) = echoed_until(config(), &columns, buy_100_at_the_open(), SQUARE_OFF_SLICE);
    assert_eq!(
        placed(&kernel.take_commands()),
        [(OrderId(2), Side::Sell, 100.0)]
    );
    let square_off = kernel.order(OrderId(2)).expect("the engine's order");
    assert_eq!(square_off.request.exit_reason, Some(ExitReason::Scheduler));
    assert!(square_off.request.reduce_only);
    assert_eq!(square_off.request.order_type, OrderType::Market);
    assert_eq!(square_off.decided_at, slice_open(SQUARE_OFF_SLICE));
    // Sent before the 15:10 slice has run: the engine still waits for it.
    assert!(matches!(
        kernel.advance(),
        Ok(oa_bt::engine::Pause::Waiting {
            next_slice: SQUARE_OFF_SLICE
        })
    ));
}

#[test]
fn the_square_off_does_not_wait_for_a_bar() {
    // No bar from 15:05 on: the square-off is sent all the same.
    let missing: Vec<u32> = (350..375).collect();
    let columns = bars(&[], &missing);
    let (mut kernel, _) = echoed_until(config(), &columns, buy_100_at_the_open(), SQUARE_OFF_SLICE);
    assert_eq!(
        placed(&kernel.take_commands()),
        [(OrderId(2), Side::Sell, 100.0)]
    );
}

#[test]
fn the_square_off_is_stamped_in_the_slice_it_is_due_in() {
    let columns = bars(&[], &[]);
    let output = run_echo(
        config(),
        &columns,
        &mut Script::new(buy_100_at_the_open()),
        &mut Echo::default(),
    );
    let due = slice_open(SQUARE_OFF_SLICE);
    let at_square_off: Vec<(u8, String)> = events(&output)
        .into_iter()
        .filter(|(ts, _, _)| *ts == due)
        .map(|(_, step, name)| (step, name))
        .collect();
    let expected = [(2, "o2 Submitted"), (2, "o2 Filled"), (2, "PositionClosed")];
    let expected: Vec<(u8, String)> = expected
        .iter()
        .map(|(step, name)| (*step, (*name).to_string()))
        .collect();
    assert_eq!(at_square_off, expected);
}

/// A session's result with the engine's exit orders taken out: what a backtest records.
fn as_backtest_records(mut output: RunOutput, engine_orders: &[OrderId]) -> RunOutput {
    for record in &mut output.fills {
        if record
            .fill
            .order
            .is_some_and(|id| engine_orders.contains(&id))
        {
            record.fill.order = None;
        }
    }
    output
        .orders
        .retain(|order| !engine_orders.contains(&order.id));
    output.events.retain(|event| match event {
        oa_bt::engine::Event::Order { order, .. } => !engine_orders.contains(order),
        _ => true,
    });
    output
}

#[test]
fn an_echoed_square_off_fills_where_the_backtest_squares_off() {
    let columns = bars(&[], &[]);
    let acts = buy_100_at_the_open();
    let backtest = run_backtest(charged_config(), &columns, &mut Script::new(acts.clone()));
    let session = run_echo(
        charged_config(),
        &columns,
        &mut Script::new(acts),
        &mut Echo::default(),
    );
    let exit = session.fills.last().expect("the square-off's fill");
    assert_eq!(exit.fill.reason.code(), "scheduler");
    assert_eq!(exit.stamp.slice_ts, slice_open(SQUARE_OFF_SLICE));
    assert_eq!(as_backtest_records(session, &[OrderId(2)]), backtest);
}

#[test]
fn a_square_off_the_venue_rejects_is_sent_again_at_the_next_open() {
    let columns = bars(&[], &[]);
    let mut kernel = session(config());
    let mut echo = Echo::rejecting([OrderId(2)]);
    let mut script = Script::new(buy_100_at_the_open());
    echo_until(
        &mut kernel,
        &columns,
        &mut script,
        &mut echo,
        SQUARE_OFF_SLICE + 1,
    );
    assert_eq!(
        kernel.order(OrderId(2)).map(|o| o.status),
        Some(OrderStatus::Rejected)
    );
    assert_eq!(
        placed(&kernel.take_commands()),
        [(OrderId(3), Side::Sell, 100.0)]
    );
}

#[test]
fn a_position_the_venue_never_closes_is_reported_open_at_the_end() {
    let columns = bars(&[], &[]);
    let mut echo = Echo::rejecting((2..=40).map(OrderId));
    let output = run_echo(
        config(),
        &columns,
        &mut Script::new(buy_100_at_the_open()),
        &mut echo,
    );
    assert_eq!(output.fills.len(), 1, "no closing fill is invented");
    assert!(output.trades.is_empty());
    let held = &output.final_positions[0];
    assert_eq!((held.product, held.quantity_units), (Product::Mis, 100.0));
    assert!(!output
        .fills
        .iter()
        .any(|record| record.fill.flags.contains(FillFlags::FILLED_AT_LAST_CLOSE)));
}

#[test]
fn a_strategy_exit_already_working_covers_the_square_off() {
    let columns = bars(&[], &[]);
    let mut acts = buy_100_at_the_open();
    acts.push(at_close(
        350,
        Act::Submit(exit(Side::Sell, 100.0, limit(600.0))),
    ));
    let (mut kernel, _) = echoed_until(config(), &columns, acts, SQUARE_OFF_SLICE);
    assert!(placed(&kernel.take_commands()).is_empty());
    assert!(kernel.order(OrderId(3)).is_none());
}

/// A crash at minute 10: open 494.00, close 494.05, a loss of about 605 on 100 units
/// bought at 500.10.
const CRASH: (u32, Ohlc) = (10, (494.0, 494.5, 493.5, 494.05));

fn daily_loss_config(base: RunConfig) -> RunConfig {
    RunConfig {
        risk: StrategyRiskConfig {
            max_daily_loss_rupees: Some(500.0),
            ..StrategyRiskConfig::default()
        },
        ..base
    }
}

#[test]
fn the_daily_loss_exit_is_sent_when_the_rule_fires() {
    let columns = bars(&[CRASH], &[]);
    let config = daily_loss_config(config());
    let (mut kernel, _) = echoed_until(config, &columns, buy_100_at_the_open(), 11);
    assert_eq!(
        placed(&kernel.take_commands()),
        [(OrderId(2), Side::Sell, 100.0)]
    );
    let risk_exit = kernel.order(OrderId(2)).expect("the engine's order");
    assert_eq!(
        risk_exit.request.exit_reason,
        Some(ExitReason::DailyLossLimit)
    );
    assert_eq!(risk_exit.decided_at, minute(11));
}

#[test]
fn an_echoed_daily_loss_exit_fills_where_the_backtest_fills_it() {
    let columns = bars(&[CRASH], &[]);
    let config = daily_loss_config(charged_config());
    let acts = buy_100_at_the_open();
    let backtest = run_backtest(config.clone(), &columns, &mut Script::new(acts.clone()));
    let session = run_echo(
        config,
        &columns,
        &mut Script::new(acts),
        &mut Echo::default(),
    );
    let exit = session.fills.last().expect("the risk exit's fill");
    assert_eq!(exit.fill.reason.code(), "daily_loss_limit");
    assert_eq!(
        (exit.stamp.slice_ts, exit.stamp.step),
        (minute(11), Step::OpenFills)
    );
    assert_eq!(as_backtest_records(session, &[OrderId(2)]), backtest);
}

#[test]
fn entries_locked_by_the_daily_loss_rule_are_withdrawn_through_the_venue() {
    let columns = bars(&[CRASH], &[]);
    let mut acts = buy_100_at_the_open();
    let resting_buy = order(Side::Buy, 10.0, limit(400.0), Product::Mis);
    acts.push(at_close(5, Act::Submit(resting_buy)));
    let (mut kernel, _) = echoed_until(daily_loss_config(config()), &columns, acts, 11);
    let commands = kernel.take_commands();
    assert!(commands.contains(&VenueCommand::Cancel { order: OrderId(2) }));
    assert_eq!(
        kernel.order(OrderId(2)).map(|o| o.status),
        Some(OrderStatus::Submitted)
    );
}

#[test]
fn the_time_exit_is_sent_when_the_session_reaches_the_open_it_is_due_at() {
    // Entered at minute 1 and held at most 5 base bars: due at the open of minute 6.
    let columns = bars(&[], &[]);
    let held_five = RunConfig {
        risk: StrategyRiskConfig {
            max_hold_bars: Some(5),
            ..StrategyRiskConfig::default()
        },
        ..config()
    };
    let (mut kernel, _) = echoed_until(held_five.clone(), &columns, buy_100_at_the_open(), 6);
    assert_eq!(
        placed(&kernel.take_commands()),
        [(OrderId(2), Side::Sell, 100.0)]
    );
    let time_exit = kernel.order(OrderId(2)).expect("the engine's order");
    assert_eq!(time_exit.request.exit_reason, Some(ExitReason::TimeExit));
    let backtest = run_backtest(held_five, &columns, &mut Script::new(buy_100_at_the_open()));
    let simulated = backtest.fills.last().expect("the backtest's time exit");
    assert_eq!(simulated.stamp.slice_ts, slice_open(6));
}

#[test]
fn an_mis_entry_still_working_when_entries_stop_is_withdrawn() {
    let columns = bars(&[], &[]);
    let resting_buy = order(Side::Buy, 10.0, limit(400.0), Product::Mis);
    let acts = vec![at_close(300, Act::Submit(resting_buy))];
    let (mut kernel, _) = echoed_until(config(), &columns, acts.clone(), ENTRY_UNTIL_SLICE - 1);
    assert!(
        kernel.take_commands().is_empty(),
        "still inside the entry window"
    );
    let (mut kernel, _) = echoed_until(config(), &columns, acts, ENTRY_UNTIL_SLICE);
    assert_eq!(
        kernel.take_commands(),
        [VenueCommand::Cancel { order: OrderId(1) }]
    );
}
