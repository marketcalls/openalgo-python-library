#![cfg(all(feature = "costs", feature = "time"))]
//! The engine's exits take over from the strategy's in a live session
//! (`engine::venue_owner`, `venue_exits`): the square-off and the daily loss exit cancel a
//! strategy exit already sent, then close the whole position.
//!
//! The test day is 2025-07-01: the square-off is at 15:10 (slice 355) and the session's
//! last slice is 374 (15:29). The bars rise 0.10 a minute from 500.00
//! (`venue_support::rising`).

mod venue_support;

use oa_bt::engine::{Engine, RunConfig, VenueCommand};
use oa_bt::exec::{OrderId, OrderRequest, OrderStatus, OrderType, QuantitySpec};
use oa_bt::model::{Product, Side};
use oa_bt::rms::{ExitReason, StrategyRiskConfig};
use venue_support::{
    at_close, bars, config, exit, market, minute, run_echo, session, wait_before, Act, Echo, Ohlc,
    Script, SBIN,
};

/// The slice that opens at the square-off, 15:10.
const SQUARE_OFF_SLICE: usize = 355;
/// The session's last slice, 15:29.
const LAST_SLICE: u32 = 374;
/// 15:29 IST in minutes since midnight: a square-off at the session's last slice.
const LAST_SLICE_MINUTE: u32 = 15 * 60 + 29;

/// `reduce(0.5)`: a reduce-only MARKET sale of half the position.
fn halve() -> OrderRequest {
    OrderRequest {
        quantity: QuantitySpec::PositionFraction(0.5),
        ..exit(Side::Sell, 1.0, OrderType::Market)
    }
}

/// A session of `config` on `bars` that buys 100 at the close of minute 0 (filled in
/// minute 1) and halves the position at the close of minute `halved`, run with no other
/// venue result until it waits before slice `until`; the commands since the buy filled
/// are left.
fn halved_before(config: RunConfig, columns: &[(u32, Ohlc)], halved: u32, until: usize) -> Engine {
    let columns = bars(columns, &[]);
    let mut kernel = session(config);
    let mut script = Script::new(vec![
        at_close(0, Act::Submit(market(Side::Buy, 100.0))),
        at_close(halved, Act::Submit(halve())),
    ]);
    wait_before(&mut kernel, &columns, &mut script, 1);
    kernel
        .apply_fill(OrderId(1), 100.0, 500.1, minute(1))
        .expect("the buy fills");
    kernel.take_commands();
    wait_before(&mut kernel, &columns, &mut script, until);
    kernel
}

fn engine_exit(kernel: &mut Engine) -> (Vec<(OrderId, f64)>, Option<ExitReason>) {
    let placed: Vec<(OrderId, f64)> = kernel
        .take_commands()
        .into_iter()
        .filter_map(|command| match command {
            VenueCommand::Place { order, units, .. } => Some((order, units)),
            _ => None,
        })
        .collect();
    let reason = placed
        .first()
        .and_then(|(id, _)| kernel.order(*id))
        .and_then(|order| order.request.exit_reason);
    (placed, reason)
}

#[test]
fn the_square_off_cancels_a_strategy_exit_sent_in_the_slice_before() {
    let last = u32::try_from(SQUARE_OFF_SLICE - 1).expect("a slice of the day");
    let mut kernel = halved_before(config(), &[], last, SQUARE_OFF_SLICE);
    let commands = kernel.take_commands();
    assert_eq!(
        commands.last(),
        Some(&VenueCommand::Cancel { order: OrderId(2) })
    );
    assert_eq!(commands.len(), 2, "the halving was placed, then cancelled");
}

#[test]
fn the_square_off_closes_the_whole_position_once_the_strategy_exit_has_ended() {
    let last = u32::try_from(SQUARE_OFF_SLICE - 1).expect("a slice of the day");
    let mut kernel = halved_before(config(), &[], last, SQUARE_OFF_SLICE);
    kernel.take_commands();
    assert!(kernel.apply_cancel(OrderId(2)).is_ok());
    assert_eq!(
        engine_exit(&mut kernel),
        (vec![(OrderId(3), 100.0)], Some(ExitReason::Scheduler))
    );
}

#[test]
fn a_square_off_at_the_sessions_last_slice_takes_over_from_a_strategy_exit() {
    let mut config = config();
    config.intraday.square_off_minute = Some(LAST_SLICE_MINUTE);
    let acts = vec![
        at_close(0, Act::Submit(market(Side::Buy, 100.0))),
        at_close(LAST_SLICE - 1, Act::Submit(halve())),
    ];
    let output = run_echo(
        config,
        &bars(&[], &[]),
        &mut Script::new(acts),
        &mut Echo::default(),
    );
    assert_eq!(output.final_positions[0].quantity_units, 0.0);
    let squared = output.fills.last().expect("the square-off's fill");
    assert_eq!(
        (
            squared.stamp.slice,
            squared.fill.quantity_units,
            squared.fill.reason.code()
        ),
        (LAST_SLICE, 100.0, "scheduler")
    );
}

/// A crash at minute 10: open 494.00, close 494.05, a loss of about 605 on 100 units
/// bought at 500.10.
const CRASH: (u32, Ohlc) = (10, (494.0, 494.5, 493.5, 494.05));

fn daily_loss_config() -> RunConfig {
    RunConfig {
        risk: StrategyRiskConfig {
            max_daily_loss_rupees: Some(500.0),
            ..StrategyRiskConfig::default()
        },
        ..config()
    }
}

#[test]
fn the_daily_loss_exit_takes_over_from_a_strategy_exit_already_sent() {
    // The halving goes out at the close of minute 8 and is still working when the crash
    // breaches the limit at the close of minute 10.
    let mut kernel = halved_before(daily_loss_config(), &[CRASH], 8, 11);
    let commands = kernel.take_commands();
    assert_eq!(
        commands.last(),
        Some(&VenueCommand::Cancel { order: OrderId(2) })
    );
    assert!(kernel.apply_cancel(OrderId(2)).is_ok());
    assert_eq!(
        engine_exit(&mut kernel),
        (vec![(OrderId(3), 100.0)], Some(ExitReason::DailyLossLimit))
    );
}

#[test]
fn a_strategy_exit_of_a_position_the_engine_exits_ends_canceled() {
    let last = u32::try_from(SQUARE_OFF_SLICE - 1).expect("a slice of the day");
    let mut kernel = halved_before(config(), &[], last, SQUARE_OFF_SLICE);
    kernel.take_commands();
    let mut script = Script::new(vec![at_close(
        u32::try_from(SQUARE_OFF_SLICE).expect("a slice of the day"),
        Act::Submit(halve()),
    )]);
    wait_before(
        &mut kernel,
        &bars(&[], &[]),
        &mut script,
        SQUARE_OFF_SLICE + 1,
    );
    let late = kernel.order(OrderId(3)).expect("the strategy's later exit");
    assert_eq!(late.status, OrderStatus::Canceled);
    let held = kernel
        .position(SBIN, Product::Mis)
        .map(|p| p.quantity_units);
    assert_eq!(held, Some(100.0), "the venue has not filled anything");
}
