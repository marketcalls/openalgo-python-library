#![cfg(all(feature = "costs", feature = "time"))]
//! The stop of a live session where it stands (`engine::venue_session`, requirement R20 of
//! the session design): nothing is sent, the working orders and the open positions are
//! reported as they are, and `finish` can be called.
//!
//! "Holding 100" is 100 units of SBIN in MIS bought at 500.10 at the open of minute 1
//! (o1, `venue_support::holding_on`). The test day 2025-07-01 has 375 slices; its last is
//! slice 374. The bars rise 0.10 a minute from 500.00 (`venue_support::rising`).

mod venue_support;

use oa_bt::engine::{
    Engine, EngineExit, Event, KernelError, OrderEventKind, RiskExitReason, VenueCommand,
    VenueError,
};
use oa_bt::exec::{OrderId, OrderStatus};
use oa_bt::model::{Product, Side};
use venue_support::{
    at_close, backtest, bars, charged_config, config, exit, holding_on, limit, market, minute,
    pause_at_close, session, wait_before, Act, Script, SBIN,
};

/// The slice after the test day's last: where a session waits while its close does.
const AFTER_THE_DAY: usize = 375;

fn to_wait(kernel: &mut Engine, slice: usize) {
    wait_before(kernel, &bars(&[], &[]), &mut Script::new(Vec::new()), slice);
}

/// Holding 100 with a take-profit LIMIT sale of the 100 at 600.00 (o2) resting, paused at
/// the close of minute 2 with no command left to take.
fn holding_with_a_target() -> Engine {
    let mut kernel = holding_on(session(config()), 100.0);
    assert!(kernel.submit(exit(Side::Sell, 100.0, limit(600.0))).is_ok());
    kernel.take_commands();
    kernel
}

#[test]
fn a_stopped_session_sends_nothing_and_reports_its_working_orders_and_position_open() {
    let mut kernel = holding_with_a_target();
    to_wait(&mut kernel, 5);
    assert!(kernel.stop_session().is_ok());
    assert!(kernel.take_commands().is_empty(), "no cancel, no exit");
    assert!(kernel.is_finished());
    let output = kernel.finish().expect("a stopped session finishes");
    let statuses: Vec<OrderStatus> = output.orders.iter().map(|order| order.status).collect();
    assert_eq!(statuses, [OrderStatus::Filled, OrderStatus::Submitted]);
    let position = output.final_positions.first().expect("the position");
    let held = (position.product, position.quantity_units);
    assert_eq!(held, (Product::Mis, 100.0));
    assert_eq!(position.average_price_rupees, Some(500.1));
}

#[test]
fn a_session_can_be_stopped_wherever_it_stands() {
    // Waiting before a slice, paused at a bar close, and waiting at the session close for
    // the venue to end its orders (the target and the unfilled square-off).
    let mut waiting = holding_with_a_target();
    to_wait(&mut waiting, 5);
    let mut at_a_close = holding_with_a_target();
    pause_at_close(&mut at_a_close, &bars(&[], &[]), 3);
    let mut closing = holding_with_a_target();
    to_wait(&mut closing, AFTER_THE_DAY);
    for (what, mut kernel) in [
        ("waiting", waiting),
        ("paused at a bar close", at_a_close),
        ("at the session close", closing),
    ] {
        assert!(kernel.stop_session().is_ok(), "{what}");
        let output = kernel.finish().expect("a stopped session finishes");
        let units = output.final_positions.first().map(|p| p.quantity_units);
        assert_eq!(units, Some(100.0), "{what}");
    }
}

#[test]
fn a_stopped_session_takes_no_more_results_or_requests_and_runs_no_more_slices() {
    let mut kernel = holding_with_a_target();
    to_wait(&mut kernel, 5);
    assert!(kernel.stop_session().is_ok());
    let session_over = Err(VenueError::SessionOver.into());
    let fill = kernel.apply_fill(OrderId(2), 100.0, 600.0, minute(5));
    assert_eq!(fill.map(|_| ()), session_over);
    let stop = EngineExit::Risk(RiskExitReason::OperatorStop);
    assert_eq!(kernel.request_exit(SBIN, Product::Mis, stop), session_over);
    assert_eq!(kernel.cancel(OrderId(2)).map(|_| ()), session_over);
    assert_eq!(kernel.advance(), Err(KernelError::RunFinished));
}

#[test]
fn stopping_returns_the_events_not_delivered_yet() {
    // The fill of o1 while the session waits before slice 1 is delivered with the next
    // bar close, which never comes.
    let mut kernel = session(config());
    let acts = vec![at_close(0, Act::Submit(market(Side::Buy, 100.0)))];
    wait_before(&mut kernel, &bars(&[], &[]), &mut Script::new(acts), 1);
    assert!(kernel
        .apply_fill(OrderId(1), 100.0, 500.1, minute(1))
        .is_ok());
    let events = kernel.stop_session().expect("the session stops");
    let filled = events.iter().any(|event| {
        matches!(
            event,
            Event::Order {
                order: OrderId(1),
                kind: OrderEventKind::Filled { .. },
                ..
            }
        )
    });
    assert!(filled, "{events:?}");
}

#[test]
fn commands_not_taken_before_the_stop_are_left_for_the_live_layer() {
    let mut kernel = session(config());
    pause_at_close(&mut kernel, &bars(&[], &[]), 0);
    assert!(kernel.submit(market(Side::Buy, 10.0)).is_ok());
    assert!(kernel.stop_session().is_ok());
    let commands = kernel.take_commands();
    assert!(matches!(
        commands.as_slice(),
        [VenueCommand::Place {
            order: OrderId(1),
            ..
        }]
    ));
}

#[test]
fn stopping_a_stopped_session_changes_nothing() {
    let mut kernel = holding_with_a_target();
    assert!(kernel.stop_session().is_ok());
    assert_eq!(kernel.stop_session(), Ok(Vec::new()));
    assert!(kernel.finish().is_ok());
}

#[test]
fn a_backtest_is_not_stopped_as_a_session() {
    let mut simulated = backtest(config(), bars(&[], &[]));
    assert_eq!(
        simulated.stop_session(),
        Err(VenueError::NotExternal.into())
    );
}

#[test]
fn a_stopped_sessions_fills_add_up_to_its_charges() {
    // The day's contract note is settled at the stop, as at a run's end.
    let mut kernel = holding_on(session(charged_config()), 100.0);
    to_wait(&mut kernel, 5);
    assert!(kernel.stop_session().is_ok());
    let charges = kernel.account().costs_rupees;
    let output = kernel.finish().expect("a stopped session finishes");
    assert!(output.charges.notes.iter().all(|note| note.settled));
    assert_eq!(output.charges.notes.len(), 1);
    let paid: f64 = output.fills.iter().map(|r| r.fill.costs_rupees).sum();
    assert!(charges > 0.0);
    assert!((paid - charges).abs() < 0.005, "{paid} against {charges}");
}
