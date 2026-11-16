#![cfg(all(feature = "costs", feature = "time"))]
//! The venue's answers to what a live session sent it (`engine::venue_answers`,
//! `venue_orders`): a modification the venue filled before confirming it, and a
//! cancellation it refused, which goes again at most once in a slice.
//!
//! The test day 2025-07-01 has 375 slices of one minute; its last is slice 374. The bars
//! rise 0.10 a minute from 500.00 (`venue_support::rising`).

mod venue_support;

use oa_bt::engine::{Engine, Pause, Submission, VenueCommand};
use oa_bt::exec::{OrderId, OrderRequest, OrderStatus, OrderType, QuantitySpec};
use oa_bt::model::{Product, Side};
use venue_support::{
    at_close, bars, config, exit, limit, market, minute, order, pause_at_close, session,
    wait_before, Act, Script,
};

/// The slice after the test day's last: where a session waits while its close does.
const AFTER_THE_DAY: usize = 375;

/// `kernel`, run on with no venue results until it waits before `slice`.
fn to_wait(kernel: &mut Engine, slice: usize) {
    wait_before(kernel, &bars(&[], &[]), &mut Script::new(Vec::new()), slice);
}

fn buy_limit(units: f64, at: f64) -> OrderRequest {
    order(Side::Buy, units, limit(at), Product::Mis)
}

/// A LIMIT buy of 100 at 450.00 (o1) decided at the close of minute 0, modified to 30 at
/// 455.00 at the close of minute 1; the venue fills 30 of it in minute 2 and then
/// confirms the modification.
fn filled_then_cut_to_30() -> Engine {
    let mut kernel = session(config());
    let acts = vec![at_close(0, Act::Submit(buy_limit(100.0, 450.0)))];
    wait_before(&mut kernel, &bars(&[], &[]), &mut Script::new(acts), 1);
    pause_at_close(&mut kernel, &bars(&[], &[]), 1);
    assert!(kernel.modify(OrderId(1), buy_limit(30.0, 455.0)).is_ok());
    to_wait(&mut kernel, 2);
    kernel.take_commands();
    let filled = minute(2) + 1;
    assert!(kernel.apply_fill(OrderId(1), 30.0, 450.0, filled).is_ok());
    assert!(kernel.apply_modify(OrderId(1)).is_ok());
    kernel
}

#[test]
fn a_modification_cut_to_the_units_filled_completes_the_order() {
    let kernel = filled_then_cut_to_30();
    let order = kernel.order(OrderId(1)).expect("the order");
    assert_eq!(
        (order.status, order.filled_units, order.quantity_units),
        (OrderStatus::Filled, 30.0, 30.0)
    );
    assert!(kernel.working_orders().is_empty());
}

#[test]
fn the_session_close_does_not_wait_for_an_order_a_modification_completed() {
    // The square-off of the 30 (o2) goes out at 15:10 and is never filled: it is the
    // only order the close cancels and waits for.
    let mut kernel = filled_then_cut_to_30();
    to_wait(&mut kernel, AFTER_THE_DAY);
    let cancelled: Vec<OrderId> = kernel
        .take_commands()
        .into_iter()
        .filter_map(|command| match command {
            VenueCommand::Cancel { order } => Some(order),
            _ => None,
        })
        .collect();
    assert_eq!(cancelled, [OrderId(2)]);
    assert!(kernel.apply_cancel(OrderId(2)).is_ok());
    assert!(matches!(kernel.advance(), Ok(Pause::Finished { .. })));
}

/// Holding 100 in MIS with a take-profit LIMIT sale of 100 at 600.00 (o2) and two
/// resting CNC LIMIT buys of 10 at 400.00 (o3, o4, never reached), `exit()` (o5) decided
/// at the close of minute 3 sends a cancellation of o2; the venue refuses it while the
/// session waits before slice 4.
fn refused_behind_a_take_profit() -> Engine {
    let mut kernel = session(config());
    let mut script = Script::new(vec![at_close(0, Act::Submit(market(Side::Buy, 100.0)))]);
    wait_before(&mut kernel, &bars(&[], &[]), &mut script, 1);
    assert!(kernel
        .apply_fill(OrderId(1), 100.0, 500.1, minute(1))
        .is_ok());
    pause_at_close(&mut kernel, &bars(&[], &[]), 3);
    assert!(kernel.submit(exit(Side::Sell, 100.0, limit(600.0))).is_ok());
    let resting = order(Side::Buy, 10.0, limit(400.0), Product::Cnc);
    for _ in 0..2 {
        let placed = kernel.submit(resting.clone());
        assert!(matches!(placed, Ok(Submission::Submitted(_))));
    }
    let whole_exit = OrderRequest {
        quantity: QuantitySpec::PositionFraction(1.0),
        ..exit(Side::Sell, 1.0, OrderType::Market)
    };
    assert!(kernel.submit(whole_exit).is_ok());
    to_wait(&mut kernel, 4);
    kernel.take_commands();
    assert!(kernel.apply_cancel_refused(OrderId(2)).is_ok());
    kernel
}

/// How many cancellations of order `id` the session has made since the last take.
fn cancels_of(kernel: &mut Engine, id: u64) -> usize {
    let cancel = VenueCommand::Cancel { order: OrderId(id) };
    kernel
        .take_commands()
        .iter()
        .filter(|command| **command == cancel)
        .count()
}

#[test]
fn a_refused_cancellation_is_sent_at_most_once_in_the_slice_it_was_refused_in() {
    // Sent in slice 3 (the decision at its close), refused in slice 4: the next result of
    // slice 4 sends it again; refused again, no later result of the slice does.
    let mut kernel = refused_behind_a_take_profit();
    assert_eq!(
        cancels_of(&mut kernel, 2),
        0,
        "a refusal sends nothing itself"
    );
    assert!(kernel.apply_cancel(OrderId(3)).is_ok());
    assert_eq!(cancels_of(&mut kernel, 2), 1);
    assert!(kernel.apply_cancel_refused(OrderId(2)).is_ok());
    assert!(kernel.apply_cancel(OrderId(4)).is_ok());
    assert_eq!(cancels_of(&mut kernel, 2), 0);
}

#[test]
fn a_refused_cancellation_goes_again_in_the_next_slice() {
    let mut kernel = refused_behind_a_take_profit();
    to_wait(&mut kernel, 5);
    assert_eq!(cancels_of(&mut kernel, 2), 1);
}
