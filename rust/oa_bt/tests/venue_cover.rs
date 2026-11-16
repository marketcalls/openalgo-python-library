#![cfg(all(feature = "costs", feature = "time"))]
//! Reduce-only orders never reverse a position in a live session (`engine::venue_cover`,
//! `venue_fills`): the reduce-only orders resting on a position's closing side fit in it
//! at submission and modification (`exit_cover`), are cut down or cancelled when it
//! shrinks, and a fill beyond it is recorded only up to it (`would_reverse`).
//!
//! Each test holds 100 units of SBIN in MIS, bought at 500.10 in minute 1, and decides at
//! the close of minute 2. The bars rise 0.10 a minute from 500.00
//! (`venue_support::rising`).

mod venue_support;

use oa_bt::engine::{Engine, KernelError, OrderRole, Submission, VenueCommand, VenueError};
use oa_bt::exec::{OrderId, OrderRequest, OrderStatus, OrderType, QuantitySpec, RejectCode};
use oa_bt::model::{Product, Side};
use venue_support::{
    at_close, bars, config, exit, frozen_at, limit, market, minute, order, pause_at_close, session,
    session_of, stop_market, wait_before, Act, Script, SBIN,
};

/// `kernel`, run on with no venue results until it waits before `slice`.
fn to_wait(kernel: &mut Engine, slice: usize) {
    wait_before(kernel, &bars(&[], &[]), &mut Script::new(Vec::new()), slice);
}

/// `kernel` holding 100 units bought in one order (o1), paused at the close of minute 2
/// with no command left to take.
fn holding_100_on(mut kernel: Engine) -> Engine {
    let acts = vec![at_close(0, Act::Submit(market(Side::Buy, 100.0)))];
    wait_before(&mut kernel, &bars(&[], &[]), &mut Script::new(acts), 1);
    kernel
        .apply_fill(OrderId(1), 100.0, 500.1, minute(1))
        .expect("the buy fills");
    pause_at_close(&mut kernel, &bars(&[], &[]), 2);
    kernel.take_commands();
    kernel
}

fn holding_100() -> Engine {
    holding_100_on(session(config()))
}

fn submit(kernel: &mut Engine, request: OrderRequest) -> Submission {
    kernel.submit(request).expect("a request the kernel reads")
}

/// A reduce-only SL-M sale at 450.00: a stop.
fn stop(units: f64) -> OrderRequest {
    exit(Side::Sell, units, stop_market(450.0))
}

/// A reduce-only LIMIT sale at 600.00: a target.
fn target(units: f64) -> OrderRequest {
    exit(Side::Sell, units, limit(600.0))
}

/// `exit()`: a reduce-only MARKET sale of the whole position.
fn whole_exit() -> OrderRequest {
    OrderRequest {
        quantity: QuantitySpec::PositionFraction(1.0),
        ..exit(Side::Sell, 1.0, OrderType::Market)
    }
}

fn held(kernel: &Engine) -> Option<f64> {
    kernel
        .position(SBIN, Product::Mis)
        .map(|position| position.quantity_units)
}

fn submitted(id: u64) -> Submission {
    Submission::Submitted(OrderId(id))
}

fn refused(id: u64, code: RejectCode) -> Submission {
    Submission::Rejected(OrderId(id), code)
}

#[test]
fn a_stop_for_the_whole_position_rests() {
    let mut kernel = holding_100();
    assert_eq!(submit(&mut kernel, stop(100.0)), submitted(2));
}

#[test]
fn a_stop_and_a_target_for_the_whole_position_cannot_both_rest() {
    let mut kernel = holding_100();
    assert_eq!(submit(&mut kernel, target(100.0)), submitted(2));
    assert_eq!(
        submit(&mut kernel, stop(100.0)),
        refused(3, RejectCode::ExitCover)
    );
}

#[test]
fn a_stop_and_a_target_for_half_the_position_each_rest() {
    let mut kernel = holding_100();
    assert_eq!(submit(&mut kernel, target(50.0)), submitted(2));
    assert_eq!(submit(&mut kernel, stop(50.0)), submitted(3));
}

#[test]
fn a_reduce_only_order_must_fit_beside_an_exit_already_sent() {
    let mut kernel = holding_100();
    assert_eq!(submit(&mut kernel, whole_exit()), submitted(2));
    assert_eq!(
        submit(&mut kernel, target(100.0)),
        refused(3, RejectCode::ExitCover)
    );
}

#[test]
fn a_reduce_only_order_placed_while_an_exit_waits_for_its_cancels_is_refused() {
    let mut kernel = holding_100();
    assert_eq!(submit(&mut kernel, target(50.0)), submitted(2));
    assert_eq!(submit(&mut kernel, whole_exit()), submitted(3));
    assert_eq!(
        kernel.take_commands(),
        vec![
            VenueCommand::Place {
                order: OrderId(2),
                instrument: SBIN,
                product: Product::Mis,
                side: Side::Sell,
                units: 50.0,
                order_type: limit(600.0),
                role: OrderRole::Exit,
            },
            VenueCommand::Cancel { order: OrderId(2) },
        ]
    );
    assert_eq!(
        submit(&mut kernel, stop(10.0)),
        refused(4, RejectCode::ExitCover)
    );
}

#[test]
fn a_sale_that_is_not_reduce_only_is_not_held_to_the_cover_rule() {
    let mut kernel = holding_100();
    assert_eq!(submit(&mut kernel, stop(100.0)), submitted(2));
    assert_eq!(submit(&mut kernel, market(Side::Sell, 40.0)), submitted(3));
}

/// Holding 100 with `resting` placed, then a plain MARKET sale of `sold` units (not
/// reduce-only) that the venue fills in minute 3; the commands of that fill are left.
fn shrunk_by(resting: &[OrderRequest], sold: f64) -> Engine {
    let mut kernel = holding_100();
    for request in resting {
        assert!(matches!(
            submit(&mut kernel, request.clone()),
            Submission::Submitted(_)
        ));
    }
    let sale = submit(&mut kernel, market(Side::Sell, sold));
    let Submission::Submitted(sale) = sale else {
        panic!("the sale is placed: {sale:?}");
    };
    to_wait(&mut kernel, 3);
    kernel.take_commands();
    kernel
        .apply_fill(sale, sold, 500.3, minute(3))
        .expect("the sale fills");
    kernel
}

#[test]
fn a_resting_stop_is_cut_down_to_what_is_left_when_the_position_shrinks() {
    let mut kernel = shrunk_by(&[stop(100.0)], 40.0);
    let cut = VenueCommand::Modify {
        order: OrderId(2),
        units: 60.0,
        order_type: stop_market(450.0),
    };
    assert_eq!(kernel.take_commands(), vec![cut]);
}

#[test]
fn the_reduce_only_orders_of_a_flat_position_are_cancelled() {
    let mut kernel = shrunk_by(&[target(100.0)], 100.0);
    let cancel = VenueCommand::Cancel { order: OrderId(2) };
    assert_eq!(kernel.take_commands(), vec![cancel]);
}

#[test]
fn the_later_of_two_reduce_only_orders_is_cut_first() {
    let mut kernel = shrunk_by(&[target(50.0), stop(50.0)], 40.0);
    let cut = VenueCommand::Modify {
        order: OrderId(3),
        units: 10.0,
        order_type: stop_market(450.0),
    };
    assert_eq!(kernel.take_commands(), vec![cut]);
}

#[test]
fn a_reduce_only_order_with_nothing_left_to_fit_is_cancelled() {
    let mut kernel = shrunk_by(&[target(50.0), stop(50.0)], 60.0);
    let commands = kernel.take_commands();
    assert_eq!(
        commands,
        vec![
            VenueCommand::Modify {
                order: OrderId(2),
                units: 40.0,
                order_type: limit(600.0),
            },
            VenueCommand::Cancel { order: OrderId(3) },
        ]
    );
}

#[test]
fn a_partly_filled_reduce_only_order_that_no_longer_fits_is_cancelled() {
    let mut kernel = holding_100();
    assert_eq!(submit(&mut kernel, stop(100.0)), submitted(2));
    assert_eq!(submit(&mut kernel, market(Side::Sell, 20.0)), submitted(3));
    to_wait(&mut kernel, 3);
    kernel.take_commands();
    let stopped = minute(3) + 1;
    assert!(kernel.apply_fill(OrderId(2), 30.0, 450.0, stopped).is_ok());
    assert!(kernel.take_commands().is_empty(), "70 left of 70 still fit");
    assert!(kernel.apply_fill(OrderId(3), 20.0, 450.0, stopped).is_ok());
    let cancel = VenueCommand::Cancel { order: OrderId(2) };
    assert_eq!(kernel.take_commands(), vec![cancel]);
}

#[test]
fn a_cut_waits_for_the_answer_to_a_modification_already_sent() {
    let mut kernel = holding_100();
    assert_eq!(submit(&mut kernel, stop(100.0)), submitted(2));
    assert_eq!(submit(&mut kernel, market(Side::Sell, 40.0)), submitted(3));
    let lower = exit(Side::Sell, 100.0, stop_market(445.0));
    assert_eq!(kernel.modify(OrderId(2), lower), Ok(submitted(2)));
    to_wait(&mut kernel, 3);
    kernel.take_commands();
    assert!(kernel
        .apply_fill(OrderId(3), 40.0, 500.3, minute(3))
        .is_ok());
    assert!(kernel.take_commands().is_empty());
    assert!(kernel.apply_modify(OrderId(2)).is_ok());
    let cut = VenueCommand::Modify {
        order: OrderId(2),
        units: 60.0,
        order_type: stop_market(445.0),
    };
    assert_eq!(kernel.take_commands(), vec![cut]);
}

#[test]
fn a_reduce_only_fill_beyond_the_position_is_recorded_up_to_it() {
    // The stop of 100 is cut to 60, but the broker fills all 100 before the cut lands.
    let mut kernel = shrunk_by(&[stop(100.0)], 40.0);
    let result = kernel.apply_fill(OrderId(2), 100.0, 450.0, minute(3) + 1);
    let refusal = VenueError::WouldReverse {
        order: OrderId(2),
        excess_units: 40.0,
    };
    assert_eq!(result, Err(KernelError::Venue(refusal)));
    assert_eq!(held(&kernel), Some(0.0));
    let stopped = kernel.order(OrderId(2)).expect("the stop");
    assert_eq!(
        (stopped.status, stopped.filled_units, stopped.quantity_units),
        (OrderStatus::Filled, 60.0, 60.0)
    );
}

#[test]
fn a_reduce_only_fill_of_a_flat_position_records_nothing_and_ends_the_order() {
    let mut kernel = shrunk_by(&[target(100.0)], 100.0);
    let result = kernel.apply_fill(OrderId(2), 100.0, 600.0, minute(3) + 1);
    let refusal = VenueError::WouldReverse {
        order: OrderId(2),
        excess_units: 100.0,
    };
    assert_eq!(result, Err(KernelError::Venue(refusal)));
    assert_eq!(held(&kernel), Some(0.0));
    let target = kernel.order(OrderId(2)).expect("the target");
    assert_eq!(
        (target.status, target.filled_units),
        (OrderStatus::Canceled, 0.0)
    );
}

#[test]
fn a_fill_beyond_the_position_of_an_order_without_reverse_is_recorded_up_to_it() {
    // A plain LIMIT sale of 100 (o2) rests when a sale of 40 fills first.
    let resting = order(Side::Sell, 100.0, limit(600.0), Product::Mis);
    let mut kernel = shrunk_by(&[resting], 40.0);
    let result = kernel.apply_fill(OrderId(2), 100.0, 600.0, minute(3) + 1);
    let refusal = VenueError::WouldReverse {
        order: OrderId(2),
        excess_units: 40.0,
    };
    assert_eq!(result, Err(KernelError::Venue(refusal)));
    assert_eq!(held(&kernel), Some(0.0));
}

#[test]
fn modifying_an_exit_counts_the_other_exits_of_its_position() {
    let mut kernel = holding_100();
    let reduce = |units| exit(Side::Sell, units, OrderType::Market);
    assert_eq!(submit(&mut kernel, reduce(50.0)), submitted(2));
    assert_eq!(submit(&mut kernel, reduce(10.0)), submitted(3));
    assert_eq!(
        kernel.modify(OrderId(2), reduce(100.0)),
        Ok(refused(2, RejectCode::ExitCover))
    );
    assert_eq!(kernel.modify(OrderId(2), reduce(90.0)), Ok(submitted(2)));
}

#[test]
fn modifying_an_exit_of_a_position_at_the_freeze_quantity_is_held_to_the_position() {
    // A freeze quantity of 100: the position may not go above it (`venue_gates`), so
    // exit() of the 100 is one order (o2), and a modification asking for 160 replaces it
    // with the 100 held, within the freeze quantity.
    let mut kernel = holding_100_on(session_of(config(), frozen_at(100)));
    assert_eq!(submit(&mut kernel, whole_exit()), submitted(2));
    assert_eq!(kernel.take_commands().len(), 1);
    let larger = exit(Side::Sell, 160.0, OrderType::Market);
    assert_eq!(kernel.modify(OrderId(2), larger), Ok(submitted(2)));
    let held_to_it = VenueCommand::Modify {
        order: OrderId(2),
        units: 100.0,
        order_type: OrderType::Market,
    };
    assert_eq!(kernel.take_commands(), vec![held_to_it]);
}
