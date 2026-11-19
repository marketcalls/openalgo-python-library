#![cfg(all(feature = "costs", feature = "time"))]
//! Reduce-only orders never reverse a position in a live session (`engine::venue_cover`):
//! with the other orders working on a position's closing side they fit in it at submission
//! and modification (`exit_cover`). The session's own fills keep them within it; how they
//! are cut down or cancelled when the position shrinks under them otherwise (at the
//! broker), and how a fill beyond it is recorded only up to it (`would_reverse`), is
//! specified in the crate (`engine/venue_cover/shrink_tests.rs`), which can make such a
//! position.
//!
//! Each test holds 100 units of SBIN in MIS, bought at 500.10 in minute 1, and decides at
//! the close of minute 2. The bars rise 0.10 a minute from 500.00
//! (`venue_support::rising`).

mod venue_support;

use oa_bt::engine::{Engine, OrderRole, Submission, VenueCommand};
use oa_bt::exec::{OrderId, OrderRequest, OrderType, QuantitySpec, RejectCode};
use oa_bt::model::{Product, Side};
use venue_support::{
    at_close, bars, config, exit, frozen_at, limit, market, minute, pause_at_close, session,
    session_of, stop_market, wait_before, Act, Script, SBIN,
};

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
fn a_sale_that_is_not_reduce_only_counts_against_the_room_a_stop_leaves() {
    // Beside a stop of the whole long a plain sale is an entry, held back while the stop
    // works (`venue_roles`); beside a stop of 60 a sale of 40 is an exit.
    let mut kernel = holding_100();
    assert_eq!(submit(&mut kernel, stop(100.0)), submitted(2));
    assert_eq!(
        submit(&mut kernel, market(Side::Sell, 40.0)),
        refused(3, RejectCode::LiveInFlight)
    );
    let mut kernel = holding_100();
    assert_eq!(submit(&mut kernel, stop(60.0)), submitted(2));
    assert_eq!(submit(&mut kernel, market(Side::Sell, 40.0)), submitted(3));
    assert_eq!(kernel.order_role(OrderId(3)), Some(OrderRole::Exit));
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
