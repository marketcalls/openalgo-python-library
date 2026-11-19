#![cfg(all(feature = "costs", feature = "time"))]
//! What a session restored from its saved state does with what the saved session had not
//! finished (`Engine::restore_live_state`, `engine::venue_state`; 16-session-design 3.6):
//! the commands it had not handed out when it saved, the strategy's exit that waited for
//! its cancels, the entry locks and the sizing cap.
//!
//! Each saved session bought SBIN with an MIS MARKET order decided at the close of minute
//! 0 (o1) and stands waiting before minute 1, its commands not taken (3.6 saves before
//! each batch). Each restarted session has caught up to that wait and decided nothing.

mod venue_support;

use oa_bt::engine::{Engine, OrderRole, StateValue, VenueCommand};
use oa_bt::exec::{OrderId, OrderStatus, OrderType, RejectCode};
use oa_bt::model::{Product, Side};
use oa_bt::rms::ExitReason;
use venue_support::{
    at_close, bars, config, exit, holding_with, market, minute, pause_at_close, placed_roles,
    session, stop_points, wait_before, Act, Script, SBIN, SECOND,
};

/// What the restart learns of a restored order, given to the restored session before its
/// commands are taken.
type Learn = Box<dyn Fn(&mut Engine, OrderId)>;

/// A saved session whose buy of `units` (o1) is decided and its Place not taken.
fn entry_decided(units: f64) -> Engine {
    let mut kernel = session(config());
    let acts = vec![at_close(0, Act::Submit(market(Side::Buy, units)))];
    wait_before(&mut kernel, &bars(&[], &[]), &mut Script::new(acts), 1);
    kernel
}

/// A session restarted and caught up to the wait before minute 1.
fn restarted() -> Engine {
    let mut kernel = session(config());
    wait_before(
        &mut kernel,
        &bars(&[], &[]),
        &mut Script::new(Vec::new()),
        1,
    );
    kernel
}

/// `state` restored into a restarted session, with the new id of saved order `saved`.
fn restored(state: &StateValue, saved: u64) -> (Engine, OrderId) {
    let mut kernel = restarted();
    let pairs = kernel
        .restore_live_state(state)
        .expect("the state restores");
    let new = pairs
        .iter()
        .find(|(old, _)| *old == OrderId(saved))
        .map(|(_, new)| *new)
        .expect("the saved order is restored");
    (kernel, new)
}

/// The value of `key` in the map `value`.
fn get<'a>(value: &'a StateValue, key: &str) -> &'a StateValue {
    let StateValue::Map(entries) = value else {
        panic!("not a map: {value:?}");
    };
    let found = entries.iter().find(|(known, _)| known == key);
    &found.unwrap_or_else(|| panic!("no {key}")).1
}

/// The items of the list `value`.
fn items(value: &StateValue) -> &[StateValue] {
    let StateValue::List(items) = value else {
        panic!("not a list: {value:?}");
    };
    items
}

#[test]
fn a_restored_order_that_ends_before_its_place_is_taken_is_never_sent() {
    let reject: Learn = Box::new(|kernel, order| {
        kernel
            .apply_reject(order, RejectCode::Venue)
            .expect("not sent before the restart");
    });
    let fill: Learn = Box::new(|kernel, order| {
        kernel
            .apply_fill(order, 100.0, 500.1, minute(1))
            .expect("filled at the venue before the restart");
    });
    let cancel: Learn = Box::new(|kernel, order| {
        kernel
            .apply_cancel(order)
            .expect("the venue never placed it");
    });
    let replaced = vec![(OrderId(2), Side::Sell, 100.0, OrderRole::LegStop)];
    // (what, the saved session, the saved order, what the restart learns of it, the
    // orders the restored session then places)
    let table = [
        (
            "an entry not sent before the restart",
            entry_decided(100.0),
            1,
            reject,
            Vec::new(),
        ),
        (
            "an entry the venue filled whole before the restart",
            entry_decided(100.0),
            1,
            fill,
            Vec::new(),
        ),
        (
            "a leg's stop the venue never placed: placed again, once",
            holding_with(100.0, stop_points(2.0)),
            2,
            cancel,
            replaced,
        ),
    ];
    for (what, saved, saved_id, learn, placed) in table {
        let (mut kernel, order) = restored(&saved.live_state(), saved_id);
        learn(&mut kernel, order);
        let commands = kernel.take_commands();
        assert!(
            commands.iter().all(|command| command.order() != order),
            "{what}: nothing is sent for the ended {order}: {commands:?}"
        );
        assert_eq!(placed_roles(&commands), placed, "{what}");
    }
}

#[test]
fn a_restored_place_withdrawn_before_it_is_taken_is_cancelled_at_the_venue() {
    // The restored stop may have reached the venue before the restart: withdrawing it
    // sends Cancel, and the stop works until the venue ends it.
    let saved = holding_with(100.0, stop_points(2.0));
    let (mut kernel, stop) = restored(&saved.live_state(), 2);
    let at = minute(1) + 10 * SECOND;
    kernel
        .apply_external_fill(
            SBIN,
            Product::Mis,
            Side::Sell,
            100.0,
            499.0,
            at,
            ExitReason::Manual,
        )
        .expect("the broker sold the long");
    let commands = kernel.take_commands();
    assert!(
        commands.contains(&VenueCommand::Cancel { order: stop }),
        "{commands:?}"
    );
    let status = kernel.order(stop).map(|order| order.status);
    assert!(status.is_some_and(OrderStatus::is_working), "{status:?}");
}

#[test]
fn a_restored_session_keeps_its_own_entry_locks_and_takes_none_of_the_saved_ones() {
    // The saved locks are the earlier start's: which of them last the day is the live
    // layer's to say (16-session-design 6.2), so it reads them in the state.
    let mut first = holding_with(100.0, stop_points(2.0));
    first.lock_entries("operator_stop").expect("a lock reason");
    let state = first.live_state();
    let saved_locks = items(get(get(&state, "desk"), "entry_locks"));
    assert_eq!(saved_locks, [StateValue::from("operator_stop")]);
    let mut kernel = restarted();
    kernel.lock_entries("starting").expect("a lock reason");
    kernel
        .restore_live_state(&state)
        .expect("the state restores");
    assert_eq!(kernel.entry_locks(), ["starting"]);
}

#[test]
fn a_restored_session_keeps_the_sizing_cap_it_set_and_takes_the_saved_one_otherwise() {
    let mut first = holding_with(100.0, stop_points(2.0));
    first.cap_sizing_equity(900_000.0).expect("a cap in rupees");
    let state = first.live_state();
    let cap_of =
        |kernel: &Engine| get(get(&kernel.live_state(), "desk"), "sizing_cap_rupees").clone();
    let mut capped = restarted();
    capped
        .cap_sizing_equity(500_000.0)
        .expect("a cap in rupees");
    capped
        .restore_live_state(&state)
        .expect("the state restores");
    assert_eq!(
        cap_of(&capped),
        StateValue::from(500_000.0),
        "the fresher cap"
    );
    let mut uncapped = restarted();
    uncapped
        .restore_live_state(&state)
        .expect("the state restores");
    assert_eq!(cap_of(&uncapped), StateValue::from(900_000.0));
}

#[test]
fn a_strategy_exit_waiting_for_its_cancels_is_saved_as_not_sent() {
    let mut kernel = holding_with(100.0, stop_points(2.0));
    kernel.take_commands();
    pause_at_close(&mut kernel, &bars(&[], &[]), 2);
    kernel
        .submit(exit(Side::Sell, 100.0, OrderType::Market))
        .expect("the exit is accepted");
    assert_eq!(
        kernel.take_commands(),
        [VenueCommand::Cancel { order: OrderId(2) }],
        "the exit waits for the stop's cancel"
    );
    let state = kernel.live_state();
    let sent: Vec<(StateValue, StateValue)> = items(get(&state, "orders"))
        .iter()
        .map(|order| (get(order, "order").clone(), get(order, "sent").clone()))
        .collect();
    assert_eq!(
        sent,
        [
            (StateValue::from(2u64), StateValue::from(true)),
            (StateValue::from(3u64), StateValue::from(false)),
        ]
    );
}
