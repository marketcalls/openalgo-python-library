#![cfg(all(feature = "costs", feature = "time"))]
//! What a live session's saved state is refused for (`Engine::restore_live_state`,
//! `LiveStateError`; 16-session-design R14): a session that cannot take a state now, a
//! state of another session, and a state that cannot be read. A refused state is refused
//! whole: the session is left as it was, and the right state is restored afterwards.
//!
//! The saved state is a session holding 100 units of SBIN bought by o1 at 500.10 with a
//! stop 2 points below (its stop o2 working), paused at the close of minute 2.

mod venue_support;

use oa_bt::engine::{
    Engine, KernelError, LiveStateError, Pause, RunConfig, StateValue, Submission, VenueError,
};
use oa_bt::exec::OrderId;
use oa_bt::model::{days_from_civil, Side};
use venue_support::{
    backtest, bars, config, holding_with, market, nifty_call, pause_at_close, session, session_of,
    stop_points, wait_before, Script, CAPITAL,
};

/// How a table row builds the session it gives the state to.
type Build<'a> = Box<dyn Fn() -> Engine + 'a>;

/// The saved state (module docs).
fn saved() -> StateValue {
    let mut kernel = holding_with(100.0, stop_points(2.0));
    pause_at_close(&mut kernel, &bars(&[], &[]), 2);
    kernel.take_commands();
    kernel.live_state()
}

/// A session restarted with `config`, caught up to the close of minute `minute`.
fn restarted_at(config: RunConfig, minute: usize) -> Engine {
    let mut kernel = session(config);
    pause_at_close(&mut kernel, &bars(&[], &[]), minute);
    kernel
}

/// The kind of a restore's result, for the tables: "restored", or the refusal's kind.
fn kind(result: Result<Vec<(OrderId, OrderId)>, KernelError>) -> String {
    match result {
        Ok(_) => "restored".to_string(),
        Err(KernelError::Venue(VenueError::NotExternal)) => "not_external".to_string(),
        Err(KernelError::LiveState(error)) => match error {
            LiveStateError::NotRestorable { .. } => "not_restorable".to_string(),
            LiveStateError::OtherSession { .. } => "other_session".to_string(),
            LiveStateError::Version { found } => format!("version {found}"),
            LiveStateError::Malformed { path, .. } => format!("malformed at {path}"),
        },
        Err(other) => format!("{other:?}"),
    }
}

/// `value` with the value at `path` (map keys from the root) replaced by `new`.
fn with(value: &StateValue, path: &[&str], new: StateValue) -> StateValue {
    let StateValue::Map(entries) = value else {
        panic!("not a map at {path:?}");
    };
    let entries = entries
        .iter()
        .map(|(key, item)| match path {
            [first] if key == first => (key.clone(), new.clone()),
            [first, rest @ ..] if key == first => (key.clone(), with(item, rest, new.clone())),
            _ => (key.clone(), item.clone()),
        })
        .collect();
    StateValue::Map(entries)
}

/// `value` without the top-level field `key`.
fn without(value: &StateValue, key: &str) -> StateValue {
    let StateValue::Map(entries) = value else {
        panic!("not a map");
    };
    StateValue::Map(entries.iter().filter(|(k, _)| k != key).cloned().collect())
}

#[test]
fn a_session_that_cannot_take_a_state_now_refuses_it() {
    let state = saved();
    let restored_once = || {
        let mut kernel = restarted_at(config(), 2);
        assert_eq!(kind(kernel.restore_live_state(&state)), "restored");
        kernel
    };
    let with_an_order = || {
        let mut kernel = restarted_at(config(), 2);
        let entry = kernel.submit(market(Side::Buy, 10.0));
        assert_eq!(entry, Ok(Submission::Submitted(OrderId(1))));
        kernel
    };
    let finished = || {
        let mut kernel = session(RunConfig {
            last_slice: Some(venue_support::minute(3)),
            ..config()
        });
        loop {
            match kernel.advance().expect("the slice runs") {
                Pause::Waiting { next_slice } => kernel.release(next_slice + 1),
                Pause::Finished { .. } => return kernel,
                _ => {}
            }
        }
    };
    let backtest = || backtest(config(), bars(&[], &[]));
    // (what, the session given the state, the refusal)
    let table: [(&str, Build<'_>, &str); 5] = [
        ("a backtest", Box::new(backtest), "not_external"),
        (
            "a session restored already",
            Box::new(restored_once),
            "not_restorable",
        ),
        (
            "a session with an order of its own",
            Box::new(with_an_order),
            "not_restorable",
        ),
        (
            "a session before the saved point",
            Box::new(|| restarted_at(config(), 1)),
            "not_restorable",
        ),
        ("a finished session", Box::new(finished), "not_restorable"),
    ];
    for (what, build, refusal) in table {
        let mut kernel = build();
        assert_eq!(kind(kernel.restore_live_state(&state)), refusal, "{what}");
    }
}

#[test]
fn a_state_of_another_session_is_refused() {
    let state = saved();
    let next_day = days_from_civil(2025, 7, 2);
    let another_day = || {
        let mut kernel = session(RunConfig {
            first_day: next_day,
            last_day: next_day,
            ..config()
        });
        wait_before(
            &mut kernel,
            &bars(&[], &[]),
            &mut Script::new(Vec::new()),
            0,
        );
        kernel
    };
    let other_instruments = || {
        let mut kernel = session_of(config(), nifty_call());
        wait_before(
            &mut kernel,
            &bars(&[], &[]),
            &mut Script::new(Vec::new()),
            3,
        );
        kernel
    };
    let another_capital = || {
        let capital = RunConfig {
            capital_rupees: CAPITAL / 2.0,
            ..config()
        };
        restarted_at(capital, 2)
    };
    // (what, the session given the state)
    let table: [(&str, Build<'_>); 3] = [
        ("another trading day", Box::new(another_day)),
        ("other instruments", Box::new(other_instruments)),
        ("another capital", Box::new(another_capital)),
    ];
    for (what, build) in table {
        let mut kernel = build();
        assert_eq!(
            kind(kernel.restore_live_state(&state)),
            "other_session",
            "{what}"
        );
    }
}

#[test]
fn a_state_that_cannot_be_read_is_refused_whole_naming_the_field() {
    let state = saved();
    let order = |id: i64| StateValue::List(vec![StateValue::Int(id)]);
    // (what, the state given, the refusal)
    let table = [
        (
            "another layout",
            with(&state, &["layout"], 2i64.into()),
            "version 2",
        ),
        (
            "a part missing",
            without(&state, "risk"),
            "malformed at live_state",
        ),
        (
            "an order named that was not saved working",
            with(&state, &["desk", "cancelling"], order(1)),
            "malformed at live_state.desk.cancelling[0]",
        ),
        (
            "a leg that is not a map",
            with(&state, &["legs"], StateValue::List(vec![StateValue::Null])),
            "malformed at live_state.legs[0]",
        ),
        (
            "a number that is text",
            with(&state, &["clock_ns"], "now".into()),
            "malformed at live_state.clock_ns",
        ),
    ];
    for (what, refused, refusal) in table {
        let mut kernel = restarted_at(config(), 2);
        assert_eq!(kind(kernel.restore_live_state(&refused)), refusal, "{what}");
        assert!(
            kernel.working_orders().is_empty(),
            "{what}: nothing restored"
        );
        assert_eq!(
            kernel.position(venue_support::SBIN, oa_bt::model::Product::Mis),
            None
        );
        assert_eq!(
            kind(kernel.restore_live_state(&state)),
            "restored",
            "{what}"
        );
    }
}

#[test]
fn a_restored_session_keeps_the_entry_locks_it_set_and_adds_the_saved_ones() {
    let mut first = holding_with(100.0, stop_points(2.0));
    first.lock_entries("reconcile").expect("a lock reason");
    pause_at_close(&mut first, &bars(&[], &[]), 2);
    let state = first.live_state();
    let mut kernel = restarted_at(config(), 2);
    kernel.lock_entries("starting").expect("a lock reason");
    assert_eq!(kind(kernel.restore_live_state(&state)), "restored");
    assert_eq!(kernel.entry_locks(), ["reconcile", "starting"]);
}

#[test]
fn a_restored_working_order_takes_the_next_id_and_keeps_its_terms() {
    let state = saved();
    let mut kernel = restarted_at(config(), 2);
    let pairs = kernel
        .restore_live_state(&state)
        .expect("the state restores");
    assert_eq!(pairs, [(OrderId(2), OrderId(1))], "the stop o2 is o1 now");
    let stop = kernel.order(OrderId(1)).expect("the restored stop");
    assert_eq!(stop.quantity_units, 100.0);
    assert_eq!(stop.request.order_type.trigger_rupees(), Some(498.1));
    let held = kernel.position(venue_support::SBIN, oa_bt::model::Product::Mis);
    assert_eq!(held.map(|position| position.quantity_units), Some(100.0));
    let add = kernel.submit(market(Side::Buy, 10.0));
    assert_eq!(
        add,
        Ok(Submission::Submitted(OrderId(2))),
        "numbered after the stop"
    );
}
