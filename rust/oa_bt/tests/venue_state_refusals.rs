#![cfg(all(feature = "costs", feature = "time"))]
//! What a live session's saved state is refused for (`Engine::restore_live_state`,
//! `LiveStateError`; 16-session-design R14): a session that cannot take a state now, a
//! state of another session, a state that cannot be read or whose parts do not hold
//! together, and a state saved while the session closed. A refused state is refused whole:
//! the session is left as it was, and the right state is restored afterwards.
//!
//! The saved state is a session holding 100 units of SBIN bought by o1 at 500.10 with a
//! stop 2 points below (its stop o2 working), paused at the close of minute 2.

mod venue_support;

use oa_bt::engine::{
    Engine, KernelError, LiveStateError, Pause, RunConfig, StateValue, Submission, VenueError,
};
use oa_bt::exec::OrderId;
use oa_bt::model::{days_from_civil, Product, Side};
use venue_support::{
    at_close, backtest, bars, config, holding_with, limit, market, nifty_call, order,
    pause_at_close, run_to_end, session, session_of, stop_points, wait_before, Act, Script,
    CAPITAL, SESSION_SLICES,
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
            LiveStateError::CarriedMismatch { .. } => "carried_mismatch".to_string(),
            LiveStateError::Version { found } => format!("version {found}"),
            LiveStateError::Malformed { path, .. } => format!("malformed at {path}"),
        },
        Err(other) => format!("{other:?}"),
    }
}

/// `value` with the value at `path` (map keys, and list indices as "0", from the root)
/// replaced by `new`.
fn with(value: &StateValue, path: &[&str], new: StateValue) -> StateValue {
    let replaced = |at: &str, item: &StateValue| match path {
        [first] if at == *first => new.clone(),
        [first, rest @ ..] if at == *first => with(item, rest, new.clone()),
        _ => item.clone(),
    };
    match value {
        StateValue::Map(entries) => StateValue::Map(
            entries
                .iter()
                .map(|(key, item)| (key.clone(), replaced(key, item)))
                .collect(),
        ),
        StateValue::List(items) => StateValue::List(
            items
                .iter()
                .enumerate()
                .map(|(index, item)| replaced(&index.to_string(), item))
                .collect(),
        ),
        other => panic!("not a map or a list at {path:?}: {other:?}"),
    }
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
            with(&state, &["layout"], 4i64.into()),
            "version 4",
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
fn a_state_that_does_not_hold_together_is_refused_naming_where() {
    let state = saved();
    // (what, the path changed, its new value, where the refusal points)
    let table: [(&str, &[&str], StateValue, &str); 7] = [
        (
            "a position its tranches do not add up to",
            &["positions", "0", "quantity_units"],
            80.0.into(),
            "live_state.positions[0]",
        ),
        (
            "a leg on the other side of its position",
            &["positions", "0", "quantity_units"],
            (-100.0).into(),
            "live_state.legs[0]",
        ),
        (
            "a leg holding other units than its position",
            &["legs", "0", "held_units"],
            50.0.into(),
            "live_state.legs[0]",
        ),
        (
            "an open position without its leg",
            &["legs"],
            StateValue::List(Vec::new()),
            "live_state.positions[0]",
        ),
        (
            "an order filled beyond its units",
            &["orders", "0", "filled_units"],
            500.0.into(),
            "live_state.orders[0]",
        ),
        (
            "a leg's stop that is not a leg stop order",
            &["orders", "0", "role"],
            "exit".into(),
            "live_state.desk.leg_stops[0]",
        ),
        (
            "an order saved as not sent that no exit waits with",
            &["orders", "0", "sent"],
            false.into(),
            "live_state.orders[0]",
        ),
    ];
    for (what, path, new, at) in table {
        let mut kernel = restarted_at(config(), 2);
        let refused = with(&state, path, new);
        let expected = format!("malformed at {at}");
        assert_eq!(
            kind(kernel.restore_live_state(&refused)),
            expected,
            "{what}"
        );
        assert!(
            kernel.working_orders().is_empty(),
            "{what}: nothing restored"
        );
    }
}

#[test]
fn a_state_saved_while_the_session_closed_is_refused_saying_so() {
    // A resting CNC LIMIT DAY order, placed at the venue, keeps the session close waiting
    // for the venue to end it; the state is saved there.
    let resting = order(Side::Buy, 10.0, limit(400.0), Product::Cnc);
    let mut first = session(config());
    let mut script = Script::new(vec![at_close(0, Act::Submit(resting))]);
    wait_before(&mut first, &bars(&[], &[]), &mut script, 1);
    assert_eq!(first.take_commands().len(), 1, "the order's Place");
    let close = SESSION_SLICES as usize;
    wait_before(&mut first, &bars(&[], &[]), &mut script, close);
    first.take_commands();
    let state = first.live_state();
    assert_eq!(
        state_field(&state, "stage"),
        &StateValue::from("closing"),
        "saved at the close"
    );
    let finished = {
        let mut kernel = session(config());
        run_to_end(&mut kernel, &bars(&[], &[]), &mut Script::new(Vec::new()));
        kernel
    };
    // (what, the session given the state)
    let table = [
        ("a session caught up to minute 2", restarted_at(config(), 2)),
        ("a session that has finished", finished),
    ];
    for (what, mut kernel) in table {
        let refused = kernel.restore_live_state(&state);
        let Err(KernelError::LiveState(LiveStateError::NotRestorable { reason })) = refused else {
            panic!("{what}: {refused:?}");
        };
        assert!(
            reason.contains("saved while the session closed"),
            "{what}: {reason}"
        );
    }
}

/// The top-level field `key` of a saved state.
fn state_field<'a>(state: &'a StateValue, key: &str) -> &'a StateValue {
    let StateValue::Map(entries) = state else {
        panic!("not a map");
    };
    let found = entries.iter().find(|(known, _)| known == key);
    &found.unwrap_or_else(|| panic!("no {key}")).1
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

/// The fields an open trade gained in layout 2: its excursion so far, its slippage so
/// far, the closing orders a price has reached that have not ended and the ticks of its
/// last slice.
const OPEN_TRADE_FIELDS_OF_LAYOUT_2: [&str; 6] = [
    "worst_rupees",
    "best_rupees",
    "inside_fill_slice",
    "slippage_rupees",
    "unconfirmed_close_orders",
    "slice_ticks",
];

/// The field an open trade gained in layout 3: the closing orders a price has reached
/// only by the terms of a modification the venue has not answered.
const OPEN_TRADE_FIELD_OF_LAYOUT_3: &str = "pending_close_orders";

/// The field an open trade gained in layout 4: the terms confirmed modifications replaced,
/// kept for the bar of the slice they were confirmed in.
const OPEN_TRADE_FIELD_OF_LAYOUT_4: &str = "replaced_terms";

/// The saved state as layout `layout` wrote it: its open trade without `fields`.
fn as_layout(layout: i64, fields: &[&str]) -> StateValue {
    let state = saved();
    let StateValue::Map(entries) = &state else {
        panic!("the state is a map");
    };
    let trades = entries.iter().find(|(key, _)| key == "open_trades");
    let Some((_, StateValue::List(trades))) = trades else {
        panic!("the state has its open trades");
    };
    let trade = fields
        .iter()
        .fold(trades[0].clone(), |trade, field| without(&trade, field));
    let earlier = with(&state, &["open_trades", "0"], trade);
    with(&earlier, &["layout"], layout.into())
}

#[test]
fn a_state_of_the_layout_before_the_excursions_is_refused_as_another_version() {
    let mut fields = OPEN_TRADE_FIELDS_OF_LAYOUT_2.to_vec();
    fields.extend([OPEN_TRADE_FIELD_OF_LAYOUT_3, OPEN_TRADE_FIELD_OF_LAYOUT_4]);
    let mut kernel = restarted_at(config(), 2);
    let earlier = as_layout(1, &fields);
    assert_eq!(kind(kernel.restore_live_state(&earlier)), "version 1");
}

#[test]
fn a_state_of_the_layout_before_the_holds_by_a_modification_is_refused_as_another_version() {
    let mut kernel = restarted_at(config(), 2);
    let fields = [OPEN_TRADE_FIELD_OF_LAYOUT_3, OPEN_TRADE_FIELD_OF_LAYOUT_4];
    let earlier = as_layout(2, &fields);
    assert_eq!(kind(kernel.restore_live_state(&earlier)), "version 2");
}

#[test]
fn a_state_of_the_layout_before_the_replaced_terms_is_refused_as_another_version() {
    let mut kernel = restarted_at(config(), 2);
    let earlier = as_layout(3, &[OPEN_TRADE_FIELD_OF_LAYOUT_4]);
    assert_eq!(kind(kernel.restore_live_state(&earlier)), "version 3");
}
