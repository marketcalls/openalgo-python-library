#![cfg(all(feature = "costs", feature = "time"))]
//! The entry gates of a live session (`engine::venue_gates`, requirements R9, R16 and R17
//! of the session design): while entries are locked (`live_lock`), while another order of
//! the position is in flight (`live_in_flight`), and beyond the freeze quantity
//! (`freeze_qty`), an entry or add is rejected at submission; exits pass them all.
//!
//! "Holding 100" is 100 units of SBIN in MIS bought at 500.10 at the open of minute 1
//! (o1, `venue_support::holding_on`); the orders below are decided at the close of
//! minute 2. The bars rise 0.10 a minute from 500.00 (`venue_support::rising`).

mod venue_support;

use oa_bt::engine::{run, Engine, KernelError, RunSetup, Submission, VenueError};
use oa_bt::exec::{InstrumentSpec, OrderId, OrderRequest, OrderType, QuantitySpec, RejectCode};
use oa_bt::market::calendar::Calendar;
use oa_bt::model::{Product, Side};
use venue_support::{
    at_close, bars, config, exit, frozen_at, holding_on, limit, market, minute, order,
    pause_at_close, session, session_of, stop_market, wait_before, Act, Script,
};

/// A session of `instrument` paused at the close of minute 2, holding `held` units bought
/// in o1 (flat with no order for 0).
fn paused_holding_of(instrument: InstrumentSpec, held: f64) -> Engine {
    let kernel = session_of(config(), instrument);
    if held > 0.0 {
        return holding_on(kernel, held);
    }
    let mut kernel = kernel;
    pause_at_close(&mut kernel, &bars(&[], &[]), 2);
    kernel
}

fn paused_holding(held: f64) -> Engine {
    paused_holding_of(venue_support::sbin(), held)
}

/// `exit()`: a reduce-only MARKET sale of the whole MIS position.
fn whole_exit() -> OrderRequest {
    OrderRequest {
        quantity: QuantitySpec::PositionFraction(1.0),
        ..exit(Side::Sell, 1.0, OrderType::Market)
    }
}

/// The reject code of `kernel`'s submission of `request`; `None` when it passed.
fn code_of(kernel: &mut Engine, request: OrderRequest) -> Option<RejectCode> {
    match kernel.submit(request) {
        Ok(Submission::Submitted(_)) => None,
        Ok(Submission::Rejected(_, code)) => Some(code),
        other => panic!("expected a submission, got {other:?}"),
    }
}

fn reversal() -> OrderRequest {
    OrderRequest {
        reverse: true,
        ..market(Side::Sell, 160.0)
    }
}

#[test]
fn while_entries_are_locked_entries_and_adds_are_rejected_and_exits_are_not() {
    let locked = Some(RejectCode::LiveLock);
    let table = [
        ("a buy from flat", 0.0, market(Side::Buy, 10.0), locked),
        ("a sale from flat", 0.0, market(Side::Sell, 10.0), locked),
        ("an add", 100.0, market(Side::Buy, 10.0), locked),
        ("a reversal", 100.0, reversal(), locked),
        (
            "a plain sale within the long",
            100.0,
            market(Side::Sell, 40.0),
            None,
        ),
        (
            "a reduce-only stop",
            100.0,
            exit(Side::Sell, 100.0, stop_market(450.0)),
            None,
        ),
        ("exit()", 100.0, whole_exit(), None),
    ];
    for (what, held, request, code) in table {
        let mut kernel = paused_holding(held);
        assert_eq!(kernel.lock_entries("feed_dead"), Ok(true));
        assert_eq!(code_of(&mut kernel, request), code, "{what}");
        if code.is_some() {
            assert!(kernel.take_commands().is_empty(), "{what}: nothing is sent");
        }
    }
}

#[test]
fn entries_stay_locked_until_every_reason_is_lifted() {
    let mut kernel = paused_holding(0.0);
    assert_eq!(kernel.lock_entries("starting"), Ok(true));
    assert_eq!(kernel.lock_entries("feed_dead"), Ok(true));
    assert_eq!(kernel.unlock_entries("starting"), Ok(true));
    let still = code_of(&mut kernel, market(Side::Buy, 10.0));
    assert_eq!(still, Some(RejectCode::LiveLock));
    assert_eq!(kernel.unlock_entries("feed_dead"), Ok(true));
    assert_eq!(code_of(&mut kernel, market(Side::Buy, 10.0)), None);
}

#[test]
fn locking_and_unlocking_say_whether_the_set_of_reasons_changed() {
    let mut kernel = session(config());
    assert_eq!(kernel.lock_entries("mode_breach"), Ok(true));
    assert_eq!(kernel.lock_entries("mode_breach"), Ok(false));
    assert_eq!(kernel.lock_entries("feed_dead"), Ok(true));
    assert_eq!(kernel.entry_locks(), ["feed_dead", "mode_breach"]);
    assert_eq!(kernel.unlock_entries("starting"), Ok(false));
    assert_eq!(kernel.unlock_entries("feed_dead"), Ok(true));
    assert_eq!(kernel.entry_locks(), ["mode_breach"]);
}

#[test]
fn entries_locked_while_the_session_waits_are_rejected_at_the_next_decision() {
    let mut kernel = session(config());
    wait_before(
        &mut kernel,
        &bars(&[], &[]),
        &mut Script::new(Vec::new()),
        2,
    );
    assert_eq!(kernel.lock_entries("operator_stop"), Ok(true));
    pause_at_close(&mut kernel, &bars(&[], &[]), 2);
    let code = code_of(&mut kernel, market(Side::Buy, 10.0));
    assert_eq!(code, Some(RejectCode::LiveLock));
}

#[test]
fn a_lock_names_its_reason_with_no_spaces_around_it() {
    // "feed_dead " and "feed_dead" would be two locks, and the first would stand after
    // the second was lifted.
    let mut kernel = session(config());
    for blank in ["", "   ", "feed_dead ", " feed_dead"] {
        for result in [kernel.lock_entries(blank), kernel.unlock_entries(blank)] {
            assert!(
                matches!(
                    result,
                    Err(KernelError::InvalidConfig {
                        field: "entry_lock",
                        ..
                    })
                ),
                "{blank:?}"
            );
        }
    }
}

#[test]
fn a_backtest_has_no_entry_locks() {
    let mut simulated = venue_support::backtest(config(), bars(&[], &[]));
    let not_external = Err(VenueError::NotExternal.into());
    assert_eq!(simulated.lock_entries("feed_dead"), not_external);
    assert_eq!(simulated.unlock_entries("feed_dead"), not_external);
    assert!(simulated.entry_locks().is_empty());
}

#[test]
fn modifying_a_working_entry_while_entries_are_locked_is_rejected() {
    let mut kernel = paused_holding(0.0);
    let resting = order(Side::Buy, 10.0, limit(450.0), Product::Mis);
    assert_eq!(code_of(&mut kernel, resting), None);
    kernel.take_commands();
    assert_eq!(kernel.lock_entries("feed_dead"), Ok(true));
    let higher = order(Side::Buy, 10.0, limit(455.0), Product::Mis);
    let refused = kernel.modify(OrderId(1), higher);
    assert_eq!(
        refused,
        Ok(Submission::Rejected(OrderId(1), RejectCode::LiveLock))
    );
    assert!(kernel.take_commands().is_empty());
}

/// A session flat at the close of minute 1 whose MARKET buy of 10 (o1), decided at the
/// close of minute 0, the venue has not filled yet.
fn entry_not_filled_yet() -> Engine {
    let mut kernel = session(config());
    let acts = vec![at_close(0, Act::Submit(market(Side::Buy, 10.0)))];
    wait_before(&mut kernel, &bars(&[], &[]), &mut Script::new(acts), 1);
    pause_at_close(&mut kernel, &bars(&[], &[]), 1);
    kernel
}

#[test]
fn an_entry_or_add_is_rejected_while_another_order_of_its_position_works() {
    let in_flight = Some(RejectCode::LiveInFlight);
    let buy = || market(Side::Buy, 10.0);
    let resting_buy = || order(Side::Buy, 10.0, limit(450.0), Product::Mis);
    let target = || exit(Side::Sell, 100.0, limit(600.0));
    let stop = || exit(Side::Sell, 100.0, stop_market(450.0));
    let table: [(&str, Engine, Vec<OrderRequest>); 5] = [
        ("an entry not filled yet", entry_not_filled_yet(), vec![]),
        ("a resting entry", paused_holding(0.0), vec![resting_buy()]),
        ("a resting target", paused_holding(100.0), vec![target()]),
        (
            "the strategy's own stop",
            paused_holding(100.0),
            vec![stop()],
        ),
        (
            "an exit waiting for its cancels",
            paused_holding(100.0),
            vec![stop(), whole_exit()],
        ),
    ];
    for (what, mut kernel, working) in table {
        for request in working {
            assert_eq!(code_of(&mut kernel, request), None, "{what}");
        }
        assert_eq!(code_of(&mut kernel, buy()), in_flight, "{what}");
    }
}

#[test]
fn an_order_of_another_position_does_not_hold_an_entry_back() {
    let mut kernel = paused_holding(0.0);
    let delivery = order(Side::Buy, 10.0, limit(450.0), Product::Cnc);
    assert_eq!(code_of(&mut kernel, delivery), None);
    assert_eq!(code_of(&mut kernel, market(Side::Buy, 10.0)), None);
}

#[test]
fn an_add_is_allowed_again_once_the_entry_in_flight_has_filled() {
    let mut kernel = session(config());
    let acts = vec![at_close(0, Act::Submit(market(Side::Buy, 10.0)))];
    wait_before(&mut kernel, &bars(&[], &[]), &mut Script::new(acts), 1);
    assert!(kernel
        .apply_fill(OrderId(1), 10.0, 500.1, minute(1))
        .is_ok());
    pause_at_close(&mut kernel, &bars(&[], &[]), 1);
    assert_eq!(code_of(&mut kernel, market(Side::Buy, 10.0)), None);
}

#[test]
fn exits_pass_while_an_add_is_in_flight() {
    let mut kernel = paused_holding(100.0);
    let resting_add = order(Side::Buy, 10.0, limit(450.0), Product::Mis);
    assert_eq!(code_of(&mut kernel, resting_add), None);
    assert_eq!(code_of(&mut kernel, market(Side::Sell, 40.0)), None);
    assert_eq!(code_of(&mut kernel, whole_exit()), None);
}

#[test]
fn modifying_the_entry_in_flight_is_not_held_back_by_itself() {
    let mut kernel = paused_holding(0.0);
    let resting = order(Side::Buy, 10.0, limit(450.0), Product::Mis);
    assert_eq!(code_of(&mut kernel, resting), None);
    let higher = order(Side::Buy, 10.0, limit(455.0), Product::Mis);
    let modified = kernel.modify(OrderId(1), higher);
    assert_eq!(modified, Ok(Submission::Submitted(OrderId(1))));
}

#[test]
fn an_entry_or_add_may_not_take_the_position_above_the_freeze_quantity() {
    // A freeze quantity of 100.
    let frozen = Some(RejectCode::FreezeQty);
    let table = [
        ("an entry of the freeze quantity", 0.0, 100.0, None),
        ("an entry above it", 0.0, 101.0, frozen),
        ("an add up to it", 60.0, 40.0, None),
        ("an add beyond it", 60.0, 41.0, frozen),
        ("an add to a position at it", 100.0, 1.0, frozen),
    ];
    for (what, held, units, code) in table {
        let mut kernel = paused_holding_of(frozen_at(100), held);
        assert_eq!(
            code_of(&mut kernel, market(Side::Buy, units)),
            code,
            "{what}"
        );
    }
}

#[test]
fn an_exit_of_a_position_at_the_freeze_quantity_passes() {
    let mut kernel = paused_holding_of(frozen_at(100), 100.0);
    assert_eq!(code_of(&mut kernel, market(Side::Sell, 100.0)), None);
    let mut kernel = paused_holding_of(frozen_at(100), 100.0);
    assert_eq!(code_of(&mut kernel, whole_exit()), None);
}

#[test]
fn a_backtest_judges_the_freeze_quantity_of_each_order_alone() {
    // A freeze quantity of 100: two buys of 60 fill, a position of 120.
    let acts = vec![
        at_close(0, Act::Submit(market(Side::Buy, 60.0))),
        at_close(1, Act::Submit(market(Side::Buy, 60.0))),
    ];
    let setup = RunSetup {
        config: config(),
        calendar: Calendar::new().expect("the embedded calendar tables parse"),
        instruments: vec![frozen_at(100)],
        bars: vec![bars(&[], &[])],
    };
    let mut simulated = Engine::new(setup).expect("a valid backtest");
    run(&mut simulated, &mut Script::new(acts)).expect("the backtest runs");
    let output = simulated.finish().expect("the backtest finished");
    let codes: Vec<Option<RejectCode>> = output.orders.iter().map(|o| o.reject_code).collect();
    assert_eq!(codes[..2], [None, None]);
    let held: f64 = output
        .fills
        .iter()
        .take(2)
        .map(|record| record.fill.quantity_units)
        .sum();
    assert_eq!(held, 120.0);
}
