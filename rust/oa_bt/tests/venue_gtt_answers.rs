#![cfg(all(feature = "costs", feature = "time"))]
//! The venue's answers about a live session's GTTs, and a fired GTT's child followed as
//! an order (design 20-positional-live 3.1 to 3.5, unit N1 item N1-2; 04-execution 4.4
//! item 3): placed, refused, modified or not, cancelled or not, ended, triggered; the
//! one-base-bar timer of an unfilled child and the engine's exit after it; the refusals
//! of an answer that does not fit; and the events the strategy sees, against a backtest
//! of the same path.
//!
//! SBIN on 2025-07-01, 1-minute bars, no charges, no slippage; 100 bought in CNC at
//! 500.10 at the 09:16 open with a stop of 4 points (trigger 496.10, child limit 491.10:
//! `gtt_venue_support`).

mod gtt_venue_support;
mod venue_support;

use gtt_venue_support::{
    at, cnc_buy, cnc_long, cnc_long_protected, gtt_commands, long_stop, stop_4, stop_and_target,
    stop_and_trail, BUY, ENTRY, G1, UNITS,
};
use oa_bt::engine::{
    Engine, GttEnd, KernelError, OrderRole, RiskExitReason, RunOutput, VenueCommand, VenueError,
    GTT_REFUSED_LOCK,
};
use oa_bt::exec::{
    FillFlags, GttId, GttStatus, OrderId, OrderRequest, OrderStatus, OrderType, RejectCode,
    Validity,
};
use oa_bt::model::{Product, Side};
use venue_support::{
    at_close, bars, config, events, limit, minute, order, placed_roles, run_backtest, run_to_end,
    session, wait_before, Act, Ohlc, Script, SBIN,
};

const G2: GttId = GttId(2);
const CHILD: OrderId = OrderId(2);

fn status_of(kernel: &Engine, gtt: GttId) -> Option<GttStatus> {
    kernel.gtt_view(gtt).map(|view| view.gtt.status)
}

fn held(kernel: &Engine) -> Option<f64> {
    kernel
        .position(SBIN, Product::Cnc)
        .map(|position| position.quantity_units)
}

fn refusal_kind(result: Result<impl std::fmt::Debug, KernelError>) -> &'static str {
    match result {
        Err(KernelError::Venue(refusal)) => refusal.kind(),
        other => panic!("not a venue refusal: {other:?}"),
    }
}

// ---------------------------------------------------------------- placed, refused

#[test]
fn a_confirmed_place_puts_the_stop_gtt_in_force_with_its_event() {
    let mut kernel = cnc_long(stop_4());
    kernel.take_commands();
    assert!(kernel.apply_gtt_placed(G1).is_ok());
    assert_eq!(status_of(&kernel, G1), Some(GttStatus::Active));
    let events = kernel.stop_session().expect("a session stops");
    let placed = events
        .iter()
        .filter(|e| matches!(e, oa_bt::engine::Event::Gtt { .. }));
    assert_eq!(placed.count(), 1);
}

#[test]
fn a_refused_stop_gtt_exits_the_leg_at_market_and_locks_entries() {
    // Decision 11.7: the leg is exited at once, reason stop_rejected, and the strategy's
    // entries are locked for the session.
    let mut kernel = cnc_long(stop_4());
    kernel.take_commands();
    assert!(kernel.apply_gtt_refused(G1).is_ok());
    assert_eq!(status_of(&kernel, G1), Some(GttStatus::Rejected));
    let exit = (
        OrderId(2),
        Side::Sell,
        UNITS,
        OrderRole::RiskExit(RiskExitReason::StopRejected),
    );
    assert_eq!(placed_roles(&kernel.take_commands()), [exit]);
    assert_eq!(kernel.entry_locks(), [GTT_REFUSED_LOCK]);
}

/// A session with a resting GTT buy of 10 at 490.00 (o1, g1), its place not answered.
fn resting_gtt_entry() -> Engine {
    let mut kernel = session(config());
    let resting = OrderRequest {
        validity: Validity::Gtt,
        ..order(Side::Buy, 10.0, limit(490.0), Product::Cnc)
    };
    let acts = vec![at_close(0, Act::Submit(resting))];
    wait_before(&mut kernel, &bars(&[], &[]), &mut Script::new(acts), 1);
    kernel.take_commands();
    kernel
}

#[test]
fn a_refused_strategy_gtt_order_ends_rejected() {
    let mut kernel = resting_gtt_entry();
    assert!(kernel.apply_gtt_refused(G1).is_ok());
    let order = kernel.order(BUY).expect("o1 is the session's");
    assert_eq!(
        (order.status, order.reject_code),
        (OrderStatus::Rejected, Some(RejectCode::Venue))
    );
}

#[test]
fn a_strategy_gtt_order_takes_no_fill_before_the_venue_fires_it() {
    let mut kernel = resting_gtt_entry();
    assert!(kernel.apply_gtt_placed(G1).is_ok());
    let fill = kernel.apply_fill(BUY, 10.0, 490.0, at(1, 5));
    assert_eq!(refusal_kind(fill), "gtt_resting");
    assert_eq!(kernel.apply_gtt_triggered(G1, at(1, 5)), Ok(BUY));
    assert!(kernel.apply_fill(BUY, 10.0, 490.0, at(1, 6)).is_ok());
    assert_eq!(status_of(&kernel, G1), Some(GttStatus::ChildFilled));
    assert_eq!(held(&kernel), Some(10.0));
}

#[test]
fn a_strategy_cancels_its_resting_gtt_order_through_the_gtt() {
    let mut kernel = resting_gtt_entry();
    assert!(kernel.apply_gtt_placed(G1).is_ok());
    assert!(kernel.cancel(BUY).is_ok());
    let cancel = gtt_commands(&kernel.take_commands());
    assert!(matches!(
        cancel[..],
        [VenueCommand::CancelGtt { gtt: G1, .. }]
    ));
    assert!(kernel.apply_gtt_cancelled(G1).is_ok());
    let order = kernel.order(BUY).expect("o1 is the session's");
    assert_eq!(order.status, OrderStatus::Canceled);
}

// ---------------------------------------------------------------- modified or not

#[test]
fn a_refused_modification_keeps_the_old_terms_and_goes_again_a_minute_later() {
    // The trail moves the stop to 501.00; the venue refuses: 496.10 stands. Thirty
    // seconds on nothing is sent; a minute after the refused one, 501.00 again.
    let mut kernel = cnc_long_protected(stop_and_trail());
    assert!(kernel.push_tick(SBIN, 503.0, at(1, 10)).is_ok());
    kernel.take_commands();
    assert!(kernel.apply_gtt_modify_refused(G1).is_ok());
    let view = kernel.gtt_view(G1).expect("g1 is the session's");
    assert_eq!(
        (view.gtt.status, view.terms, view.pending),
        (GttStatus::Active, long_stop(UNITS, 496.1, 491.1), None)
    );
    assert!(kernel.push_tick(SBIN, 503.0, at(1, 40)).is_ok());
    assert!(kernel.take_commands().is_empty());
    assert!(kernel.push_tick(SBIN, 503.0, at(2, 10)).is_ok());
    let again = VenueCommand::ModifyGtt {
        gtt: G1,
        terms: long_stop(UNITS, 501.0, 495.95),
    };
    assert_eq!(kernel.take_commands(), [again]);
}

#[test]
fn a_confirmed_modification_rests_on_the_terms_sent() {
    let mut kernel = cnc_long_protected(stop_and_trail());
    assert!(kernel.push_tick(SBIN, 503.0, at(1, 10)).is_ok());
    kernel.take_commands();
    let pending = kernel.gtt_view(G1).and_then(|view| view.pending);
    assert_eq!(pending, Some(long_stop(UNITS, 501.0, 495.95)));
    assert!(kernel.apply_gtt_modified(G1).is_ok());
    let view = kernel.gtt_view(G1).expect("g1 is the session's");
    assert_eq!(
        (view.terms, view.pending),
        (long_stop(UNITS, 501.0, 495.95), None)
    );
}

// ---------------------------------------------------------------- cancelled or not

#[test]
fn a_stop_gtt_cancelled_outside_the_session_is_placed_again() {
    // Not asked for: the GTT ended outside the session; the leg gets g2 at its stop.
    let mut kernel = cnc_long_protected(stop_4());
    assert!(kernel.apply_gtt_cancelled(G1).is_ok());
    assert_eq!(status_of(&kernel, G1), Some(GttStatus::VenueEnded));
    let again = VenueCommand::PlaceGtt {
        gtt: G2,
        order: None,
        terms: long_stop(UNITS, 496.1, 491.1),
    };
    assert_eq!(kernel.take_commands(), [again]);
}

#[test]
fn a_gtt_ending_while_its_modification_waits_is_placed_again_at_the_legs_stop() {
    // The trail asked for 501.00 and g1 ended outside the session before the answer:
    // nothing of g1 waits any more, and g2 goes out at the leg's stop, 501.00.
    let mut kernel = cnc_long_protected(stop_and_trail());
    assert!(kernel.push_tick(SBIN, 503.0, at(1, 10)).is_ok());
    kernel.take_commands();
    assert!(kernel.apply_gtt_cancelled(G1).is_ok());
    let view = kernel.gtt_view(G1).expect("g1 is the session's");
    assert_eq!(
        (view.gtt.status, view.pending),
        (GttStatus::VenueEnded, None)
    );
    let again = VenueCommand::PlaceGtt {
        gtt: G2,
        order: None,
        terms: long_stop(UNITS, 501.0, 495.95),
    };
    assert_eq!(kernel.take_commands(), [again]);
}

#[test]
fn a_refused_cancellation_works_on_and_goes_again_in_a_later_slice() {
    // The target's exit waits for g1's cancellation; the venue refuses it in minute 1:
    // g1 rests, nothing is sent again in minute 1, and the cancellation goes once the
    // session reaches minute 2.
    let mut kernel = cnc_long_protected(stop_and_target());
    assert!(kernel.push_tick(SBIN, 506.1, at(1, 20)).is_ok());
    kernel.take_commands();
    assert!(kernel.apply_gtt_cancel_refused(G1).is_ok());
    assert_eq!(status_of(&kernel, G1), Some(GttStatus::Active));
    assert!(kernel.push_tick(SBIN, 506.2, at(1, 40)).is_ok());
    assert!(kernel.take_commands().is_empty());
    wait_before(
        &mut kernel,
        &bars(&[], &[]),
        &mut Script::new(Vec::new()),
        2,
    );
    let cancel = gtt_commands(&kernel.take_commands());
    assert!(matches!(
        cancel[..],
        [VenueCommand::CancelGtt { gtt: G1, .. }]
    ));
}

// ---------------------------------------------------------------- ended

#[test]
fn an_expired_stop_gtt_is_placed_again() {
    let mut kernel = cnc_long_protected(stop_4());
    assert!(kernel.apply_gtt_ended(G1, GttEnd::Expired).is_ok());
    let again = gtt_commands(&kernel.take_commands());
    assert!(matches!(
        again[..],
        [VenueCommand::PlaceGtt { gtt: G2, .. }]
    ));
}

#[test]
fn a_leg_whose_stop_gtt_is_rejected_a_second_time_is_exited_at_market() {
    let mut kernel = cnc_long_protected(stop_4());
    assert!(kernel.apply_gtt_ended(G1, GttEnd::Rejected).is_ok());
    let again = gtt_commands(&kernel.take_commands());
    assert!(matches!(
        again[..],
        [VenueCommand::PlaceGtt { gtt: G2, .. }]
    ));
    assert!(kernel.apply_gtt_placed(G2).is_ok());
    assert!(kernel.apply_gtt_ended(G2, GttEnd::Rejected).is_ok());
    let exit = (
        OrderId(2),
        Side::Sell,
        UNITS,
        OrderRole::RiskExit(RiskExitReason::StopRejected),
    );
    assert_eq!(placed_roles(&kernel.take_commands()), [exit]);
}

// ---------------------------------------------------------------- triggered and the child

#[test]
fn a_fired_stop_gtt_is_followed_as_its_child_order() {
    // The child is a LIMIT sale of the GTT's 100 at its limit 491.10, the leg's stop, and
    // it fills at 496.10: the position closes for the stop's reason, with the GTT's id.
    let mut kernel = cnc_long_protected(stop_4());
    assert_eq!(kernel.apply_gtt_triggered(G1, at(1, 30)), Ok(CHILD));
    let child = kernel
        .order(CHILD)
        .expect("the child is the session's")
        .clone();
    assert_eq!(
        (
            child.request.side,
            child.quantity_units,
            child.request.order_type
        ),
        (
            Side::Sell,
            UNITS,
            OrderType::Limit {
                limit_rupees: 491.1
            }
        )
    );
    assert_eq!(kernel.order_role(CHILD), Some(OrderRole::LegStop));
    assert_eq!(kernel.gtt_view(G1).and_then(|view| view.child), Some(CHILD));
    assert!(kernel.take_commands().is_empty(), "the venue placed it");
    assert!(kernel.apply_fill(CHILD, UNITS, 496.1, at(1, 31)).is_ok());
    assert_eq!(held(&kernel), Some(0.0));
    assert_eq!(status_of(&kernel, G1), Some(GttStatus::ChildFilled));
    let output = finished(kernel);
    let exit = output.fills[1].fill;
    assert_eq!((exit.gtt, exit.reason.code()), (Some(G1), "leg_sl"));
}

#[test]
fn a_child_has_no_order_before_the_venue_fires_its_gtt() {
    let mut kernel = cnc_long_protected(stop_4());
    let fill = kernel.apply_fill(CHILD, UNITS, 496.1, at(1, 31));
    assert!(matches!(fill, Err(KernelError::UnknownOrder(CHILD))));
    assert_eq!(held(&kernel), Some(UNITS));
}

#[test]
fn a_child_not_filled_through_a_base_bar_is_cancelled_then_the_leg_exits_flagged() {
    // Fired at 09:16:30: at 09:17:20 the child still has a minute to go; at 09:17:31 it
    // is cancelled, and only once the venue confirms it does the engine's exit go.
    let mut kernel = cnc_long_protected(stop_4());
    assert_eq!(kernel.apply_gtt_triggered(G1, at(1, 30)), Ok(CHILD));
    assert!(kernel.push_tick(SBIN, 489.0, at(2, 20)).is_ok());
    assert!(kernel.take_commands().is_empty());
    assert!(kernel.push_tick(SBIN, 489.0, at(2, 31)).is_ok());
    assert_eq!(
        kernel.take_commands(),
        [VenueCommand::Cancel { order: CHILD }]
    );
    assert!(kernel.apply_cancel(CHILD).is_ok());
    assert_eq!(status_of(&kernel, G1), Some(GttStatus::ChildCanceled));
    let exit = (
        OrderId(3),
        Side::Sell,
        UNITS,
        OrderRole::RiskExit(RiskExitReason::StopUnfilled),
    );
    assert_eq!(placed_roles(&kernel.take_commands()), [exit]);
    assert!(kernel
        .apply_fill(OrderId(3), UNITS, 489.0, at(2, 32))
        .is_ok());
    let output = finished(kernel);
    let fill = output.fills[1].fill;
    assert_eq!(fill.reason.code(), "leg_sl");
    assert!(fill.flags.contains(FillFlags::GTT_CHILD_UNFILLED));
}

#[test]
fn a_target_waiting_for_the_stop_gtts_cancel_gives_way_to_its_child() {
    // The target's exit (o2) waits for g1's cancellation, and the venue fires g1 first:
    // its child (o3) holds the position, and the target's exit ends canceled.
    let mut kernel = cnc_long_protected(stop_and_target());
    assert!(kernel.push_tick(SBIN, 506.1, at(1, 20)).is_ok());
    kernel.take_commands();
    assert_eq!(kernel.apply_gtt_triggered(G1, at(1, 21)), Ok(OrderId(3)));
    let target = kernel.order(OrderId(2)).map(|o| o.status);
    assert_eq!(target, Some(OrderStatus::Canceled));
    assert!(kernel.take_commands().is_empty());
}

// ---------------------------------------------------------------- refused answers

#[test]
fn an_answer_for_a_gtt_the_session_does_not_have_is_refused() {
    let mut kernel = cnc_long_protected(stop_4());
    assert_eq!(
        refusal_kind(kernel.apply_gtt_placed(GttId(9))),
        "unknown_gtt"
    );
}

#[test]
fn a_second_answer_to_one_request_is_refused_and_changes_nothing() {
    let mut kernel = cnc_long_protected(stop_4());
    let second = kernel.apply_gtt_placed(G1);
    assert_eq!(
        second,
        Err(KernelError::Venue(VenueError::GttState {
            gtt: G1,
            status: GttStatus::Active,
            answer: "placed",
        }))
    );
    assert_eq!(refusal_kind(kernel.apply_gtt_modified(G1)), "gtt_state");
    assert_eq!(
        refusal_kind(kernel.apply_gtt_cancel_refused(G1)),
        "gtt_state"
    );
    assert_eq!(status_of(&kernel, G1), Some(GttStatus::Active));
    assert!(kernel.take_commands().is_empty());
}

#[test]
fn the_gtts_of_a_position_are_queried_by_instrument_and_product() {
    let mut kernel = cnc_long_protected(stop_4());
    assert!(kernel.apply_gtt_cancelled(G1).is_ok());
    let views = kernel.gtt_views(SBIN, Product::Cnc);
    let states: Vec<(GttId, GttStatus)> = views.iter().map(|v| (v.gtt.id, v.gtt.status)).collect();
    assert_eq!(
        states,
        [(G1, GttStatus::VenueEnded), (G2, GttStatus::Requested)]
    );
    assert!(kernel.gtt_views(SBIN, Product::Mis).is_empty());
}

#[test]
fn a_session_close_waits_for_the_answer_to_a_gtt_request() {
    // g1's place is not answered when the day ends: the close waits for it.
    let mut kernel = cnc_long(stop_4());
    kernel.take_commands();
    let last = venue_support::SESSION_SLICES as usize;
    wait_before(
        &mut kernel,
        &bars(&[], &[]),
        &mut Script::new(Vec::new()),
        last,
    );
    let pause = kernel.advance().expect("the close runs");
    assert!(matches!(pause, oa_bt::engine::Pause::Waiting { .. }));
    assert!(kernel.apply_gtt_placed(G1).is_ok());
    let pause = kernel.advance().expect("the close completes");
    assert!(matches!(pause, oa_bt::engine::Pause::Finished { .. }));
}

// ---------------------------------------------------------------- against the backtest

/// Minute 3 falls through the stop from inside: (500.30, 500.40, 495.00, 495.50).
const THROUGH_THE_STOP: Ohlc = (500.3, 500.4, 495.0, 495.5);

fn finished(mut kernel: Engine) -> RunOutput {
    run_to_end(&mut kernel, &bars(&[], &[]), &mut Script::new(Vec::new()));
    kernel.finish().expect("the session finished")
}

/// The names of `output`'s events, without those of the orders in `skipped`.
fn names(output: &RunOutput, skipped: &[OrderId]) -> Vec<String> {
    let skipped: Vec<String> = skipped.iter().map(|id| format!("{id} ")).collect();
    events(output)
        .into_iter()
        .map(|(_, _, name)| name)
        .filter(|name| !skipped.iter().any(|order| name.starts_with(order.as_str())))
        .collect()
}

#[test]
fn the_strategy_sees_a_stop_gtt_fire_as_in_a_backtest() {
    // The backtest fires g1 inside minute 3 at its trigger 496.10. The session's venue
    // fires it there too and fills its child at 496.10. The strategy sees the same events
    // in the same order, but for the child's own order events, which the backtest's
    // engine-held fill has none of.
    let bars_through = bars(&[(3, THROUGH_THE_STOP)], &[]);
    let acts = vec![at_close(0, Act::Submit(cnc_buy(UNITS, stop_4())))];
    let backtest = run_backtest(config(), &bars_through, &mut Script::new(acts));
    let expected = [
        "o1 Submitted",
        "o1 Filled",
        "PositionOpened",
        "GttPlaced g1",
        "GttTriggered g1",
        "leg_sl_hit",
        "GttChildFilled g1",
        "PositionClosed",
    ];
    assert_eq!(names(&backtest, &[]), expected);
    let mut kernel = cnc_long_protected(stop_4());
    wait_before(&mut kernel, &bars_through, &mut Script::new(Vec::new()), 3);
    assert_eq!(kernel.apply_gtt_triggered(G1, minute(3) + 1), Ok(CHILD));
    assert!(kernel
        .apply_fill(CHILD, UNITS, 496.1, minute(3) + 1)
        .is_ok());
    let live = finished(kernel);
    assert_eq!(names(&live, &[CHILD]), expected);
    let child = ["o2 Submitted", "o2 Filled"];
    let all = names(&live, &[]);
    assert!(child.iter().all(|name| all.iter().any(|n| n == name)));
    let trade = |output: &RunOutput| {
        let t = &output.trades[0];
        (
            t.entry_price_rupees,
            t.exit_price_rupees,
            t.gross_pnl_rupees,
        )
    };
    assert_eq!(trade(&live), trade(&backtest));
    assert_eq!(trade(&live), (ENTRY, 496.1, -400.0));
}
