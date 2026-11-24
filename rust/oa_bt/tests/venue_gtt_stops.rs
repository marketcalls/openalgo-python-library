#![cfg(all(feature = "costs", feature = "time"))]
//! A CNC or NRML leg's stop held at the broker as one SINGLE GTT in a live session, and
//! its targets held on ticks (design 20-positional-live sections 2 and 3.6, unit N1 item
//! N1-1): the GTT placed with the entry's fill, moved by a trail at most once a minute,
//! cancelled before a target's exit of the whole position, cut before a partial one, and
//! sent in the direction of its trigger. A session never fires a GTT itself.
//!
//! SBIN on 2025-07-01, 1-minute bars, no charges, no slippage; 100 bought in CNC at
//! 500.10 at the 09:16 open (`gtt_venue_support`).

mod gtt_venue_support;
mod venue_support;

use gtt_venue_support::{
    at, cnc_long, cnc_long_protected, gtt_commands, long_stop, order_commands, stop_4,
    stop_and_target, stop_and_trail, stop_and_two_targets, BUY, ENTRY, G1, UNITS,
};
use oa_bt::engine::{GttTerms, OrderRole, VenueCommand};
use oa_bt::exec::{
    GttId, GttStatus, InstrumentSpec, OrderId, OrderRequest, OrderStatus, OrderType, Validity,
};
use oa_bt::model::{days_from_civil, Exchange, InstrumentKind, Product, Side};
use oa_bt::rms::LegRules;
use venue_support::{
    at_close, bars, config, limit, minute, order, placed_roles, run_to_end, sbin, session,
    session_of, wait_before, Act, Script, SBIN,
};

#[test]
fn a_cnc_entry_that_fills_places_one_stop_gtt_at_the_stop() {
    // 100 bought at 500.10, stop 4 points: the trigger 496.10, below the price; its child
    // sells 1 percent below it, 491.139, down to the tick: 491.10. No stop order.
    let mut kernel = cnc_long(stop_4());
    let expected = VenueCommand::PlaceGtt {
        gtt: G1,
        order: None,
        terms: long_stop(UNITS, 496.1, 491.1),
    };
    assert_eq!(kernel.take_commands(), [expected]);
}

#[test]
fn a_stop_gtt_is_not_in_force_until_the_venue_confirms_it() {
    let mut kernel = cnc_long(stop_4());
    kernel.take_commands();
    let view = kernel.gtt_view(G1).expect("g1 is the session's");
    assert_eq!(view.gtt.status, GttStatus::Requested);
    assert!(kernel.apply_gtt_placed(G1).is_ok());
    let view = kernel.gtt_view(G1).expect("g1 is the session's");
    assert_eq!((view.gtt.status, view.child), (GttStatus::Active, None));
}

#[test]
fn a_leg_without_a_stop_places_no_gtt() {
    let mut kernel = cnc_long(LegRules::default());
    assert!(kernel.take_commands().is_empty());
}

#[test]
fn a_trail_moves_the_stop_gtt_at_most_once_a_minute() {
    // 503.00 arms the 2-point trail: stop 501.00, child 495.99 down to 495.95, sent at
    // once. 504.00 twenty seconds later moves it to 502.00, held: one minute has not
    // passed since the last modification. A tick a minute after it sends 502.00 (child
    // 496.98 down to 496.95).
    let mut kernel = cnc_long_protected(stop_and_trail());
    assert!(kernel.push_tick(SBIN, 503.0, at(1, 10)).is_ok());
    let first = VenueCommand::ModifyGtt {
        gtt: G1,
        terms: long_stop(UNITS, 501.0, 495.95),
    };
    assert_eq!(kernel.take_commands(), [first]);
    assert!(kernel.apply_gtt_modified(G1).is_ok());
    assert!(kernel.push_tick(SBIN, 504.0, at(1, 30)).is_ok());
    assert!(kernel.take_commands().is_empty(), "held for the minute");
    assert!(kernel.push_tick(SBIN, 503.5, at(2, 10)).is_ok());
    let second = VenueCommand::ModifyGtt {
        gtt: G1,
        terms: long_stop(UNITS, 502.0, 496.95),
    };
    assert_eq!(kernel.take_commands(), [second]);
}

#[test]
fn a_modification_waits_for_the_answer_to_the_one_before() {
    // The first move goes; while it waits, a second one is the leg's and goes with the
    // answer once the minute has passed.
    let mut kernel = cnc_long_protected(stop_and_trail());
    assert!(kernel.push_tick(SBIN, 503.0, at(1, 10)).is_ok());
    kernel.take_commands();
    assert!(kernel.push_tick(SBIN, 505.0, at(2, 20)).is_ok());
    assert!(
        kernel.take_commands().is_empty(),
        "one waits for its answer"
    );
    assert!(kernel.apply_gtt_modified(G1).is_ok());
    let newest = VenueCommand::ModifyGtt {
        gtt: G1,
        terms: long_stop(UNITS, 503.0, 497.95),
    };
    assert_eq!(kernel.take_commands(), [newest]);
}

#[test]
fn a_target_on_a_tick_cancels_the_stop_gtt_and_exits_only_once_the_cancel_is_confirmed() {
    // The target 506.10 is reached: the exit of all 100 waits for g1's cancellation.
    let mut kernel = cnc_long_protected(stop_and_target());
    assert!(kernel.push_tick(SBIN, 506.1, at(1, 20)).is_ok());
    let commands = kernel.take_commands();
    let cancel = VenueCommand::CancelGtt {
        gtt: G1,
        terms: long_stop(UNITS, 496.1, 491.1),
    };
    assert_eq!(commands, [cancel]);
    assert!(kernel.apply_gtt_cancelled(G1).is_ok());
    let exit = (OrderId(2), Side::Sell, UNITS, OrderRole::Exit);
    assert_eq!(placed_roles(&kernel.take_commands()), [exit]);
    assert!(kernel
        .apply_fill(OrderId(2), UNITS, 506.1, at(1, 21))
        .is_ok());
    assert_eq!(
        kernel
            .position(SBIN, Product::Cnc)
            .map(|p| p.quantity_units),
        Some(0.0)
    );
}

#[test]
fn a_partial_target_cuts_the_stop_gtt_to_the_units_left_before_its_exit() {
    // The first target (506.10) sells half: g1 is cut to the 50 left, and the exit of 50
    // goes once the venue confirms the cut.
    let mut kernel = cnc_long_protected(stop_and_two_targets());
    assert!(kernel.push_tick(SBIN, 506.1, at(1, 20)).is_ok());
    let cut = VenueCommand::ModifyGtt {
        gtt: G1,
        terms: long_stop(50.0, 496.1, 491.1),
    };
    assert_eq!(kernel.take_commands(), [cut]);
    assert!(kernel.apply_gtt_modified(G1).is_ok());
    let exit = (OrderId(2), Side::Sell, 50.0, OrderRole::Exit);
    assert_eq!(placed_roles(&kernel.take_commands()), [exit]);
}

#[test]
fn a_strategy_partial_exit_cuts_the_stop_gtt_first() {
    // reduce() of 40 units at the 09:16 close: g1 is cut to 60, then the sale goes.
    let mut kernel = cnc_long_protected(stop_4());
    let sale = OrderRequest {
        reduce_only: true,
        ..order(Side::Sell, 40.0, OrderType::Market, Product::Cnc)
    };
    let acts = vec![at_close(1, Act::Submit(sale))];
    wait_before(&mut kernel, &bars(&[], &[]), &mut Script::new(acts), 2);
    let cut = VenueCommand::ModifyGtt {
        gtt: G1,
        terms: long_stop(60.0, 496.1, 491.1),
    };
    assert_eq!(kernel.take_commands(), [cut]);
    assert!(kernel.apply_gtt_modified(G1).is_ok());
    let sold = (OrderId(2), Side::Sell, 40.0, OrderRole::Exit);
    assert_eq!(placed_roles(&kernel.take_commands()), [sold]);
}

/// A NIFTY future: NFO, lot 75, tick 0.05.
fn nifty_future() -> InstrumentSpec {
    InstrumentSpec {
        symbol: "NIFTY31JUL25FUT".to_string(),
        exchange: Exchange::Nfo,
        kind: InstrumentKind::Future,
        lot_size_units: 75,
        last_trading_day: Some(days_from_civil(2025, 7, 31)),
        ..sbin()
    }
}

#[test]
fn an_nrml_short_futures_stop_gtt_lies_above_the_price_and_says_so() {
    // One lot sold at 25000.00, stop 50 points: the trigger 25050.00 above the price; its
    // child buys 1 percent above it, 25300.50.
    let mut kernel = session_of(config(), nifty_future());
    let short = OrderRequest {
        leg_rules: LegRules {
            stop: Some(gtt_venue_support::points(50.0)),
            ..LegRules::default()
        },
        ..order(Side::Sell, 75.0, OrderType::Market, Product::Nrml)
    };
    let acts = vec![at_close(0, Act::Submit(short))];
    wait_before(&mut kernel, &bars(&[], &[]), &mut Script::new(acts), 1);
    kernel.take_commands();
    assert!(kernel.apply_fill(BUY, 75.0, 25_000.0, minute(1)).is_ok());
    let terms = GttTerms {
        instrument: SBIN,
        product: Product::Nrml,
        side: Side::Buy,
        units: 75.0,
        trigger_rupees: 25_050.0,
        limit_rupees: 25_300.5,
        trigger_below: false,
    };
    let expected = VenueCommand::PlaceGtt {
        gtt: G1,
        order: None,
        terms,
    };
    assert_eq!(kernel.take_commands(), [expected]);
}

#[test]
fn a_session_never_fires_a_stop_gtt_itself() {
    // Minute 1's bar runs down to 480.00, through the trigger 496.10: the kernel fires
    // nothing; the GTT rests until the venue says it fired.
    let mut kernel = cnc_long_protected(stop_4());
    let fall = (500.1, 500.2, 480.0, 481.0);
    let mut script = Script::new(Vec::new());
    wait_before(&mut kernel, &bars(&[(1, fall)], &[]), &mut script, 3);
    assert!(kernel.push_tick(SBIN, 480.0, at(2, 5)).is_ok());
    assert!(kernel.take_commands().is_empty());
    let view = kernel.gtt_view(G1).expect("g1 is the session's");
    assert_eq!(view.gtt.status, GttStatus::Active);
    assert_eq!(
        kernel
            .position(SBIN, Product::Cnc)
            .map(|p| p.quantity_units),
        Some(UNITS)
    );
}

#[test]
fn a_strategy_gtt_entry_rests_as_a_gtt_and_not_as_an_order() {
    // A GTT buy of 10 at 490.00 below the last close (500.15): its trigger 490.00 below
    // the price, its child buying 1 percent above it, 494.90. No order is placed, and it
    // is not withdrawn at the session close.
    let mut kernel = session(config());
    let resting = OrderRequest {
        validity: Validity::Gtt,
        ..order(Side::Buy, 10.0, limit(490.0), Product::Cnc)
    };
    let acts = vec![at_close(0, Act::Submit(resting))];
    wait_before(&mut kernel, &bars(&[], &[]), &mut Script::new(acts), 1);
    let terms = GttTerms {
        instrument: SBIN,
        product: Product::Cnc,
        side: Side::Buy,
        units: 10.0,
        trigger_rupees: 490.0,
        limit_rupees: 494.9,
        trigger_below: true,
    };
    let expected = VenueCommand::PlaceGtt {
        gtt: GttId(1),
        order: Some(BUY),
        terms,
    };
    assert_eq!(kernel.take_commands(), [expected]);
    assert!(kernel.apply_gtt_placed(GttId(1)).is_ok());
    // The session close has no DAY order to end: it completes at once.
    run_to_end(&mut kernel, &bars(&[], &[]), &mut Script::new(Vec::new()));
    assert!(order_commands(&kernel.take_commands()).is_empty());
    let order = kernel.order(BUY).expect("o1 is the session's");
    assert_eq!(order.status, OrderStatus::Submitted);
}

#[test]
fn an_mis_leg_keeps_its_stop_order_and_sends_no_gtt() {
    // The 2.4.1 rule, unchanged: the MIS stop is an SL-M order of the leg.
    let mut kernel = session(config());
    let request = OrderRequest {
        leg_rules: stop_4(),
        ..order(Side::Buy, UNITS, OrderType::Market, Product::Mis)
    };
    let acts = vec![at_close(0, Act::Submit(request))];
    wait_before(&mut kernel, &bars(&[], &[]), &mut Script::new(acts), 1);
    kernel.take_commands();
    assert!(kernel.apply_fill(BUY, UNITS, ENTRY, minute(1)).is_ok());
    let commands = kernel.take_commands();
    assert!(gtt_commands(&commands).is_empty());
    let stop = (OrderId(2), Side::Sell, UNITS, OrderRole::LegStop);
    assert_eq!(placed_roles(&commands), [stop]);
}

/// The broker sells `units` of the CNC long by itself at 499.00 at 09:16:40.
fn sold_by_the_broker(kernel: &mut oa_bt::engine::Engine, units: f64) {
    let sold = kernel.apply_external_fill(
        SBIN,
        Product::Cnc,
        Side::Sell,
        units,
        499.0,
        at(1, 40),
        oa_bt::rms::ExitReason::Manual,
    );
    assert!(sold.is_ok(), "{sold:?}");
}

#[test]
fn a_sale_the_broker_made_by_itself_cuts_the_stop_gtt_to_what_is_left() {
    let mut kernel = cnc_long_protected(stop_4());
    sold_by_the_broker(&mut kernel, 30.0);
    let cut = VenueCommand::ModifyGtt {
        gtt: G1,
        terms: long_stop(70.0, 496.1, 491.1),
    };
    assert_eq!(kernel.take_commands(), [cut]);
}

#[test]
fn a_sale_the_broker_made_of_the_whole_position_cancels_the_stop_gtt() {
    let mut kernel = cnc_long_protected(stop_4());
    sold_by_the_broker(&mut kernel, UNITS);
    let cancel = VenueCommand::CancelGtt {
        gtt: G1,
        terms: long_stop(UNITS, 496.1, 491.1),
    };
    assert_eq!(kernel.take_commands(), [cancel]);
}

#[test]
fn a_resting_sale_beside_a_stop_gtt_for_the_whole_position_is_refused_as_beside_a_stop() {
    // The GTT covers the 100 held: a reduce-only LIMIT sale of 100 has no room left on the
    // closing side (exit_cover), as beside an MIS leg's stop.
    let mut kernel = cnc_long_protected(stop_4());
    let sale = OrderRequest {
        reduce_only: true,
        ..order(Side::Sell, UNITS, limit(520.0), Product::Cnc)
    };
    let acts = vec![at_close(1, Act::Submit(sale))];
    let mut script = Script::new(acts);
    wait_before(&mut kernel, &bars(&[], &[]), &mut script, 2);
    let rejected =
        oa_bt::engine::Submission::Rejected(OrderId(2), oa_bt::exec::RejectCode::ExitCover);
    assert_eq!(script.submissions, [rejected]);
}
