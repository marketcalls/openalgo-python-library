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
    at, cnc_buy, cnc_long, cnc_long_protected, gtt_commands, long_stop, nifty_future,
    order_commands, stop_4, stop_and_target, stop_and_trail, stop_and_two_targets, BUY, ENTRY, G1,
    UNITS,
};
use oa_bt::engine::{Engine, GttTerms, KernelError, OrderRole, Submission, VenueCommand};
use oa_bt::exec::{
    GttId, GttStatus, OrderId, OrderRequest, OrderStatus, OrderType, RejectCode, Validity,
};
use oa_bt::model::{Product, Side};
use oa_bt::rms::LegRules;
use venue_support::{
    at_close, bars, config, limit, minute, order, pause_at_close, placed_roles, run_to_end,
    session, session_of, wait_before, Act, Script, SBIN,
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

// ---------------------------------------------------------------- a stop GTT left behind

/// The CNC long of `cnc_long_protected(stop_4())` sold by the broker by itself at 09:16:40
/// and g1's cancellation refused: g1 rests on a flat position.
fn flat_with_g1_resting() -> Engine {
    let mut kernel = cnc_long_protected(stop_4());
    sold_by_the_broker(&mut kernel, UNITS);
    let cancel = gtt_commands(&kernel.take_commands());
    assert!(matches!(
        cancel[..],
        [VenueCommand::CancelGtt { gtt: G1, .. }]
    ));
    assert!(kernel.apply_gtt_cancel_refused(G1).is_ok());
    kernel
}

#[test]
fn a_stop_gtt_left_resting_on_a_flat_position_keeps_entries_out() {
    // A new CNC buy at the 09:16 close is rejected live_in_flight: its stop GTT beside g1
    // would make two sales at the broker, as an MIS entry beside a stop a closed leg left
    // working is rejected.
    let mut kernel = flat_with_g1_resting();
    let acts = vec![at_close(1, Act::Submit(cnc_buy(10.0, stop_4())))];
    let mut script = Script::new(acts);
    wait_before(&mut kernel, &bars(&[], &[]), &mut script, 2);
    assert!(matches!(
        script.submissions[..],
        [Submission::Rejected(_, RejectCode::LiveInFlight)]
    ));
}

/// The CNC long of `cnc_long_protected(stop_4())` with a resting LIMIT buy of 10 at
/// 495.00 with `rules` (o2) placed at the 09:16 close. At 09:17:10 the broker sells the
/// 100 by itself, the venue refuses g1's cancellation, and at 09:17:20 o2 fills and opens
/// a new leg from flat; the commands of that fill are left.
fn new_leg_beside_g1(rules: LegRules) -> Engine {
    new_leg_of_beside_g1(stop_4(), rules)
}

/// `new_leg_beside_g1` with the first long bought with `first` (a stop of 4 points, so
/// that g1 is the same) and o2 placed with `rules`.
fn new_leg_of_beside_g1(first: LegRules, rules: LegRules) -> Engine {
    let mut kernel = cnc_long_protected(first);
    let add = OrderRequest {
        leg_rules: rules,
        ..order(Side::Buy, 10.0, limit(495.0), Product::Cnc)
    };
    let acts = vec![at_close(1, Act::Submit(add))];
    wait_before(&mut kernel, &bars(&[], &[]), &mut Script::new(acts), 2);
    kernel.take_commands();
    let sold = kernel.apply_external_fill(
        SBIN,
        Product::Cnc,
        Side::Sell,
        UNITS,
        499.0,
        at(2, 10),
        oa_bt::rms::ExitReason::Manual,
    );
    assert!(sold.is_ok(), "{sold:?}");
    kernel.take_commands();
    assert!(kernel.apply_gtt_cancel_refused(G1).is_ok());
    assert!(kernel
        .apply_fill(OrderId(2), 10.0, 495.0, at(2, 20))
        .is_ok());
    kernel
}

#[test]
fn a_stop_gtt_an_earlier_leg_left_resting_is_cancelled_again_in_a_later_slice() {
    // g1 is no leg's stop any more and still rests: nothing more in minute 2, the slice
    // of the refusal, and its cancellation goes again once the session reaches minute 3.
    let mut kernel = new_leg_beside_g1(LegRules::default());
    assert!(gtt_commands(&kernel.take_commands()).is_empty());
    wait_before(
        &mut kernel,
        &bars(&[], &[]),
        &mut Script::new(Vec::new()),
        3,
    );
    let again = gtt_commands(&kernel.take_commands());
    assert!(matches!(
        again[..],
        [VenueCommand::CancelGtt { gtt: G1, .. }]
    ));
}

#[test]
fn a_new_legs_stop_gtt_waits_for_the_cancellation_of_the_one_its_earlier_leg_left() {
    // o2 carries a stop of 4 points: 491.00, its child 1 percent below, 486.09 down to the
    // tick, 486.05. That GTT beside g1 would be a second sale at the broker, so it goes
    // only once the venue has cancelled g1.
    let mut kernel = new_leg_beside_g1(stop_4());
    assert!(gtt_commands(&kernel.take_commands()).is_empty());
    wait_before(
        &mut kernel,
        &bars(&[], &[]),
        &mut Script::new(Vec::new()),
        3,
    );
    let again = gtt_commands(&kernel.take_commands());
    assert!(matches!(
        again[..],
        [VenueCommand::CancelGtt { gtt: G1, .. }]
    ));
    assert!(kernel.apply_gtt_cancelled(G1).is_ok());
    let placed = VenueCommand::PlaceGtt {
        gtt: GttId(2),
        order: None,
        terms: long_stop(10.0, 491.0, 486.05),
    };
    assert_eq!(gtt_commands(&kernel.take_commands()), [placed]);
}

#[test]
fn a_resting_exit_of_a_new_leg_beside_the_stop_gtt_its_earlier_leg_left_is_refused_exit_cover() {
    // g1 still sells 100 at 496.10 at the broker until the venue takes its cancellation:
    // a reduce-only LIMIT sale of the new leg's 10 at 505.00 at the 09:17 close would sell
    // them a second time, as beside the leg's own stop GTT (exit_cover).
    let mut kernel = new_leg_beside_g1(LegRules::default());
    kernel.take_commands();
    let sale = OrderRequest {
        reduce_only: true,
        ..order(Side::Sell, 10.0, limit(505.0), Product::Cnc)
    };
    let mut script = Script::new(vec![at_close(2, Act::Submit(sale))]);
    wait_before(&mut kernel, &bars(&[], &[]), &mut script, 3);
    let refused = Submission::Rejected(OrderId(3), RejectCode::ExitCover);
    assert_eq!(script.submissions, [refused]);
}

#[test]
fn a_target_exit_of_a_new_leg_waits_for_the_cancellation_of_the_stop_gtt_its_earlier_leg_left() {
    // The new leg (10 bought at 495.00) has its stop at 491.00, whose GTT waits for g1's
    // cancellation, and its target 6 points up, 501.00. The target is reached at
    // 09:17:30: the exit of the 10 waits while g1 rests, g1's cancellation goes again in
    // minute 3 (the venue refused it in minute 2), and the exit goes once the venue has
    // taken it, with no stop GTT for the leg it closes.
    let mut kernel = new_leg_of_beside_g1(stop_and_target(), stop_and_target());
    kernel.take_commands();
    assert!(kernel.push_tick(SBIN, 501.0, at(2, 30)).is_ok());
    assert!(placed_roles(&kernel.take_commands()).is_empty());
    let mut script = Script::new(Vec::new());
    wait_before(&mut kernel, &bars(&[], &[]), &mut script, 3);
    let commands = kernel.take_commands();
    assert!(placed_roles(&commands).is_empty());
    assert!(matches!(
        gtt_commands(&commands)[..],
        [VenueCommand::CancelGtt { gtt: G1, .. }]
    ));
    assert!(kernel.apply_gtt_cancelled(G1).is_ok());
    let commands = kernel.take_commands();
    let exit = (OrderId(3), Side::Sell, 10.0, OrderRole::Exit);
    assert_eq!(placed_roles(&commands), [exit]);
    assert!(gtt_commands(&commands).is_empty());
}

// ---------------------------------------------------------------- GTT orders against it

#[test]
fn a_gtt_order_against_the_position_held_is_not_built_in_a_session() {
    // A GTT sale of 10 at 495.00 that is not reduce-only, beside g1: both could fire at the
    // broker, a second sale of the position (on CNC, of the user's own holdings). A
    // backtest cancels the stop GTT before such an exit fills; a session cannot.
    let mut kernel = cnc_long_protected(stop_4());
    pause_at_close(&mut kernel, &bars(&[], &[]), 1);
    let sale = OrderRequest {
        validity: Validity::Gtt,
        ..order(Side::Sell, 10.0, limit(495.0), Product::Cnc)
    };
    let refused = kernel.submit(sale);
    assert!(
        matches!(
            refused,
            Err(KernelError::NotBuilt {
                feature: "a GTT order against the position held, in a live session",
                ..
            })
        ),
        "{refused:?}"
    );
    assert!(kernel.take_commands().is_empty());
}

// ---------------------------------------------------------------- one trigger move a minute

#[test]
fn a_cut_within_a_minute_of_a_trail_keeps_the_resting_trigger_until_the_minute_passes() {
    // 503.00 at 09:16:10 trails the stop to 501.00 (child 495.95), confirmed. 506.10 at
    // 09:16:30 would trail it to 504.10 and reaches the first target (half): the cut to 50
    // goes at once on the resting trigger 501.00, and 504.10 (child 1 percent below,
    // 499.059, down to the tick, 499.05) goes a minute after the cut, at 09:17:30.
    let rules = LegRules {
        trail: stop_and_trail().trail,
        ..stop_and_two_targets()
    };
    let mut kernel = cnc_long_protected(rules);
    assert!(kernel.push_tick(SBIN, 503.0, at(1, 10)).is_ok());
    kernel.take_commands();
    assert!(kernel.apply_gtt_modified(G1).is_ok());
    assert!(kernel.push_tick(SBIN, 506.1, at(1, 30)).is_ok());
    let cut = VenueCommand::ModifyGtt {
        gtt: G1,
        terms: long_stop(50.0, 501.0, 495.95),
    };
    assert_eq!(kernel.take_commands(), [cut]);
    assert!(kernel.apply_gtt_modified(G1).is_ok());
    kernel.take_commands();
    assert!(kernel.push_tick(SBIN, 506.0, at(2, 20)).is_ok());
    assert!(gtt_commands(&kernel.take_commands()).is_empty());
    assert!(kernel.push_tick(SBIN, 506.0, at(2, 30)).is_ok());
    let trailed = VenueCommand::ModifyGtt {
        gtt: G1,
        terms: long_stop(50.0, 504.1, 499.05),
    };
    assert_eq!(kernel.take_commands(), [trailed]);
}

#[test]
fn a_cut_goes_at_once_after_the_venue_refused_a_trigger_move() {
    // 503.00 at 09:16:10 trails the stop to 501.00, and the venue refuses that. 506.10 at
    // 09:16:20 reaches the first target (half): the cut to 50 on the resting trigger
    // 496.10 is not the request refused, and the target's exit waits for it, so it goes
    // at once; the trigger (504.10 by now) waits for the minute since 09:16:10.
    let rules = LegRules {
        trail: stop_and_trail().trail,
        ..stop_and_two_targets()
    };
    let mut kernel = cnc_long_protected(rules);
    assert!(kernel.push_tick(SBIN, 503.0, at(1, 10)).is_ok());
    kernel.take_commands();
    assert!(kernel.apply_gtt_modify_refused(G1).is_ok());
    assert!(kernel.take_commands().is_empty());
    assert!(kernel.push_tick(SBIN, 506.1, at(1, 20)).is_ok());
    let cut = VenueCommand::ModifyGtt {
        gtt: G1,
        terms: long_stop(50.0, 496.1, 491.1),
    };
    assert_eq!(kernel.take_commands(), [cut]);
}

#[test]
fn a_cut_the_venue_refused_is_sent_again_only_once_the_minute_has_passed() {
    // The first target's cut to 50 (09:16:20) is refused. The same cut sent again at once
    // would be refused again, in a loop with the venue: it goes again a minute after it was
    // sent, with the next tick from 09:17:20.
    let mut kernel = cnc_long_protected(stop_and_two_targets());
    assert!(kernel.push_tick(SBIN, 506.1, at(1, 20)).is_ok());
    kernel.take_commands();
    assert!(kernel.apply_gtt_modify_refused(G1).is_ok());
    assert!(kernel.push_tick(SBIN, 506.0, at(1, 50)).is_ok());
    assert!(kernel.take_commands().is_empty());
    assert!(kernel.push_tick(SBIN, 506.0, at(2, 25)).is_ok());
    let cut = VenueCommand::ModifyGtt {
        gtt: G1,
        terms: long_stop(50.0, 496.1, 491.1),
    };
    assert_eq!(kernel.take_commands(), [cut]);
}
