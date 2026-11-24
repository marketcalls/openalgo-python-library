#![cfg(all(feature = "costs", feature = "time"))]
//! `protect` for a CNC or NRML leg whose stop is a GTT (design 20-positional-live 3.4,
//! review M11; unit N1 item N1-3): a new stop GTT at the leg's stop, also while the market
//! is closed; an exit at market flagged `stop_crossed` when the price is already beyond
//! the stop; and the outcomes `protect` already reports, kept for GTT legs.
//!
//! SBIN on 2025-07-01, 1-minute bars, no charges, no slippage; 100 bought in CNC at
//! 500.10 at the 09:16 open with a stop of 4 points (trigger 496.10, child limit 491.10:
//! `gtt_venue_support`).

mod gtt_venue_support;
mod venue_support;

use gtt_venue_support::{
    at, cnc_long_protected, gtt_commands, long_stop, stop_4, stop_and_target, G1, UNITS,
};
use oa_bt::engine::{Engine, OrderRole, Pause, Protect, RiskExitReason, VenueCommand};
use oa_bt::exec::{FillFlags, GttId, OrderId, OrderRequest, RejectCode};
use oa_bt::model::{Product, Side};
use oa_bt::rms::LegRules;
use venue_support::{
    at_close, bars, limit, order, placed_roles, run_to_end, wait_before, Act, Script, SBIN,
};

const G2: GttId = GttId(2);
const TARGET_EXIT: OrderId = OrderId(2);

/// A leg with no stop at the broker: its target (506.10) reached on a tick cancelled g1,
/// and the broker rejected the exit (o2) that followed, so the position is open and
/// unprotected; no command is left.
fn unprotected_after_a_failed_exit() -> Engine {
    let mut kernel = cnc_long_protected(stop_and_target());
    assert!(kernel.push_tick(SBIN, 506.1, at(1, 20)).is_ok());
    kernel.take_commands();
    assert!(kernel.apply_gtt_cancelled(G1).is_ok());
    kernel.take_commands();
    assert!(kernel.apply_reject(TARGET_EXIT, RejectCode::Venue).is_ok());
    assert!(
        kernel.take_commands().is_empty(),
        "the kernel places nothing by itself"
    );
    kernel
}

#[test]
fn protect_places_a_new_stop_gtt_at_the_legs_stop() {
    let mut kernel = unprotected_after_a_failed_exit();
    assert_eq!(kernel.protect(SBIN, Product::Cnc), Ok(Protect::Placed));
    let placed = VenueCommand::PlaceGtt {
        gtt: G2,
        order: None,
        terms: long_stop(UNITS, 496.1, 491.1),
    };
    assert_eq!(kernel.take_commands(), [placed]);
    assert_eq!(kernel.protect(SBIN, Product::Cnc), Ok(Protect::StopWorks));
}

#[test]
fn protect_exits_a_leg_whose_stop_the_price_has_crossed_flagged_stop_crossed() {
    // The last price 495.00 is below the stop 496.10: no GTT, an exit at market whose
    // fill is flagged stop_crossed, reason leg_sl.
    let mut kernel = unprotected_after_a_failed_exit();
    assert!(kernel.push_tick(SBIN, 495.0, at(1, 50)).is_ok());
    assert_eq!(kernel.protect(SBIN, Product::Cnc), Ok(Protect::StopCrossed));
    let commands = kernel.take_commands();
    assert!(gtt_commands(&commands).is_empty());
    let exit = (
        OrderId(3),
        Side::Sell,
        UNITS,
        OrderRole::RiskExit(RiskExitReason::StopUnfilled),
    );
    assert_eq!(placed_roles(&commands), [exit]);
    assert!(kernel
        .apply_fill(OrderId(3), UNITS, 494.9, at(1, 51))
        .is_ok());
    run_to_end(&mut kernel, &bars(&[], &[]), &mut Script::new(Vec::new()));
    let output = kernel.finish().expect("the session finished");
    let fill = output.fills[1].fill;
    assert_eq!(fill.reason.code(), "leg_sl");
    assert!(fill.flags.contains(FillFlags::STOP_CROSSED));
}

#[test]
fn a_stop_gtt_that_ends_while_the_session_closes_is_placed_again_with_the_market_closed() {
    // A resting LIMIT add of 10 at 400.00 keeps the session close waiting for its
    // cancellation. g1 ends at the venue meanwhile: g2 is placed all the same, since
    // brokers take GTTs outside market hours.
    let mut kernel = cnc_long_protected(stop_4());
    let add = OrderRequest {
        leg_rules: stop_4(),
        ..order(Side::Buy, 10.0, limit(400.0), Product::Cnc)
    };
    let last = venue_support::SESSION_SLICES as usize;
    let acts = vec![at_close(1, Act::Submit(add))];
    wait_before(&mut kernel, &bars(&[], &[]), &mut Script::new(acts), last);
    // The add's place, then its cancellation at the close.
    let closing = kernel.take_commands();
    let cancel = VenueCommand::Cancel { order: OrderId(2) };
    assert_eq!(closing.last(), Some(&cancel));
    assert!(kernel.apply_gtt_cancelled(G1).is_ok());
    let again = gtt_commands(&kernel.take_commands());
    assert!(matches!(
        again[..],
        [VenueCommand::PlaceGtt { gtt: G2, .. }]
    ));
    assert_eq!(kernel.protect(SBIN, Product::Cnc), Ok(Protect::StopWorks));
    assert!(kernel.apply_cancel(OrderId(2)).is_ok());
    assert!(matches!(kernel.advance(), Ok(Pause::Waiting { .. })));
    assert!(kernel.apply_gtt_placed(G2).is_ok());
    assert!(matches!(kernel.advance(), Ok(Pause::Finished { .. })));
}

#[test]
fn protect_keeps_its_other_outcomes_for_a_gtt_leg() {
    // Flat before any entry; no stop level on a leg without one; exiting while a target's
    // exit waits for the GTT's cancellation.
    let mut flat = venue_support::session(venue_support::config());
    let script = &mut Script::new(Vec::new());
    wait_before(&mut flat, &bars(&[], &[]), script, 1);
    assert_eq!(flat.protect(SBIN, Product::Cnc), Ok(Protect::Flat));
    let mut no_stop = gtt_venue_support::cnc_long(LegRules::default());
    assert_eq!(
        no_stop.protect(SBIN, Product::Cnc),
        Ok(Protect::NoStopLevel)
    );
    let mut exiting = cnc_long_protected(stop_and_target());
    assert!(exiting.push_tick(SBIN, 506.1, at(1, 20)).is_ok());
    assert_eq!(exiting.protect(SBIN, Product::Cnc), Ok(Protect::Exiting));
}

#[test]
fn the_protect_outcomes_end_with_stop_crossed() {
    assert_eq!(Protect::StopCrossed.code(), "stop_crossed");
    assert_eq!(Protect::CODES.last(), Some(&"stop_crossed"));
}
