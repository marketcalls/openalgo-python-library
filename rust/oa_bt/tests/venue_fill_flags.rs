#![cfg(all(feature = "costs", feature = "time"))]
//! The flags a live session's venue reports with a fill (`Engine::apply_fill_flagged`;
//! design 20-positional-live 3.3 and 4.3, unit N4f item N4f-1): `price_unverified`, a
//! fill whose price is not known (a fired GTT's child the session never saw, recorded at
//! the child's limit), and `missed_day`, a stop GTT that fired on a trading day no session
//! ran. Only these two come from a venue, and never into a backtest. They travel as the
//! kernel's own flags do: on the fill, on the trade the fill belongs to, and through a
//! same-day restart and a carry to the next trading day.
//!
//! SBIN, 1-minute bars, no charges, no slippage. The long: 100 bought in CNC at 500.10 at
//! the 09:16 open of 2025-07-01 with a stop of 4 points, a stop GTT (g1) triggering at
//! 496.10 with its child selling at 491.10 (`gtt_venue_support`). The GTT entry: a CNC
//! buy of 10 resting as a GTT (g1) at 490.00, decided at the 09:15 close.

mod gtt_venue_support;
mod venue_support;

use gtt_venue_support::{at, cnc_long_protected, stop_4, BUY, G1, UNITS};
use oa_bt::engine::{
    Engine, EngineExit, KernelError, RiskExitReason, RunConfig, RunOutput, StateValue,
    VenueCommand, VenueError,
};
use oa_bt::exec::{FillFlags, OrderId, OrderRequest, OrderStatus, Validity};
use oa_bt::model::{days_from_civil, ts_from_ist, Day, Product, Side};
use oa_bt::rms::ExitReason;
use venue_support::{
    at_close, backtest, bars, config, limit, minute, order, placed_roles, run_to_end, session,
    wait_before, Act, Script, NSE_OPEN_MINUTE, SBIN,
};

/// The child of the long's stop GTT: the session's second order.
const CHILD: OrderId = OrderId(2);
/// The child's limit, rupees: 1 percent below the 496.10 trigger, down to the tick.
const CHILD_LIMIT: f64 = 491.1;
/// The long's loss when its child is recorded at its limit: (491.10 - 500.10) x 100.
const LOSS_AT_THE_LIMIT: f64 = -900.0;

fn unverified() -> FillFlags {
    FillFlags::PRICE_UNVERIFIED
}

fn unverified_on_a_missed_day() -> FillFlags {
    FillFlags::PRICE_UNVERIFIED.with(FillFlags::MISSED_DAY)
}

fn held(kernel: &Engine) -> Option<f64> {
    kernel
        .position(SBIN, Product::Cnc)
        .map(|position| position.quantity_units)
}

/// `kernel` run to its end, the venue confirming the close's cancellations, and its output.
fn finished(mut kernel: Engine) -> RunOutput {
    run_to_end(&mut kernel, &bars(&[], &[]), &mut Script::new(Vec::new()));
    kernel.finish().expect("the session finished")
}

/// The long whose stop GTT the venue fired at 09:16:30: its child (o2) followed, unfilled.
fn long_with_child_fired() -> Engine {
    let mut kernel = cnc_long_protected(stop_4());
    assert_eq!(kernel.apply_gtt_triggered(G1, at(1, 30)), Ok(CHILD));
    kernel
}

#[test]
fn a_missed_childs_fill_at_its_limit_closes_the_trade_flagged_price_unverified() {
    // The session never saw the child's fill; the position fell by the GTT's units, so the
    // fall is the child's fill, at the child's limit, its price not known.
    let mut kernel = long_with_child_fired();
    let filled = kernel.apply_fill_flagged(CHILD, UNITS, CHILD_LIMIT, at(1, 31), unverified());
    assert!(filled.is_ok(), "{filled:?}");
    assert_eq!(held(&kernel), Some(0.0));
    let output = finished(kernel);
    let (entry, exit) = (output.fills[0].fill, output.fills[1].fill);
    assert_eq!(entry.flags, FillFlags::NONE);
    assert_eq!(
        (exit.price_rupees, exit.gtt, exit.reason.code()),
        (CHILD_LIMIT, Some(G1), "leg_sl")
    );
    assert_eq!(exit.flags.names(), vec!["price_unverified"]);
    let trade = output.trades[0];
    assert_eq!(
        (trade.exit_price_rupees, trade.gross_pnl_rupees),
        (CHILD_LIMIT, LOSS_AT_THE_LIMIT)
    );
    assert_eq!(trade.flags.names(), vec!["price_unverified"]);
}

#[test]
fn a_stop_that_fired_on_a_missed_day_carries_both_flags() {
    let mut kernel = long_with_child_fired();
    let flags = unverified_on_a_missed_day();
    let filled = kernel.apply_fill_flagged(CHILD, UNITS, CHILD_LIMIT, at(1, 31), flags);
    assert!(filled.is_ok(), "{filled:?}");
    let output = finished(kernel);
    let names = vec!["price_unverified", "missed_day"];
    assert_eq!(output.fills[1].fill.flags.names(), names);
    assert_eq!(output.trades[0].flags.names(), names);
}

#[test]
fn a_plain_fill_and_its_trade_carry_no_flag() {
    let mut kernel = long_with_child_fired();
    assert!(kernel.apply_fill(CHILD, UNITS, 496.1, at(1, 31)).is_ok());
    let output = finished(kernel);
    assert_eq!(output.fills[1].fill.flags, FillFlags::NONE);
    assert_eq!(output.trades[0].flags, FillFlags::NONE);
}

#[test]
fn a_flag_the_kernel_sets_itself_is_refused_from_the_venue_and_nothing_is_recorded() {
    // The five flags of the kernel's own fills, alone and beside a venue's flag.
    let kernels_own = [
        FillFlags::FILLED_AT_LAST_CLOSE,
        FillFlags::GTT_CHILD_UNFILLED,
        FillFlags::EXTERNAL,
        FillFlags::STOP_CROSSED,
        FillFlags::LATE_DECISION,
    ];
    for own in kernels_own {
        for flags in [own, own.with(FillFlags::PRICE_UNVERIFIED)] {
            let mut kernel = long_with_child_fired();
            let refused = kernel.apply_fill_flagged(CHILD, UNITS, CHILD_LIMIT, at(1, 31), flags);
            let Err(KernelError::Venue(refusal @ VenueError::InvalidFill { .. })) = refused else {
                panic!(
                    "{:?} refused as an invalid fill: {refused:?}",
                    flags.names()
                );
            };
            let message = refusal.to_string();
            assert!(
                message.contains("price_unverified") && message.contains("missed_day"),
                "{message}"
            );
            assert_eq!(held(&kernel), Some(UNITS), "{:?}", flags.names());
            let child = kernel.order(CHILD).expect("the child is the session's");
            assert_eq!((child.filled_units, child.status.is_working()), (0.0, true));
        }
    }
}

#[test]
fn a_bit_that_names_no_flag_is_refused_by_its_value_and_nothing_is_recorded() {
    // Bit 128 is no fill flag (`FillFlags::NAMED`): the refusal says so by its value
    // rather than naming no flag (unit N4f round 1, finding 5).
    let mut kernel = long_with_child_fired();
    let unknown = FillFlags(128).with(FillFlags::PRICE_UNVERIFIED);
    let refused = kernel.apply_fill_flagged(CHILD, UNITS, CHILD_LIMIT, at(1, 31), unknown);
    let Err(KernelError::Venue(refusal @ VenueError::InvalidFill { .. })) = refused else {
        panic!("refused as an invalid fill: {refused:?}");
    };
    let message = refusal.to_string();
    assert!(
        message.contains("bits 0b10000000 name no fill flag"),
        "{message}"
    );
    assert_eq!(held(&kernel), Some(UNITS));
}

#[test]
fn a_backtest_takes_no_flagged_fill() {
    let mut kernel = backtest(config(), bars(&[], &[]));
    let refused = kernel.apply_fill_flagged(OrderId(1), 10.0, 500.0, minute(1), unverified());
    assert!(
        matches!(refused, Err(KernelError::Venue(VenueError::NotExternal))),
        "{refused:?}"
    );
}

/// A session whose CNC buy of 10 resting as a GTT (o1, g1) the venue fired at 09:16:05,
/// its child filled at 490.00 with `flags`, waiting before minute 1.
fn gtt_entry_filled(flags: FillFlags) -> Engine {
    let mut kernel = session(config());
    let resting = OrderRequest {
        validity: Validity::Gtt,
        ..order(Side::Buy, 10.0, limit(490.0), Product::Cnc)
    };
    let acts = vec![at_close(0, Act::Submit(resting))];
    wait_before(&mut kernel, &bars(&[], &[]), &mut Script::new(acts), 1);
    kernel.take_commands();
    kernel.apply_gtt_placed(G1).expect("the venue confirms g1");
    assert_eq!(kernel.apply_gtt_triggered(G1, at(1, 5)), Ok(BUY));
    kernel
        .apply_fill_flagged(BUY, 10.0, 490.0, at(1, 6), flags)
        .expect("the entry's child fills");
    kernel
}

/// The operator's exit of `kernel`'s 10, filled by the venue at 495.00 at `filled_at`,
/// then the session stopped: its output.
fn exited_at_495(mut kernel: Engine, filled_at: i64) -> RunOutput {
    let exit = EngineExit::Risk(RiskExitReason::OperatorStop);
    kernel
        .request_exit(SBIN, Product::Cnc, exit)
        .expect("the session asks for the exit");
    let placed = placed_roles(&kernel.take_commands());
    assert_eq!(placed.len(), 1, "{placed:?}");
    let (id, side, units, _) = placed[0];
    assert_eq!((side, units), (Side::Sell, 10.0));
    kernel
        .apply_fill(id, 10.0, 495.0, filled_at)
        .expect("the exit fills");
    kernel.stop_session().expect("the session stops");
    kernel.finish().expect("the session finished")
}

/// A session restarted on 2025-07-01, caught up to the wait before minute 1, with `state`
/// restored.
fn restarted_with(state: &StateValue) -> Engine {
    let mut kernel = session(config());
    wait_before(
        &mut kernel,
        &bars(&[], &[]),
        &mut Script::new(Vec::new()),
        1,
    );
    kernel
        .restore_live_state(state)
        .expect("the saved state restores");
    kernel
}

#[test]
fn a_same_day_restart_keeps_the_flags_of_a_trade_still_open() {
    // Bought at 490.00, price unverified, before the restart; sold at 495.00 after it: the
    // trade the restarted session closes shows the entry's flag, and its own exit fill none.
    let state = gtt_entry_filled(unverified()).live_state();
    let output = exited_at_495(restarted_with(&state), at(1, 40));
    let trade = output.trades[0];
    assert_eq!(
        (
            trade.entry_price_rupees,
            trade.exit_price_rupees,
            trade.gross_pnl_rupees
        ),
        (490.0, 495.0, 50.0)
    );
    assert_eq!(trade.flags.names(), vec!["price_unverified"]);
    assert_eq!(output.fills.len(), 1, "only the exit is this start's fill");
    assert_eq!(output.fills[0].fill.flags, FillFlags::NONE);
}

/// `state` without the field `field` in any item of its part `part`.
fn without_field(state: &StateValue, part: &str, field: &str) -> StateValue {
    let mut state = state.clone();
    let StateValue::Map(parts) = &mut state else {
        panic!("a saved state is a map");
    };
    for (key, value) in parts.iter_mut() {
        let StateValue::List(items) = value else {
            continue;
        };
        if key != part {
            continue;
        }
        for item in items.iter_mut() {
            if let StateValue::Map(fields) = item {
                fields.retain(|(name, _)| name != field);
            }
        }
    }
    state
}

#[test]
fn a_state_saved_before_trades_kept_their_flags_restores_with_none() {
    // Layout 6 as written before the open trades carried their flags: it still restores,
    // and the trade then knows of no flag from before the restart.
    let state = gtt_entry_filled(unverified()).live_state();
    let older = without_field(&state, "open_trades", "flags");
    assert_ne!(older, state, "the saved open trade had its flags");
    let output = exited_at_495(restarted_with(&older), at(1, 40));
    assert_eq!(output.trades[0].flags, FillFlags::NONE);
}

fn thursday() -> Day {
    days_from_civil(2025, 7, 3)
}

#[test]
fn a_stop_gtt_that_fired_on_a_missed_day_is_recorded_at_the_next_start_with_both_flags() {
    // The long is carried from Tue 07-01; no session runs on Wed 07-02, when g1 fires and
    // its child fills at the broker. Thursday's session finds g1 gone and the position
    // fallen by its 100: before the open it records the child's fill at the child's limit,
    // flagged price_unverified and missed_day, in Thursday's books.
    let mut tuesday = cnc_long_protected(stop_4());
    run_to_end(&mut tuesday, &bars(&[], &[]), &mut Script::new(Vec::new()));
    let carried = tuesday.carried_state().expect("Tuesday has closed");
    let on_thursday = RunConfig {
        first_day: thursday(),
        last_day: thursday(),
        ..config()
    };
    let mut kernel = session(on_thursday);
    let ids = kernel
        .restore_carried_state(&carried)
        .expect("Thursday takes it");
    let (_, stop) = ids.gtts[0];
    let before_open = ts_from_ist(thursday(), NSE_OPEN_MINUTE) - 60_000_000_000;
    let child = kernel
        .apply_gtt_triggered(stop, before_open)
        .expect("the start finds g1 fired");
    let flags = unverified_on_a_missed_day();
    let filled = kernel.apply_fill_flagged(child, UNITS, CHILD_LIMIT, before_open, flags);
    assert!(filled.is_ok(), "{filled:?}");
    assert_eq!(held(&kernel), Some(0.0));
    assert_eq!(kernel.take_commands(), Vec::<VenueCommand>::new());
    kernel.stop_session().expect("the session stops");
    let output = kernel.finish().expect("the session finished");
    let exit = output.fills[0].fill;
    assert_eq!(
        (exit.price_rupees, exit.reason.code()),
        (CHILD_LIMIT, ExitReason::LegSl.code())
    );
    assert_eq!(exit.flags.names(), vec!["price_unverified", "missed_day"]);
    let trade = output.trades[0];
    assert_eq!(
        (trade.entry_price_rupees, trade.gross_pnl_rupees),
        (500.1, LOSS_AT_THE_LIMIT)
    );
    assert_eq!(trade.flags, unverified_on_a_missed_day());
}

#[test]
fn the_child_order_ends_filled_with_its_flagged_fill() {
    let mut kernel = long_with_child_fired();
    assert!(kernel
        .apply_fill_flagged(CHILD, UNITS, CHILD_LIMIT, at(1, 31), unverified())
        .is_ok());
    let child = kernel.order(CHILD).expect("the child is the session's");
    assert_eq!(
        (child.status, child.filled_units, child.average_fill_rupees),
        (OrderStatus::Filled, UNITS, CHILD_LIMIT)
    );
}
