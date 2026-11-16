#![cfg(all(feature = "costs", feature = "time"))]
//! The session close of a live session (`engine::venue_clock`): the venue ends orders,
//! not the clock. A DAY order a backtest would expire is cancelled through the venue and
//! works until the venue's result, the close waits for it, and the session cannot finish
//! while an order works.
//!
//! The test day 2025-07-01 has 375 slices of one minute (09:15 to 15:30); its last is
//! slice 374. The bars rise 0.10 a minute from 500.00 (`venue_support::rising`).

mod venue_support;

use oa_bt::engine::{Engine, KernelError, Pause, RunConfig, Step, VenueCommand};
use oa_bt::exec::{OrderId, OrderStatus};
use oa_bt::model::{days_from_civil, Product, Side};
use venue_support::{
    at_close, bars, config, limit, minute, order, session, wait_before, Act, Script,
};

/// The slice after the test day's last: where a session waits while its close does.
const AFTER_THE_DAY: usize = 375;

/// A CNC buy LIMIT of 10 at 400.00 (o1, never reached) decided at the close of minute
/// 300 in a session of `config`, run to its first session close.
fn resting_at_the_close(config: RunConfig) -> Engine {
    let mut kernel = session(config);
    let resting = order(Side::Buy, 10.0, limit(400.0), Product::Cnc);
    let acts = vec![at_close(300, Act::Submit(resting))];
    wait_before(
        &mut kernel,
        &bars(&[], &[]),
        &mut Script::new(acts),
        AFTER_THE_DAY,
    );
    kernel
}

fn status(kernel: &Engine) -> OrderStatus {
    kernel.order(OrderId(1)).expect("the order").status
}

#[test]
fn a_day_order_working_at_the_session_close_is_cancelled_through_the_venue() {
    let mut kernel = resting_at_the_close(config());
    let cancel = VenueCommand::Cancel { order: OrderId(1) };
    assert_eq!(kernel.take_commands().last(), Some(&cancel));
    assert_eq!(status(&kernel), OrderStatus::Submitted, "not expired");
}

#[test]
fn the_session_does_not_finish_while_an_order_works() {
    let mut kernel = resting_at_the_close(config());
    let waiting = Pause::Waiting {
        next_slice: AFTER_THE_DAY,
    };
    assert_eq!(kernel.advance(), Ok(waiting));
    assert!(kernel.apply_cancel(OrderId(1)).is_ok());
    assert!(matches!(kernel.advance(), Ok(Pause::Finished { .. })));
}

#[test]
fn a_fill_the_venue_reports_before_the_cancellation_is_recorded_in_the_closing_slice() {
    let mut kernel = resting_at_the_close(config());
    let late = minute(374) + 59_000_000_000;
    let stamp = kernel
        .apply_fill(OrderId(1), 10.0, 400.0, late)
        .expect("the fill is recorded");
    assert_eq!((stamp.slice, stamp.step), (374, Step::Levels));
    assert!(matches!(kernel.advance(), Ok(Pause::Finished { .. })));
    let output = kernel.finish().expect("the session finished");
    assert_eq!(output.final_positions[0].quantity_units, 10.0);
    let last_level = output.levels.last().expect("the position's row");
    assert_eq!((last_level.slice, last_level.quantity_units), (374, 10.0));
}

#[test]
fn a_cancellation_the_venue_refuses_at_the_close_is_sent_again() {
    let mut kernel = resting_at_the_close(config());
    kernel.take_commands();
    assert!(kernel.apply_cancel_refused(OrderId(1)).is_ok());
    assert!(matches!(kernel.advance(), Ok(Pause::Waiting { .. })));
    let cancel = VenueCommand::Cancel { order: OrderId(1) };
    assert_eq!(kernel.take_commands(), vec![cancel]);
}

#[test]
fn a_late_fill_of_a_first_day_order_is_recorded_on_that_day() {
    // A two-day session: the first day's close waits for o1, so the broker's fill at
    // 15:29:59 is recorded on its day, not refused as expired.
    let two_days = RunConfig {
        last_day: days_from_civil(2025, 7, 2),
        ..config()
    };
    let mut kernel = resting_at_the_close(two_days);
    let late = minute(374) + 59_000_000_000;
    let stamp = kernel.apply_fill(OrderId(1), 10.0, 400.0, late);
    assert_eq!(stamp.map(|s| s.slice), Ok(374));
    assert_eq!(status(&kernel), OrderStatus::Filled);
    let next_day = Pause::Waiting {
        next_slice: AFTER_THE_DAY,
    };
    assert_eq!(kernel.advance(), Ok(next_day));
    let refused = kernel.apply_fill(OrderId(1), 1.0, 400.0, late);
    assert!(matches!(refused, Err(KernelError::Venue(_))));
}
