#![cfg(all(feature = "costs", feature = "time"))]
//! A live session's orders and the venue's results (`engine::venue`, `venue_fills`): what
//! the session sends, and how it records the fills, rejections and cancellations the
//! venue sends back.
//!
//! The bars rise 0.10 a minute from 500.00 (`venue_support::rising`).

mod venue_support;

use oa_bt::engine::{run, Engine, RunConfig, RunOutput, Step, Venue, VenueCommand};
use oa_bt::exec::{
    FillAt, OrderId, OrderRequest, OrderStatus, OrderType, QuantitySpec, RejectCode,
};
use oa_bt::market::slippage::SlippageSpec;
use oa_bt::model::{Product, Side};
use venue_support::{
    at_close, backtest, bars, config, exit, limit, market, minute, order, pause_at_close,
    run_to_end, session, slice_open, stop_market, wait_before, Act, Script, SBIN,
};

/// A session run without a venue until it waits before `slice`, with `script`.
fn waiting_before(slice: usize, script: &mut Script) -> Engine {
    let mut kernel = session(config());
    wait_before(&mut kernel, &bars(&[], &[]), script, slice);
    kernel
}

/// Run a session to its end with no more results from the venue; its result.
fn finished(mut kernel: Engine, script: &mut Script) -> RunOutput {
    run_to_end(&mut kernel, &bars(&[], &[]), script);
    kernel.finish().expect("the session finished")
}

fn submit_at(k: u32, request: OrderRequest) -> (i64, Act) {
    at_close(k, Act::Submit(request))
}

fn status(kernel: &Engine, id: u64) -> OrderStatus {
    kernel.order(OrderId(id)).expect("a known order").status
}

#[test]
fn an_order_that_passes_the_checks_is_sent_at_once() {
    let mut kernel = session(config());
    pause_at_close(&mut kernel, &bars(&[], &[]), 0);
    let stop = OrderType::StopLimit {
        trigger_rupees: 503.4,
        limit_rupees: 503.6,
    };
    let request = order(Side::Buy, 100.0, stop, Product::Mis);
    assert!(kernel.submit(request).is_ok());
    let place = VenueCommand::Place {
        order: OrderId(1),
        instrument: SBIN,
        product: Product::Mis,
        side: Side::Buy,
        units: 100.0,
        order_type: stop,
    };
    assert_eq!(kernel.take_commands(), vec![place]);
}

#[test]
fn the_commands_are_handed_over_once() {
    let mut kernel = waiting_before(
        1,
        &mut Script::new(vec![submit_at(0, market(Side::Buy, 10.0))]),
    );
    assert_eq!(kernel.take_commands().len(), 1);
    assert!(kernel.take_commands().is_empty());
}

#[test]
fn an_order_refused_by_a_submission_check_is_not_sent() {
    // 500.03 is off the 0.05 tick grid: rejected `tick`, as in a backtest.
    let mut script = Script::new(vec![submit_at(
        0,
        order(Side::Buy, 10.0, limit(500.03), Product::Mis),
    )]);
    let mut kernel = waiting_before(1, &mut script);
    assert!(kernel.take_commands().is_empty());
    assert_eq!(status(&kernel, 1), OrderStatus::Rejected);
    assert_eq!(
        kernel.order(OrderId(1)).and_then(|o| o.reject_code),
        Some(RejectCode::Tick)
    );
}

#[test]
fn a_session_fills_no_order_itself() {
    // Without a venue the MARKET order decided at minute 0 still works at minute 30.
    let mut kernel = waiting_before(
        30,
        &mut Script::new(vec![submit_at(0, market(Side::Buy, 10.0))]),
    );
    assert_eq!(status(&kernel, 1), OrderStatus::Submitted);
    assert!(kernel.position(SBIN, Product::Mis).is_none());
    assert_eq!(kernel.take_commands().len(), 1);
}

#[test]
fn a_backtest_sends_no_commands_and_a_session_names_its_venue() {
    let mut simulated = backtest(config(), bars(&[], &[]));
    run(
        &mut simulated,
        &mut Script::new(vec![submit_at(0, market(Side::Buy, 10.0))]),
    )
    .expect("the backtest completes");
    assert_eq!(simulated.venue(), Venue::Simulated);
    assert!(simulated.take_commands().is_empty());
    assert_eq!(session(config()).venue(), Venue::External);
}

#[test]
fn cancel_sends_a_cancel_and_the_order_works_until_the_venue_confirms_it() {
    let mut kernel = session(config());
    let columns = bars(&[], &[]);
    let mut script = Script::new(vec![submit_at(
        0,
        order(Side::Buy, 10.0, limit(400.0), Product::Mis),
    )]);
    wait_before(&mut kernel, &columns, &mut script, 2);
    pause_at_close(&mut kernel, &columns, 2);
    assert_eq!(kernel.cancel(OrderId(1)), Ok(OrderStatus::Submitted));
    let commands = kernel.take_commands();
    assert_eq!(
        commands.last(),
        Some(&VenueCommand::Cancel { order: OrderId(1) })
    );
    assert_eq!(status(&kernel, 1), OrderStatus::Submitted);
    wait_before(&mut kernel, &columns, &mut script, 3);
    let stamp = kernel.apply_cancel(OrderId(1)).expect("the venue confirms");
    assert_eq!((stamp.slice_ts, stamp.step), (slice_open(3), Step::Levels));
    assert_eq!(status(&kernel, 1), OrderStatus::Canceled);
}

#[test]
fn a_second_cancel_of_an_order_sends_nothing_more() {
    let mut kernel = session(config());
    let columns = bars(&[], &[]);
    let mut script = Script::new(vec![submit_at(
        0,
        order(Side::Buy, 10.0, limit(400.0), Product::Mis),
    )]);
    wait_before(&mut kernel, &columns, &mut script, 2);
    assert_eq!(kernel.take_commands().len(), 1);
    pause_at_close(&mut kernel, &columns, 2);
    for _ in 0..2 {
        assert!(kernel.cancel(OrderId(1)).is_ok());
    }
    assert_eq!(
        kernel.take_commands(),
        vec![VenueCommand::Cancel { order: OrderId(1) }]
    );
}

#[test]
fn modify_replaces_the_order_and_sends_the_replacement() {
    let replacement = order(Side::Buy, 20.0, limit(450.0), Product::Mis);
    let acts = vec![
        submit_at(0, order(Side::Buy, 10.0, limit(400.0), Product::Mis)),
        at_close(2, Act::Modify(OrderId(1), replacement)),
    ];
    let mut kernel = waiting_before(3, &mut Script::new(acts));
    let modify = VenueCommand::Modify {
        order: OrderId(1),
        units: 20.0,
        order_type: limit(450.0),
    };
    assert_eq!(kernel.take_commands().last(), Some(&modify));
    let replaced = kernel.order(OrderId(1)).expect("the order");
    assert_eq!(
        (replaced.quantity_units, replaced.request.order_type),
        (20.0, limit(450.0))
    );
}

#[test]
fn partial_fills_accumulate_until_the_order_has_its_units() {
    let mut kernel = waiting_before(
        1,
        &mut Script::new(vec![submit_at(0, market(Side::Buy, 100.0))]),
    );
    assert!(kernel
        .apply_fill(OrderId(1), 40.0, 500.1, minute(1))
        .is_ok());
    let part = kernel.order(OrderId(1)).expect("the order");
    assert_eq!(
        (part.status, part.filled_units),
        (OrderStatus::PartiallyFilled, 40.0)
    );
    assert!(kernel
        .apply_fill(OrderId(1), 60.0, 500.2, minute(1) + 1)
        .is_ok());
    let done = kernel.order(OrderId(1)).expect("the order");
    assert_eq!(
        (done.status, done.filled_units),
        (OrderStatus::Filled, 100.0)
    );
    // (40 x 500.10 + 60 x 500.20) / 100
    assert!((done.average_fill_rupees - 500.16).abs() < 1e-9);
    let held = kernel
        .position(SBIN, Product::Mis)
        .map(|p| p.quantity_units);
    assert_eq!(held, Some(100.0));
}

#[test]
fn a_fill_takes_the_venues_price_with_no_slippage() {
    let slipping = RunConfig {
        slippage: SlippageSpec::Ticks(4),
        ..config()
    };
    let mut kernel = session(slipping);
    wait_before(
        &mut kernel,
        &bars(&[], &[]),
        &mut Script::new(vec![submit_at(0, market(Side::Buy, 10.0))]),
        1,
    );
    assert!(kernel
        .apply_fill(OrderId(1), 10.0, 500.1, minute(1))
        .is_ok());
    let average = kernel
        .position(SBIN, Product::Mis)
        .map(|p| p.average_price_rupees);
    assert_eq!(average, Some(500.1));
}

#[test]
fn a_fill_at_the_slices_open_is_at_the_open_and_one_inside_it_at_a_level() {
    let acts = vec![
        submit_at(0, market(Side::Buy, 10.0)),
        submit_at(0, market(Side::Buy, 10.0)),
    ];
    let mut script = Script::new(acts);
    let mut kernel = waiting_before(1, &mut script);
    assert!(kernel
        .apply_fill(OrderId(1), 10.0, 500.1, minute(1))
        .is_ok());
    assert!(kernel
        .apply_fill(OrderId(2), 10.0, 500.2, minute(1) + 30_000_000_000)
        .is_ok());
    let output = finished(kernel, &mut script);
    let at: Vec<FillAt> = output.fills.iter().map(|record| record.fill.at).collect();
    assert_eq!(at[..2], [FillAt::Open, FillAt::Level]);
}

#[test]
fn a_fill_reported_after_its_slice_has_run_is_recorded_in_the_slice_in_progress() {
    let mut kernel = waiting_before(
        5,
        &mut Script::new(vec![submit_at(0, market(Side::Buy, 10.0))]),
    );
    let stamp = kernel
        .apply_fill(OrderId(1), 10.0, 500.1, minute(1))
        .expect("a late fill");
    assert_eq!((stamp.slice, stamp.slice_ts), (5, slice_open(5)));
}

#[test]
fn a_fill_timed_ahead_of_the_slice_in_progress_is_recorded_in_it() {
    let mut kernel = waiting_before(
        5,
        &mut Script::new(vec![submit_at(0, market(Side::Buy, 10.0))]),
    );
    let stamp = kernel
        .apply_fill(OrderId(1), 10.0, 500.1, minute(9))
        .expect("a fill");
    assert_eq!((stamp.slice, stamp.step), (5, Step::OpenFills));
}

#[test]
fn the_fill_of_a_stop_order_writes_its_trigger_first() {
    let mut script = Script::new(vec![submit_at(
        0,
        order(Side::Buy, 10.0, stop_market(501.0), Product::Mis),
    )]);
    let mut kernel = waiting_before(9, &mut script);
    assert!(kernel
        .apply_fill(OrderId(1), 10.0, 501.0, minute(9) + 1)
        .is_ok());
    let output = finished(kernel, &mut script);
    let names: Vec<String> = venue_support::events(&output)
        .into_iter()
        .filter(|(ts, _, _)| *ts == slice_open(9))
        .map(|(_, step, name)| format!("{step} {name}"))
        .collect();
    assert_eq!(names, ["3 o1 Triggered", "3 o1 Filled", "3 PositionOpened"]);
}

#[test]
fn a_rejection_from_the_venue_ends_the_order_with_its_reason() {
    let mut kernel = waiting_before(
        1,
        &mut Script::new(vec![submit_at(0, market(Side::Buy, 10.0))]),
    );
    let stamp = kernel
        .apply_reject(OrderId(1), RejectCode::Venue)
        .expect("a rejection");
    assert_eq!((stamp.slice, stamp.step), (1, Step::OpenFills));
    let rejected = kernel.order(OrderId(1)).expect("the order");
    assert_eq!(
        (rejected.status, rejected.reject_code),
        (OrderStatus::Rejected, Some(RejectCode::Venue))
    );
}

#[test]
fn a_partly_filled_order_keeps_its_fill_when_the_venue_rejects_the_rest() {
    let mut kernel = waiting_before(
        1,
        &mut Script::new(vec![submit_at(0, market(Side::Buy, 100.0))]),
    );
    assert!(kernel
        .apply_fill(OrderId(1), 40.0, 500.1, minute(1))
        .is_ok());
    assert!(kernel.apply_reject(OrderId(1), RejectCode::Venue).is_ok());
    let order = kernel.order(OrderId(1)).expect("the order");
    assert_eq!(
        (order.status, order.filled_units),
        (OrderStatus::Rejected, 40.0)
    );
    let held = kernel
        .position(SBIN, Product::Mis)
        .map(|p| p.quantity_units);
    assert_eq!(held, Some(40.0));
}

/// A session holding 100 units of SBIN in MIS from minute 1, waiting before slice 3.
fn holding_100() -> (Engine, Script) {
    let mut script = Script::new(vec![submit_at(0, market(Side::Buy, 100.0))]);
    let mut kernel = waiting_before(1, &mut script);
    assert!(kernel
        .apply_fill(OrderId(1), 100.0, 500.1, minute(1))
        .is_ok());
    wait_before(&mut kernel, &bars(&[], &[]), &mut script, 3);
    assert_eq!(kernel.take_commands().len(), 1);
    (kernel, script)
}

/// The units of the commands that place an order, in order.
fn placed_units(kernel: &mut Engine) -> Vec<(OrderId, f64)> {
    kernel
        .take_commands()
        .into_iter()
        .filter_map(|command| match command {
            VenueCommand::Place { order, units, .. } => Some((order, units)),
            _ => None,
        })
        .collect()
}

fn fraction_exit(fraction: f64) -> OrderRequest {
    OrderRequest {
        reduce_only: true,
        product: Product::Mis,
        ..OrderRequest::market(SBIN, Side::Sell, QuantitySpec::PositionFraction(fraction))
    }
}

#[test]
fn a_fraction_exit_is_sent_with_the_units_of_the_position_at_the_decision() {
    let (mut kernel, _) = holding_100();
    pause_at_close(&mut kernel, &bars(&[], &[]), 3);
    assert!(kernel.submit(fraction_exit(0.5)).is_ok());
    assert_eq!(placed_units(&mut kernel), [(OrderId(2), 50.0)]);
    assert_eq!(
        kernel.order(OrderId(2)).map(|o| o.quantity_units),
        Some(50.0)
    );
}

#[test]
fn an_exit_is_held_to_the_part_of_the_position_no_working_exit_covers() {
    let (mut kernel, _) = holding_100();
    pause_at_close(&mut kernel, &bars(&[], &[]), 3);
    assert!(kernel.submit(exit(Side::Sell, 60.0, limit(600.0))).is_ok());
    assert!(kernel.submit(fraction_exit(1.0)).is_ok());
    assert_eq!(
        placed_units(&mut kernel),
        [(OrderId(2), 60.0), (OrderId(3), 40.0)]
    );
}

#[test]
fn an_exit_with_nothing_left_to_reduce_ends_canceled_and_sends_nothing() {
    let mut kernel = session(config());
    pause_at_close(&mut kernel, &bars(&[], &[]), 0);
    assert!(kernel.submit(fraction_exit(1.0)).is_ok());
    assert!(kernel.take_commands().is_empty());
    assert_eq!(status(&kernel, 1), OrderStatus::Canceled);
}

#[test]
fn a_fill_that_takes_the_position_through_zero_closes_it_then_opens_the_other_side() {
    let (mut kernel, mut script) = holding_100();
    pause_at_close(&mut kernel, &bars(&[], &[]), 3);
    let reverse = OrderRequest {
        reverse: true,
        ..market(Side::Sell, 150.0)
    };
    assert!(kernel.submit(reverse).is_ok());
    wait_before(&mut kernel, &bars(&[], &[]), &mut script, 4);
    assert!(kernel
        .apply_fill(OrderId(2), 150.0, 500.4, minute(4))
        .is_ok());
    let output = finished(kernel, &mut script);
    let parts: Vec<(f64, &str)> = output
        .fills
        .iter()
        .filter(|record| record.fill.order == Some(OrderId(2)))
        .map(|record| (record.fill.quantity_units, record.fill.reason.code()))
        .collect();
    assert_eq!(parts, [(100.0, "signal"), (50.0, "entry")]);
}
