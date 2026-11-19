//! A position that shrinks under the orders sent to close it (`venue_cover`): they are cut
//! down to what is left or cancelled, and a fill of one beyond the position is recorded
//! only up to it (`venue_fills`, `VenueError::WouldReverse`), a plain exit's as a
//! reduce-only order's.
//!
//! The room on a position's closing side keeps the session's own fills from shrinking it
//! under its closing orders (`closing_room`); the broker can, with a sale of its own
//! (`Engine::apply_external_fill`, `venue_test_kit::sold_at_the_broker`). Each test holds
//! 100 units of SBIN in MIS bought at 500.10 in minute 1 (`venue_test_kit::holding_100`),
//! decides at the close of minute 2, and the broker sells while the session waits before
//! slice 3.

use crate::engine::venue_test_kit::{
    held, holding_100, market, minute, sold_at_the_broker, wait_before,
};
use crate::engine::{Engine, KernelError, Submission, VenueCommand, VenueError};
use crate::exec::{OrderId, OrderRequest, OrderStatus, OrderType};
use crate::market::margin::MarginError;
use crate::model::Side;

/// The instant of the broker's sale: inside slice 3, which the session waits for.
fn in_slice_3() -> i64 {
    minute(3) + 1
}

const STOP: OrderType = OrderType::StopMarket {
    trigger_rupees: 450.0,
};
const TARGET: OrderType = OrderType::Limit {
    limit_rupees: 600.0,
};

/// A reduce-only SL-M sale of `units` at 450.00: a stop.
fn stop(units: f64) -> OrderRequest {
    OrderRequest {
        reduce_only: true,
        order_type: STOP,
        ..market(Side::Sell, units)
    }
}

/// A reduce-only LIMIT sale of `units` at 600.00: a target.
fn target(units: f64) -> OrderRequest {
    OrderRequest {
        reduce_only: true,
        order_type: TARGET,
        ..market(Side::Sell, units)
    }
}

/// A plain LIMIT sale of `units` at 600.00, not reduce-only: within the long, an exit.
fn plain_target(units: f64) -> OrderRequest {
    OrderRequest {
        order_type: TARGET,
        ..market(Side::Sell, units)
    }
}

fn accept(kernel: &mut Engine, request: OrderRequest) {
    let accepted = kernel.submit(request);
    assert!(
        matches!(accepted, Ok(Submission::Submitted(_))),
        "{accepted:?}"
    );
}

/// Holding 100 with `resting` accepted, waiting before slice 3 with no command left to
/// take; then the broker sells `sold` of the long at 500.30. The commands of that sale are
/// left for the test.
fn shrunk_by(resting: &[OrderRequest], sold: f64) -> Engine {
    let mut kernel = holding_100();
    for request in resting {
        accept(&mut kernel, request.clone());
    }
    wait_before(&mut kernel, 3);
    kernel.take_commands();
    sold_at_the_broker(&mut kernel, sold, 500.3, in_slice_3());
    kernel
}

fn status_and_units(kernel: &Engine, id: u64) -> (OrderStatus, f64, f64) {
    let order = kernel.order(OrderId(id)).expect("the order");
    (order.status, order.filled_units, order.quantity_units)
}

#[test]
fn a_resting_stop_is_cut_down_to_what_is_left_when_the_position_shrinks() {
    let mut kernel = shrunk_by(&[stop(100.0)], 40.0);
    let cut = VenueCommand::Modify {
        order: OrderId(2),
        units: 60.0,
        order_type: STOP,
    };
    assert_eq!(kernel.take_commands(), [cut]);
}

#[test]
fn the_orders_sent_to_close_a_flat_position_are_cancelled() {
    for (what, resting) in [
        ("a reduce-only target", target(100.0)),
        ("a plain sale of the long", plain_target(100.0)),
    ] {
        let mut kernel = shrunk_by(&[resting], 100.0);
        let cancel = VenueCommand::Cancel { order: OrderId(2) };
        assert_eq!(kernel.take_commands(), [cancel], "{what}");
    }
}

#[test]
fn the_later_of_two_reduce_only_orders_is_cut_first() {
    let mut kernel = shrunk_by(&[target(50.0), stop(50.0)], 40.0);
    let cut = VenueCommand::Modify {
        order: OrderId(3),
        units: 10.0,
        order_type: STOP,
    };
    assert_eq!(kernel.take_commands(), [cut]);
}

#[test]
fn a_reduce_only_order_with_nothing_left_to_fit_is_cancelled() {
    let mut kernel = shrunk_by(&[target(50.0), stop(50.0)], 60.0);
    assert_eq!(
        kernel.take_commands(),
        [
            VenueCommand::Modify {
                order: OrderId(2),
                units: 40.0,
                order_type: TARGET,
            },
            VenueCommand::Cancel { order: OrderId(3) },
        ]
    );
}

#[test]
fn a_partly_filled_reduce_only_order_that_no_longer_fits_is_cancelled() {
    let mut kernel = holding_100();
    accept(&mut kernel, stop(100.0));
    wait_before(&mut kernel, 3);
    kernel.take_commands();
    let stopped = minute(3) + 1;
    assert!(kernel.apply_fill(OrderId(2), 30.0, 450.0, stopped).is_ok());
    assert!(kernel.take_commands().is_empty(), "70 left of 70 still fit");
    sold_at_the_broker(&mut kernel, 20.0, 450.0, in_slice_3());
    let cancel = VenueCommand::Cancel { order: OrderId(2) };
    assert_eq!(kernel.take_commands(), [cancel]);
}

#[test]
fn a_cut_waits_for_the_answer_to_a_modification_already_sent() {
    let mut kernel = holding_100();
    accept(&mut kernel, stop(100.0));
    let lower = OrderRequest {
        order_type: OrderType::StopMarket {
            trigger_rupees: 445.0,
        },
        ..stop(100.0)
    };
    assert_eq!(
        kernel.modify(OrderId(2), lower.clone()),
        Ok(Submission::Submitted(OrderId(2)))
    );
    wait_before(&mut kernel, 3);
    kernel.take_commands();
    sold_at_the_broker(&mut kernel, 40.0, 500.3, in_slice_3());
    assert!(kernel.take_commands().is_empty());
    assert!(kernel.apply_modify(OrderId(2)).is_ok());
    let cut = VenueCommand::Modify {
        order: OrderId(2),
        units: 60.0,
        order_type: lower.order_type,
    };
    assert_eq!(kernel.take_commands(), [cut]);
}

#[test]
fn a_reduce_only_fill_beyond_the_position_is_recorded_up_to_it() {
    // The stop of 100 is cut to 60, but the broker fills all 100 before the cut lands.
    let mut kernel = shrunk_by(&[stop(100.0)], 40.0);
    let result = kernel.apply_fill(OrderId(2), 100.0, 450.0, minute(3) + 1);
    let refusal = VenueError::WouldReverse {
        order: OrderId(2),
        excess_units: 40.0,
    };
    assert_eq!(result, Err(KernelError::Venue(refusal)));
    assert_eq!(held(&kernel), 0.0);
    let filled = (OrderStatus::Filled, 60.0, 60.0);
    assert_eq!(status_and_units(&kernel, 2), filled);
}

#[test]
fn a_fill_of_an_order_sent_to_close_a_flat_position_records_nothing_and_ends_it() {
    // The broker sold the whole long, and then fills the resting sale before its cancel
    // lands: a plain exit opens no short, as a reduce-only one does not.
    for (what, resting) in [
        ("a reduce-only target", target(100.0)),
        ("a plain sale of the long", plain_target(100.0)),
    ] {
        let mut kernel = shrunk_by(&[resting], 100.0);
        let result = kernel.apply_fill(OrderId(2), 100.0, 600.0, minute(3) + 1);
        let refusal = VenueError::WouldReverse {
            order: OrderId(2),
            excess_units: 100.0,
        };
        assert_eq!(result, Err(KernelError::Venue(refusal)), "{what}");
        assert_eq!(held(&kernel), 0.0, "{what}");
        let ended = status_and_units(&kernel, 2);
        assert_eq!((ended.0, ended.1), (OrderStatus::Canceled, 0.0), "{what}");
    }
}

#[test]
fn a_plain_sales_fill_beyond_the_position_is_recorded_up_to_it() {
    let mut kernel = shrunk_by(&[plain_target(100.0)], 40.0);
    let result = kernel.apply_fill(OrderId(2), 100.0, 600.0, minute(3) + 1);
    let refusal = VenueError::WouldReverse {
        order: OrderId(2),
        excess_units: 40.0,
    };
    assert_eq!(result, Err(KernelError::Venue(refusal)));
    assert_eq!(held(&kernel), 0.0);
}

#[test]
fn a_fill_the_margin_rule_refuses_is_recorded_not_at_all() {
    // A reversal of the long (o2: 250 with reverse=True, 150 opening a short) passes the
    // margin rule at submission. The rule refuses it at the fill only if what it judges
    // changed since, which no call makes; here the order's leverage is raised in the book
    // to 50x, above what the margin table allows an MIS equity position. Nothing of the
    // fill is recorded, not even the 100 that close the long.
    let mut kernel = holding_100();
    let reversal = OrderRequest {
        reverse: true,
        ..market(Side::Sell, 250.0)
    };
    accept(&mut kernel, reversal);
    let booked = kernel.book.get_mut(OrderId(2)).expect("the reversal");
    booked.request.leverage = Some(50.0);
    wait_before(&mut kernel, 3);
    let result = kernel.apply_fill(OrderId(2), 250.0, 500.3, minute(3));
    assert!(
        matches!(
            result,
            Err(KernelError::Margin(MarginError::InvalidRequest { .. }))
        ),
        "{result:?}"
    );
    assert_eq!(held(&kernel), 100.0);
    let untouched = (OrderStatus::Submitted, 0.0, 250.0);
    assert_eq!(status_and_units(&kernel, 2), untouched);
}
