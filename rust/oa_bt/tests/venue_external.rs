#![cfg(all(feature = "costs", feature = "time"))]
//! A live session's fill with no order of its own (`Engine::apply_external_fill`,
//! `engine::venue_external`; 16-session-design R15): the broker closed or reduced the
//! position by itself. It is recorded through the one fill path, flagged "external", and
//! the orders working to close the position follow it; one that would not only close or
//! reduce is refused whole.
//!
//! Each test holds 100 units of SBIN bought by o1 at 500.10 at the open of minute 1, with
//! a stop 2 points below it (o2, at 498.10) placed, while the session waits
//! before minute 1.

mod venue_support;

use oa_bt::engine::{Engine, EventStamp, KernelError, OrderRole, Step, VenueCommand, VenueError};
use oa_bt::exec::{FillAt, FillFlags, FillReason, InstrumentId, OrderId};
use oa_bt::model::{Product, Side};
use oa_bt::portfolio::PositionKey;
use oa_bt::rms::ExitReason;
use venue_support::{
    backtest, bars, config, holding_with, minute, placed_roles, stop_points, SBIN, SECOND,
};

/// A row of the refusal table: (what, product, side, units, price, the refusal).
type Row = (
    &'static str,
    Product,
    Side,
    f64,
    f64,
    Result<EventStamp, KernelError>,
);

/// The broker's own fill: 10 s into minute 1.
fn broker_time() -> i64 {
    minute(1) + 10 * SECOND
}

/// Holding 100 with its stop o2 placed (module docs), the commands taken.
fn holding() -> Engine {
    let mut kernel = holding_with(100.0, stop_points(2.0));
    let stop = placed_roles(&kernel.take_commands());
    assert_eq!(stop, [(OrderId(2), Side::Sell, 100.0, OrderRole::LegStop)]);
    kernel
}

fn held(kernel: &Engine) -> f64 {
    kernel
        .position(SBIN, Product::Mis)
        .map_or(0.0, |position| position.quantity_units)
}

/// The broker sells `units` of the long at 499.00, for `reason`.
fn broker_sells(
    kernel: &mut Engine,
    units: f64,
    reason: ExitReason,
) -> Result<EventStamp, KernelError> {
    kernel.apply_external_fill(
        SBIN,
        Product::Mis,
        Side::Sell,
        units,
        499.0,
        broker_time(),
        reason,
    )
}

#[test]
fn a_broker_sale_of_part_of_the_long_is_recorded_and_the_stop_cut_to_what_is_left() {
    let mut kernel = holding();
    let stamp = broker_sells(&mut kernel, 40.0, ExitReason::Manual).expect("recorded");
    assert_eq!((stamp.slice, stamp.step), (1, Step::Levels));
    assert_eq!(held(&kernel), 60.0);
    let leg = kernel.leg(SBIN, Product::Mis).expect("the leg is open");
    assert_eq!(leg.held_units(), 60.0);
    let cut = VenueCommand::Modify {
        order: OrderId(2),
        units: 60.0,
        order_type: oa_bt::exec::OrderType::StopMarket {
            trigger_rupees: 498.1,
        },
    };
    assert_eq!(kernel.take_commands(), [cut]);
    let account = kernel.account();
    assert_eq!(
        account.realized_rupees, -44.0,
        "40 units at 499.00 against 500.10"
    );
}

#[test]
fn a_broker_sale_of_the_whole_long_closes_the_trade_for_its_reason_and_cancels_the_stop() {
    let mut kernel = holding();
    broker_sells(&mut kernel, 100.0, ExitReason::Scheduler).expect("recorded");
    assert_eq!(held(&kernel), 0.0);
    assert!(kernel.leg(SBIN, Product::Mis).is_none(), "the leg closed");
    let cancel = VenueCommand::Cancel { order: OrderId(2) };
    assert_eq!(kernel.take_commands(), [cancel]);
    kernel
        .apply_cancel(OrderId(2))
        .expect("the venue cancels the stop");
    kernel
        .stop_session()
        .expect("the session stops where it stands");
    let output = kernel.finish().expect("the session is over");
    let sale = output.fills.last().expect("the broker's sale");
    let fill = sale.fill;
    assert_eq!(fill.order, None);
    assert_eq!(fill.flags, FillFlags::EXTERNAL);
    assert_eq!(fill.flags.names(), ["external"]);
    assert_eq!((fill.at, fill.price_rupees), (FillAt::Level, 499.0));
    assert_eq!(fill.reason, FillReason::Exit(ExitReason::Scheduler));
    let trade = output.trades.last().expect("the closed trade");
    assert_eq!(trade.exit_reason, FillReason::Exit(ExitReason::Scheduler));
    assert_eq!(trade.exit_price_rupees, 499.0);
}

#[test]
fn a_broker_fill_at_the_slice_open_is_recorded_at_the_open() {
    let mut kernel = holding();
    let at_open = minute(1);
    let reason = ExitReason::Manual;
    let sold =
        kernel.apply_external_fill(SBIN, Product::Mis, Side::Sell, 10.0, 500.1, at_open, reason);
    assert!(sold.is_ok(), "{sold:?}");
    kernel.stop_session().expect("the session stops");
    let output = kernel.finish().expect("the session is over");
    assert_eq!(output.fills.last().map(|f| f.fill.at), Some(FillAt::Open));
}

#[test]
fn a_fill_that_would_not_only_close_or_reduce_the_position_is_refused_whole() {
    let key = PositionKey {
        instrument: SBIN,
        product: Product::Mis,
    };
    let beyond = |side: Side, units: f64, held_units: f64| {
        Err(KernelError::Venue(VenueError::BeyondPosition {
            key,
            side,
            units,
            held_units,
        }))
    };
    let cnc_key = PositionKey {
        instrument: SBIN,
        product: Product::Cnc,
    };
    // (what, product, side, units, price, the refusal)
    let table: [Row; 7] = [
        (
            "a buy on the long",
            Product::Mis,
            Side::Buy,
            10.0,
            499.0,
            beyond(Side::Buy, 10.0, 100.0),
        ),
        (
            "through zero",
            Product::Mis,
            Side::Sell,
            101.0,
            499.0,
            beyond(Side::Sell, 101.0, 100.0),
        ),
        (
            "of a flat position",
            Product::Cnc,
            Side::Sell,
            10.0,
            499.0,
            Err(KernelError::Venue(VenueError::BeyondPosition {
                key: cnc_key,
                side: Side::Sell,
                units: 10.0,
                held_units: 0.0,
            })),
        ),
        (
            "no units",
            Product::Mis,
            Side::Sell,
            0.0,
            499.0,
            Err(invalid()),
        ),
        (
            "units that are not a number",
            Product::Mis,
            Side::Sell,
            f64::NAN,
            499.0,
            Err(invalid()),
        ),
        (
            "no price",
            Product::Mis,
            Side::Sell,
            10.0,
            0.0,
            Err(invalid()),
        ),
        (
            "a price that is not finite",
            Product::Mis,
            Side::Sell,
            10.0,
            f64::INFINITY,
            Err(invalid()),
        ),
    ];
    for (what, product, side, units, price, refusal) in table {
        let mut kernel = holding();
        let result = kernel.apply_external_fill(
            SBIN,
            product,
            side,
            units,
            price,
            broker_time(),
            ExitReason::Manual,
        );
        let same = match (&result, &refusal) {
            (
                Err(KernelError::Venue(VenueError::InvalidFill { .. })),
                Err(KernelError::Venue(VenueError::InvalidFill { .. })),
            ) => true,
            _ => result == refusal,
        };
        assert!(same, "{what}: {result:?}");
        assert_eq!(held(&kernel), 100.0, "{what}: nothing recorded");
        assert!(kernel.take_commands().is_empty(), "{what}: nothing sent");
    }
}

fn invalid() -> KernelError {
    KernelError::Venue(VenueError::InvalidFill {
        reason: String::new(),
    })
}

#[test]
fn a_fill_with_no_order_is_refused_when_no_result_can_be_recorded() {
    let sell = |kernel: &mut Engine| broker_sells(kernel, 10.0, ExitReason::Manual);
    let mut at_a_bar_close = holding();
    venue_support::pause_at_close(&mut at_a_bar_close, &bars(&[], &[]), 2);
    assert_eq!(
        sell(&mut at_a_bar_close),
        Err(VenueError::AtBarClose.into())
    );
    let mut a_backtest = backtest(config(), bars(&[], &[]));
    assert_eq!(sell(&mut a_backtest), Err(VenueError::NotExternal.into()));
    let mut over = holding();
    over.stop_session().expect("the session stops");
    assert_eq!(sell(&mut over), Err(VenueError::SessionOver.into()));
    let mut unknown = holding();
    let stranger = unknown.apply_external_fill(
        InstrumentId(7),
        Product::Mis,
        Side::Sell,
        10.0,
        499.0,
        broker_time(),
        ExitReason::Manual,
    );
    assert!(
        matches!(
            stranger,
            Err(KernelError::InvalidConfig {
                field: "symbol",
                ..
            })
        ),
        "{stranger:?}"
    );
}
