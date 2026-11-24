#![cfg(all(feature = "costs", feature = "time"))]
//! A leg's broker-held stop and the other orders on its position's closing side in a live
//! session (`engine::venue_legs`, `venue_stop_terms`, `venue_owner`, `venue_gates`;
//! 16-session-design 4.5, requirements R2, R4 and R19): the stop never works beyond the
//! room the orders sent to close the position leave, and takes that room from the resting
//! ones; a plain MARKET order against the position goes out after the stop is cut down or
//! cancelled for it; and a stop left working by a closed leg keeps the next entry out.
//!
//! Every test buys 100 SBIN in MIS (o1, decided at the close of minute 0 and filled at
//! 500.10 at the open of minute 1). With `stop_points(3.0)` the stop is o2 at 497.10.

mod venue_support;

use oa_bt::engine::{
    Engine, EngineExit, ExitRequest, OrderRole, Pause, Protect, RiskExitReason, Submission,
    VenueCommand,
};
use oa_bt::exec::{OrderId, OrderRequest, OrderType, RejectCode};
use oa_bt::model::{Product, Side};
use oa_bt::rms::{Distance, LegRules, TrailSpec};
use venue_support::{
    at_second, bars, exit, holding_with, limit, market, modified, order, pause_at_close,
    placed_roles, stop_points, SBIN,
};

const STOP: OrderId = OrderId(2);
const STOP_LEVEL: f64 = 497.1;

fn sl_m(trigger_rupees: f64) -> OrderType {
    OrderType::StopMarket { trigger_rupees }
}

fn waiting(kernel: &mut Engine) {
    assert!(matches!(kernel.advance(), Ok(Pause::Waiting { .. })));
}

/// A trail 2 points behind the best price, armed once the price is 2 points in favour; no
/// stop rule, so the leg has no stop level until it arms.
fn trail_only() -> LegRules {
    LegRules {
        trail: Some(TrailSpec {
            trigger: Distance::Points(2.0),
            step: None,
        }),
        ..LegRules::default()
    }
}

/// Holding 100 with `rules`, with `resting` (o2 when the leg has no stop yet) placed at
/// the close of minute 1, waiting before minute 2 with every command taken.
fn holding_beside(rules: LegRules, resting: OrderRequest) -> Engine {
    let mut kernel = holding_with(100.0, rules);
    kernel.take_commands();
    pause_at_close(&mut kernel, &bars(&[], &[]), 1);
    assert!(matches!(
        kernel.submit(resting),
        Ok(Submission::Submitted(_))
    ));
    waiting(&mut kernel);
    kernel.take_commands();
    kernel
}

/// The stop a trail armed at 502.20 places: 2 points below, at 500.20.
const TRAILED: f64 = 500.2;

fn place_stop(order: u64, units: f64) -> VenueCommand {
    VenueCommand::Place {
        order: OrderId(order),
        instrument: SBIN,
        product: Product::Mis,
        side: Side::Sell,
        units,
        order_type: sl_m(TRAILED),
        role: OrderRole::LegStop,
    }
}

#[test]
fn a_stop_that_arms_beside_a_resting_order_sent_to_close_takes_its_room_first() {
    // (what, the resting order o2, the commands when the trail arms, after o2's cancel)
    let plain_sale = order(Side::Sell, 100.0, limit(520.0), Product::Mis);
    let rows = [
        (
            "a plain sale of the whole position",
            plain_sale,
            vec![VenueCommand::Cancel { order: OrderId(2) }],
            vec![place_stop(3, 100.0)],
        ),
        (
            "a reduce-only sale of part of it",
            exit(Side::Sell, 40.0, limit(520.0)),
            vec![
                VenueCommand::Cancel { order: OrderId(2) },
                place_stop(3, 60.0),
            ],
            vec![VenueCommand::Modify {
                order: OrderId(3),
                units: 100.0,
                order_type: sl_m(TRAILED),
            }],
        ),
    ];
    for (what, resting, armed, after_cancel) in rows {
        let mut kernel = holding_beside(trail_only(), resting);
        assert_eq!(
            kernel.order_role(OrderId(2)),
            Some(OrderRole::Exit),
            "{what}"
        );
        kernel
            .push_tick(SBIN, 502.2, at_second(2, 5))
            .expect("the tick is taken");
        assert_eq!(kernel.take_commands(), armed, "{what}: the trail arms");
        kernel
            .apply_cancel(OrderId(2))
            .expect("the venue cancels the resting order");
        assert_eq!(kernel.take_commands(), after_cancel, "{what}: its cancel");
        if modified(&after_cancel).is_empty() {
            continue;
        }
        kernel
            .apply_modify(OrderId(3))
            .expect("the venue confirms the stop's units");
        assert!(
            kernel.take_commands().is_empty(),
            "{what}: nothing is modified again"
        );
    }
}

#[test]
fn a_stop_kept_within_its_room_is_not_modified_again_at_the_next_open() {
    let resting = exit(Side::Sell, 40.0, limit(520.0));
    let mut kernel = holding_beside(trail_only(), resting);
    kernel
        .push_tick(SBIN, 502.2, at_second(2, 5))
        .expect("the tick is taken");
    kernel.take_commands();
    kernel
        .apply_cancel(OrderId(2))
        .expect("the venue cancels the resting order");
    kernel.take_commands();
    kernel
        .apply_modify(OrderId(3))
        .expect("the venue confirms the stop's units");
    pause_at_close(&mut kernel, &bars(&[], &[]), 2);
    waiting(&mut kernel);
    assert!(modified(&kernel.take_commands()).is_empty());
}

#[test]
fn protect_takes_the_room_of_a_resting_order_sent_to_close_before_it_places_the_stop() {
    // The kill switch's exit cancels the stop and the venue rejects it; the strategy then
    // rests a reduce-only sale of the position (o4) where no stop works.
    let mut kernel = holding_with(100.0, stop_points(3.0));
    kernel.take_commands();
    let kill = EngineExit::Risk(RiskExitReason::KillSwitch);
    let asked = kernel.request_exit(SBIN, Product::Mis, kill);
    assert_eq!(asked, Ok(ExitRequest::WaitsForCancels));
    assert!(kernel.apply_cancel(STOP).is_ok());
    assert!(kernel.apply_reject(OrderId(3), RejectCode::Venue).is_ok());
    kernel.take_commands();
    pause_at_close(&mut kernel, &bars(&[], &[]), 1);
    let target = exit(Side::Sell, 100.0, limit(520.0));
    assert_eq!(kernel.submit(target), Ok(Submission::Submitted(OrderId(4))));
    waiting(&mut kernel);
    kernel.take_commands();
    let outcome = kernel.protect(SBIN, Product::Mis).map(Protect::code);
    assert_eq!(outcome, Ok("waits_for_cancels"));
    assert_eq!(
        kernel.take_commands(),
        [VenueCommand::Cancel { order: OrderId(4) }]
    );
    kernel
        .apply_cancel(OrderId(4))
        .expect("the venue cancels the sale");
    let expected = vec![(OrderId(5), Side::Sell, 100.0, OrderRole::LegStop)];
    assert_eq!(placed_roles(&kernel.take_commands()), expected);
    let stop = kernel.order(OrderId(5)).expect("the new stop");
    assert_eq!(stop.request.order_type, sl_m(STOP_LEVEL));
}

/// Holding 100 with its stop o2 resting, paused at the close of minute 1.
fn deciding_beside_the_stop() -> Engine {
    let mut kernel = holding_with(100.0, stop_points(3.0));
    kernel.take_commands();
    pause_at_close(&mut kernel, &bars(&[], &[]), 1);
    kernel
}

/// What a plain MARKET order (o3) against the long sends beside the stop: at submission,
/// and once the venue has answered what that sent.
struct PlainBesideTheStop {
    what: &'static str,
    request: OrderRequest,
    role: OrderRole,
    first: VenueCommand,
    then: Vec<(OrderId, Side, f64, OrderRole)>,
}

#[test]
fn a_plain_market_order_against_the_position_goes_out_after_the_stop_is_cut_or_cancelled() {
    let reversal = OrderRequest {
        reverse: true,
        ..market(Side::Sell, 200.0)
    };
    let rows = [
        PlainBesideTheStop {
            what: "a plain sale of the whole position: the stop is cancelled first",
            request: market(Side::Sell, 100.0),
            role: OrderRole::Exit,
            first: VenueCommand::Cancel { order: STOP },
            then: vec![(OrderId(3), Side::Sell, 100.0, OrderRole::Exit)],
        },
        PlainBesideTheStop {
            what: "a plain sale of part of it: the stop is cut down to the rest first",
            request: market(Side::Sell, 40.0),
            role: OrderRole::Exit,
            first: VenueCommand::Modify {
                order: STOP,
                units: 60.0,
                order_type: sl_m(STOP_LEVEL),
            },
            then: vec![(OrderId(3), Side::Sell, 40.0, OrderRole::Exit)],
        },
        PlainBesideTheStop {
            what: "a reversal: the stop is cancelled first, then it closes and opens",
            request: reversal,
            role: OrderRole::Entry,
            first: VenueCommand::Cancel { order: STOP },
            then: vec![(OrderId(3), Side::Sell, 200.0, OrderRole::Entry)],
        },
    ];
    for row in rows {
        let what = row.what;
        let mut kernel = deciding_beside_the_stop();
        assert_eq!(
            kernel.submit(row.request),
            Ok(Submission::Submitted(OrderId(3))),
            "{what}"
        );
        assert_eq!(kernel.order_role(OrderId(3)), Some(row.role), "{what}");
        assert_eq!(kernel.take_commands(), [row.first], "{what}: first");
        waiting(&mut kernel);
        kernel.take_commands();
        let answered = match row.first {
            VenueCommand::Cancel { order } => kernel.apply_cancel(order),
            VenueCommand::Modify { order, .. } => kernel.apply_modify(order),
            VenueCommand::Place { order, .. } => panic!("{what}: {order} placed first"),
            other => panic!("{what}: an MIS leg sends no GTT command: {other:?}"),
        };
        assert!(answered.is_ok(), "{what}");
        assert_eq!(
            placed_roles(&kernel.take_commands()),
            row.then,
            "{what}: then"
        );
    }
}

#[test]
fn a_plain_sale_closes_the_position_beside_its_stop_while_entries_are_locked() {
    let mut kernel = deciding_beside_the_stop();
    assert_eq!(kernel.lock_entries("feed_dead"), Ok(true));
    assert_eq!(
        kernel.submit(market(Side::Sell, 100.0)),
        Ok(Submission::Submitted(OrderId(3)))
    );
    assert_eq!(
        kernel.take_commands(),
        [VenueCommand::Cancel { order: STOP }]
    );
}

#[test]
fn a_resting_order_beside_the_stop_is_not_modified_into_a_market_one() {
    // A plain LIMIT sale beside the stop is an entry (it rests beyond the room the stop
    // leaves); at market it would fill beside the stop.
    let mut kernel = deciding_beside_the_stop();
    let resting = order(Side::Sell, 100.0, limit(520.0), Product::Mis);
    assert_eq!(
        kernel.submit(resting),
        Ok(Submission::Submitted(OrderId(3)))
    );
    kernel.take_commands();
    pause_at_close(&mut kernel, &bars(&[], &[]), 2);
    let refused = kernel.modify(OrderId(3), market(Side::Sell, 100.0));
    assert!(refused.is_err(), "{refused:?}");
    assert!(modified(&kernel.take_commands()).is_empty());
}

#[test]
fn a_closed_legs_stop_still_working_keeps_the_next_entry_out() {
    // A resting plain sale (o3) closes the long beside the stop; the venue refuses the
    // stop's cancel, so o2 works on with no leg left to hold it.
    let mut kernel = deciding_beside_the_stop();
    let resting = order(Side::Sell, 100.0, limit(500.5), Product::Mis);
    assert_eq!(
        kernel.submit(resting),
        Ok(Submission::Submitted(OrderId(3)))
    );
    waiting(&mut kernel);
    kernel.take_commands();
    kernel
        .apply_fill(OrderId(3), 100.0, 500.5, at_second(2, 5))
        .expect("the sale fills");
    assert_eq!(
        kernel.take_commands(),
        [VenueCommand::Cancel { order: STOP }]
    );
    kernel
        .apply_cancel_refused(STOP)
        .expect("the venue refuses the cancel");
    pause_at_close(&mut kernel, &bars(&[], &[]), 2);
    let entry = OrderRequest {
        leg_rules: stop_points(5.0),
        ..market(Side::Buy, 100.0)
    };
    assert_eq!(
        kernel.submit(entry),
        Ok(Submission::Rejected(OrderId(4), RejectCode::LiveInFlight))
    );
    assert!(kernel.order(STOP).is_some_and(|o| o.status.is_working()));
    assert_eq!(
        kernel
            .position(SBIN, Product::Mis)
            .map(|p| p.quantity_units),
        Some(0.0)
    );
}
