#![cfg(all(feature = "costs", feature = "time"))]
//! A leg's broker-held stop in a live session (`engine::venue_legs`; decision D31,
//! 16-session-design 4.1 and 4.10, requirements R2, R3 and R19).
//!
//! Every test buys SBIN (or the instrument a row names) in MIS with a stop 3 points below
//! the fill: o1 is the buy, decided at the close of minute 0 and filled at 500.10 at the
//! open of minute 1, so the stop is at 497.10 and the kernel's first stop order is o2.

mod venue_support;

use oa_bt::engine::{
    Engine, EngineExit, KernelError, OrderRole, Pause, RiskExitReason, Submission, VenueCommand,
    VenueError,
};
use oa_bt::exec::{
    FillReason, InstrumentSpec, OrderId, OrderRequest, OrderStatus, OrderType, RejectCode,
};
use oa_bt::model::{days_from_civil, Exchange, InstrumentKind, Product, Side};
use oa_bt::rms::ExitReason;
use venue_support::{
    at_second, bars, entry_waiting, exit, holding_with, limit, market, minute, modified,
    nifty_call, order, pause_at_close, placed_roles, run_to_end, sbin, stop_market, stop_points,
    Script, SBIN,
};

const ENTRY: OrderId = OrderId(1);
const STOP: OrderId = OrderId(2);
/// 500.10 less the 3-point stop.
const STOP_LEVEL: f64 = 497.1;

/// What a step does to the session: a fill, or the venue's answer or end of an order.
type Ending = fn(&mut Engine) -> Result<(), KernelError>;

/// An order placed: (order, side, units, role).
type Placed = (OrderId, Side, f64, OrderRole);

fn sl_m(trigger_rupees: f64) -> OrderType {
    OrderType::StopMarket { trigger_rupees }
}

/// An NFO future: lot 75, tick 0.05.
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

/// The (order, side, units, order type, role) of the orders `commands` place.
fn placed(commands: &[VenueCommand]) -> Vec<(OrderId, Side, f64, OrderType, OrderRole)> {
    commands
        .iter()
        .filter_map(|command| match *command {
            VenueCommand::Place {
                order,
                side,
                units,
                order_type,
                role,
                ..
            } => Some((order, side, units, order_type, role)),
            _ => None,
        })
        .collect()
}

#[test]
fn the_entry_fill_places_the_stop_for_its_units_in_the_segments_shape() {
    // 5 percent below a sell stop at 497.10 is 472.245, down to the tick 472.20.
    let rows = [
        ("cash equity", sbin(), 100.0, sl_m(STOP_LEVEL)),
        ("a future", nifty_future(), 75.0, sl_m(STOP_LEVEL)),
        (
            "an option",
            nifty_call(),
            75.0,
            OrderType::StopLimit {
                trigger_rupees: STOP_LEVEL,
                limit_rupees: 472.2,
            },
        ),
    ];
    for (what, instrument, units, shape) in rows {
        let mut kernel = entry_waiting(instrument, units, stop_points(3.0));
        kernel
            .apply_fill(ENTRY, units, 500.1, minute(1))
            .expect("the buy fills");
        let expected = vec![(STOP, Side::Sell, units, shape, OrderRole::LegStop)];
        assert_eq!(placed(&kernel.take_commands()), expected, "{what}");
        let stop = kernel.order(STOP).expect("the stop is in the book");
        assert!(stop.request.reduce_only, "{what}");
        assert_eq!(stop.request.exit_reason, Some(ExitReason::LegSl), "{what}");
    }
}

#[test]
fn each_further_fill_of_the_entry_modifies_the_stops_units_one_answer_at_a_time() {
    let mut kernel = entry_waiting(sbin(), 100.0, stop_points(3.0));
    // (fill of the buy at second s, or the venue's answer to the stop's modification;
    // the stop's modifications that follow)
    let first = kernel.apply_fill(ENTRY, 40.0, 500.1, minute(1));
    assert!(first.is_ok());
    let placed_stop = placed(&kernel.take_commands());
    assert_eq!(placed_stop[0].2, 40.0, "the stop covers the units filled");
    let steps: [(&str, Ending, Vec<f64>); 4] = [
        (
            "30 more fill",
            |k| {
                k.apply_fill(ENTRY, 30.0, 500.1, at_second(1, 10))
                    .map(|_| ())
            },
            vec![70.0],
        ),
        (
            "the last 30 fill while that modification waits",
            |k| {
                k.apply_fill(ENTRY, 30.0, 500.1, at_second(1, 20))
                    .map(|_| ())
            },
            vec![],
        ),
        (
            "the venue confirms it",
            |k| k.apply_modify(STOP).map(|_| ()),
            vec![100.0],
        ),
        ("and the next", |k| k.apply_modify(STOP).map(|_| ()), vec![]),
    ];
    for (what, step, units) in steps {
        step(&mut kernel).expect("the step is taken");
        let modifications = modified(&kernel.take_commands());
        let expected: Vec<(OrderId, f64, OrderType)> = units
            .iter()
            .map(|units| (STOP, *units, sl_m(STOP_LEVEL)))
            .collect();
        assert_eq!(modifications, expected, "{what}");
    }
}

#[test]
fn an_add_modifies_the_stops_units_to_the_position() {
    let mut kernel = holding_with(100.0, stop_points(3.0));
    kernel.take_commands();
    pause_at_close(&mut kernel, &bars(&[], &[]), 1);
    let add = OrderRequest {
        leg_rules: stop_points(3.0),
        ..market(Side::Buy, 50.0)
    };
    assert_eq!(kernel.submit(add), Ok(Submission::Submitted(OrderId(3))));
    assert_eq!(kernel.order_role(OrderId(3)), Some(OrderRole::Add));
    assert!(matches!(kernel.advance(), Ok(Pause::Waiting { .. })));
    kernel.take_commands();
    kernel
        .apply_fill(OrderId(3), 50.0, 500.1, minute(2))
        .expect("the add fills");
    let expected = vec![(STOP, 150.0, sl_m(STOP_LEVEL))];
    assert_eq!(modified(&kernel.take_commands()), expected);
}

#[test]
fn a_stop_reached_on_a_tick_is_left_to_the_broker() {
    let mut kernel = holding_with(100.0, stop_points(3.0));
    kernel.take_commands();
    kernel
        .push_tick(SBIN, 496.0, at_second(1, 5))
        .expect("the tick is taken");
    assert!(kernel.take_commands().is_empty());
    let held = kernel
        .position(SBIN, Product::Mis)
        .map(|p| p.quantity_units);
    assert_eq!(held, Some(100.0));
}

#[test]
fn the_stops_own_fill_closes_the_leg_as_a_leg_stop() {
    let mut kernel = holding_with(100.0, stop_points(3.0));
    kernel.take_commands();
    kernel
        .apply_fill(STOP, 100.0, 497.0, at_second(1, 5))
        .expect("the stop fills");
    assert!(kernel.leg(SBIN, Product::Mis).is_none());
    run_to_end(&mut kernel, &bars(&[], &[]), &mut Script::new(Vec::new()));
    let output = kernel.finish().expect("the session finished");
    let trade = &output.trades[0];
    assert_eq!(trade.exit_reason, FillReason::Exit(ExitReason::LegSl));
    assert_eq!(trade.exit_price_rupees, 497.0);
    let names: Vec<String> = venue_support::events(&output)
        .into_iter()
        .filter(|(ts, _, _)| *ts == minute(1))
        .map(|(_, _, name)| name)
        .collect();
    let stop_events: Vec<&str> = names
        .iter()
        .map(String::as_str)
        .filter(|name| name.starts_with("o2") || *name == "leg_sl_hit")
        .collect();
    assert_eq!(
        stop_events,
        ["o2 Submitted", "o2 Triggered", "leg_sl_hit", "o2 Filled"]
    );
}

#[test]
fn a_stop_that_ends_without_a_fill_while_its_leg_is_open_protects_the_leg() {
    let risk = |reason| OrderRole::RiskExit(reason);
    let rows: [(&str, Ending, Vec<Placed>); 6] = [
        (
            "rejected by the broker: exit at market",
            |k| k.apply_reject(STOP, RejectCode::Venue).map(|_| ()),
            vec![(
                OrderId(3),
                Side::Sell,
                100.0,
                risk(RiskExitReason::StopRejected),
            )],
        ),
        (
            "cancelled by the broker: placed again",
            |k| k.apply_cancel(STOP).map(|_| ()),
            vec![(OrderId(3), Side::Sell, 100.0, OrderRole::LegStop)],
        ),
        (
            "placed again and rejected: exit at market",
            |k| {
                k.apply_cancel(STOP)?;
                k.take_commands();
                k.apply_reject(OrderId(3), RejectCode::Venue).map(|_| ())
            },
            vec![(
                OrderId(4),
                Side::Sell,
                100.0,
                risk(RiskExitReason::StopRejected),
            )],
        ),
        (
            "placed again and cancelled by the broker: exit at market",
            |k| {
                k.apply_cancel(STOP)?;
                k.take_commands();
                k.apply_cancel(OrderId(3)).map(|_| ())
            },
            vec![(
                OrderId(4),
                Side::Sell,
                100.0,
                risk(RiskExitReason::Unprotected),
            )],
        ),
        (
            "cancelled on the session's request: nothing",
            |k| {
                k.cancel(STOP)?;
                k.take_commands();
                k.apply_cancel(STOP).map(|_| ())
            },
            vec![],
        ),
        (
            "rejected while the square-off waits for it: only the square-off",
            |k| {
                k.request_exit(SBIN, Product::Mis, EngineExit::SquareOff)?;
                k.take_commands();
                k.apply_reject(STOP, RejectCode::Venue).map(|_| ())
            },
            vec![(OrderId(3), Side::Sell, 100.0, OrderRole::SquareOff)],
        ),
    ];
    for (what, ending, expected) in rows {
        let mut kernel = holding_with(100.0, stop_points(3.0));
        kernel.take_commands();
        ending(&mut kernel).expect("the venue's result is taken");
        assert_eq!(placed_roles(&kernel.take_commands()), expected, "{what}");
    }
}

#[test]
fn a_re_placed_stop_has_the_leg_level_and_the_units_the_exits_leave() {
    let mut kernel = holding_with(100.0, stop_points(3.0));
    kernel.take_commands();
    kernel.apply_cancel(STOP).expect("the broker cancels it");
    let expected = vec![(
        OrderId(3),
        Side::Sell,
        100.0,
        sl_m(STOP_LEVEL),
        OrderRole::LegStop,
    )];
    assert_eq!(placed(&kernel.take_commands()), expected);
}

#[test]
fn a_stop_that_filled_its_units_while_the_leg_holds_more_exits_the_rest() {
    // The buy fills 40, the stop goes out for 40; 60 more fill and the stop's modification
    // to 100 waits when the broker fills the stop's 40: 60 are left without a stop.
    let mut kernel = entry_waiting(sbin(), 100.0, stop_points(3.0));
    assert!(kernel.apply_fill(ENTRY, 40.0, 500.1, minute(1)).is_ok());
    assert!(kernel
        .apply_fill(ENTRY, 60.0, 500.1, at_second(1, 5))
        .is_ok());
    kernel.take_commands();
    kernel
        .apply_fill(STOP, 40.0, 497.0, at_second(1, 9))
        .expect("the stop fills");
    let expected = vec![(
        OrderId(3),
        Side::Sell,
        60.0,
        OrderRole::RiskExit(RiskExitReason::StopUnfilled),
    )];
    assert_eq!(placed_roles(&kernel.take_commands()), expected);
}

/// The session holding 100 with its stop o2 resting, paused at the close of minute 1.
fn deciding_beside_the_stop() -> Engine {
    let mut kernel = holding_with(100.0, stop_points(3.0));
    kernel.take_commands();
    pause_at_close(&mut kernel, &bars(&[], &[]), 1);
    kernel
}

#[test]
fn the_stop_counts_in_the_room_on_the_positions_closing_side() {
    // A plain sale resting beside the stop could leave the position short if both filled,
    // so it is an entry; a reduce-only order for the position beside it does not fit. (A
    // plain MARKET sale goes once the stop has made way for it: venue_leg_room.rs.)
    let rows = [
        (
            "a plain resting sale of the position",
            order(Side::Sell, 100.0, limit(510.0), Product::Mis),
            Some(OrderRole::Entry),
        ),
        (
            "a plain resting sale of part of it",
            order(Side::Sell, 40.0, limit(510.0), Product::Mis),
            Some(OrderRole::Entry),
        ),
        (
            "a resting target of the position",
            exit(Side::Sell, 100.0, limit(510.0)),
            None,
        ),
        (
            "a second stop",
            exit(Side::Sell, 100.0, stop_market(495.0)),
            None,
        ),
    ];
    for (what, request, role) in rows {
        let mut kernel = deciding_beside_the_stop();
        let submitted = kernel.submit(request).expect("the order is judged");
        match role {
            Some(role) => {
                assert_eq!(submitted, Submission::Submitted(OrderId(3)), "{what}");
                assert_eq!(kernel.order_role(OrderId(3)), Some(role), "{what}");
            }
            None => assert_eq!(
                submitted,
                Submission::Rejected(OrderId(3), RejectCode::ExitCover),
                "{what}"
            ),
        }
    }
}

#[test]
fn a_stop_fill_beyond_the_position_records_the_closing_part_and_refuses_the_rest() {
    // A plain sale resting beside the stop fills first (the stop is cut down or cancelled
    // for it), then the broker fills the stop's 100 on its old terms: only what was still
    // open is recorded (16-session-design R19, reconciled with the partial record of
    // `venue_fills`).
    // (units sold, the stop's units recorded, refused, the stop afterwards)
    let rows = [
        (60.0, 40.0, 60.0, OrderStatus::Filled),
        (100.0, 0.0, 100.0, OrderStatus::Canceled),
    ];
    for (sold, recorded, refused, status) in rows {
        let mut kernel = deciding_beside_the_stop();
        let sale = OrderId(3);
        let resting = order(Side::Sell, sold, limit(500.5), Product::Mis);
        assert!(kernel.submit(resting).is_ok());
        assert!(matches!(kernel.advance(), Ok(Pause::Waiting { .. })));
        assert!(kernel.apply_fill(sale, sold, 500.5, minute(2)).is_ok());
        let result = kernel.apply_fill(STOP, 100.0, 497.0, at_second(2, 5));
        let expected = KernelError::Venue(VenueError::WouldReverse {
            order: STOP,
            excess_units: refused,
        });
        assert_eq!(result, Err(expected), "{sold} sold");
        let stop = kernel.order(STOP).expect("the stop is in the book");
        assert_eq!(
            (stop.filled_units, stop.status),
            (recorded, status),
            "{sold} sold"
        );
        let held = kernel
            .position(SBIN, Product::Mis)
            .map(|p| p.quantity_units);
        assert_eq!(held, Some(0.0), "{sold} sold");
    }
}
