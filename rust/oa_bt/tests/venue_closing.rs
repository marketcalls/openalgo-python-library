#![cfg(all(feature = "costs", feature = "time"))]
//! The room on a position's closing side in a live session (`engine::venue_cover`,
//! `engine::venue_roles`): a plain order against the position is an exit only within what
//! the orders working to close it leave; beyond that it is an entry, which the entry gates
//! judge, since a fill of those orders could leave the position flat and this one would
//! then open the other side. A reduce-only order beyond the room is rejected `exit_cover`.
//!
//! "Holding 100" is 100 units of SBIN in MIS bought at 500.10 at the open of minute 1
//! (o1, `venue_support::holding_on`); the orders below are decided at the close of
//! minute 2 unless a test says otherwise. The bars rise 0.10 a minute from 500.00
//! (`venue_support::rising`).

mod venue_support;

use oa_bt::engine::{Engine, EngineExit, ExitRequest, OrderRole, RiskExitReason, Submission};
use oa_bt::exec::{InstrumentSpec, OrderId, OrderRequest, OrderType, QuantitySpec, RejectCode};
use oa_bt::model::{Product, Side};
use venue_support::{
    bars, config, exit, frozen_at, holding_on, limit, market, minute, order, pause_at_close,
    placed_roles, session, session_of, stop_market, wait_before, Script, SBIN,
};

const OPERATOR_STOP: EngineExit = EngineExit::Risk(RiskExitReason::OperatorStop);

/// Holding 100 of `instrument`, paused at the close of minute 2.
fn holding_100_of(instrument: InstrumentSpec) -> Engine {
    holding_on(session_of(config(), instrument), 100.0)
}

fn holding_100() -> Engine {
    holding_on(session(config()), 100.0)
}

/// `exit()`: a reduce-only MARKET sale of the whole MIS position.
fn whole_exit() -> OrderRequest {
    OrderRequest {
        quantity: QuantitySpec::PositionFraction(1.0),
        ..exit(Side::Sell, 1.0, OrderType::Market)
    }
}

/// A plain LIMIT sale of `units` at `limit_rupees`: not reduce-only.
fn resting_sale(units: f64, limit_rupees: f64) -> OrderRequest {
    order(Side::Sell, units, limit(limit_rupees), Product::Mis)
}

/// What `kernel` makes of `request`: the reject code, `None` when it passed.
fn code_of(kernel: &mut Engine, request: OrderRequest) -> Option<RejectCode> {
    match kernel.submit(request) {
        Ok(Submission::Submitted(_)) => None,
        Ok(Submission::Rejected(_, code)) => Some(code),
        other => panic!("expected a submission, got {other:?}"),
    }
}

/// The signed units of SBIN held in MIS.
fn held(kernel: &Engine) -> f64 {
    kernel
        .position(SBIN, Product::Mis)
        .map_or(0.0, |position| position.quantity_units)
}

#[test]
fn plain_sales_that_add_up_to_the_position_are_exits() {
    let mut kernel = holding_100();
    assert_eq!(code_of(&mut kernel, market(Side::Sell, 40.0)), None);
    assert_eq!(code_of(&mut kernel, resting_sale(60.0, 510.0)), None);
    assert_eq!(
        placed_roles(&kernel.take_commands()),
        [
            (OrderId(2), Side::Sell, 40.0, OrderRole::Exit),
            (OrderId(3), Side::Sell, 60.0, OrderRole::Exit),
        ]
    );
}

/// A plain sale submitted while holding 100, after `working` orders were accepted, with
/// entries locked for `lock` first if it names a reason.
struct LateSale {
    what: &'static str,
    lock: Option<&'static str>,
    working: Vec<OrderRequest>,
    sale: OrderRequest,
    code: Option<RejectCode>,
}

#[test]
fn a_plain_sale_beyond_what_the_orders_closing_the_long_leave_is_an_entry_the_gates_judge() {
    let in_flight = Some(RejectCode::LiveInFlight);
    let locked = Some(RejectCode::LiveLock);
    let stop = || exit(Side::Sell, 100.0, stop_market(450.0));
    let table = [
        LateSale {
            what: "two plain sales of the long",
            lock: None,
            working: vec![market(Side::Sell, 100.0)],
            sale: market(Side::Sell, 100.0),
            code: in_flight,
        },
        LateSale {
            what: "a sale beyond what an earlier one leaves",
            lock: None,
            working: vec![market(Side::Sell, 40.0)],
            sale: market(Side::Sell, 61.0),
            code: in_flight,
        },
        LateSale {
            what: "two resting sales of the long while entries are locked",
            lock: Some("feed_dead"),
            working: vec![resting_sale(100.0, 505.0)],
            sale: resting_sale(100.0, 506.0),
            code: locked,
        },
        LateSale {
            what: "a plain sale after exit() while entries are locked for a failed exit",
            lock: Some("exit_failed"),
            working: vec![whole_exit()],
            sale: market(Side::Sell, 100.0),
            code: locked,
        },
        LateSale {
            what: "a plain sale beside the strategy's own stop",
            lock: None,
            working: vec![stop()],
            sale: market(Side::Sell, 40.0),
            code: in_flight,
        },
        LateSale {
            what: "a plain sale while exit() waits for its cancels",
            lock: None,
            working: vec![stop(), whole_exit()],
            sale: market(Side::Sell, 10.0),
            code: in_flight,
        },
    ];
    for case in table {
        let what = case.what;
        let mut kernel = holding_100();
        if let Some(reason) = case.lock {
            assert_eq!(kernel.lock_entries(reason), Ok(true), "{what}");
        }
        for request in case.working {
            assert_eq!(code_of(&mut kernel, request), None, "{what}");
        }
        kernel.take_commands();
        assert_eq!(code_of(&mut kernel, case.sale), case.code, "{what}");
        assert!(placed_roles(&kernel.take_commands()).is_empty(), "{what}");
    }
}

#[test]
fn a_second_plain_sale_while_the_first_ones_fill_is_not_reported_is_held_back() {
    // sell(100) at the close of minute 2 is not reported filled by the close of minute 3,
    // where the strategy, still seeing the long, sells 100 again: held back, so the late
    // fill of the first closes the long and nothing opens a short.
    let mut kernel = holding_100();
    assert_eq!(code_of(&mut kernel, market(Side::Sell, 100.0)), None);
    kernel.take_commands();
    pause_at_close(&mut kernel, &bars(&[], &[]), 3);
    let again = kernel.submit(market(Side::Sell, 100.0));
    assert_eq!(
        again,
        Ok(Submission::Rejected(OrderId(3), RejectCode::LiveInFlight))
    );
    wait_before(
        &mut kernel,
        &bars(&[], &[]),
        &mut Script::new(Vec::new()),
        4,
    );
    assert!(kernel
        .apply_fill(OrderId(2), 100.0, 500.3, minute(3))
        .is_ok());
    assert_eq!(held(&kernel), 0.0);
}

#[test]
fn plain_sales_never_take_the_position_past_the_freeze_quantity() {
    // A freeze quantity of 100 and a long of 100: one resting sale of the 100 is its exit,
    // and the next two are held back.
    let mut kernel = holding_100_of(frozen_at(100));
    let codes: Vec<Option<RejectCode>> = [505.0, 506.0, 507.0]
        .into_iter()
        .map(|limit_rupees| code_of(&mut kernel, resting_sale(100.0, limit_rupees)))
        .collect();
    assert_eq!(
        codes,
        [
            None,
            Some(RejectCode::LiveInFlight),
            Some(RejectCode::LiveInFlight)
        ]
    );
    wait_before(
        &mut kernel,
        &bars(&[], &[]),
        &mut Script::new(Vec::new()),
        3,
    );
    assert!(kernel
        .apply_fill(OrderId(2), 100.0, 505.0, minute(3) + 1)
        .is_ok());
    assert_eq!(held(&kernel), 0.0);
}

#[test]
fn during_a_safe_stop_the_strategy_cannot_sell_past_the_sessions_exit() {
    // Entries locked for operator_stop, the session's exit of the 100 sent and not filled
    // yet: at the next bar close the strategy's sell(100) is an entry, locked out.
    let mut kernel = holding_100();
    wait_before(
        &mut kernel,
        &bars(&[], &[]),
        &mut Script::new(Vec::new()),
        3,
    );
    assert_eq!(kernel.lock_entries("operator_stop"), Ok(true));
    assert_eq!(
        kernel.request_exit(SBIN, Product::Mis, OPERATOR_STOP),
        Ok(ExitRequest::Sent)
    );
    kernel.take_commands();
    pause_at_close(&mut kernel, &bars(&[], &[]), 3);
    let sale = kernel.submit(market(Side::Sell, 100.0));
    assert_eq!(
        sale,
        Ok(Submission::Rejected(OrderId(3), RejectCode::LiveLock))
    );
    assert!(kernel.take_commands().is_empty());
}

#[test]
fn a_reduce_only_order_beyond_what_a_plain_sale_of_the_long_leaves_is_rejected_exit_cover() {
    let table = [
        (
            "a stop beside a sale of the whole long",
            100.0,
            100.0,
            Some(RejectCode::ExitCover),
        ),
        (
            "a stop beyond what a partial sale leaves",
            60.0,
            50.0,
            Some(RejectCode::ExitCover),
        ),
        ("a stop of what a partial sale leaves", 60.0, 40.0, None),
    ];
    for (what, sold, stop_units, code) in table {
        let mut kernel = holding_100();
        assert_eq!(
            code_of(&mut kernel, resting_sale(sold, 520.0)),
            None,
            "{what}"
        );
        let stop = exit(Side::Sell, stop_units, stop_market(450.0));
        assert_eq!(code_of(&mut kernel, stop), code, "{what}");
    }
}
