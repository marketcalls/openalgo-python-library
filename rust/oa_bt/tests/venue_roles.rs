#![cfg(all(feature = "costs", feature = "time"))]
//! What each order of a live session is for (`engine::venue_roles`, requirement R1 of the
//! session design): every `VenueCommand::Place` carries its order's role, and
//! `Engine::order_role` reports it.
//!
//! The test day is 2025-07-01 on 1-minute slices; the square-off is at 15:10 (slice 355).
//! The bars rise 0.10 a minute from 500.00 (`venue_support::rising`).

mod venue_support;

use oa_bt::engine::{
    run, BarColumns, Engine, EngineExit, ExitRequest, KernelError, OrderRole, RiskExitReason,
    RunConfig, Submission, VenueCommand,
};
use oa_bt::exec::{ExecError, OrderId, OrderRequest, OrderType, QuantitySpec};
use oa_bt::model::{Product, Side};
use oa_bt::rms::{ExitReason, StrategyRiskConfig};
use venue_support::{
    at_close, backtest, bars, config, echo_until, exit, holding_on, limit, market, order,
    pause_at_close, placed_roles, session, stop_market, wait_before, Act, Echo, Ohlc, Script, SBIN,
};

/// The slice that opens at the square-off, 15:10.
const SQUARE_OFF_SLICE: usize = 355;
/// A crash at minute 10: open 494.00, close 494.05, a loss of about 605 on 100 units
/// bought at 500.10.
const CRASH: (u32, Ohlc) = (10, (494.0, 494.5, 493.5, 494.05));

/// `exit()`: a reduce-only MARKET sale of the whole MIS position.
fn whole_exit() -> OrderRequest {
    OrderRequest {
        quantity: QuantitySpec::PositionFraction(1.0),
        ..exit(Side::Sell, 1.0, OrderType::Market)
    }
}

/// A session paused at the close of minute 2, holding `held` units bought in o1 (flat
/// with no order for 0).
fn paused_holding(held: f64) -> Engine {
    if held > 0.0 {
        return holding_on(session(config()), held);
    }
    let mut kernel = session(config());
    pause_at_close(&mut kernel, &bars(&[], &[]), 2);
    kernel
}

/// The role `request`'s Place carries and the one `order_role` reports, for a session
/// holding `held` units that submits it at the close of minute 2.
fn role_of(held: f64, request: OrderRequest) -> (OrderRole, Option<OrderRole>) {
    let mut kernel = paused_holding(held);
    let Ok(Submission::Submitted(id)) = kernel.submit(request) else {
        panic!("the order passes the checks");
    };
    let placed = placed_roles(&kernel.take_commands());
    let carried = placed
        .iter()
        .find(|(order, ..)| *order == id)
        .map(|(.., role)| *role)
        .expect("the order is placed");
    (carried, kernel.order_role(id))
}

#[test]
fn a_strategy_order_carries_the_role_the_position_gives_it() {
    let reversal = OrderRequest {
        reverse: true,
        ..market(Side::Sell, 160.0)
    };
    let table = [
        (
            "a buy from flat",
            0.0,
            market(Side::Buy, 10.0),
            OrderRole::Entry,
        ),
        (
            "a sale from flat",
            0.0,
            market(Side::Sell, 10.0),
            OrderRole::Entry,
        ),
        (
            "a buy while long",
            100.0,
            market(Side::Buy, 10.0),
            OrderRole::Add,
        ),
        (
            "a plain sale within the long",
            100.0,
            market(Side::Sell, 40.0),
            OrderRole::Exit,
        ),
        (
            "a reduce-only target",
            100.0,
            exit(Side::Sell, 100.0, limit(600.0)),
            OrderRole::Exit,
        ),
        (
            "a reduce-only stop",
            100.0,
            exit(Side::Sell, 100.0, stop_market(450.0)),
            OrderRole::Exit,
        ),
        ("exit()", 100.0, whole_exit(), OrderRole::Exit),
        ("a reversal of the long", 100.0, reversal, OrderRole::Entry),
    ];
    for (what, held, request, role) in table {
        assert_eq!(role_of(held, request), (role, Some(role)), "{what}");
    }
}

/// Holding 100 with a plain LIMIT sale of the 100 at 520.00 (o2, an exit), paused at the
/// close of minute 2 with no command left to take.
fn holding_with_a_plain_target() -> Engine {
    let mut kernel = paused_holding(100.0);
    let sale = order(Side::Sell, 100.0, limit(520.0), Product::Mis);
    assert_eq!(kernel.submit(sale), Ok(Submission::Submitted(OrderId(2))));
    kernel.take_commands();
    kernel
}

#[test]
fn a_modification_that_would_change_an_orders_role_is_refused() {
    // Into a reversal: 250 with reverse=True would open a short of 150.
    let mut kernel = holding_with_a_plain_target();
    let reversal = OrderRequest {
        reverse: true,
        ..order(Side::Sell, 250.0, limit(520.0), Product::Mis)
    };
    let refused = kernel.modify(OrderId(2), reversal);
    assert!(
        matches!(
            refused,
            Err(KernelError::Exec(ExecError::InvalidRequest { .. }))
        ),
        "{refused:?}"
    );
    assert!(kernel.take_commands().is_empty(), "nothing is sent");
    assert_eq!(kernel.order_role(OrderId(2)), Some(OrderRole::Exit));
    let order = kernel.order(OrderId(2)).expect("the sale");
    assert_eq!(
        (order.quantity_units, order.request.reverse),
        (100.0, false)
    );
}

#[test]
fn a_modification_that_keeps_an_orders_role_is_sent() {
    let mut kernel = holding_with_a_plain_target();
    let lower = order(Side::Sell, 100.0, limit(515.0), Product::Mis);
    assert_eq!(
        kernel.modify(OrderId(2), lower),
        Ok(Submission::Submitted(OrderId(2)))
    );
    let modify = VenueCommand::Modify {
        order: OrderId(2),
        units: 100.0,
        order_type: limit(515.0),
    };
    assert_eq!(kernel.take_commands(), [modify]);
    assert_eq!(kernel.order_role(OrderId(2)), Some(OrderRole::Exit));
}

#[test]
fn an_order_never_accepted_has_no_role() {
    // 500.03 is off the 0.05 tick grid: rejected `tick` at submission, never sent.
    let mut kernel = paused_holding(0.0);
    let off_tick = OrderRequest {
        order_type: limit(500.03),
        ..market(Side::Buy, 10.0)
    };
    let Ok(Submission::Rejected(id, _)) = kernel.submit(off_tick) else {
        panic!("the order is rejected");
    };
    assert_eq!(kernel.order_role(id), None);
    assert_eq!(kernel.order_role(OrderId(99)), None);
}

#[test]
fn a_backtests_orders_have_no_role() {
    let mut simulated = backtest(config(), bars(&[], &[]));
    let acts = vec![at_close(0, Act::Submit(market(Side::Buy, 10.0)))];
    simulated.hold();
    simulated.release(2);
    run(&mut simulated, &mut Script::new(acts)).expect("the released slices run");
    assert!(simulated.order(OrderId(1)).is_some());
    assert_eq!(simulated.order_role(OrderId(1)), None);
}

/// The role and the trade's exit reason of the engine's first order (o2) after the 100
/// bought at the open, echoed on `columns` until the session waits before `slice`.
fn engine_exit_of(
    config: RunConfig,
    columns: &BarColumns,
    slice: usize,
) -> (OrderRole, Option<ExitReason>) {
    let mut kernel = session(config);
    let acts = vec![at_close(0, Act::Submit(market(Side::Buy, 100.0)))];
    let mut echo = Echo::default();
    echo_until(
        &mut kernel,
        columns,
        &mut Script::new(acts),
        &mut echo,
        slice,
    );
    let placed = placed_roles(&kernel.take_commands());
    let Some(&(order, _, _, role)) = placed.first() else {
        panic!("the engine's exit is placed before slice {slice}");
    };
    assert_eq!(order, OrderId(2));
    let exit_reason = kernel.order(order).and_then(|o| o.request.exit_reason);
    (role, exit_reason)
}

fn with_risk(risk: StrategyRiskConfig) -> RunConfig {
    RunConfig { risk, ..config() }
}

#[test]
fn each_engine_exit_carries_its_role_and_the_backtests_exit_reason() {
    let plain = bars(&[], &[]);
    let crash = bars(&[CRASH], &[]);
    let daily_loss = with_risk(StrategyRiskConfig {
        max_daily_loss_rupees: Some(500.0),
        ..StrategyRiskConfig::default()
    });
    // A 0.005 percent drawdown of the 1 crore is 500: the crash takes more.
    let drawdown = with_risk(StrategyRiskConfig {
        drawdown_halt_percent: Some(0.005),
        ..StrategyRiskConfig::default()
    });
    let held_five = with_risk(StrategyRiskConfig {
        max_hold_bars: Some(5),
        ..StrategyRiskConfig::default()
    });
    let table = [
        (
            "the square-off",
            config(),
            &plain,
            SQUARE_OFF_SLICE,
            OrderRole::SquareOff,
            ExitReason::Scheduler,
        ),
        (
            "the daily loss limit",
            daily_loss,
            &crash,
            11,
            OrderRole::RiskExit(RiskExitReason::DailyLoss),
            ExitReason::DailyLossLimit,
        ),
        (
            "the drawdown halt",
            drawdown,
            &crash,
            11,
            OrderRole::RiskExit(RiskExitReason::Drawdown),
            ExitReason::DrawdownHalt,
        ),
        (
            "the time exit",
            held_five,
            &plain,
            6,
            OrderRole::RiskExit(RiskExitReason::TimeExit),
            ExitReason::TimeExit,
        ),
    ];
    for (what, config, columns, slice, role, reason) in table {
        assert_eq!(
            engine_exit_of(config, columns, slice),
            (role, Some(reason)),
            "{what}"
        );
    }
}

#[test]
fn an_exit_the_session_asks_for_carries_its_reason() {
    let table = [
        (
            EngineExit::SquareOff,
            OrderRole::SquareOff,
            ExitReason::Scheduler,
        ),
        (
            EngineExit::Risk(RiskExitReason::Unprotected),
            OrderRole::RiskExit(RiskExitReason::Unprotected),
            ExitReason::Manual,
        ),
        (
            EngineExit::Risk(RiskExitReason::StopRejected),
            OrderRole::RiskExit(RiskExitReason::StopRejected),
            ExitReason::Manual,
        ),
        (
            EngineExit::Risk(RiskExitReason::StopUnfilled),
            OrderRole::RiskExit(RiskExitReason::StopUnfilled),
            ExitReason::LegSl,
        ),
        (
            EngineExit::Risk(RiskExitReason::KillSwitch),
            OrderRole::RiskExit(RiskExitReason::KillSwitch),
            ExitReason::Manual,
        ),
        (
            EngineExit::Risk(RiskExitReason::AccountLoss),
            OrderRole::RiskExit(RiskExitReason::AccountLoss),
            ExitReason::Manual,
        ),
        (
            EngineExit::Risk(RiskExitReason::OperatorStop),
            OrderRole::RiskExit(RiskExitReason::OperatorStop),
            ExitReason::Manual,
        ),
    ];
    for (exit, role, reason) in table {
        let mut kernel = holding_on(session(config()), 100.0);
        wait_before(
            &mut kernel,
            &bars(&[], &[]),
            &mut Script::new(Vec::new()),
            3,
        );
        let asked = kernel.request_exit(SBIN, Product::Mis, exit);
        assert_eq!(asked, Ok(ExitRequest::Sent), "{exit:?}");
        assert_eq!(
            placed_roles(&kernel.take_commands()),
            [(OrderId(2), Side::Sell, 100.0, role)],
            "{exit:?}"
        );
        let exit_reason = kernel.order(OrderId(2)).and_then(|o| o.request.exit_reason);
        assert_eq!(exit_reason, Some(reason), "{exit:?}");
    }
}
