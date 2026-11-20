#![cfg(all(feature = "costs", feature = "time"))]
//! A live session's member of an instrument group (`engine::venue_group`; 18-multi-strategy
//! 4.13 and 4.14, requirements KR10 to KR12): the side the other members hold or are
//! entering (`set_group_side`, reject `group_other_side`), the market orders only of a
//! shared instrument (`set_shared`, reject `shared_resting`), and a closing order's
//! margin rejection (`apply_margin_reject`), after which the kernel sends nothing that
//! closes the position by itself until the session asks: no exit at market, no stop placed
//! again or raised, no exit that waited, none at later opens.
//!
//! "Holding 100" is 100 units of SBIN in MIS bought at 500.10 at the open of minute 1
//! (o1, `venue_support::holding_on`); the orders of the gate tests are decided at the
//! close of minute 2. The bars rise 0.10 a minute from 500.00 (`venue_support::rising`).

mod venue_support;

use oa_bt::engine::{
    BarColumns, Engine, EngineExit, EventStamp, ExitRequest, GroupSide, KernelError, OrderRole,
    Pause, Protect, RiskExitReason, RunConfig, Step, Submission, VenueCommand, VenueError,
};
use oa_bt::exec::{
    ExecError, InstrumentId, OrderId, OrderRequest, OrderStatus, OrderType, QuantitySpec,
    RejectCode,
};
use oa_bt::model::{Product, Side};
use oa_bt::rms::{ExitReason, StrategyRiskConfig};
use venue_support::{
    at_close, at_second, backtest, bars, config, echo_until, entry_waiting, exit, holding_on,
    holding_with, limit, market, minute, modified, order, pause_at_close, placed_roles, sbin,
    session, stop_limit, stop_market, stop_points, wait_before, Act, Echo, Ohlc, Script, SBIN,
};

const STOP: OrderId = OrderId(2);
/// 500.10 less the 3-point stop.
const STOP_LEVEL: f64 = 497.1;

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

/// `exit()`: a reduce-only MARKET sale of the whole MIS position.
fn whole_exit() -> OrderRequest {
    OrderRequest {
        quantity: QuantitySpec::PositionFraction(1.0),
        ..exit(Side::Sell, 1.0, OrderType::Market)
    }
}

fn reversal() -> OrderRequest {
    OrderRequest {
        reverse: true,
        ..market(Side::Sell, 160.0)
    }
}

/// The reject code of `kernel`'s submission of `request`; `None` when it passed.
fn code_of(kernel: &mut Engine, request: OrderRequest) -> Option<RejectCode> {
    match kernel.submit(request) {
        Ok(Submission::Submitted(_)) => None,
        Ok(Submission::Rejected(_, code)) => Some(code),
        other => panic!("expected a submission, got {other:?}"),
    }
}

fn sbin_mis_side(kernel: &mut Engine, side: Option<GroupSide>) {
    let set = kernel.set_group_side(SBIN, Product::Mis, side);
    assert_eq!(set, Ok(()));
}

fn sbin_mis_shared(kernel: &mut Engine, shared: bool) {
    assert_eq!(kernel.set_shared(SBIN, Product::Mis, shared), Ok(()));
}

#[test]
fn an_order_that_would_open_or_add_on_the_other_side_of_the_group_is_rejected() {
    let other_side = Some(RejectCode::GroupOtherSide);
    let long = Some(GroupSide::Long);
    let short = Some(GroupSide::Short);
    let stop = || exit(Side::Sell, 100.0, stop_market(450.0));
    let table = [
        (
            "long: a buy from flat",
            long,
            0.0,
            market(Side::Buy, 10.0),
            None,
        ),
        (
            "long: a sale from flat",
            long,
            0.0,
            market(Side::Sell, 10.0),
            other_side,
        ),
        (
            "long: a resting sale from flat",
            long,
            0.0,
            order(Side::Sell, 10.0, limit(520.0), Product::Mis),
            other_side,
        ),
        ("long: an add", long, 100.0, market(Side::Buy, 10.0), None),
        (
            "long: a plain sale within the long",
            long,
            100.0,
            market(Side::Sell, 40.0),
            None,
        ),
        ("long: exit()", long, 100.0, whole_exit(), None),
        ("long: a reduce-only stop", long, 100.0, stop(), None),
        ("long: a reversal", long, 100.0, reversal(), other_side),
        (
            "short: a buy from flat",
            short,
            0.0,
            market(Side::Buy, 10.0),
            other_side,
        ),
        (
            "short: a sale from flat",
            short,
            0.0,
            market(Side::Sell, 10.0),
            None,
        ),
        (
            "short: an add to a long",
            short,
            100.0,
            market(Side::Buy, 10.0),
            other_side,
        ),
        ("short: exit()", short, 100.0, whole_exit(), None),
        (
            "none: a sale from flat",
            None,
            0.0,
            market(Side::Sell, 10.0),
            None,
        ),
    ];
    for (what, side, held, request, code) in table {
        let mut kernel = paused_holding(held);
        sbin_mis_side(&mut kernel, side);
        assert_eq!(code_of(&mut kernel, request), code, "{what}");
        if code.is_some() {
            assert!(kernel.take_commands().is_empty(), "{what}: nothing is sent");
        }
    }
}

#[test]
fn a_group_side_cleared_lets_the_other_side_trade_again() {
    let mut kernel = paused_holding(0.0);
    sbin_mis_side(&mut kernel, Some(GroupSide::Long));
    sbin_mis_side(&mut kernel, None);
    assert_eq!(code_of(&mut kernel, market(Side::Sell, 10.0)), None);
}

#[test]
fn a_strategy_order_that_rests_on_a_shared_instrument_is_rejected() {
    let resting = Some(RejectCode::SharedResting);
    let table = [
        ("a market buy from flat", 0.0, market(Side::Buy, 10.0), None),
        (
            "a limit buy from flat",
            0.0,
            order(Side::Buy, 10.0, limit(495.0), Product::Mis),
            resting,
        ),
        (
            "an SL-M buy from flat",
            0.0,
            order(Side::Buy, 10.0, stop_market(510.0), Product::Mis),
            resting,
        ),
        (
            "an SL buy from flat",
            0.0,
            order(Side::Buy, 10.0, stop_limit(510.0, 511.0), Product::Mis),
            resting,
        ),
        ("a market add", 100.0, market(Side::Buy, 10.0), None),
        (
            "a resting target",
            100.0,
            exit(Side::Sell, 100.0, limit(600.0)),
            resting,
        ),
        (
            "a resting stop",
            100.0,
            exit(Side::Sell, 100.0, stop_market(450.0)),
            resting,
        ),
        ("exit()", 100.0, whole_exit(), None),
        (
            "a plain market sale within the long",
            100.0,
            market(Side::Sell, 40.0),
            None,
        ),
    ];
    for (what, held, request, code) in table {
        let mut kernel = paused_holding(held);
        sbin_mis_shared(&mut kernel, true);
        assert_eq!(code_of(&mut kernel, request), code, "{what}");
        if code.is_some() {
            assert!(kernel.take_commands().is_empty(), "{what}: nothing is sent");
        }
    }
}

#[test]
fn an_instrument_no_longer_shared_takes_resting_orders_again() {
    let mut kernel = paused_holding(0.0);
    sbin_mis_shared(&mut kernel, true);
    sbin_mis_shared(&mut kernel, false);
    let resting_buy = order(Side::Buy, 10.0, limit(495.0), Product::Mis);
    assert_eq!(code_of(&mut kernel, resting_buy), None);
}

#[test]
fn a_shared_instruments_refusals_come_before_the_entry_locks() {
    // Both refusals stand whatever the locks do, so the code names them first.
    let mut other_side = paused_holding(0.0);
    sbin_mis_side(&mut other_side, Some(GroupSide::Long));
    assert_eq!(other_side.lock_entries("feed_dead"), Ok(true));
    let code = code_of(&mut other_side, market(Side::Sell, 10.0));
    assert_eq!(code, Some(RejectCode::GroupOtherSide));
    let mut resting = paused_holding(0.0);
    sbin_mis_shared(&mut resting, true);
    assert_eq!(resting.lock_entries("feed_dead"), Ok(true));
    let resting_buy = order(Side::Buy, 10.0, limit(495.0), Product::Mis);
    assert_eq!(
        code_of(&mut resting, resting_buy),
        Some(RejectCode::SharedResting)
    );
}

#[test]
fn a_modification_is_judged_by_the_gates_of_its_instrument_as_a_submission_is() {
    // A resting order placed before the gate was set (o1), modified after it.
    type Gate = fn(&mut Engine);
    let table: [(&str, OrderRequest, OrderRequest, Gate, RejectCode); 2] = [
        (
            "the other side",
            order(Side::Sell, 10.0, limit(520.0), Product::Mis),
            order(Side::Sell, 10.0, limit(525.0), Product::Mis),
            |k| sbin_mis_side(k, Some(GroupSide::Long)),
            RejectCode::GroupOtherSide,
        ),
        (
            "a shared instrument",
            order(Side::Buy, 10.0, limit(450.0), Product::Mis),
            order(Side::Buy, 10.0, limit(455.0), Product::Mis),
            |k| sbin_mis_shared(k, true),
            RejectCode::SharedResting,
        ),
    ];
    for (what, placed, replacement, gate, code) in table {
        let mut kernel = paused_holding(0.0);
        assert_eq!(code_of(&mut kernel, placed), None, "{what}");
        kernel.take_commands();
        gate(&mut kernel);
        let refused = kernel.modify(OrderId(1), replacement);
        assert_eq!(
            refused,
            Ok(Submission::Rejected(OrderId(1), code)),
            "{what}"
        );
        assert!(kernel.take_commands().is_empty(), "{what}: nothing is sent");
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

/// A shared SBIN whose other members are short, where this member bought 100 with a stop
/// 3 points below (o1, filled at 500.10 at the open of minute 1); the fill's commands are
/// left for the test.
fn shared_and_holding_with_a_stop() -> Engine {
    let mut kernel = entry_waiting(sbin(), 100.0, stop_points(3.0));
    sbin_mis_shared(&mut kernel, true);
    sbin_mis_side(&mut kernel, Some(GroupSide::Short));
    kernel
        .apply_fill(OrderId(1), 100.0, 500.1, minute(1))
        .expect("the buy fills");
    kernel
}

#[test]
fn the_legs_stop_on_a_shared_instrument_is_placed_as_on_any_other() {
    let mut kernel = shared_and_holding_with_a_stop();
    let stop = (
        STOP,
        Side::Sell,
        100.0,
        stop_market(STOP_LEVEL),
        OrderRole::LegStop,
    );
    assert_eq!(placed(&kernel.take_commands()), [stop]);
}

#[test]
fn the_reprice_of_a_triggered_stop_on_a_shared_instrument_is_sent() {
    let mut kernel = shared_and_holding_with_a_stop();
    kernel.take_commands();
    assert_eq!(kernel.reprice_exit(STOP, 497.0), Ok(true));
    assert_eq!(
        modified(&kernel.take_commands()),
        [(STOP, 100.0, limit(497.0))]
    );
}

#[test]
fn the_engines_exit_at_a_limit_on_a_shared_instrument_is_sent() {
    // The price rule's fallback: the stop is cancelled, then the exit goes at the limit.
    let mut kernel = shared_and_holding_with_a_stop();
    kernel.take_commands();
    let unfilled = EngineExit::Risk(RiskExitReason::StopUnfilled);
    let asked = kernel.request_limit_exit(SBIN, Product::Mis, unfilled, 496.0);
    assert_eq!(asked, Ok(ExitRequest::WaitsForCancels));
    kernel.take_commands();
    assert!(kernel.apply_cancel(STOP).is_ok());
    let exit = (
        OrderId(3),
        Side::Sell,
        100.0,
        limit(496.0),
        OrderRole::RiskExit(RiskExitReason::StopUnfilled),
    );
    assert_eq!(placed(&kernel.take_commands()), [exit]);
}

#[test]
fn the_gates_are_a_live_sessions_controls() {
    let mut simulated = backtest(config(), bars(&[], &[]));
    let not_external = Err(VenueError::NotExternal.into());
    let long = Some(GroupSide::Long);
    assert_eq!(
        simulated.set_group_side(SBIN, Product::Mis, long),
        not_external
    );
    assert_eq!(simulated.set_shared(SBIN, Product::Mis, true), not_external);
    let mut kernel = session(config());
    let unknown = InstrumentId(5);
    for result in [
        kernel.set_group_side(unknown, Product::Mis, long),
        kernel.set_shared(unknown, Product::Mis, true),
    ] {
        assert!(
            matches!(
                result,
                Err(KernelError::InvalidConfig {
                    field: "symbol",
                    ..
                })
            ),
            "{result:?}"
        );
    }
}

#[test]
fn a_group_side_is_named_long_or_short() {
    assert_eq!(GroupSide::CODES, ["long", "short"]);
    for side in [GroupSide::Long, GroupSide::Short] {
        assert_eq!(GroupSide::from_code(side.code()), Some(side));
    }
    assert_eq!(GroupSide::from_code("BUY"), None);
}

/// A step that ends an order of the session: the venue's result, or its answer.
type Ending = fn(&mut Engine) -> Result<(), KernelError>;

/// An order placed: (order, side, units, role).
type Placed = (OrderId, Side, f64, OrderRole);

/// An order placed, with the exit reason its fills record.
type PlacedExit = (OrderId, Option<ExitReason>);

#[test]
fn a_stop_rejected_for_margin_is_neither_exited_at_market_nor_placed_again() {
    // Beside each row, the stop rejected the usual way (R3) exits the leg at market.
    let rows: [(&str, Ending, Vec<Placed>); 3] = [
        (
            "rejected: exit at market",
            |k| k.apply_reject(STOP, RejectCode::Venue).map(|_| ()),
            vec![(
                OrderId(3),
                Side::Sell,
                100.0,
                OrderRole::RiskExit(RiskExitReason::StopRejected),
            )],
        ),
        (
            "rejected for margin: nothing",
            |k| k.apply_margin_reject(STOP).map(|_| ()),
            vec![],
        ),
        (
            "rejected for margin while the square-off waits for it: nothing, it waits",
            |k| {
                k.request_exit(SBIN, Product::Mis, EngineExit::SquareOff)?;
                k.take_commands();
                k.apply_margin_reject(STOP).map(|_| ())
            },
            vec![],
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
fn an_exit_held_by_a_margin_rejection_goes_when_the_session_asks_for_an_exit() {
    // The square-off waited for the stop's cancellation (o2) when the stop was rejected
    // for margin; the session's safe stop then sends it (o3).
    let mut kernel = holding_with(100.0, stop_points(3.0));
    kernel.take_commands();
    let asked = kernel.request_exit(SBIN, Product::Mis, EngineExit::SquareOff);
    assert_eq!(asked, Ok(ExitRequest::WaitsForCancels));
    kernel.take_commands();
    assert!(kernel.apply_margin_reject(STOP).is_ok());
    assert!(kernel.take_commands().is_empty(), "the square-off waits");
    let stop = EngineExit::Risk(RiskExitReason::OperatorStop);
    let asked = kernel.request_exit(SBIN, Product::Mis, stop);
    assert_eq!(asked, Ok(ExitRequest::AlreadyExiting));
    let square_off = (OrderId(3), Side::Sell, 100.0, OrderRole::SquareOff);
    assert_eq!(placed_roles(&kernel.take_commands()), [square_off]);
}

#[test]
fn a_stop_rejected_for_margin_ends_rejected_where_the_venues_results_are_recorded() {
    // The session waits before minute 1, so the result is recorded in slice 1, in step 3
    // where a backtest fills an SL-M order.
    let mut kernel = holding_with(100.0, stop_points(3.0));
    kernel.take_commands();
    let stamp = EventStamp {
        slice: 1,
        slice_ts: minute(1),
        step: Step::Levels,
    };
    assert_eq!(kernel.apply_margin_reject(STOP), Ok(stamp));
    let stop = kernel.order(STOP).expect("the stop");
    assert_eq!(
        (stop.status, stop.reject_code),
        (OrderStatus::Rejected, Some(RejectCode::Venue))
    );
}

#[test]
fn a_leg_whose_stop_was_rejected_for_margin_gets_no_new_stop_at_later_opens() {
    let mut kernel = holding_with(100.0, stop_points(3.0));
    kernel.take_commands();
    assert!(kernel.apply_margin_reject(STOP).is_ok());
    let mut script = Script::new(Vec::new());
    wait_before(&mut kernel, &bars(&[], &[]), &mut script, 6);
    assert!(kernel.take_commands().is_empty());
}

#[test]
fn protect_places_the_stop_again_after_its_margin_rejection() {
    let mut kernel = holding_with(100.0, stop_points(3.0));
    kernel.take_commands();
    assert!(kernel.apply_margin_reject(STOP).is_ok());
    assert_eq!(kernel.protect(SBIN, Product::Mis), Ok(Protect::Placed));
    let stop = (
        OrderId(3),
        Side::Sell,
        100.0,
        stop_market(STOP_LEVEL),
        OrderRole::LegStop,
    );
    assert_eq!(placed(&kernel.take_commands()), [stop]);
}

#[test]
fn a_strategy_exit_rejected_for_margin_ends_rejected_and_nothing_follows() {
    // Holding 100, a plain sale of 40 (o2) decided at the close of minute 2.
    let mut kernel = paused_holding(100.0);
    assert_eq!(code_of(&mut kernel, market(Side::Sell, 40.0)), None);
    kernel.take_commands();
    let mut script = Script::new(Vec::new());
    wait_before(&mut kernel, &bars(&[], &[]), &mut script, 3);
    assert!(kernel.apply_margin_reject(OrderId(2)).is_ok());
    let sale = kernel.order(OrderId(2)).map(|o| o.status);
    assert_eq!(sale, Some(OrderStatus::Rejected));
    wait_before(&mut kernel, &bars(&[], &[]), &mut script, 6);
    assert!(kernel.take_commands().is_empty());
}

/// `reduce(fraction)`: a reduce-only MARKET sale of that fraction of the position.
fn reduce(fraction: f64) -> OrderRequest {
    OrderRequest {
        quantity: QuantitySpec::PositionFraction(fraction),
        ..exit(Side::Sell, 1.0, OrderType::Market)
    }
}

/// Holding 100 with its stop (o2), the strategy halves the position at the close of
/// minute 1 (o3): the stop is cut to 50 first, the venue confirms the cut, and o3 goes for
/// 50 while the session waits before minute 2; every command is taken.
fn half_exit_sent() -> Engine {
    let mut kernel = holding_with(100.0, stop_points(3.0));
    kernel.take_commands();
    pause_at_close(&mut kernel, &bars(&[], &[]), 1);
    let submitted = kernel.submit(reduce(0.5));
    assert_eq!(submitted, Ok(Submission::Submitted(OrderId(3))));
    let cut = (STOP, 50.0, stop_market(STOP_LEVEL));
    assert_eq!(modified(&kernel.take_commands()), [cut]);
    assert!(matches!(kernel.advance(), Ok(Pause::Waiting { .. })));
    kernel
        .apply_modify(STOP)
        .expect("the venue confirms the cut");
    let sent = placed_roles(&kernel.take_commands());
    assert_eq!(sent, [(OrderId(3), Side::Sell, 50.0, OrderRole::Exit)]);
    kernel
}

/// A modification: (order, units, order type).
type Modified = (OrderId, f64, OrderType);

#[test]
fn a_partial_exit_rejected_for_margin_leaves_the_stop_cut_down() {
    // Rejected the usual way, the exit ended unfilled, so the stop covers the whole
    // position again (R4). Rejected for margin, the broker may not hold the 100: the
    // stop is not raised.
    let rows: [(&str, Ending, Vec<Modified>); 2] = [
        (
            "rejected",
            |k| k.apply_reject(OrderId(3), RejectCode::Venue).map(|_| ()),
            vec![(STOP, 100.0, stop_market(STOP_LEVEL))],
        ),
        (
            "rejected for margin",
            |k| k.apply_margin_reject(OrderId(3)).map(|_| ()),
            vec![],
        ),
    ];
    for (what, ending, expected) in rows {
        let mut kernel = half_exit_sent();
        ending(&mut kernel).expect("the venue's result is taken");
        let commands = kernel.take_commands();
        assert_eq!(modified(&commands), expected, "{what}");
        assert!(
            placed_roles(&commands).is_empty(),
            "{what}: nothing is placed"
        );
    }
}

#[test]
fn protect_puts_the_stop_back_over_the_position_after_a_partial_exits_margin_rejection() {
    let mut kernel = half_exit_sent();
    assert!(kernel.apply_margin_reject(OrderId(3)).is_ok());
    assert_eq!(kernel.protect(SBIN, Product::Mis), Ok(Protect::StopWorks));
    let raised = (STOP, 100.0, stop_market(STOP_LEVEL));
    assert_eq!(modified(&kernel.take_commands()), [raised]);
}

/// `half_exit_sent` with o3 rejected for margin (the stop stays at 50 of the 100), then the
/// strategy's `exit` decided at the close of minute 2 (o4), the session waiting before
/// minute 4; every command is taken.
fn exit_made_while_held(exit: OrderRequest) -> Engine {
    let mut kernel = half_exit_sent();
    assert!(kernel.apply_margin_reject(OrderId(3)).is_ok());
    let mut script = Script::new(vec![at_close(2, Act::Submit(exit))]);
    wait_before(&mut kernel, &bars(&[], &[]), &mut script, 4);
    assert_eq!(script.submissions, [Submission::Submitted(OrderId(4))]);
    kernel
}

#[test]
fn an_exit_the_strategy_makes_while_a_margin_rejection_holds_the_position_waits() {
    // Nothing is cancelled or sent for it, and the stop is not cut for it: exit() would
    // cancel the stop first, reduce(0.75) would cut it to the 25 that stay.
    for (what, exit) in [("exit()", whole_exit()), ("reduce(0.75)", reduce(0.75))] {
        let mut kernel = exit_made_while_held(exit);
        assert!(kernel.take_commands().is_empty(), "{what}");
        let waiting = kernel.order(OrderId(4)).map(|o| o.status);
        assert_eq!(waiting, Some(OrderStatus::Submitted), "{what}");
    }
}

#[test]
fn protect_moves_on_an_exit_the_hold_kept_waiting() {
    // The strategy's exit() cancels the stop first, as it would have when it was made.
    let mut kernel = exit_made_while_held(whole_exit());
    kernel.take_commands();
    assert_eq!(kernel.protect(SBIN, Product::Mis), Ok(Protect::Exiting));
    assert_eq!(
        kernel.take_commands(),
        [VenueCommand::Cancel { order: STOP }]
    );
}

#[test]
fn the_sessions_exit_takes_over_from_an_exit_the_hold_kept_waiting() {
    let mut kernel = exit_made_while_held(whole_exit());
    kernel.take_commands();
    let stop = EngineExit::Risk(RiskExitReason::OperatorStop);
    let asked = kernel.request_exit(SBIN, Product::Mis, stop);
    assert_eq!(asked, Ok(ExitRequest::WaitsForCancels));
    assert_eq!(
        kernel.take_commands(),
        [VenueCommand::Cancel { order: STOP }]
    );
    let strategy_exit = kernel.order(OrderId(4)).map(|o| o.status);
    assert_eq!(strategy_exit, Some(OrderStatus::Canceled));
}

/// The slice that opens at the square-off, 15:10.
const SQUARE_OFF_SLICE: usize = 355;

/// Holding 100 with no stop (o1), a plain sale of 40 (o2) decided at the close of minute
/// 2, sent, and the session waiting before minute 3.
fn sale_sent() -> Engine {
    let mut kernel = paused_holding(100.0);
    assert_eq!(code_of(&mut kernel, market(Side::Sell, 40.0)), None);
    let mut script = Script::new(Vec::new());
    wait_before(&mut kernel, &bars(&[], &[]), &mut script, 3);
    kernel.take_commands();
    kernel
}

/// A session in the state a row of a table starts from.
type Holding = fn() -> Engine;

/// Holding 100 with its stop (o2), the session waiting before minute 1.
fn stop_placed() -> Engine {
    let mut kernel = holding_with(100.0, stop_points(3.0));
    kernel.take_commands();
    kernel
}

#[test]
fn a_position_held_by_a_margin_rejection_gets_no_square_off() {
    // Any closing order rejected for margin holds the position; a sale rejected the usual
    // way leaves it to the square-off (o3).
    let square_off = (OrderId(3), Side::Sell, 100.0, OrderRole::SquareOff);
    let rows: [(&str, Holding, Ending, Vec<Placed>); 3] = [
        (
            "a sale rejected",
            sale_sent,
            |k| k.apply_reject(OrderId(2), RejectCode::Venue).map(|_| ()),
            vec![square_off],
        ),
        (
            "a sale rejected for margin",
            sale_sent,
            |k| k.apply_margin_reject(OrderId(2)).map(|_| ()),
            vec![],
        ),
        (
            "the leg's stop rejected for margin",
            stop_placed,
            |k| k.apply_margin_reject(STOP).map(|_| ()),
            vec![],
        ),
    ];
    for (what, holding, ending, expected) in rows {
        let mut kernel = holding();
        ending(&mut kernel).expect("the venue's result is taken");
        let mut script = Script::new(Vec::new());
        wait_before(&mut kernel, &bars(&[], &[]), &mut script, SQUARE_OFF_SLICE);
        assert_eq!(placed_roles(&kernel.take_commands()), expected, "{what}");
    }
}

/// A crash at minute 10: open 494.00, close 494.05, a loss of about 605 on 100 units
/// bought at 500.10, past the daily loss limit of 500.
const CRASH: (u32, Ohlc) = (10, (494.0, 494.5, 493.5, 494.05));

fn daily_loss_config() -> RunConfig {
    RunConfig {
        risk: StrategyRiskConfig {
            max_daily_loss_rupees: Some(500.0),
            ..StrategyRiskConfig::default()
        },
        ..config()
    }
}

fn crash_bars() -> BarColumns {
    bars(&[CRASH], &[])
}

/// Long 100 from minute 1 (o1, filled by the echo venue); the crash breaches the daily
/// loss limit at the close of minute 10, and its exit (o2) is sent. The session waits
/// before slice 11 with o2's command taken.
fn daily_loss_exit_sent() -> Engine {
    let mut kernel = session(daily_loss_config());
    let acts = vec![at_close(0, Act::Submit(market(Side::Buy, 100.0)))];
    let mut echo = Echo::default();
    echo_until(
        &mut kernel,
        &crash_bars(),
        &mut Script::new(acts),
        &mut echo,
        11,
    );
    let sent = placed_roles(&kernel.take_commands());
    let daily_loss = OrderRole::RiskExit(RiskExitReason::DailyLoss);
    assert_eq!(sent, [(OrderId(2), Side::Sell, 100.0, daily_loss)]);
    kernel
}

/// Advance `kernel` to the wait before `slice` with nothing filled, and return the orders
/// placed meanwhile with their exit reasons.
fn placed_by(kernel: &mut Engine, slice: usize) -> Vec<PlacedExit> {
    wait_before(kernel, &crash_bars(), &mut Script::new(Vec::new()), slice);
    let commands = kernel.take_commands();
    placed_roles(&commands)
        .iter()
        .map(|(id, ..)| (*id, kernel.order(*id).and_then(|o| o.request.exit_reason)))
        .collect()
}

#[test]
fn an_engine_exit_rejected_for_margin_is_not_sent_again_at_later_opens() {
    // Rejected the usual way, the daily loss exit goes again at the next open (o3).
    let rows: [(&str, Ending, Vec<PlacedExit>); 2] = [
        (
            "rejected",
            |k| k.apply_reject(OrderId(2), RejectCode::Venue).map(|_| ()),
            vec![(OrderId(3), Some(ExitReason::DailyLossLimit))],
        ),
        (
            "rejected for margin",
            |k| k.apply_margin_reject(OrderId(2)).map(|_| ()),
            vec![],
        ),
    ];
    for (what, ending, expected) in rows {
        let mut kernel = daily_loss_exit_sent();
        ending(&mut kernel).expect("the venue's result is taken");
        assert_eq!(placed_by(&mut kernel, 14), expected, "{what}");
    }
}

#[test]
fn an_exit_the_session_asks_for_lets_the_engines_exit_go_again_at_later_opens() {
    // After the margin rejection the session exits (o3, Ctrl+C); that exit is rejected
    // the usual way, so the daily loss exit goes again at the next open (o4).
    let mut kernel = daily_loss_exit_sent();
    assert!(kernel.apply_margin_reject(OrderId(2)).is_ok());
    let stop = EngineExit::Risk(RiskExitReason::OperatorStop);
    let asked = kernel.request_exit(SBIN, Product::Mis, stop);
    assert_eq!(asked, Ok(ExitRequest::Sent));
    kernel.take_commands();
    assert!(kernel.apply_reject(OrderId(3), RejectCode::Venue).is_ok());
    let expected = [(OrderId(4), Some(ExitReason::DailyLossLimit))];
    assert_eq!(placed_by(&mut kernel, 13), expected);
}

#[test]
fn protect_lets_the_engines_exit_go_again_at_later_opens() {
    // The leg has no stop rule, so nothing is placed; the session has asked all the same.
    let mut kernel = daily_loss_exit_sent();
    assert!(kernel.apply_margin_reject(OrderId(2)).is_ok());
    assert_eq!(kernel.protect(SBIN, Product::Mis), Ok(Protect::NoStopLevel));
    let expected = [(OrderId(3), Some(ExitReason::DailyLossLimit))];
    assert_eq!(placed_by(&mut kernel, 13), expected);
}

#[test]
fn a_new_leg_is_exited_by_the_engine_as_any_leg_is() {
    // Held at most 3 base bars: the first leg (from minute 1) is due at the open of
    // minute 4 (o2), rejected for margin; the broker then sells the 100 by itself, a buy
    // decided at the close of minute 4 (o3) fills at the open of minute 5, and that leg's
    // time exit is due at the open of minute 8 (o4).
    let held_three = RunConfig {
        risk: StrategyRiskConfig {
            max_hold_bars: Some(3),
            ..StrategyRiskConfig::default()
        },
        ..config()
    };
    let mut kernel = session(held_three);
    let acts = vec![at_close(0, Act::Submit(market(Side::Buy, 100.0)))];
    echo_until(
        &mut kernel,
        &bars(&[], &[]),
        &mut Script::new(acts),
        &mut Echo::default(),
        4,
    );
    let time_exit = OrderRole::RiskExit(RiskExitReason::TimeExit);
    let first = placed_roles(&kernel.take_commands());
    assert_eq!(first, [(OrderId(2), Side::Sell, 100.0, time_exit)]);
    assert!(kernel.apply_margin_reject(OrderId(2)).is_ok());
    let sold = kernel.apply_external_fill(
        SBIN,
        Product::Mis,
        Side::Sell,
        100.0,
        500.4,
        at_second(4, 1),
        ExitReason::Manual,
    );
    assert!(sold.is_ok(), "{sold:?}");
    let buy = vec![at_close(4, Act::Submit(market(Side::Buy, 100.0)))];
    wait_before(&mut kernel, &bars(&[], &[]), &mut Script::new(buy), 5);
    kernel.take_commands();
    assert!(kernel
        .apply_fill(OrderId(3), 100.0, 500.5, minute(5))
        .is_ok());
    let mut script = Script::new(Vec::new());
    wait_before(&mut kernel, &bars(&[], &[]), &mut script, 8);
    let second = placed_roles(&kernel.take_commands());
    assert_eq!(second, [(OrderId(4), Side::Sell, 100.0, time_exit)]);
}

#[test]
fn only_a_closing_order_takes_a_margin_rejection() {
    // A buy from flat (o1), decided at the close of minute 2, still works after.
    let mut kernel = paused_holding(0.0);
    assert_eq!(code_of(&mut kernel, market(Side::Buy, 10.0)), None);
    kernel.take_commands();
    let mut script = Script::new(Vec::new());
    wait_before(&mut kernel, &bars(&[], &[]), &mut script, 3);
    let refused = kernel.apply_margin_reject(OrderId(1));
    assert!(
        matches!(
            refused,
            Err(KernelError::Exec(ExecError::InvalidRequest { .. }))
        ),
        "{refused:?}"
    );
    let entry = kernel.order(OrderId(1)).map(|o| o.status);
    assert_eq!(entry, Some(OrderStatus::Submitted));
}

#[test]
fn a_margin_rejection_is_refused_as_any_venue_result_is() {
    let mut simulated = backtest(config(), bars(&[], &[]));
    let not_external = Err(VenueError::NotExternal.into());
    assert_eq!(simulated.apply_margin_reject(OrderId(1)), not_external);
    let mut kernel = paused_holding(100.0);
    let at_bar_close = Err(VenueError::AtBarClose.into());
    assert_eq!(kernel.apply_margin_reject(OrderId(1)), at_bar_close);
    let mut script = Script::new(Vec::new());
    wait_before(&mut kernel, &bars(&[], &[]), &mut script, 3);
    let ended = kernel.apply_margin_reject(OrderId(1));
    let filled = VenueError::OrderEnded {
        order: OrderId(1),
        status: OrderStatus::Filled,
    };
    assert_eq!(ended, Err(filled.into()));
}
