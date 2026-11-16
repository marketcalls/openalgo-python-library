#![cfg(all(feature = "costs", feature = "time"))]
//! Exits of a leg whose stop the broker holds (`engine::venue_owner`, `venue_legs`,
//! `venue_protect`; 16-session-design 4.5 to 4.8, requirements R4 and R7): a partial exit
//! cuts the stop down first, a full one cancels it first, and the session's own calls
//! protect the leg again, reprice a stop the price has passed and exit at a limit.
//!
//! Every test buys 100 SBIN in MIS (o1, filled at 500.10 at the open of minute 1) with a
//! stop 3 points below (o2 at 497.10), unless a row says otherwise.

mod venue_support;

use oa_bt::engine::{
    Engine, EngineExit, ExitRequest, KernelError, OrderRole, Pause, Protect, RiskExitReason,
    Submission, VenueCommand, VenueError,
};
use oa_bt::exec::{ExecError, OrderId, OrderRequest, OrderType, QuantitySpec, RejectCode};
use oa_bt::model::{Product, Side};
use oa_bt::rms::{AfterFirstTarget, Distance, ExitReason, LegRules, TargetSpec, TrailSpec};
use venue_support::{
    at_second, backtest, bars, config, exit, holding_with, limit, market, minute, modified,
    pause_at_close, placed_roles, session, stop_points, SBIN,
};

const STOP: OrderId = OrderId(2);
const STOP_LEVEL: f64 = 497.1;

fn sl_m(trigger_rupees: f64) -> OrderType {
    OrderType::StopMarket { trigger_rupees }
}

/// `exit()` of a fraction of the position: a reduce-only MARKET order.
fn reduce(fraction: f64) -> OrderRequest {
    OrderRequest {
        quantity: QuantitySpec::PositionFraction(fraction),
        ..exit(Side::Sell, 1.0, OrderType::Market)
    }
}

/// The stop 3 points below and two targets, half the position each, 2 and 4 points above:
/// 502.10 and 504.10.
fn stop_and_two_targets() -> LegRules {
    let target = |points| TargetSpec {
        distance: Distance::Points(points),
        fraction: 0.5,
    };
    LegRules {
        targets: vec![target(2.0), target(4.0)],
        ..stop_points(3.0)
    }
}

/// Holding 100 with `rules`, the stop's place taken, waiting before minute 1.
fn holding(rules: LegRules) -> Engine {
    let mut kernel = holding_with(100.0, rules);
    kernel.take_commands();
    kernel
}

/// Holding 100 with its stop, paused at the close of minute 1.
fn deciding() -> Engine {
    let mut kernel = holding(stop_points(3.0));
    pause_at_close(&mut kernel, &bars(&[], &[]), 1);
    kernel
}

fn waiting(kernel: &mut Engine) {
    assert!(matches!(kernel.advance(), Ok(Pause::Waiting { .. })));
}

#[test]
fn a_partial_exit_waits_while_the_stop_is_cut_down_then_goes() {
    let mut kernel = deciding();
    assert_eq!(
        kernel.submit(reduce(0.5)),
        Ok(Submission::Submitted(OrderId(3)))
    );
    let commands = kernel.take_commands();
    assert_eq!(modified(&commands), [(STOP, 50.0, sl_m(STOP_LEVEL))]);
    assert!(placed_roles(&commands).is_empty(), "nothing is placed yet");
    waiting(&mut kernel);
    assert!(kernel.take_commands().is_empty());
    kernel
        .apply_modify(STOP)
        .expect("the venue confirms the cut");
    let expected = vec![(OrderId(3), Side::Sell, 50.0, OrderRole::Exit)];
    assert_eq!(placed_roles(&kernel.take_commands()), expected);
    kernel
        .apply_fill(OrderId(3), 50.0, 500.5, minute(2))
        .expect("the exit fills");
    assert!(
        kernel.take_commands().is_empty(),
        "the stop covers what is left"
    );
}

#[test]
fn a_full_exit_cancels_the_stop_first() {
    let mut kernel = deciding();
    let full = exit(Side::Sell, 100.0, OrderType::Market);
    assert_eq!(kernel.submit(full), Ok(Submission::Submitted(OrderId(3))));
    let cancelled = kernel.take_commands();
    assert_eq!(cancelled, [VenueCommand::Cancel { order: STOP }]);
    waiting(&mut kernel);
    kernel.take_commands();
    kernel.apply_cancel(STOP).expect("the stop is cancelled");
    let expected = vec![(OrderId(3), Side::Sell, 100.0, OrderRole::Exit)];
    assert_eq!(placed_roles(&kernel.take_commands()), expected);
}

#[test]
fn a_stop_the_broker_cancels_while_a_partial_exit_waits_is_placed_again_for_the_rest() {
    // The partial exit does not close the leg: the half that stays gets its stop again,
    // and the exit then goes beside it.
    let mut kernel = deciding();
    assert_eq!(
        kernel.submit(reduce(0.5)),
        Ok(Submission::Submitted(OrderId(3)))
    );
    waiting(&mut kernel);
    kernel.take_commands();
    kernel
        .apply_cancel(STOP)
        .expect("the broker cancels the stop");
    let expected = vec![
        (OrderId(4), Side::Sell, 50.0, OrderRole::LegStop),
        (OrderId(3), Side::Sell, 50.0, OrderRole::Exit),
    ];
    assert_eq!(placed_roles(&kernel.take_commands()), expected);
}

/// One step of a tick-driven script: what the venue or the feed does, at the session's
/// clock, and the commands the session sends after it.
type Step = (
    &'static str,
    fn(&mut Engine) -> Result<(), KernelError>,
    Vec<VenueCommand>,
);

fn run_steps(kernel: &mut Engine, steps: Vec<Step>) {
    for (what, step, expected) in steps {
        step(kernel).expect("the step is taken");
        assert_eq!(kernel.take_commands(), expected, "{what}");
    }
}

fn modify_stop(units: f64, trigger: f64) -> VenueCommand {
    VenueCommand::Modify {
        order: STOP,
        units,
        order_type: sl_m(trigger),
    }
}

fn place_exit(order: u64, units: f64) -> VenueCommand {
    VenueCommand::Place {
        order: OrderId(order),
        instrument: SBIN,
        product: Product::Mis,
        side: Side::Sell,
        units,
        order_type: OrderType::Market,
        role: OrderRole::Exit,
    }
}

#[test]
fn a_partial_target_cuts_the_stop_and_a_refused_cut_is_sent_again_after_the_throttle() {
    let mut kernel = holding(stop_and_two_targets());
    let steps: Vec<Step> = vec![
        (
            "the first target is reached: the stop is cut to the half that stays",
            |k| k.push_tick(SBIN, 502.1, at_second(1, 10)),
            vec![modify_stop(50.0, STOP_LEVEL)],
        ),
        (
            "the venue refuses the cut: the target's exit waits",
            |k| k.apply_modify_refused(STOP),
            vec![],
        ),
        (
            "4 s after the cut was sent: nothing yet",
            |k| k.push_tick(SBIN, 502.2, at_second(1, 14)),
            vec![],
        ),
        (
            "5 s after: the cut goes again",
            |k| k.push_tick(SBIN, 502.2, at_second(1, 15)),
            vec![modify_stop(50.0, STOP_LEVEL)],
        ),
        (
            "confirmed: the target's exit goes",
            |k| k.apply_modify(STOP).map(|_| ()),
            vec![place_exit(3, 50.0)],
        ),
        (
            "it fills",
            |k| {
                k.apply_fill(OrderId(3), 50.0, 502.1, at_second(1, 16))
                    .map(|_| ())
            },
            vec![],
        ),
        (
            "the second target takes the rest: the stop is cancelled first",
            |k| k.push_tick(SBIN, 504.1, at_second(1, 30)),
            vec![VenueCommand::Cancel { order: STOP }],
        ),
        (
            "then the exit goes",
            |k| k.apply_cancel(STOP).map(|_| ()),
            vec![place_exit(4, 50.0)],
        ),
    ];
    run_steps(&mut kernel, steps);
    let target_exit = kernel.order(OrderId(3)).expect("the target's exit");
    assert_eq!(target_exit.request.exit_reason, Some(ExitReason::LegTarget));
}

#[test]
fn a_filled_first_target_moves_the_stop_to_breakeven_after_the_throttle() {
    let rules = LegRules {
        after_first_target: AfterFirstTarget::Breakeven,
        ..stop_and_two_targets()
    };
    let mut kernel = holding(rules);
    let steps: Vec<Step> = vec![
        (
            "the first target is reached",
            |k| k.push_tick(SBIN, 502.1, at_second(1, 10)),
            vec![modify_stop(50.0, STOP_LEVEL)],
        ),
        (
            "the cut is confirmed",
            |k| k.apply_modify(STOP).map(|_| ()),
            vec![place_exit(3, 50.0)],
        ),
        (
            "the target fills 2 s after the cut was sent: the stop moves to 500.10 later",
            |k| {
                k.apply_fill(OrderId(3), 50.0, 502.1, at_second(1, 12))
                    .map(|_| ())
            },
            vec![],
        ),
        (
            "5 s after the cut",
            |k| k.push_tick(SBIN, 502.0, at_second(1, 15)),
            vec![modify_stop(50.0, 500.1)],
        ),
    ];
    run_steps(&mut kernel, steps);
}

#[test]
fn protect_places_the_stop_again_only_when_the_leg_has_none_and_no_exit_works() {
    type Setup = fn() -> Engine;
    let rows: [(&str, Setup, Protect); 5] = [
        (
            "flat",
            || {
                let mut kernel = session(config());
                pause_at_close(&mut kernel, &bars(&[], &[]), 0);
                kernel
            },
            Protect::Flat,
        ),
        (
            "an entry with no stop rule",
            || holding(LegRules::default()),
            Protect::NoStopLevel,
        ),
        (
            "its stop works",
            || holding(stop_points(3.0)),
            Protect::StopWorks,
        ),
        (
            "an exit waits for the stop's cancel",
            || {
                let mut kernel = holding(stop_points(3.0));
                let asked = kernel.request_exit(SBIN, Product::Mis, EngineExit::SquareOff);
                assert_eq!(asked, Ok(ExitRequest::WaitsForCancels));
                kernel
            },
            Protect::Exiting,
        ),
        (
            "the stop was cancelled for an exit the venue rejected",
            || {
                let mut kernel = holding(stop_points(3.0));
                let kill = EngineExit::Risk(RiskExitReason::KillSwitch);
                assert!(kernel.request_exit(SBIN, Product::Mis, kill).is_ok());
                assert!(kernel.apply_cancel(STOP).is_ok());
                assert!(kernel.apply_reject(OrderId(3), RejectCode::Venue).is_ok());
                kernel.take_commands();
                kernel
            },
            Protect::Placed,
        ),
    ];
    for (what, setup, outcome) in rows {
        let mut kernel = setup();
        assert_eq!(kernel.protect(SBIN, Product::Mis), Ok(outcome), "{what}");
    }
}

#[test]
fn a_protected_leg_gets_its_stop_at_its_level_for_the_position() {
    let mut kernel = holding(stop_points(3.0));
    let kill = EngineExit::Risk(RiskExitReason::KillSwitch);
    assert!(kernel.request_exit(SBIN, Product::Mis, kill).is_ok());
    assert!(kernel.apply_cancel(STOP).is_ok());
    assert!(kernel.apply_reject(OrderId(3), RejectCode::Venue).is_ok());
    kernel.take_commands();
    assert_eq!(kernel.protect(SBIN, Product::Mis), Ok(Protect::Placed));
    let expected = vec![(OrderId(4), Side::Sell, 100.0, OrderRole::LegStop)];
    let commands = kernel.take_commands();
    assert_eq!(placed_roles(&commands), expected);
    let stop = kernel.order(OrderId(4)).expect("the new stop");
    assert_eq!(stop.request.order_type, sl_m(STOP_LEVEL));
}

#[test]
fn a_stop_is_repriced_to_a_limit_for_its_total_units_partly_filled_or_not() {
    // (units the stop filled before, the reprice's units)
    for (filled, units) in [(0.0, 100.0), (30.0, 100.0)] {
        let mut kernel = holding(stop_points(3.0));
        if filled > 0.0 {
            assert!(kernel
                .apply_fill(STOP, filled, 497.0, at_second(1, 5))
                .is_ok());
        }
        assert_eq!(
            kernel.reprice_exit(STOP, 494.0),
            Ok(true),
            "{filled} filled"
        );
        let expected = vec![(STOP, units, limit(494.0))];
        assert_eq!(
            modified(&kernel.take_commands()),
            expected,
            "{filled} filled"
        );
    }
}

#[test]
fn a_reprice_waits_for_the_answer_to_a_trailing_modification() {
    let rules = LegRules {
        trail: Some(TrailSpec {
            trigger: Distance::Points(2.0),
            step: None,
        }),
        ..stop_points(3.0)
    };
    let mut kernel = holding(rules);
    let steps: Vec<Step> = vec![
        (
            "the trail arms and moves the stop to 500.10",
            |k| k.push_tick(SBIN, 502.1, at_second(1, 10)),
            vec![modify_stop(100.0, 500.1)],
        ),
        (
            "the reprice waits for that modification's answer",
            |k| k.reprice_exit(STOP, 495.0).map(|sent| assert!(!sent)),
            vec![],
        ),
        (
            "it goes with the answer",
            |k| k.apply_modify(STOP).map(|_| ()),
            vec![VenueCommand::Modify {
                order: STOP,
                units: 100.0,
                order_type: limit(495.0),
            }],
        ),
        (
            "a repriced stop no longer trails",
            |k| k.push_tick(SBIN, 510.0, at_second(1, 30)),
            vec![],
        ),
    ];
    run_steps(&mut kernel, steps);
}

#[test]
fn a_repriced_stop_exits_the_position_for_the_strategys_later_exits() {
    let mut kernel = deciding();
    assert_eq!(kernel.reprice_exit(STOP, 495.0), Ok(true));
    kernel.take_commands();
    // A partial exit has nothing left to close beside a stop that exits it all.
    assert_eq!(
        kernel.submit(reduce(0.5)),
        Ok(Submission::Submitted(OrderId(3)))
    );
    let status = kernel.order(OrderId(3)).map(|order| order.status.code());
    assert_eq!(status, Some("Canceled"));
    // And an entry waits for it, as for any order in flight.
    let add = kernel.submit(market(Side::Buy, 10.0));
    assert_eq!(
        add,
        Ok(Submission::Rejected(OrderId(4), RejectCode::LiveInFlight))
    );
}

fn invalid(result: Result<bool, KernelError>) -> bool {
    matches!(
        result,
        Err(KernelError::Exec(ExecError::InvalidRequest { .. }))
    )
}

#[test]
fn only_a_working_stop_or_engine_exit_is_repriced_at_a_price_on_the_tick() {
    let mut kernel = holding(stop_points(3.0));
    assert!(invalid(kernel.reprice_exit(STOP, 495.03)), "off the tick");
    assert!(invalid(kernel.reprice_exit(STOP, 0.0)), "not a price");
    assert_eq!(
        kernel.reprice_exit(OrderId(1), 495.0),
        Err(KernelError::Venue(VenueError::OrderEnded {
            order: OrderId(1),
            status: oa_bt::exec::OrderStatus::Filled,
        })),
        "the filled entry"
    );
    pause_at_close(&mut kernel, &bars(&[], &[]), 1);
    let resting_add = OrderRequest {
        order_type: limit(495.0),
        ..market(Side::Buy, 10.0)
    };
    assert!(matches!(
        kernel.submit(resting_add),
        Ok(Submission::Submitted(_))
    ));
    assert!(invalid(kernel.reprice_exit(OrderId(3), 495.0)), "an add");
    let mut simulated = backtest(config(), bars(&[], &[]));
    assert_eq!(
        simulated.reprice_exit(STOP, 495.0),
        Err(KernelError::Venue(VenueError::NotExternal))
    );
}

#[test]
fn an_exit_at_a_limit_cancels_the_stop_then_goes_as_a_limit_and_is_repriced() {
    let mut kernel = holding(stop_points(3.0));
    let unfilled = EngineExit::Risk(RiskExitReason::StopUnfilled);
    let asked = kernel.request_limit_exit(SBIN, Product::Mis, unfilled, 490.0);
    assert_eq!(asked, Ok(ExitRequest::WaitsForCancels));
    assert_eq!(
        kernel.take_commands(),
        [VenueCommand::Cancel { order: STOP }]
    );
    kernel.apply_cancel(STOP).expect("the stop is cancelled");
    let expected = vec![VenueCommand::Place {
        order: OrderId(3),
        instrument: SBIN,
        product: Product::Mis,
        side: Side::Sell,
        units: 100.0,
        order_type: limit(490.0),
        role: OrderRole::RiskExit(RiskExitReason::StopUnfilled),
    }];
    assert_eq!(kernel.take_commands(), expected);
    assert_eq!(kernel.reprice_exit(OrderId(3), 489.0), Ok(true));
    assert_eq!(
        modified(&kernel.take_commands()),
        [(OrderId(3), 100.0, limit(489.0))]
    );
    let off_tick = kernel.request_limit_exit(SBIN, Product::Mis, unfilled, 489.01);
    assert!(matches!(
        off_tick,
        Err(KernelError::Exec(ExecError::InvalidRequest { .. }))
    ));
}
