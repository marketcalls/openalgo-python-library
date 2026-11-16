#![cfg(all(feature = "costs", feature = "time"))]
//! Tick mode in a live session (`engine::venue_ticks`; 03-architecture 4.2, last
//! paragraph; 16-session-design 4.3 and 4.4, requirements R5 and R6): a tick ratchets the
//! legs, exits a target it reaches, marks the position and tests the daily loss limit,
//! and the broker's stop follows a trail under its throttle.
//!
//! The position is 100 SBIN in MIS bought at 500.10 at the open of minute 1 (o1), with a
//! stop 3 points below (o2 at 497.10) where a test gives one.

mod venue_support;

use oa_bt::engine::{
    Engine, EngineExit, ExitRequest, KernelError, OrderRole, Pause, RiskExitReason, RunConfig,
    VenueCommand, VenueError,
};
use oa_bt::exec::{InstrumentId, OrderId, OrderType};
use oa_bt::model::{Product, Side};
use oa_bt::rms::{BreakevenSpec, Distance, LegRules, StrategyRiskConfig, TradingState, TrailSpec};
use venue_support::{
    at_close, at_second, backtest, bars, config, holding_with, market, minute, pause_at_close,
    placed_roles, session, stop_points, wait_before, Act, Script, SBIN,
};

const STOP: OrderId = OrderId(2);

/// Holding 100 with `rules`, the commands of the fill taken, waiting before minute 1.
fn holding(rules: LegRules) -> Engine {
    let mut kernel = holding_with(100.0, rules);
    kernel.take_commands();
    kernel
}

fn trail(points: f64) -> Option<TrailSpec> {
    Some(TrailSpec {
        trigger: Distance::Points(points),
        step: None,
    })
}

#[test]
fn a_tick_is_refused_with_the_reason_it_cannot_be_taken() {
    let mut paused = session(config());
    pause_at_close(&mut paused, &bars(&[], &[]), 0);
    let mut stopped = session(config());
    assert!(stopped.stop_session().is_ok());
    let mut simulated = backtest(config(), bars(&[], &[]));
    let venue = |error| Err(KernelError::Venue(error));
    let refusals = [
        (
            "in a backtest",
            simulated.push_tick(SBIN, 500.0, minute(0)),
            venue(VenueError::NotExternal),
        ),
        (
            "at a bar close",
            paused.push_tick(SBIN, 500.0, minute(1)),
            venue(VenueError::AtBarClose),
        ),
        (
            "after the session",
            stopped.push_tick(SBIN, 500.0, minute(0)),
            venue(VenueError::SessionOver),
        ),
    ];
    for (what, result, expected) in refusals {
        assert_eq!(result, expected, "{what}");
    }
    for price in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        let mut kernel = holding(LegRules::default());
        let result = kernel.push_tick(SBIN, price, at_second(1, 1));
        assert!(
            matches!(result, Err(KernelError::InvalidData { .. })),
            "{price}: {result:?}"
        );
    }
    let mut kernel = holding(LegRules::default());
    let unknown = kernel.push_tick(InstrumentId(5), 500.0, at_second(1, 1));
    assert!(matches!(
        unknown,
        Err(KernelError::InvalidConfig {
            field: "symbol",
            ..
        })
    ));
}

#[test]
fn a_tick_marks_the_position_at_its_price() {
    let mut kernel = holding(LegRules::default());
    kernel
        .push_tick(SBIN, 505.0, at_second(1, 1))
        .expect("the tick is taken");
    let mark = kernel
        .position(SBIN, Product::Mis)
        .and_then(|p| p.mark_rupees);
    assert_eq!(mark, Some(505.0));
    let open_pnl = kernel.account().unrealized_rupees;
    assert!((open_pnl - 490.0).abs() < 1e-6, "{open_pnl}");
}

#[test]
fn a_tick_past_the_daily_loss_limit_exits_the_position_at_market() {
    let limited = RunConfig {
        risk: StrategyRiskConfig {
            max_daily_loss_rupees: Some(1_000.0),
            ..StrategyRiskConfig::default()
        },
        ..config()
    };
    let mut kernel = session(limited);
    let acts = vec![at_close(0, Act::Submit(market(Side::Buy, 100.0)))];
    wait_before(&mut kernel, &bars(&[], &[]), &mut Script::new(acts), 1);
    kernel.take_commands();
    assert!(kernel
        .apply_fill(OrderId(1), 100.0, 500.1, minute(1))
        .is_ok());
    // (the tick's price, the risk exits it sends): a loss of Rs 1,000 or more breaches.
    let ticks = [
        (491.0, vec![]),
        (
            490.0,
            vec![(OrderId(2), Side::Sell, 100.0, RiskExitReason::DailyLoss)],
        ),
        (489.0, vec![]),
    ];
    for (second, (price, exits)) in (1..).zip(ticks) {
        kernel
            .push_tick(SBIN, price, at_second(1, second))
            .expect("the tick is taken");
        let expected: Vec<(OrderId, Side, f64, OrderRole)> = exits
            .into_iter()
            .map(|(order, side, units, reason)| (order, side, units, OrderRole::RiskExit(reason)))
            .collect();
        assert_eq!(placed_roles(&kernel.take_commands()), expected, "{price}");
    }
    assert_eq!(kernel.trading_state(), TradingState::Reducing);
}

#[test]
fn a_target_reached_on_a_tick_exits_once_while_its_exit_works() {
    let rules = LegRules {
        target: Some(Distance::Points(2.0)),
        ..LegRules::default()
    };
    let mut kernel = holding(rules);
    let ticks = [
        (502.0, vec![]),
        (
            502.1,
            vec![(OrderId(2), Side::Sell, 100.0, OrderRole::Exit)],
        ),
        (502.4, vec![]),
    ];
    for (second, (price, expected)) in (1..).zip(ticks) {
        kernel
            .push_tick(SBIN, price, at_second(1, second))
            .expect("the tick is taken");
        assert_eq!(placed_roles(&kernel.take_commands()), expected, "{price}");
    }
}

#[test]
fn a_target_reached_while_the_engines_exit_closes_the_position_makes_no_exit() {
    let rules = LegRules {
        target: Some(Distance::Points(2.0)),
        ..LegRules::default()
    };
    let mut kernel = holding(rules);
    let square_off = kernel.request_exit(SBIN, Product::Mis, EngineExit::SquareOff);
    assert_eq!(square_off, Ok(ExitRequest::Sent));
    kernel.take_commands();
    for second in [1, 2, 3] {
        kernel
            .push_tick(SBIN, 502.5, at_second(1, second))
            .expect("the tick is taken");
        assert!(kernel.take_commands().is_empty(), "second {second}");
    }
    assert!(
        kernel.order(OrderId(3)).is_none(),
        "no target exit was made"
    );
}

#[test]
fn a_breakeven_armed_on_a_tick_moves_the_brokers_stop() {
    let rules = LegRules {
        breakeven: Some(BreakevenSpec {
            trigger: Distance::Points(2.0),
            offset: None,
        }),
        ..stop_points(3.0)
    };
    let mut kernel = holding(rules);
    kernel
        .push_tick(SBIN, 502.1, at_second(1, 1))
        .expect("the tick is taken");
    let leg = kernel.leg(SBIN, Product::Mis).expect("the leg");
    assert!(leg.breakeven.is_some_and(|state| state.armed));
    let expected = [VenueCommand::Modify {
        order: STOP,
        units: 100.0,
        order_type: OrderType::StopMarket {
            trigger_rupees: 500.1,
        },
    }];
    assert_eq!(kernel.take_commands(), expected);
}

/// A step of the trailing script: the tick's price at second s of minute 1, or the venue's
/// answer to the stop's modification (`None`); the trigger the session then sends.
type TrailStep = (&'static str, Option<(f64, i64)>, Answer, Option<f64>);

#[derive(Clone, Copy, Debug)]
enum Answer {
    None,
    Confirmed,
    Refused,
}

#[test]
fn the_brokers_stop_trails_by_a_quarter_percent_at_most_once_every_five_seconds() {
    // Trail 2 points behind the best price once 2 points up; the stop starts at 497.10.
    // 0.25 percent of 500.10 is 1.25 rupees.
    let rules = LegRules {
        trail: trail(2.0),
        ..stop_points(3.0)
    };
    let mut kernel = holding(rules);
    let steps: [TrailStep; 10] = [
        (
            "armed at 502.10: the stop goes to 500.10",
            Some((502.1, 10)),
            Answer::None,
            Some(500.1),
        ),
        ("confirmed", None, Answer::Confirmed, None),
        (
            "501.00 is 0.90 above: too little",
            Some((503.0, 11)),
            Answer::None,
            None,
        ),
        (
            "501.40 is enough, 2 s after the last",
            Some((503.4, 12)),
            Answer::None,
            None,
        ),
        (
            "5 s after the last",
            Some((503.4, 15)),
            Answer::None,
            Some(501.4),
        ),
        (
            "504.00 while that one waits",
            Some((506.0, 17)),
            Answer::None,
            None,
        ),
        (
            "confirmed 2 s after it was sent: the throttle holds",
            None,
            Answer::Confirmed,
            None,
        ),
        (
            "5 s after the last",
            Some((506.0, 20)),
            Answer::None,
            Some(504.0),
        ),
        ("refused: the old stop stands", None, Answer::Refused, None),
        (
            "sent again 5 s later",
            Some((506.0, 25)),
            Answer::None,
            Some(504.0),
        ),
    ];
    for (what, tick, answer, trigger) in steps {
        let result = match (tick, answer) {
            (Some((price, second)), _) => kernel.push_tick(SBIN, price, at_second(1, second)),
            (None, Answer::Confirmed) => kernel.apply_modify(STOP).map(|_| ()),
            (None, Answer::Refused) => kernel.apply_modify_refused(STOP),
            (None, Answer::None) => Ok(()),
        };
        result.expect("the step is taken");
        let expected: Vec<VenueCommand> = trigger
            .map(|trigger_rupees| VenueCommand::Modify {
                order: STOP,
                units: 100.0,
                order_type: OrderType::StopMarket { trigger_rupees },
            })
            .into_iter()
            .collect();
        assert_eq!(kernel.take_commands(), expected, "{what}");
    }
}

#[test]
fn a_trail_with_no_stop_rule_places_the_brokers_stop_once_it_arms() {
    let rules = LegRules {
        trail: trail(2.0),
        ..LegRules::default()
    };
    let mut kernel = holding(rules);
    kernel
        .push_tick(SBIN, 501.0, at_second(1, 1))
        .expect("the tick is taken");
    assert!(kernel.take_commands().is_empty(), "not armed: no stop yet");
    kernel
        .push_tick(SBIN, 502.1, at_second(1, 2))
        .expect("the tick is taken");
    let expected = vec![(STOP, Side::Sell, 100.0, OrderRole::LegStop)];
    assert_eq!(placed_roles(&kernel.take_commands()), expected);
}

#[test]
fn a_session_does_not_ratchet_on_its_bars() {
    // The bar of minute 1 reaches 502.60, past the trail's trigger, but only ticks move the
    // legs of a live session: the broker's stop stays at 497.10.
    let rules = LegRules {
        trail: trail(2.0),
        ..stop_points(3.0)
    };
    let mut kernel = holding(rules);
    let high_bar = [(1, (500.1, 502.6, 500.0, 502.5))];
    pause_at_close(&mut kernel, &bars(&high_bar, &[]), 1);
    assert!(matches!(kernel.advance(), Ok(Pause::Waiting { .. })));
    assert!(kernel.take_commands().is_empty());
    let leg = kernel.leg(SBIN, Product::Mis).expect("the leg");
    assert_eq!(leg.stop.map(|stop| stop.level_rupees), Some(497.1));
}
