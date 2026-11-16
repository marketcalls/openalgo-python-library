#![cfg(all(feature = "costs", feature = "time"))]
//! What a live session refuses: venue results it cannot record (`venue_fills`), and the
//! parts of a backtest its venue cannot carry yet (`engine::venue`), each with a typed
//! error.

mod venue_support;

use oa_bt::engine::{
    run, Engine, FuturesSeries, KernelError, Pause, Pending, QuantityMode, RollRule, RunConfig,
    RunSetup, Submission, TargetRequest, VenueError,
};
use oa_bt::exec::{OrderId, OrderRequest, OrderStatus, OrderType, RejectCode, Validity};
use oa_bt::market::calendar::Calendar;
use oa_bt::model::{Product, Side};
use oa_bt::rms::{BreakevenSpec, Distance, LegRules, TrailSpec};
use oa_bt::sizing::TargetSize;
use venue_support::{
    at_close, backtest, bars, config, limit, market, minute, nifty_call, order, pause_at_close,
    run_to_end, sbin, session, wait_before, Act, Script, SBIN,
};

/// Every result a venue sends, each as a call on `kernel` for order `id`.
fn every_result(kernel: &mut Engine, id: OrderId) -> [Result<(), KernelError>; 6] {
    [
        kernel.apply_fill(id, 10.0, 500.1, minute(1)).map(|_| ()),
        kernel.apply_reject(id, RejectCode::Venue).map(|_| ()),
        kernel.apply_cancel(id).map(|_| ()),
        kernel.apply_cancel_refused(id),
        kernel.apply_modify(id).map(|_| ()),
        kernel.apply_modify_refused(id),
    ]
}

/// A session with one MIS MARKET buy of 100 decided at minute 0, waiting before slice 1.
fn one_order_waiting() -> Engine {
    let mut kernel = session(config());
    let acts = vec![at_close(0, Act::Submit(market(Side::Buy, 100.0)))];
    wait_before(&mut kernel, &bars(&[], &[]), &mut Script::new(acts), 1);
    kernel
}

fn venue_error(result: Result<(), KernelError>) -> VenueError {
    match result {
        Err(KernelError::Venue(error)) => error,
        other => panic!("expected a venue error, got {other:?}"),
    }
}

#[test]
fn a_backtest_takes_no_result_from_a_venue() {
    let mut simulated = backtest(config(), bars(&[], &[]));
    let acts = vec![at_close(0, Act::Submit(market(Side::Buy, 10.0)))];
    let mut script = Script::new(acts);
    simulated.hold();
    simulated.release(1);
    run(&mut simulated, &mut script).expect("the released slice runs");
    for result in every_result(&mut simulated, OrderId(1)) {
        assert_eq!(venue_error(result), VenueError::NotExternal);
    }
}

#[test]
fn a_result_for_an_unknown_order_is_refused() {
    let mut kernel = one_order_waiting();
    for result in every_result(&mut kernel, OrderId(9)) {
        assert_eq!(result, Err(KernelError::UnknownOrder(OrderId(9))));
    }
}

#[test]
fn a_result_for_an_order_that_has_ended_is_refused() {
    let mut kernel = one_order_waiting();
    assert!(kernel
        .apply_fill(OrderId(1), 100.0, 500.1, minute(1))
        .is_ok());
    let ended = VenueError::OrderEnded {
        order: OrderId(1),
        status: OrderStatus::Filled,
    };
    for result in every_result(&mut kernel, OrderId(1)) {
        assert_eq!(venue_error(result), ended);
    }
}

#[test]
fn a_fill_of_more_units_than_the_order_has_left_is_refused() {
    let mut kernel = one_order_waiting();
    assert!(kernel
        .apply_fill(OrderId(1), 40.0, 500.1, minute(1))
        .is_ok());
    let result = kernel
        .apply_fill(OrderId(1), 61.0, 500.1, minute(1))
        .map(|_| ());
    let error = venue_error(result);
    let expected = VenueError::TooManyUnits {
        order: OrderId(1),
        units: 61.0,
        left_units: 60.0,
    };
    assert_eq!(error, expected);
    assert!(KernelError::Venue(error).is_input_error());
}

#[test]
fn a_fill_of_units_that_are_not_above_zero_is_refused() {
    let mut kernel = one_order_waiting();
    for units in [0.0, -10.0, f64::NAN, f64::INFINITY] {
        let result = kernel
            .apply_fill(OrderId(1), units, 500.1, minute(1))
            .map(|_| ());
        let error = venue_error(result);
        assert!(matches!(error, VenueError::InvalidFill { .. }), "{units}");
        assert!(KernelError::Venue(error).is_input_error(), "{units}");
    }
}

#[test]
fn a_fill_at_a_price_that_is_not_above_zero_is_refused() {
    let mut kernel = one_order_waiting();
    for price in [0.0, -500.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let result = kernel
            .apply_fill(OrderId(1), 10.0, price, minute(1))
            .map(|_| ());
        assert!(
            matches!(venue_error(result), VenueError::InvalidFill { .. }),
            "{price}"
        );
    }
    assert_eq!(kernel.order(OrderId(1)).map(|o| o.filled_units), Some(0.0));
}

#[test]
fn a_result_while_the_engine_is_paused_at_a_bar_close_is_refused() {
    let mut kernel = session(config());
    pause_at_close(&mut kernel, &bars(&[], &[]), 0);
    assert!(kernel.submit(market(Side::Buy, 10.0)).is_ok());
    for result in every_result(&mut kernel, OrderId(1)) {
        assert_eq!(venue_error(result), VenueError::AtBarClose);
    }
}

#[test]
fn a_result_after_the_session_finished_is_refused() {
    let mut kernel = one_order_waiting();
    run_to_end(&mut kernel, &bars(&[], &[]), &mut Script::new(Vec::new()));
    for result in every_result(&mut kernel, OrderId(1)) {
        assert_eq!(venue_error(result), VenueError::SessionOver);
    }
}

#[test]
fn an_answer_to_nothing_the_venue_was_sent_is_refused() {
    let mut kernel = one_order_waiting();
    let not_pending = |what| VenueError::NotPending {
        order: OrderId(1),
        what,
    };
    let cancel = kernel.apply_cancel_refused(OrderId(1));
    assert_eq!(venue_error(cancel), not_pending(Pending::Cancel));
    for result in [
        kernel.apply_modify(OrderId(1)).map(|_| ()),
        kernel.apply_modify_refused(OrderId(1)),
    ] {
        assert_eq!(venue_error(result), not_pending(Pending::Modify));
    }
}

#[test]
fn a_modification_confirmed_below_the_units_filled_is_refused() {
    let mut kernel = session(config());
    let columns = bars(&[], &[]);
    let resting = order(Side::Buy, 100.0, limit(400.0), Product::Mis);
    let acts = vec![
        at_close(0, Act::Submit(resting)),
        at_close(
            1,
            Act::Modify(
                OrderId(1),
                order(Side::Buy, 30.0, limit(401.0), Product::Mis),
            ),
        ),
    ];
    wait_before(&mut kernel, &columns, &mut Script::new(acts), 2);
    assert!(kernel
        .apply_fill(OrderId(1), 50.0, 400.0, minute(2))
        .is_ok());
    let below = VenueError::ModifyBelowFilled {
        order: OrderId(1),
        units: 30.0,
        filled_units: 50.0,
    };
    assert_eq!(
        venue_error(kernel.apply_modify(OrderId(1)).map(|_| ())),
        below
    );
}

#[test]
fn leverage_the_margin_table_does_not_allow_is_refused_before_it_is_sent() {
    let request = OrderRequest {
        leverage: Some(50.0),
        ..market(Side::Buy, 10.0)
    };
    let mut kernel = session(config());
    kernel.release(1);
    assert!(matches!(kernel.advance(), Ok(Pause::BarClose { .. })));
    let refused = kernel.submit(request);
    assert!(
        matches!(refused, Err(KernelError::Margin(_))),
        "{refused:?}"
    );
    assert!(kernel.take_commands().is_empty());
}

#[test]
fn a_short_option_is_refused_before_it_is_sent() {
    let setup = |bars| RunSetup {
        config: config(),
        calendar: Calendar::new().expect("the embedded calendar tables parse"),
        instruments: vec![nifty_call()],
        bars: vec![bars],
    };
    let mut kernel = Engine::new_session(setup(Default::default())).expect("a session");
    kernel.release(1);
    assert!(matches!(kernel.advance(), Ok(Pause::BarClose { .. })));
    let sell = order(Side::Sell, 75.0, OrderType::Market, Product::Nrml);
    let refused = kernel.submit(sell).map(|_| ());
    assert_eq!(
        not_built(refused),
        "selling an option to open a short position (short-option margin)"
    );
    assert!(kernel.take_commands().is_empty());
}

#[test]
fn an_order_whose_margin_is_above_the_free_cash_is_rejected_as_in_a_backtest() {
    // 30,000 shares near 500.00 are 1.5 crore, above the capital of 1 crore: a backtest
    // rejects the order `margin` at its fill bar, a session when it is submitted.
    let large = market(Side::Buy, 30_000.0);
    let acts = vec![at_close(0, Act::Submit(large.clone()))];
    let simulated = venue_support::run_backtest(config(), &bars(&[], &[]), &mut Script::new(acts));
    assert_eq!(simulated.orders[0].reject_code, Some(RejectCode::Margin));
    let mut kernel = session(config());
    pause_at_close(&mut kernel, &bars(&[], &[]), 0);
    let live = kernel.submit(large);
    assert_eq!(
        live,
        Ok(Submission::Rejected(OrderId(1), RejectCode::Margin))
    );
    assert!(kernel.take_commands().is_empty());
}

#[test]
fn an_entry_the_working_orders_leave_no_cash_for_is_rejected_as_in_a_backtest() {
    // Two buys of 12,000 near 500.00 in one bar, 60 lakh each, against a capital of 1
    // crore, on MIS and on CNC (two positions, so nothing is in flight on the second's):
    // a backtest fills the first and rejects the second `margin` at their fill bar, a
    // session rejects the second when it is submitted.
    let buy = |product| order(Side::Buy, 12_000.0, OrderType::Market, product);
    let acts = vec![
        at_close(0, Act::Submit(buy(Product::Mis))),
        at_close(0, Act::Submit(buy(Product::Cnc))),
    ];
    let simulated = venue_support::run_backtest(config(), &bars(&[], &[]), &mut Script::new(acts));
    let codes: Vec<Option<RejectCode>> = simulated.orders.iter().map(|o| o.reject_code).collect();
    assert_eq!(codes[..2], [None, Some(RejectCode::Margin)]);
    let mut kernel = session(config());
    pause_at_close(&mut kernel, &bars(&[], &[]), 0);
    let first = kernel.submit(buy(Product::Mis));
    assert_eq!(first, Ok(Submission::Submitted(OrderId(1))));
    assert_eq!(
        kernel.submit(buy(Product::Cnc)),
        Ok(Submission::Rejected(OrderId(2), RejectCode::Margin))
    );
}

#[test]
fn a_fill_the_margin_rule_refuses_is_recorded_not_at_all() {
    // Long 10; a sale of 10 that may reverse, with leverage 50 (o2), and a LIMIT sale of
    // 5 (o3) are sent; o3 fills, so o2's 10 units would close 5 and open 5 short at a
    // leverage the margin table refuses: nothing of the fill is recorded, not even its
    // closing part.
    let mut kernel = session(config());
    let columns = bars(&[], &[]);
    let mut script = Script::new(vec![at_close(0, Act::Submit(market(Side::Buy, 10.0)))]);
    wait_before(&mut kernel, &columns, &mut script, 1);
    assert!(kernel
        .apply_fill(OrderId(1), 10.0, 500.0, minute(1))
        .is_ok());
    pause_at_close(&mut kernel, &columns, 1);
    let leveraged_sale = OrderRequest {
        reverse: true,
        leverage: Some(50.0),
        ..market(Side::Sell, 10.0)
    };
    assert!(kernel.submit(leveraged_sale).is_ok());
    assert!(kernel
        .submit(order(Side::Sell, 5.0, limit(500.0), Product::Mis))
        .is_ok());
    wait_before(&mut kernel, &columns, &mut script, 2);
    assert!(kernel.apply_fill(OrderId(3), 5.0, 500.0, minute(2)).is_ok());
    assert!(kernel
        .apply_fill(OrderId(2), 10.0, 500.0, minute(2))
        .is_err());
    let held = kernel
        .position(SBIN, Product::Mis)
        .map(|p| p.quantity_units);
    assert_eq!(held, Some(5.0));
    let sale = kernel.order(OrderId(2)).expect("the sale");
    assert_eq!(
        (sale.status, sale.filled_units),
        (OrderStatus::Submitted, 0.0)
    );
}

/// What submitting `request` at the first bar close of a session returns.
fn submitted_in_session(request: OrderRequest) -> Result<(), KernelError> {
    let mut kernel = session(config());
    pause_at_close(&mut kernel, &bars(&[], &[]), 0);
    kernel.submit(request).map(|_| ())
}

fn not_built(result: Result<(), KernelError>) -> &'static str {
    match result {
        Err(error) => error
            .not_built_feature()
            .unwrap_or("an error that is not NotBuilt"),
        Ok(()) => "accepted",
    }
}

#[test]
fn leg_rules_are_refused_in_a_session_until_they_are_built() {
    let points = |value| Distance::Points(value);
    let rows = [
        LegRules {
            stop: Some(points(4.0)),
            ..LegRules::default()
        },
        LegRules {
            target: Some(points(6.0)),
            ..LegRules::default()
        },
        LegRules {
            stop: Some(points(4.0)),
            trail: Some(TrailSpec {
                trigger: points(2.0),
                step: None,
            }),
            ..LegRules::default()
        },
        LegRules {
            stop: Some(points(4.0)),
            breakeven: Some(BreakevenSpec {
                trigger: points(2.0),
                offset: None,
            }),
            ..LegRules::default()
        },
    ];
    for rules in rows {
        let request = OrderRequest {
            leg_rules: rules.clone(),
            ..market(Side::Buy, 10.0)
        };
        let feature = not_built(submitted_in_session(request));
        assert!(
            feature.starts_with("stop=, target="),
            "{rules:?}: {feature}"
        );
    }
}

#[test]
fn a_gtt_order_is_refused_in_a_session() {
    let request = OrderRequest {
        validity: Validity::Gtt,
        ..order(Side::Buy, 10.0, limit(490.0), Product::Cnc)
    };
    assert_eq!(
        not_built(submitted_in_session(request)),
        "a GTT order in a live session"
    );
}

#[test]
fn a_rebalance_is_refused_in_a_session() {
    let mut kernel = session(config());
    pause_at_close(&mut kernel, &bars(&[], &[]), 0);
    let target = TargetRequest {
        instrument: SBIN,
        product: Product::Cnc,
        size: TargetSize::Units(10.0),
        leverage: None,
    };
    let result = kernel.submit_rebalance(target).map(|_| ());
    assert_eq!(not_built(result), "rebalance() in a live session");
}

#[test]
fn a_futures_series_is_refused_in_a_session() {
    let mut kernel = session(config());
    let series = FuturesSeries {
        contracts: Vec::new(),
        rule: RollRule::Roll,
    };
    let result = kernel.add_futures_series(series).map(|_| ());
    let feature = not_built(result);
    assert!(feature.starts_with("a futures series"), "{feature}");
}

#[test]
fn modifying_a_partly_filled_order_is_refused_in_a_session() {
    let mut kernel = one_order_waiting();
    assert!(kernel
        .apply_fill(OrderId(1), 40.0, 500.1, minute(1))
        .is_ok());
    pause_at_close(&mut kernel, &bars(&[], &[]), 1);
    let result = kernel
        .modify(OrderId(1), market(Side::Buy, 80.0))
        .map(|_| ());
    let feature = not_built(result);
    assert!(
        feature.starts_with("modifying a partly filled order"),
        "{feature}"
    );
}

#[test]
fn a_session_in_fractional_quantities_is_refused() {
    let fractional = RunConfig {
        quantity_mode: QuantityMode::Fractional,
        ..config()
    };
    let setup = RunSetup {
        config: fractional,
        calendar: Calendar::new().expect("the embedded calendar tables parse"),
        instruments: vec![sbin()],
        bars: vec![Default::default()],
    };
    let result = Engine::new_session(setup).map(|_| ());
    assert!(matches!(
        result,
        Err(KernelError::InvalidConfig {
            field: "quantity_mode",
            ..
        })
    ));
}

#[test]
fn a_backtest_keeps_its_leg_rules_and_gtt_orders() {
    // The refusals are the session's: the same requests pass in a backtest.
    let mut simulated = backtest(config(), bars(&[], &[]));
    simulated.hold();
    simulated.release(1);
    assert!(matches!(simulated.advance(), Ok(Pause::BarClose { .. })));
    let stop = LegRules {
        stop: Some(Distance::Points(4.0)),
        ..LegRules::default()
    };
    let with_stop = OrderRequest {
        leg_rules: stop,
        ..market(Side::Buy, 10.0)
    };
    assert!(simulated.submit(with_stop).is_ok());
    let gtt = OrderRequest {
        validity: Validity::Gtt,
        ..order(Side::Buy, 10.0, limit(490.0), Product::Cnc)
    };
    assert!(simulated.submit(gtt).is_ok());
}
