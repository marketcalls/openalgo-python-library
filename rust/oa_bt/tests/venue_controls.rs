#![cfg(all(feature = "costs", feature = "time"))]
//! A live session's controls (`engine::venue_session`, requirements R7, R10 and R18 of
//! the session design): the exit the live layer asks for, a cancellation while the
//! session waits, the cap on the engine default's sizing equity, and the daily loss
//! measure. The entry locks and gates are in `venue_gates.rs`, the stop in
//! `venue_stop.rs`.
//!
//! "Holding 100" is 100 units of SBIN in MIS bought at 500.10 at the open of minute 1
//! (o1, `venue_support::holding_on`). The bars rise 0.10 a minute from 500.00
//! (`venue_support::rising`): bar k opens at 500.00 + 0.10 k and closes 0.05 above.

mod venue_support;

use oa_bt::engine::{
    run, Engine, EngineExit, KernelError, OrderRole, RiskExitReason, RunConfig, Submission,
    VenueCommand, VenueError,
};
use oa_bt::exec::{InstrumentId, OrderId, OrderRequest, OrderStatus, OrderType, QuantitySpec};
use oa_bt::model::{Product, Side};
use oa_bt::rms::{RiskBasis, StrategyRiskConfig};
use oa_bt::sizing::{SizerSpec, SkipReason};
use venue_support::{
    at_close, backtest, bars, charged_config, close_of, config, exit, holding_on, limit, market,
    minute, order, pause_at_close, placed_roles, session, stop_market, wait_before, Act, Ohlc,
    Script, SBIN,
};

const OPERATOR_STOP: EngineExit = EngineExit::Risk(RiskExitReason::OperatorStop);
const OPERATOR_STOP_ROLE: OrderRole = OrderRole::RiskExit(RiskExitReason::OperatorStop);

/// `kernel` run on, with no venue results and nothing decided, until it waits before
/// `slice`.
fn to_wait(kernel: &mut Engine, slice: usize) {
    wait_before(kernel, &bars(&[], &[]), &mut Script::new(Vec::new()), slice);
}

/// Holding 100, waiting before slice 3.
fn waiting_with_100() -> Engine {
    let mut kernel = holding_on(session(config()), 100.0);
    to_wait(&mut kernel, 3);
    kernel
}

/// A reduce-only SL-M sale of the 100 at 450.00: the strategy's own stop.
fn resting_stop() -> OrderRequest {
    exit(Side::Sell, 100.0, stop_market(450.0))
}

/// `exit()`: a reduce-only MARKET sale of the whole MIS position.
fn whole_exit() -> OrderRequest {
    OrderRequest {
        quantity: QuantitySpec::PositionFraction(1.0),
        ..exit(Side::Sell, 1.0, OrderType::Market)
    }
}

fn status(kernel: &Engine, id: u64) -> Option<OrderStatus> {
    kernel.order(OrderId(id)).map(|order| order.status)
}

#[test]
fn an_exit_asked_for_while_waiting_sells_the_whole_position_at_market() {
    let mut kernel = waiting_with_100();
    let asked = kernel.request_exit(SBIN, Product::Mis, OPERATOR_STOP);
    assert_eq!(asked, Ok(()));
    assert_eq!(
        placed_roles(&kernel.take_commands()),
        [(OrderId(2), Side::Sell, 100.0, OPERATOR_STOP_ROLE)]
    );
    let sent = kernel.order(OrderId(2)).expect("the exit");
    let terms = (sent.request.order_type, sent.request.reduce_only);
    assert_eq!(terms, (OrderType::Market, true));
    assert_eq!(sent.decided_at, minute(3), "decided at the open of slice 3");
}

#[test]
fn an_exit_asked_for_at_a_bar_close_is_decided_at_that_close() {
    let mut kernel = holding_on(session(config()), 100.0);
    let asked = kernel.request_exit(SBIN, Product::Mis, OPERATOR_STOP);
    assert_eq!(asked, Ok(()));
    let sent = kernel.order(OrderId(2)).expect("the exit");
    assert_eq!(sent.decided_at, close_of(2));
}

#[test]
fn an_exit_asked_for_cancels_the_orders_on_the_closing_side_first() {
    let mut kernel = holding_on(session(config()), 100.0);
    assert!(kernel.submit(resting_stop()).is_ok());
    to_wait(&mut kernel, 3);
    kernel.take_commands();
    assert_eq!(
        kernel.request_exit(SBIN, Product::Mis, OPERATOR_STOP),
        Ok(())
    );
    let cancel = VenueCommand::Cancel { order: OrderId(2) };
    assert_eq!(kernel.take_commands(), vec![cancel], "no exit order yet");
    assert!(kernel.apply_cancel(OrderId(2)).is_ok());
    assert_eq!(
        placed_roles(&kernel.take_commands()),
        [(OrderId(3), Side::Sell, 100.0, OPERATOR_STOP_ROLE)]
    );
}

#[test]
fn an_exit_asked_for_takes_over_from_the_strategys_exit() {
    // exit() (o3) waits for the cancellation of the strategy's stop (o2): the session's
    // exit ends it, and is sent for the whole position once the stop has ended.
    let mut kernel = holding_on(session(config()), 100.0);
    assert!(kernel.submit(resting_stop()).is_ok());
    assert!(kernel.submit(whole_exit()).is_ok());
    to_wait(&mut kernel, 3);
    kernel.take_commands();
    assert_eq!(
        kernel.request_exit(SBIN, Product::Mis, OPERATOR_STOP),
        Ok(())
    );
    assert_eq!(status(&kernel, 3), Some(OrderStatus::Canceled));
    assert!(
        kernel.take_commands().is_empty(),
        "the stop's cancel is out"
    );
    assert!(kernel.apply_cancel(OrderId(2)).is_ok());
    assert_eq!(
        placed_roles(&kernel.take_commands()),
        [(OrderId(4), Side::Sell, 100.0, OPERATOR_STOP_ROLE)]
    );
}

#[test]
fn an_exit_asked_for_a_flat_position_sends_nothing() {
    let mut kernel = session(config());
    to_wait(&mut kernel, 3);
    assert_eq!(
        kernel.request_exit(SBIN, Product::Mis, OPERATOR_STOP),
        Ok(())
    );
    assert!(kernel.take_commands().is_empty());
}

#[test]
fn an_exit_asked_for_a_position_an_engine_exit_holds_sends_nothing_more() {
    let mut kernel = waiting_with_100();
    assert_eq!(
        kernel.request_exit(SBIN, Product::Mis, OPERATOR_STOP),
        Ok(())
    );
    kernel.take_commands();
    let again = kernel.request_exit(SBIN, Product::Mis, EngineExit::SquareOff);
    assert_eq!(again, Ok(()));
    assert!(kernel.take_commands().is_empty());
    assert_eq!(kernel.order_role(OrderId(2)), Some(OPERATOR_STOP_ROLE));
}

#[test]
fn an_exit_is_asked_for_only_of_a_live_session_in_progress() {
    let mut simulated = backtest(config(), bars(&[], &[]));
    let in_a_backtest = simulated.request_exit(SBIN, Product::Mis, OPERATOR_STOP);
    assert_eq!(in_a_backtest, Err(VenueError::NotExternal.into()));
    let mut kernel = waiting_with_100();
    let unknown = kernel.request_exit(InstrumentId(7), Product::Mis, OPERATOR_STOP);
    assert!(matches!(
        unknown,
        Err(KernelError::InvalidConfig {
            field: "symbol",
            ..
        })
    ));
    assert!(kernel.stop_session().is_ok());
    let stopped = kernel.request_exit(SBIN, Product::Mis, OPERATOR_STOP);
    assert_eq!(stopped, Err(VenueError::SessionOver.into()));
}

/// A session with a resting LIMIT buy of 10 at 450.00 (o1) decided at the close of
/// minute 0, waiting before slice 2 with no command left to take.
fn resting_buy_waiting() -> Engine {
    let mut kernel = session(config());
    let buy = order(Side::Buy, 10.0, limit(450.0), Product::Mis);
    let acts = vec![at_close(0, Act::Submit(buy))];
    wait_before(&mut kernel, &bars(&[], &[]), &mut Script::new(acts), 2);
    kernel.take_commands();
    kernel
}

fn cancels_of_o1(kernel: &mut Engine) -> usize {
    let cancel = VenueCommand::Cancel { order: OrderId(1) };
    let commands = kernel.take_commands();
    commands
        .iter()
        .filter(|command| **command == cancel)
        .count()
}

#[test]
fn a_cancellation_is_sent_while_the_session_waits() {
    let mut kernel = resting_buy_waiting();
    assert_eq!(kernel.cancel(OrderId(1)), Ok(OrderStatus::Submitted));
    assert_eq!(cancels_of_o1(&mut kernel), 1);
    assert_eq!(status(&kernel, 1), Some(OrderStatus::Submitted));
}

#[test]
fn a_cancellation_the_venue_refused_is_sent_again_on_request_in_the_same_slice() {
    let mut kernel = resting_buy_waiting();
    assert!(kernel.cancel(OrderId(1)).is_ok());
    kernel.take_commands();
    assert!(kernel.apply_cancel_refused(OrderId(1)).is_ok());
    assert!(kernel.cancel(OrderId(1)).is_ok());
    assert_eq!(cancels_of_o1(&mut kernel), 1);
}

#[test]
fn a_cancellation_waiting_for_the_venues_answer_is_not_sent_twice() {
    let mut kernel = resting_buy_waiting();
    assert!(kernel.cancel(OrderId(1)).is_ok());
    assert!(kernel.cancel(OrderId(1)).is_ok());
    assert_eq!(cancels_of_o1(&mut kernel), 1);
}

#[test]
fn a_backtest_cancels_only_at_a_decision_point() {
    let mut simulated = backtest(config(), bars(&[], &[]));
    let buy = order(Side::Buy, 10.0, limit(450.0), Product::Mis);
    simulated.hold();
    simulated.release(1);
    let mut script = Script::new(vec![at_close(0, Act::Submit(buy))]);
    run(&mut simulated, &mut script).expect("the released slice runs");
    assert_eq!(
        simulated.cancel(OrderId(1)),
        Err(KernelError::NotAtDecisionPoint)
    );
}

/// An MIS buy the engine default sizes.
fn default_sized_buy() -> OrderRequest {
    OrderRequest {
        product: Product::Mis,
        ..OrderRequest::market(
            SBIN,
            Side::Buy,
            QuantitySpec::Sizer(SizerSpec::EngineDefault),
        )
    }
}

/// What a session submitting `request` at the close of minute 0 gives it, with the
/// sizing cap `cap` set first, and the units of the order it made, if any.
fn sized_with(cap: Option<f64>, request: OrderRequest) -> (Submission, Option<f64>) {
    let mut kernel = session(config());
    pause_at_close(&mut kernel, &bars(&[], &[]), 0);
    if let Some(cap) = cap {
        assert_eq!(kernel.cap_sizing_equity(cap), Ok(()));
    }
    let outcome = kernel.submit(request).expect("a request the kernel reads");
    let units = match outcome {
        Submission::Submitted(id) | Submission::Rejected(id, _) => {
            kernel.order(id).map(|order| order.quantity_units)
        }
        Submission::Skipped(_) => None,
    };
    (outcome, units)
}

#[test]
fn the_engine_default_sizes_from_the_lesser_of_the_equity_and_the_cap() {
    // The last close is 500.05 and 2 percent of the budget is kept back: the 1 crore
    // equity buys floor(98,00,000 / 500.05) = 19,598 units, a cap of 1,00,000 buys
    // floor(98,000 / 500.05) = 195.
    let table = [
        ("no cap", None, 19_598.0),
        ("a cap below the equity", Some(100_000.0), 195.0),
        ("a cap above the equity", Some(20_000_000.0), 19_598.0),
    ];
    for (what, cap, units) in table {
        let (outcome, sized) = sized_with(cap, default_sized_buy());
        assert_eq!(outcome, Submission::Submitted(OrderId(1)), "{what}");
        assert_eq!(sized, Some(units), "{what}");
    }
}

#[test]
fn a_cap_of_nothing_leaves_the_engine_default_nothing_to_buy() {
    let (outcome, _) = sized_with(Some(0.0), default_sized_buy());
    assert_eq!(outcome, Submission::Skipped(SkipReason::RoundedToZero));
}

#[test]
fn an_order_of_a_given_quantity_is_not_capped() {
    // 1,000 units near 500.00 are worth about 5 lakh, above the cap of 1,00,000.
    let (outcome, units) = sized_with(Some(100_000.0), market(Side::Buy, 1_000.0));
    assert_eq!(outcome, Submission::Submitted(OrderId(1)));
    assert_eq!(units, Some(1_000.0));
}

#[test]
fn the_sizing_cap_is_rupees_of_zero_or_more_in_a_live_session() {
    let mut kernel = session(config());
    for rupees in [-1.0, f64::NAN, f64::INFINITY] {
        let refused = kernel.cap_sizing_equity(rupees);
        assert!(
            matches!(
                refused,
                Err(KernelError::InvalidConfig {
                    field: "sizing_equity_cap",
                    ..
                })
            ),
            "{rupees}"
        );
    }
    let mut simulated = backtest(config(), bars(&[], &[]));
    let in_a_backtest = simulated.cap_sizing_equity(100_000.0);
    assert_eq!(in_a_backtest, Err(VenueError::NotExternal.into()));
}

/// A crash at minute 10: it closes at 494.05.
const CRASH: (u32, Ohlc) = (10, (494.0, 494.5, 493.5, 494.05));

#[test]
fn the_daily_loss_measure_is_the_days_pnl_of_the_strategy() {
    // Bought 100 at 500.10. Marked at the close of minute 2 (500.25): +15.00. Marked at
    // the crash's close (494.05): -605.00. Sold at 500.40 in minute 4: +30.00, flat.
    let flat = session(config());
    assert_eq!(flat.daily_loss_measure(), 0.0);
    let mut marked = waiting_with_100();
    assert_eq!(marked.daily_loss_measure(), 15.0);
    let crash = bars(&[CRASH], &[]);
    wait_before(&mut marked, &crash, &mut Script::new(Vec::new()), 11);
    assert_eq!(marked.daily_loss_measure(), -605.0);
    let mut closed = holding_on(session(config()), 100.0);
    assert!(closed.submit(market(Side::Sell, 100.0)).is_ok());
    to_wait(&mut closed, 4);
    assert!(closed
        .apply_fill(OrderId(2), 100.0, 500.4, minute(4) + 1)
        .is_ok());
    assert_eq!(closed.daily_loss_measure(), 30.0);
}

#[test]
fn the_daily_loss_measure_on_the_net_basis_is_less_the_charges() {
    let net = RunConfig {
        risk: StrategyRiskConfig {
            risk_basis: RiskBasis::Net,
            ..StrategyRiskConfig::default()
        },
        ..charged_config()
    };
    let mut kernel = holding_on(session(net), 100.0);
    to_wait(&mut kernel, 3);
    let charges = kernel.account().costs_rupees;
    assert!(charges > 0.0, "the buy was charged");
    let expected = 15.0 - charges;
    assert!((kernel.daily_loss_measure() - expected).abs() < 0.005);
}
