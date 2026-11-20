//! The strategy's daily limits in one engine (18-multi-strategy 2.5a rules 1 to 5 and 11),
//! one behaviour per test.
//!
//! SBIN on NSE in July 2025 with 10,00,000 of capital, no charges and no slippage, on
//! 1-minute bars flat at 100.00 unless a test says otherwise. An order decided at the
//! close of slice k fills at the open of slice k + 1. Every expected value is worked by
//! hand from those prices.

use crate::engine::account_test_kit::{
    events, fills, flat, held, july_2025, market, minute_bars, pause_at, Setup,
    MINUTES_PER_SESSION, SBIN,
};
use crate::engine::{Engine, Submission};
use crate::exec::{FillAt, OrderId, OrderRequest, OrderType, QuantitySpec, RejectCode};
use crate::model::{Product, Side};
use crate::rms::{AccountTrades, StrategyRiskConfig};

/// The engine of a one-day run flat at 100.00 with `risk`.
fn flat_day(risk: StrategyRiskConfig) -> Engine {
    let day = july_2025(1);
    Setup {
        risk,
        ..Setup::minutes(day, day)
    }
    .engine(minute_bars(&[day], |_, _| flat(100.0)))
}

/// A strategy capped at `trades` entries a day.
fn capped(trades: u32) -> StrategyRiskConfig {
    StrategyRiskConfig {
        max_trades_per_day: Some(trades),
        ..StrategyRiskConfig::default()
    }
}

/// A buy of `units` in MIS.
fn buy(units: f64) -> OrderRequest {
    market(Side::Buy, units, Product::Mis)
}

/// The exit of the whole MIS position.
fn exit_all(side: Side) -> OrderRequest {
    OrderRequest {
        product: Product::Mis,
        reduce_only: true,
        ..OrderRequest::market(SBIN, side, QuantitySpec::PositionFraction(1.0))
    }
}

/// A sale of `units` with `reverse=True`.
fn reverse_sell(units: f64) -> OrderRequest {
    OrderRequest {
        reverse: true,
        ..market(Side::Sell, units, Product::Mis)
    }
}

/// A LIMIT buy of `units` at `limit_rupees` in MIS.
fn limit_buy(units: f64, limit_rupees: f64) -> OrderRequest {
    OrderRequest {
        order_type: OrderType::Limit { limit_rupees },
        ..buy(units)
    }
}

/// The outcome of a submission, in words.
fn outcome(submission: Submission) -> String {
    match submission {
        Submission::Submitted(_) => "submitted".to_string(),
        Submission::Rejected(_, code) => format!("rejected {}", code.code()),
        Submission::Skipped(reason) => format!("skipped {reason:?}"),
    }
}

/// Run `script` (orders by the slice whose close decides them, in slice order) and return
/// each order's outcome in words, in the order submitted.
fn run(engine: &mut Engine, script: &[(usize, OrderRequest)]) -> Vec<String> {
    let mut outcomes = Vec::new();
    let mut at = None;
    for (slice, request) in script {
        if at != Some(*slice) {
            pause_at(engine, *slice);
            at = Some(*slice);
        }
        let submitted = engine.submit(request.clone()).expect("a usable order");
        outcomes.push(outcome(submitted));
    }
    outcomes
}

#[test]
fn an_entry_at_the_cap_is_refused_and_the_exit_before_it_was_not() {
    // Cap 1: the entry of slice 0 counts; the exit of slice 2 is never refused; the entry
    // of slice 4 finds the count at the cap.
    let mut engine = flat_day(capped(1));
    let script = [(0, buy(100.0)), (2, exit_all(Side::Sell)), (4, buy(100.0))];
    let outcomes = run(&mut engine, &script);
    assert_eq!(outcomes, ["submitted", "submitted", "rejected trade_limit"]);
    assert_eq!(engine.trades_today(), 1);
}

#[test]
fn an_add_is_not_a_trade() {
    // Cap 1: the entry counts, the add of slice 2 to the open long does not.
    let mut engine = flat_day(capped(1));
    let outcomes = run(&mut engine, &[(0, buy(100.0)), (2, buy(50.0))]);
    assert_eq!(outcomes, ["submitted", "submitted"]);
    pause_at(&mut engine, 3);
    assert_eq!(held(&engine, Product::Mis), 150.0);
    assert_eq!(engine.trades_today(), 1);
}

#[test]
fn a_working_entry_counts_so_entries_decided_together_cannot_pass_the_cap() {
    // Cap 1: a LIMIT buy at 90.00 rests (no bar trades there) and counts while it works;
    // a market buy of the next slice is refused.
    let mut engine = flat_day(capped(1));
    let outcomes = run(&mut engine, &[(0, limit_buy(100.0, 90.0)), (1, buy(100.0))]);
    assert_eq!(outcomes, ["submitted", "rejected trade_limit"]);
}

#[test]
fn an_entry_that_ends_with_no_fill_stops_counting() {
    // Cap 1: the resting LIMIT buy is cancelled at slice 1, so the entry of slice 2 is the
    // day's first that counts.
    let mut engine = flat_day(capped(1));
    run(&mut engine, &[(0, limit_buy(100.0, 90.0))]);
    pause_at(&mut engine, 1);
    engine.cancel(OrderId(1)).expect("the order works");
    assert_eq!(engine.trades_today(), 0);
    let outcomes = run(&mut engine, &[(2, buy(100.0))]);
    assert_eq!(outcomes, ["submitted"]);
    assert_eq!(engine.trades_today(), 1);
}

#[test]
fn a_filled_entry_stays_counted_after_its_position_closes() {
    let mut engine = flat_day(capped(2));
    run(&mut engine, &[(0, buy(100.0)), (2, exit_all(Side::Sell))]);
    pause_at(&mut engine, 4);
    assert_eq!(
        (held(&engine, Product::Mis), engine.trades_today()),
        (0.0, 1)
    );
}

#[test]
fn the_count_starts_again_on_the_next_session_day() {
    // Cap 1 over two days: the second day's first entry is allowed.
    let (first, second) = (july_2025(1), july_2025(2));
    let mut engine = Setup {
        risk: capped(1),
        ..Setup::minutes(first, second)
    }
    .engine(minute_bars(&[first, second], |_, _| flat(100.0)));
    let next_day = MINUTES_PER_SESSION as usize;
    let script = [
        (0, buy(100.0)),
        (2, exit_all(Side::Sell)),
        (4, buy(100.0)),
        (next_day, buy(100.0)),
    ];
    let outcomes = run(&mut engine, &script);
    assert_eq!(
        outcomes,
        [
            "submitted",
            "submitted",
            "rejected trade_limit",
            "submitted"
        ]
    );
}

#[test]
fn a_reversal_at_the_cap_is_cut_to_its_close() {
    // Cap 1: long 100 from slice 1; the sale of 200 with reverse=True at slice 2 closes the
    // long at slice 3's open (100.00) and its opening 100 is refused trade_limit.
    let mut engine = flat_day(capped(1));
    let outcomes = run(&mut engine, &[(0, buy(100.0)), (2, reverse_sell(200.0))]);
    assert_eq!(outcomes, ["submitted", "submitted"]);
    pause_at(&mut engine, 3);
    assert_eq!(held(&engine, Product::Mis), 0.0);
    let close = engine.order(OrderId(2)).expect("the close");
    let rest = engine.order(OrderId(3)).expect("the opening part");
    assert_eq!(
        (close.quantity_units, close.request.reduce_only),
        (100.0, true)
    );
    assert_eq!(
        (rest.quantity_units, rest.request.side, rest.reject_code),
        (100.0, Side::Sell, Some(RejectCode::TradeLimit))
    );
    let sold: Vec<_> = fills(&engine)
        .into_iter()
        .filter(|fill| fill.2 == Side::Sell)
        .map(|fill| (fill.0, fill.3, fill.4, fill.5, fill.6))
        .collect();
    assert_eq!(sold, [(3, 100.0, 100.0, FillAt::Open, "signal")]);
}

#[test]
fn a_reversal_under_the_cap_reverses_and_counts() {
    let mut engine = flat_day(capped(2));
    run(&mut engine, &[(0, buy(100.0)), (2, reverse_sell(200.0))]);
    pause_at(&mut engine, 3);
    assert_eq!(held(&engine, Product::Mis), -100.0);
    assert_eq!(engine.trades_today(), 2);
}

#[test]
fn the_accounts_cap_counts_the_other_strategies_entries() {
    // The account's cap of 3 with 2 entries of the others: the strategy's first entry
    // passes, its second is refused account_trade_limit.
    let mut engine = flat_day(StrategyRiskConfig::default());
    let account = AccountTrades {
        limit: 3,
        others: 2,
    };
    engine
        .set_account_trades(Some(account))
        .expect("a usable cap");
    let script = [(0, buy(100.0)), (2, exit_all(Side::Sell)), (4, buy(100.0))];
    let outcomes = run(&mut engine, &script);
    assert_eq!(
        outcomes,
        ["submitted", "submitted", "rejected account_trade_limit"]
    );
}

#[test]
fn an_account_cap_of_zero_is_refused() {
    let mut engine = flat_day(StrategyRiskConfig::default());
    let zero = AccountTrades {
        limit: 0,
        others: 0,
    };
    assert!(engine.set_account_trades(Some(zero)).is_err());
}

/// A strategy whose orders are worth at most `rupees`.
fn value_capped(rupees: f64) -> StrategyRiskConfig {
    StrategyRiskConfig {
        max_order_value_rupees: Some(rupees),
        ..StrategyRiskConfig::default()
    }
}

#[test]
fn the_order_value_cap_refuses_an_entry_or_add_above_it() {
    // Cap 10,000.00 at a last close of 100.00: 101 units are 10,100.00, refused; 100 are
    // 10,000.00, allowed; an add of 1 unit (100.00) is allowed, an add of 101 refused.
    let mut engine = flat_day(value_capped(10_000.0));
    let script = [
        (0, buy(101.0)),
        (0, buy(100.0)),
        (2, buy(101.0)),
        (2, buy(1.0)),
    ];
    let outcomes = run(&mut engine, &script);
    assert_eq!(
        outcomes,
        [
            "rejected order_value_limit",
            "submitted",
            "rejected order_value_limit",
            "submitted"
        ]
    );
}

#[test]
fn a_limit_order_is_judged_at_its_limit() {
    // 100 units at a limit of 100.05 are 10,005.00, above a cap of 10,000.00, though the
    // last close is 100.00.
    let mut engine = flat_day(value_capped(10_000.0));
    let outcomes = run(&mut engine, &[(0, limit_buy(100.0, 100.05))]);
    assert_eq!(outcomes, ["rejected order_value_limit"]);
}

#[test]
fn the_order_value_cap_never_refuses_an_exit() {
    // Long 100 bought at 100.00 (10,000.00, at the cap); the price rises to 150.00 from
    // slice 2, so the exit of slice 3 is worth 15,000.00: it is never refused.
    let day = july_2025(1);
    let mut engine = Setup {
        risk: value_capped(10_000.0),
        ..Setup::minutes(day, day)
    }
    .engine(minute_bars(&[day], |_, k| {
        if k < 2 {
            flat(100.0)
        } else {
            flat(150.0)
        }
    }));
    let outcomes = run(&mut engine, &[(0, buy(100.0)), (3, exit_all(Side::Sell))]);
    assert_eq!(outcomes, ["submitted", "submitted"]);
    pause_at(&mut engine, 4);
    assert_eq!(held(&engine, Product::Mis), 0.0);
}

#[test]
fn a_reversal_whose_opening_part_is_above_the_value_cap_is_cut_to_its_close() {
    // Long 50 (5,000.00); a sale of 200 with reverse=True opens 150 short, worth 15,000.00
    // above a cap of 10,000.00: the long is closed and the 150 refused.
    let mut engine = flat_day(value_capped(10_000.0));
    run(&mut engine, &[(0, buy(50.0)), (2, reverse_sell(200.0))]);
    pause_at(&mut engine, 3);
    assert_eq!(held(&engine, Product::Mis), 0.0);
    let rest = engine.order(OrderId(3)).expect("the opening part");
    assert_eq!(
        (rest.quantity_units, rest.reject_code),
        (150.0, Some(RejectCode::OrderValueLimit))
    );
}

/// The day of the profit target: SBIN flat at 100.00 to minute 4, then flat at 110.00; a
/// buy of 100 at slice 0 (filled at 100.00 at slice 1) and a LIMIT buy of 10 at 90.00 at
/// slice 1, which rests; a target of `target` rupees. The engine stands at slice 5's pause.
fn profit_day(target: f64) -> Engine {
    let day = july_2025(1);
    let mut engine = Setup {
        risk: StrategyRiskConfig {
            daily_profit_target_rupees: Some(target),
            ..StrategyRiskConfig::default()
        },
        ..Setup::minutes(day, day)
    }
    .engine(minute_bars(&[day], |_, k| {
        if k < 5 {
            flat(100.0)
        } else {
            flat(110.0)
        }
    }));
    run(&mut engine, &[(0, buy(100.0)), (1, limit_buy(10.0, 90.0))]);
    pause_at(&mut engine, 5);
    engine
}

#[test]
fn the_profit_target_reached_records_its_event_and_reduces() {
    // At the close of slice 5 the long of 100 is marked at 110.00: +1,000.00, the target.
    let engine = profit_day(1_000.0);
    let at_five: Vec<String> = events(&engine)
        .into_iter()
        .filter(|(slice, step, _)| (*slice, *step) == (5, 6))
        .map(|(_, _, name)| name)
        .collect();
    assert_eq!(
        at_five,
        ["daily_profit_hit", "state ACTIVE -> REDUCING", "Canceled"]
    );
}

#[test]
fn the_profit_target_exits_every_position_at_the_next_open() {
    let mut engine = profit_day(1_000.0);
    pause_at(&mut engine, 6);
    let last = fills(&engine).pop().expect("the exit");
    assert_eq!(
        (last.0, last.2, last.3, last.4, last.6),
        (6, Side::Sell, 100.0, 110.0, "daily_profit_target")
    );
}

#[test]
fn entries_after_the_profit_target_are_refused_until_the_next_session() {
    let mut engine = profit_day(1_000.0);
    let outcomes = run(&mut engine, &[(7, buy(10.0))]);
    assert_eq!(outcomes, ["rejected daily_profit_gate"]);
}

#[test]
fn a_profit_below_the_target_changes_nothing() {
    // +1,000.00 against a target of 1,000.01: no event, the long stays open.
    let mut engine = profit_day(1_000.01);
    pause_at(&mut engine, 7);
    let hit = events(&engine)
        .into_iter()
        .any(|(_, _, name)| name == "daily_profit_hit");
    assert!(!hit);
    assert_eq!(held(&engine, Product::Mis), 100.0);
}
