//! The account's daily loss limit at a round (18-multi-strategy 2.5, KR4 through the fund),
//! worked by hand: two 1-minute strategies long SBIN and a daily one with an order
//! resting, an account limit of Rs 5,000, no charges and no slippage.
//!
//! SBIN is flat at 100.00 on 2025-07-07 until 09:25, then flat at 95.00. The first
//! strategy holds 600 and the second 400, both bought at 100.00 at 09:16; at the round
//! closing at 09:26 their measures are 600 x -5 = -3,000 and 400 x -5 = -2,000, so the
//! account's measure is -5,000: at the limit, a breach. Every engine takes it: the two
//! minute strategies record it at once (that slice's step 6) and exit at the next open,
//! 09:26, at 95.00; the second's resting LIMIT buy is cancelled with it, and its buy
//! decided at that close is refused `account_loss_gate`. The daily strategy's slice is
//! open until 15:30: its resting buy is cancelled at that slice's step 3 and the breach
//! recorded at its step 6.

use crate::engine::{AccountBreach, AccountRule, Event, EventStamp, RunOutput, Step};
use crate::exec::{OrderId, OrderStatus};
use crate::model::{Product, Side};
use crate::rms::RiskEventName;

use super::super::{FundOutput, Pool, Refusal, RefusedOrder};
use super::kit::{account, fills, flat, in_fund, ist, july, limit_buy, market};
use super::kit::{Run, Script, Symbol};

const CAPITAL: f64 = 100_000.0;
const LIMIT: f64 = 5_000.0;

/// The run of the module docs: (each engine's output, the fund's).
fn breached() -> (Vec<RunOutput>, FundOutput) {
    let minutes = || {
        let run = Run::minutes(Symbol::Sbin, july(7), july(7), CAPITAL);
        run.engine(|_, k| if k < 10 { flat(100.0) } else { flat(95.0) })
    };
    let daily = Run::daily(Symbol::Sbin, july(4), july(7), CAPITAL);
    let daily = daily.engine(|day, _| {
        if day == 0 {
            flat(100.0)
        } else {
            [100.0, 100.0, 95.0, 95.0]
        }
    });
    let first = Script::new(&[(0, market(Side::Buy, 600.0, Product::Mis))]);
    let second = Script::new(&[
        (0, market(Side::Buy, 400.0, Product::Mis)),
        (5, limit_buy(100.0, 90.0, Product::Mis)),
        (10, market(Side::Buy, 100.0, Product::Mis)),
    ]);
    let resting = Script::new(&[(0, limit_buy(100.0, 50.0, Product::Cnc))]);
    let config = account(Pool::Shared, None, Some(LIMIT));
    in_fund(
        config,
        vec![minutes(), minutes(), daily],
        &[first, second, resting],
    )
}

fn stamp(slice: u32, slice_ts: i64, step: Step) -> EventStamp {
    EventStamp {
        slice,
        slice_ts,
        step,
    }
}

#[test]
fn the_account_breaches_once_at_the_round_its_measure_reaches_the_limit() {
    let (_, fund) = breached();
    let at = ist(july(7), 9, 26);
    let breach = AccountBreach {
        at,
        day: july(7),
        rule: AccountRule::Loss,
        limit_rupees: LIMIT,
        measure_rupees: -5_000.0,
    };
    assert_eq!(fund.breaches, vec![breach]);
    let row = fund
        .account
        .ts
        .iter()
        .position(|&ts| ts == at)
        .expect("the round");
    assert_eq!(fund.account.daily_loss_rupees[row], -5_000.0);
    // Equity: 3,00,000 - 3,00,000 + 97,000 + 98,000 + 1,00,000.
    assert_eq!(fund.account.equity_rupees[row], 295_000.0);
    // The exits realise the loss: the measure stays at -5,000 to the close, no second
    // breach.
    assert_eq!(fund.account.daily_loss_rupees.last(), Some(&-5_000.0));
}

#[test]
fn every_minute_strategy_exits_at_the_next_open_for_the_account_limit() {
    let (outputs, _) = breached();
    let exits = [(0, 600.0), (1, 400.0)];
    for (strategy, units) in exits {
        let expected = vec![
            (1, 2, Side::Buy, units, 100.0, "entry"),
            (11, 2, Side::Sell, units, 95.0, "account_loss_limit"),
        ];
        assert_eq!(fills(&outputs[strategy]), expected, "strategy {strategy}");
    }
}

#[test]
fn the_account_ends_resting_entries_and_refuses_new_ones_after_a_breach() {
    let (_, fund) = breached();
    let minute_10 = ist(july(7), 9, 25);
    let daily_open = ist(july(7), 9, 15);
    let row = |strategy, order, stamp, refusal| RefusedOrder {
        strategy,
        order: OrderId(order),
        stamp,
        refusal,
    };
    let expected = vec![
        // Phase A: the second's resting LIMIT, cancelled where the breach is recorded.
        row(
            1,
            2,
            stamp(10, minute_10, Step::AggregateRules),
            Refusal::AccountLoss,
        ),
        // Phase C of the same round: its buy decided at that close.
        row(
            1,
            3,
            stamp(10, minute_10, Step::Callbacks),
            Refusal::AccountLossGate,
        ),
        // 15:30: the daily strategy's resting buy, cancelled before its slice's levels.
        row(
            2,
            1,
            stamp(1, daily_open, Step::Levels),
            Refusal::AccountLoss,
        ),
    ];
    assert_eq!(fund.refused, expected);
}

#[test]
fn a_strategy_whose_slice_is_open_records_the_breach_at_that_slice_step_6() {
    let (outputs, _) = breached();
    let daily = &outputs[2];
    let hits: Vec<EventStamp> = daily
        .events
        .iter()
        .filter_map(|event| match event {
            Event::Risk { stamp, event, .. } if event.name == RiskEventName::AccountLossHit => {
                Some(*stamp)
            }
            _ => None,
        })
        .collect();
    let daily_open = ist(july(7), 9, 15);
    assert_eq!(hits, vec![stamp(1, daily_open, Step::AggregateRules)]);
    assert_eq!(daily.orders[0].status, OrderStatus::Canceled);
}
