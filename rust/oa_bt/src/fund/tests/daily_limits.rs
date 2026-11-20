//! The account's daily profit target and trade cap through the fund (18-multi-strategy
//! 2.5a rules 6 and 7), worked by hand.
//!
//! Two 1-minute strategies on SBIN on 2025-07-07, Rs 1,00,000 each, no charges and no
//! slippage; an order decided at the close of slice k fills at the open of slice k + 1
//! (slice k opens at 09:15 + k IST).
//!
//! - **Profit target** of Rs 1,000: SBIN is flat at 100.00 to minute 4, then at 105.00.
//!   Each strategy buys 100 at slice 0 (filled at 100.00 at slice 1); at the round closing
//!   at 09:21 (slice 5) each is up 100 x 5 = 500, the account 1,000: the target. Both
//!   record `account_profit_hit` at slice 5, step 6, exit at slice 6's open at 105.00 with
//!   the reason `account_profit_target`, and the first's buy decided at slice 7 is refused
//!   `account_profit_gate`.
//! - **Trade cap** of 2 entries, in a separate pool: the first buys at slice 0, the second
//!   at slice 0 (one entry each: 2 in all, the second's allowed since only the first's
//!   counted when it decided), the first exits at slice 2 and buys again at slice 4: its
//!   own 1 and the second's 1 reach the cap, so it is refused `account_trade_limit`. No
//!   position is exited for it.

use crate::engine::{AccountBreach, AccountRule, EventStamp, Step};
use crate::exec::{OrderId, QuantitySpec};
use crate::model::{Product, Side};

use super::super::{Pool, Refusal, RefusedOrder};
use super::kit::{account, fills, flat, in_fund, ist, july, market, Run, Script, Symbol, ONE};
use crate::exec::OrderRequest;

const CAPITAL: f64 = 100_000.0;

fn engine(price: impl Fn(u32) -> f64) -> crate::engine::Engine {
    Run::minutes(Symbol::Sbin, july(7), july(7), CAPITAL).engine(|_, k| flat(price(k)))
}

fn exit_all() -> OrderRequest {
    OrderRequest {
        product: Product::Mis,
        reduce_only: true,
        ..OrderRequest::market(ONE, Side::Sell, QuantitySpec::PositionFraction(1.0))
    }
}

#[test]
fn the_accounts_profit_target_exits_every_strategy_and_refuses_their_entries() {
    let rising = |k| if k < 5 { 100.0 } else { 105.0 };
    let first = Script::new(&[
        (0, market(Side::Buy, 100.0, Product::Mis)),
        (7, market(Side::Buy, 10.0, Product::Mis)),
    ]);
    let second = Script::new(&[(0, market(Side::Buy, 100.0, Product::Mis))]);
    let config = super::super::FundConfig {
        account_daily_profit_target_rupees: Some(1_000.0),
        ..account(Pool::Shared, None, None)
    };
    let (outputs, fund) = in_fund(
        config,
        vec![engine(rising), engine(rising)],
        &[first, second],
    );
    let reached = AccountBreach {
        at: ist(july(7), 9, 21),
        day: july(7),
        rule: AccountRule::Profit,
        limit_rupees: 1_000.0,
        measure_rupees: 1_000.0,
    };
    assert_eq!(fund.breaches, vec![reached]);
    for output in &outputs {
        let expected = vec![
            (1, 2, Side::Buy, 100.0, 100.0, "entry"),
            (6, 2, Side::Sell, 100.0, 105.0, "account_profit_target"),
        ];
        assert_eq!(fills(output), expected);
    }
    let refused = RefusedOrder {
        strategy: 0,
        order: OrderId(2),
        stamp: EventStamp {
            slice: 7,
            slice_ts: ist(july(7), 9, 22),
            step: Step::Callbacks,
        },
        refusal: Refusal::AccountProfitGate,
    };
    assert_eq!(fund.refused, vec![refused]);
}

#[test]
fn the_accounts_trade_cap_counts_every_strategys_entries_and_exits_nothing() {
    let first = Script::new(&[
        (0, market(Side::Buy, 100.0, Product::Mis)),
        (2, exit_all()),
        (4, market(Side::Buy, 100.0, Product::Mis)),
    ]);
    let second = Script::new(&[(0, market(Side::Buy, 100.0, Product::Mis))]);
    let config = super::super::FundConfig {
        account_max_trades_per_day: Some(2),
        ..account(Pool::Separate, None, None)
    };
    let (outputs, fund) = in_fund(
        config,
        vec![engine(|_| 100.0), engine(|_| 100.0)],
        &[first, second],
    );
    let refused = RefusedOrder {
        strategy: 0,
        order: OrderId(3),
        stamp: EventStamp {
            slice: 4,
            slice_ts: ist(july(7), 9, 19),
            step: Step::Callbacks,
        },
        refusal: Refusal::AccountTradeLimit,
    };
    assert_eq!(fund.refused, vec![refused]);
    // The second strategy's long is not touched: only the MIS square-off closes it.
    let second_fills: Vec<_> = fills(&outputs[1]).into_iter().map(|f| f.5).collect();
    assert_eq!(second_fills, ["entry", "scheduler"]);
    assert!(fund.breaches.is_empty());
}

#[test]
fn a_fund_refuses_an_unusable_target_or_trade_cap() {
    let engines = [engine(|_| 100.0)];
    let lent: Vec<_> = engines.iter().collect();
    let target = super::super::FundConfig {
        account_daily_profit_target_rupees: Some(0.0),
        ..account(Pool::Shared, None, None)
    };
    let cap = super::super::FundConfig {
        account_max_trades_per_day: Some(0),
        ..account(Pool::Shared, None, None)
    };
    let field = |config| match super::super::Fund::new(config, &lent) {
        Err(super::super::FundError::InvalidConfig { field, .. }) => Some(field),
        _ => None,
    };
    assert_eq!(field(target), Some("account_daily_profit_target"));
    assert_eq!(field(cap), Some("account_max_trades_per_day"));
}
