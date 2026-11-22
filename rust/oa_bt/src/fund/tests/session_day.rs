//! The account's measure counts each strategy on the round's session day only
//! (18-multi-strategy 2.5, H3, KR9), through the fund, worked by hand.
//!
//! An NSE strategy on 2025-07-07 only buys 600 SBIN MIS at 100.00 at 09:16; SBIN trades at
//! 90.00 from 10:55 (minute 100), and the position is squared off at 15:10 at 90.00: its
//! measure is -6,000 on the 7th, where its run ends. An MCX strategy on the 7th and 8th
//! buys one lot (100) of CRUDEOIL at 6,000 at 09:03 on the 7th and holds it; crude falls
//! to 5,970, a loss of 3,000. The account's limit is Rs 8,000, the strategies' capitals
//! Rs 1,00,000 and Rs 5,00,000, no charges and no slippage.
//!
//! - Crude falls on the 8th: at MCX's rounds of the 8th the NSE strategy's -6,000 is of
//!   the 7th and counts 0, so the account's measure is -3,000; no breach (counting
//!   yesterday's loss would make -9,000, a breach).
//! - Crude falls at 20:40 on the 7th: at the round closing at 20:41 the account's measure
//!   is -6,000 - 3,000 = -9,000, a breach; the NSE strategy's run has ended, and the MCX
//!   strategy exits at the next open, 20:41, at 5,970.

use crate::engine::{AccountBreach, AccountRule, RunOutput};
use crate::model::{Product, Side};

use super::super::{FundOutput, Pool};
use super::kit::{account, fills, flat, in_fund, ist, july, market, Run, Script, Symbol};

const LIMIT: f64 = 8_000.0;

/// The run of the module docs, crude trading at 5,970 from slice `k` of day `day` (0 the
/// 7th, 1 the 8th) on.
fn across_exchanges(day: usize, k: u32) -> (Vec<RunOutput>, FundOutput) {
    let sbin = Run::minutes(Symbol::Sbin, july(7), july(7), 100_000.0);
    let sbin = sbin.engine(|_, minute| {
        if minute < 100 {
            flat(100.0)
        } else {
            flat(90.0)
        }
    });
    let crude = Run::minutes(Symbol::Crude, july(7), july(8), 500_000.0);
    let crude = crude.engine(|d, minute| match (d, minute) {
        (d, _) if d > day => flat(5_970.0),
        (d, minute) if d == day && minute >= k => flat(5_970.0),
        _ => flat(6_000.0),
    });
    let scripts = [
        Script::new(&[(0, market(Side::Buy, 600.0, Product::Mis))]),
        Script::new(&[(2, market(Side::Buy, 100.0, Product::Nrml))]),
    ];
    in_fund(
        account(Pool::Shared, None, Some(LIMIT)),
        vec![sbin, crude],
        &scripts,
    )
}

/// The account's daily loss measure at the round closing at `at`.
fn measure_at(fund: &FundOutput, at: i64) -> f64 {
    let rows = &fund.account;
    let row = rows
        .ts
        .iter()
        .position(|&ts| ts == at)
        .expect("a round at that close");
    rows.daily_loss_rupees[row]
}

#[test]
fn yesterday_nse_loss_counts_nothing_at_today_mcx_rounds() {
    let (_, fund) = across_exchanges(1, 5);
    assert_eq!(fund.breaches, vec![]);
    assert_eq!(measure_at(&fund, ist(july(7), 15, 30)), -6_000.0);
    // Crude at 5,970 from 09:05 on the 8th: the round closing at 09:06.
    assert_eq!(measure_at(&fund, ist(july(8), 9, 6)), -3_000.0);
}

#[test]
fn the_same_day_losses_of_both_exchanges_breach_at_an_mcx_evening_round() {
    let (outputs, fund) = across_exchanges(0, 700);
    let at = ist(july(7), 20, 41);
    let breach = AccountBreach {
        at,
        day: july(7),
        rule: AccountRule::Loss,
        limit_rupees: LIMIT,
        measure_rupees: -9_000.0,
    };
    assert_eq!(fund.breaches, vec![breach]);
    let exit = (701, 2, Side::Sell, 100.0, 5_970.0, "account_loss_limit");
    assert_eq!(fills(&outputs[1]).last(), Some(&exit));
}
