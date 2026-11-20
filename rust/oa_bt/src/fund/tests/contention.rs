//! Contention for one account in a shared pool (18-multi-strategy 2.3 and 2.4), worked by
//! hand. Every strategy trades SBIN (cash equity at 1x margin, no charges, no slippage)
//! on 2025-07-07 (and 08), with Rs 1,00,000 of capital; the account's free cash is
//! `F = C - sum(c_i) + sum(f_i)`, and an opening fill needs its margin within the
//! strategy's own free cash and within `F`.

use crate::engine::{Engine, EventStamp, Step};
use crate::exec::OrderId;
use crate::model::{Product, Side};

use super::super::{FundOutput, Pool, Refusal, RefusedOrder};
use super::kit::{account, fills, flat, in_fund, ist, july, market, with_stop};
use super::kit::{Run, Script, Symbol};

const CAPITAL: f64 = 100_000.0;

/// A 1-minute SBIN strategy on 2025-07-07 with Rs 1,00,000 on bars priced by `price(k)`.
fn minutes_of(price: impl Fn(u32) -> [f64; 4]) -> Engine {
    let run = Run::minutes(Symbol::Sbin, july(7), july(7), CAPITAL);
    run.engine(|_, k| price(k))
}

/// The refusal of strategy `strategy`'s order `order` at `step` of its slice `slice`,
/// which opens at `slice_ts`.
fn refused(strategy: usize, order: u64, (slice, slice_ts): (u32, i64), step: Step) -> RefusedOrder {
    RefusedOrder {
        strategy,
        order: OrderId(order),
        stamp: EventStamp {
            slice,
            slice_ts,
            step,
        },
        refusal: Refusal::AccountMargin,
    }
}

/// The account's row at the round closing at `at`: (equity, free cash, margin).
fn row_at(fund: &FundOutput, at: i64) -> (f64, f64, f64) {
    let rows = &fund.account;
    let index = rows
        .ts
        .iter()
        .position(|&ts| ts == at)
        .expect("a round at that close");
    (
        rows.equity_rupees[index],
        rows.free_cash_rupees[index],
        rows.margin_rupees[index],
    )
}

#[test]
fn the_first_strategy_in_call_order_takes_the_account_at_a_shared_open() {
    // An account of Rs 1,00,000 for two strategies of Rs 1,00,000 each, SBIN flat at 100.
    // Both buy 600 at the close of minute 0; at the open of minute 1 the first fills
    // (60,000 of margin, F = 1,00,000 - 2,00,000 + 40,000 + 1,00,000 = 40,000) and the
    // second's 60,000 is beyond F: refused account_margin, though its own free cash is
    // 1,00,000. The first's position is squared off at 15:10 (minute 355), MIS's default.
    let buy = Script::new(&[(0, market(Side::Buy, 600.0, Product::Mis))]);
    let engines = vec![minutes_of(|_| flat(100.0)), minutes_of(|_| flat(100.0))];
    let config = account(Pool::Shared, Some(CAPITAL), None);
    let (outputs, fund) = in_fund(config, engines, &[buy.clone(), buy]);
    let minute_1 = (1, ist(july(7), 9, 16));
    let square_off = (355, 2, Side::Sell, 600.0, 100.0, "scheduler");
    assert_eq!(
        fills(&outputs[0]),
        vec![(1, 2, Side::Buy, 600.0, 100.0, "entry"), square_off]
    );
    assert_eq!(fills(&outputs[1]), vec![]);
    assert_eq!(fund.refused, vec![refused(1, 1, minute_1, Step::OpenFills)]);
    // The round closing at 09:17: equity 1,00,000, free cash 40,000, margin 60,000.
    assert_eq!(
        row_at(&fund, ist(july(7), 9, 17)),
        (100_000.0, 40_000.0, 60_000.0)
    );
}

#[test]
fn a_stop_inside_a_bar_frees_no_cash_for_another_strategy_at_that_bar_open() {
    // The account of the test above. The first strategy buys 600 at the close of minute
    // 0 with a stop 5 below (95); the second buys 600 at the close of minute 1. Minute 2
    // opens at 100 and falls to 90: the second's fill at its open (step 2) meets F =
    // 40,000 and is refused, and only then does the first's stop fill at 95 (step 3),
    // freeing 57,000. Filling the second after the stop would have looked inside the bar.
    let first = Script::new(&[(0, with_stop(Side::Buy, 600.0, Product::Mis, 5.0))]);
    let second = Script::new(&[(1, market(Side::Buy, 600.0, Product::Mis))]);
    let falling = |k: u32| match k {
        0 | 1 => flat(100.0),
        2 => [100.0, 100.0, 90.0, 92.0],
        _ => flat(92.0),
    };
    let engines = vec![minutes_of(falling), minutes_of(falling)];
    let config = account(Pool::Shared, Some(CAPITAL), None);
    let (outputs, fund) = in_fund(config, engines, &[first, second]);
    let first_fills = fills(&outputs[0]);
    assert_eq!(first_fills[0], (1, 2, Side::Buy, 600.0, 100.0, "entry"));
    let (slice, step, side, units, price, _) = first_fills[1];
    assert_eq!(
        (slice, step, side, units, price),
        (2, 3, Side::Sell, 600.0, 95.0)
    );
    assert_eq!(fills(&outputs[1]), vec![]);
    let minute_2 = (2, ist(july(7), 9, 17));
    assert_eq!(fund.refused, vec![refused(1, 1, minute_2, Step::OpenFills)]);
}

/// The second strategy's opening fill of a buy of `units` at 80.00 after the first's
/// open loss (test below) as (slice, units, price), if any, and the account's refusals.
fn after_an_open_loss(units: f64) -> (Option<(u32, f64, f64)>, Vec<RefusedOrder>) {
    let dropping = |k: u32| if k < 2 { flat(100.0) } else { flat(80.0) };
    let first = Script::new(&[(0, market(Side::Buy, 900.0, Product::Mis))]);
    let second = Script::new(&[(2, market(Side::Buy, units, Product::Mis))]);
    let engines = vec![minutes_of(dropping), minutes_of(dropping)];
    let (outputs, fund) = in_fund(account(Pool::Shared, None, None), engines, &[first, second]);
    let opened = fills(&outputs[1])
        .first()
        .map(|fill| (fill.0, fill.3, fill.4));
    (opened, fund.refused)
}

#[test]
fn one_strategy_open_loss_blocks_another_strategy_entry() {
    // The account is the sum of the capitals, Rs 2,00,000. The first strategy buys 900 at
    // 100.00 (90,000 of margin); SBIN falls to 80.00 from minute 2: its equity is 82,000
    // and its free cash 82,000 - 90,000 = -8,000, so F = 2,00,000 - 2,00,000 - 8,000 +
    // 1,00,000 = 92,000. The second buys at the close of minute 2, filling at 80.00 at
    // the open of minute 3: 1,100 units need 88,000 (filled); 1,200 need 96,000, within
    // its own 1,00,000 but beyond F (refused account_margin).
    let minute_3 = (3, ist(july(7), 9, 18));
    let cases = [
        (1_100.0, Some((3, 1_100.0, 80.0)), vec![]),
        (
            1_200.0,
            None,
            vec![refused(1, 1, minute_3, Step::OpenFills)],
        ),
    ];
    for (units, filled, refusals) in cases {
        assert_eq!(
            after_an_open_loss(units),
            (filled, refusals),
            "{units} units"
        );
    }
}

#[test]
fn a_daily_strategy_open_fill_takes_the_account_cash_at_its_open() {
    // A daily strategy buys 600 CNC at the close of 2025-07-07, filling at 100.00 at the
    // open of the 08th (09:15), where its slice opens; a 1-minute strategy buys 600 at the
    // close of 09:15-09:16 that day. The account of Rs 1,00,000 has F = 40,000 from 09:15,
    // whichever comes first in call order: the minute strategy's fill at 09:16 is refused.
    // The daily strategy's margin is the account's from its open, though its slice closes
    // at 15:30: the round closing at 09:17 shows margin 60,000 and free cash 40,000.
    let daily = || Run::daily(Symbol::Sbin, july(7), july(8), CAPITAL).engine(|_, _| flat(100.0));
    let minutes =
        || Run::minutes(Symbol::Sbin, july(7), july(8), CAPITAL).engine(|_, _| flat(100.0));
    let daily_buy = Script::new(&[(0, market(Side::Buy, 600.0, Product::Cnc))]);
    // The minute strategy's slice 375 is 09:15-09:16 on the 08th.
    let minute_buy = Script::new(&[(375, market(Side::Buy, 600.0, Product::Mis))]);
    let at_0916 = (376, ist(july(8), 9, 16));
    let config = account(Pool::Shared, Some(CAPITAL), None);
    for daily_first in [true, false] {
        let (engines, scripts, minute) = match daily_first {
            true => (
                vec![daily(), minutes()],
                vec![daily_buy.clone(), minute_buy.clone()],
                1,
            ),
            false => (
                vec![minutes(), daily()],
                vec![minute_buy.clone(), daily_buy.clone()],
                0,
            ),
        };
        let (outputs, fund) = in_fund(config, engines, &scripts);
        let daily_output = &outputs[1 - minute];
        assert_eq!(
            fills(daily_output),
            vec![(1, 2, Side::Buy, 600.0, 100.0, "entry")]
        );
        assert_eq!(
            fund.refused,
            vec![refused(minute, 1, at_0916, Step::OpenFills)]
        );
        let row = row_at(&fund, ist(july(8), 9, 17));
        assert_eq!(
            row,
            (100_000.0, 40_000.0, 60_000.0),
            "daily first: {daily_first}"
        );
    }
}
