//! The merged time line (18-multi-strategy 2.3, 2.6): engines on other exchanges take only
//! the instants of their own grids, the account's rows follow every round, and engines
//! that end at different times leave the fund going with their last figures in the
//! ledger. Each engine's output is its output alone.

use crate::engine::Engine;
use crate::model::{Product, Side};

use super::super::Pool;
use super::kit::{account, alone, exit_all, flat, in_fund, ist, july, market, wave};
use super::kit::{Run, Script, Symbol, MCX_MINUTES, NSE_MINUTES};

#[test]
fn an_nse_and_an_mcx_strategy_share_the_account_minute_by_minute() {
    // SBIN on NSE (09:15 to 15:30) and a CRUDEOIL future on MCX (09:00 to 23:30) on
    // 2025-07-07. Every NSE close is a 1-minute MCX close too, so the rounds are MCX's 870
    // closes, from 09:01 to 23:30; the NSE engine takes part in 375 of them.
    let sbin = || Run::minutes(Symbol::Sbin, july(7), july(7), 1_000_000.0);
    let crude = || Run::minutes(Symbol::Crude, july(7), july(7), 500_000.0);
    let sbin_script = Script::new(&[
        (5, market(Side::Buy, 100.0, Product::Mis)),
        (200, exit_all(Side::Sell, Product::Mis)),
    ]);
    let crude_script = Script::new(&[
        (30, market(Side::Buy, 100.0, Product::Nrml)),
        (700, exit_all(Side::Sell, Product::Nrml)),
    ]);
    let engines = || -> Vec<Engine> {
        vec![
            sbin().engine(|day, k| wave(100.0, day, k)),
            crude().engine(|day, k| wave(6_000.0, day, k)),
        ]
    };
    let solo: Vec<_> = engines()
        .into_iter()
        .zip([&sbin_script, &crude_script])
        .map(|(engine, script)| alone(engine, script))
        .collect();
    let scripts = [sbin_script, crude_script];
    let (outputs, fund) = in_fund(account(Pool::Separate, None, None), engines(), &scripts);
    assert_eq!(outputs, solo);
    let rows = &fund.account.ts;
    assert_eq!(rows.len(), MCX_MINUTES as usize);
    assert_eq!(
        (rows.first(), rows.last()),
        (Some(&ist(july(7), 9, 1)), Some(&ist(july(7), 23, 30)))
    );
    assert_eq!(solo[0].equity.ts.len(), NSE_MINUTES as usize);
}

#[test]
fn an_engine_that_ends_first_leaves_its_last_figures_in_the_account() {
    // SBIN runs on 2025-07-07 and 08, INFY on the 07th only, flat at 100.00 and 200.00, in
    // a separate pool of Rs 3,00,000 (1,00,000 and 2,00,000). INFY buys 100 CNC at 200.00
    // and ends at 15:30 on the 07th holding it; the fund goes on with SBIN through the
    // 08th, and every row of the 08th still holds INFY's equity of 2,00,000 and its margin
    // of 20,000: equity 3,00,000, margin 20,000, free cash 2,80,000.
    let sbin = Run::minutes(Symbol::Sbin, july(7), july(8), 100_000.0);
    let infy = Run::minutes(Symbol::Infy, july(7), july(7), 200_000.0);
    let infy_buy = Script::new(&[(5, market(Side::Buy, 100.0, Product::Cnc))]);
    let engines = || {
        vec![
            sbin.engine(|_, _| flat(100.0)),
            infy.engine(|_, _| flat(200.0)),
        ]
    };
    let scripts = [Script::default(), infy_buy];
    let solo: Vec<_> = engines()
        .into_iter()
        .zip(&scripts)
        .map(|(e, s)| alone(e, s))
        .collect();
    let (outputs, fund) = in_fund(account(Pool::Separate, None, None), engines(), &scripts);
    assert_eq!(outputs, solo);
    let rows = &fund.account;
    assert_eq!(rows.ts.len(), 2 * NSE_MINUTES as usize);
    let last = rows.ts.len() - 1;
    assert_eq!(rows.ts[last], ist(july(8), 15, 30));
    let figures = (
        rows.equity_rupees[last],
        rows.margin_rupees[last],
        rows.free_cash_rupees[last],
    );
    assert_eq!(figures, (300_000.0, 20_000.0, 280_000.0));
}

#[test]
fn a_daily_engine_beside_a_minute_engine_takes_part_in_the_round_at_its_close() {
    // A daily SBIN strategy and a 1-minute one on 2025-07-07 and 08: 750 rounds, the
    // daily slice closing at the 15:30 round of each day with the minute one's last
    // slice. Each output is its output alone.
    let daily = Run::daily(Symbol::Sbin, july(7), july(8), 100_000.0);
    let minutes = Run::minutes(Symbol::Sbin, july(7), july(8), 100_000.0);
    let daily_script = Script::new(&[(0, market(Side::Buy, 300.0, Product::Cnc))]);
    let minute_script = Script::new(&[
        (380, market(Side::Buy, 200.0, Product::Mis)),
        (500, exit_all(Side::Sell, Product::Mis)),
    ]);
    let engines = || {
        vec![
            daily.engine(|day, _| wave(100.0, day, 0)),
            minutes.engine(|day, k| wave(100.0, day, k)),
        ]
    };
    let scripts = [daily_script, minute_script];
    let solo: Vec<_> = engines()
        .into_iter()
        .zip(&scripts)
        .map(|(e, s)| alone(e, s))
        .collect();
    let (outputs, fund) = in_fund(account(Pool::Shared, None, None), engines(), &scripts);
    assert_eq!(outputs, solo);
    assert_eq!(fund.account.ts.len(), 2 * NSE_MINUTES as usize);
}
