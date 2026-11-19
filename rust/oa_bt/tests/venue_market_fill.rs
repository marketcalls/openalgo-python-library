#![cfg(all(feature = "costs", feature = "time"))]
//! The price a backtest fills a MARKET order at, asked for a bar the caller names
//! (`engine::market_fill`): what a live session's day summary sets the broker's MARKET
//! fills against (16-session-design 1.4, decided item 11).
//!
//! SBIN on NSE, tick 0.05, on the test day 2025-07-01 (minute k opens at 09:15 + k).

mod venue_support;

use oa_bt::engine::{KernelError, RunConfig};
use oa_bt::exec::InstrumentId;
use oa_bt::market::slippage::SlippageSpec;
use oa_bt::model::Side;
use venue_support::{
    at_close, bars, config, market, minute, rising, run_backtest, session, Act, Script, SBIN,
};

/// The test day's config with `slippage`.
fn slipping(slippage: SlippageSpec) -> RunConfig {
    RunConfig {
        slippage,
        ..config()
    }
}

#[test]
fn a_buy_pays_the_open_plus_the_runs_slippage_and_a_sale_receives_the_open_less_it() {
    // Two ticks of slippage are 0.10 rupees.
    let kernel = session(slipping(SlippageSpec::Ticks(2)));
    let at_open = |side| kernel.market_fill_price(SBIN, side, minute(3), 500.0);
    assert_eq!(at_open(Side::Buy), Ok(500.1));
    assert_eq!(at_open(Side::Sell), Ok(499.9));
}

#[test]
fn the_default_slippage_is_the_tables_row_rounded_to_the_tick_away_from_the_trader() {
    // SBIN is large or mid cap equity: the default table's EQ_LARGE_MID row, 2 bps with a
    // floor of one tick. At an open of 817.55, 2 bps is 0.16351 rupees: a buyer pays
    // 817.71351, rounded up to the tick, 817.75; a seller receives 817.38649, rounded down,
    // 817.35. At an open of 10.00, 2 bps is 0.002, below the floor of one tick, 0.05.
    let kernel = session(slipping(SlippageSpec::SegmentDefault));
    let price = |side, open| kernel.market_fill_price(SBIN, side, minute(3), open);
    assert_eq!(price(Side::Buy, 817.55), Ok(817.75));
    assert_eq!(price(Side::Sell, 817.55), Ok(817.35));
    assert_eq!(price(Side::Buy, 10.0), Ok(10.05));
}

#[test]
fn the_price_is_the_backtests_fill_of_a_market_order_at_the_same_open() {
    // A backtest of the day with the default slippage: a MARKET buy of 10 decided at the
    // close of minute 0 fills at the open of minute 1, 500.10, worse by the slippage.
    let default = slipping(SlippageSpec::SegmentDefault);
    let acts = vec![at_close(0, Act::Submit(market(Side::Buy, 10.0)))];
    let output = run_backtest(default.clone(), &bars(&[], &[]), &mut Script::new(acts));
    let filled = output.fills.first().map(|record| record.fill.price_rupees);
    let (open, _, _, _) = rising(1);
    let asked = session(default).market_fill_price(SBIN, Side::Buy, minute(1), open);
    assert_eq!(asked.ok(), filled);
}

#[test]
fn asking_records_no_use_of_the_default_table() {
    // The run's results name the market-table rows its fills used; asking for a price is
    // no fill.
    let mut kernel = session(slipping(SlippageSpec::SegmentDefault));
    assert!(kernel
        .market_fill_price(SBIN, Side::Buy, minute(3), 500.0)
        .is_ok());
    assert!(kernel.stop_session().is_ok());
    let output = kernel.finish().expect("a stopped session finishes");
    assert!(
        output.charges.tables.is_empty(),
        "{:?}",
        output.charges.tables
    );
}

#[test]
fn an_instant_that_opens_no_slice_or_an_open_not_above_zero_is_refused() {
    let kernel = session(config());
    let off_grid = kernel.market_fill_price(SBIN, Side::Buy, minute(3) + 1, 500.0);
    assert!(
        matches!(off_grid, Err(KernelError::InvalidData { .. })),
        "{off_grid:?}"
    );
    for open in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        let refused = kernel.market_fill_price(SBIN, Side::Buy, minute(3), open);
        assert!(
            matches!(refused, Err(KernelError::InvalidData { .. })),
            "{open}"
        );
    }
}

#[test]
fn an_instrument_the_run_does_not_trade_is_refused() {
    let kernel = session(config());
    let unknown = kernel.market_fill_price(InstrumentId(1), Side::Buy, minute(3), 500.0);
    assert!(matches!(
        unknown,
        Err(KernelError::InvalidConfig {
            field: "symbol",
            ..
        })
    ));
}
