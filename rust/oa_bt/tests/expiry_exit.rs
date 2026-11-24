#![cfg(all(feature = "costs", feature = "time"))]
//! A derivatives position exited on its positional exit day, before the exchange settles
//! the contract by delivery (design 20-positional-live section 9; unit N4 item N4-2), in a
//! backtest: at the first base-bar open at or after the exchange's square-off time, reason
//! `expiry_exit`, at the open plus slippage; at the session's last close, flagged, when no
//! bar comes at or after it. A contract with no positional exit day (an index future)
//! settles at expiry as before.
//!
//! NFO contracts whose last trading day is Thu 2025-07-31, the NFO square-off of that date
//! 15:10 IST (minute 355 of the session), slippage one tick (0.05) on a MARKET fill, no
//! charges, NRML. A MARKET buy of one lot is decided at the 09:15 close and fills at the
//! 09:16 open.

use oa_bt::engine::{
    run, BarColumns, CostConfig, Engine, FillTiming, IntradayOverrides, Perturbation, QuantityMode,
    RunConfig, RunOutput, RunSetup, SettlementKind, StrategyInterval,
};
use oa_bt::exec::{
    FillAt, FillFlags, FillReason, InstrumentId, InstrumentSpec, OrderRequest, OrderType,
    QuantitySpec,
};
use oa_bt::market::calendar::{BaseInterval, Calendar};
use oa_bt::market::facts::MarketFacts;
use oa_bt::market::margin::MarginModel;
use oa_bt::market::slippage::SlippageSpec;
use oa_bt::model::{days_from_civil, ts_from_ist, Day, Exchange, InstrumentKind, Product, Side};
use oa_bt::rms::{ExitReason, StrategyRiskConfig};
use oa_bt::sizing::SizingControls;

mod venue_support;

use venue_support::{Act, Script};

const CONTRACT: InstrumentId = InstrumentId(0);
const OPEN_MINUTE: u32 = 9 * 60 + 15;
const CLOSE_MINUTE: u32 = 15 * 60 + 30;
/// Slices of an NFO session on the 1-minute grid (09:15 to 15:30).
const SLICES: u32 = 375;
/// The minute of the session that opens at the 15:10 square-off.
const SQUARE_OFF: u32 = 355;

fn last_day() -> Day {
    days_from_civil(2025, 7, 31)
}

/// A contract on NFO, lot 75, tick 0.05, whose last trading day is 2025-07-31, with the
/// positional exit day `exit_day`.
fn contract(symbol: &str, kind: InstrumentKind, exit_day: Option<Day>) -> InstrumentSpec {
    InstrumentSpec {
        symbol: symbol.to_string(),
        exchange: Exchange::Nfo,
        kind,
        lot_size_units: 75,
        tick_size_rupees: 0.05,
        multiplier: 1.0,
        freeze_quantity_units: None,
        cost_segment: None,
        last_trading_day: Some(last_day()),
        positional_exit_day: exit_day,
    }
}

/// A stock future, physically settled: its exit day is its last trading day.
fn stock_future() -> InstrumentSpec {
    contract("SBIN31JUL25FUT", InstrumentKind::Future, Some(last_day()))
}

/// An index future, settled in cash: no exit day.
fn index_future() -> InstrumentSpec {
    contract("NIFTY31JUL25FUT", InstrumentKind::Future, None)
}

fn index_option(exit_day: Option<Day>) -> InstrumentSpec {
    contract("NIFTY31JUL2525000CE", InstrumentKind::Option, exit_day)
}

/// 1-minute bars of 2025-07-31 flat at `price` (high and low 0.50 from it), the 15:10
/// bar opening at `square_off_open`.
fn minute_bars(price: f64, square_off_open: f64) -> BarColumns {
    let mut columns = BarColumns::default();
    for k in 0..SLICES {
        let open = if k == SQUARE_OFF {
            square_off_open
        } else {
            price
        };
        columns
            .ts_open
            .push(ts_from_ist(last_day(), OPEN_MINUTE + k));
        columns.open.push(open);
        columns.high.push(open.max(price) + 0.5);
        columns.low.push(open.min(price) - 0.5);
        columns.close.push(price);
        columns.volume.push(1_000.0);
    }
    columns
}

fn config(base: BaseInterval, strategy: StrategyInterval, first: Day) -> RunConfig {
    RunConfig {
        first_day: first,
        last_day: last_day(),
        first_slice: None,
        last_slice: None,
        present: None,
        base,
        strategy_interval: strategy,
        fill: FillTiming::NextOpen,
        quantity_mode: QuantityMode::Venue,
        capital_rupees: 100_000_000.0,
        costs: CostConfig::Off,
        slippage: SlippageSpec::Ticks(1),
        margin: MarginModel,
        market: MarketFacts::default(),
        risk: StrategyRiskConfig::default(),
        sizing: SizingControls::default(),
        intraday: IntradayOverrides::default(),
        deliver_events: false,
        perturbation: Perturbation::default(),
    }
}

fn nrml_buy() -> OrderRequest {
    OrderRequest {
        product: Product::Nrml,
        order_type: OrderType::Market,
        ..OrderRequest::market(CONTRACT, Side::Buy, QuantitySpec::Units(75.0))
    }
}

/// The backtest of `bars` on 1-minute slices of 2025-07-31, bought at the 09:15 close.
fn minute_run(instrument: InstrumentSpec, bars: BarColumns) -> RunOutput {
    let minute = 60_000_000_000;
    let config = config(
        BaseInterval::Intraday(minute),
        StrategyInterval::Intraday(minute),
        last_day(),
    );
    let decided = ts_from_ist(last_day(), OPEN_MINUTE + 1);
    run_with(config, instrument, bars, decided)
}

fn run_with(
    config: RunConfig,
    instrument: InstrumentSpec,
    bars: BarColumns,
    decided: i64,
) -> RunOutput {
    let setup = RunSetup {
        config,
        calendar: Calendar::new().expect("the embedded calendar tables parse"),
        instruments: vec![instrument],
        bars: vec![bars],
    };
    let mut kernel = Engine::new(setup).expect("a valid backtest");
    let mut script = Script::new(vec![(decided, Act::Submit(nrml_buy()))]);
    run(&mut kernel, &mut script).expect("the backtest completes");
    kernel.finish().expect("the backtest finished")
}

const EXPIRY_EXIT: FillReason = FillReason::Exit(ExitReason::ExpiryExit);

#[test]
fn a_stock_future_held_into_its_last_day_exits_at_the_square_off_open_plus_slippage() {
    // The 15:10 bar opens at 1,002.00: sold there less a tick, 1,001.95.
    let output = minute_run(stock_future(), minute_bars(1000.0, 1002.0));
    assert_eq!(output.fills.len(), 2, "{:?}", output.fills);
    let exit = &output.fills[1];
    assert_eq!(exit.stamp.slice, SQUARE_OFF);
    let fill = &exit.fill;
    assert_eq!(
        (fill.price_rupees, fill.at, fill.reason),
        (1001.95, FillAt::Open, EXPIRY_EXIT)
    );
    assert_eq!(fill.flags, FillFlags::NONE);
    assert_eq!(output.trades.len(), 1);
    assert_eq!(output.trades[0].exit_reason, EXPIRY_EXIT);
}

#[test]
fn the_exited_stock_future_is_never_settled_at_expiry() {
    let output = minute_run(stock_future(), minute_bars(1000.0, 1002.0));
    let expiry: Vec<_> = output
        .futures
        .settlements
        .iter()
        .filter(|record| record.kind == SettlementKind::Expiry)
        .collect();
    assert!(
        expiry.iter().all(|record| record.quantity_units == 0.0),
        "{expiry:?}"
    );
    assert!(output
        .final_positions
        .iter()
        .all(|p| p.quantity_units == 0.0));
}

#[test]
fn an_index_future_still_settles_at_expiry_at_its_last_close() {
    // No exit day: the position is settled in step 8 at the last close, 1,000.00,
    // reason expiry, flagged, with no slippage.
    let output = minute_run(index_future(), minute_bars(1000.0, 1002.0));
    let exit = &output.fills[1].fill;
    let expiry = FillReason::Exit(ExitReason::Expiry);
    assert_eq!(
        (exit.price_rupees, exit.at, exit.reason),
        (1000.0, FillAt::Close, expiry)
    );
    assert!(exit.flags.contains(FillFlags::FILLED_AT_LAST_CLOSE));
}

#[test]
fn an_index_option_held_to_expiry_exits_at_the_square_off() {
    // An option was left unsettled before: with no exit day it is still open at the end.
    let held = minute_run(index_option(None), minute_bars(120.0, 125.0));
    assert_eq!(held.fills.len(), 1);
    assert_eq!(held.final_positions[0].quantity_units, 75.0);
    // With its last trading day as its exit day it is sold at the 15:10 open, 125.00 less
    // a tick: 124.95.
    let exited = minute_run(index_option(Some(last_day())), minute_bars(120.0, 125.0));
    let exit = &exited.fills[1].fill;
    assert_eq!((exit.price_rupees, exit.reason), (124.95, EXPIRY_EXIT));
}

#[test]
fn with_no_bar_after_the_square_off_the_position_exits_at_the_last_close_flagged() {
    // Daily bars of 2025-07-30 and 07-31: no bar opens at or after 15:10, so the position
    // bought at the 07-31 open is sold at that session's close, 1,010.00, flagged, less a
    // tick of slippage as the engine's `eod` exit of an MIS position takes it: 1,009.95.
    let mut bars = BarColumns::default();
    let prev = days_from_civil(2025, 7, 30);
    for (day, close) in [(prev, 1000.0), (last_day(), 1010.0)] {
        bars.ts_open.push(ts_from_ist(day, OPEN_MINUTE));
        bars.open.push(1000.0);
        bars.high.push(1015.0);
        bars.low.push(995.0);
        bars.close.push(close);
        bars.volume.push(1_000.0);
    }
    let config = config(BaseInterval::Daily, StrategyInterval::Daily, prev);
    let decided = ts_from_ist(prev, CLOSE_MINUTE);
    let output = run_with(config, stock_future(), bars, decided);
    let exit = &output.fills[1].fill;
    assert_eq!(
        (exit.price_rupees, exit.at, exit.reason),
        (1009.95, FillAt::Close, EXPIRY_EXIT)
    );
    assert!(exit.flags.contains(FillFlags::FILLED_AT_LAST_CLOSE));
}

#[test]
fn an_mcx_delivery_contract_exits_at_mcx_square_off_on_its_exit_day() {
    // GOLD on MCX: Tue 2025-07-29 is its exit day (the trading day before its tender
    // period). The MCX session of that date runs 09:00 to 23:30, its square-off 20 minutes
    // before the close: 23:10, minute 850. That bar opens at 98,010.00: sold at 98,009.00
    // (one tick of 1.00).
    let exit_day = days_from_civil(2025, 7, 29);
    let gold = InstrumentSpec {
        symbol: "GOLD05AUG25FUT".to_string(),
        exchange: Exchange::Mcx,
        kind: InstrumentKind::Future,
        lot_size_units: 1,
        tick_size_rupees: 1.0,
        multiplier: 100.0,
        freeze_quantity_units: None,
        cost_segment: None,
        last_trading_day: Some(days_from_civil(2025, 8, 5)),
        positional_exit_day: Some(exit_day),
    };
    let mcx_open = 9 * 60;
    let minute = 60_000_000_000;
    let mut bars = BarColumns::default();
    for k in 0..870 {
        let open = if k == 850 { 98_010.0 } else { 98_000.0 };
        bars.ts_open.push(ts_from_ist(exit_day, mcx_open + k));
        bars.open.push(open);
        bars.high.push(open + 5.0);
        bars.low.push(98_000.0_f64.min(open) - 5.0);
        bars.close.push(98_000.0);
        bars.volume.push(10.0);
    }
    let config = RunConfig {
        first_day: exit_day,
        last_day: exit_day,
        slippage: SlippageSpec::Ticks(1),
        ..config(
            BaseInterval::Intraday(minute),
            StrategyInterval::Intraday(minute),
            exit_day,
        )
    };
    let decided = ts_from_ist(exit_day, mcx_open + 1);
    let request = OrderRequest {
        quantity: QuantitySpec::Units(1.0),
        ..nrml_buy()
    };
    let setup = RunSetup {
        config,
        calendar: Calendar::new().expect("the embedded calendar tables parse"),
        instruments: vec![gold],
        bars: vec![bars],
    };
    let mut kernel = Engine::new(setup).expect("a valid backtest");
    let mut script = Script::new(vec![(decided, Act::Submit(request))]);
    run(&mut kernel, &mut script).expect("the backtest completes");
    let output = kernel.finish().expect("the backtest finished");
    let exit = &output.fills[1];
    assert_eq!(exit.stamp.slice, 850);
    assert_eq!(
        (exit.fill.price_rupees, exit.fill.reason),
        (98_009.0, EXPIRY_EXIT)
    );
}
