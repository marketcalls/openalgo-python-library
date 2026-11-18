#![cfg(all(feature = "costs", feature = "time"))]
//! The MAE and MFE of futures trades (`engine::excursion`): the contract multiplier in
//! the running P&L, and a roll, which closes the near contract's trade at its open and
//! opens the next contract's trade there. No charges, no slippage, NRML.

mod order_support;

use oa_bt::engine::{
    run, BarColumns, CostConfig, Driver, Engine, FillTiming, FuturesSeries, IntradayOverrides,
    KernelError, Pause, Perturbation, QuantityMode, RollRule, RunConfig, RunOutput, RunSetup,
    SeriesContract, StrategyInterval, TradeRecord,
};
use oa_bt::exec::{InstrumentId, InstrumentSpec, OrderRequest, OrderType, QuantitySpec};
use oa_bt::market::calendar::{BaseInterval, Calendar};
use oa_bt::market::facts::{InstrumentMarket, MarketFacts};
use oa_bt::market::margin::MarginModel;
use oa_bt::market::slippage::SlippageSpec;
use oa_bt::model::{days_from_civil, ts_from_ist, Day, Exchange, InstrumentKind, Product, Side};
use oa_bt::rms::StrategyRiskConfig;
use oa_bt::sizing::SizingControls;
use order_support::{
    engine, exit_all, minute, minute_bars, minute_config, order, run_script, Act, Ohlc,
};

/// (MAE, MFE, gross P&L) of a trade, rupees.
fn excursion(trade: &TradeRecord) -> (f64, f64, f64) {
    (trade.mae_rupees, trade.mfe_rupees, trade.gross_pnl_rupees)
}

#[test]
fn the_running_pnl_of_a_future_counts_its_multiplier() {
    // One unit of a gold future quoted per 10 grams, traded in 1 kg units: a one-rupee
    // move is Rs 100 a unit. Bought at the open of minute 2 (97,000): its bar (97,000,
    // 97,050, 96,900, 97,020) is +5,000 at the high and -10,000 at the low; sold at the
    // open of minute 3 (97,030): +3,000.
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
    };
    let bars = [
        (2, (97_000.0, 97_050.0, 96_900.0, 97_020.0)),
        (3, (97_030.0, 97_200.0, 96_000.0, 97_100.0)),
    ];
    let kernel = engine(minute_config(), gold, minute_bars(10, 97_000.0, &bars));
    let acts = vec![
        (
            minute(2),
            Act::Submit(order(Side::Buy, 1.0, OrderType::Market, Product::Nrml)),
        ),
        (minute(3), Act::Submit(exit_all(Side::Buy, Product::Nrml))),
    ];
    let (output, _) = run_script(kernel, acts);
    assert_eq!(output.trades.len(), 1);
    assert_eq!(excursion(&output.trades[0]), (-10_000.0, 5_000.0, 3_000.0));
}

// ---------------------------------------------------------------- a roll

const NEAR: InstrumentId = InstrumentId(0);
const NEXT: InstrumentId = InstrumentId(1);
const OPEN_MINUTE: u32 = 9 * 60 + 15;
const CLOSE_MINUTE: u32 = 15 * 60 + 30;

/// A July 2025 date; day 32 is 2025-08-01.
fn july(day: u32) -> Day {
    days_from_civil(2025, 7, 1) + day as i32 - 1
}

fn future(symbol: &str, last_day: Day) -> InstrumentSpec {
    InstrumentSpec {
        symbol: symbol.to_string(),
        exchange: Exchange::Nfo,
        kind: InstrumentKind::Future,
        lot_size_units: 75,
        tick_size_rupees: 0.05,
        multiplier: 1.0,
        freeze_quantity_units: None,
        cost_segment: None,
        last_trading_day: Some(last_day),
    }
}

/// Daily bars (open, high, low, close) of the given July days.
fn daily(rows: &[(u32, Ohlc)]) -> BarColumns {
    let mut columns = BarColumns::default();
    for &(day, (open, high, low, close)) in rows {
        columns.ts_open.push(ts_from_ist(july(day), OPEN_MINUTE));
        columns.open.push(open);
        columns.high.push(high);
        columns.low.push(low);
        columns.close.push(close);
        columns.volume.push(1_000.0);
    }
    columns
}

fn roll_config() -> RunConfig {
    let nifty = InstrumentMarket {
        underlying: Some("NIFTY".to_string()),
        ..InstrumentMarket::default()
    };
    RunConfig {
        first_day: july(25),
        last_day: july(32),
        first_slice: None,
        last_slice: None,
        present: None,
        base: BaseInterval::Daily,
        strategy_interval: StrategyInterval::Daily,
        fill: FillTiming::NextOpen,
        quantity_mode: QuantityMode::Venue,
        capital_rupees: 10_000_000.0,
        costs: CostConfig::Off,
        slippage: SlippageSpec::Zero,
        margin: MarginModel,
        market: MarketFacts {
            instruments: vec![nifty.clone(), nifty],
        },
        risk: StrategyRiskConfig::default(),
        sizing: SizingControls::default(),
        intraday: IntradayOverrides::default(),
        deliver_events: false,
        perturbation: Perturbation::default(),
    }
}

/// Buys 75 of the near contract at the close of 07-25 and sells the next contract's
/// position at the close of 07-31.
struct BuyThenSell;

impl Driver for BuyThenSell {
    fn on_pause(&mut self, engine: &mut Engine, pause: &Pause) -> Result<(), KernelError> {
        let Pause::BarClose { decided_at, .. } = pause else {
            return Ok(());
        };
        let (instrument, side) = if *decided_at == ts_from_ist(july(25), CLOSE_MINUTE) {
            (NEAR, Side::Buy)
        } else if *decided_at == ts_from_ist(july(31), CLOSE_MINUTE) {
            (NEXT, Side::Sell)
        } else {
            return Ok(());
        };
        let request = OrderRequest {
            product: Product::Nrml,
            ..OrderRequest::market(instrument, side, QuantitySpec::Units(75.0))
        };
        engine.submit(request)?;
        Ok(())
    }
}

fn rolled() -> RunOutput {
    // The near contract: 25,000.00 open and close, 10 points either way, every session
    // to the roll at the 07-30 open (25,040.00). The next contract opens 07-30 at
    // 25,135.00 and trades at 25,100.00 on 07-31 and 08-01.
    let near = daily(&[
        (25, (25_000.0, 25_010.0, 24_990.0, 25_000.0)),
        (28, (25_000.0, 25_010.0, 24_990.0, 25_000.0)),
        (29, (25_000.0, 25_010.0, 24_990.0, 25_000.0)),
        (30, (25_040.0, 25_300.0, 24_700.0, 25_040.0)),
        (31, (25_040.0, 25_050.0, 25_030.0, 25_040.0)),
    ]);
    let next = daily(&[
        (25, (25_100.0, 25_110.0, 25_090.0, 25_100.0)),
        (28, (25_100.0, 25_110.0, 25_090.0, 25_100.0)),
        (29, (25_100.0, 25_110.0, 25_090.0, 25_100.0)),
        (30, (25_135.0, 25_145.0, 25_125.0, 25_135.0)),
        (31, (25_100.0, 25_110.0, 25_090.0, 25_100.0)),
        (32, (25_100.0, 25_110.0, 25_090.0, 25_100.0)),
    ]);
    let instruments = vec![
        future("NIFTY31JUL25FUT", july(31)),
        future("NIFTY28AUG25FUT", days_from_civil(2025, 8, 28)),
    ];
    let calendar = Calendar::new().expect("the embedded calendar tables parse");
    let mut kernel = Engine::new(RunSetup {
        config: roll_config(),
        calendar,
        instruments,
        bars: vec![near, next],
    })
    .expect("a valid setup");
    let series = FuturesSeries {
        contracts: vec![
            SeriesContract {
                instrument: NEAR,
                hand_over: Some(july(30)),
            },
            SeriesContract {
                instrument: NEXT,
                hand_over: None,
            },
        ],
        rule: RollRule::Roll,
    };
    assert_eq!(kernel.add_futures_series(series), Ok(0));
    run(&mut kernel, &mut BuyThenSell).expect("the run completes");
    kernel.finish().expect("the run finished")
}

#[test]
fn a_roll_closes_the_near_trade_at_its_open_and_opens_the_next_trade_there() {
    // Near: 75 bought at the 07-28 open (25,000.00); 07-28 and 07-29 count whole (10
    // points either way: -750 and +750); the roll at the 07-30 open (25,040.00) counts
    // that open only (+3,000), not the 07-30 range of the near contract.
    // Next: 75 bought at the 07-30 open (25,135.00), whose whole bar counts (+750 at
    // 25,145.00, -750 at 25,125.00); 07-31 counts whole (25,090.00: -3,375); sold at
    // the 08-01 open (25,100.00): -2,625.
    let output = rolled();
    let trades: Vec<(u32, &str, (f64, f64, f64))> = output
        .trades
        .iter()
        .map(|trade| {
            (
                trade.instrument.0,
                trade.exit_reason.code(),
                excursion(trade),
            )
        })
        .collect();
    assert_eq!(
        trades,
        [
            (0, "roll", (-750.0, 3_000.0, 3_000.0)),
            (1, "signal", (-3_375.0, 750.0, -2_625.0)),
        ]
    );
}
