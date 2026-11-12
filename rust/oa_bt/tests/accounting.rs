#![cfg(all(feature = "costs", feature = "time"))]
//! Charges, default slippage, margin and the engine default size, end to end through
//! the slice sequence.
//!
//! One NSE equity on 2025-07-01, 1-minute bars over the whole session. Bars opening
//! before 09:45 open at 1,000.00 and later ones at 1,100.00 (high and low 1.00 either
//! side, close at the open). A 15-minute strategy buys at the 09:15 bar's close (fill at
//! the 09:30 open, 1,000.00) and exits at the 09:30 bar's close (fill at the 09:45 open,
//! 1,100.00).

use oa_bt::engine::{
    run, BarColumns, CostConfig, Driver, Engine, FillTiming, IntradayOverrides, KernelError, Pause,
    QuantityMode, RunConfig, RunOutput, RunSetup, StrategyInterval, Submission,
};
use oa_bt::exec::{InstrumentId, InstrumentSpec, OrderRequest, QuantitySpec, RejectCode};
use oa_bt::market::calendar::{BaseInterval, Calendar};
use oa_bt::market::costs::{ContractNoteAggregator, CostModel, Segment};
use oa_bt::market::facts::MarketFacts;
use oa_bt::market::margin::MarginModel;
use oa_bt::market::slippage::SlippageSpec;
use oa_bt::model::{days_from_civil, ts_from_ist, Day, Exchange, InstrumentKind, Product, Side};
use oa_bt::rms::StrategyRiskConfig;
use oa_bt::sizing::{SizerSpec, SizingControls};

const MINUTE: i64 = 60_000_000_000;
const SESSION_MINUTES: u32 = 375;
const CAPITAL: f64 = 1_000_000.0;
const SBIN: InstrumentId = InstrumentId(0);

fn day() -> Day {
    days_from_civil(2025, 7, 1)
}

fn at(hour: u32, minute: u32) -> i64 {
    ts_from_ist(day(), hour * 60 + minute)
}

fn bars() -> BarColumns {
    let mut columns = BarColumns::default();
    for k in 0..SESSION_MINUTES {
        let open = if k < 30 { 1_000.0 } else { 1_100.0 };
        columns.ts_open.push(ts_from_ist(day(), 9 * 60 + 15 + k));
        columns.open.push(open);
        columns.high.push(open + 1.0);
        columns.low.push(open - 1.0);
        columns.close.push(open);
        columns.volume.push(1_000.0);
    }
    columns
}

fn sbin() -> InstrumentSpec {
    InstrumentSpec {
        symbol: "SBIN".to_string(),
        exchange: Exchange::Nse,
        kind: InstrumentKind::Equity,
        lot_size_units: 1,
        tick_size_rupees: 0.05,
        multiplier: 1.0,
        freeze_quantity_units: None,
        cost_segment: None,
        last_trading_day: None,
    }
}

fn model() -> CostModel {
    CostModel::builtin().expect("the built-in schedules parse")
}

fn config(costs: CostConfig, slippage: SlippageSpec, mode: QuantityMode) -> RunConfig {
    RunConfig {
        first_day: day(),
        last_day: day(),
        first_slice: None,
        last_slice: None,
        present: None,
        base: BaseInterval::Intraday(MINUTE),
        strategy_interval: StrategyInterval::Intraday(15 * MINUTE),
        fill: FillTiming::NextOpen,
        quantity_mode: mode,
        capital_rupees: CAPITAL,
        costs,
        slippage,
        margin: MarginModel,
        market: MarketFacts::default(),
        risk: StrategyRiskConfig::default(),
        sizing: SizingControls::default(),
        intraday: IntradayOverrides::default(),
        deliver_events: false,
    }
}

/// Places `entry` at the 09:15 bar's close and exits the whole position at the 09:30
/// bar's close.
struct RoundTrip {
    entry: OrderRequest,
    submissions: Vec<Submission>,
}

impl Driver for RoundTrip {
    fn on_pause(&mut self, engine: &mut Engine, pause: &Pause) -> Result<(), KernelError> {
        let Pause::BarClose { bars, .. } = pause else {
            return Ok(());
        };
        for bar in bars {
            let product = self.entry.product;
            let request = if bar.ts_open == at(9, 15) {
                self.entry.clone()
            } else if bar.ts_open == at(9, 30) {
                let held = engine
                    .position(SBIN, product)
                    .map_or(0.0, |p| p.quantity_units);
                let side = if held > 0.0 { Side::Sell } else { Side::Buy };
                OrderRequest {
                    reduce_only: true,
                    product,
                    ..OrderRequest::market(SBIN, side, QuantitySpec::PositionFraction(1.0))
                }
            } else {
                continue;
            };
            self.submissions.push(engine.submit(request)?);
        }
        Ok(())
    }
}

fn run_trip(config: RunConfig, entry: OrderRequest) -> (RunOutput, Vec<Submission>) {
    let setup = RunSetup {
        config,
        calendar: Calendar::new().expect("the embedded calendar tables parse"),
        instruments: vec![sbin()],
        bars: vec![bars()],
    };
    let mut kernel = Engine::new(setup).expect("a valid setup");
    let mut driver = RoundTrip {
        entry,
        submissions: Vec::new(),
    };
    run(&mut kernel, &mut driver).expect("the run completes");
    (
        kernel.finish().expect("the run finished"),
        driver.submissions,
    )
}

fn buy(units: f64, product: Product) -> OrderRequest {
    OrderRequest {
        product,
        ..OrderRequest::market(SBIN, Side::Buy, QuantitySpec::Units(units))
    }
}

fn charged(units: f64) -> RunOutput {
    let costs = CostConfig::Model(model());
    let config = config(costs, SlippageSpec::Zero, QuantityMode::Venue);
    run_trip(config, buy(units, Product::Mis)).0
}

#[test]
fn each_fill_costs_what_the_cost_model_prices_it_at() {
    let output = charged(400.0);
    let model = model();
    let price = |side, price| {
        let cost = model.fill_cost(Segment::NseEqIntraday, side, price, 400.0, day());
        cost.map(|c| c.total()).expect("priced")
    };
    let costs: Vec<f64> = output.fills.iter().map(|r| r.fill.costs_rupees).collect();
    assert_eq!(
        costs,
        vec![price(Side::Buy, 1_000.0), price(Side::Sell, 1_100.0)]
    );
    let trade_costs = output.trades[0].costs_rupees;
    assert!((trade_costs - costs.iter().sum::<f64>()).abs() < 1e-9);
}

#[test]
fn the_ledger_charges_what_the_days_contract_note_charges() {
    let output = charged(400.0);
    let model = model();
    let mut notes = ContractNoteAggregator::new();
    for (side, price) in [(Side::Buy, 1_000.0), (Side::Sell, 1_100.0)] {
        let cost = model.fill_cost(Segment::NseEqIntraday, side, price, 400.0, day());
        notes.add(day(), Segment::NseEqIntraday, &cost.expect("priced"));
    }
    let expected = notes.total_charged();
    let last_costs = output.equity.costs_rupees.last().copied();
    assert_eq!(last_costs, Some(expected));
    let settled = &output.charges.notes;
    assert_eq!(settled.len(), 1);
    assert!(settled[0].settled);
    assert_eq!(settled[0].note.charged_total, expected);
}

#[test]
fn the_default_slippage_of_cash_equity_is_two_bps_rounded_away_from_the_trader() {
    // Buy at 1,000.00 + 2 bps (0.20) = 1,000.20; sell at 1,100.00 - 0.22 = 1,099.78,
    // rounded down to the 0.05 tick: 1,099.75.
    let config = config(
        CostConfig::Off,
        SlippageSpec::SegmentDefault,
        QuantityMode::Venue,
    );
    let (output, _) = run_trip(config, buy(400.0, Product::Mis));
    let prices: Vec<f64> = output.fills.iter().map(|r| r.fill.price_rupees).collect();
    assert_eq!(prices, vec![1_000.2, 1_099.75]);
    let rows: Vec<&str> = output
        .charges
        .tables
        .iter()
        .map(|u| u.row.id.as_str())
        .collect();
    assert_eq!(rows, vec!["MIS_EQ/2", "EQ_LARGE_MID/1"]);
}

#[test]
fn an_entry_beyond_the_free_cash_is_rejected_for_margin_at_the_fill() {
    // 1,001 shares at 1,000.00 on delivery need 10,01,000 of the 10,00,000 capital.
    let config = config(CostConfig::Off, SlippageSpec::Zero, QuantityMode::Venue);
    let (output, _) = run_trip(config, buy(1_001.0, Product::Cnc));
    assert!(output.fills.is_empty());
    assert_eq!(output.orders[0].reject_code, Some(RejectCode::Margin));
}

#[test]
fn intraday_leverage_asked_for_blocks_value_over_leverage() {
    // 4,000 shares at 1,000.00 are worth 40,00,000; at 5x they block 8,00,000.
    let config = config(CostConfig::Off, SlippageSpec::Zero, QuantityMode::Venue);
    let leveraged = OrderRequest {
        leverage: Some(5.0),
        ..buy(4_000.0, Product::Mis)
    };
    let (output, _) = run_trip(config, leveraged);
    assert_eq!(output.fills.len(), 2);
    let entry_margin = output.equity.margin_rupees[15];
    assert_eq!(entry_margin, 800_000.0);
}

#[test]
fn the_engine_default_spends_the_capital_at_the_fill_in_fractional_mode() {
    // No costs, no slippage: 10,00,000 / 1,000.00 = 1,000 shares at the 09:30 open.
    let config = config(
        CostConfig::Off,
        SlippageSpec::Zero,
        QuantityMode::Fractional,
    );
    let entry = OrderRequest {
        product: Product::Mis,
        ..OrderRequest::market(
            SBIN,
            Side::Buy,
            QuantitySpec::Sizer(SizerSpec::EngineDefault),
        )
    };
    let (output, _) = run_trip(config, entry);
    assert_eq!(output.fills[0].fill.quantity_units, 1_000.0);
}

#[test]
fn the_engine_default_keeps_the_cash_buffer_in_venue_mode() {
    // At the decision close 1,000.00: 10,00,000 x 0.98 / 1,000.00 = 980 shares.
    let config = config(CostConfig::Off, SlippageSpec::Zero, QuantityMode::Venue);
    let entry = OrderRequest {
        product: Product::Mis,
        ..OrderRequest::market(
            SBIN,
            Side::Buy,
            QuantitySpec::Sizer(SizerSpec::EngineDefault),
        )
    };
    let (output, _) = run_trip(config, entry);
    assert_eq!(output.fills[0].fill.quantity_units, 980.0);
}
