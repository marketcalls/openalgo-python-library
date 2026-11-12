#![cfg(all(feature = "costs", feature = "time"))]
//! The credit for selling delivery holdings (04-execution 5, `market::sale_credit`), end
//! to end through the slice sequence: the free cash a strategy sees after CNC fills.
//!
//! One instrument at 1,000.00 on every 1-minute bar of the sessions of 2023-03-01, 02
//! and 03 (high and low 1.00 either side), capital 10,00,000, no charges and no
//! slippage: an NSE share, or an NFO future traded on CNC. On these dates a broker
//! credited 80 percent of a sale of holdings for trading on the day of the sale and the
//! rest on the next trading day. A 15-minute strategy decides at each bar close (09:30,
//! 09:45, ...) and its MARKET orders fill at the next open, at 1,000.00.

use oa_bt::engine::{
    run, BarColumns, CostConfig, Driver, Engine, FillTiming, IntradayOverrides, KernelError, Pause,
    QuantityMode, RunConfig, RunSetup, StrategyInterval,
};
use oa_bt::exec::{InstrumentId, InstrumentSpec, OrderRequest, QuantitySpec};
use oa_bt::market::calendar::{BaseInterval, Calendar};
use oa_bt::market::facts::MarketFacts;
use oa_bt::market::margin::MarginModel;
use oa_bt::market::slippage::SlippageSpec;
use oa_bt::model::{days_from_civil, ts_from_ist, Day, Exchange, InstrumentKind, Product, Side};
use oa_bt::rms::StrategyRiskConfig;
use oa_bt::sizing::SizingControls;

const MINUTE: i64 = 60_000_000_000;
const SESSION_OPEN_MINUTE: u32 = 9 * 60 + 15;
const SESSION_MINUTES: u32 = 375;
const SESSIONS: [u32; 3] = [1, 2, 3];
const CAPITAL: f64 = 1_000_000.0;
const PRICE: f64 = 1_000.0;
const HELD: InstrumentId = InstrumentId(0);

fn march(day: u32) -> Day {
    days_from_civil(2023, 3, day)
}

/// UTC nanoseconds of an IST clock time on 2023-03-`day`.
fn at(day: u32, hour: u32, minute: u32) -> i64 {
    ts_from_ist(march(day), hour * 60 + minute)
}

fn bars() -> BarColumns {
    let mut columns = BarColumns::default();
    for day in SESSIONS {
        for k in 0..SESSION_MINUTES {
            columns
                .ts_open
                .push(ts_from_ist(march(day), SESSION_OPEN_MINUTE + k));
            columns.open.push(PRICE);
            columns.high.push(PRICE + 1.0);
            columns.low.push(PRICE - 1.0);
            columns.close.push(PRICE);
            columns.volume.push(1_000.0);
        }
    }
    columns
}

/// An NSE share, or an NFO future (one unit a lot) that expires after the run.
fn instrument(kind: InstrumentKind) -> InstrumentSpec {
    let future = kind == InstrumentKind::Future;
    InstrumentSpec {
        symbol: if future { "NIFTY30MAR23FUT" } else { "SBIN" }.to_string(),
        exchange: if future { Exchange::Nfo } else { Exchange::Nse },
        kind,
        lot_size_units: 1,
        tick_size_rupees: 0.05,
        multiplier: 1.0,
        freeze_quantity_units: None,
        cost_segment: None,
        last_trading_day: future.then(|| march(30)),
    }
}

fn config() -> RunConfig {
    RunConfig {
        first_day: march(1),
        last_day: march(3),
        first_slice: None,
        last_slice: None,
        present: None,
        base: BaseInterval::Intraday(MINUTE),
        strategy_interval: StrategyInterval::Intraday(15 * MINUTE),
        fill: FillTiming::NextOpen,
        quantity_mode: QuantityMode::Venue,
        capital_rupees: CAPITAL,
        costs: CostConfig::Off,
        slippage: SlippageSpec::Zero,
        margin: MarginModel,
        market: MarketFacts::default(),
        risk: StrategyRiskConfig::default(),
        sizing: SizingControls::default(),
        intraday: IntradayOverrides::default(),
        deliver_events: false,
    }
}

/// A CNC MARKET order for `units`: a buy, or a sale that only reduces.
fn order(side: Side, units: f64) -> OrderRequest {
    OrderRequest {
        product: Product::Cnc,
        reduce_only: side == Side::Sell,
        ..OrderRequest::market(HELD, side, QuantitySpec::Units(units))
    }
}

/// Submits each order of `plan` at its decision instant and keeps the free cash seen at
/// every decision.
struct Scripted {
    plan: Vec<(i64, OrderRequest)>,
    cash_seen: Vec<(i64, f64)>,
}

impl Driver for Scripted {
    fn on_pause(&mut self, engine: &mut Engine, pause: &Pause) -> Result<(), KernelError> {
        let Pause::BarClose { decided_at, .. } = pause else {
            return Ok(());
        };
        self.cash_seen
            .push((*decided_at, engine.account().free_cash_rupees));
        for (at, request) in &self.plan {
            if at == decided_at {
                engine.submit(request.clone())?;
            }
        }
        Ok(())
    }
}

/// The free cash seen at each decision of a run on `kind` that places `plan`.
fn cash_seen(kind: InstrumentKind, plan: &[(i64, OrderRequest)]) -> Vec<(i64, f64)> {
    let setup = RunSetup {
        config: config(),
        calendar: Calendar::new().expect("the embedded calendar tables parse"),
        instruments: vec![instrument(kind)],
        bars: vec![bars()],
    };
    let mut kernel = Engine::new(setup).expect("a valid setup");
    let mut driver = Scripted {
        plan: plan.to_vec(),
        cash_seen: Vec::new(),
    };
    run(&mut kernel, &mut driver).expect("the run completes");
    driver.cash_seen
}

fn cash_at(seen: &[(i64, f64)], decided_at: i64) -> Option<f64> {
    seen.iter()
        .find(|(at, _)| *at == decided_at)
        .map(|(_, cash)| *cash)
}

/// 900 bought at the 03-01 09:30 open (9,00,000 paid, 1,00,000 left) and sold whole at
/// the 03-02 09:30 open.
fn holdings_sold_next_day() -> Vec<(i64, OrderRequest)> {
    vec![
        (at(1, 9, 30), order(Side::Buy, 900.0)),
        (at(2, 9, 30), order(Side::Sell, 900.0)),
    ]
}

#[test]
fn a_sale_of_holdings_leaves_the_part_credited_the_next_day_out_of_the_free_cash() {
    // The 9,00,000 of the sale are holdings: 20 percent, 1,80,000, wait for the next
    // trading day, so 10,00,000 - 1,80,000 = 8,20,000 are free.
    let seen = cash_seen(InstrumentKind::Equity, &holdings_sold_next_day());
    assert_eq!(cash_at(&seen, at(2, 9, 45)), Some(820_000.0));
}

#[test]
fn the_part_withheld_is_credited_at_the_session_close() {
    // Still withheld at the session's last decision (15:30, before the close step);
    // credited by the next session's first.
    let seen = cash_seen(InstrumentKind::Equity, &holdings_sold_next_day());
    assert_eq!(cash_at(&seen, at(2, 15, 30)), Some(820_000.0));
    assert_eq!(cash_at(&seen, at(3, 9, 30)), Some(1_000_000.0));
}

#[test]
fn a_sale_of_units_bought_in_the_same_session_is_credited_in_full() {
    // Bought at the 09:30 open and sold at the 09:45 open of 03-01: an intraday trade.
    let plan = [
        (at(1, 9, 30), order(Side::Buy, 900.0)),
        (at(1, 9, 45), order(Side::Sell, 900.0)),
    ];
    let seen = cash_seen(InstrumentKind::Equity, &plan);
    assert_eq!(cash_at(&seen, at(1, 10, 0)), Some(1_000_000.0));
}

#[test]
fn a_sale_matches_the_units_bought_in_the_session_first() {
    // 900 held from 03-01; on 03-02 100 more are bought at the 09:30 open (the last
    // 1,00,000 of cash) and 1,000 sold at the 09:45 open: 100 are an intraday trade and
    // 900 are holdings, of whose 9,00,000 the 1,80,000 wait.
    let plan = [
        (at(1, 9, 30), order(Side::Buy, 900.0)),
        (at(2, 9, 30), order(Side::Buy, 100.0)),
        (at(2, 9, 45), order(Side::Sell, 1_000.0)),
    ];
    let seen = cash_seen(InstrumentKind::Equity, &plan);
    assert_eq!(cash_at(&seen, at(2, 9, 45)), Some(0.0));
    assert_eq!(cash_at(&seen, at(2, 10, 0)), Some(820_000.0));
}

#[test]
fn a_future_on_cnc_blocks_its_margin_not_its_whole_value() {
    // 900 units at 1,000.00 block 20 percent of the 9,00,000 contract value (the
    // default futures margin), 1,80,000: a future is margined, never paid for in full.
    let plan = [(at(1, 9, 30), order(Side::Buy, 900.0))];
    let seen = cash_seen(InstrumentKind::Future, &plan);
    assert_eq!(cash_at(&seen, at(1, 9, 45)), Some(820_000.0));
}

#[test]
fn a_sale_of_a_future_on_cnc_withholds_nothing() {
    // The sale credit is a rule of shares: the margin of a future sold the day after
    // it was bought is free at once.
    let seen = cash_seen(InstrumentKind::Future, &holdings_sold_next_day());
    assert_eq!(cash_at(&seen, at(2, 9, 45)), Some(1_000_000.0));
}
