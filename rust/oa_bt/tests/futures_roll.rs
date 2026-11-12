#![cfg(all(feature = "costs", feature = "time"))]
//! Futures series in the kernel (04-execution 7; SPEC-19, SPEC-20, GTT-9): the roll at
//! the roll session's open, the level shift by the spread, a level crossed at the near
//! contract's open, a new lot size, settlement at expiry and the daily mark to market.
//!
//! Two NIFTY contracts on NFO with daily bars: the July contract (last trading day
//! 2025-07-31) and the August one (2025-08-28), sessions 2025-07-25 to 2025-08-01. A
//! strategy buys at the close of 2025-07-25, so the entry fills at the 07-28 open. No
//! charges, no slippage, NRML, venue quantities.

use oa_bt::engine::{
    run, BarColumns, CostConfig, Driver, Engine, Event, FillTiming, FuturesSeries,
    IntradayOverrides, KernelError, Pause, QuantityMode, RollRule, RunConfig, RunOutput, RunSetup,
    SeriesContract, SettlementKind, Step, StrategyInterval,
};
use oa_bt::exec::{
    FillAt, InstrumentId, InstrumentSpec, OrderRequest, OrderStatus, OrderType, QuantitySpec,
};
use oa_bt::market::calendar::{BaseInterval, Calendar};
use oa_bt::market::facts::{InstrumentMarket, MarketFacts};
use oa_bt::market::margin::MarginModel;
use oa_bt::market::slippage::SlippageSpec;
use oa_bt::model::{days_from_civil, ts_from_ist, Day, Exchange, InstrumentKind, Product, Side};
use oa_bt::rms::{Distance, LegRules, StrategyRiskConfig};

const NEAR: InstrumentId = InstrumentId(0);
const NEXT: InstrumentId = InstrumentId(1);
/// 09:15 and 15:30 IST in minutes since midnight.
const OPEN_MINUTE: u32 = 9 * 60 + 15;
const CLOSE_MINUTE: u32 = 15 * 60 + 30;
const SESSIONS: [u32; 6] = [25, 28, 29, 30, 31, 32];

/// One bar: (open, high, low, close), rupees.
type Ohlc = (f64, f64, f64, f64);

/// A July 2025 date; day 32 is 2025-08-01.
fn july(day: u32) -> Day {
    days_from_civil(2025, 7, 1) + day as i32 - 1
}

fn future(symbol: &str, lot: u32, last_day: Day) -> InstrumentSpec {
    InstrumentSpec {
        symbol: symbol.to_string(),
        exchange: Exchange::Nfo,
        kind: InstrumentKind::Future,
        lot_size_units: lot,
        tick_size_rupees: 0.05,
        multiplier: 1.0,
        freeze_quantity_units: None,
        cost_segment: None,
        last_trading_day: Some(last_day),
    }
}

fn bars(rows: &[(u32, Ohlc)]) -> BarColumns {
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

fn flat(price: f64) -> Ohlc {
    (price, price + 10.0, price - 10.0, price)
}

/// The near contract flat at 25,000.00 and the next at 25,100.00 every session, unless
/// a row is given.
fn market(near: &[(u32, Ohlc)], next: &[(u32, Ohlc)]) -> Vec<BarColumns> {
    let pick = |rows: &[(u32, Ohlc)], day: u32, base: f64| {
        rows.iter()
            .find(|(d, _)| *d == day)
            .map_or(flat(base), |(_, bar)| *bar)
    };
    let near_days: Vec<(u32, Ohlc)> = SESSIONS[..5]
        .iter()
        .map(|&d| (d, pick(near, d, 25_000.0)))
        .collect();
    let next_days: Vec<(u32, Ohlc)> = SESSIONS
        .iter()
        .map(|&d| (d, pick(next, d, 25_100.0)))
        .collect();
    vec![bars(&near_days), bars(&next_days)]
}

fn config() -> RunConfig {
    let nifty = InstrumentMarket {
        underlying: Some("NIFTY".to_string()),
        ..InstrumentMarket::default()
    };
    RunConfig {
        first_day: july(25),
        last_day: july(32),
        first_slice: None,
        last_slice: None,
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
        intraday: IntradayOverrides::default(),
        deliver_events: false,
    }
}

fn engine(next_lot: u32, bars: Vec<BarColumns>) -> Engine {
    let instruments = vec![
        future("NIFTY31JUL25FUT", 75, july(31)),
        future("NIFTY28AUG25FUT", next_lot, days_from_civil(2025, 8, 28)),
    ];
    let calendar = Calendar::new().expect("the embedded calendar tables parse");
    Engine::new(RunSetup {
        config: config(),
        calendar,
        instruments,
        bars,
    })
    .expect("a valid setup")
}

fn series(rule: RollRule, hand_over: Day) -> FuturesSeries {
    FuturesSeries {
        contracts: vec![
            SeriesContract {
                instrument: NEAR,
                hand_over: Some(hand_over),
            },
            SeriesContract {
                instrument: NEXT,
                hand_over: None,
            },
        ],
        rule,
    }
}

/// Buys `units` of the near contract, NRML, at the close of 2025-07-25.
struct Buyer {
    units: f64,
    rules: LegRules,
}

impl Driver for Buyer {
    fn on_pause(&mut self, engine: &mut Engine, pause: &Pause) -> Result<(), KernelError> {
        let Pause::BarClose { decided_at, .. } = pause else {
            return Ok(());
        };
        if *decided_at == ts_from_ist(july(25), CLOSE_MINUTE) {
            let request = OrderRequest {
                product: Product::Nrml,
                leg_rules: self.rules.clone(),
                ..OrderRequest::market(NEAR, Side::Buy, QuantitySpec::Units(self.units))
            };
            engine.submit(request)?;
        }
        Ok(())
    }
}

/// Places a buy LIMIT on the near contract far below the market at the close of
/// 2025-07-29: valid for the roll session, it is still working at the roll.
struct RestingBuyer;

impl Driver for RestingBuyer {
    fn on_pause(&mut self, engine: &mut Engine, pause: &Pause) -> Result<(), KernelError> {
        let Pause::BarClose { decided_at, .. } = pause else {
            return Ok(());
        };
        if *decided_at == ts_from_ist(july(29), CLOSE_MINUTE) {
            let request = OrderRequest {
                product: Product::Nrml,
                order_type: OrderType::Limit {
                    limit_rupees: 24_000.0,
                },
                ..OrderRequest::market(NEAR, Side::Buy, QuantitySpec::Units(75.0))
            };
            engine.submit(request)?;
        }
        Ok(())
    }
}

fn stop(points: f64) -> LegRules {
    LegRules {
        stop: Some(Distance::Points(points)),
        ..LegRules::default()
    }
}

fn run_roll(mut kernel: Engine, units: f64, rules: LegRules) -> RunOutput {
    let mut driver = Buyer { units, rules };
    run(&mut kernel, &mut driver).expect("the run completes");
    kernel.finish().expect("the run finished")
}

/// (day of July, step, instrument, side, units, price, where, reason).
type FillRow = (u32, Step, u32, Side, f64, f64, FillAt, &'static str);

fn fills(output: &RunOutput) -> Vec<FillRow> {
    output
        .fills
        .iter()
        .map(|record| {
            let f = record.fill;
            let day =
                (record.stamp.slice_ts - ts_from_ist(july(0), OPEN_MINUTE)) / 86_400_000_000_000;
            let at = (day as u32, record.stamp.step, f.instrument.0, f.side);
            (
                at.0,
                at.1,
                at.2,
                at.3,
                f.quantity_units,
                f.price_rupees,
                f.at,
                f.reason.code(),
            )
        })
        .collect()
}

fn names_at(output: &RunOutput, day: u32) -> Vec<String> {
    let open = ts_from_ist(july(day), OPEN_MINUTE);
    output
        .events
        .iter()
        .filter(|event| event.stamp().slice_ts == open)
        .map(|event| match event {
            Event::Position { change, .. } => change.code().to_string(),
            Event::Gtt { event, .. } => event.name.code().to_string(),
            Event::Risk { event, .. } => event.name.code().to_string(),
            Event::Order { kind, .. } => kind.code().to_string(),
            Event::TradingState { .. } => "state".to_string(),
        })
        .collect()
}

fn rolled(rules: LegRules, near: &[(u32, Ohlc)], next: &[(u32, Ohlc)]) -> RunOutput {
    let mut kernel = engine(75, market(near, next));
    let added = kernel.add_futures_series(series(RollRule::Roll, july(30)));
    assert_eq!(added, Ok(0));
    run_roll(kernel, 75.0, rules)
}

// ---------------------------------------------------------------- the roll

#[test]
fn the_switch_is_the_roll_sessions_first_open_with_the_spread_of_the_raw_opens() {
    let mut kernel = engine(75, market(&[(30, flat(25_040.0))], &[(30, flat(25_135.0))]));
    assert_eq!(
        kernel.add_futures_series(series(RollRule::Roll, july(30))),
        Ok(0)
    );
    let switch = kernel.futures_switches()[0];
    assert_eq!(switch.slice_ts, ts_from_ist(july(30), OPEN_MINUTE));
    // 25,135.00 - 25,040.00.
    assert_eq!(switch.spread_rupees, 95.0);
    assert!(switch.rolls_positions);
}

#[test]
fn a_roll_closes_the_near_contract_and_opens_the_next_at_their_opens_in_step_2() {
    let output = rolled(
        stop(150.0),
        &[(30, flat(25_040.0))],
        &[(30, flat(25_135.0))],
    );
    let rows = fills(&output);
    assert_eq!(
        rows[1..],
        [
            (
                30,
                Step::OpenFills,
                0,
                Side::Sell,
                75.0,
                25_040.0,
                FillAt::Open,
                "roll"
            ),
            (
                30,
                Step::OpenFills,
                1,
                Side::Buy,
                75.0,
                25_135.0,
                FillAt::Open,
                "roll"
            ),
        ]
    );
    assert!(output.fills[1..].iter().all(|r| r.fill.order.is_none()));
    // Cancel the near contract's GTT, roll, then a new GTT on the next contract.
    let names = names_at(&output, 30);
    let expected = [
        "GttCanceled",
        "PositionClosed",
        "PositionOpened",
        "GttPlaced",
    ];
    assert_eq!(names, expected);
}

#[test]
fn the_rolled_stop_and_the_entry_reference_move_by_the_spread() {
    // Entry 25,000.00 on 07-28, stop 150 points: 24,850.00. Spread 95.00: 24,945.00.
    let output = rolled(
        stop(150.0),
        &[(30, flat(25_040.0))],
        &[(30, flat(25_135.0))],
    );
    let row = output
        .levels
        .iter()
        .find(|row| row.instrument == NEXT)
        .expect("a level row for the next contract");
    assert_eq!(row.stop.map(|s| s.level_rupees), Some(24_945.0));
    assert_eq!(row.average_price_rupees, Some(25_135.0));
    let gtt = output.events.iter().rev().find_map(|event| match event {
        Event::Gtt { event, .. } => event.stop_trigger_rupees,
        _ => None,
    });
    assert_eq!(gtt, Some(24_945.0));
}

#[test]
fn a_stop_crossed_at_the_near_open_exits_there_and_nothing_rolls() {
    // The stop 24,850.00 is crossed at the near contract's roll-session open 24,800.00:
    // its GTT fires in step 2, the child (limit 24,850 x 0.99 = 24,601.50) fills at the
    // open, reason leg_sl, and the position is not rolled.
    let gap = (24_800.0, 24_810.0, 24_700.0, 24_750.0);
    let output = rolled(stop(150.0), &[(30, gap)], &[]);
    let rows = fills(&output);
    assert_eq!(rows.len(), 2);
    assert_eq!(
        rows[1],
        (
            30,
            Step::OpenFills,
            0,
            Side::Sell,
            75.0,
            24_800.0,
            FillAt::Open,
            "leg_sl"
        )
    );
    let names = names_at(&output, 30);
    let expected = [
        "GttTriggered",
        "leg_sl_hit",
        "GttChildFilled",
        "PositionClosed",
    ];
    assert_eq!(names, expected);
}

#[test]
fn a_new_lot_size_rolls_the_nearest_whole_number_of_lots() {
    // 150 units (two lots of 75) roll into two lots of 65: 130 units.
    let mut kernel = engine(65, market(&[], &[]));
    assert_eq!(
        kernel.add_futures_series(series(RollRule::Roll, july(30))),
        Ok(0)
    );
    let output = run_roll(kernel, 150.0, stop(150.0));
    let rows = fills(&output);
    assert_eq!((rows[2].2, rows[2].4, rows[2].7), (1, 130.0, "roll"));
    let row = output.levels.iter().find(|row| row.instrument == NEXT);
    assert_eq!(
        row.and_then(|r| r.stop.map(|s| s.quantity_units)),
        Some(130.0)
    );
}

#[test]
fn a_missing_near_bar_moves_the_roll_to_the_next_session_with_both_bars() {
    let mut bars = market(&[], &[]);
    // The near contract has no bar on the roll session, 07-30.
    let near: Vec<(u32, Ohlc)> = [25, 28, 29, 31]
        .iter()
        .map(|&day| (day, flat(25_000.0)))
        .collect();
    bars[0] = self::bars(&near);
    let mut kernel = engine(75, bars);
    assert_eq!(
        kernel.add_futures_series(series(RollRule::Roll, july(30))),
        Ok(0)
    );
    assert_eq!(
        kernel.futures_switches()[0].slice_ts,
        ts_from_ist(july(31), OPEN_MINUTE)
    );
}

#[test]
fn the_near_contracts_working_orders_are_cancelled_at_the_roll() {
    let mut kernel = engine(75, market(&[], &[]));
    assert_eq!(
        kernel.add_futures_series(series(RollRule::Roll, july(30))),
        Ok(0)
    );
    run(&mut kernel, &mut RestingBuyer).expect("the run completes");
    let output = kernel.finish().expect("the run finished");
    assert_eq!(output.orders[0].status, OrderStatus::Canceled);
    assert_eq!(names_at(&output, 30), ["Canceled"]);
    assert!(output.fills.is_empty());
}

#[test]
fn a_roll_after_the_last_trading_day_is_refused() {
    let mut kernel = engine(75, market(&[], &[]));
    let late = kernel.add_futures_series(series(RollRule::Roll, july(32)));
    assert!(matches!(
        late,
        Err(KernelError::InvalidConfig { field: "roll", .. })
    ));
}

// ---------------------------------------------------------------- held to expiry

fn held_to_expiry() -> RunOutput {
    let mut kernel = engine(75, market(&[(31, flat(25_200.0))], &[]));
    let added = kernel.add_futures_series(series(RollRule::SettleAtExpiry, july(32)));
    assert_eq!(added, Ok(0));
    run_roll(kernel, 75.0, stop(500.0))
}

#[test]
fn a_position_held_to_expiry_is_settled_at_the_last_close_on_the_expiry_session() {
    let output = held_to_expiry();
    let rows = fills(&output);
    assert_eq!(
        rows[1],
        (
            31,
            Step::SessionClose,
            0,
            Side::Sell,
            75.0,
            25_200.0,
            FillAt::Close,
            "expiry"
        )
    );
    assert_eq!(
        output.fills[1].fill.flags.names(),
        vec!["filled_at_last_close"]
    );
    assert_eq!(output.fills[1].fill.costs_rupees, 0.0);
    // Settled, the GTT is cancelled first.
    assert_eq!(names_at(&output, 31), ["GttCanceled", "PositionClosed"]);
}

#[test]
fn the_daily_mark_to_market_posts_each_sessions_change_and_the_expiry() {
    // Entry 25,000.00 on 07-28; closes 25,000.00 on 07-28, 29 and 30, settled at
    // 25,200.00 on 07-31: postings 0, 0, 0, then (25,200 - 25,000) x 75 = 15,000.
    let output = held_to_expiry();
    let postings: Vec<(Day, f64, SettlementKind)> = output
        .futures
        .settlements
        .iter()
        .map(|r| (r.day, r.amount_rupees, r.kind))
        .collect();
    assert_eq!(
        postings,
        vec![
            (july(28), 0.0, SettlementKind::Daily),
            (july(29), 0.0, SettlementKind::Daily),
            (july(30), 0.0, SettlementKind::Daily),
            (july(31), 15_000.0, SettlementKind::Expiry),
        ]
    );
    let total: f64 = postings.iter().map(|p| p.1).sum();
    assert_eq!(total, output.gross_pnl.total_rupees);
}

#[test]
fn the_datas_settlement_price_is_used_instead_of_the_last_close() {
    let mut kernel = engine(75, market(&[], &[]));
    assert_eq!(
        kernel.add_futures_series(series(RollRule::SettleAtExpiry, july(32))),
        Ok(0)
    );
    let prices = vec![(july(28), 25_010.0), (july(31), 25_100.0)];
    assert_eq!(kernel.set_settlement_prices(NEAR, prices), Ok(()));
    let output = run_roll(kernel, 75.0, stop(500.0));
    let first = output.futures.settlements[0];
    // (25,010 - 25,000) x 75 = 750, not flagged.
    assert_eq!((first.amount_rupees, first.last_close_used), (750.0, false));
    let rows = fills(&output);
    assert_eq!((rows[1].5, rows[1].7), (25_100.0, "expiry"));
    assert!(output.fills[1].fill.flags.names().is_empty());
}

#[test]
fn a_settlement_switch_takes_its_spread_from_the_last_trading_days_closes() {
    let near = [(31, (25_000.0, 25_050.0, 24_950.0, 25_020.0))];
    let next = [(31, (25_100.0, 25_150.0, 25_050.0, 25_110.0))];
    let mut kernel = engine(75, market(&near, &next));
    let added = kernel.add_futures_series(series(RollRule::SettleAtExpiry, july(32)));
    assert_eq!(added, Ok(0));
    let switch = kernel.futures_switches()[0];
    // 25,110.00 - 25,020.00 at the close of 07-31; the next contract from 08-01.
    assert_eq!(switch.spread_rupees, 90.0);
    assert_eq!(switch.slice_ts, ts_from_ist(july(32), OPEN_MINUTE));
    assert!(!switch.rolls_positions);
}
