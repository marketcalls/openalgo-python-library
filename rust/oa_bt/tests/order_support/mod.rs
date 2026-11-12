//! Shared harness of the resting-order and GTT tests: instruments, bars written per
//! bar, run configurations, a scripted driver that submits, modifies and cancels at
//! given decision instants, and readable views of the fills, events and GTTs.
//!
//! Every test file that includes this module uses a different part of it, so unused
//! items are allowed here rather than duplicated per file.
#![allow(dead_code)]

use oa_bt::engine::{
    run, BarColumns, CostConfig, Driver, Engine, Event, FillTiming, IntradayOverrides, KernelError,
    OrderEventKind, Pause, QuantityMode, RunConfig, RunOutput, RunSetup, Step, StrategyInterval,
    Submission,
};
use oa_bt::exec::{
    FillAt, Gtt, InstrumentId, InstrumentSpec, OrderId, OrderRequest, OrderType, QuantitySpec,
    Validity,
};
use oa_bt::market::calendar::{BaseInterval, Calendar};
use oa_bt::market::facts::MarketFacts;
use oa_bt::market::margin::MarginModel;
use oa_bt::market::slippage::SlippageSpec;
use oa_bt::model::{days_from_civil, ts_from_ist, Day, Exchange, InstrumentKind, Product, Side};
use oa_bt::rms::{LegRules, StrategyRiskConfig};
use oa_bt::sizing::SizingControls;

pub const MINUTE: i64 = 60_000_000_000;
/// 09:15 IST, the NSE session open, in minutes since midnight.
pub const NSE_OPEN_MINUTE: u32 = 9 * 60 + 15;
/// 15:30 IST, the NSE session close.
pub const NSE_CLOSE_MINUTE: u32 = 15 * 60 + 30;
pub const CAPITAL: f64 = 10_000_000.0;
pub const SBIN: InstrumentId = InstrumentId(0);

/// One bar: (open, high, low, close), rupees.
pub type Ohlc = (f64, f64, f64, f64);

pub fn july_2025(day: u32) -> Day {
    days_from_civil(2025, 7, day)
}

/// The open of minute `k` of the 2025-07-01 session (k = 0 opens at 09:15).
pub fn minute(k: u32) -> i64 {
    ts_from_ist(july_2025(1), NSE_OPEN_MINUTE + k)
}

/// The session open of a day: the time a daily bar carries.
pub fn day_open(day: Day) -> i64 {
    ts_from_ist(day, NSE_OPEN_MINUTE)
}

/// The session close of a day: the decision instant of its daily bar.
pub fn day_close(day: Day) -> i64 {
    ts_from_ist(day, NSE_CLOSE_MINUTE)
}

/// Run a daily strategy on `bars` (one per session day, in order) with the decisions
/// made at the given days' closes; the output and the run's GTTs and script.
pub fn run_daily(
    bars: &[(Day, Ohlc)],
    acts: Vec<(Day, Act)>,
    adjust: impl FnOnce(&mut RunConfig),
) -> (RunOutput, Vec<Gtt>, Script) {
    let first = bars.first().map_or(july_2025(1), |(day, _)| *day);
    let last = bars.last().map_or(first, |(day, _)| *day);
    let mut config = daily_config(first, last);
    adjust(&mut config);
    let mut kernel = engine(config, equity(), daily_bars(bars));
    let acts = acts
        .into_iter()
        .map(|(day, act)| (day_close(day), act))
        .collect();
    let script = run_kept(&mut kernel, acts);
    let gtts = kernel.gtts().to_vec();
    (kernel.finish().expect("the run finished"), gtts, script)
}

/// An NSE cash-equity instrument: lot 1, tick 0.05, multiplier 1.
pub fn equity() -> InstrumentSpec {
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

/// An NFO option: lot 75, tick 0.05.
pub fn option() -> InstrumentSpec {
    InstrumentSpec {
        symbol: "NIFTY31JUL2525000CE".to_string(),
        exchange: Exchange::Nfo,
        kind: InstrumentKind::Option,
        lot_size_units: 75,
        tick_size_rupees: 0.05,
        ..equity()
    }
}

fn push(columns: &mut BarColumns, ts: i64, (open, high, low, close): Ohlc) {
    columns.ts_open.push(ts);
    columns.open.push(open);
    columns.high.push(high);
    columns.low.push(low);
    columns.close.push(close);
    columns.volume.push(1_000.0);
}

/// One-minute bars of 2025-07-01 for `count` minutes: flat at `flat` unless a minute is
/// in `overrides`.
pub fn minute_bars(count: u32, flat: f64, overrides: &[(u32, Ohlc)]) -> BarColumns {
    let mut columns = BarColumns::default();
    for k in 0..count {
        let bar = overrides
            .iter()
            .find(|(m, _)| *m == k)
            .map_or((flat, flat, flat, flat), |(_, bar)| *bar);
        push(&mut columns, minute(k), bar);
    }
    columns
}

/// Daily bars, one per session day given.
pub fn daily_bars(days: &[(Day, Ohlc)]) -> BarColumns {
    let mut columns = BarColumns::default();
    for (day, bar) in days {
        push(&mut columns, day_open(*day), *bar);
    }
    columns
}

fn config(days: (Day, Day), base: BaseInterval, interval: StrategyInterval) -> RunConfig {
    RunConfig {
        first_day: days.0,
        last_day: days.1,
        first_slice: None,
        last_slice: None,
        present: None,
        base,
        strategy_interval: interval,
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

/// A one-minute strategy on one-minute bars, on 2025-07-01.
pub fn minute_config() -> RunConfig {
    let day = july_2025(1);
    config(
        (day, day),
        BaseInterval::Intraday(MINUTE),
        StrategyInterval::Intraday(MINUTE),
    )
}

/// A daily strategy on daily bars from `first` to `last`.
pub fn daily_config(first: Day, last: Day) -> RunConfig {
    config((first, last), BaseInterval::Daily, StrategyInterval::Daily)
}

/// The engine of a run on the embedded calendar.
pub fn engine(config: RunConfig, instrument: InstrumentSpec, bars: BarColumns) -> Engine {
    let calendar = Calendar::new().expect("the embedded calendar tables parse");
    Engine::new(RunSetup {
        config,
        calendar,
        instruments: vec![instrument],
        bars: vec![bars],
    })
    .expect("a valid setup")
}

/// A request for `units` of SBIN on `side` in `product`, priced as `order_type`.
pub fn order(side: Side, units: f64, order_type: OrderType, product: Product) -> OrderRequest {
    OrderRequest {
        order_type,
        product,
        ..OrderRequest::market(SBIN, side, QuantitySpec::Units(units))
    }
}

pub fn limit(limit_rupees: f64) -> OrderType {
    OrderType::Limit { limit_rupees }
}

pub fn stop_market(trigger_rupees: f64) -> OrderType {
    OrderType::StopMarket { trigger_rupees }
}

pub fn stop_limit(trigger_rupees: f64, limit_rupees: f64) -> OrderType {
    OrderType::StopLimit {
        trigger_rupees,
        limit_rupees,
    }
}

/// A CNC market buy with leg rules.
pub fn positional_buy(units: f64, rules: LegRules) -> OrderRequest {
    OrderRequest {
        leg_rules: rules,
        ..order(Side::Buy, units, OrderType::Market, Product::Cnc)
    }
}

/// A CNC GTT order.
pub fn gtt_order(side: Side, units: f64, order_type: OrderType) -> OrderRequest {
    OrderRequest {
        validity: Validity::Gtt,
        ..order(side, units, order_type, Product::Cnc)
    }
}

/// An order that closes the whole position of SBIN in `product`.
pub fn exit_all(held_side: Side, product: Product) -> OrderRequest {
    let side = match held_side {
        Side::Buy => Side::Sell,
        Side::Sell => Side::Buy,
    };
    OrderRequest {
        reduce_only: true,
        product,
        ..OrderRequest::market(SBIN, side, QuantitySpec::PositionFraction(1.0))
    }
}

/// What the script does at a decision instant.
#[derive(Clone, Debug)]
pub enum Act {
    Submit(OrderRequest),
    Modify(OrderId, OrderRequest),
    Cancel(OrderId),
}

/// A driver that acts at given decision instants (strategy-bar closes, UTC
/// nanoseconds) and keeps what each call returned.
pub struct Script {
    acts: Vec<(i64, Act)>,
    pub submissions: Vec<Submission>,
    pub modifications: Vec<Result<Submission, KernelError>>,
}

impl Driver for Script {
    fn on_pause(&mut self, engine: &mut Engine, pause: &Pause) -> Result<(), KernelError> {
        let Pause::BarClose { decided_at, .. } = pause else {
            return Ok(());
        };
        let due: Vec<Act> = self
            .acts
            .iter()
            .filter(|(instant, _)| instant == decided_at)
            .map(|(_, act)| act.clone())
            .collect();
        for act in due {
            match act {
                Act::Submit(request) => self.submissions.push(engine.submit(request)?),
                Act::Modify(id, request) => self.modifications.push(engine.modify(id, request)),
                Act::Cancel(id) => {
                    engine.cancel(id)?;
                }
            }
        }
        Ok(())
    }
}

/// Run a script to the end; the finished engine's output and the script.
pub fn run_script(mut kernel: Engine, acts: Vec<(i64, Act)>) -> (RunOutput, Script) {
    let mut script = Script {
        acts,
        submissions: Vec::new(),
        modifications: Vec::new(),
    };
    run(&mut kernel, &mut script).expect("the run completes");
    (kernel.finish().expect("the run finished"), script)
}

/// Run a script and keep the engine (for its GTTs); it is finished but not consumed.
pub fn run_kept(kernel: &mut Engine, acts: Vec<(i64, Act)>) -> Script {
    let mut script = Script {
        acts,
        submissions: Vec::new(),
        modifications: Vec::new(),
    };
    run(kernel, &mut script).expect("the run completes");
    script
}

/// One fill as (slice open, step, side, units, price, where, reason).
pub type FillRow = (i64, Step, Side, f64, f64, FillAt, &'static str);

pub fn fills(output: &RunOutput) -> Vec<FillRow> {
    output
        .fills
        .iter()
        .map(|record| {
            let fill = record.fill;
            (
                record.stamp.slice_ts,
                record.stamp.step,
                fill.side,
                fill.quantity_units,
                fill.price_rupees,
                fill.at,
                fill.reason.code(),
            )
        })
        .collect()
}

/// One event as (slice open, step number, name): a rejection with its code.
pub fn events(output: &RunOutput) -> Vec<(i64, u8, String)> {
    output
        .events
        .iter()
        .map(|event| {
            let stamp = event.stamp();
            let name = match event {
                Event::Order {
                    kind: OrderEventKind::Rejected(code),
                    ..
                } => format!("Rejected {}", code.code()),
                Event::Order { kind, .. } => kind.code().to_string(),
                Event::Position { change, .. } => change.code().to_string(),
                Event::Risk { event, .. } => event.name.code().to_string(),
                Event::TradingState { from, to, .. } => {
                    format!("state {} -> {}", from.code(), to.code())
                }
                Event::Gtt { event, .. } => format!("{} {}", event.name.code(), event.gtt),
            };
            (stamp.slice_ts, stamp.step.number(), name)
        })
        .collect()
}

/// The events of one slice, by name only.
pub fn events_at(output: &RunOutput, slice_ts: i64) -> Vec<String> {
    events(output)
        .into_iter()
        .filter(|(ts, _, _)| *ts == slice_ts)
        .map(|(_, _, name)| name)
        .collect()
}
