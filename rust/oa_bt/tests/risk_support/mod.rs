//! Shared harness of the strategy and session risk-rule tests: instruments, bars,
//! configurations, a scripted driver and a readable summary of the event stream.
//!
//! Every test file that includes this module uses a different part of it, so unused
//! items are allowed here rather than duplicated per file.
#![allow(dead_code)]

use oa_bt::engine::{
    run, BarColumns, CostConfig, Driver, Engine, Event, FillTiming, IntradayOverrides, KernelError,
    OrderEventKind, Pause, QuantityMode, RunConfig, RunOutput, RunSetup, Step, StrategyInterval,
    Submission,
};
use oa_bt::exec::{FillAt, InstrumentId, InstrumentSpec, OrderRequest, QuantitySpec};
use oa_bt::market::calendar::{BaseInterval, Calendar};
use oa_bt::market::facts::MarketFacts;
use oa_bt::market::margin::MarginModel;
use oa_bt::market::slippage::SlippageSpec;
use oa_bt::model::{days_from_civil, ts_from_ist, Day, Exchange, InstrumentKind, Product, Side};
use oa_bt::rms::{StrategyRiskConfig, TradingState};

pub const MINUTES_15: i64 = 15 * 60_000_000_000;
/// 09:15 IST, the NSE session open, in minutes since midnight.
pub const NSE_OPEN_MINUTE: u32 = 9 * 60 + 15;
/// 15:30 IST, the NSE session close.
pub const NSE_CLOSE_MINUTE: u32 = 15 * 60 + 30;
/// Fifteen-minute bars in one NSE session (09:15 to 15:30).
pub const BARS_15M_PER_SESSION: usize = 25;

pub fn july_2025(day: u32) -> Day {
    days_from_civil(2025, 7, day)
}

/// An IST clock time on a day, as UTC nanoseconds.
pub fn at(day: Day, hour: u32, minute: u32) -> i64 {
    ts_from_ist(day, hour * 60 + minute)
}

/// Open of the 15-minute bar `k` of a session (k = 0 opens at 09:15).
pub fn bar_open(day: Day, k: usize) -> i64 {
    ts_from_ist(day, NSE_OPEN_MINUTE + 15 * k as u32)
}

/// Close of the 15-minute bar `k`: the decision instant of that bar.
pub fn bar_close(day: Day, k: usize) -> i64 {
    bar_open(day, k + 1)
}

/// The decision instant of a daily bar: the session close.
pub fn day_close(day: Day) -> i64 {
    ts_from_ist(day, NSE_CLOSE_MINUTE)
}

/// An NSE cash-equity instrument: lot 1, tick 0.05, multiplier 1.
pub fn equity(symbol: &str) -> InstrumentSpec {
    InstrumentSpec {
        symbol: symbol.to_string(),
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

/// Append one bar; the high and low are the open and close's extremes.
pub fn push_bar(columns: &mut BarColumns, ts_open: i64, open: f64, close: f64) {
    push_ohlc(
        columns,
        ts_open,
        [open, open.max(close), open.min(close), close],
    );
}

/// Append one bar with all four prices.
pub fn push_ohlc(columns: &mut BarColumns, ts_open: i64, [open, high, low, close]: [f64; 4]) {
    columns.ts_open.push(ts_open);
    columns.open.push(open);
    columns.high.push(high);
    columns.low.push(low);
    columns.close.push(close);
    columns.volume.push(1_000.0);
}

/// Fifteen-minute bars of whole sessions: `price(day_index, k)` gives (open, close) of
/// bar k, or `None` for a missing bar.
pub fn sessions_15m(
    days: &[Day],
    price: impl Fn(usize, usize) -> Option<(f64, f64)>,
) -> BarColumns {
    let mut columns = BarColumns::default();
    for (index, &day) in days.iter().enumerate() {
        for k in 0..BARS_15M_PER_SESSION {
            if let Some((open, close)) = price(index, k) {
                push_bar(&mut columns, bar_open(day, k), open, close);
            }
        }
    }
    columns
}

/// A run configuration with no charges, no slippage, venue quantities and the given
/// intervals, capital and risk rules.
pub fn config(
    days: (Day, Day),
    base: BaseInterval,
    interval: StrategyInterval,
    capital_rupees: f64,
    risk: StrategyRiskConfig,
) -> RunConfig {
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
        capital_rupees,
        costs: CostConfig::Off,
        slippage: SlippageSpec::Zero,
        margin: MarginModel,
        market: MarketFacts::default(),
        risk,
        intraday: IntradayOverrides::default(),
        deliver_events: false,
    }
}

/// A 15-minute strategy on a 15-minute base.
pub fn intraday_config(days: (Day, Day), risk: StrategyRiskConfig) -> RunConfig {
    let interval = StrategyInterval::Intraday(MINUTES_15);
    config(
        days,
        BaseInterval::Intraday(MINUTES_15),
        interval,
        1_000_000.0,
        risk,
    )
}

/// The engine of a run on the embedded calendar.
pub fn engine(
    config: RunConfig,
    instruments: Vec<InstrumentSpec>,
    bars: Vec<BarColumns>,
) -> Result<Engine, KernelError> {
    let calendar = Calendar::new().expect("the embedded calendar tables parse");
    Engine::new(RunSetup {
        config,
        calendar,
        instruments,
        bars,
    })
}

/// What the script does at a decision instant.
#[derive(Clone, Copy, Debug)]
pub enum Action {
    /// Buy units of an instrument (CNC).
    Buy(InstrumentId, f64),
    /// Exit the whole CNC position of an instrument.
    Exit(InstrumentId),
}

/// A driver that acts at given decision instants (strategy-bar closes) and keeps what
/// `submit` returned and the trading state it saw at each decision.
pub struct Script {
    actions: Vec<(i64, Action)>,
    pub submissions: Vec<(i64, Submission)>,
    pub states: Vec<(i64, TradingState)>,
}

impl Script {
    pub fn new(actions: Vec<(i64, Action)>) -> Script {
        Script {
            actions,
            submissions: Vec::new(),
            states: Vec::new(),
        }
    }
}

fn request(engine: &Engine, action: Action) -> OrderRequest {
    match action {
        Action::Buy(instrument, units) => {
            OrderRequest::market(instrument, Side::Buy, QuantitySpec::Units(units))
        }
        Action::Exit(instrument) => {
            let held = engine
                .position(instrument, Product::Cnc)
                .map_or(0.0, |p| p.quantity_units);
            let side = if held > 0.0 { Side::Sell } else { Side::Buy };
            OrderRequest {
                reduce_only: true,
                ..OrderRequest::market(instrument, side, QuantitySpec::PositionFraction(1.0))
            }
        }
    }
}

impl Driver for Script {
    fn on_pause(&mut self, engine: &mut Engine, pause: &Pause) -> Result<(), KernelError> {
        let Pause::BarClose { decided_at, .. } = pause else {
            return Ok(());
        };
        self.states.push((*decided_at, engine.trading_state()));
        let due: Vec<Action> = self
            .actions
            .iter()
            .filter(|(instant, _)| instant == decided_at)
            .map(|(_, action)| *action)
            .collect();
        for action in due {
            let submitted = engine.submit(request(engine, action))?;
            self.submissions.push((*decided_at, submitted));
        }
        Ok(())
    }
}

/// Run a script to the end; the result and the script.
pub fn run_script(mut kernel: Engine, actions: Vec<(i64, Action)>) -> (RunOutput, Script) {
    let mut script = Script::new(actions);
    run(&mut kernel, &mut script).expect("the run completes");
    (kernel.finish().expect("the run finished"), script)
}

/// One event as (slice open, step number, name): order events by name (a rejection
/// with its code), position events by change, risk events by rule, and trading-state
/// changes as "state FROM -> TO".
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
                Event::Gtt { event, .. } => event.name.code().to_string(),
            };
            (stamp.slice_ts, stamp.step.number(), name)
        })
        .collect()
}

/// One fill as (slice open, step, side, units, price, where, reason, has an order,
/// filled at a last close).
pub type FillRow = (i64, Step, Side, f64, f64, FillAt, &'static str, bool, bool);

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
                fill.order.is_some(),
                !fill.flags.names().is_empty(),
            )
        })
        .collect()
}

/// The outcome of a submission as text: "submitted" or "rejected <code>".
pub fn outcome(submission: &Submission) -> String {
    match submission {
        Submission::Submitted(_) => "submitted".to_string(),
        Submission::Rejected(_, code) => format!("rejected {}", code.code()),
        Submission::Skipped(reason) => format!("skipped {}", reason.code()),
    }
}

/// The outcomes of a script's submissions, in order.
pub fn outcomes(script: &Script) -> Vec<String> {
    script
        .submissions
        .iter()
        .map(|(_, submission)| outcome(submission))
        .collect()
}
