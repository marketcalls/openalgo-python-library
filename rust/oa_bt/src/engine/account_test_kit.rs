//! A harness for the unit tests of the phases of a slice (`steps`) and of a portfolio
//! run's account in one engine (`account_link`, `account_breach`): backtests of SBIN (and
//! INFY, where a test needs a second instrument) on NSE in July 2025 with no charges and
//! no slippage and a capital of 10,00,000, driven a phase at a time as a portfolio run's
//! fund drives them.

use crate::exec::{FillAt, InstrumentId, InstrumentSpec, OrderRequest, QuantitySpec};
use crate::market::calendar::{BaseInterval, Calendar};
use crate::market::facts::MarketFacts;
use crate::market::margin::MarginModel;
use crate::market::slippage::SlippageSpec;
use crate::model::{days_from_civil, ts_from_ist, Day, Exchange, InstrumentKind, Product, Side};
use crate::portfolio::PositionKey;
use crate::rms::StrategyRiskConfig;
use crate::sizing::SizingControls;

use super::clock::SliceInfo;
use super::config::{
    BarColumns, CostConfig, FillTiming, IntradayOverrides, Perturbation, QuantityMode, RunConfig,
    RunSetup, StrategyInterval,
};
use super::event::{Event, OrderEventKind, Step};
use super::kernel::Engine;
use super::pause::Pause;

pub(super) const SBIN: InstrumentId = InstrumentId(0);
pub(super) const INFY: InstrumentId = InstrumentId(1);
/// The symbols of a run's instruments, in registration order (`SBIN`, `INFY`).
const SYMBOLS: [&str; 2] = ["SBIN", "INFY"];
/// The capital of every run, rupees.
pub(super) const CAPITAL_RUPEES: f64 = 1_000_000.0;
/// One-minute slices in an NSE session (09:15 to 15:30).
pub(super) const MINUTES_PER_SESSION: u32 = 375;
const MINUTE_NS: i64 = 60_000_000_000;
/// 09:15 IST, the NSE session open, in minutes since midnight.
const NSE_OPEN_MINUTE: u32 = 9 * 60 + 15;

/// A day of July 2025, as days since 1970-01-01.
pub(super) fn july_2025(day: u32) -> Day {
    days_from_civil(2025, 7, day)
}

/// An IST wall-clock time of `day`, UTC nanoseconds.
pub(super) fn ist(day: Day, hour: u32, minute: u32) -> i64 {
    ts_from_ist(day, hour * 60 + minute)
}

/// An NSE equity with a lot of 1 and a tick of 0.05.
fn nse_equity(symbol: &str) -> InstrumentSpec {
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

/// How a run is set up: its days, base and strategy interval, quantity mode and risk.
#[derive(Clone, Copy)]
pub(super) struct Setup {
    pub(super) days: (Day, Day),
    pub(super) daily: bool,
    pub(super) quantity_mode: QuantityMode,
    pub(super) risk: StrategyRiskConfig,
}

impl Setup {
    /// A 1-minute strategy on 1-minute bars from `first` to `last`, whole shares.
    pub(super) fn minutes(first: Day, last: Day) -> Setup {
        Setup {
            days: (first, last),
            daily: false,
            quantity_mode: QuantityMode::Venue,
            risk: StrategyRiskConfig::default(),
        }
    }

    /// A daily strategy on daily bars from `first` to `last`, whole shares.
    pub(super) fn daily(first: Day, last: Day) -> Setup {
        Setup {
            daily: true,
            ..Setup::minutes(first, last)
        }
    }

    fn config(self) -> RunConfig {
        let (base, interval) = match self.daily {
            true => (BaseInterval::Daily, StrategyInterval::Daily),
            false => (
                BaseInterval::Intraday(MINUTE_NS),
                StrategyInterval::Intraday(MINUTE_NS),
            ),
        };
        RunConfig {
            first_day: self.days.0,
            last_day: self.days.1,
            first_slice: None,
            last_slice: None,
            present: None,
            base,
            strategy_interval: interval,
            fill: FillTiming::NextOpen,
            quantity_mode: self.quantity_mode,
            capital_rupees: CAPITAL_RUPEES,
            costs: CostConfig::Off,
            slippage: SlippageSpec::Zero,
            margin: MarginModel,
            market: MarketFacts::default(),
            risk: self.risk,
            sizing: SizingControls::default(),
            intraday: IntradayOverrides::default(),
            deliver_events: false,
            perturbation: Perturbation::default(),
        }
    }

    /// The engine of this run on `bars` of SBIN.
    pub(super) fn engine(self, bars: BarColumns) -> Engine {
        self.engine_over(vec![bars])
    }

    /// The engine of this run on one `BarColumns` per instrument: SBIN's, then INFY's.
    pub(super) fn engine_over(self, bars: Vec<BarColumns>) -> Engine {
        assert!(bars.len() <= SYMBOLS.len(), "bars of SBIN and INFY at most");
        let instruments = SYMBOLS[..bars.len()]
            .iter()
            .map(|symbol| nse_equity(symbol))
            .collect();
        let setup = RunSetup {
            config: self.config(),
            calendar: Calendar::new().expect("the embedded calendar tables parse"),
            instruments,
            bars,
        };
        Engine::new(setup).expect("a valid run")
    }
}

fn push(columns: &mut BarColumns, ts_open: i64, [open, high, low, close]: [f64; 4]) {
    columns.ts_open.push(ts_open);
    columns.open.push(open);
    columns.high.push(high);
    columns.low.push(low);
    columns.close.push(close);
    columns.volume.push(1_000.0);
}

/// One-minute bars of every minute of `days`: `price(day index, k)` is the (open, high,
/// low, close) of minute `k`, which opens at 09:15 + k.
pub(super) fn minute_bars(days: &[Day], price: impl Fn(usize, u32) -> [f64; 4]) -> BarColumns {
    let mut columns = BarColumns::default();
    for (index, &day) in days.iter().enumerate() {
        for k in 0..MINUTES_PER_SESSION {
            push(
                &mut columns,
                ts_from_ist(day, NSE_OPEN_MINUTE + k),
                price(index, k),
            );
        }
    }
    columns
}

/// A flat bar at `price`.
pub(super) fn flat(price: f64) -> [f64; 4] {
    [price; 4]
}

/// Daily bars, each opening at 09:15 IST of its day.
pub(super) fn daily_bars(rows: &[(Day, [f64; 4])]) -> BarColumns {
    let mut columns = BarColumns::default();
    for &(day, prices) in rows {
        push(&mut columns, ts_from_ist(day, NSE_OPEN_MINUTE), prices);
    }
    columns
}

/// A MARKET order of `units` of SBIN on `side` in `product`.
pub(super) fn market(side: Side, units: f64, product: Product) -> OrderRequest {
    OrderRequest {
        product,
        ..OrderRequest::market(SBIN, side, QuantitySpec::Units(units))
    }
}

/// The slice at `index` of the run's grid.
pub(super) fn slice(engine: &Engine, index: usize) -> SliceInfo {
    engine.timeline.slice(index).expect("a slice of the run")
}

/// Advance to the step-7 pause of slice `index` (every slice of these runs pauses there).
pub(super) fn pause_at(engine: &mut Engine, index: usize) {
    loop {
        match engine.advance().expect("the slice runs") {
            Pause::BarClose { stamp, .. } if stamp.slice as usize >= index => return,
            Pause::Finished { .. } => panic!("the run finished before slice {index}"),
            _ => {}
        }
    }
}

/// From the step-7 pause of slice `index`, end it: the engine stands before the next.
pub(super) fn end_paused(engine: &mut Engine, index: usize) {
    let ended = engine.end_slice(&slice(engine, index));
    assert_eq!(ended, Ok(None), "a backtest's slice ends without a pause");
}

/// From before slice `index`, run its open phase and its levels phase, as a fund runs a
/// participant of the round closing at the slice's close up to that round's account phase.
pub(super) fn open_and_level(engine: &mut Engine, index: usize) {
    let slice = slice(engine, index);
    let opened = engine.open_phase(&slice);
    assert_eq!(opened, Ok(None), "no timer is due");
    engine.levels_phase(&slice).expect("the levels phase runs");
}

/// Advance to the end of the run.
pub(super) fn run_to_end(engine: &mut Engine) {
    while !matches!(
        engine.advance().expect("the slice runs"),
        Pause::Finished { .. }
    ) {}
}

/// The signed units of SBIN held in `product`.
pub(super) fn held(engine: &Engine, product: Product) -> f64 {
    engine.held_units(PositionKey {
        instrument: SBIN,
        product,
    })
}

/// Every event so far as (slice index, step number, name): order events by name (a
/// rejection with its code), position events by change, risk events by rule.
pub(super) fn events(engine: &Engine) -> Vec<(u32, u8, String)> {
    engine
        .record
        .events
        .iter()
        .map(|event| {
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
            let stamp = event.stamp();
            (stamp.slice, stamp.step.number(), name)
        })
        .collect()
}

/// One fill as (slice index, step, side, units, price, where, reason, flagged).
pub(super) type FillRow = (u32, Step, Side, f64, f64, FillAt, &'static str, bool);

/// Every fill so far (`FillRow`).
pub(super) fn fills(engine: &Engine) -> Vec<FillRow> {
    engine
        .record
        .fills
        .iter()
        .map(|record| {
            let fill = record.fill;
            (
                record.stamp.slice,
                record.stamp.step,
                fill.side,
                fill.quantity_units,
                fill.price_rupees,
                fill.at,
                fill.reason.code(),
                !fill.flags.names().is_empty(),
            )
        })
        .collect()
}
