//! The engines, bars and scripted strategies of the fund's tests.
//!
//! Every run is a backtest on the embedded calendar with no charges and no slippage, in
//! whole shares, of one instrument: SBIN or INFY on NSE (cash equity, lot 1, tick 0.05),
//! or a CRUDEOIL future on MCX (lot 100, tick 1.00), on 1-minute or daily bars. Cash
//! equity in MIS and CNC blocks its whole value as margin (1x), a future 20 percent of
//! its value (the default of D23). A strategy is a script: the orders it submits at the
//! step 7 pause of a slice, by slice index.

use crate::engine::{
    run, BarColumns, CostConfig, Driver, Engine, FillTiming, IntradayOverrides, KernelError, Pause,
    Perturbation, QuantityMode, RunConfig, RunOutput, RunSetup, StrategyInterval,
};
use crate::exec::{InstrumentId, InstrumentSpec, OrderRequest, OrderType, QuantitySpec};
use crate::market::calendar::{BaseInterval, Calendar};
use crate::market::facts::MarketFacts;
use crate::market::margin::MarginModel;
use crate::market::slippage::SlippageSpec;
use crate::model::{days_from_civil, ts_from_ist, Day, Exchange, InstrumentKind, Product, Side};
use crate::rms::{Distance, StrategyRiskConfig};
use crate::sizing::SizingControls;

use super::super::{Fund, FundConfig, FundOutput, FundPause, Pool};

/// The one instrument of every run.
pub(super) const ONE: InstrumentId = InstrumentId(0);
const MINUTE_NS: i64 = 60_000_000_000;
/// 09:15 IST, the NSE session open, in minutes since midnight.
const NSE_OPEN: u32 = 9 * 60 + 15;
/// One-minute slices in an NSE session (09:15 to 15:30).
pub(super) const NSE_MINUTES: u32 = 375;
/// 09:00 IST, the MCX session open, in minutes since midnight.
const MCX_OPEN: u32 = 9 * 60;
/// One-minute slices in an MCX session of July 2025 (09:00 to 23:30, the close while the
/// United States keep daylight saving time; the embedded calendar's MCX row).
pub(super) const MCX_MINUTES: u32 = 870;

/// A day of July 2025, as days since 1970-01-01.
pub(super) fn july(day: u32) -> Day {
    days_from_civil(2025, 7, day)
}

/// An IST wall-clock time of `day`, UTC nanoseconds.
pub(super) fn ist(day: Day, hour: u32, minute: u32) -> i64 {
    ts_from_ist(day, hour * 60 + minute)
}

/// The instrument a run trades.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Symbol {
    Sbin,
    Infy,
    Crude,
}

impl Symbol {
    fn spec(self) -> InstrumentSpec {
        let (symbol, exchange, kind, lot, tick) = match self {
            Symbol::Sbin => ("SBIN", Exchange::Nse, InstrumentKind::Equity, 1, 0.05),
            Symbol::Infy => ("INFY", Exchange::Nse, InstrumentKind::Equity, 1, 0.05),
            Symbol::Crude => (
                "CRUDEOIL25JULFUT",
                Exchange::Mcx,
                InstrumentKind::Future,
                100,
                1.0,
            ),
        };
        InstrumentSpec {
            symbol: symbol.to_string(),
            exchange,
            kind,
            lot_size_units: lot,
            tick_size_rupees: tick,
            multiplier: 1.0,
            freeze_quantity_units: None,
            cost_segment: None,
            last_trading_day: (kind == InstrumentKind::Future).then(|| july(18)),
        }
    }

    /// The session's open (minutes since midnight) and its 1-minute slices.
    fn session(self) -> (u32, u32) {
        match self {
            Symbol::Crude => (MCX_OPEN, MCX_MINUTES),
            _ => (NSE_OPEN, NSE_MINUTES),
        }
    }
}

/// How a run is set up.
#[derive(Clone, Copy, Debug)]
pub(super) struct Run {
    pub(super) symbol: Symbol,
    pub(super) days: (Day, Day),
    pub(super) daily: bool,
    pub(super) capital_rupees: f64,
    pub(super) max_daily_loss_rupees: Option<f64>,
}

impl Run {
    /// A 1-minute strategy on `symbol` from `first` to `last` with `capital_rupees`.
    pub(super) fn minutes(symbol: Symbol, first: Day, last: Day, capital_rupees: f64) -> Run {
        Run {
            symbol,
            days: (first, last),
            daily: false,
            capital_rupees,
            max_daily_loss_rupees: None,
        }
    }

    /// A daily strategy on `symbol` from `first` to `last` with `capital_rupees`.
    pub(super) fn daily(symbol: Symbol, first: Day, last: Day, capital_rupees: f64) -> Run {
        Run {
            daily: true,
            ..Run::minutes(symbol, first, last, capital_rupees)
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
            quantity_mode: QuantityMode::Venue,
            capital_rupees: self.capital_rupees,
            costs: CostConfig::Off,
            slippage: SlippageSpec::Zero,
            margin: MarginModel,
            market: MarketFacts::default(),
            risk: StrategyRiskConfig {
                max_daily_loss_rupees: self.max_daily_loss_rupees,
                ..StrategyRiskConfig::default()
            },
            sizing: SizingControls::default(),
            intraday: IntradayOverrides::default(),
            deliver_events: false,
            perturbation: Perturbation::default(),
        }
    }

    /// The run's engine on bars priced by `price(day index, slice of the day)` as (open,
    /// high, low, close); a daily run's slice of the day is 0.
    pub(super) fn engine(self, price: impl Fn(usize, u32) -> [f64; 4]) -> Engine {
        Engine::new(self.setup(price)).expect("a valid run")
    }

    /// The run's setup on bars priced as `engine` prices them.
    pub(super) fn setup(self, price: impl Fn(usize, u32) -> [f64; 4]) -> RunSetup {
        RunSetup {
            config: self.config(),
            calendar: Calendar::new().expect("the embedded calendar tables parse"),
            instruments: vec![self.symbol.spec()],
            bars: vec![self.bars(price)],
        }
    }

    /// One bar per slice of every weekday of the run (the tests' days are trading days).
    fn bars(self, price: impl Fn(usize, u32) -> [f64; 4]) -> BarColumns {
        let mut columns = BarColumns::default();
        let (open, minutes) = self.symbol.session();
        let days = (self.days.0..=self.days.1).filter(|&day| weekday(day));
        for (index, day) in days.enumerate() {
            let slices = if self.daily { 1 } else { minutes };
            for k in 0..slices {
                let [o, h, l, c] = price(index, k);
                columns.ts_open.push(ts_from_ist(day, open + k));
                columns.open.push(o);
                columns.high.push(h);
                columns.low.push(l);
                columns.close.push(c);
                columns.volume.push(1_000.0);
            }
        }
        columns
    }
}

/// True for Monday to Friday (1970-01-01 was a Thursday).
fn weekday(day: Day) -> bool {
    (i64::from(day) + 3).rem_euclid(7) < 5
}

/// A flat bar at `price`.
pub(super) fn flat(price: f64) -> [f64; 4] {
    [price; 4]
}

/// A bar of a moving market around `base` rupees, in whole rupees (on every tick grid
/// of the tests): its open and close step over 0 to 20 rupees above `base` from one slice
/// to the next, its high and low one rupee beyond them.
pub(super) fn wave(base: f64, day: usize, k: u32) -> [f64; 4] {
    let step = |a: usize, b: usize| ((k as usize * a + day * b) % 21) as f64;
    let (open, close) = (base + step(37, 11), base + step(53, 7));
    [open, open.max(close) + 1.0, open.min(close) - 1.0, close]
}

/// A MARKET order of `units` on `side` in `product` with a stop `points` rupees away.
pub(super) fn with_stop(side: Side, units: f64, product: Product, points: f64) -> OrderRequest {
    let mut order = market(side, units, product);
    order.leg_rules.stop = Some(Distance::Points(points));
    order
}

/// The exit of the whole position in `product`: a MARKET order on `side`, reduce-only.
pub(super) fn exit_all(side: Side, product: Product) -> OrderRequest {
    OrderRequest {
        product,
        reduce_only: true,
        ..OrderRequest::market(ONE, side, QuantitySpec::PositionFraction(1.0))
    }
}

/// A MARKET order of `units` on `side` in `product`.
pub(super) fn market(side: Side, units: f64, product: Product) -> OrderRequest {
    OrderRequest {
        product,
        ..OrderRequest::market(ONE, side, QuantitySpec::Units(units))
    }
}

/// A LIMIT DAY buy of `units` at `limit_rupees` in `product`.
pub(super) fn limit_buy(units: f64, limit_rupees: f64, product: Product) -> OrderRequest {
    OrderRequest {
        order_type: OrderType::Limit { limit_rupees },
        ..market(Side::Buy, units, product)
    }
}

/// A strategy: the orders it submits at the step 7 pause of a slice, and at the timer
/// pause (step 2) of a slice, by slice index.
#[derive(Clone, Debug, Default)]
pub(super) struct Script {
    pub(super) orders: Vec<(u32, OrderRequest)>,
    pub(super) at_timers: Vec<(u32, OrderRequest)>,
}

impl Script {
    pub(super) fn new(orders: &[(u32, OrderRequest)]) -> Script {
        Script {
            orders: orders.to_vec(),
            at_timers: Vec::new(),
        }
    }
}

impl Driver for Script {
    fn on_pause(&mut self, engine: &mut Engine, pause: &Pause) -> Result<(), KernelError> {
        let (orders, slice) = match pause {
            Pause::BarClose { stamp, .. } => (&self.orders, stamp.slice),
            Pause::Timers { stamp, .. } => (&self.at_timers, stamp.slice),
            Pause::Finished { .. } | Pause::Waiting { .. } => return Ok(()),
        };
        for (_, request) in orders.iter().filter(|(at, _)| *at == slice) {
            engine.submit(request.clone())?;
        }
        Ok(())
    }
}

/// The run of one engine alone with its script, to its end.
pub(super) fn alone(mut engine: Engine, script: &Script) -> RunOutput {
    run(&mut engine, &mut script.clone()).expect("the run runs");
    engine.finish().expect("finished")
}

/// A fund of `pool` over `engines` with their scripts, run to its end: each engine's
/// output, in call order, and the fund's.
pub(super) fn in_fund(
    config: FundConfig,
    mut engines: Vec<Engine>,
    scripts: &[Script],
) -> (Vec<RunOutput>, FundOutput) {
    let fund = Fund::new(config, &engines.iter().collect::<Vec<_>>()).expect("a valid fund");
    let fund = drive(fund, &mut engines, scripts);
    let outputs = engines.into_iter().map(|e| e.finish().expect("finished"));
    (outputs.collect(), fund.finish().expect("the fund finished"))
}

/// Advance `fund` to its end, deciding every pause with the strategy's script.
pub(super) fn drive(mut fund: Fund, engines: &mut [Engine], scripts: &[Script]) -> Fund {
    let mut scripts = scripts.to_vec();
    loop {
        let mut lent: Vec<&mut Engine> = engines.iter_mut().collect();
        match fund.advance(&mut lent).expect("the fund advances") {
            FundPause::Finished => return fund,
            FundPause::Strategy { strategy, pause } => {
                let decided = scripts[strategy].on_pause(lent[strategy], &pause);
                decided.expect("the script decides");
            }
        }
    }
}

/// A fund's configuration: `pool`, the account's capital and its daily loss limit.
pub(super) fn account(pool: Pool, capital: Option<f64>, limit: Option<f64>) -> FundConfig {
    FundConfig {
        pool,
        account_capital_rupees: capital,
        account_max_daily_loss_rupees: limit,
    }
}

/// Every fill of an output as (slice, step number, side, units, price, exit reason).
pub(super) fn fills(output: &RunOutput) -> Vec<(u32, u8, Side, f64, f64, &'static str)> {
    output
        .fills
        .iter()
        .map(|record| {
            let fill = record.fill;
            let (stamp, reason) = (record.stamp, fill.reason.code());
            (
                stamp.slice,
                stamp.step.number(),
                fill.side,
                fill.quantity_units,
                fill.price_rupees,
                reason,
            )
        })
        .collect()
}
