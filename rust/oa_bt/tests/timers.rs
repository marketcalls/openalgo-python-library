#![cfg(all(feature = "costs", feature = "time"))]
//! Timers (`self.at`, `self.every`) and the strategy bars of a whole run, through the
//! stepping API.
//!
//! Intraday cases use one instrument with 1-minute base bars on 2025-07-01 (a Tuesday)
//! from 09:15 IST: the bar opening k minutes after 09:15 opens at 500.00 + 0.10 k. Daily
//! cases use one daily bar per session, opening at 100 + the day of the month.

use oa_bt::data::resample::CalendarUnit;
use oa_bt::engine::{
    BarColumns, CostConfig, Engine, FillTiming, IntradayOverrides, KernelError, Pause,
    QuantityMode, RunConfig, RunSetup, Step, StrategyInterval, Submission, TimerCount, TimerId,
    TimerSpec,
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
const SBIN: InstrumentId = InstrumentId(0);

fn july(day: u32) -> Day {
    days_from_civil(2025, 7, day)
}

/// IST clock time on 2025-07-01, as UTC nanoseconds.
fn at(hour: u32, minute: u32) -> i64 {
    ts_from_ist(july(1), hour * 60 + minute)
}

fn minute_bars(count: u32, missing: &[u32]) -> BarColumns {
    let mut columns = BarColumns::default();
    for k in (0..count).filter(|k| !missing.contains(k)) {
        let open = 500.0 + 0.10 * f64::from(k);
        columns
            .ts_open
            .push(ts_from_ist(july(1), SESSION_OPEN_MINUTE + k));
        push_prices(&mut columns, open);
    }
    columns
}

/// One daily bar per given session date, opening at 09:15.
fn daily_bars(days: &[Day]) -> BarColumns {
    let mut columns = BarColumns::default();
    for (number, &day) in days.iter().enumerate() {
        columns.ts_open.push(ts_from_ist(day, SESSION_OPEN_MINUTE));
        push_prices(&mut columns, 100.0 + number as f64);
    }
    columns
}

fn push_prices(columns: &mut BarColumns, open: f64) {
    columns.open.push(open);
    columns.high.push(open + 0.50);
    columns.low.push(open - 0.50);
    columns.close.push(open + 0.05);
    columns.volume.push(1_000.0);
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

fn config(
    first_day: Day,
    last_day: Day,
    base: BaseInterval,
    interval: StrategyInterval,
) -> RunConfig {
    RunConfig {
        first_day,
        last_day,
        first_slice: None,
        last_slice: None,
        present: None,
        base,
        strategy_interval: interval,
        fill: FillTiming::NextOpen,
        quantity_mode: QuantityMode::Venue,
        capital_rupees: 1_000_000.0,
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

fn engine(config: RunConfig, bars: BarColumns) -> Engine {
    let setup = RunSetup {
        config,
        calendar: Calendar::new().expect("the embedded calendar tables parse"),
        instruments: vec![sbin()],
        bars: vec![bars],
    };
    Engine::new(setup).expect("a valid setup")
}

/// A one-day intraday run: 1-minute base, 15-minute strategy.
fn intraday(bars: BarColumns) -> Engine {
    let interval = StrategyInterval::Intraday(15 * MINUTE);
    engine(
        config(july(1), july(1), BaseInterval::Intraday(MINUTE), interval),
        bars,
    )
}

fn daily(first: Day, last: Day, bars: BarColumns) -> Engine {
    let interval = StrategyInterval::Daily;
    engine(config(first, last, BaseInterval::Daily, interval), bars)
}

fn buy(units: f64, product: Product) -> OrderRequest {
    OrderRequest {
        product,
        ..OrderRequest::market(SBIN, Side::Buy, QuantitySpec::Units(units))
    }
}

/// Advance to the end; the (decision instant, timers) of every timer pause.
fn timer_pauses(kernel: &mut Engine) -> Vec<(i64, Vec<TimerId>)> {
    let mut seen = Vec::new();
    loop {
        match kernel.advance().expect("the run advances") {
            Pause::Timers {
                decided_at, timers, ..
            } => seen.push((decided_at, timers)),
            Pause::Finished { .. } => return seen,
            Pause::BarClose { .. } => {}
        }
    }
}

fn at_time(hour: u32, minute: u32) -> TimerSpec {
    TimerSpec::At {
        minute_of_day: hour * 60 + minute,
    }
}

#[test]
fn an_at_timer_pauses_in_step_2_of_the_slice_opening_at_its_time() {
    let mut kernel = intraday(minute_bars(375, &[]));
    let timer = kernel.add_timer(at_time(9, 20)).expect("before the run");
    let pause = kernel.advance().expect("the run advances");
    assert_eq!(
        pause,
        Pause::Timers {
            stamp: oa_bt::engine::EventStamp {
                slice: 5,
                slice_ts: at(9, 20),
                step: Step::OpenFills,
            },
            decided_at: at(9, 20),
            timers: vec![timer],
        }
    );
    assert_eq!(kernel.decision_instant(), Some(at(9, 20)));
}

#[test]
fn an_order_placed_by_a_timer_fills_at_that_same_open() {
    let mut kernel = intraday(minute_bars(375, &[]));
    kernel.add_timer(at_time(9, 20)).expect("before the run");
    let pause = kernel.advance().expect("the run advances");
    assert!(matches!(pause, Pause::Timers { .. }));
    let submitted = kernel
        .submit(buy(10.0, Product::Cnc))
        .expect("at a decision point");
    assert!(matches!(submitted, Submission::Submitted(_)));
    while !matches!(kernel.advance().expect("advances"), Pause::Finished { .. }) {}
    let output = kernel.finish().expect("finished");
    let fills: Vec<(i64, Step, f64)> = output
        .fills
        .iter()
        .map(|r| (r.stamp.slice_ts, r.stamp.step, r.fill.price_rupees))
        .collect();
    // The 09:20 bar is k = 5: it opens at 500.50.
    assert_eq!(fills, vec![(at(9, 20), Step::OpenFills, 500.5)]);
    assert_eq!(output.orders[0].decided_at, at(9, 20));
}

#[test]
fn a_timer_before_the_session_open_fires_at_the_first_slice() {
    let mut kernel = intraday(minute_bars(375, &[]));
    let timer = kernel.add_timer(at_time(9, 0)).expect("before the run");
    assert_eq!(timer_pauses(&mut kernel), vec![(at(9, 15), vec![timer])]);
}

#[test]
fn an_at_timer_fires_once_per_session() {
    let mut kernel = intraday(minute_bars(375, &[]));
    let timer = kernel.add_timer(at_time(10, 0)).expect("before the run");
    assert_eq!(timer_pauses(&mut kernel), vec![(at(10, 0), vec![timer])]);
    assert_eq!(
        kernel.timer_counts(),
        vec![TimerCount {
            timer,
            fired: 1,
            missed: 0
        }]
    );
}

#[test]
fn a_session_with_no_slice_at_or_after_the_time_counts_as_missed() {
    let mut kernel = intraday(minute_bars(375, &[]));
    let timer = kernel.add_timer(at_time(15, 45)).expect("before the run");
    assert!(timer_pauses(&mut kernel).is_empty());
    assert_eq!(
        kernel.timer_counts(),
        vec![TimerCount {
            timer,
            fired: 0,
            missed: 1
        }]
    );
}

#[test]
fn timers_due_at_the_same_open_fire_together_in_registration_order() {
    let mut kernel = intraday(minute_bars(375, &[]));
    let first = kernel.add_timer(at_time(9, 30)).expect("before the run");
    let second = kernel.add_timer(at_time(9, 25)).expect("before the run");
    assert_eq!(
        timer_pauses(&mut kernel),
        vec![(at(9, 25), vec![second]), (at(9, 30), vec![first])]
    );
    let mut kernel = intraday(minute_bars(375, &[]));
    let first = kernel.add_timer(at_time(9, 30)).expect("before the run");
    let second = kernel.add_timer(at_time(9, 30)).expect("before the run");
    assert_eq!(
        timer_pauses(&mut kernel),
        vec![(at(9, 30), vec![first, second])]
    );
}

#[test]
fn a_timer_pause_comes_after_the_orders_due_at_that_open_and_carries_no_events() {
    let mut kernel = intraday(minute_bars(375, &[]));
    kernel.add_timer(at_time(9, 30)).expect("before the run");
    // The 09:15 strategy bar closes at 09:30: buy there, to fill at the 09:30 open.
    let pause = kernel.advance().expect("advances");
    assert!(matches!(pause, Pause::BarClose { decided_at, .. } if decided_at == at(9, 30)));
    kernel
        .submit(buy(10.0, Product::Cnc))
        .expect("at a decision point");
    let pause = kernel.advance().expect("advances");
    assert!(matches!(pause, Pause::Timers { decided_at, .. } if decided_at == at(9, 30)));
    assert!(pause.events().is_empty());
    let held = kernel
        .position(SBIN, Product::Cnc)
        .map(|p| p.quantity_units);
    assert_eq!(held, Some(10.0));
}

#[test]
fn timers_are_registered_before_the_run_starts() {
    let mut kernel = intraday(minute_bars(375, &[]));
    kernel.advance().expect("advances");
    let error = kernel.add_timer(at_time(9, 30)).expect_err("too late");
    assert!(matches!(
        error,
        KernelError::InvalidConfig { field: "timer", .. }
    ));
}

#[test]
fn a_timer_time_outside_the_day_is_refused() {
    let mut kernel = intraday(minute_bars(375, &[]));
    let error = kernel
        .add_timer(TimerSpec::At {
            minute_of_day: 24 * 60,
        })
        .expect_err("no such time");
    assert!(matches!(
        error,
        KernelError::InvalidConfig { field: "timer", .. }
    ));
}

#[test]
fn every_is_refused_on_an_intraday_strategy() {
    let mut kernel = intraday(minute_bars(375, &[]));
    let error = kernel
        .add_timer(TimerSpec::Every(CalendarUnit::Week))
        .expect_err("intraday interval");
    assert!(matches!(
        error,
        KernelError::InvalidConfig { field: "timer", .. }
    ));
}

#[test]
fn every_week_fires_at_the_open_after_the_first_session_of_the_week() {
    // Tuesday 1 July to Friday 11 July 2025. Monday 30 June opened the first week, so it
    // does not fire there; Monday 7 July opens the second, so it fires at the Tuesday
    // 8 July open.
    let days: Vec<Day> = [1, 2, 3, 4, 7, 8, 9, 10, 11].map(july).to_vec();
    let mut kernel = daily(july(1), july(11), daily_bars(&days));
    let timer = kernel
        .add_timer(TimerSpec::Every(CalendarUnit::Week))
        .expect("before the run");
    let tuesday_open = ts_from_ist(july(8), SESSION_OPEN_MINUTE);
    assert_eq!(timer_pauses(&mut kernel), vec![(tuesday_open, vec![timer])]);
}

#[test]
fn every_month_counts_the_first_session_of_the_run_when_it_opens_the_month() {
    // Friday 1 August 2025 is August's first session; Thursday 31 July is not in the
    // run, so the calendar decides. It fires at the Monday 4 August open.
    let august = |day| days_from_civil(2025, 8, day);
    let days: Vec<Day> = [1, 4, 5].map(august).to_vec();
    let mut kernel = daily(august(1), august(5), daily_bars(&days));
    let timer = kernel
        .add_timer(TimerSpec::Every(CalendarUnit::Month))
        .expect("before the run");
    let monday_open = ts_from_ist(august(4), SESSION_OPEN_MINUTE);
    assert_eq!(timer_pauses(&mut kernel), vec![(monday_open, vec![timer])]);
}

#[test]
fn an_order_placed_by_an_every_timer_fills_at_that_open() {
    let days: Vec<Day> = [1, 2, 3, 4, 7, 8, 9].map(july).to_vec();
    let mut kernel = daily(july(1), july(9), daily_bars(&days));
    kernel
        .add_timer(TimerSpec::Every(CalendarUnit::Week))
        .expect("before the run");
    loop {
        match kernel.advance().expect("advances") {
            Pause::Timers { .. } => {
                kernel
                    .submit(buy(10.0, Product::Cnc))
                    .expect("at a decision point");
            }
            Pause::Finished { .. } => break,
            Pause::BarClose { .. } => {}
        }
    }
    let output = kernel.finish().expect("finished");
    let tuesday_open = ts_from_ist(july(8), SESSION_OPEN_MINUTE);
    // 8 July is the sixth session: it opens at 105.
    let fills: Vec<(i64, f64)> = output
        .fills
        .iter()
        .map(|r| (r.stamp.slice_ts, r.fill.price_rupees))
        .collect();
    assert_eq!(fills, vec![(tuesday_open, 105.0)]);
}

#[test]
fn the_strategy_bars_of_the_run_are_the_bars_the_pauses_deliver() {
    // Base bars k = 20 to 29 are missing: the 09:30 strategy bar is built from the
    // rest, and nothing differs between the pauses and the whole-run series.
    let missing: Vec<u32> = (20..30).collect();
    let mut kernel = intraday(minute_bars(375, &missing));
    let mut delivered = Vec::new();
    loop {
        match kernel.advance().expect("advances") {
            Pause::BarClose { bars, .. } => delivered.extend(bars),
            Pause::Finished { .. } => break,
            Pause::Timers { .. } => {}
        }
    }
    let whole = kernel.strategy_bars(SBIN);
    assert_eq!(whole.len(), 25);
    assert_eq!(whole, delivered);
    assert!(kernel.strategy_bars(InstrumentId(7)).is_empty());
}
