#![cfg(all(feature = "costs", feature = "time"))]
//! Where a run ends (03-architecture 4.2, "End of run"): at the earliest of the end of
//! its range, the close of the last base bar any instrument has, and the present.
//!
//! One NSE instrument on 2025-07-01 (session 09:15 to 15:30 IST), 1-minute base bars, a
//! 15-minute MIS strategy that buys 100 units at the first strategy-bar close and holds.
//! The bar opening k minutes after 09:15 opens at 500.00 + 0.10 k and closes 0.05 higher,
//! so every price below can be read off the inputs.

use oa_bt::engine::{
    run, BarColumns, CostConfig, Driver, Engine, FillTiming, IntradayOverrides, KernelError, Pause,
    QuantityMode, RunConfig, RunEnd, RunEndCause, RunOutput, RunSetup, StrategyInterval,
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
const SECOND: i64 = 1_000_000_000;
const SESSION_OPEN_MINUTE: u32 = 9 * 60 + 15;
/// Minutes of the 2025-07-01 NSE session, 09:15 to 15:30.
const SESSION_MINUTES: u32 = 375;
const SBIN: InstrumentId = InstrumentId(0);

fn day() -> Day {
    days_from_civil(2025, 7, 1)
}

/// IST clock time on the test day, as UTC nanoseconds.
fn at(hour: u32, minute: u32) -> i64 {
    ts_from_ist(day(), hour * 60 + minute)
}

/// Base bars from 09:15 for `count` minutes, leaving out the minutes in `missing`.
fn bars(count: u32, missing: &[u32]) -> BarColumns {
    let mut columns = BarColumns::default();
    for k in (0..count).filter(|k| !missing.contains(k)) {
        let open = 500.0 + 0.10 * f64::from(k);
        columns
            .ts_open
            .push(ts_from_ist(day(), SESSION_OPEN_MINUTE + k));
        columns.open.push(open);
        columns.high.push(open + 0.50);
        columns.low.push(open - 0.50);
        columns.close.push(open + 0.05);
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

fn config(present: Option<i64>) -> RunConfig {
    RunConfig {
        first_day: day(),
        last_day: day(),
        first_slice: None,
        last_slice: None,
        present,
        base: BaseInterval::Intraday(MINUTE),
        strategy_interval: StrategyInterval::Intraday(15 * MINUTE),
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

fn engine(bars: BarColumns, present: Option<i64>) -> Result<Engine, KernelError> {
    let calendar = Calendar::new().expect("the embedded calendar tables parse");
    Engine::new(RunSetup {
        config: config(present),
        calendar,
        instruments: vec![sbin()],
        bars: vec![bars],
    })
}

/// Buys 100 MIS units at the first strategy-bar close, then holds.
struct BuyAndHold {
    bought: bool,
}

impl Driver for BuyAndHold {
    fn on_pause(&mut self, engine: &mut Engine, pause: &Pause) -> Result<(), KernelError> {
        if self.bought || !matches!(pause, Pause::BarClose { .. }) {
            return Ok(());
        }
        self.bought = true;
        let request = OrderRequest {
            product: Product::Mis,
            ..OrderRequest::market(SBIN, Side::Buy, QuantitySpec::Units(100.0))
        };
        engine.submit(request)?;
        Ok(())
    }
}

fn run_to_end(bars: BarColumns, present: Option<i64>) -> RunOutput {
    let mut kernel = engine(bars, present).expect("a valid setup");
    run(&mut kernel, &mut BuyAndHold { bought: false }).expect("the run completes");
    kernel.finish().expect("the run finished")
}

#[test]
fn bars_that_stop_at_noon_end_the_run_at_the_close_of_the_last_bar() {
    // 165 bars: the last opens at 11:59 and closes at 12:00.
    let output = run_to_end(bars(165, &[]), None);
    assert_eq!(output.equity.ts.last(), Some(&at(12, 0)));
    assert_eq!(
        output.run_end,
        RunEnd {
            close: at(12, 0),
            range_close: at(15, 30),
            cause: Some(RunEndCause::LastBar),
        }
    );
}

#[test]
fn a_position_held_at_the_end_of_the_run_is_reported_open_not_squared_off() {
    let output = run_to_end(bars(165, &[]), None);
    // Only the entry: no eod exit at 15:29 at the 12:00 price.
    assert_eq!(output.fills.len(), 1);
    assert!(output.trades.is_empty());
    let held = output.final_positions[0];
    assert_eq!(held.quantity_units, 100.0);
    // The 11:59 bar (k = 164) closes at 500.00 + 16.40 + 0.05.
    let mark = held.mark_rupees.unwrap_or(f64::NAN);
    assert!((mark - 516.45).abs() < 1e-9, "marked at {mark}");
}

#[test]
fn the_present_ends_the_run_at_the_last_slice_closed_by_then() {
    let present = at(11, 0) + 30 * SECOND;
    let output = run_to_end(bars(SESSION_MINUTES, &[]), Some(present));
    assert_eq!(output.equity.ts.last(), Some(&at(11, 0)));
    assert_eq!(output.run_end.cause, Some(RunEndCause::Present));
}

#[test]
fn bars_that_reach_the_end_of_the_range_end_the_run_there_with_no_cause() {
    let output = run_to_end(bars(SESSION_MINUTES, &[]), Some(at(18, 0)));
    assert_eq!(
        output.run_end,
        RunEnd {
            close: at(15, 30),
            range_close: at(15, 30),
            cause: None,
        }
    );
}

/// Monday 2025-06-30, the NSE session before the test day.
fn monday() -> Day {
    days_from_civil(2025, 6, 30)
}

/// Base bars from 09:15 on Monday 2025-06-30 for `monday_count` minutes, then on the
/// test day for `tuesday_count` minutes.
fn two_sessions(monday_count: u32, tuesday_count: u32) -> BarColumns {
    let mut columns = BarColumns::default();
    for (session, count) in [(monday(), monday_count), (day(), tuesday_count)] {
        for k in 0..count {
            let open = 500.0 + 0.10 * f64::from(k);
            columns
                .ts_open
                .push(ts_from_ist(session, SESSION_OPEN_MINUTE + k));
            columns.open.push(open);
            columns.high.push(open + 0.50);
            columns.low.push(open - 0.50);
            columns.close.push(open + 0.05);
            columns.volume.push(1_000.0);
        }
    }
    columns
}

/// A run over Monday and the test day.
fn run_two_sessions(bars: BarColumns, present: Option<i64>) -> RunOutput {
    let calendar = Calendar::new().expect("the embedded calendar tables parse");
    let mut kernel = Engine::new(RunSetup {
        config: RunConfig {
            first_day: monday(),
            ..config(present)
        },
        calendar,
        instruments: vec![sbin()],
        bars: vec![bars],
    })
    .expect("a valid setup");
    run(&mut kernel, &mut BuyAndHold { bought: false }).expect("the run completes");
    kernel.finish().expect("the run finished")
}

/// Bars from 09:15 to 15:14 IST: the last opens at 15:14 and closes at 15:15, as a
/// stock with a closing auction trades continuously only until 15:15.
const UNTIL_THE_CLOSING_AUCTION: u32 = 360;

#[test]
fn sessions_whose_bars_all_stop_at_the_same_time_end_the_run_at_the_end_of_its_range() {
    let bars = two_sessions(UNTIL_THE_CLOSING_AUCTION, UNTIL_THE_CLOSING_AUCTION);
    let output = run_two_sessions(bars, None);
    assert_eq!(output.equity.ts.last(), Some(&at(15, 30)));
    assert_eq!(
        output.run_end,
        RunEnd {
            close: at(15, 30),
            range_close: at(15, 30),
            cause: None,
        }
    );
}

#[test]
fn a_last_session_that_stops_earlier_than_the_one_before_ends_the_run_at_its_last_bar() {
    // Monday is complete; on the test day the last bar opens at 11:59 and closes at 12:00.
    let output = run_two_sessions(two_sessions(SESSION_MINUTES, 165), None);
    assert_eq!(
        output.run_end,
        RunEnd {
            close: at(12, 0),
            range_close: at(15, 30),
            cause: Some(RunEndCause::LastBar),
        }
    );
}

#[test]
fn the_present_after_the_usual_last_bar_ends_the_run_at_the_present_not_at_the_bar() {
    let bars = two_sessions(UNTIL_THE_CLOSING_AUCTION, UNTIL_THE_CLOSING_AUCTION);
    let output = run_two_sessions(bars, Some(at(15, 20) + 30 * SECOND));
    assert_eq!(output.run_end.close, at(15, 20));
    assert_eq!(output.run_end.cause, Some(RunEndCause::Present));
}

#[test]
fn a_missing_bar_inside_the_run_is_still_a_slice() {
    let output = run_to_end(bars(165, &[100]), None);
    assert_eq!(output.equity.ts.len(), 165);
}

#[test]
fn a_range_with_no_bar_is_refused_naming_the_data() {
    let refused = engine(BarColumns::default(), None).err();
    assert!(matches!(
        refused,
        Some(KernelError::InvalidConfig { field: "data", .. })
    ));
}

#[test]
fn a_present_before_the_first_slice_closes_is_refused_naming_now() {
    let refused = engine(bars(SESSION_MINUTES, &[]), Some(at(9, 15))).err();
    assert!(matches!(
        refused,
        Some(KernelError::InvalidConfig { field: "now", .. })
    ));
}
