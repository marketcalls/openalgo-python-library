#![cfg(all(feature = "costs", feature = "time"))]
//! The walking skeleton of the kernel, end to end through the slice sequence: one
//! instrument, 1-minute base bars, a 15-minute strategy that buys at one bar close and
//! exits at a later close, fills at the next base-bar open, no costs and no slippage.
//!
//! Bars are written so that every expected number can be read off the inputs: the bar
//! opening k minutes after 09:15 IST opens at 500.00 + 0.10 k, its high is the open plus
//! 0.50, its low the open minus 0.50 and its close the open plus 0.05.

use oa_bt::engine::{
    run, BarColumns, CostConfig, Driver, Engine, Event, FillTiming, IntradayOverrides, KernelError,
    Pause, QuantityMode, RunConfig, RunOutput, RunSetup, Step, StrategyInterval, Submission,
};
use oa_bt::exec::{InstrumentId, InstrumentSpec, OrderRequest, QuantitySpec, RejectCode};
use oa_bt::market::calendar::{BaseInterval, Calendar};
use oa_bt::market::facts::MarketFacts;
use oa_bt::market::margin::MarginModel;
use oa_bt::market::slippage::SlippageSpec;
use oa_bt::model::{days_from_civil, ts_from_ist, Day, Exchange, InstrumentKind, Product, Side};
use oa_bt::portfolio::PositionChange;
use oa_bt::rms::{Distance, LegRules, StrategyRiskConfig};

const MINUTE: i64 = 60_000_000_000;
const SESSION_OPEN_MINUTE: u32 = 9 * 60 + 15;
const CAPITAL: f64 = 1_000_000.0;
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

fn config(last_slice: Option<i64>) -> RunConfig {
    RunConfig {
        first_day: day(),
        last_day: day(),
        first_slice: None,
        last_slice,
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
        intraday: IntradayOverrides::default(),
        deliver_events: false,
    }
}

fn engine(bars: BarColumns, last_slice: Option<i64>) -> Engine {
    let calendar = Calendar::new().expect("the embedded calendar tables parse");
    let setup = RunSetup {
        config: config(last_slice),
        calendar,
        instruments: vec![sbin()],
        bars: vec![bars],
    };
    Engine::new(setup).expect("a valid setup")
}

/// What the script does at a strategy-bar close.
#[derive(Clone)]
enum Action {
    Buy(f64, Product),
    Exit(Product),
    Order(Box<OrderRequest>),
}

/// A driver that acts at given strategy-bar opens and keeps what `submit` returned.
struct Script {
    actions: Vec<(i64, Action)>,
    submissions: Vec<Submission>,
}

impl Script {
    fn new(actions: Vec<(i64, Action)>) -> Script {
        Script {
            actions,
            submissions: Vec::new(),
        }
    }
}

fn exit_request(product: Product, held: f64) -> OrderRequest {
    let side = if held > 0.0 { Side::Sell } else { Side::Buy };
    OrderRequest {
        reduce_only: true,
        product,
        ..OrderRequest::market(SBIN, side, QuantitySpec::PositionFraction(1.0))
    }
}

impl Driver for Script {
    fn on_pause(&mut self, engine: &mut Engine, pause: &Pause) -> Result<(), KernelError> {
        let Pause::BarClose { bars, .. } = pause else {
            return Ok(());
        };
        for bar in bars {
            let due: Vec<Action> = self
                .actions
                .iter()
                .filter(|(open, _)| *open == bar.ts_open)
                .map(|(_, action)| action.clone())
                .collect();
            for action in due {
                let request = match action {
                    Action::Buy(units, product) => OrderRequest {
                        product,
                        ..OrderRequest::market(SBIN, Side::Buy, QuantitySpec::Units(units))
                    },
                    Action::Exit(product) => {
                        let held = engine
                            .position(SBIN, product)
                            .map_or(0.0, |p| p.quantity_units);
                        exit_request(product, held)
                    }
                    Action::Order(request) => *request,
                };
                self.submissions.push(engine.submit(request)?);
            }
        }
        Ok(())
    }
}

fn run_script(
    bars: BarColumns,
    last_slice: Option<i64>,
    actions: Vec<(i64, Action)>,
) -> (RunOutput, Vec<Submission>) {
    let mut kernel = engine(bars, last_slice);
    let mut script = Script::new(actions);
    run(&mut kernel, &mut script).expect("the run completes");
    (
        kernel.finish().expect("the run finished"),
        script.submissions,
    )
}

fn buy_then_exit() -> Vec<(i64, Action)> {
    vec![
        (at(9, 15), Action::Buy(100.0, Product::Cnc)),
        (at(9, 30), Action::Exit(Product::Cnc)),
    ]
}

#[test]
fn a_buy_and_an_exit_fill_at_the_next_base_bar_open() {
    let (output, _) = run_script(bars(45, &[]), Some(at(9, 59)), buy_then_exit());
    let fills: Vec<(i64, Step, Side, f64, f64, &str)> = output
        .fills
        .iter()
        .map(|r| {
            let f = r.fill;
            (
                r.stamp.slice_ts,
                r.stamp.step,
                f.side,
                f.quantity_units,
                f.price_rupees,
                f.reason.code(),
            )
        })
        .collect();
    assert_eq!(
        fills,
        vec![
            (at(9, 30), Step::OpenFills, Side::Buy, 100.0, 501.5, "entry"),
            (
                at(9, 45),
                Step::OpenFills,
                Side::Sell,
                100.0,
                503.0,
                "signal"
            ),
        ]
    );
}

#[test]
fn the_gross_pnl_is_the_difference_of_the_two_opens() {
    let (output, _) = run_script(bars(45, &[]), Some(at(9, 59)), buy_then_exit());
    // (503.00 - 501.50) x 100
    assert!((output.gross_pnl.realized_rupees - 150.0).abs() < 1e-9);
    assert_eq!(output.gross_pnl.unrealized_rupees, 0.0);
    let last_equity = output.equity.equity_rupees.last().copied();
    assert!(last_equity.is_some_and(|equity| (equity - (CAPITAL + 150.0)).abs() < 1e-6));
    assert_eq!(output.equity.ts.len(), 45);
}

#[test]
fn the_position_history_has_a_row_for_each_change() {
    let (output, _) = run_script(bars(45, &[]), Some(at(9, 59)), buy_then_exit());
    let rows: Vec<(i64, f64, Option<f64>)> = output
        .levels
        .iter()
        .map(|row| (row.slice_ts, row.quantity_units, row.average_price_rupees))
        .collect();
    assert_eq!(
        rows,
        vec![(at(9, 30), 100.0, Some(501.5)), (at(9, 45), 0.0, None)]
    );
}

#[test]
fn the_round_trip_is_one_closed_trade() {
    let (output, _) = run_script(bars(45, &[]), Some(at(9, 59)), buy_then_exit());
    assert_eq!(output.trades.len(), 1);
    let trade = output.trades[0];
    assert_eq!((trade.entry_ts, trade.exit_ts), (at(9, 30), at(9, 45)));
    assert_eq!(
        (trade.entry_price_rupees, trade.exit_price_rupees),
        (501.5, 503.0)
    );
    assert_eq!(trade.direction, Side::Buy);
    assert!((trade.net_pnl_rupees - 150.0).abs() < 1e-9);
}

/// (slice open, step, name) of every event.
fn event_names(output: &RunOutput) -> Vec<(i64, Step, &'static str)> {
    output
        .events
        .iter()
        .map(|event| match event {
            Event::Order { stamp, kind, .. } => (stamp.slice_ts, stamp.step, kind.code()),
            Event::Position { stamp, change, .. } => (stamp.slice_ts, stamp.step, change.code()),
            Event::Risk { stamp, event, .. } => (stamp.slice_ts, stamp.step, event.name.code()),
            Event::TradingState { stamp, to, .. } => (stamp.slice_ts, stamp.step, to.code()),
        })
        .collect()
}

#[test]
fn the_event_stream_follows_the_fixture_vocabulary() {
    let (output, _) = run_script(bars(45, &[]), Some(at(9, 59)), buy_then_exit());
    assert_eq!(
        event_names(&output),
        vec![
            (at(9, 29), Step::Callbacks, "Submitted"),
            (at(9, 30), Step::OpenFills, "Filled"),
            (at(9, 30), Step::OpenFills, PositionChange::Opened.code()),
            (at(9, 44), Step::Callbacks, "Submitted"),
            (at(9, 45), Step::OpenFills, "Filled"),
            (at(9, 45), Step::OpenFills, PositionChange::Closed.code()),
        ]
    );
}

#[test]
fn an_order_whose_fill_bar_is_missing_fills_at_the_next_bar() {
    // SPEC-06: the 09:30 bar (k = 15) is missing; the order fills at the 09:31 open.
    let actions = vec![(at(9, 15), Action::Buy(100.0, Product::Cnc))];
    let (output, _) = run_script(bars(21, &[15]), Some(at(9, 35)), actions);
    assert_eq!(output.fills.len(), 1);
    assert_eq!(output.fills[0].stamp.slice_ts, at(9, 31));
    assert_eq!(output.fills[0].fill.price_rupees, 501.6);
    // Open at the end: marked at the 09:35 close, 502.05, and not a trade.
    assert!((output.gross_pnl.unrealized_rupees - 45.0).abs() < 1e-9);
    assert!(output.trades.is_empty());
}

#[test]
fn an_open_position_without_a_bar_is_marked_stale() {
    let actions = vec![(at(9, 15), Action::Buy(100.0, Product::Cnc))];
    let (output, _) = run_script(bars(21, &[17]), Some(at(9, 35)), actions);
    // The 09:32 bar is missing: the 09:31 close, 501.65, carries over.
    assert_eq!(output.stale_marks.len(), 1);
    let mark = output.stale_marks[0];
    assert_eq!(mark.slice_ts, at(9, 32));
    assert!((mark.mark_rupees - 501.65).abs() < 1e-9);
}

#[test]
fn an_intraday_position_is_squared_off_at_the_square_off_open() {
    let actions = vec![(at(9, 15), Action::Buy(100.0, Product::Mis))];
    let (output, _) = run_script(bars(375, &[]), None, actions);
    let last = output.fills.last().expect("two fills");
    // The default square-off on 2025-07-01 is 15:10: k = 355, open 535.50.
    assert_eq!(last.stamp.slice_ts, at(15, 10));
    assert_eq!(last.fill.reason.code(), "scheduler");
    assert_eq!(last.fill.price_rupees, 535.5);
}

#[test]
fn a_last_bar_signal_is_rejected_as_market_closed() {
    // The 15:15 strategy bar closes with the session: its fill would be the next day.
    let actions = vec![(at(15, 15), Action::Buy(10.0, Product::Cnc))];
    let (output, submissions) = run_script(bars(375, &[]), None, actions);
    assert!(matches!(
        submissions[..],
        [Submission::Rejected(_, RejectCode::MarketClosed)]
    ));
    assert!(output.fills.is_empty());
}

#[test]
fn an_intraday_entry_after_entry_until_is_rejected() {
    let actions = vec![(at(14, 45), Action::Buy(10.0, Product::Mis))];
    let (_, submissions) = run_script(bars(375, &[]), None, actions);
    assert!(matches!(
        submissions[..],
        [Submission::Rejected(_, RejectCode::EntryWindow)]
    ));
}

#[test]
fn leg_rules_on_a_delivery_position_are_refused_as_not_built() {
    // On CNC and NRML they are GTT orders (04-execution 4.4); MIS legs are engine-held.
    let rules = LegRules {
        stop: Some(Distance::Points(4.0)),
        ..LegRules::default()
    };
    let request = OrderRequest {
        leg_rules: rules,
        ..OrderRequest::market(SBIN, Side::Buy, QuantitySpec::Units(100.0))
    };
    let mut kernel = engine(bars(45, &[]), Some(at(9, 59)));
    let mut script = Script::new(vec![(at(9, 15), Action::Order(Box::new(request)))]);
    let error = run(&mut kernel, &mut script).expect_err("GTT-held stops are not built yet");
    assert!(error.not_built_feature().is_some(), "{error}");
}

#[test]
fn the_same_inputs_give_the_same_result() {
    let first = run_script(bars(375, &[40, 41]), None, buy_then_exit());
    let second = run_script(bars(375, &[40, 41]), None, buy_then_exit());
    assert_eq!(first.0, second.0);
}

#[test]
fn orders_are_refused_outside_a_decision_point() {
    let mut kernel = engine(bars(45, &[]), Some(at(9, 59)));
    let request = OrderRequest::market(SBIN, Side::Buy, QuantitySpec::Units(1.0));
    assert_eq!(kernel.submit(request), Err(KernelError::NotAtDecisionPoint));
}
