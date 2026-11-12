#![cfg(all(feature = "costs", feature = "time"))]
//! Leg rules and `reverse=True` through the whole slice sequence: what the engine adds
//! around the leg state (the exit fill, its reason and target index, slippage, the
//! events, the order states) and what it refuses.
//!
//! One NSE session on 2025-07-01, 1-minute base bars from 09:15, a 5-minute strategy,
//! MIS. Bars are flat at 500.00 unless a test overrides a minute, so every level and
//! fill below can be worked out from the overrides alone.

use oa_bt::engine::{
    run, BarColumns, CostConfig, Driver, Engine, Event, FillTiming, IntradayOverrides, KernelError,
    Pause, QuantityMode, RunConfig, RunOutput, RunSetup, Step, StrategyInterval, Submission,
};
use oa_bt::exec::{
    FillAt, InstrumentId, InstrumentSpec, OrderRequest, OrderStatus, QuantitySpec, RejectCode,
};
use oa_bt::market::calendar::{BaseInterval, Calendar};
use oa_bt::market::margin::MarginModel;
use oa_bt::market::slippage::SlippageSpec;
use oa_bt::model::{days_from_civil, ts_from_ist, Day, Exchange, InstrumentKind, Product, Side};
use oa_bt::rms::{Distance, LegRules, RmsError, StrategyRiskConfig, TargetSpec, TrailSpec};

const MINUTE: i64 = 60_000_000_000;
const SESSION_OPEN_MINUTE: u32 = 9 * 60 + 15;
const CAPITAL: f64 = 10_000_000.0;
const MINUTES: u32 = 60;
const FLAT: f64 = 500.0;

fn day() -> Day {
    days_from_civil(2025, 7, 1)
}

/// IST clock time on the test day, as UTC nanoseconds.
fn at(hour: u32, minute: u32) -> i64 {
    ts_from_ist(day(), hour * 60 + minute)
}

/// (minute from 09:15, open, high, low, close).
type Row = (u32, f64, f64, f64, f64);

fn bars(overrides: &[Row]) -> BarColumns {
    let mut columns = BarColumns::default();
    for k in 0..MINUTES {
        let (open, high, low, close) = overrides
            .iter()
            .find(|row| row.0 == k)
            .map_or((FLAT, FLAT, FLAT, FLAT), |row| (row.1, row.2, row.3, row.4));
        columns
            .ts_open
            .push(ts_from_ist(day(), SESSION_OPEN_MINUTE + k));
        columns.open.push(open);
        columns.high.push(high);
        columns.low.push(low);
        columns.close.push(close);
        columns.volume.push(1_000.0);
    }
    columns
}

fn equity(symbol: &str) -> InstrumentSpec {
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

fn engine(instruments: Vec<(&str, BarColumns)>, slippage: SlippageSpec) -> Engine {
    let config = RunConfig {
        first_day: day(),
        last_day: day(),
        first_slice: None,
        last_slice: Some(at(10, 14)),
        base: BaseInterval::Intraday(MINUTE),
        strategy_interval: StrategyInterval::Intraday(5 * MINUTE),
        fill: FillTiming::NextOpen,
        quantity_mode: QuantityMode::Venue,
        capital_rupees: CAPITAL,
        costs: CostConfig::Off,
        slippage,
        margin: MarginModel,
        risk: StrategyRiskConfig::default(),
        intraday: IntradayOverrides::default(),
        deliver_events: false,
    };
    let (specs, columns): (Vec<_>, Vec<_>) = instruments
        .into_iter()
        .map(|(symbol, bars)| (equity(symbol), bars))
        .unzip();
    let setup = RunSetup {
        config,
        calendar: Calendar::new().expect("the embedded calendar tables parse"),
        instruments: specs,
        bars: columns,
    };
    Engine::new(setup).expect("a valid setup")
}

/// Orders to place at given strategy-bar opens, per symbol.
struct Script {
    orders: Vec<(i64, &'static str, OrderRequest)>,
    submissions: Vec<Submission>,
}

impl Driver for Script {
    fn on_pause(&mut self, engine: &mut Engine, pause: &Pause) -> Result<(), KernelError> {
        let Pause::BarClose { bars, .. } = pause else {
            return Ok(());
        };
        for bar in bars {
            let symbol = engine.instruments()[bar.instrument.index()].symbol.clone();
            let due: Vec<OrderRequest> = self
                .orders
                .iter()
                .filter(|(open, name, _)| *open == bar.ts_open && *name == symbol)
                .map(|(_, _, request)| OrderRequest {
                    instrument: bar.instrument,
                    ..request.clone()
                })
                .collect();
            for request in due {
                self.submissions.push(engine.submit(request)?);
            }
        }
        Ok(())
    }
}

fn order(side: Side, units: f64, rules: LegRules) -> OrderRequest {
    OrderRequest {
        product: Product::Mis,
        leg_rules: rules,
        ..OrderRequest::market(InstrumentId(0), side, QuantitySpec::Units(units))
    }
}

fn run_script(
    mut kernel: Engine,
    orders: Vec<(i64, &'static str, OrderRequest)>,
) -> Result<(RunOutput, Vec<Submission>), KernelError> {
    let mut script = Script {
        orders,
        submissions: Vec::new(),
    };
    run(&mut kernel, &mut script)?;
    Ok((kernel.finish()?, script.submissions))
}

fn stop_and_target(stop: f64, target: f64) -> LegRules {
    LegRules {
        stop: Some(Distance::Points(stop)),
        target: Some(Distance::Points(target)),
        ..LegRules::default()
    }
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
        })
        .collect()
}

#[test]
fn a_target_exit_fills_at_its_level_less_slippage_with_its_reason_and_index() {
    // One tick of slippage. Entry at the 09:20 open: BUY at 500.00 + 0.05 = 500.05;
    // stop 495.05, target 503.05. The 09:21 bar reaches 503.50: SELL at 503.05 - 0.05.
    let kernel = engine(
        vec![("SBIN", bars(&[(6, 501.0, 503.5, 500.5, 503.0)]))],
        SlippageSpec::Ticks(1),
    );
    let orders = vec![(
        at(9, 15),
        "SBIN",
        order(Side::Buy, 100.0, stop_and_target(5.0, 3.0)),
    )];
    let (output, _) = run_script(kernel, orders).expect("the run completes");
    let fills: Vec<_> = output
        .fills
        .iter()
        .map(|r| {
            (
                r.stamp.slice_ts,
                r.stamp.step,
                r.fill.price_rupees,
                r.fill.at,
            )
        })
        .collect();
    assert_eq!(
        fills,
        vec![
            (at(9, 20), Step::OpenFills, 500.05, FillAt::Open),
            (at(9, 21), Step::Levels, 503.0, FillAt::Level),
        ]
    );
    let exit = output.fills[1].fill;
    assert_eq!(
        (exit.reason.code(), exit.target_index),
        ("leg_target", Some(0))
    );
    assert_eq!(exit.order, None);
}

#[test]
fn a_level_exit_emits_its_risk_event_before_the_position_event() {
    let kernel = engine(
        vec![("SBIN", bars(&[(6, 501.0, 503.5, 500.5, 503.0)]))],
        SlippageSpec::Zero,
    );
    let orders = vec![(
        at(9, 15),
        "SBIN",
        order(Side::Buy, 100.0, stop_and_target(5.0, 3.0)),
    )];
    let (output, _) = run_script(kernel, orders).expect("the run completes");
    let at_exit: Vec<_> = event_names(&output)
        .into_iter()
        .filter(|(slice, _, _)| *slice == at(9, 21))
        .collect();
    assert_eq!(
        at_exit,
        vec![
            (at(9, 21), Step::Levels, "leg_target_hit"),
            (at(9, 21), Step::Levels, "PositionClosed"),
        ]
    );
}

#[test]
fn reverse_closes_the_position_and_opens_the_rest_as_two_fills_of_one_order() {
    // Long 100 from the 09:20 open; at the 09:20 bar's close sell 150 with reverse=True:
    // at the 09:25 open SELL 100 closes (reason signal), SELL 50 opens a short (entry).
    let kernel = engine(vec![("SBIN", bars(&[]))], SlippageSpec::Zero);
    let reverse = OrderRequest {
        reverse: true,
        ..order(Side::Sell, 150.0, LegRules::default())
    };
    let orders = vec![
        (
            at(9, 15),
            "SBIN",
            order(Side::Buy, 100.0, LegRules::default()),
        ),
        (at(9, 20), "SBIN", reverse),
    ];
    let (output, _) = run_script(kernel, orders).expect("the run completes");
    let fills: Vec<_> = output
        .fills
        .iter()
        .map(|r| {
            (
                r.stamp.slice_ts,
                r.fill.side,
                r.fill.quantity_units,
                r.fill.reason.code(),
            )
        })
        .collect();
    assert_eq!(
        fills,
        vec![
            (at(9, 20), Side::Buy, 100.0, "entry"),
            (at(9, 25), Side::Sell, 100.0, "signal"),
            (at(9, 25), Side::Sell, 50.0, "entry"),
        ]
    );
    let second = &output.orders[1];
    assert_eq!(
        (second.status, second.filled_units),
        (OrderStatus::Filled, 150.0)
    );
    let at_reversal: Vec<_> = event_names(&output)
        .into_iter()
        .filter(|(slice, _, _)| *slice == at(9, 25))
        .map(|(_, _, name)| name)
        .collect();
    assert_eq!(
        at_reversal,
        vec!["Filled", "PositionClosed", "Filled", "PositionOpened"]
    );
    assert_eq!(output.final_positions[0].quantity_units, -50.0);
}

#[test]
fn an_order_beyond_the_position_without_reverse_is_rejected() {
    let kernel = engine(vec![("SBIN", bars(&[]))], SlippageSpec::Zero);
    let orders = vec![
        (
            at(9, 15),
            "SBIN",
            order(Side::Buy, 100.0, LegRules::default()),
        ),
        (
            at(9, 20),
            "SBIN",
            order(Side::Sell, 150.0, LegRules::default()),
        ),
    ];
    let (output, submissions) = run_script(kernel, orders).expect("the run completes");
    assert!(matches!(
        submissions[..],
        [
            Submission::Submitted(_),
            Submission::Rejected(_, RejectCode::ReverseNotAllowed)
        ]
    ));
    assert_eq!(output.fills.len(), 1);
    assert_eq!(output.final_positions[0].quantity_units, 100.0);
}

#[test]
fn an_add_that_changes_the_leg_rules_is_a_configuration_error() {
    let kernel = engine(vec![("SBIN", bars(&[]))], SlippageSpec::Zero);
    let orders = vec![
        (
            at(9, 15),
            "SBIN",
            order(Side::Buy, 100.0, stop_and_target(5.0, 3.0)),
        ),
        (
            at(9, 20),
            "SBIN",
            order(Side::Buy, 50.0, stop_and_target(6.0, 3.0)),
        ),
    ];
    let error = run_script(kernel, orders).map(|_| ());
    assert!(
        matches!(error, Err(KernelError::Rms(RmsError::InvalidRule { .. }))),
        "{error:?}"
    );
}

#[test]
fn an_add_that_repeats_the_leg_rules_is_accepted() {
    let kernel = engine(vec![("SBIN", bars(&[]))], SlippageSpec::Zero);
    let orders = vec![
        (
            at(9, 15),
            "SBIN",
            order(Side::Buy, 100.0, stop_and_target(5.0, 3.0)),
        ),
        (
            at(9, 20),
            "SBIN",
            order(Side::Buy, 50.0, stop_and_target(5.0, 3.0)),
        ),
    ];
    let (output, _) = run_script(kernel, orders).expect("the run completes");
    assert_eq!(output.final_positions[0].quantity_units, 150.0);
}

#[test]
fn leg_rules_on_an_order_that_only_reduces_are_a_configuration_error() {
    let kernel = engine(vec![("SBIN", bars(&[]))], SlippageSpec::Zero);
    let orders = vec![
        (
            at(9, 15),
            "SBIN",
            order(Side::Buy, 100.0, LegRules::default()),
        ),
        (
            at(9, 20),
            "SBIN",
            order(Side::Sell, 50.0, stop_and_target(5.0, 3.0)),
        ),
    ];
    let error = run_script(kernel, orders).map(|_| ());
    assert!(
        matches!(error, Err(KernelError::Rms(RmsError::InvalidRule { .. }))),
        "{error:?}"
    );
}

#[test]
fn a_reversing_order_gives_its_leg_rules_to_the_position_it_opens() {
    // Long 100; sell 150 with reverse=True and a 5-point stop: the short of 50 opened
    // at 500.00 has its stop at 505.00.
    let kernel = engine(vec![("SBIN", bars(&[]))], SlippageSpec::Zero);
    let reverse = OrderRequest {
        reverse: true,
        ..order(Side::Sell, 150.0, stop_and_target(5.0, 3.0))
    };
    let orders = vec![
        (
            at(9, 15),
            "SBIN",
            order(Side::Buy, 100.0, LegRules::default()),
        ),
        (at(9, 20), "SBIN", reverse),
    ];
    let (output, _) = run_script(kernel, orders).expect("the run completes");
    let short = output.levels.last().and_then(|row| row.stop);
    assert_eq!(
        short.map(|stop| (stop.level_rupees, stop.quantity_units)),
        Some((505.0, 50.0))
    );
}

/// A wandering path of whole ticks from a seed, so the two instruments differ.
fn wandering(seed: u64) -> Vec<Row> {
    let mut state = seed;
    let mut step = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state % 41) as f64 * 0.05 - 1.0
    };
    let mut close = FLAT;
    (0..MINUTES)
        .map(|k| {
            let open = close + step();
            let (a, b) = (open + step(), open + step());
            let high = open.max(a).max(b);
            let low = open.min(a).min(b);
            close = if k % 2 == 0 { high } else { low };
            (k, open, high, low, close)
        })
        .collect()
}

fn busy_rules() -> LegRules {
    LegRules {
        stop: Some(Distance::Points(2.0)),
        targets: vec![
            TargetSpec {
                distance: Distance::Points(1.0),
                fraction: 0.5,
            },
            TargetSpec {
                distance: Distance::Points(2.5),
                fraction: 0.3,
            },
        ],
        trail: Some(TrailSpec {
            trigger: Distance::Points(1.5),
            step: None,
        }),
        ..LegRules::default()
    }
}

/// Per symbol: every fill and every position-history row, without instrument ids.
fn per_symbol(output: &RunOutput, symbol: &str) -> (Vec<String>, Vec<String>) {
    let id = output
        .instruments
        .iter()
        .position(|spec| spec.symbol == symbol)
        .map(|index| InstrumentId(index as u32));
    let fills = output
        .fills
        .iter()
        .filter(|r| Some(r.fill.instrument) == id)
        .map(|r| {
            let f = r.fill;
            let order = f.order.is_some();
            let when = (r.stamp.slice_ts, r.stamp.step);
            let what = (
                f.side,
                f.quantity_units,
                f.price_rupees,
                f.at,
                f.reason,
                f.target_index,
            );
            format!("{when:?} {what:?} order {order}")
        })
        .collect();
    let levels = output
        .levels
        .iter()
        .filter(|row| Some(row.instrument) == id)
        .map(|row| {
            let state = (row.quantity_units, row.stop, &row.targets, row.trail);
            format!("{} {state:?} {}", row.slice_ts, row.remainder_units)
        })
        .collect();
    (fills, levels)
}

#[test]
fn results_do_not_depend_on_the_registration_order_of_the_legs() {
    for seed in [3_u64, 11, 29, 47, 83] {
        let (a, b) = (bars(&wandering(seed)), bars(&wandering(seed + 1_000)));
        let entries = || {
            vec![
                (at(9, 15), "SBIN", order(Side::Buy, 100.0, busy_rules())),
                (at(9, 15), "TCS", order(Side::Sell, 60.0, busy_rules())),
            ]
        };
        let forward = engine(
            vec![("SBIN", a.clone()), ("TCS", b.clone())],
            SlippageSpec::Zero,
        );
        let backward = engine(vec![("TCS", b), ("SBIN", a)], SlippageSpec::Zero);
        let (one, _) = run_script(forward, entries()).expect("the forward run completes");
        let (two, _) = run_script(backward, entries()).expect("the backward run completes");
        for symbol in ["SBIN", "TCS"] {
            assert_eq!(
                per_symbol(&one, symbol),
                per_symbol(&two, symbol),
                "seed {seed}"
            );
        }
        assert!(
            one.fills.len() > 2,
            "seed {seed}: the legs exit at their levels"
        );
    }
}
