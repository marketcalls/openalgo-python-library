#![cfg(all(feature = "costs", feature = "time"))]
//! Condition strategies through the Rust signal driver (14-strategy-api 4.1, SPEC-29):
//! `run_signals` makes, at every strategy-bar close, the decision of the equivalent
//! `on_bar`, so its run equals the run of that `on_bar` written out as a script driver.
//!
//! Bars are 15-minute MIS bars on the embedded NSE calendar, two sessions (2025-07-01
//! and 2025-07-02), with no charges and no slippage unless a test says otherwise.

mod risk_support;

use oa_bt::engine::{
    run, run_signals, CostConfig, Driver, Engine, EntrySignal, KernelError, Pause, RunConfig,
    RunOutput, SignalArrays, SkippedEntry, StrategyBar,
};
use oa_bt::exec::{InstrumentId, OrderRequest, QuantitySpec};
use oa_bt::market::costs::CostModel;
use oa_bt::market::slippage::SlippageSpec;
use oa_bt::model::{Product, Side};
use oa_bt::rms::{Distance, LegRules, StrategyRiskConfig, TargetSpec};
use oa_bt::sizing::{SizerSpec, SkipReason};
use risk_support::*;

const SBIN: InstrumentId = InstrumentId(0);
const SESSIONS: usize = 2;
const BARS: usize = SESSIONS * BARS_15M_PER_SESSION;

fn days() -> [oa_bt::model::Day; SESSIONS] {
    [july_2025(1), july_2025(2)]
}

/// A small deterministic generator for the property runs (test code only; the kernel
/// itself has no randomness).
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0 >> 33
    }

    /// True with probability `percent` / 100.
    fn chance(&mut self, percent: u64) -> bool {
        self.next() % 100 < percent
    }
}

/// Bars that walk by whole ticks: each bar opens at the last close and moves up to
/// 2.00 rupees either way.
fn walk(rng: &mut Lcg) -> oa_bt::engine::BarColumns {
    let mut closes = vec![500.0];
    for _ in 0..BARS {
        let ticks = (rng.next() % 81) as f64 - 40.0;
        let last = closes[closes.len() - 1];
        closes.push(((last + ticks * 0.05) * 20.0).round() / 20.0);
    }
    sessions_15m(&days(), |session, k| {
        let bar = session * BARS_15M_PER_SESSION + k;
        Some((closes[bar], closes[bar + 1]))
    })
}

fn engine_for(bars: oa_bt::engine::BarColumns) -> Engine {
    engine_with(bars, CostConfig::Off, SlippageSpec::Zero)
}

fn engine_with(
    bars: oa_bt::engine::BarColumns,
    costs: CostConfig,
    slippage: SlippageSpec,
) -> Engine {
    let days = days();
    let config = RunConfig {
        costs,
        slippage,
        ..intraday_config((days[0], days[1]), StrategyRiskConfig::default())
    };
    engine(config, vec![equity("SBIN")], vec![bars]).expect("a valid setup")
}

/// The built-in charges of the default broker and the dated default slippage table:
/// what `bt.backtest` uses when a run names neither.
fn charges_and_slippage() -> (CostConfig, SlippageSpec) {
    let model = CostModel::builtin().expect("the built-in schedules parse");
    (CostConfig::Model(model), SlippageSpec::SegmentDefault)
}

fn bar_opens() -> Vec<i64> {
    days()
        .iter()
        .flat_map(|&day| (0..BARS_15M_PER_SESSION).map(move |k| bar_open(day, k)))
        .collect()
}

fn entry_order(side: Side, units: f64, rules: LegRules) -> OrderRequest {
    OrderRequest {
        product: Product::Mis,
        leg_rules: rules,
        ..OrderRequest::market(SBIN, side, QuantitySpec::Units(units))
    }
}

fn signals(when: [Vec<bool>; 4], pyramiding: u32, rules: LegRules) -> SignalArrays {
    let [entry, exit, short, cover] = when;
    SignalArrays {
        instrument: SBIN,
        product: Product::Mis,
        bar_opens: bar_opens(),
        entry: Some(EntrySignal {
            when: entry,
            order: entry_order(Side::Buy, 10.0, rules.clone()),
        }),
        exit: Some(exit),
        short: Some(EntrySignal {
            when: short,
            order: entry_order(Side::Sell, 10.0, rules),
        }),
        cover: Some(cover),
        pyramiding,
    }
}

/// The `on_bar` of 14-strategy-api 4.1 written out literally, as an event strategy
/// would write it, reading `self.position` and placing `buy`, `sell` and `exit()`.
struct LiteralOnBar<'a> {
    signals: &'a SignalArrays,
}

impl LiteralOnBar<'_> {
    fn on_bar(&self, engine: &mut Engine, bar: &StrategyBar) -> Result<(), KernelError> {
        let s = self.signals;
        let i = s.bar_opens.binary_search(&bar.ts_open).expect("aligned");
        let value = |array: &Vec<bool>| array[i];
        let (entry, short) = (s.entry.as_ref().expect("e"), s.short.as_ref().expect("s"));
        let (exit, cover) = (s.exit.as_ref().expect("x"), s.cover.as_ref().expect("c"));
        let position = engine.position(SBIN, Product::Mis);
        let held = position.map_or(0.0, |p| p.quantity_units);
        let adds = position.map_or(0, |p| p.adds);
        let flat = held == 0.0 && !engine.has_pending_entry(SBIN);
        let exit_order = |side| OrderRequest {
            product: Product::Mis,
            reduce_only: true,
            ..OrderRequest::market(SBIN, side, QuantitySpec::PositionFraction(1.0))
        };
        if flat {
            if value(&entry.when) {
                engine.submit(entry.order.clone())?;
            } else if value(&short.when) {
                engine.submit(short.order.clone())?;
            }
        } else if held > 0.0 {
            if value(exit) {
                engine.submit(exit_order(Side::Sell))?;
            } else if value(&entry.when) && adds < s.pyramiding {
                engine.submit(entry.order.clone())?;
            }
        } else if held < 0.0 {
            if value(cover) {
                engine.submit(exit_order(Side::Buy))?;
            } else if value(&short.when) && adds < s.pyramiding {
                engine.submit(short.order.clone())?;
            }
        }
        Ok(())
    }
}

impl Driver for LiteralOnBar<'_> {
    fn on_pause(&mut self, engine: &mut Engine, pause: &Pause) -> Result<(), KernelError> {
        if let Pause::BarClose { bars, .. } = pause {
            for bar in bars.iter().filter(|bar| bar.instrument == SBIN) {
                self.on_bar(engine, bar)?;
            }
        }
        Ok(())
    }
}

fn with_signal_driver(bars: oa_bt::engine::BarColumns, signals: &SignalArrays) -> RunOutput {
    signal_run(engine_for(bars), signals)
}

fn signal_run(mut kernel: Engine, signals: &SignalArrays) -> RunOutput {
    run_signals(&mut kernel, signals).expect("the run completes");
    kernel.finish().expect("finished")
}

fn literal_run(mut kernel: Engine, signals: &SignalArrays) -> RunOutput {
    run(&mut kernel, &mut LiteralOnBar { signals }).expect("the run completes");
    kernel.finish().expect("finished")
}

fn random_flags(rng: &mut Lcg, percent: u64) -> Vec<bool> {
    (0..BARS).map(|_| rng.chance(percent)).collect()
}

fn stop_and_target() -> LegRules {
    LegRules {
        stop: Some(Distance::Points(1.0)),
        targets: vec![TargetSpec {
            distance: Distance::Points(1.5),
            fraction: 1.0,
        }],
        ..LegRules::default()
    }
}

/// Sixty seeded runs of random arrays on random bars: the run of the signal driver
/// equals the run of the literal `on_bar`, every table bit for bit (`RunOutput`'s
/// equality compares every float exactly).
fn assert_equal_runs_on_random_arrays_and_bars(costs: CostConfig, slippage: SlippageSpec) {
    for seed in 0..60u64 {
        let mut rng = Lcg(seed);
        let bars = walk(&mut rng);
        let when = [
            random_flags(&mut rng, 30),
            random_flags(&mut rng, 30),
            random_flags(&mut rng, 30),
            random_flags(&mut rng, 30),
        ];
        let rules = if seed % 2 == 0 {
            stop_and_target()
        } else {
            LegRules::default()
        };
        let signals = signals(when, (seed % 3) as u32, rules);
        let kernel = || engine_with(bars.clone(), costs.clone(), slippage);
        let by_signals = signal_run(kernel(), &signals);
        let by_on_bar = literal_run(kernel(), &signals);
        assert!(!by_on_bar.fills.is_empty(), "seed {seed} trades");
        assert_eq!(by_signals, by_on_bar, "seed {seed}");
    }
}

#[test]
fn the_signal_driver_equals_the_literal_on_bar_on_random_arrays_and_bars() {
    assert_equal_runs_on_random_arrays_and_bars(CostConfig::Off, SlippageSpec::Zero);
}

#[test]
fn the_signal_driver_equals_the_literal_on_bar_with_charges_and_slippage() {
    let (costs, slippage) = charges_and_slippage();
    assert_equal_runs_on_random_arrays_and_bars(costs, slippage);
}

/// Every bar closes at 500.00 and opens there too, so nothing but the conditions acts.
fn flat_bars() -> oa_bt::engine::BarColumns {
    sessions_15m(&days(), |_, _| Some((500.0, 500.0)))
}

fn first_session(values: &[usize]) -> Vec<bool> {
    (0..BARS).map(|bar| values.contains(&bar)).collect()
}

/// (side, units) of every fill in order.
fn sides(output: &RunOutput) -> Vec<(Side, f64)> {
    output
        .fills
        .iter()
        .map(|record| (record.fill.side, record.fill.quantity_units))
        .collect()
}

#[test]
fn a_condition_that_stays_true_adds_up_to_pyramiding_and_no_more() {
    // Entry true on bars 0 to 5: entry at bar 0, adds at bars 1 and 2 (pyramiding 2),
    // then nothing until the square-off closes the 30 units.
    let when = [
        first_session(&[0, 1, 2, 3, 4, 5]),
        vec![false; BARS],
        vec![false; BARS],
        vec![false; BARS],
    ];
    let output = with_signal_driver(flat_bars(), &signals(when, 2, LegRules::default()));
    let buys: Vec<_> = sides(&output)
        .into_iter()
        .filter(|(side, _)| *side == Side::Buy)
        .collect();
    assert_eq!(buys, vec![(Side::Buy, 10.0); 3]);
}

#[test]
fn exit_wins_over_adding_and_long_entry_wins_over_short() {
    // Bar 0: entry and short both true while flat, so a buy. Bar 1: in the long, exit and
    // entry both true, so the exit (pyramiding 5 would allow an add). Bar 2: flat again
    // with only the short true, so a sell. Bar 3: cover and short true, so the cover.
    let when = [
        first_session(&[0, 1]),
        first_session(&[1]),
        first_session(&[0, 2, 3]),
        first_session(&[3]),
    ];
    let output = with_signal_driver(flat_bars(), &signals(when, 5, LegRules::default()));
    assert_eq!(
        sides(&output),
        vec![
            (Side::Buy, 10.0),
            (Side::Sell, 10.0),
            (Side::Sell, 10.0),
            (Side::Buy, 10.0)
        ]
    );
}

#[test]
fn arrays_of_the_wrong_length_are_refused_before_the_run() {
    let mut kernel = engine_for(flat_bars());
    let when = [
        vec![true; BARS - 1],
        vec![false; BARS],
        vec![false; BARS],
        vec![false; BARS],
    ];
    let error = run_signals(&mut kernel, &signals(when, 0, LegRules::default()));
    assert!(
        matches!(error, Err(KernelError::InvalidConfig { field: "conditions", ref reason })
            if reason.contains("entry_when has 49 values for 50 strategy bars")),
        "{error:?}"
    );
}

#[test]
fn an_entry_order_on_the_wrong_side_is_refused() {
    let mut signals = signals(
        [
            vec![false; BARS],
            vec![false; BARS],
            vec![false; BARS],
            vec![false; BARS],
        ],
        0,
        LegRules::default(),
    );
    if let Some(entry) = signals.entry.as_mut() {
        entry.order.side = Side::Sell;
    }
    let error = run_signals(&mut engine_for(flat_bars()), &signals);
    assert!(
        matches!(error, Err(KernelError::InvalidConfig { ref reason, .. })
            if reason.contains("entry_when order is on the wrong side")),
        "{error:?}"
    );
}

#[test]
fn arrays_that_miss_a_strategy_bar_stop_the_run_at_that_bar() {
    let mut signals = signals(
        [
            vec![false; BARS],
            vec![false; BARS],
            vec![false; BARS],
            vec![false; BARS],
        ],
        0,
        LegRules::default(),
    );
    // Shift every open by one minute: no array value belongs to a strategy bar.
    signals.bar_opens = bar_opens().iter().map(|ts| ts + 60_000_000_000).collect();
    let error = run_signals(&mut engine_for(flat_bars()), &signals);
    assert!(
        matches!(error, Err(KernelError::InvalidConfig { ref reason, .. })
            if reason.contains("no value for the strategy bar opening at")),
        "{error:?}"
    );
}

#[test]
fn an_entry_the_sizer_skips_is_returned_with_its_decision_instant() {
    // 100 rupees of a 500.00 share is 0.2 shares, which rounds down to none (venue).
    let mut signals = signals(
        [
            first_session(&[0]),
            vec![false; BARS],
            vec![false; BARS],
            vec![false; BARS],
        ],
        0,
        LegRules::default(),
    );
    if let Some(entry) = signals.entry.as_mut() {
        entry.order.quantity = QuantitySpec::Sizer(SizerSpec::Value { rupees: 100.0 });
    }
    let mut kernel = engine_for(flat_bars());
    let skipped = run_signals(&mut kernel, &signals).expect("the run completes");
    let expected = SkippedEntry {
        decided_at: bar_close(july_2025(1), 0),
        reason: SkipReason::RoundedToZero,
    };
    assert_eq!(skipped, vec![expected]);
    assert!(kernel.finish().expect("finished").fills.is_empty());
}

#[test]
fn undeclared_conditions_never_trade() {
    let signals = SignalArrays {
        instrument: SBIN,
        product: Product::Mis,
        bar_opens: bar_opens(),
        entry: None,
        exit: None,
        short: None,
        cover: None,
        pyramiding: 0,
    };
    let output = with_signal_driver(flat_bars(), &signals);
    assert!(output.fills.is_empty());
}
