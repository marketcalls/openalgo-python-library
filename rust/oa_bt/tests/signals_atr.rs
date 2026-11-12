#![cfg(all(feature = "costs", feature = "time"))]
//! ATR distances and the ATR-risk sizer in condition strategies (14-strategy-api 4.1,
//! SPEC-25): the signal driver puts the deciding bar's ATR into the order, as an
//! `on_bar` order resolves its ATR when it is placed.
//!
//! Fifteen-minute MIS bars on the embedded NSE calendar, two sessions (2025-07-01 and
//! 2025-07-02), capital 10,00,000, no charges and no slippage.

mod risk_support;

use oa_bt::engine::{
    run, run_signals, AtrInput, AtrSlot, Driver, Engine, EntrySignal, KernelError, Pause,
    RunOutput, SignalArrays, StrategyBar,
};
use oa_bt::exec::{InstrumentId, OrderRequest, QuantitySpec};
use oa_bt::model::{Product, Side};
use oa_bt::rms::{Distance, LegRules, StrategyRiskConfig, TrailSpec};
use oa_bt::sizing::SizerSpec;
use risk_support::*;

const SBIN: InstrumentId = InstrumentId(0);
const BARS: usize = 2 * BARS_15M_PER_SESSION;
const PERIOD: u32 = 14;

fn days() -> [oa_bt::model::Day; 2] {
    [july_2025(1), july_2025(2)]
}

fn bar_opens() -> Vec<i64> {
    days()
        .iter()
        .flat_map(|&day| (0..BARS_15M_PER_SESSION).map(move |k| bar_open(day, k)))
        .collect()
}

fn engine_for(bars: oa_bt::engine::BarColumns) -> Engine {
    let days = days();
    let config = intraday_config((days[0], days[1]), StrategyRiskConfig::default());
    engine(config, vec![equity("SBIN")], vec![bars]).expect("a valid setup")
}

/// Every bar opens and closes at 500.00.
fn flat_bars() -> oa_bt::engine::BarColumns {
    sessions_15m(&days(), |_, _| Some((500.0, 500.0)))
}

/// Bars that walk by whole ticks from 500.00, a fixed pattern of up to 1.95 a bar.
fn walking_bars() -> oa_bt::engine::BarColumns {
    let mut closes = vec![500.0];
    for bar in 0..BARS {
        let ticks = ((bar * 37) % 79) as f64 - 39.0;
        let last = closes[closes.len() - 1];
        closes.push(((last + ticks * 0.05) * 20.0).round() / 20.0);
    }
    sessions_15m(&days(), |session, k| {
        let bar = session * BARS_15M_PER_SESSION + k;
        Some((closes[bar], closes[bar + 1]))
    })
}

fn atr_stop(multiple: f64) -> LegRules {
    LegRules {
        stop: Some(Distance::Atr {
            multiple,
            atr_rupees: f64::NAN,
        }),
        ..LegRules::default()
    }
}

fn buy(quantity: QuantitySpec, rules: LegRules) -> OrderRequest {
    OrderRequest {
        product: Product::Mis,
        leg_rules: rules,
        ..OrderRequest::market(SBIN, Side::Buy, quantity)
    }
}

fn entry_only(when: Vec<bool>, order: OrderRequest, atr: Vec<AtrInput>) -> SignalArrays {
    SignalArrays {
        instrument: SBIN,
        product: Product::Mis,
        bar_opens: bar_opens(),
        entry: Some(EntrySignal { when, order, atr }),
        exit: None,
        short: None,
        cover: None,
        pyramiding: 0,
    }
}

fn at_bars(bars: &[usize]) -> Vec<bool> {
    (0..BARS).map(|bar| bars.contains(&bar)).collect()
}

fn stop_input(rupees: Vec<f64>) -> AtrInput {
    AtrInput {
        slot: AtrSlot::Stop,
        period: PERIOD,
        rupees,
    }
}

#[test]
fn an_atr_stop_takes_the_atr_of_the_deciding_bar() {
    // Entry decided at bar 2, where the ATR is 4.00: it fills at 500.00 and its stop is
    // 2 x 4.00 below, at 492.00.
    let mut atr = vec![f64::NAN; BARS];
    atr[2] = 4.0;
    atr[3] = 9.0;
    let order = buy(QuantitySpec::Units(10.0), atr_stop(2.0));
    let signals = entry_only(at_bars(&[2]), order, vec![stop_input(atr)]);
    let mut kernel = engine_for(flat_bars());
    run_signals(&mut kernel, &signals).expect("the run completes");
    let output = kernel.finish().expect("finished");
    let opened = output
        .levels
        .iter()
        .find(|row| row.quantity_units == 10.0)
        .expect("the position opened");
    assert_eq!(opened.stop.map(|stop| stop.level_rupees), Some(492.0));
}

#[test]
fn the_atr_risk_sizer_takes_the_atr_of_the_deciding_bar() {
    // 1 percent of 10,00,000 is 10,000 of risk; the ATR at bar 0 is 5.00 and the stop 2
    // ATRs away, 10.00 a share: 1,000 shares.
    let sizer = SizerSpec::AtrRisk {
        risk_percent: 1.0,
        multiple: 2.0,
        atr_rupees: f64::NAN,
    };
    let mut atr = vec![f64::NAN; BARS];
    atr[0] = 5.0;
    let input = AtrInput {
        slot: AtrSlot::Sizer,
        period: PERIOD,
        rupees: atr,
    };
    let order = buy(QuantitySpec::Sizer(sizer), LegRules::default());
    let signals = entry_only(at_bars(&[0]), order, vec![input]);
    let mut kernel = engine_for(flat_bars());
    run_signals(&mut kernel, &signals).expect("the run completes");
    let output = kernel.finish().expect("finished");
    assert_eq!(output.fills[0].fill.quantity_units, 1_000.0);
}

#[test]
fn an_atr_distance_without_a_value_at_the_deciding_bar_stops_the_run() {
    let order = buy(QuantitySpec::Units(10.0), atr_stop(2.0));
    let signals = entry_only(at_bars(&[0]), order, vec![stop_input(vec![f64::NAN; BARS])]);
    let error = run_signals(&mut engine_for(flat_bars()), &signals);
    assert!(
        matches!(error, Err(KernelError::InvalidConfig { field: "conditions", ref reason })
            if reason.contains("bt.atr(period=14) has no positive value")),
        "{error:?}"
    );
}

#[test]
fn atr_values_must_be_given_for_exactly_the_atrs_the_order_names() {
    let with_atr = buy(QuantitySpec::Units(10.0), atr_stop(2.0));
    let without_atr = buy(QuantitySpec::Units(10.0), LegRules::default());
    let target_input = AtrInput {
        slot: AtrSlot::Target,
        period: PERIOD,
        rupees: vec![4.0; BARS],
    };
    let cases = [
        (with_atr.clone(), vec![], "names ATRs at [Stop]"),
        (
            without_atr,
            vec![target_input],
            "values were given for [Target]",
        ),
        (
            with_atr,
            vec![stop_input(vec![4.0; 3])],
            "has 3 values for 50",
        ),
    ];
    for (order, inputs, message) in cases {
        let signals = entry_only(at_bars(&[0]), order, inputs);
        let error = run_signals(&mut engine_for(flat_bars()), &signals);
        assert!(
            matches!(error, Err(KernelError::InvalidConfig { ref reason, .. })
                if reason.contains(message)),
            "{message}: {error:?}"
        );
    }
}

/// The entry and exit of 14-strategy-api 4.1 written out, putting each bar's ATR into
/// the order it places, as `on_bar` resolves `bt.atr` when it orders.
struct LiteralOnBar<'a> {
    signals: &'a SignalArrays,
    exit: &'a [bool],
    atr: &'a [f64],
}

impl LiteralOnBar<'_> {
    fn on_bar(&self, engine: &mut Engine, bar: &StrategyBar) -> Result<(), KernelError> {
        let i = self
            .signals
            .bar_opens
            .binary_search(&bar.ts_open)
            .expect("aligned");
        let entry = self.signals.entry.as_ref().expect("an entry");
        let held = engine
            .position(SBIN, Product::Mis)
            .map_or(0.0, |p| p.quantity_units);
        if held == 0.0 && !engine.has_pending_entry(SBIN) && entry.when[i] {
            let mut order = entry.order.clone();
            let atr = |multiple| Distance::Atr {
                multiple,
                atr_rupees: self.atr[i],
            };
            order.leg_rules.stop = Some(atr(1.5));
            order.leg_rules.trail = Some(TrailSpec {
                trigger: atr(2.0),
                step: None,
            });
            engine.submit(order)?;
        } else if held > 0.0 && self.exit[i] {
            engine.submit(OrderRequest {
                product: Product::Mis,
                reduce_only: true,
                ..OrderRequest::market(SBIN, Side::Sell, QuantitySpec::PositionFraction(1.0))
            })?;
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

#[test]
fn a_condition_with_atr_rules_equals_the_on_bar_that_resolves_the_atr_when_it_orders() {
    // An ATR stop of 1.5 and an ATR trail of 2, entries every 7th bar and exits every
    // 11th, with an ATR that changes every bar.
    let rules = LegRules {
        stop: Some(Distance::Atr {
            multiple: 1.5,
            atr_rupees: f64::NAN,
        }),
        trail: Some(TrailSpec {
            trigger: Distance::Atr {
                multiple: 2.0,
                atr_rupees: f64::NAN,
            },
            step: None,
        }),
        ..LegRules::default()
    };
    let atr: Vec<f64> = (0..BARS).map(|bar| 0.5 + (bar % 9) as f64 * 0.25).collect();
    let when: Vec<bool> = (0..BARS).map(|bar| bar % 7 == 0).collect();
    let exit: Vec<bool> = (0..BARS).map(|bar| bar % 11 == 5).collect();
    let inputs = [AtrSlot::Stop, AtrSlot::TrailTrigger].map(|slot| AtrInput {
        slot,
        period: PERIOD,
        rupees: atr.clone(),
    });
    let mut signals = entry_only(when, buy(QuantitySpec::Units(10.0), rules), inputs.to_vec());
    signals.exit = Some(exit.clone());
    let mut by_signals = engine_for(walking_bars());
    run_signals(&mut by_signals, &signals).expect("the run completes");
    let by_signals: RunOutput = by_signals.finish().expect("finished");
    let mut by_on_bar = engine_for(walking_bars());
    let mut literal = LiteralOnBar {
        signals: &signals,
        exit: &exit,
        atr: &atr,
    };
    run(&mut by_on_bar, &mut literal).expect("the run completes");
    let by_on_bar = by_on_bar.finish().expect("finished");
    assert!(by_on_bar.fills.len() > 4, "the run trades");
    assert_eq!(by_signals, by_on_bar);
}
