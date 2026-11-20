//! KR8 (18-multi-strategy 2.8): an engine in a fund runs exactly as it runs alone when the
//! account never binds, every table of its `RunOutput` equal: a fund of one, a separate
//! pool of three (one of them on a daily base beside two 1-minute ones), and a shared pool
//! whose account is the sum of the capitals and never runs short. Condition strategies are
//! decided inline by their Rust drivers, as `run_signals` decides them; a timer pause goes
//! to its driver in the open phase. A fund run twice gives the same outputs.

use crate::engine::{run_signals, Engine, EntrySignal, RunOutput, SignalArrays, Step, TimerSpec};
use crate::exec::{OrderRequest, QuantitySpec};
use crate::model::{Product, Side};

use super::super::{Fund, Pool};
use super::kit::{account, alone, drive, exit_all, in_fund, july, market, wave, with_stop};
use super::kit::{Run, Script, Symbol, ONE};

/// SBIN on 1-minute bars of 2025-07-07 and 08 with 10,00,000: an MIS long with a stop,
/// an add, its exit, an MIS short and its cover on the first day; a CNC buy, half of it
/// sold, the rest held to the end on the second.
fn sbin_minutes() -> (Engine, Script) {
    let run = Run::minutes(Symbol::Sbin, july(7), july(8), 1_000_000.0);
    let reduce = OrderRequest {
        reduce_only: true,
        ..market(Side::Sell, 100.0, Product::Cnc)
    };
    let script = Script::new(&[
        (3, with_stop(Side::Buy, 100.0, Product::Mis, 2.0)),
        (10, market(Side::Buy, 50.0, Product::Mis)),
        (40, exit_all(Side::Sell, Product::Mis)),
        (60, market(Side::Sell, 80.0, Product::Mis)),
        (70, exit_all(Side::Buy, Product::Mis)),
        (380, market(Side::Buy, 200.0, Product::Cnc)),
        (600, reduce),
    ]);
    (run.engine(|day, k| wave(100.0, day, k)), script)
}

/// INFY on 1-minute bars of the same days with 5,00,000: MIS round trips of 300.
fn infy_minutes() -> (Engine, Script) {
    let run = Run::minutes(Symbol::Infy, july(7), july(8), 500_000.0);
    let script = Script::new(&[
        (20, market(Side::Buy, 300.0, Product::Mis)),
        (90, exit_all(Side::Sell, Product::Mis)),
        (400, market(Side::Sell, 300.0, Product::Mis)),
        (500, exit_all(Side::Buy, Product::Mis)),
    ]);
    (run.engine(|day, k| wave(1_500.0, day, k)), script)
}

/// SBIN on daily bars of 2025-07-01 to 08 with 3,00,000: a CNC buy of 500, its exit.
fn sbin_daily() -> (Engine, Script) {
    let run = Run::daily(Symbol::Sbin, july(1), july(8), 300_000.0);
    let script = Script::new(&[
        (0, market(Side::Buy, 500.0, Product::Cnc)),
        (3, exit_all(Side::Sell, Product::Cnc)),
    ]);
    (run.engine(|day, _| wave(100.0, day, 0)), script)
}

/// The three strategies of the module docs, in call order.
fn three() -> (Vec<Engine>, Vec<Script>) {
    let (parts, scripts) = [sbin_minutes(), infy_minutes(), sbin_daily()]
        .into_iter()
        .unzip();
    (parts, scripts)
}

/// Each of `strategies` run alone.
fn each_alone(strategies: (Vec<Engine>, Vec<Script>)) -> Vec<RunOutput> {
    let (engines, scripts) = strategies;
    engines
        .into_iter()
        .zip(&scripts)
        .map(|(e, s)| alone(e, s))
        .collect()
}

#[test]
fn a_fund_of_one_runs_the_engine_as_it_runs_alone() {
    let (engine, script) = sbin_minutes();
    let solo = alone(engine.clone(), &script);
    // The stop exits the first long inside a bar (step 3), so the levels phase trades too.
    let stopped = solo
        .fills
        .iter()
        .any(|record| record.stamp.step == Step::Levels);
    assert!(
        stopped && solo.fills.len() >= 7,
        "the script trades: {:?}",
        solo.fills
    );
    for pool in [Pool::Shared, Pool::Separate] {
        let scripts = std::slice::from_ref(&script);
        let (outputs, _) = in_fund(account(pool, None, None), vec![engine.clone()], scripts);
        assert_eq!(outputs, vec![solo.clone()], "{pool:?}");
    }
    // An account limit that is never reached changes nothing either.
    let limited = account(Pool::Shared, None, Some(10_000_000.0));
    let (outputs, fund) = in_fund(limited, vec![engine], &[script]);
    assert_eq!((outputs, fund.breaches.len()), (vec![solo], 0));
}

#[test]
fn a_separate_pool_of_three_runs_each_as_it_runs_alone() {
    let solo = each_alone(three());
    let (engines, scripts) = three();
    let (outputs, fund) = in_fund(account(Pool::Separate, None, None), engines, &scripts);
    assert_eq!(outputs, solo);
    assert!(fund.refused.is_empty());
}

#[test]
fn a_shared_pool_whose_account_never_runs_short_runs_each_as_alone() {
    let solo = each_alone(three());
    let (engines, scripts) = three();
    let (outputs, fund) = in_fund(account(Pool::Shared, None, None), engines, &scripts);
    assert_eq!(outputs, solo);
    assert!(fund.refused.is_empty());
}

/// The condition arrays of SBIN's 1-minute run: an MIS entry of 100 every 50 bars, its
/// exit 20 bars later.
fn sbin_conditions(engine: &Engine) -> SignalArrays {
    let bar_opens: Vec<i64> = engine
        .strategy_bars(ONE)
        .iter()
        .map(|b| b.ts_open)
        .collect();
    let entry = (0..bar_opens.len()).map(|i| i % 50 == 5).collect();
    let exit = (0..bar_opens.len()).map(|i| i % 50 == 25).collect();
    let order = OrderRequest {
        product: Product::Mis,
        ..OrderRequest::market(ONE, Side::Buy, QuantitySpec::Units(100.0))
    };
    SignalArrays {
        instrument: ONE,
        product: Product::Mis,
        bar_opens,
        entry: Some(EntrySignal {
            when: entry,
            order,
            atr: Vec::new(),
        }),
        exit: Some(exit),
        short: None,
        cover: None,
        pyramiding: 0,
    }
}

#[test]
fn a_condition_strategy_in_a_fund_is_decided_inline_as_run_signals_decides_it() {
    let (conditions, _) = sbin_minutes();
    let arrays = sbin_conditions(&conditions);
    let mut solo = conditions.clone();
    let solo_skipped = run_signals(&mut solo, &arrays).expect("the conditions run");
    let solo = solo.finish().expect("finished");
    let (events, script) = infy_minutes();
    let solo_events = alone(events.clone(), &script);

    let mut engines = vec![conditions, events];
    let lent: Vec<&Engine> = engines.iter().collect();
    let mut fund = Fund::new(account(Pool::Separate, None, None), &lent).expect("a fund");
    fund.set_signals(0, arrays, &engines[0])
        .expect("the arrays fit");
    let fund = drive(fund, &mut engines, &[Script::default(), script]);
    let outputs: Vec<_> = engines
        .into_iter()
        .map(|e| e.finish().expect("finished"))
        .collect();
    assert_eq!(outputs, vec![solo.clone(), solo_events]);
    assert!(
        solo.trades.len() >= 10,
        "the conditions trade: {}",
        solo.trades.len()
    );
    assert_eq!(
        fund.finish().expect("finished").skipped,
        vec![solo_skipped, vec![]]
    );
}

#[test]
fn a_timer_pause_goes_to_its_driver_and_its_orders_fill_at_that_open() {
    // SBIN with a timer at 10:00 each session buys 100 MIS at the timer's pause, which
    // fills at that same open (minute 45 of the 7th, step 2), beside INFY in a shared pool.
    let (mut timed, _) = sbin_minutes();
    timed
        .add_timer(TimerSpec::At {
            minute_of_day: 10 * 60,
        })
        .expect("before the first advance");
    let script = Script {
        at_timers: vec![(45, market(Side::Buy, 100.0, Product::Mis))],
        ..Script::default()
    };
    let (infy, infy_script) = infy_minutes();
    let solo = vec![
        alone(timed.clone(), &script),
        alone(infy.clone(), &infy_script),
    ];
    let scripts = [script, infy_script];
    let (outputs, _) = in_fund(
        account(Pool::Shared, None, None),
        vec![timed, infy],
        &scripts,
    );
    assert_eq!(outputs, solo);
    let at_timer = &outputs[0].fills[0];
    assert_eq!(
        (at_timer.stamp.slice, at_timer.stamp.step),
        (45, Step::OpenFills)
    );
}

#[test]
fn a_fund_run_twice_gives_the_same_outputs() {
    let first = {
        let (engines, scripts) = three();
        in_fund(
            account(Pool::Shared, Some(500_000.0), Some(20_000.0)),
            engines,
            &scripts,
        )
    };
    let (engines, scripts) = three();
    let second = in_fund(
        account(Pool::Shared, Some(500_000.0), Some(20_000.0)),
        engines,
        &scripts,
    );
    assert_eq!(first, second);
}
