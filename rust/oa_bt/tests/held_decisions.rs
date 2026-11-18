#![cfg(all(feature = "costs", feature = "time"))]
//! Decisions held for a delay (design 17, section 2.14; `delay_entries`, `delay_exits`):
//! every decision the strategy makes about an order reaches the market late, its
//! cancellation and its modification as much as its placement, the order working on its
//! old terms meanwhile; an order decided by a timer inside a strategy bar counts its delay
//! from that bar's close; a delay longer than a month of sessions still fills, and one
//! that runs past the session's close or the run's last strategy bar is refused with its
//! own code. Every expected price and instant is worked by hand from the bars written in
//! each test (NSE, July and August 2025, no charges, no slippage).

mod order_support;

use oa_bt::engine::{Pause, RunConfig, RunOutput, StrategyInterval, Submission, TimerSpec};
use oa_bt::exec::{OrderId, OrderStatus, OrderType};
use oa_bt::market::calendar::Calendar;
use oa_bt::model::{days_from_civil, ts_from_ist, Day, Exchange, Product, Side};
use order_support::{
    day_open, engine, equity, events, fills, limit, minute, minute_bars, minute_config, order,
    run_daily, run_script, Act, Ohlc, MINUTE, NSE_OPEN_MINUTE,
};

/// Bars whose minute k opens at 100 + k, high open + 2, low open - 1, close open + 1:
/// minute 1 is (101, 103, 100, 102), minute 2 (102, 104, 101, 103), minute 3 (103, 105,
/// 102, 104).
fn rising(count: u32) -> Vec<(u32, Ohlc)> {
    (0..count)
        .map(|k| {
            let open = 100.0 + f64::from(k);
            (k, (open, open + 2.0, open - 1.0, open + 1.0))
        })
        .collect()
}

fn held(entries: u32, exits: u32, mut config: RunConfig) -> RunConfig {
    config.perturbation.delay_entries = entries;
    config.perturbation.delay_exits = exits;
    config
}

fn limit_buy(price: f64) -> oa_bt::exec::OrderRequest {
    order(Side::Buy, 10.0, limit(price), Product::Cnc)
}

/// A one-minute strategy with entries held one bar, scripted at the given decisions.
fn entries_held_one_bar(acts: Vec<(i64, Act)>) -> RunOutput {
    let config = held(1, 0, minute_config());
    let kernel = engine(config, equity(), minute_bars(10, 100.0, &rising(10)));
    run_script(kernel, acts).0
}

/// The (slice open, step) of the first event named `name`.
fn first_event(output: &RunOutput, name: &str) -> Option<(i64, u8)> {
    events(output)
        .into_iter()
        .find(|(_, _, event)| event == name)
        .map(|(slice, step, _)| (slice, step))
}

#[test]
fn a_held_orders_cancellation_is_held_too_and_the_order_works_until_it_arrives() {
    // The buy limit at 101, decided at minute 0's close, works from minute 1's close
    // (minute(2)). Its cancellation, decided there, arrives one bar later, at minute 2's
    // close: minute 2 (102, 104, 101, 103) reaches 101 first, and the order fills there.
    let output = entries_held_one_bar(vec![
        (minute(1), Act::Submit(limit_buy(101.0))),
        (minute(2), Act::Cancel(OrderId(1))),
    ]);
    let rows = fills(&output);
    assert_eq!(rows.len(), 1, "{rows:?}");
    assert_eq!(
        (rows[0].0, rows[0].2, rows[0].4),
        (minute(2), Side::Buy, 101.0)
    );
    assert_eq!(output.orders[0].status, OrderStatus::Filled);
    assert_eq!(first_event(&output, "Canceled"), None);
}

#[test]
fn a_cancellation_takes_effect_when_its_delay_ends() {
    // A buy limit at 90 never fills; cancelled at minute(2) with entries held one bar, it
    // is cancelled at the open of minute 3, the first slice after minute 2's close.
    let output = entries_held_one_bar(vec![
        (minute(1), Act::Submit(limit_buy(90.0))),
        (minute(2), Act::Cancel(OrderId(1))),
    ]);
    assert_eq!(first_event(&output, "Canceled"), Some((minute(3), 2)));
    assert_eq!(output.orders[0].status, OrderStatus::Canceled);
}

#[test]
fn a_modification_leaves_the_old_terms_working_until_it_arrives() {
    // The buy limit at 102 works from minute(2); the change to 90, decided there, arrives
    // at minute 2's close. Minute 2 opens at 102: the old terms fill at its open.
    let output = entries_held_one_bar(vec![
        (minute(1), Act::Submit(limit_buy(102.0))),
        (minute(2), Act::Modify(OrderId(1), limit_buy(90.0))),
    ]);
    let rows = fills(&output);
    assert_eq!(rows.len(), 1, "{rows:?}");
    assert_eq!((rows[0].0, rows[0].4), (minute(2), 102.0));
    assert_eq!(first_event(&output, "Modified"), None);
}

#[test]
fn a_modification_takes_effect_when_its_delay_ends() {
    // The buy limit at 90 works from minute(2); the change to 104 arrives at minute 2's
    // close and is stamped at minute 3's open, which it fills at (103, at or below 104).
    let output = entries_held_one_bar(vec![
        (minute(1), Act::Submit(limit_buy(90.0))),
        (minute(2), Act::Modify(OrderId(1), limit_buy(104.0))),
    ]);
    assert_eq!(first_event(&output, "Modified"), Some((minute(3), 2)));
    let rows = fills(&output);
    assert_eq!((rows[0].0, rows[0].4), (minute(3), 103.0));
}

/// A 15-minute strategy on one-minute bars of 2025-07-01 with entries held `delay` bars;
/// it buys from a timer at 09:20. The price the buy fills at.
fn timer_buy_price(delay: u32) -> f64 {
    let mut config = held(delay, 0, minute_config());
    config.strategy_interval = StrategyInterval::Intraday(15 * MINUTE);
    let mut kernel = engine(config, equity(), minute_bars(375, 100.0, &rising(375)));
    kernel
        .add_timer(TimerSpec::At {
            minute_of_day: 9 * 60 + 20,
        })
        .expect("before the run");
    loop {
        match kernel.advance().expect("the run advances") {
            Pause::Timers { .. } => {
                let buy = order(Side::Buy, 10.0, OrderType::Market, Product::Cnc);
                let placed = kernel.submit(buy).expect("at a decision point");
                assert!(matches!(placed, Submission::Submitted(_)), "{placed:?}");
            }
            Pause::Finished { .. } => break,
            _ => {}
        }
    }
    let output = kernel.finish().expect("the run finished");
    fills(&output)[0].4
}

#[test]
fn a_timers_order_counts_its_delay_from_the_close_of_the_bar_it_is_inside() {
    // The timer fires at 09:20, inside the 09:15 to 09:30 bar. On time the buy fills at
    // 09:20 (minute 5, 105); held one bar, from the close of the 09:30 to 09:45 bar, at
    // 09:45 (minute 30, 130); held two, at 10:00 (minute 45, 145).
    assert_eq!(timer_buy_price(0), 105.0);
    assert_eq!(timer_buy_price(1), 130.0);
    assert_eq!(timer_buy_price(2), 145.0);
}

/// The NSE sessions from 2025-07-01 on, `count` of them.
fn sessions(count: usize) -> Vec<Day> {
    let calendar = Calendar::new().expect("the embedded calendar tables parse");
    (0..120)
        .map(|ahead| days_from_civil(2025, 7, 1) + ahead)
        .filter(|day| calendar.session(Exchange::Nse, *day).is_some())
        .take(count)
        .collect()
}

/// Daily bars of `days`, session n opening at 100 + n with a range of 2 around it.
fn numbered(days: &[Day]) -> Vec<(Day, Ohlc)> {
    days.iter()
        .enumerate()
        .map(|(n, day)| {
            let open = 100.0 + n as f64;
            (*day, (open, open + 1.0, open - 1.0, open))
        })
        .collect()
}

#[test]
fn a_daily_order_held_longer_than_a_month_of_sessions_fills() {
    // Decided at session 0's close (07-01) and held 25 sessions, to session 25's close: it
    // fills at session 26's open (08-06), 126, 36 calendar days after the decision.
    let days = sessions(30);
    let buy = order(Side::Buy, 10.0, OrderType::Market, Product::Cnc);
    let (output, _, _) = run_daily(&numbered(&days), vec![(days[0], Act::Submit(buy))], |c| {
        c.perturbation.delay_entries = 25;
    });
    let rows = fills(&output);
    assert_eq!(rows.len(), 1, "{rows:?}");
    assert_eq!((rows[0].0, rows[0].4), (day_open(days[26]), 126.0));
}

#[test]
fn an_order_held_past_the_runs_last_strategy_bar_is_refused_with_its_own_code() {
    // Five sessions; a buy decided at session 3's close and held two would wait to session
    // 5's close, after the run.
    let days = sessions(5);
    let buy = order(Side::Buy, 10.0, OrderType::Market, Product::Cnc);
    let (output, _, script) = run_daily(&numbered(&days), vec![(days[3], Act::Submit(buy))], |c| {
        c.perturbation.delay_entries = 2;
    });
    assert!(fills(&output).is_empty());
    assert!(matches!(
        script.submissions[0],
        Submission::Rejected(_, code) if code.code() == "delay_past_close"
    ));
}

#[test]
fn an_intraday_order_held_past_its_sessions_close_is_refused_with_its_own_code() {
    // The session's last one-minute bar closes at 15:30; an entry decided at 15:29 (the
    // close of minute 373) and held one bar would wait to 15:30.
    let config = held(1, 0, minute_config());
    let kernel = engine(config, equity(), minute_bars(375, 100.0, &[]));
    let decided = ts_from_ist(order_support::july_2025(1), NSE_OPEN_MINUTE + 374);
    let buy = order(Side::Buy, 1.0, OrderType::Market, Product::Cnc);
    let (output, script) = run_script(kernel, vec![(decided, Act::Submit(buy))]);
    assert!(fills(&output).is_empty());
    assert!(matches!(
        script.submissions[0],
        Submission::Rejected(_, code) if code.code() == "delay_past_close"
    ));
}
