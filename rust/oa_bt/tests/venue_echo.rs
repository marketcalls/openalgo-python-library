#![cfg(all(feature = "costs", feature = "time"))]
//! A live session whose venue fills every order where the backtest's simulator does (the
//! echo venue of `venue_support`) is the backtest: its `RunOutput` equals the backtest's,
//! field for field, with zero slippage and the same costs, but that the venue priced its
//! fills, so none has slippage modelled (`venue_support::venue_priced`).
//!
//! The bars rise 0.10 a minute from 500.00 (`venue_support::rising`), so every fill
//! below can be read off them: open 500.00 + 0.10 k at minute k, high the open plus 0.50,
//! low the open less 0.50.

mod venue_support;

use oa_bt::engine::{run, Pause, RunConfig, RunOutput, StrategyInterval, TimerSpec};
use oa_bt::exec::{FillAt, OrderRequest, OrderType};
use oa_bt::model::{ts_from_ist, Product, Side};
use venue_support::{
    at_close, backtest, bars, charged_config, config, day, drive_echo, exit, limit, market, minute,
    order, run_backtest, run_echo, session, stop_limit, stop_market, venue_priced, Act, Echo, Ohlc,
    Script, MINUTE,
};

/// A gap down at minute 50: it opens at 495.00, through a sell stop at 503.00.
const GAP: (u32, Ohlc) = (50, (495.0, 496.0, 494.0, 495.5));
/// Minutes with no bar: an order decided at minute 40 waits for minute 43 (SPEC-06).
const MISSING: [u32; 2] = [41, 42];

fn mis(side: Side, units: f64, order_type: OrderType) -> OrderRequest {
    order(side, units, order_type, Product::Mis)
}

/// Every order kind and fill of the simulator, decided at the close of minute k:
///
/// | k | Order | Fills |
/// |---|---|---|
/// | 0 | buy 100 MARKET | minute 1 at the open, 500.10 |
/// | 5 | sell 100 LIMIT 501.50, exit | minute 10 at its limit (high 501.50) |
/// | 12 | buy 50 LIMIT 502.00 | minute 13 at the open, 501.30 |
/// | 14 | sell 50 SL-M 501.20, exit | minute 15 at its trigger (low 501.00) |
/// | 20 | sell 30 MARKET | minute 21 at the open, 502.10: short |
/// | 25 | buy 60 MARKET, reverse | minute 26 at the open, 502.60: close 30, open 30 |
/// | 30 | buy 20 SL 503.40, limit 503.60 | minute 31 at its trigger (high 503.60) |
/// | 40 | sell 50 MARKET, exit | minute 43 at the open, 504.30 (no bars at 41, 42) |
/// | 45 | buy 10 MARKET | minute 46 at the open, 504.60 |
/// | 47 | sell 10 SL-M 503.00, exit | minute 50 at the open of the gap, 495.00 |
/// | 57 | buy 10 LIMIT 506.00 | minute 58 at the open, 505.80 |
/// | 65 | sell 10 MARKET, exit | minute 66 at the open, 506.60 |
///
/// A modification and a cancellation are not in it: the venue confirms them, so their
/// events are written when it does (`venue_session.rs`), not at the decision.
fn every_order_kind() -> Vec<(i64, Act)> {
    let reverse = OrderRequest {
        reverse: true,
        ..market(Side::Buy, 60.0)
    };
    vec![
        at_close(0, Act::Submit(market(Side::Buy, 100.0))),
        at_close(5, Act::Submit(exit(Side::Sell, 100.0, limit(501.5)))),
        at_close(12, Act::Submit(mis(Side::Buy, 50.0, limit(502.0)))),
        at_close(14, Act::Submit(exit(Side::Sell, 50.0, stop_market(501.2)))),
        at_close(20, Act::Submit(market(Side::Sell, 30.0))),
        at_close(25, Act::Submit(reverse)),
        at_close(
            30,
            Act::Submit(mis(Side::Buy, 20.0, stop_limit(503.4, 503.6))),
        ),
        at_close(40, Act::Submit(exit(Side::Sell, 50.0, OrderType::Market))),
        at_close(45, Act::Submit(market(Side::Buy, 10.0))),
        at_close(47, Act::Submit(exit(Side::Sell, 10.0, stop_market(503.0)))),
        at_close(57, Act::Submit(mis(Side::Buy, 10.0, limit(506.0)))),
        at_close(65, Act::Submit(exit(Side::Sell, 10.0, OrderType::Market))),
    ]
}

/// The echoed session and the backtest of the same script on the same bars.
fn both(config: RunConfig, acts: Vec<(i64, Act)>) -> (RunOutput, RunOutput) {
    let columns = bars(&[GAP], &MISSING);
    let backtest = run_backtest(config.clone(), &columns, &mut Script::new(acts.clone()));
    let session = run_echo(
        config,
        &columns,
        &mut Script::new(acts),
        &mut Echo::default(),
    );
    (session, backtest)
}

#[test]
fn a_session_with_an_echo_venue_is_the_backtest_of_its_bars() {
    let (session, backtest) = both(charged_config(), every_order_kind());
    assert_eq!(session, venue_priced(backtest));
}

#[test]
fn the_echoed_session_fills_every_order_kind_where_its_table_says() {
    // The equality above holds for a run that does what its table says, not for one in
    // which nothing fills.
    let (session, _) = both(config(), every_order_kind());
    let fills: Vec<(i64, f64, f64, FillAt)> = session
        .fills
        .iter()
        .map(|r| {
            (
                r.stamp.slice_ts,
                r.fill.quantity_units,
                r.fill.price_rupees,
                r.fill.at,
            )
        })
        .collect();
    let expected = [
        (1, 100.0, 500.1, FillAt::Open),
        (10, 100.0, 501.5, FillAt::Level),
        (13, 50.0, 501.3, FillAt::Open),
        (15, 50.0, 501.2, FillAt::Level),
        (21, 30.0, 502.1, FillAt::Open),
        (26, 30.0, 502.6, FillAt::Open),
        (26, 30.0, 502.6, FillAt::Open),
        (31, 20.0, 503.4, FillAt::Level),
        (43, 50.0, 504.3, FillAt::Open),
        (46, 10.0, 504.6, FillAt::Open),
        (50, 10.0, 495.0, FillAt::Open),
        (58, 10.0, 505.8, FillAt::Open),
        (66, 10.0, 506.6, FillAt::Open),
    ];
    assert_eq!(fills.len(), expected.len());
    for (fill, (k, units, price, at)) in fills.iter().zip(expected) {
        assert_eq!(
            (fill.0, fill.1, fill.3),
            (minute(k), units, at),
            "minute {k}"
        );
        assert!(
            (fill.2 - price).abs() < 1e-9,
            "minute {k}: {} for {price}",
            fill.2
        );
    }
}

/// A 15-minute strategy on 1-minute bars, with charges.
fn fifteen_minutes() -> RunConfig {
    RunConfig {
        strategy_interval: StrategyInterval::Intraday(15 * MINUTE),
        ..charged_config()
    }
}

#[test]
fn a_session_on_a_15_minute_strategy_is_the_backtest_of_its_bars() {
    let acts = vec![
        at_close(14, Act::Submit(market(Side::Buy, 100.0))),
        at_close(44, Act::Submit(exit(Side::Sell, 100.0, OrderType::Market))),
    ];
    let (session, backtest) = both(fifteen_minutes(), acts);
    assert_eq!(session.fills.len(), 2);
    assert_eq!(session, venue_priced(backtest));
}

/// 10:05 IST in minutes since midnight: the open of minute 50, between two 15-minute
/// closes.
const FIVE_PAST_TEN: u32 = 10 * 60 + 5;

#[test]
fn a_timer_fires_when_the_session_reaches_its_open() {
    // The 10:05 timer fires at the close of minute 49, before minute 50 has run: a live
    // session reaches 10:05 then.
    let timer = TimerSpec::At {
        minute_of_day: FIVE_PAST_TEN,
    };
    let mut live = session(fifteen_minutes());
    live.add_timer(timer).expect("a timer before the session");
    let columns = bars(&[], &[]);
    let mut script = Script::new(Vec::new());
    venue_support::wait_before(&mut live, &columns, &mut script, 49);
    live.release(50);
    let fired = live.advance().expect("minute 49 runs");
    let Pause::Timers {
        stamp, decided_at, ..
    } = fired
    else {
        panic!("expected the timer, got {fired:?}");
    };
    assert_eq!(
        (stamp.slice, decided_at),
        (50, ts_from_ist(day(), FIVE_PAST_TEN))
    );
    assert_eq!(live.advance(), Ok(Pause::Waiting { next_slice: 50 }));
}

#[test]
fn a_session_whose_timer_places_an_order_is_the_backtest_of_its_bars() {
    // A timer at 10:05 buys at that open (minute 50), sent as the session reaches it and
    // filled by the venue in minute 50, as a backtest fills it; the exit is decided at the
    // 10:30 close.
    let at_timer = ts_from_ist(day(), FIVE_PAST_TEN);
    let acts = vec![
        (at_timer, Act::Submit(market(Side::Buy, 40.0))),
        at_close(74, Act::Submit(exit(Side::Sell, 40.0, OrderType::Market))),
    ];
    let timer = TimerSpec::At {
        minute_of_day: FIVE_PAST_TEN,
    };
    let columns = bars(&[], &[]);
    let mut simulated = backtest(fifteen_minutes(), columns.clone());
    simulated.add_timer(timer).expect("a timer before the run");
    run(&mut simulated, &mut Script::new(acts.clone())).expect("the backtest completes");
    let backtest = simulated.finish().expect("the backtest finished");
    let mut live = session(fifteen_minutes());
    live.add_timer(timer).expect("a timer before the session");
    let output = drive_echo(live, &columns, &mut Script::new(acts), &mut Echo::default());
    assert_eq!(output.fills.len(), 2);
    assert_eq!(output.fills[0].stamp.slice_ts, at_timer);
    assert_eq!(output, venue_priced(backtest));
}
