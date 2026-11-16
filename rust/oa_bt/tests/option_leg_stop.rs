#![cfg(all(feature = "costs", feature = "time"))]
//! A backtest gives an option leg's stop the stop-limit shape the exchange takes for it
//! (`engine::legs`, `exec::segment::venue_order_type`; 16-session-design R12, PRD
//! 15-phase2-plan step 5): its limit 5 percent beyond the stop, a gap through the limit
//! filling at the next bar. Cash equity keeps its stop-market rule.
//!
//! The strategy buys at the close of minute 0 with a stop 3 points below, so the position
//! opens at 500.10 at the open of minute 1 with its stop at 497.10; the option's limit is 5
//! percent below that, 472.245, down to the tick 472.20. The bars rise 0.10 a minute from
//! 500.00 (`venue_support::rising`) until the rows below override them. No slippage.

mod venue_support;

use oa_bt::engine::{run, Engine, RunOutput, RunSetup};
use oa_bt::exec::{FillAt, FillReason, InstrumentSpec, OrderRequest};
use oa_bt::market::calendar::Calendar;
use oa_bt::model::Side;
use oa_bt::rms::ExitReason;
use venue_support::{
    at_close, bars, config, market, minute, nifty_call, sbin, stop_points, Act, Ohlc, Script,
};

/// A backtest of `instrument` on the rising bars with `overrides`, buying 75 units.
fn backtest_of(instrument: InstrumentSpec, overrides: &[(u32, Ohlc)]) -> RunOutput {
    let setup = RunSetup {
        config: config(),
        calendar: Calendar::new().expect("the embedded calendar tables parse"),
        instruments: vec![instrument],
        bars: vec![bars(overrides, &[])],
    };
    let mut kernel = Engine::new(setup).expect("a valid backtest");
    let buy = OrderRequest {
        leg_rules: stop_points(3.0),
        ..market(Side::Buy, 75.0)
    };
    let mut script = Script::new(vec![at_close(0, Act::Submit(buy))]);
    run(&mut kernel, &mut script).expect("the backtest completes");
    kernel.finish().expect("the backtest finished")
}

/// A row of the stop table: what, the instrument, the bars overridden, and the stop's
/// fill (minute, price, where in the bar).
type StopRow = (
    &'static str,
    InstrumentSpec,
    Vec<(u32, Ohlc)>,
    (u32, f64, FillAt),
);

#[test]
fn an_option_legs_stop_fills_as_a_stop_limit_and_a_gap_through_its_limit_at_the_next_bar() {
    let rows: [StopRow; 5] = [
        (
            "an option's stop reached inside the range fills at the stop",
            nifty_call(),
            vec![(5, (500.5, 501.0, 497.0, 500.6))],
            (5, 497.1, FillAt::Level),
        ),
        (
            "an option gapped between its stop and limit fills at the open",
            nifty_call(),
            vec![(5, (490.0, 491.0, 489.0, 490.5))],
            (5, 490.0, FillAt::Open),
        ),
        (
            "an option gapped beyond its limit fills at the limit when the bar comes back to it",
            nifty_call(),
            vec![(5, (470.0, 473.0, 469.0, 471.0))],
            (5, 472.2, FillAt::Level),
        ),
        (
            "an option gapped beyond its limit for the whole bar fills at the next open",
            nifty_call(),
            vec![
                (5, (470.0, 471.0, 469.0, 470.5)),
                (6, (468.0, 469.0, 467.0, 468.5)),
            ],
            (6, 468.0, FillAt::Open),
        ),
        (
            "a share gapped through its stop fills at the open, as before",
            sbin(),
            vec![
                (5, (470.0, 471.0, 469.0, 470.5)),
                (6, (468.0, 469.0, 467.0, 468.5)),
            ],
            (5, 470.0, FillAt::Open),
        ),
    ];
    for (what, instrument, overrides, (k, price, at)) in rows {
        let output = backtest_of(instrument, &overrides);
        let exit = output.fills.last().expect("the stop's fill");
        assert_eq!(
            (exit.stamp.slice_ts, exit.fill.at, exit.fill.reason),
            (minute(k), at, FillReason::Exit(ExitReason::LegSl)),
            "{what}"
        );
        assert!(
            (exit.fill.price_rupees - price).abs() < 1e-9,
            "{what}: {}",
            exit.fill.price_rupees
        );
        assert_eq!(exit.fill.quantity_units, 75.0, "{what}");
        assert_eq!(output.trades.len(), 1, "{what}");
    }
}

#[test]
fn an_option_stop_gapped_through_its_limit_writes_its_hit_in_the_bar_that_reached_it() {
    let overrides = [
        (5, (470.0, 471.0, 469.0, 470.5)),
        (6, (468.0, 469.0, 467.0, 468.5)),
    ];
    let output = backtest_of(nifty_call(), &overrides);
    let hits: Vec<i64> = venue_support::events(&output)
        .into_iter()
        .filter(|(_, _, name)| name == "leg_sl_hit")
        .map(|(ts, _, _)| ts)
        .collect();
    assert_eq!(hits, [minute(5)]);
}
