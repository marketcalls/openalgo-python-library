#![cfg(all(feature = "costs", feature = "time"))]
//! A live session's MAE and MFE (`engine::excursion`): every tick taken while a trade is
//! open counts, besides the bars handed over, but one through a broker-held stop that the
//! venue has not reported filling yet; and a fill the venue priced has no slippage
//! modelled.
//!
//! 100 SBIN in MIS bought by the venue at 500.10 at the open of minute 1; minute 1's bar
//! is (500.10, 500.60, 499.60, 500.15); the position is sold by the venue at 502.00
//! thirty seconds into minute 2, or by its stop at 497.10.

mod venue_support;

use oa_bt::engine::{Engine, RunOutput};
use oa_bt::exec::{OrderId, OrderType};
use oa_bt::model::Side;
use oa_bt::rms::LegRules;
use venue_support::{
    at_close, at_second, bars, exit, holding_with, run_to_end, stop_points, wait_before, Act,
    Script, SBIN,
};

/// The session above with the ticks `prices` taken in minute 1, one a second.
fn session_with_ticks(prices: &[f64]) -> RunOutput {
    let mut kernel: Engine = holding_with(100.0, LegRules::default());
    kernel.take_commands();
    for (second, price) in (1..).zip(prices) {
        kernel
            .push_tick(SBIN, *price, at_second(1, second))
            .expect("the tick is taken");
    }
    let sell = Act::Submit(exit(Side::Sell, 100.0, OrderType::Market));
    let mut script = Script::new(vec![at_close(1, sell)]);
    wait_before(&mut kernel, &bars(&[], &[]), &mut script, 2);
    kernel.take_commands();
    kernel
        .apply_fill(OrderId(2), 100.0, 502.0, at_second(2, 30))
        .expect("the sell fills");
    run_to_end(&mut kernel, &bars(&[], &[]), &mut script);
    kernel.finish().expect("the session finished")
}

#[test]
fn every_tick_while_the_trade_is_open_counts() {
    // Without ticks: minute 1's bar, held from its open, is -50 at 499.60 and +50 at
    // 500.60; the sell at 502.00 is +190. With ticks at 499.00 (-110) and 503.00 (+290)
    // those count too.
    let table = [
        (vec![], (-50.0, 190.0)),
        (vec![499.0, 503.0], (-110.0, 290.0)),
    ];
    for (ticks, expected) in table {
        let output = session_with_ticks(&ticks);
        assert_eq!(output.trades.len(), 1, "{ticks:?}");
        let trade = output.trades[0];
        assert_eq!((trade.mae_rupees, trade.mfe_rupees), expected, "{ticks:?}");
        assert_eq!(trade.gross_pnl_rupees, 190.0, "{ticks:?}");
    }
}

#[test]
fn a_tick_through_the_brokers_stop_does_not_count_before_the_stop_is_reported_filled() {
    // The leg's stop 3 points below is the broker's (o2 at 497.10). Ticks at 501.50
    // (+140) and 499.00 (-110) count; the tick at 496.00 comes through the stop before
    // the venue reports its fill at 497.10 (-300): the stop may have closed the position
    // already, so 496.00 (-410) does not count.
    let mut kernel = holding_with(100.0, stop_points(3.0));
    kernel.take_commands();
    for (second, price) in [(1, 501.5), (2, 499.0), (3, 496.0)] {
        kernel
            .push_tick(SBIN, price, at_second(1, second))
            .expect("the tick is taken");
    }
    kernel
        .apply_fill(OrderId(2), 100.0, 497.1, at_second(1, 4))
        .expect("the stop fills");
    let mut script = Script::new(Vec::new());
    run_to_end(&mut kernel, &bars(&[], &[]), &mut script);
    let output = kernel.finish().expect("the session finished");
    assert_eq!(output.trades.len(), 1);
    let trade = output.trades[0];
    assert_eq!(
        (trade.mae_rupees, trade.mfe_rupees, trade.gross_pnl_rupees),
        (-300.0, 140.0, -300.0)
    );
}

#[test]
fn a_fill_the_venue_priced_has_no_slippage_modelled() {
    let output = session_with_ticks(&[]);
    let fills: Vec<Option<f64>> = output.fills.iter().map(|f| f.slippage_rupees).collect();
    assert_eq!(fills, [None, None]);
    assert_eq!(output.trades[0].slippage_rupees, None);
}
