#![cfg(all(feature = "costs", feature = "time"))]
//! A live session's MAE and MFE (`engine::excursion`, `engine::excursion_hold`): every
//! tick taken while a trade is open counts, besides the bars handed over, but none once a
//! price has reached a working order on the trade's closing side (a broker-held stop, a
//! limit, a MARKET exit), until the venue reports that order filled or ended: the venue
//! may have closed the position before it reports so. A bar is judged by the prices its
//! ticks did not carry. And a fill the venue priced has no slippage modelled.
//!
//! 100 SBIN in MIS bought by the venue at 500.10 at the open of minute 1; minute 1's bar
//! is (500.10, 500.60, 499.60, 500.15) unless a test gives another; the position is sold
//! by the venue at 502.00 thirty seconds into minute 2, or by its stop at 497.10.

mod venue_support;

use oa_bt::engine::{Engine, RunOutput, StateValue, TradeRecord, VenueCommand};
use oa_bt::exec::{OrderId, OrderType};
use oa_bt::model::Side;
use oa_bt::rms::{Distance, LegRules, TrailSpec};
use venue_support::{
    at_close, at_second, bars, config, exit, holding_with, limit, run_to_end, session, stop_points,
    wait_before, Act, Script, SBIN,
};

/// The one trade of a session run to its end.
fn only_trade(mut kernel: Engine) -> TradeRecord {
    run_to_end(&mut kernel, &bars(&[], &[]), &mut Script::new(Vec::new()));
    let output = kernel.finish().expect("the session finished");
    assert_eq!(output.trades.len(), 1);
    output.trades[0]
}

/// (MAE, MFE, gross P&L) of a trade, rupees.
fn excursion(trade: &TradeRecord) -> (f64, f64, f64) {
    (trade.mae_rupees, trade.mfe_rupees, trade.gross_pnl_rupees)
}

/// 100 SBIN held with the broker's stop (o2) 3 points below the buy, at 497.10, and the
/// ticks `prices` taken in minute 1, one a second.
fn stopped_with_ticks(prices: &[f64]) -> Engine {
    let mut kernel = holding_with(100.0, stop_points(3.0));
    kernel.take_commands();
    for (second, price) in (1..).zip(prices) {
        kernel
            .push_tick(SBIN, *price, at_second(1, second))
            .expect("the tick is taken");
    }
    kernel
}

/// The session above with the ticks `prices` taken in minute 1, one a second.
fn session_with_ticks(prices: &[f64]) -> RunOutput {
    let mut kernel = holding_with(100.0, LegRules::default());
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
    let mut kernel = stopped_with_ticks(&[501.5, 499.0, 496.0]);
    kernel
        .apply_fill(OrderId(2), 100.0, 497.1, at_second(1, 4))
        .expect("the stop fills");
    assert_eq!(excursion(&only_trade(kernel)), (-300.0, 140.0, -300.0));
}

#[test]
fn a_fill_the_venue_priced_has_no_slippage_modelled() {
    let output = session_with_ticks(&[]);
    let fills: Vec<Option<f64>> = output.fills.iter().map(|f| f.slippage_rupees).collect();
    assert_eq!(fills, [None, None]);
    assert_eq!(output.trades[0].slippage_rupees, None);
}

#[test]
fn a_bar_that_reached_the_stop_does_not_count_when_handed_over_before_the_stops_fill() {
    // Minute 1's bar (500.10, 501.50, 495.00, 496.00) reaches the stop at 497.10 and is
    // handed over before the venue reports the stop's fill at 497.10 one second into
    // minute 2 (-300): the stop may have closed the position anywhere in the bar, so
    // neither its high (+140) nor its low (-510) counts. Ticks taken before one reached
    // the stop count: 501.50 (+140) and 499.00 (-110).
    let minute_1 = (500.1, 501.5, 495.0, 496.0);
    let table = [
        (vec![], (-300.0, 0.0, -300.0)),
        (vec![501.5, 499.0, 496.0], (-300.0, 140.0, -300.0)),
    ];
    for (ticks, expected) in table {
        let mut kernel = stopped_with_ticks(&ticks);
        let mut script = Script::new(Vec::new());
        wait_before(&mut kernel, &bars(&[(1, minute_1)], &[]), &mut script, 2);
        kernel
            .apply_fill(OrderId(2), 100.0, 497.1, at_second(2, 1))
            .expect("the stop fills");
        assert_eq!(excursion(&only_trade(kernel)), expected, "{ticks:?}");
    }
}

#[test]
fn a_bar_handed_over_while_a_market_exit_works_does_not_count() {
    // The sell (o2) decided at the close of minute 1 fills at 502.00 at the start of
    // minute 2 (+190), and the venue reports it only after minute 2's bar (502.00,
    // 510.00, 490.00, 505.00) is handed over: the position may have been closed anywhere
    // in that bar, so neither +990 at its high nor -1,010 at its low counts. Minute 1's
    // bar, held through, counts: -50 at 499.60.
    let mut kernel = holding_with(100.0, LegRules::default());
    kernel.take_commands();
    let sell = Act::Submit(exit(Side::Sell, 100.0, OrderType::Market));
    let mut script = Script::new(vec![at_close(1, sell)]);
    wait_before(&mut kernel, &bars(&[], &[]), &mut script, 2);
    kernel.take_commands();
    let minute_2 = (502.0, 510.0, 490.0, 505.0);
    wait_before(&mut kernel, &bars(&[(2, minute_2)], &[]), &mut script, 3);
    kernel
        .apply_fill(OrderId(2), 100.0, 502.0, at_second(3, 1))
        .expect("the sell fills");
    assert_eq!(excursion(&only_trade(kernel)), (-50.0, 190.0, 190.0));
}

#[test]
fn no_tick_counts_after_one_reached_the_stop_until_the_venue_reports_it() {
    // Ticks at 501.50 (+140), 496.00 (through the stop at 497.10) and 505.00 (a bounce,
    // +490), then the stop's fill at 497.10 (-300): the stop may have filled at the
    // second tick, so the bounce may have come after the position was closed.
    let mut kernel = stopped_with_ticks(&[501.5, 496.0, 505.0]);
    kernel
        .apply_fill(OrderId(2), 100.0, 497.1, at_second(1, 4))
        .expect("the stop fills");
    assert_eq!(excursion(&only_trade(kernel)), (-300.0, 140.0, -300.0));
}

#[test]
fn ticks_count_again_once_the_venue_ends_unfilled_the_closing_order_they_reached() {
    // A sell limit at 503.00 (o2) decided at the close of minute 1 works in minute 2. A
    // tick at 503.00 reaches it and one at 502.80 (+270) follows: the limit may have
    // filled, so neither counts. The venue then cancels o2 of its own accord, unfilled,
    // and a tick at 502.50 (+240) counts. Minute 2 has no bar; minute 3's (500.30,
    // 500.80, 499.80, 500.35) counts, and the sell (o3) decided at its close fills at
    // 501.00 (+90). Minute 1's bar gives the MAE, -50.
    let mut kernel = holding_with(100.0, LegRules::default());
    kernel.take_commands();
    let target = Act::Submit(exit(Side::Sell, 100.0, limit(503.0)));
    let sell = Act::Submit(exit(Side::Sell, 100.0, OrderType::Market));
    let mut script = Script::new(vec![at_close(1, target), at_close(3, sell)]);
    let no_minute_2 = bars(&[], &[2]);
    wait_before(&mut kernel, &no_minute_2, &mut script, 2);
    kernel.take_commands();
    for (second, price) in [(1, 503.0), (2, 502.8)] {
        kernel
            .push_tick(SBIN, price, at_second(2, second))
            .expect("the tick is taken");
    }
    kernel.apply_cancel(OrderId(2)).expect("the venue cancels");
    kernel
        .push_tick(SBIN, 502.5, at_second(2, 3))
        .expect("the tick is taken");
    wait_before(&mut kernel, &no_minute_2, &mut script, 4);
    kernel.take_commands();
    kernel
        .apply_fill(OrderId(3), 100.0, 501.0, at_second(4, 1))
        .expect("the sell fills");
    assert_eq!(excursion(&only_trade(kernel)), (-50.0, 240.0, 90.0));
}

/// A session restarted, caught up to the wait before minute 1, and given `state`; the
/// new id of the saved order o2.
fn restored(state: &StateValue) -> (Engine, OrderId) {
    let mut kernel = session(config());
    let mut script = Script::new(Vec::new());
    wait_before(&mut kernel, &bars(&[], &[]), &mut script, 1);
    let pairs = kernel
        .restore_live_state(state)
        .expect("the state restores");
    let (_, stop) = pairs
        .into_iter()
        .find(|(saved, _)| *saved == OrderId(2))
        .expect("the stop is restored");
    (kernel, stop)
}

#[test]
fn a_restart_keeps_the_ticks_held_back_after_one_reached_the_stop() {
    // As above, with a restart between the tick through the stop and the bounce: the
    // bounce at 505.00 (+490) does not count after the restart either.
    let saved = stopped_with_ticks(&[501.5, 496.0]).live_state();
    let (mut kernel, stop) = restored(&saved);
    kernel
        .push_tick(SBIN, 505.0, at_second(1, 3))
        .expect("the tick is taken");
    kernel
        .apply_fill(stop, 100.0, 497.1, at_second(1, 4))
        .expect("the stop fills");
    assert_eq!(excursion(&only_trade(kernel)), (-300.0, 140.0, -300.0));
}

/// The tick `price` at `at`, the venue confirming at once each modification it makes.
fn tick_confirmed(kernel: &mut Engine, price: f64, at: i64) {
    kernel
        .push_tick(SBIN, price, at)
        .expect("the tick is taken");
    for command in kernel.take_commands() {
        if let VenueCommand::Modify { order, .. } = command {
            kernel.apply_modify(order).expect("the venue confirms");
        }
    }
}

#[test]
fn a_bar_its_ticks_carried_counts_though_the_stop_trailed_above_its_low_inside_it() {
    // The broker's stop (o2) starts at 497.10 and trails 2 points behind the best price.
    // Minute 1 ticks at 499.00 (-110), 504.00 (+390; the stop moves to 502.00) and 503.00;
    // its bar (500.10, 504.00, 499.00, 503.00) reached 502.00 only at its low, before the
    // move, and its ticks carried its low and high, so it holds nothing back. Minute 2
    // ticks at 506.00 (+590; the stop moves to 504.00) and 503.50, through the stop,
    // which fills at 504.00 (+390).
    let rules = LegRules {
        trail: Some(TrailSpec {
            trigger: Distance::Points(2.0),
            step: None,
        }),
        ..stop_points(3.0)
    };
    let mut kernel = holding_with(100.0, rules);
    kernel.take_commands();
    for (second, price) in [(1, 499.0), (2, 504.0), (3, 503.0)] {
        tick_confirmed(&mut kernel, price, at_second(1, second));
    }
    let minute_1 = (500.1, 504.0, 499.0, 503.0);
    let mut script = Script::new(Vec::new());
    wait_before(&mut kernel, &bars(&[(1, minute_1)], &[]), &mut script, 2);
    for (second, price) in [(1, 506.0), (2, 503.5)] {
        tick_confirmed(&mut kernel, price, at_second(2, second));
    }
    let stop = kernel.order(OrderId(2)).map(|o| o.request.order_type);
    assert_eq!(
        stop,
        Some(OrderType::StopMarket {
            trigger_rupees: 504.0
        })
    );
    kernel
        .apply_fill(OrderId(2), 100.0, 504.0, at_second(2, 3))
        .expect("the stop fills");
    assert_eq!(excursion(&only_trade(kernel)), (-110.0, 590.0, 390.0));
}
