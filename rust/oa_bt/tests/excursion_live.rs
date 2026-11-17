#![cfg(all(feature = "costs", feature = "time"))]
//! A live session's MAE and MFE (`engine::excursion`, `engine::excursion_hold`): every
//! tick taken while a trade is open counts, besides the bars handed over, but none once a
//! price has reached a working order on the trade's closing side (a broker-held stop, a
//! limit, a MARKET exit), until the venue reports that order filled or ended: the venue
//! may have closed the position before it reports so. An order whose modification waits
//! for the venue's answer is judged by its old terms and by its new ones until then. A
//! bar is judged by the prices its ticks did not carry, and the bar of the minute in
//! which the venue confirmed a modification by the terms it replaced as well. And a fill
//! the venue priced has no slippage modelled.
//!
//! 100 SBIN in MIS bought by the venue at 500.10 at the open of minute 1; minute k's bar
//! opens at 500.00 + 0.10 k, its high 0.50 above the open, its low 0.50 below and its
//! close 0.05 above (minute 1: 500.10, 500.60, 499.60, 500.15), unless a test gives
//! another; the position is sold by the venue at 502.00 thirty seconds into minute 2, or
//! by its stop at 497.10.

mod venue_support;

use oa_bt::engine::{Engine, RunOutput, StateValue, TradeRecord, VenueCommand};
use oa_bt::exec::{OrderId, OrderRequest, OrderType};
use oa_bt::model::Side;
use oa_bt::rms::{Distance, LegRules, TrailSpec};
use venue_support::{
    at_close, at_second, bars, config, entry_waiting, exit, holding_with, limit, market, minute,
    modified, placed_roles, run_to_end, sbin, session, stop_market, stop_points, wait_before, Act,
    Script, SBIN,
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

/// A session restarted, caught up to the wait before minute `minute`, and given `state`;
/// the new id of the saved order o2.
fn restored(state: &StateValue, minute: usize) -> (Engine, OrderId) {
    let mut kernel = session(config());
    let mut script = Script::new(Vec::new());
    wait_before(&mut kernel, &bars(&[], &[]), &mut script, minute);
    let pairs = kernel
        .restore_live_state(state)
        .expect("the state restores");
    let (_, order) = pairs
        .into_iter()
        .find(|(saved, _)| *saved == OrderId(2))
        .expect("o2 is restored");
    (kernel, order)
}

#[test]
fn a_restart_keeps_the_ticks_held_back_after_one_reached_the_stop() {
    // As above, with a restart between the tick through the stop and the bounce: the
    // bounce at 505.00 (+490) does not count after the restart either.
    let saved = stopped_with_ticks(&[501.5, 496.0]).live_state();
    let (mut kernel, stop) = restored(&saved, 1);
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

/// A stop 3 points from the fill that trails 2 points behind the best price.
fn trailing() -> LegRules {
    LegRules {
        trail: Some(TrailSpec {
            trigger: Distance::Points(2.0),
            step: None,
        }),
        ..stop_points(3.0)
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
    let mut kernel = holding_with(100.0, trailing());
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

/// The ticks `ticks` of minute 1, each (second, price).
fn take_ticks(kernel: &mut Engine, ticks: &[(i64, f64)]) {
    for (second, price) in ticks {
        kernel
            .push_tick(SBIN, *price, at_second(1, *second))
            .expect("the tick is taken");
    }
}

/// 100 SBIN held with the trailing stop (o2) at 497.10, and a tick at 504.00 (+390) one
/// second into minute 1 that moves it to 502.00: the modification is sent and the venue
/// has not answered it.
fn trail_in_flight() -> Engine {
    let mut kernel = holding_with(100.0, trailing());
    kernel.take_commands();
    take_ticks(&mut kernel, &[(1, 504.0)]);
    let sent = modified(&kernel.take_commands());
    assert_eq!(sent, [(OrderId(2), 100.0, stop_market(502.0))]);
    kernel
}

#[test]
fn a_tick_through_the_terms_of_a_stops_modification_in_flight_holds_the_trade_back() {
    // A tick at 501.50 crosses the stop's new trigger, 502.00, before the venue confirms
    // the move: the venue may have applied it and filled the stop, so neither 501.50 nor
    // the bounce at 510.00 (+990), before or after the confirmation, counts. The stop
    // fills at 501.50 (+140); the tick at 504.00 gives the MFE.
    for bounce_before_confirmation in [false, true] {
        let mut kernel = trail_in_flight();
        take_ticks(&mut kernel, &[(2, 501.5)]);
        if bounce_before_confirmation {
            take_ticks(&mut kernel, &[(3, 510.0)]);
        }
        kernel.apply_modify(OrderId(2)).expect("the venue confirms");
        if !bounce_before_confirmation {
            take_ticks(&mut kernel, &[(3, 510.0)]);
        }
        kernel
            .apply_fill(OrderId(2), 100.0, 501.5, at_second(1, 4))
            .expect("the stop fills");
        assert_eq!(
            excursion(&only_trade(kernel)),
            (0.0, 390.0, 140.0),
            "bounce before the confirmation: {bounce_before_confirmation}"
        );
    }
}

#[test]
fn a_tick_between_the_old_and_the_new_trigger_of_a_modification_in_flight_does_not_count() {
    // A tick at 499.00 (-110), above the stop's old trigger 497.10 and below its new one
    // 502.00, comes before the venue confirms the move: the stop may have filled already.
    // The venue confirms, and the stop fills at 501.80 (+170).
    let mut kernel = trail_in_flight();
    take_ticks(&mut kernel, &[(2, 499.0)]);
    kernel.apply_modify(OrderId(2)).expect("the venue confirms");
    kernel
        .apply_fill(OrderId(2), 100.0, 501.8, at_second(1, 3))
        .expect("the stop fills");
    assert_eq!(excursion(&only_trade(kernel)), (0.0, 390.0, 170.0));
}

#[test]
fn a_short_is_held_back_by_the_terms_of_its_stops_modification_in_flight() {
    // 100 SBIN sold short at 500.10 with the broker's buy stop (o2) at 503.10 trailing 2
    // points: a tick at 496.00 (+410) moves it to 498.00. A tick at 498.50 crosses 498.00
    // before the venue confirms the move, so neither it (+160) nor the fall to 490.00
    // (+1,010) after the confirmation counts. The stop fills at 498.50 (+160).
    let request = OrderRequest {
        leg_rules: trailing(),
        ..market(Side::Sell, 100.0)
    };
    let mut kernel = session(config());
    let mut script = Script::new(vec![at_close(0, Act::Submit(request))]);
    wait_before(&mut kernel, &bars(&[], &[]), &mut script, 1);
    kernel.take_commands();
    kernel
        .apply_fill(OrderId(1), 100.0, 500.1, minute(1))
        .expect("the sell fills");
    kernel.take_commands();
    take_ticks(&mut kernel, &[(1, 496.0)]);
    let sent = modified(&kernel.take_commands());
    assert_eq!(sent, [(OrderId(2), 100.0, stop_market(498.0))]);
    take_ticks(&mut kernel, &[(2, 498.5)]);
    kernel.apply_modify(OrderId(2)).expect("the venue confirms");
    take_ticks(&mut kernel, &[(3, 490.0)]);
    kernel
        .apply_fill(OrderId(2), 100.0, 498.5, at_second(1, 4))
        .expect("the stop fills");
    assert_eq!(excursion(&only_trade(kernel)), (0.0, 410.0, 160.0));
}

/// `trail_in_flight` with ticks at 501.50, through the new trigger 502.00, and 506.00
/// (+590) two and three seconds into minute 1, before the venue answers.
fn crossed_in_flight() -> Engine {
    let mut kernel = trail_in_flight();
    take_ticks(&mut kernel, &[(2, 501.5), (3, 506.0)]);
    kernel
}

/// The venue refuses the stop's move (`stop`), then ticks at 505.00 (+490) and 496.00,
/// through the old trigger 497.10, and the stop fills at 497.10 (-300).
fn refused_then_stopped(mut kernel: Engine, stop: OrderId) -> TradeRecord {
    kernel
        .apply_modify_refused(stop)
        .expect("the venue refuses");
    take_ticks(&mut kernel, &[(4, 505.0), (5, 496.0)]);
    kernel
        .apply_fill(stop, 100.0, 497.1, at_second(1, 6))
        .expect("the stop fills");
    only_trade(kernel)
}

#[test]
fn a_refused_modification_leaves_the_trade_judged_by_the_old_terms_again() {
    // Ticks at 501.50 and 506.00 come while the stop's move to 502.00 is in flight: the
    // first crosses 502.00, so neither counts. The venue refuses the move: the stop worked
    // at 497.10 throughout, which neither tick reached, so the trade is judged by 497.10
    // again: the tick at 505.00 (+490) counts, the one at 496.00 reaches 497.10 and does
    // not.
    let trade = refused_then_stopped(crossed_in_flight(), OrderId(2));
    assert_eq!(excursion(&trade), (-300.0, 490.0, -300.0));
}

#[test]
fn a_restart_keeps_a_trade_held_back_by_a_modification_in_flight_until_the_venue_answers() {
    // As above, with a restart between the tick at 506.00 and the venue's refusal: the
    // restored session holds the trade back by the move in flight, and the refusal
    // releases it as it would have without the restart.
    let saved = crossed_in_flight().live_state();
    let (kernel, stop) = restored(&saved, 1);
    let trade = refused_then_stopped(kernel, stop);
    assert_eq!(excursion(&trade), (-300.0, 490.0, -300.0));
}

#[test]
fn a_bar_price_the_ticks_after_a_mid_bar_entry_did_not_carry_holds_the_trade_back() {
    // The buy (o1) fills at 500.10 thirty seconds into minute 1 and the broker's stop
    // (o2) works at 497.10. Ticks after the fill at 500.50 (+40) and 501.00 (+90) count.
    // Minute 1's bar (500.00, 501.00, 496.00, 500.80) has a low the ticks did not carry:
    // a tick feed does not carry every traded price, and that low may have traded after
    // the fill, through the stop, as well as before it. So the trade is held back until
    // the venue reports its closing orders ended: minute 2's ticks at 510.00 (+990),
    // 520.00 (+1,990) and 500.30, and its bar, count neither while the stop works nor
    // while the sell (o3) decided at the close of minute 2 works. The sell fills at 500.20
    // (+10). An under-claim if the low came before the fill, never an over-claim.
    let mut kernel = entry_waiting(sbin(), 100.0, stop_points(3.0));
    kernel
        .apply_fill(OrderId(1), 100.0, 500.1, at_second(1, 30))
        .expect("the buy fills");
    kernel.take_commands();
    for (second, price) in [(31, 500.5), (40, 501.0)] {
        kernel
            .push_tick(SBIN, price, at_second(1, second))
            .expect("the tick is taken");
    }
    let minute_1 = (500.0, 501.0, 496.0, 500.8);
    let sell = Act::Submit(exit(Side::Sell, 100.0, OrderType::Market));
    let mut script = Script::new(vec![at_close(2, sell)]);
    wait_before(&mut kernel, &bars(&[(1, minute_1)], &[]), &mut script, 2);
    for (second, price) in [(10, 510.0), (20, 520.0), (40, 500.3)] {
        kernel
            .push_tick(SBIN, price, at_second(2, second))
            .expect("the tick is taken");
    }
    let minute_2 = (500.8, 520.0, 500.3, 500.3);
    let both = bars(&[(1, minute_1), (2, minute_2)], &[]);
    wait_before(&mut kernel, &both, &mut script, 3);
    let cancels = kernel.take_commands();
    assert!(cancels.contains(&VenueCommand::Cancel { order: OrderId(2) }));
    kernel.apply_cancel(OrderId(2)).expect("the venue cancels");
    let sells = kernel.take_commands();
    assert!(matches!(
        sells.as_slice(),
        [VenueCommand::Place {
            order: OrderId(3),
            side: Side::Sell,
            ..
        }]
    ));
    kernel
        .apply_fill(OrderId(3), 100.0, 500.2, at_second(3, 1))
        .expect("the sell fills");
    assert_eq!(excursion(&only_trade(kernel)), (0.0, 90.0, 10.0));
}

/// A closing order of the strategy moved further away from the price, and part filled on
/// its old terms while the move was at the venue.
struct Loosened {
    /// The sell for the whole (o2), decided at the close of minute 1.
    first: OrderType,
    /// Its terms after the move decided at the close of minute 2.
    moved: OrderType,
    /// The trade's one tick of minute 3, five seconds in, before the venue confirms.
    tick: f64,
    /// Minute 3's bar, handed over after the confirmation.
    minute_3: (f64, f64, f64, f64),
    /// The price of the 40 the venue sold on the old terms, reported one second into
    /// minute 4.
    part_fill: f64,
}

/// The trade of `case`, the session restarted between the venue's confirmation and minute
/// 3's bar when `restart`. After the part fill the venue cancels the rest of o2 of its own
/// accord, and the strategy sells the 60 left by MARKET at the close of minute 4, filled at
/// 500.50 one second into minute 5.
fn loosened_then_part_filled(case: &Loosened, restart: bool) -> TradeRecord {
    let mut kernel = holding_with(100.0, LegRules::default());
    kernel.take_commands();
    let first = Act::Submit(exit(Side::Sell, 100.0, case.first));
    let moved = Act::Modify(OrderId(2), exit(Side::Sell, 100.0, case.moved));
    let mut script = Script::new(vec![at_close(1, first), at_close(2, moved)]);
    wait_before(&mut kernel, &bars(&[], &[]), &mut script, 2);
    kernel.take_commands();
    wait_before(&mut kernel, &bars(&[], &[]), &mut script, 3);
    let sent = modified(&kernel.take_commands());
    assert_eq!(sent, [(OrderId(2), 100.0, case.moved)]);
    kernel
        .push_tick(SBIN, case.tick, at_second(3, 5))
        .expect("the tick is taken");
    kernel.apply_modify(OrderId(2)).expect("the venue confirms");
    let (mut kernel, order) = match restart {
        true => restored(&kernel.live_state(), 3),
        false => (kernel, OrderId(2)),
    };
    let rest = Act::Submit(exit(Side::Sell, 60.0, OrderType::Market));
    let mut script = Script::new(vec![at_close(4, rest)]);
    let minute_3 = bars(&[(3, case.minute_3)], &[]);
    wait_before(&mut kernel, &minute_3, &mut script, 4);
    kernel
        .apply_fill(order, 40.0, case.part_fill, at_second(4, 1))
        .expect("the venue reports the part sold");
    kernel
        .apply_cancel(order)
        .expect("the venue cancels the rest");
    wait_before(&mut kernel, &minute_3, &mut script, 5);
    let placed = placed_roles(&kernel.take_commands());
    assert_eq!(placed.len(), 1, "{placed:?}");
    kernel
        .apply_fill(placed[0].0, 60.0, 500.5, at_second(5, 1))
        .expect("the rest is sold");
    only_trade(kernel)
}

/// The strategy's sell limit at 503.00 moved out to 505.00; minute 3 ticks at 502.00 and
/// its bar is (502.00, 504.00, 501.50, 503.50); the venue sold 40 at 503.00.
const TARGET_MOVED_OUT: Loosened = Loosened {
    first: OrderType::Limit {
        limit_rupees: 503.0,
    },
    moved: OrderType::Limit {
        limit_rupees: 505.0,
    },
    tick: 502.0,
    minute_3: (502.0, 504.0, 501.5, 503.5),
    part_fill: 503.0,
};

#[test]
fn a_bar_reaching_the_terms_a_modification_confirmed_inside_it_replaced_does_not_count() {
    // The strategy's sell limit (o2) at 503.00 is moved out to 505.00. Minute 3 ticks at
    // 502.00 (+190), the venue confirms the move, then minute 3's bar is handed over. Its
    // high 504.00, which no tick carried, reaches 503.00 but not 505.00: it may have traded
    // while 503.00 still worked, and the venue did sell 40 at 503.00 then, which it reports
    // after the bar (+290). So the bar does not count: +390 at its high with the whole
    // position, where the trade saw at most 40 x 2.90 + 60 x 3.90 = 350. The 60 left are
    // sold at 500.50 (+140). Minute 1's bar gives the MAE, -50.
    let trade = loosened_then_part_filled(&TARGET_MOVED_OUT, false);
    assert_eq!(excursion(&trade), (-50.0, 290.0, 140.0));
}

#[test]
fn a_stop_moved_further_away_and_confirmed_inside_a_bar_holds_the_trade_on_its_old_trigger() {
    // The strategy's sell stop (o2) at 497.00 is moved out to 495.00. Minute 3 ticks at
    // 498.00 (-210), the venue confirms the move, then minute 3's bar (498.00, 499.00,
    // 496.00, 498.50) is handed over. Its low 496.00, which no tick carried, reaches 497.00
    // but not 495.00: it may have traded while 497.00 still worked, and the venue did sell
    // 40 at 497.00 then, which it reports after the bar (-310). So the bar does not count:
    // -410 at its low with the whole position, where the trade saw at least 40 x -3.10 +
    // 60 x -4.10 = -370. The 60 left are sold at 500.50 (-100). Minute 2's bar gives the
    // MFE, +60.
    let stop_moved_out = Loosened {
        first: stop_market(497.0),
        moved: stop_market(495.0),
        tick: 498.0,
        minute_3: (498.0, 499.0, 496.0, 498.5),
        part_fill: 497.0,
    };
    let trade = loosened_then_part_filled(&stop_moved_out, false);
    assert_eq!(excursion(&trade), (-310.0, 60.0, -100.0));
}

#[test]
fn a_restart_keeps_the_terms_a_modification_replaced_until_the_bar_is_judged() {
    // The limit moved out to 505.00 above, with a restart between the venue's confirmation
    // and minute 3's bar: the restored session judges the bar by 503.00 too.
    let trade = loosened_then_part_filled(&TARGET_MOVED_OUT, true);
    assert_eq!(excursion(&trade), (-50.0, 290.0, 140.0));
}

#[test]
fn a_trail_confirmed_inside_a_bar_leaves_the_bar_judged_as_before() {
    // The broker's stop (o2) at 497.10 trails 2 points behind the best price. Minute 2
    // ticks at 504.00 (+390; the stop moves to 502.00, which the venue confirms at once)
    // and 503.50 (+340). Its bar (503.00, 504.00, 502.50, 503.50) has a low no tick
    // carried, 502.50 (+240), which reaches neither the new trigger nor the one it
    // replaced: a trail only tightens the stop, so a price that reached its old trigger
    // reaches the new one too, and the bar counts as before. Minute 3 ticks at 506.00
    // (+590; the stop moves to 504.00) and 503.50, through it; the stop fills at 504.00
    // (+390). Were the trade held back on minute 2's bar, the MFE would be 390.
    let mut kernel = holding_with(100.0, trailing());
    kernel.take_commands();
    let minute_2 = bars(&[(2, (503.0, 504.0, 502.5, 503.5))], &[]);
    let mut script = Script::new(Vec::new());
    wait_before(&mut kernel, &minute_2, &mut script, 2);
    for (second, price) in [(1, 504.0), (2, 503.5)] {
        tick_confirmed(&mut kernel, price, at_second(2, second));
    }
    wait_before(&mut kernel, &minute_2, &mut script, 3);
    for (second, price) in [(1, 506.0), (2, 503.5)] {
        tick_confirmed(&mut kernel, price, at_second(3, second));
    }
    kernel
        .apply_fill(OrderId(2), 100.0, 504.0, at_second(3, 3))
        .expect("the stop fills");
    assert_eq!(excursion(&only_trade(kernel)), (-50.0, 590.0, 390.0));
}
