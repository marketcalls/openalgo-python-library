#![cfg(all(feature = "costs", feature = "time"))]
//! A live session's MAE and MFE around a modification the venue confirmed
//! (`engine::excursion_hold`): the terms it replaced may have filled part of the order
//! before the confirmation, and the venue may report that fill late. So every tick of the
//! slice the confirmation is recorded in, before it and after it, is judged against the
//! replaced terms too, and so are the ticks and the bar of the slice after it: the kernel
//! records a confirmation in the slice in progress, which in a live session lags real time
//! by the close grace (the bar of a minute is handed over a second after it closes).
//!
//! 100 SBIN in MIS bought by the venue at 500.10 at the open of minute 1; minute k's bar
//! opens at 500.00 + 0.10 k, its high 0.50 above the open, its low 0.50 below and its close
//! 0.05 above (minute 1: 500.10, 500.60, 499.60, 500.15; minute 2: 500.20, 500.70, 499.70,
//! 500.25), unless a test gives another. Minutes 1 and 2 give the trade -50 and +60 at
//! their extremes. A running P&L is gross, rupees: 100 x (price - 500.10) while the 100 are
//! held, 40 x 2.90 = 116.00 realized plus 60 x (price - 500.10) once 40 are sold at 503.00.

mod venue_support;

use oa_bt::engine::{Engine, TradeRecord};
use oa_bt::exec::{OrderId, OrderType};
use oa_bt::model::Side;
use oa_bt::rms::{Distance, LegRules, TrailSpec};
use venue_support::{
    at_close, at_second, bars, exit, holding_with, limit, modified, placed_roles, run_to_end,
    stop_market, stop_points, wait_before, Act, Script, SBIN,
};

/// One bar: (open, high, low, close), rupees.
type Ohlc = (f64, f64, f64, f64);

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

/// A closing order of the strategy moved further away from the price, confirmed by the
/// venue inside minute 3, and part filled on its old terms while the move was at the venue.
struct Loosened {
    /// The sell for the whole (o2), decided at the close of minute 1.
    first: OrderType,
    /// Its terms after the move decided at the close of minute 2.
    moved: OrderType,
    /// The trade's ticks of minute 3 before the venue confirms: (second, price).
    before: &'static [(i64, f64)],
    /// The trade's ticks of minute 3 after the confirmation.
    after: &'static [(i64, f64)],
    /// Minute 3's bar, handed over after the ticks.
    minute_3: Ohlc,
    /// The price of the 40 the venue sold on the old terms, reported one second into
    /// minute 4.
    part_fill: f64,
}

/// The trade of `case`. After the part fill the venue cancels the rest of o2 of its own
/// accord, and the strategy sells the 60 left by MARKET at the close of minute 4, filled at
/// 500.50 one second into minute 5. Minute 4's bar is 500.40, 500.90, 499.90, 500.45: +164
/// at its high and +104 at its low with 60 held; the last sale +140.
fn loosened(case: &Loosened) -> TradeRecord {
    let mut kernel = holding_with(100.0, LegRules::default());
    kernel.take_commands();
    let first = Act::Submit(exit(Side::Sell, 100.0, case.first));
    let moved = Act::Modify(OrderId(2), exit(Side::Sell, 100.0, case.moved));
    let rest = Act::Submit(exit(Side::Sell, 60.0, OrderType::Market));
    let acts = vec![at_close(1, first), at_close(2, moved), at_close(4, rest)];
    let mut script = Script::new(acts);
    wait_before(&mut kernel, &bars(&[], &[]), &mut script, 2);
    kernel.take_commands();
    wait_before(&mut kernel, &bars(&[], &[]), &mut script, 3);
    let sent = modified(&kernel.take_commands());
    assert_eq!(sent, [(OrderId(2), 100.0, case.moved)]);
    take_ticks(&mut kernel, 3, case.before);
    kernel.apply_modify(OrderId(2)).expect("the venue confirms");
    take_ticks(&mut kernel, 3, case.after);
    let minute_3 = bars(&[(3, case.minute_3)], &[]);
    wait_before(&mut kernel, &minute_3, &mut script, 4);
    kernel
        .apply_fill(OrderId(2), 40.0, case.part_fill, at_second(4, 1))
        .expect("the venue reports the part sold");
    kernel
        .apply_cancel(OrderId(2))
        .expect("the venue cancels the rest");
    wait_before(&mut kernel, &minute_3, &mut script, 5);
    sell_the_rest(&mut kernel, 5);
    only_trade(kernel)
}

/// The ticks `prices` of minute `k`: (second, price).
fn take_ticks(kernel: &mut Engine, k: u32, prices: &[(i64, f64)]) {
    for &(second, price) in prices {
        kernel
            .push_tick(SBIN, price, at_second(k, second))
            .expect("the tick is taken");
    }
}

/// The strategy's MARKET sale of the 60 left, placed now, filled at 500.50 one second into
/// minute `k`.
fn sell_the_rest(kernel: &mut Engine, k: u32) {
    let placed = placed_roles(&kernel.take_commands());
    assert_eq!(placed.len(), 1, "{placed:?}");
    kernel
        .apply_fill(placed[0].0, 60.0, 500.5, at_second(k, 1))
        .expect("the rest is sold");
}

#[test]
fn a_tick_after_a_confirmation_that_reaches_the_replaced_limit_does_not_count() {
    // The strategy's sell limit (o2) at 503.00 is moved out to 505.00. Minute 3 ticks at
    // 502.00 (+190), the venue confirms the move, then minute 3 ticks at 504.50, between
    // the old limit and the new. The venue sold 40 at 503.00 on the old terms, before the
    // confirmation and so before the 504.50 tick: the trade then held 60, and saw at most
    // 40 x 2.90 + 60 x 4.40 = 380, not the +440 of the whole position at 504.50. So that
    // tick, which reaches the replaced 503.00, holds the trade back until the venue reports
    // o2 ended. Minute 3's bar (502.00, 504.50, 501.50, 504.00) adds its low only, while
    // held. The part fill counts at 503.00 (+290).
    let case = Loosened {
        first: limit(503.0),
        moved: limit(505.0),
        before: &[(5, 502.0)],
        after: &[(30, 504.5)],
        minute_3: (502.0, 504.5, 501.5, 504.0),
        part_fill: 503.0,
    };
    assert_eq!(excursion(&loosened(&case)), (-50.0, 290.0, 140.0));
}

#[test]
fn a_tick_after_a_confirmation_that_reaches_the_replaced_stop_does_not_count() {
    // The strategy's sell stop (o2) at 497.00 is moved out to 495.00. Minute 3 ticks at
    // 498.00 (-210), the venue confirms the move, then minute 3 ticks at 496.00, between
    // the old trigger and the new. The venue sold 40 at 497.00 on the old trigger before
    // the confirmation: the trade saw no worse than 40 x -3.10 + 60 x -4.10 = -370, not the
    // -410 of the whole position at 496.00. The part fill counts at 497.00 (-310); minute
    // 4's bar gives -76 and -136 with 60 held, the last sale -100.
    let case = Loosened {
        first: stop_market(497.0),
        moved: stop_market(495.0),
        before: &[(5, 498.0)],
        after: &[(30, 496.0)],
        minute_3: (498.0, 499.0, 496.0, 498.5),
        part_fill: 497.0,
    };
    assert_eq!(excursion(&loosened(&case)), (-310.0, 60.0, -100.0));
}

#[test]
fn a_tick_before_the_confirmation_that_reaches_the_working_limit_holds_the_trade_back() {
    // The control: the 504.50 tick comes before the confirmation, while 503.00 works.
    let case = Loosened {
        first: limit(503.0),
        moved: limit(505.0),
        before: &[(5, 502.0), (30, 504.5)],
        after: &[],
        minute_3: (502.0, 504.5, 501.5, 504.0),
        part_fill: 503.0,
    };
    assert_eq!(excursion(&loosened(&case)), (-50.0, 290.0, 140.0));
}

#[test]
fn the_bar_after_the_slice_of_a_confirmation_is_judged_by_the_replaced_terms_too() {
    // The venue's confirmation of the move of o2's limit from 503.00 to 505.00 reaches the
    // kernel while it waits before slice 3, after a tick at 502.00 (+190): it is recorded
    // in slice 3, though the venue may have applied it in the first moments of minute 4,
    // the bar of minute 3 being handed over a second after its close. Minute 3's bar
    // (502.00, 502.50, 501.50, 502.20) reaches neither limit and counts (+240 at its high,
    // +140 at its low). Minute 4's (502.20, 504.00, 501.80, 503.00), with no tick, reaches
    // the old 503.00 only: if 504.00 traded before the venue applied the move, the trade
    // saw at most 40 x 2.90 + 60 x 3.90 = 350, not the +390 of the whole position. So it
    // does not count. The venue reports 40 sold at 503.00 one second into minute 5 (+290)
    // and cancels the rest; minute 5's bar (500.50, 501.00, 500.00, 500.55) gives +170 and
    // +110 with 60 held, and the 60 are sold at 500.50 one second into minute 6 (+140).
    let mut kernel = holding_with(100.0, LegRules::default());
    kernel.take_commands();
    let first = Act::Submit(exit(Side::Sell, 100.0, limit(503.0)));
    let moved = Act::Modify(OrderId(2), exit(Side::Sell, 100.0, limit(505.0)));
    let rest = Act::Submit(exit(Side::Sell, 60.0, OrderType::Market));
    let acts = vec![at_close(1, first), at_close(2, moved), at_close(5, rest)];
    let mut script = Script::new(acts);
    wait_before(&mut kernel, &bars(&[], &[]), &mut script, 3);
    kernel.take_commands();
    take_ticks(&mut kernel, 3, &[(5, 502.0)]);
    kernel.apply_modify(OrderId(2)).expect("the venue confirms");
    let minutes_3_and_4 = bars(
        &[
            (3, (502.0, 502.5, 501.5, 502.2)),
            (4, (502.2, 504.0, 501.8, 503.0)),
        ],
        &[],
    );
    wait_before(&mut kernel, &minutes_3_and_4, &mut script, 5);
    kernel
        .apply_fill(OrderId(2), 40.0, 503.0, at_second(5, 1))
        .expect("the venue reports the part sold");
    kernel
        .apply_cancel(OrderId(2))
        .expect("the venue cancels the rest");
    wait_before(&mut kernel, &minutes_3_and_4, &mut script, 6);
    sell_the_rest(&mut kernel, 6);
    assert_eq!(excursion(&only_trade(kernel)), (-50.0, 290.0, 140.0));
}

#[test]
fn a_tick_of_the_slice_after_a_confirmation_is_judged_by_the_replaced_terms_too() {
    // As above, with no bar of minute 4 but a tick at 504.00 four seconds into it: it
    // reaches the replaced 503.00 only, and may have traded before the venue applied the
    // move, so it does not count.
    let mut kernel = holding_with(100.0, LegRules::default());
    kernel.take_commands();
    let first = Act::Submit(exit(Side::Sell, 100.0, limit(503.0)));
    let moved = Act::Modify(OrderId(2), exit(Side::Sell, 100.0, limit(505.0)));
    let rest = Act::Submit(exit(Side::Sell, 60.0, OrderType::Market));
    let acts = vec![at_close(1, first), at_close(2, moved), at_close(5, rest)];
    let mut script = Script::new(acts);
    wait_before(&mut kernel, &bars(&[], &[]), &mut script, 3);
    kernel.take_commands();
    take_ticks(&mut kernel, 3, &[(5, 502.0)]);
    kernel.apply_modify(OrderId(2)).expect("the venue confirms");
    let minute_3_only = bars(&[(3, (502.0, 502.5, 501.5, 502.2))], &[4]);
    wait_before(&mut kernel, &minute_3_only, &mut script, 4);
    take_ticks(&mut kernel, 4, &[(4, 504.0)]);
    wait_before(&mut kernel, &minute_3_only, &mut script, 5);
    kernel
        .apply_fill(OrderId(2), 40.0, 503.0, at_second(5, 1))
        .expect("the venue reports the part sold");
    kernel
        .apply_cancel(OrderId(2))
        .expect("the venue cancels the rest");
    wait_before(&mut kernel, &minute_3_only, &mut script, 6);
    sell_the_rest(&mut kernel, 6);
    assert_eq!(excursion(&only_trade(kernel)), (-50.0, 290.0, 140.0));
}

#[test]
fn a_trail_confirmed_in_a_slice_leaves_the_ticks_of_the_next_slice_judged_as_before() {
    // The broker's stop (o2) at 497.10 trails 2 points behind the best price. Minute 2, which
    // has no bar, ticks at 504.00 (+390; the stop moves to 502.00, which the venue confirms
    // at once). Minute 3
    // ticks at 503.00 (+290): it reaches neither the new trigger nor the replaced 497.10,
    // kept for the slice after the confirmation, since a trail only tightens the stop. Then
    // 506.00 (+590; the stop moves to 504.00) and 503.50, through it; the stop fills at
    // 504.00 (+390).
    let mut kernel = holding_with(100.0, trailing());
    kernel.take_commands();
    let mut script = Script::new(Vec::new());
    wait_before(&mut kernel, &bars(&[], &[2]), &mut script, 2);
    tick_confirmed(&mut kernel, 504.0, at_second(2, 1));
    wait_before(&mut kernel, &bars(&[], &[2]), &mut script, 3);
    for (second, price) in [(1, 503.0), (2, 506.0), (3, 503.5)] {
        tick_confirmed(&mut kernel, price, at_second(3, second));
    }
    kernel
        .apply_fill(OrderId(2), 100.0, 504.0, at_second(3, 4))
        .expect("the stop fills");
    assert_eq!(excursion(&only_trade(kernel)), (-50.0, 590.0, 390.0));
}

/// A stop 3 points below the buy that trails 2 points behind the best price.
fn trailing() -> LegRules {
    LegRules {
        trail: Some(TrailSpec {
            trigger: Distance::Points(2.0),
            step: None,
        }),
        ..stop_points(3.0)
    }
}

/// A tick at `price` at `at`, then the venue's confirmation of each modification it made.
fn tick_confirmed(kernel: &mut Engine, price: f64, at: i64) {
    kernel
        .push_tick(SBIN, price, at)
        .expect("the tick is taken");
    for (order, _, _) in modified(&kernel.take_commands()) {
        kernel.apply_modify(order).expect("the venue confirms");
    }
}
