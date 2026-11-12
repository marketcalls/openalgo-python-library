#![cfg(all(feature = "costs", feature = "time"))]
//! Resting DAY orders through the slice sequence (03-architecture 4.2 step 3,
//! 04-execution 4.1 and 4.2): LIMIT, SL-M and SL matched against the base bar, the
//! conversion of an unfilled stop-limit exit, modification and the tick check.
//!
//! One NSE session on 2025-07-01, one-minute bars, a one-minute strategy, MIS. Every bar
//! is flat at 100.00 unless a test overrides its minute. Every order is decided at the
//! close of minute 0 (09:16) and is first tested against the bar of minute 1, so each
//! expected fill can be read off the overrides.

mod order_support;

use oa_bt::engine::{RunOutput, Step, Submission};
use oa_bt::exec::{FillAt, OrderId, OrderRequest, OrderStatus, OrderType, RejectCode};
use oa_bt::market::slippage::SlippageSpec;
use oa_bt::model::{Product, Side};
use order_support::{
    engine, events_at, fills, limit, minute, minute_bars, minute_config, option, order, run_script,
    stop_limit, stop_market, Act, Ohlc,
};

const BARS: u32 = 10;
const FLAT: f64 = 100.0;

/// Run the orders decided at the given minutes' closes against the overridden bars.
fn run_orders(
    acts: Vec<(u32, Act)>,
    overrides: &[(u32, Ohlc)],
    slippage: SlippageSpec,
) -> (RunOutput, Vec<Submission>) {
    let mut config = minute_config();
    config.slippage = slippage;
    config.last_slice = Some(minute(BARS - 1));
    let kernel = engine(
        config,
        order_support::equity(),
        minute_bars(BARS, FLAT, overrides),
    );
    let acts = acts
        .into_iter()
        .map(|(k, act)| (minute(k + 1), act))
        .collect();
    let (output, script) = run_script(kernel, acts);
    (output, script.submissions)
}

/// One order decided at the close of minute 0, MIS, no slippage.
fn run_one(request: OrderRequest, overrides: &[(u32, Ohlc)]) -> RunOutput {
    run_orders(
        vec![(0, Act::Submit(request))],
        overrides,
        SlippageSpec::Zero,
    )
    .0
}

fn buy(order_type: OrderType) -> OrderRequest {
    order(Side::Buy, 10.0, order_type, Product::Mis)
}

fn sell(order_type: OrderType) -> OrderRequest {
    order(Side::Sell, 10.0, order_type, Product::Mis)
}

/// (minute, where, price) of every fill.
fn fill_points(output: &RunOutput) -> Vec<(i64, Step, FillAt, f64)> {
    fills(output)
        .into_iter()
        .map(|(ts, step, _, _, price, at, _)| (ts, step, at, price))
        .collect()
}

// Minute 1 opens at 100.00 and ranges from 99.00 to 101.00.
const RANGE: (u32, Ohlc) = (1, (100.0, 101.0, 99.0, 100.5));

// ---------------------------------------------------------------- LIMIT

#[test]
fn a_limit_buy_at_the_open_fills_at_the_open_in_step_3() {
    let output = run_one(buy(limit(100.0)), &[RANGE]);
    assert_eq!(
        fill_points(&output),
        [(minute(1), Step::Levels, FillAt::Open, 100.0)]
    );
}

#[test]
fn a_limit_buy_above_the_open_fills_at_the_better_open_without_slippage() {
    // Two ticks of slippage would make a market buy 100.10; a resting limit takes none.
    let acts = vec![(0, Act::Submit(buy(limit(100.5))))];
    let (output, _) = run_orders(acts, &[RANGE], SlippageSpec::Ticks(2));
    assert_eq!(
        fill_points(&output),
        [(minute(1), Step::Levels, FillAt::Open, 100.0)]
    );
}

#[test]
fn a_limit_buy_below_the_open_inside_the_range_fills_at_its_limit_without_slippage() {
    let acts = vec![(0, Act::Submit(buy(limit(99.5))))];
    let (output, _) = run_orders(acts, &[RANGE], SlippageSpec::Ticks(2));
    assert_eq!(
        fill_points(&output),
        [(minute(1), Step::Levels, FillAt::Level, 99.5)]
    );
}

#[test]
fn a_limit_buy_below_the_range_rests_until_a_bar_reaches_it() {
    // 98.95 is below minute 1's low of 99.00; minute 2 dips to 98.90.
    let dip = (2, (99.0, 99.5, 98.9, 99.2));
    let output = run_one(buy(limit(98.95)), &[RANGE, dip]);
    assert_eq!(
        fill_points(&output),
        [(minute(2), Step::Levels, FillAt::Level, 98.95)]
    );
}

#[test]
fn a_limit_sell_at_the_open_fills_at_the_open() {
    let output = run_one(sell(limit(100.0)), &[RANGE]);
    assert_eq!(
        fill_points(&output),
        [(minute(1), Step::Levels, FillAt::Open, 100.0)]
    );
}

#[test]
fn a_limit_sell_below_the_open_fills_at_the_better_open() {
    let output = run_one(sell(limit(99.5)), &[RANGE]);
    assert_eq!(
        fill_points(&output),
        [(minute(1), Step::Levels, FillAt::Open, 100.0)]
    );
}

#[test]
fn a_limit_sell_above_the_open_inside_the_range_fills_at_its_limit() {
    let output = run_one(sell(limit(100.8)), &[RANGE]);
    assert_eq!(
        fill_points(&output),
        [(minute(1), Step::Levels, FillAt::Level, 100.8)]
    );
}

#[test]
fn a_limit_sell_above_the_range_does_not_fill() {
    let output = run_one(sell(limit(101.05)), &[RANGE]);
    assert!(output.fills.is_empty());
    assert_eq!(output.orders[0].status, OrderStatus::Submitted);
}

#[test]
fn a_resting_limit_is_never_tested_against_its_decision_bar() {
    // Minute 0's own low of 98.00 would fill a buy at 99.00; the order is decided at
    // minute 0's close and first meets minute 1, flat at 100.00.
    let decision_bar = (0, (100.0, 100.0, 98.0, 100.0));
    let output = run_one(buy(limit(99.0)), &[decision_bar]);
    assert!(output.fills.is_empty());
}

// ---------------------------------------------------------------- SL-M

#[test]
fn a_buy_stop_market_triggered_at_the_open_fills_at_the_open_plus_slippage() {
    // Trigger 100.50; minute 1 gaps up to 101.00; two ticks of slippage: 101.10.
    let gap = (1, (101.0, 101.5, 100.8, 101.2));
    let acts = vec![(0, Act::Submit(buy(stop_market(100.5))))];
    let (output, _) = run_orders(acts, &[gap], SlippageSpec::Ticks(2));
    assert_eq!(
        fill_points(&output),
        [(minute(1), Step::Levels, FillAt::Open, 101.1)]
    );
}

#[test]
fn a_buy_stop_market_triggered_inside_the_range_fills_at_its_trigger_plus_slippage() {
    let acts = vec![(0, Act::Submit(buy(stop_market(100.5))))];
    let (output, _) = run_orders(acts, &[RANGE], SlippageSpec::Ticks(2));
    assert_eq!(
        fill_points(&output),
        [(minute(1), Step::Levels, FillAt::Level, 100.6)]
    );
    assert_eq!(output.orders[0].status, OrderStatus::Filled);
}

#[test]
fn a_sell_stop_market_triggered_at_the_open_fills_at_the_open() {
    let gap = (1, (98.5, 99.0, 98.0, 98.8));
    let output = run_one(sell(stop_market(99.5)), &[gap]);
    assert_eq!(
        fill_points(&output),
        [(minute(1), Step::Levels, FillAt::Open, 98.5)]
    );
}

#[test]
fn a_sell_stop_market_triggered_inside_the_range_fills_at_its_trigger() {
    let output = run_one(sell(stop_market(99.5)), &[RANGE]);
    assert_eq!(
        fill_points(&output),
        [(minute(1), Step::Levels, FillAt::Level, 99.5)]
    );
}

#[test]
fn a_stop_order_writes_triggered_before_its_fill() {
    let output = run_one(buy(stop_market(100.5)), &[RANGE]);
    assert_eq!(
        events_at(&output, minute(1)),
        ["Triggered", "Filled", "PositionOpened"]
    );
}

// ---------------------------------------------------------------- SL

#[test]
fn a_stop_limit_marketable_at_its_trigger_fills_there_plus_slippage_held_at_its_limit() {
    // Trigger 100.50, limit 100.70: two ticks give 100.60; five ticks would give
    // 100.75, held at the limit 100.70 (GTT-5).
    let request = buy(stop_limit(100.5, 100.7));
    let two = run_orders(
        vec![(0, Act::Submit(request.clone()))],
        &[RANGE],
        SlippageSpec::Ticks(2),
    );
    let five = run_orders(
        vec![(0, Act::Submit(request))],
        &[RANGE],
        SlippageSpec::Ticks(5),
    );
    assert_eq!(fill_points(&two.0)[0].3, 100.6);
    assert_eq!(fill_points(&five.0)[0].3, 100.7);
}

#[test]
fn a_stop_limit_triggered_at_the_open_beyond_its_limit_fills_at_the_limit_when_the_bar_comes_back()
{
    // Minute 1 opens at 101.00, beyond the limit 100.70, and trades back to 100.60.
    let gap = (1, (101.0, 101.2, 100.6, 101.0));
    let acts = vec![(0, Act::Submit(buy(stop_limit(100.5, 100.7))))];
    let (output, _) = run_orders(acts, &[gap], SlippageSpec::Ticks(2));
    assert_eq!(
        fill_points(&output),
        [(minute(1), Step::Levels, FillAt::Level, 100.7)]
    );
}

#[test]
fn a_triggered_stop_limit_entry_that_does_not_fill_rests_as_a_limit() {
    // Minute 1 gaps to 101.50 and stays above the limit 100.70; minute 2 dips to 100.60.
    let gap = (1, (101.5, 101.8, 101.0, 101.2));
    let dip = (2, (101.0, 101.2, 100.6, 101.0));
    let output = run_one(buy(stop_limit(100.5, 100.7)), &[gap, dip]);
    assert_eq!(
        fill_points(&output),
        [(minute(2), Step::Levels, FillAt::Level, 100.7)]
    );
    assert_eq!(events_at(&output, minute(1)), ["Triggered"]);
}

#[test]
fn a_stop_limit_exit_not_filled_within_its_bar_fills_at_the_next_open_plus_slippage() {
    // Long 10 from minute 1's open (100.00). The sell stop decided at minute 1's close
    // (trigger 99.50, limit 99.30) meets minute 2, which opens at 99.00 and never comes
    // back to 99.30: converted, it sells at minute 3's open 98.80 less two ticks.
    let gap = (2, (99.0, 99.2, 98.6, 98.9));
    let next = (3, (98.8, 99.0, 98.5, 98.7));
    let acts = vec![
        (0, Act::Submit(buy(OrderType::Market))),
        (1, Act::Submit(sell(stop_limit(99.5, 99.3)))),
    ];
    let (output, _) = run_orders(acts, &[gap, next], SlippageSpec::Ticks(2));
    let points = fill_points(&output);
    assert_eq!(points[1], (minute(3), Step::OpenFills, FillAt::Open, 98.7));
    assert_eq!(events_at(&output, minute(2)), ["Triggered"]);
}

// ---------------------------------------------------------------- validity, modify, tick

#[test]
fn an_unfilled_limit_expires_at_the_session_close() {
    let mut config = minute_config();
    config.last_slice = None;
    // Bars through the session: a run ends with its last bar (`run_end`).
    let session_minutes = order_support::NSE_CLOSE_MINUTE - order_support::NSE_OPEN_MINUTE;
    let kernel = engine(
        config,
        order_support::equity(),
        minute_bars(session_minutes, FLAT, &[]),
    );
    let (output, _) = run_script(kernel, vec![(minute(1), Act::Submit(buy(limit(99.0))))]);
    assert_eq!(output.orders[0].status, OrderStatus::Expired);
}

#[test]
fn a_modify_replaces_the_order_from_the_next_bar() {
    // A buy limit at 98.00 is moved to 99.50 at minute 1's close; minute 2 dips to
    // 99.40 and fills it at 99.50. Minute 1's own dip to 99.20 is before the modify.
    let dip_1 = (1, (100.0, 100.0, 99.2, 100.0));
    let dip_2 = (2, (100.0, 100.5, 99.4, 100.0));
    let acts = vec![
        (0, Act::Submit(buy(limit(98.0)))),
        (1, Act::Modify(OrderId(1), buy(limit(99.5)))),
    ];
    let (output, _) = run_orders(acts, &[dip_1, dip_2], SlippageSpec::Zero);
    assert_eq!(
        fill_points(&output),
        [(minute(2), Step::Levels, FillAt::Level, 99.5)]
    );
    assert_eq!(events_at(&output, minute(1)), ["Modified"]);
    assert_eq!(
        output.orders[0].request.order_type,
        OrderType::Limit { limit_rupees: 99.5 }
    );
}

#[test]
fn a_modify_that_fails_a_check_leaves_the_order_as_it_was() {
    let mut config = minute_config();
    config.last_slice = Some(minute(BARS - 1));
    let mut kernel = engine(
        config,
        order_support::equity(),
        minute_bars(BARS, FLAT, &[]),
    );
    let acts = vec![
        (minute(1), Act::Submit(buy(limit(98.0)))),
        (minute(2), Act::Modify(OrderId(1), buy(limit(98.02)))),
    ];
    let script = order_support::run_kept(&mut kernel, acts);
    assert!(matches!(
        script.modifications[..],
        [Ok(Submission::Rejected(OrderId(1), RejectCode::Tick))]
    ));
    let kept = kernel
        .order(OrderId(1))
        .map(|o| (o.status, o.request.order_type));
    assert_eq!(
        kept,
        Some((
            OrderStatus::Submitted,
            OrderType::Limit { limit_rupees: 98.0 }
        ))
    );
}

#[test]
fn a_modify_may_not_change_the_side() {
    let mut config = minute_config();
    config.last_slice = Some(minute(BARS - 1));
    let mut kernel = engine(
        config,
        order_support::equity(),
        minute_bars(BARS, FLAT, &[]),
    );
    let acts = vec![
        (minute(1), Act::Submit(buy(limit(98.0)))),
        (minute(2), Act::Modify(OrderId(1), sell(limit(102.0)))),
    ];
    let script = order_support::run_kept(&mut kernel, acts);
    assert!(script.modifications[0]
        .as_ref()
        .is_err_and(|error| error.is_input_error()));
}

#[test]
fn a_limit_or_trigger_off_the_tick_grid_is_rejected_tick() {
    let acts = vec![
        (0, Act::Submit(buy(limit(100.02)))),
        (0, Act::Submit(sell(stop_market(99.53)))),
        (0, Act::Submit(buy(stop_limit(100.5, 100.71)))),
    ];
    let (_, submissions) = run_orders(acts, &[], SlippageSpec::Zero);
    let codes: Vec<Option<RejectCode>> = submissions
        .iter()
        .map(|s| match s {
            Submission::Rejected(_, code) => Some(*code),
            _ => None,
        })
        .collect();
    assert_eq!(codes, [Some(RejectCode::Tick); 3]);
}

#[test]
fn an_options_stop_market_is_sent_as_a_stop_limit_five_percent_beyond_its_trigger() {
    // 04-execution 4.1: a buy stop at 120.30 on an option rests as an SL with its
    // limit 5 percent above, 126.315 rounded up to the tick: 126.35.
    let mut config = minute_config();
    config.last_slice = Some(minute(BARS - 1));
    let mut kernel = engine(config, option(), minute_bars(BARS, FLAT, &[]));
    let request = OrderRequest {
        quantity: oa_bt::exec::QuantitySpec::Units(75.0),
        ..buy(stop_market(120.3))
    };
    order_support::run_kept(&mut kernel, vec![(minute(1), Act::Submit(request))]);
    let sent = kernel.order(OrderId(1)).map(|o| o.request.order_type);
    assert_eq!(sent, Some(stop_limit(120.3, 126.35)));
}
