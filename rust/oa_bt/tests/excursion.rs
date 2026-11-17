#![cfg(all(feature = "costs", feature = "time"))]
//! A trade's MAE and MFE (`engine::excursion`) and the slippage of its fills, through
//! whole backtests: which prices of the base bars count, long and short, entries and
//! exits at the open and inside a bar, adds and partial exits, a gap between sessions,
//! and what slippage cost each fill and each trade.
//!
//! SBIN on 1-minute bars of 2025-07-01 (flat at 100.00 unless a minute is given), MIS,
//! 10 units, no charges, no slippage unless a test sets it. An order decided at the
//! close of minute k - 1 (the instant `minute(k)`) fills at the open of minute k. Every
//! expected figure is worked by hand: units x (price - entry) for a long, the reverse
//! for a short, plus what partial exits realized.

mod order_support;

use oa_bt::engine::{RunOutput, TradeRecord};
use oa_bt::exec::{OrderRequest, OrderType};
use oa_bt::market::slippage::SlippageSpec;
use oa_bt::model::{Product, Side};
use oa_bt::rms::{Distance, LegRules};
use order_support::{
    engine, exit_all, july_2025, limit, minute, minute_bars, minute_config, order, run_daily,
    run_script, stop_market, Act, Ohlc,
};

const UNITS: f64 = 10.0;

/// Run `acts` (decision instant, act) on 2025-07-01's minute bars `overrides`, flat at
/// 100.00 elsewhere, with `slippage`.
fn run_minutes(
    overrides: &[(u32, Ohlc)],
    acts: Vec<(i64, Act)>,
    slippage: SlippageSpec,
) -> RunOutput {
    let mut config = minute_config();
    config.slippage = slippage;
    let kernel = engine(
        config,
        order_support::equity(),
        minute_bars(10, 100.0, overrides),
    );
    run_script(kernel, acts).0
}

fn mis(side: Side, order_type: OrderType) -> OrderRequest {
    order(side, UNITS, order_type, Product::Mis)
}

fn with_rules(side: Side, rules: LegRules) -> OrderRequest {
    OrderRequest {
        leg_rules: rules,
        ..mis(side, OrderType::Market)
    }
}

/// The only trade of a run.
fn only_trade(output: &RunOutput) -> TradeRecord {
    assert_eq!(output.trades.len(), 1, "{:?}", output.trades);
    output.trades[0]
}

/// (MAE, MFE, gross P&L) of a trade, rupees.
fn excursion(trade: &TradeRecord) -> (f64, f64, f64) {
    (trade.mae_rupees, trade.mfe_rupees, trade.gross_pnl_rupees)
}

#[test]
fn a_next_open_trade_counts_its_whole_slices_and_only_the_open_of_its_exit_slice() {
    // Entry at the open of minute 2 (100.00), exit at the open of minute 4 (104.50).
    // Minute 2 (100, 103, 98, 101) and minute 3 (101, 105, 99.50, 104) count whole;
    // minute 4 counts its open only, so its 110.00 and 90.00 never count.
    // Long: worst 10 x (98 - 100) = -20, best 10 x (105 - 100) = 50, gross 45.
    // Short: worst 10 x (100 - 105) = -50, best 10 x (100 - 98) = 20, gross -45.
    let bars = [
        (2, (100.0, 103.0, 98.0, 101.0)),
        (3, (101.0, 105.0, 99.5, 104.0)),
        (4, (104.5, 110.0, 90.0, 100.0)),
    ];
    let table = [
        (Side::Buy, (-20.0, 50.0, 45.0)),
        (Side::Sell, (-50.0, 20.0, -45.0)),
    ];
    for (side, expected) in table {
        let acts = vec![
            (minute(2), Act::Submit(mis(side, OrderType::Market))),
            (minute(4), Act::Submit(exit_all(side, Product::Mis))),
        ];
        let output = run_minutes(&bars, acts, SlippageSpec::Zero);
        assert_eq!(excursion(&only_trade(&output)), expected, "{side:?}");
    }
}

#[test]
fn an_exit_inside_a_slice_counts_its_open_and_its_fill_but_not_its_range() {
    // Long 10 at 100.00 from the open of minute 2, whose (100, 101, 99, 100.50) counts:
    // -10 and +10. Minute 3 opens at 100.50 (+5) and reaches the level inside the bar:
    // the stop at 98.00 fills at -20 (the low 97.00 would be -30); the target at 103.00
    // fills at +30 (the high 104.00 would be +40, the low 98.00 -20).
    let stop = LegRules {
        stop: Some(Distance::Points(2.0)),
        ..LegRules::default()
    };
    let target = LegRules {
        target: Some(Distance::Points(3.0)),
        ..LegRules::default()
    };
    let table = [
        (
            "stop",
            stop,
            (100.5, 102.0, 97.0, 97.5),
            "leg_sl",
            (-20.0, 10.0, -20.0),
        ),
        (
            "target",
            target,
            (100.5, 104.0, 98.0, 103.5),
            "leg_target",
            (-10.0, 30.0, 30.0),
        ),
    ];
    for (what, rules, exit_bar, reason, expected) in table {
        let bars = [(2, (100.0, 101.0, 99.0, 100.5)), (3, exit_bar)];
        let acts = vec![(minute(2), Act::Submit(with_rules(Side::Buy, rules)))];
        let trade = only_trade(&run_minutes(&bars, acts, SlippageSpec::Zero));
        assert_eq!(trade.exit_reason.code(), reason, "{what}");
        assert_eq!(excursion(&trade), expected, "{what}");
    }
}

#[test]
fn a_stop_entry_inside_a_slice_counts_its_fill_and_the_close_not_the_range() {
    // A buy stop at 101.00 fills inside minute 2 (100, 103, 99, 102.50): the fill (0)
    // and the close (+15) count, not the low 99.00 (-20) nor the high 103.00 (+20).
    // Minute 3 (102.50, 104, 100, 103) counts whole: +30 and -10. The exit at the open
    // of minute 4 (103.00): +20.
    let bars = [
        (2, (100.0, 103.0, 99.0, 102.5)),
        (3, (102.5, 104.0, 100.0, 103.0)),
        (4, (103.0, 103.5, 102.0, 103.0)),
    ];
    let acts = vec![
        (minute(2), Act::Submit(mis(Side::Buy, stop_market(101.0)))),
        (minute(4), Act::Submit(exit_all(Side::Buy, Product::Mis))),
    ];
    let trade = only_trade(&run_minutes(&bars, acts, SlippageSpec::Zero));
    assert_eq!(trade.entry_price_rupees, 101.0);
    assert_eq!(excursion(&trade), (-10.0, 30.0, 20.0));
}

#[test]
fn an_entry_and_an_exit_inside_one_slice_count_only_their_two_fills() {
    // A buy stop at 101.00 and a sell limit at 104.00 both fill inside minute 2 (100,
    // 105, 99, 102): 0 and +30. The low 99.00 (-20), the high 105.00 (+40) and the
    // close do not count.
    let bars = [(2, (100.0, 105.0, 99.0, 102.0))];
    let acts = vec![
        (minute(2), Act::Submit(mis(Side::Buy, stop_market(101.0)))),
        (minute(2), Act::Submit(mis(Side::Sell, limit(104.0)))),
    ];
    let trade = only_trade(&run_minutes(&bars, acts, SlippageSpec::Zero));
    assert_eq!((trade.entry_slice, trade.exit_slice), (2, 2));
    assert_eq!(excursion(&trade), (0.0, 30.0, 30.0));
}

#[test]
fn an_add_and_a_partial_exit_count_with_the_position_held_around_them() {
    // Buy 10 at 100.00 (minute 2: 102 is +20, 99 is -10); add 10 at 101.00 (minute 3,
    // average 100.50: its open with 10 held is +10, then 20 held: 103 is +50, 100 is
    // -10); sell 10 at 102.00 (minute 4, +15 realized, 10 left at 100.50: 105 is 15 + 45
    // = +60, 97 is 15 - 35 = -20); sell the last 10 at 99.00 (minute 5): gross 0.
    let bars = [
        (2, (100.0, 102.0, 99.0, 101.0)),
        (3, (101.0, 103.0, 100.0, 102.0)),
        (4, (102.0, 105.0, 97.0, 99.0)),
        (5, (99.0, 99.0, 99.0, 99.0)),
    ];
    let acts = vec![
        (minute(2), Act::Submit(mis(Side::Buy, OrderType::Market))),
        (minute(3), Act::Submit(mis(Side::Buy, OrderType::Market))),
        (minute(4), Act::Submit(mis(Side::Sell, OrderType::Market))),
        (minute(5), Act::Submit(mis(Side::Sell, OrderType::Market))),
    ];
    let trade = only_trade(&run_minutes(&bars, acts, SlippageSpec::Zero));
    assert_eq!((trade.fills, trade.max_units), (4, 20.0));
    assert_eq!(excursion(&trade), (-20.0, 60.0, 0.0));
}

#[test]
fn a_delivery_trade_counts_the_gap_of_a_later_session() {
    // Daily bars. Bought at the 07-02 open (102.00): (102, 104, 101, 103) is +20 and
    // -10; 07-03 gaps down to (95, 96, 92, 94): -100 at its low; sold at the 07-04 open
    // (94.50): -75. The 07-04 range (99.00, 93.00) does not count.
    let bars = [
        (july_2025(1), (100.0, 101.0, 99.0, 100.0)),
        (july_2025(2), (102.0, 104.0, 101.0, 103.0)),
        (july_2025(3), (95.0, 96.0, 92.0, 94.0)),
        (july_2025(4), (94.5, 99.0, 93.0, 98.0)),
    ];
    let acts = vec![
        (
            july_2025(1),
            Act::Submit(order(Side::Buy, UNITS, OrderType::Market, Product::Cnc)),
        ),
        (july_2025(3), Act::Submit(exit_all(Side::Buy, Product::Cnc))),
    ];
    let (output, _, _) = run_daily(&bars, acts, |_| {});
    assert_eq!(excursion(&only_trade(&output)), (-100.0, 20.0, -75.0));
}

#[test]
fn every_fill_records_what_its_slippage_cost_and_each_trade_their_sum() {
    // Two ticks of 0.05 on 10 units cost 1.00 a fill taking the market; the resting
    // limit takes none. Trade 1: a buy stop at 101.00 fills at 101.10 (1.00), the sell
    // limit at 104.00 (0.00). Trade 2: a buy at the open of minute 4 (100.00) fills at
    // 100.10 (1.00), its exit at the open of minute 5 at 99.90 (1.00).
    let bars = [(2, (100.0, 105.0, 99.0, 102.0))];
    let acts = vec![
        (minute(2), Act::Submit(mis(Side::Buy, stop_market(101.0)))),
        (minute(2), Act::Submit(mis(Side::Sell, limit(104.0)))),
        (minute(4), Act::Submit(mis(Side::Buy, OrderType::Market))),
        (minute(5), Act::Submit(exit_all(Side::Buy, Product::Mis))),
    ];
    let output = run_minutes(&bars, acts, SlippageSpec::Ticks(2));
    let fills: Vec<(f64, Option<f64>)> = output
        .fills
        .iter()
        .map(|record| (record.fill.price_rupees, record.slippage_rupees))
        .collect();
    let expected = [
        (101.1, Some(1.0)),
        (104.0, Some(0.0)),
        (100.1, Some(1.0)),
        (99.9, Some(1.0)),
    ];
    assert_eq!(fills, expected);
    let trades: Vec<(Option<f64>, f64)> = output
        .trades
        .iter()
        .map(|trade| (trade.slippage_rupees, trade.gross_pnl_rupees))
        .collect();
    // The slippage is in the fill prices, so in the gross P&L already: 10 x (104.00 -
    // 101.10) = 29.00 and 10 x (99.90 - 100.10) = -2.00.
    assert_eq!(trades, [(Some(1.0), 29.0), (Some(2.0), -2.0)]);
}
