//! Win/loss ratio, Kelly fraction, risk of ruin, CPC index and losing trade streaks.

use super::{assert_close, hours};
use crate::metrics::{risk_of_ruin, trade_ratios, TradeColumns, TradeRatios};

/// Ratios of trades closing an hour apart with the given net P&L and no costs.
fn ratios(net: &[f64], capital: f64) -> TradeRatios {
    let exits: Vec<i64> = (1..=net.len() as i64).map(hours).collect();
    let entries: Vec<i64> = exits.iter().map(|t| t - hours(1) / 2).collect();
    let costs = vec![0.0; net.len()];
    let trades = TradeColumns {
        entry_ts: &entries,
        exit_ts: &exits,
        gross_pnl: net,
        net_pnl: net,
        costs: &costs,
    };
    trade_ratios(&trades, capital).unwrap()
}

#[test]
fn win_loss_kelly_and_cpc_follow_their_definitions() {
    let r = ratios(&[100.0, -50.0, 200.0, -50.0, 0.0], 10_000.0);
    // Two wins and two losses; the breakeven trade is neither.
    assert_close(r.win_loss_ratio, 1.0);
    // p = 2 / 4 = 0.5 and payoff 150 / 50 = 3: 0.5 - 0.5 / 3.
    assert_close(r.kelly_criterion, 0.5 - 0.5 / 3.0);
    // Profit factor 300 / 100 x win rate 2 / 5 x payoff 3.
    assert_close(r.cpc_index, 3.0 * 0.4 * 3.0);
    assert_eq!(r.loss_streaks, vec![1, 1]);
}

#[test]
fn the_risk_of_ruin_is_the_lundberg_bound_of_the_trades() {
    // Trades of +2 and -1 rupees, equally likely: (exp(-2g) + exp(g)) / 2 = 1 gives
    // y^3 - 2y^2 + 1 = 0 for y = exp(g), whose root above 1 is the golden ratio. From
    // 10 rupees the bound is exp(-10 g) = golden ratio ^ -10.
    let golden = (1.0 + 5.0_f64.sqrt()) / 2.0;
    assert_close(risk_of_ruin(&[2.0, -1.0], 10.0), golden.powi(-10));
    // Scaling every trade and the capital together leaves it unchanged.
    assert_close(risk_of_ruin(&[2000.0, -1000.0], 10_000.0), golden.powi(-10));
}

#[test]
fn ruin_is_certain_without_an_edge_and_impossible_without_a_loss() {
    assert_eq!(risk_of_ruin(&[100.0, -100.0], 1_000.0), 1.0);
    assert_eq!(risk_of_ruin(&[-10.0], 1_000.0), 1.0);
    assert_eq!(risk_of_ruin(&[10.0, 0.0], 1_000.0), 0.0);
    assert!(risk_of_ruin(&[], 1_000.0).is_nan());
    assert!(risk_of_ruin(&[10.0, -5.0], 0.0).is_nan());
}

#[test]
fn all_winning_trades_have_an_infinite_win_loss_ratio_and_no_kelly() {
    let r = ratios(&[10.0, 20.0], 1_000.0);
    assert_eq!(r.win_loss_ratio, f64::INFINITY);
    assert!(r.kelly_criterion.is_nan());
    assert_eq!(r.risk_of_ruin, 0.0);
    assert!(r.loss_streaks.is_empty());
}

#[test]
fn all_losing_trades_make_ruin_certain() {
    let r = ratios(&[-10.0, -20.0, -5.0], 1_000.0);
    assert_eq!(r.win_loss_ratio, 0.0);
    assert!(r.kelly_criterion.is_nan());
    assert_eq!(r.risk_of_ruin, 1.0);
    assert_eq!(r.loss_streaks, vec![3]);
}

#[test]
fn no_trade_has_no_ratio() {
    let r = ratios(&[], 1_000.0);
    assert!(r.win_loss_ratio.is_nan() && r.kelly_criterion.is_nan());
    assert!(r.risk_of_ruin.is_nan() && r.cpc_index.is_nan());
    assert!(r.loss_streaks.is_empty());
}
