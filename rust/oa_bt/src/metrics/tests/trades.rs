//! Trade statistics, counted per closed trade.

use super::{assert_close, hours};
use crate::metrics::{trade_stats, MetricsError, TradeColumns};

/// Six trades given out of exit order; by exit time their net P&L reads
/// +100, -50, 0, +200, -25, -25.
struct Table {
    entry: Vec<i64>,
    exit: Vec<i64>,
    gross: Vec<f64>,
    net: Vec<f64>,
    costs: Vec<f64>,
}

impl Table {
    fn new(exit_hours: &[i64], net: &[f64]) -> Table {
        Table {
            entry: vec![0; net.len()],
            exit: exit_hours.iter().map(|&h| hours(h)).collect(),
            gross: net.to_vec(),
            net: net.to_vec(),
            costs: vec![0.0; net.len()],
        }
    }

    fn columns(&self) -> TradeColumns<'_> {
        TradeColumns {
            entry_ts: &self.entry,
            exit_ts: &self.exit,
            gross_pnl: &self.gross,
            net_pnl: &self.net,
            costs: &self.costs,
        }
    }
}

fn six_trades() -> Table {
    Table::new(
        &[4, 1, 6, 2, 5, 3],
        &[200.0, 100.0, -25.0, -50.0, -25.0, 0.0],
    )
}

#[test]
fn counts_rates_and_averages() {
    let stats = trade_stats(&six_trades().columns()).unwrap();
    assert_eq!((stats.trades, stats.wins, stats.losses), (6, 2, 3));
    assert_close(stats.win_rate, 2.0 / 6.0);
    assert_close(stats.average_win, 150.0);
    assert_close(stats.average_loss, -100.0 / 3.0);
    assert_close(stats.expectancy, 200.0 / 6.0);
    assert_close(stats.largest_win, 200.0);
    assert_close(stats.largest_loss, -50.0);
}

#[test]
fn profit_factor_and_payoff_ratio() {
    let stats = trade_stats(&six_trades().columns()).unwrap();
    // Wins 300, losses -100.
    assert_close(stats.profit_factor, 3.0);
    // 150 / (100 / 3).
    assert_close(stats.payoff_ratio, 4.5);
}

#[test]
fn runs_follow_exit_order_and_a_breakeven_trade_breaks_them() {
    // Exit order: W L 0 W L L.
    let stats = trade_stats(&six_trades().columns()).unwrap();
    assert_eq!(
        (stats.max_consecutive_wins, stats.max_consecutive_losses),
        (1, 2)
    );
}

#[test]
fn trades_exiting_together_are_ordered_by_entry() {
    let mut table = Table::new(&[5, 5, 5], &[-1.0, 1.0, 1.0]);
    table.entry = vec![hours(3), hours(1), hours(2)];
    // By entry: +1, +1, -1.
    let stats = trade_stats(&table.columns()).unwrap();
    assert_eq!(stats.max_consecutive_wins, 2);
}

#[test]
fn no_trades_leave_every_rate_undefined_and_every_sum_zero() {
    let stats = trade_stats(&Table::new(&[], &[]).columns()).unwrap();
    assert_eq!((stats.trades, stats.max_consecutive_wins), (0, 0));
    for value in [
        stats.win_rate,
        stats.average_win,
        stats.average_loss,
        stats.profit_factor,
        stats.expectancy,
        stats.payoff_ratio,
        stats.largest_win,
        stats.largest_loss,
        stats.average_holding,
        stats.cost_share_of_gross,
    ] {
        assert!(value.is_nan());
    }
    assert_eq!(
        (stats.gross_pnl, stats.net_pnl, stats.total_costs),
        (0.0, 0.0, 0.0)
    );
}

#[test]
fn one_winning_trade_has_an_infinite_profit_factor_and_no_loss_figures() {
    let stats = trade_stats(&Table::new(&[1], &[40.0]).columns()).unwrap();
    assert_eq!((stats.trades, stats.win_rate), (1, 1.0));
    assert_eq!(stats.profit_factor, f64::INFINITY);
    assert!(stats.average_loss.is_nan() && stats.payoff_ratio.is_nan());
    assert!(stats.largest_loss.is_nan());
}

#[test]
fn only_losses_give_a_profit_factor_of_zero() {
    let stats = trade_stats(&Table::new(&[1, 2], &[-1.0, -2.0]).columns()).unwrap();
    assert_eq!((stats.profit_factor, stats.win_rate), (0.0, 0.0));
}

#[test]
fn only_breakeven_trades_leave_the_profit_factor_undefined() {
    let stats = trade_stats(&Table::new(&[1, 2], &[0.0, 0.0]).columns()).unwrap();
    assert!(stats.profit_factor.is_nan());
    assert_eq!((stats.wins, stats.losses, stats.win_rate), (0, 0, 0.0));
}

#[test]
fn average_holding_time_is_the_mean_of_exit_minus_entry() {
    let mut table = Table::new(&[2, 5], &[1.0, 1.0]);
    table.entry = vec![hours(1), hours(2)];
    // One hour and three hours.
    let stats = trade_stats(&table.columns()).unwrap();
    assert_eq!(stats.average_holding, hours(2) as f64);
}

#[test]
fn costs_are_summed_and_compared_with_gross_pnl() {
    let mut table = Table::new(&[1, 2], &[100.0, -50.0]);
    (table.gross, table.costs) = (vec![120.0, -40.0], vec![20.0, 10.0]);
    let stats = trade_stats(&table.columns()).unwrap();
    assert_eq!(
        (stats.gross_pnl, stats.total_costs, stats.net_pnl),
        (80.0, 30.0, 50.0)
    );
    assert_close(stats.cost_share_of_gross, 30.0 / 80.0);
}

#[test]
fn the_cost_share_is_undefined_when_gross_pnl_is_not_positive() {
    let mut table = Table::new(&[1], &[-60.0]);
    (table.gross, table.costs) = (vec![-50.0], vec![10.0]);
    assert!(trade_stats(&table.columns())
        .unwrap()
        .cost_share_of_gross
        .is_nan());
}

#[test]
fn net_pnl_must_be_gross_pnl_minus_costs_to_the_paisa() {
    let mut table = Table::new(&[1], &[100.0]);
    (table.gross, table.costs) = (vec![110.0], vec![9.995]);
    assert!(trade_stats(&table.columns()).is_ok());
    table.costs = vec![9.0];
    assert_eq!(
        trade_stats(&table.columns()),
        Err(MetricsError::PnlMismatch {
            index: 0,
            gross: 110.0,
            costs: 9.0,
            net: 100.0
        })
    );
}

#[test]
fn a_trade_exiting_before_it_enters_is_refused() {
    let mut table = Table::new(&[1, 1], &[1.0, 1.0]);
    table.entry = vec![0, hours(2)];
    assert_eq!(
        trade_stats(&table.columns()),
        Err(MetricsError::ExitBeforeEntry { index: 1 })
    );
}
