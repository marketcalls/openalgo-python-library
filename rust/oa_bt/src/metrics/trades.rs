//! Statistics of closed trades (round trips), counted per trade, never per bar.
//!
//! Definitions, on the net P&L `p_1..p_n` of the closed trades, in rupees:
//!
//! - A win is a trade with `p > 0` and a loss one with `p < 0`. A trade with `p = 0`
//!   is neither, but it counts in the number of trades and breaks a run.
//! - Win rate: wins divided by all closed trades.
//! - Average win: mean net P&L of the wins. Average loss: mean net P&L of the losses
//!   (negative).
//! - Profit factor: the sum of the wins divided by the magnitude of the sum of the
//!   losses.
//! - Expectancy: mean net P&L per trade, `(p_1 + ... + p_n) / n`.
//! - Payoff ratio: average win divided by the magnitude of the average loss.
//! - Largest win: the highest net P&L among wins. Largest loss: the lowest among
//!   losses (negative).
//! - Consecutive wins and losses: the longest unbroken run of wins, and of losses,
//!   with trades ordered by exit time (then entry time, then input order).
//! - Average holding time: mean of exit time minus entry time, in nanoseconds.
//! - Gross P&L, net P&L and costs: the sums over the trades, in rupees.
//! - Cost share of gross P&L: total costs divided by gross P&L, when gross P&L is
//!   positive.
//!
//! Undefined cases: with no trades every rate, average and extreme is NaN and the sums
//! are zero. Average win, largest win and payoff ratio are NaN without a win, and the
//! loss figures without a loss. Profit factor is infinite with wins and no loss and
//! NaN with neither. Cost share is NaN when gross P&L is zero or negative.

use crate::model::TsNanos;

use super::curve::{check_finite, check_length};
use super::error::MetricsError;
use super::PNL_TOLERANCE_RUPEES;

/// Closed trades as columns, one row per trade. Times are UTC nanoseconds and money is
/// rupees; `net_pnl` must equal `gross_pnl - costs` within `PNL_TOLERANCE_RUPEES`.
#[derive(Clone, Copy, Debug)]
pub struct TradeColumns<'a> {
    pub entry_ts: &'a [TsNanos],
    pub exit_ts: &'a [TsNanos],
    pub gross_pnl: &'a [f64],
    pub net_pnl: &'a [f64],
    pub costs: &'a [f64],
}

/// Trade statistics (definitions in the module docs).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TradeStats {
    pub trades: usize,
    pub wins: usize,
    pub losses: usize,
    /// Fraction of trades that won.
    pub win_rate: f64,
    /// Rupees.
    pub average_win: f64,
    /// Rupees, negative.
    pub average_loss: f64,
    pub profit_factor: f64,
    /// Rupees per trade.
    pub expectancy: f64,
    pub payoff_ratio: f64,
    /// Rupees.
    pub largest_win: f64,
    /// Rupees, negative.
    pub largest_loss: f64,
    pub max_consecutive_wins: usize,
    pub max_consecutive_losses: usize,
    /// Nanoseconds.
    pub average_holding: f64,
    /// Rupees.
    pub gross_pnl: f64,
    /// Rupees.
    pub net_pnl: f64,
    /// Rupees.
    pub total_costs: f64,
    /// Fraction of gross P&L taken by costs.
    pub cost_share_of_gross: f64,
}

impl TradeColumns<'_> {
    fn check(&self) -> Result<(), MetricsError> {
        let n = self.entry_ts.len();
        check_length(self.exit_ts.len(), n, "trade exit times")?;
        check_length(self.gross_pnl.len(), n, "trade gross P&L")?;
        check_length(self.net_pnl.len(), n, "trade net P&L")?;
        check_length(self.costs.len(), n, "trade costs")?;
        check_finite(self.gross_pnl, "trade gross P&L")?;
        check_finite(self.net_pnl, "trade net P&L")?;
        check_finite(self.costs, "trade costs")?;
        let times = self.entry_ts.iter().zip(self.exit_ts);
        if let Some(index) = times.clone().position(|(entry, exit)| exit < entry) {
            return Err(MetricsError::ExitBeforeEntry { index });
        }
        let rows = self.gross_pnl.iter().zip(self.costs).zip(self.net_pnl);
        for (index, ((&gross, &costs), &net)) in rows.enumerate() {
            let tolerance = PNL_TOLERANCE_RUPEES + f64::EPSILON * gross.abs();
            if (gross - costs - net).abs() > tolerance {
                return Err(MetricsError::PnlMismatch {
                    index,
                    gross,
                    costs,
                    net,
                });
            }
        }
        Ok(())
    }

    /// Net P&L in the order of exit time, then entry time, then input order.
    fn net_by_exit(&self) -> Vec<f64> {
        let mut order: Vec<usize> = (0..self.entry_ts.len()).collect();
        order.sort_by_key(|&i| (self.exit_ts.get(i), self.entry_ts.get(i)));
        order
            .into_iter()
            .filter_map(|i| self.net_pnl.get(i).copied())
            .collect()
    }
}

fn ratio_or_infinite(numerator: f64, denominator: f64) -> f64 {
    if denominator > 0.0 {
        numerator / denominator
    } else if numerator > 0.0 {
        f64::INFINITY
    } else {
        f64::NAN
    }
}

/// Longest runs of wins and of losses in a sequence of net P&L.
fn longest_runs(net: &[f64]) -> (usize, usize) {
    let (mut wins, mut losses, mut best_wins, mut best_losses) = (0, 0, 0, 0);
    for &p in net {
        (wins, losses) = if p > 0.0 {
            (wins + 1, 0)
        } else if p < 0.0 {
            (0, losses + 1)
        } else {
            (0, 0)
        };
        best_wins = usize::max(best_wins, wins);
        best_losses = usize::max(best_losses, losses);
    }
    (best_wins, best_losses)
}

fn mean_or_nan(sum: f64, count: usize) -> f64 {
    if count == 0 {
        f64::NAN
    } else {
        sum / count as f64
    }
}

/// Statistics of closed trades (definitions and undefined cases in the module docs).
///
/// Errors: `LengthMismatch` and `NotFinite` for malformed columns; `ExitBeforeEntry`;
/// `PnlMismatch` when net P&L is not gross P&L minus costs.
pub fn trade_stats(trades: &TradeColumns) -> Result<TradeStats, MetricsError> {
    trades.check()?;
    let net = trades.net_by_exit();
    let n = net.len();
    let wins: Vec<f64> = net.iter().copied().filter(|&p| p > 0.0).collect();
    let losses: Vec<f64> = net.iter().copied().filter(|&p| p < 0.0).collect();
    let (win_sum, loss_sum) = (wins.iter().sum::<f64>(), losses.iter().sum::<f64>());
    let average_win = mean_or_nan(win_sum, wins.len());
    let average_loss = mean_or_nan(loss_sum, losses.len());
    let extreme = |values: &[f64], pick: fn(f64, f64) -> f64| {
        values.iter().copied().reduce(pick).unwrap_or(f64::NAN)
    };
    let (max_consecutive_wins, max_consecutive_losses) = longest_runs(&net);
    let holding: f64 = trades
        .entry_ts
        .iter()
        .zip(trades.exit_ts)
        .map(|(&entry, &exit)| exit.saturating_sub(entry) as f64)
        .sum();
    let gross_pnl: f64 = trades.gross_pnl.iter().sum();
    let total_costs: f64 = trades.costs.iter().sum();
    let net_pnl: f64 = net.iter().sum();
    Ok(TradeStats {
        trades: n,
        wins: wins.len(),
        losses: losses.len(),
        win_rate: mean_or_nan(wins.len() as f64, n),
        average_win,
        average_loss,
        profit_factor: ratio_or_infinite(win_sum, loss_sum.abs()),
        expectancy: mean_or_nan(net_pnl, n),
        payoff_ratio: average_win / average_loss.abs(),
        largest_win: extreme(&wins, f64::max),
        largest_loss: extreme(&losses, f64::min),
        max_consecutive_wins,
        max_consecutive_losses,
        average_holding: mean_or_nan(holding, n),
        gross_pnl,
        net_pnl,
        total_costs,
        cost_share_of_gross: if gross_pnl > 0.0 {
            total_costs / gross_pnl
        } else {
            f64::NAN
        },
    })
}
