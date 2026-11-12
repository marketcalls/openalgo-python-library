//! Ratios of closed trades that size or judge an edge: the win/loss count ratio, the
//! Kelly fraction, the risk of ruin, the CPC index and the losing streaks.
//!
//! Definitions, on the net P&L `p_1..p_n` of the closed trades in rupees, with wins,
//! losses, win rate, profit factor and payoff ratio as `trades` defines them:
//!
//! - Win/loss ratio: winning trades over losing trades (a count ratio).
//! - Kelly criterion (Kelly 1956, "A New Interpretation of Information Rate", Bell
//!   System Technical Journal 35(4), for a bet won with probability `p` at odds `b`):
//!   `f = p - (1 - p) / b`, with `p` the share of wins among the trades that won or lost
//!   and `b` the payoff ratio. The fraction of capital to stake per trade that maximises
//!   long-run growth; negative when the edge is negative.
//! - Risk of ruin: the trades are taken as a random walk of independent draws from the
//!   run's own net P&L, at fixed rupee size, starting from the capital `u`. The
//!   adjustment coefficient `g > 0` solves `mean(exp(-g p_i)) = 1`, and the Lundberg
//!   bound `exp(-g u)` (Asmussen and Albrecher 2010, Ruin Probabilities, World
//!   Scientific, the Lundberg inequality) bounds the probability of ever losing the
//!   capital. It is 1 when the mean net P&L is zero or below (the walk has no upward
//!   drift) and 0 when no trade lost.
//! - CPC index: profit factor x win rate x payoff ratio, a composite of the three trade
//!   figures; above 1 is commonly read as a robust edge.
//! - Losing streaks: the length of every maximal run of losing trades, in exit order (as
//!   `trades` orders them); a breakeven trade breaks a run.
//!
//! Undefined cases: with no trade every figure is NaN. The win/loss ratio is infinite
//! with wins and no loss. The Kelly fraction is NaN without both a win and a loss.

use super::error::MetricsError;
use super::moments::mean;
use super::sessions::loss_streaks;
use super::tails::ratio_or_infinite;
use super::trades::{trade_stats, TradeColumns};

/// Upper limit on the doublings that bracket the adjustment coefficient, and on the
/// bisection steps that find it: 2^200 covers any coefficient a finite sample can
/// have, and 200 halvings reach the resolution of an f64 from any bracket.
const MAX_SEARCH_STEPS: usize = 200;

/// Trade ratios (definitions in the module docs).
#[derive(Clone, Debug, PartialEq)]
pub struct TradeRatios {
    pub win_loss_ratio: f64,
    /// Fraction of capital per trade.
    pub kelly_criterion: f64,
    /// Probability, a fraction.
    pub risk_of_ruin: f64,
    pub cpc_index: f64,
    /// Length of each losing streak, in trades, in exit order.
    pub loss_streaks: Vec<usize>,
}

/// `mean(exp(-g x)) - 1` for P&L scaled to at most 1 in magnitude.
fn excess_growth(scaled: &[f64], g: f64) -> f64 {
    mean(&scaled.iter().map(|x| (-g * x).exp()).collect::<Vec<_>>()) - 1.0
}

/// The risk of ruin of the trades' net P&L `net_pnl` (rupees) from `capital` rupees
/// (module docs). NaN for no trade or a capital that is not positive.
pub fn risk_of_ruin(net_pnl: &[f64], capital: f64) -> f64 {
    let finite = net_pnl.iter().all(|p| p.is_finite());
    if net_pnl.is_empty() || !finite || !(capital.is_finite() && capital > 0.0) {
        return f64::NAN;
    }
    if net_pnl.iter().all(|&p| p >= 0.0) {
        return 0.0;
    }
    if mean(net_pnl) <= 0.0 {
        return 1.0;
    }
    // Scaled to the largest trade, the coefficient is found in units of 1 / scale.
    let scale = net_pnl.iter().fold(0.0_f64, |m, p| m.max(p.abs()));
    let scaled: Vec<f64> = net_pnl.iter().map(|p| p / scale).collect();
    let mut high = 1.0;
    for _ in 0..MAX_SEARCH_STEPS {
        if excess_growth(&scaled, high) > 0.0 {
            break;
        }
        high *= 2.0;
    }
    let mut low = 0.0;
    for _ in 0..MAX_SEARCH_STEPS {
        let middle = (low + high) / 2.0;
        if middle <= low || middle >= high {
            break;
        }
        if excess_growth(&scaled, middle) > 0.0 {
            high = middle;
        } else {
            low = middle;
        }
    }
    (-high * capital / scale).exp()
}

/// Trade ratios of closed trades from a capital of `capital` rupees (module docs).
///
/// Errors: those of `trades::trade_stats`.
pub fn trade_ratios(trades: &TradeColumns, capital: f64) -> Result<TradeRatios, MetricsError> {
    let stats = trade_stats(trades)?;
    let net = trades.net_by_exit();
    let decided = stats.wins + stats.losses;
    let kelly = if stats.wins == 0 || stats.losses == 0 {
        f64::NAN
    } else {
        let p = stats.wins as f64 / decided as f64;
        p - (1.0 - p) / stats.payoff_ratio
    };
    let counts = ratio_or_infinite(stats.wins as f64, stats.losses as f64);
    Ok(TradeRatios {
        win_loss_ratio: if stats.trades == 0 { f64::NAN } else { counts },
        kelly_criterion: kelly,
        risk_of_ruin: risk_of_ruin(&net, capital),
        cpc_index: stats.profit_factor * stats.win_rate * stats.payoff_ratio,
        loss_streaks: loss_streaks(&net),
    })
}
