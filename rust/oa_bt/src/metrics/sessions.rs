//! Counting statistics of session returns: average and geometric mean, best and worst
//! session, winning and losing sessions, and losing streaks.
//!
//! Definitions, for session returns `r_1..r_n` (fractions) in date order:
//!
//! - Expected return: the geometric mean session return
//!   `((1 + r_1) ... (1 + r_n))^(1/n) - 1`, the per-session rate that compounds to the
//!   run's growth; -1 once a session loses everything (`1 + r <= 0`).
//! - Average return: the arithmetic mean `(r_1 + ... + r_n) / n`, flat sessions included.
//! - Best and worst session: the highest and the lowest return.
//! - A winning session has `r > 0` and a losing one `r < 0`; a flat session is neither.
//!   Session win rate: winning sessions over the sessions that won or lost.
//! - Average winning (losing) session: the mean return of the winning (losing) sessions.
//! - Consecutive winning (losing) sessions: the longest unbroken run; a flat session
//!   breaks a run.
//! - Losing streaks: the length of every maximal run of losing sessions, in date order.
//!
//! Undefined cases: with no session the means, extremes and the win rate are NaN; the
//! win rate is NaN when no session won or lost, and an average of winning (losing)
//! sessions is NaN without one.

use super::curve::check_finite;
use super::error::MetricsError;
use super::moments::mean;
use super::trades::longest_runs;

/// Counting statistics of session returns (definitions in the module docs). Returns are
/// fractions.
#[derive(Clone, Debug, PartialEq)]
pub struct SessionStats {
    pub sessions: usize,
    pub expected_return: f64,
    pub average_return: f64,
    pub best: f64,
    pub worst: f64,
    pub winning: usize,
    pub losing: usize,
    pub win_rate: f64,
    pub average_winning: f64,
    pub average_losing: f64,
    pub max_consecutive_wins: usize,
    pub max_consecutive_losses: usize,
    /// Length of each losing streak, in sessions, in date order.
    pub loss_streaks: Vec<usize>,
}

/// The length of every maximal run of negative values, in order; zero breaks a run.
pub fn loss_streaks(values: &[f64]) -> Vec<usize> {
    let mut streaks = Vec::new();
    let mut run = 0;
    for &value in values {
        if value < 0.0 {
            run += 1;
        } else if run > 0 {
            streaks.push(run);
            run = 0;
        }
    }
    if run > 0 {
        streaks.push(run);
    }
    streaks
}

/// The geometric mean of returns (module docs); NaN for none.
pub fn geometric_mean(returns: &[f64]) -> f64 {
    if returns.is_empty() {
        return f64::NAN;
    }
    if returns.iter().any(|&r| r <= -1.0) {
        return -1.0;
    }
    // Logarithms keep a long product of growth factors from overflowing.
    let log_growth: f64 = returns.iter().map(|r| r.ln_1p()).sum();
    (log_growth / returns.len() as f64).exp_m1()
}

/// Counting statistics of session returns in date order (module docs).
///
/// Errors: `NotFinite` for a return that is NaN or infinite.
pub fn session_stats(returns: &[f64]) -> Result<SessionStats, MetricsError> {
    check_finite(returns, "session return")?;
    let wins: Vec<f64> = returns.iter().copied().filter(|&r| r > 0.0).collect();
    let losses: Vec<f64> = returns.iter().copied().filter(|&r| r < 0.0).collect();
    let decided = wins.len() + losses.len();
    let extreme = |pick: fn(f64, f64) -> f64| returns.iter().copied().reduce(pick);
    let (max_consecutive_wins, max_consecutive_losses) = longest_runs(returns);
    Ok(SessionStats {
        sessions: returns.len(),
        expected_return: geometric_mean(returns),
        average_return: mean(returns),
        best: extreme(f64::max).unwrap_or(f64::NAN),
        worst: extreme(f64::min).unwrap_or(f64::NAN),
        winning: wins.len(),
        losing: losses.len(),
        win_rate: if decided == 0 {
            f64::NAN
        } else {
            wins.len() as f64 / decided as f64
        },
        average_winning: mean(&wins),
        average_losing: mean(&losses),
        max_consecutive_wins,
        max_consecutive_losses,
        loss_streaks: loss_streaks(returns),
    })
}
