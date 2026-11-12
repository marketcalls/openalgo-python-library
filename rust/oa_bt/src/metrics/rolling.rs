//! Rolling statistics: volatility, Sharpe, Sortino, session win rate and beta over a
//! trailing window of sessions.
//!
//! Definition: the value at session `i` is the statistic of the `w` sessions ending at
//! `i` (sessions `i - w + 1` to `i`), computed exactly as for the whole run: volatility,
//! Sharpe and Sortino as `ratios` defines them (annualised by `sqrt(N)`, with the
//! per-session risk-free rate subtracted), the session win rate as `sessions` defines
//! it, and beta as `benchmark` defines it on the sessions of the window where both
//! returns are known. The first `w - 1` sessions have no full window: NaN.

use super::benchmark::benchmark_stats;
use super::curve::{check_finite, check_length, check_not_infinite};
use super::error::MetricsError;
use super::ratios::return_stats;
use super::sessions::session_stats;

/// Rolling statistics, one value per session (definitions in the module docs).
#[derive(Clone, Debug, PartialEq)]
pub struct RollingStats {
    /// Annualised, a fraction.
    pub volatility: Vec<f64>,
    pub sharpe: Vec<f64>,
    pub sortino: Vec<f64>,
    /// Fraction of the window's sessions that won, among those that won or lost.
    pub win_rate: Vec<f64>,
    /// Beta against the benchmark; empty without a benchmark.
    pub beta: Vec<f64>,
}

/// Rolling statistics of session returns over windows of `window` sessions (module
/// docs). `benchmark` holds the benchmark's session returns on the same dates (NaN where
/// unknown), or is `None`. `periods_per_year` and `risk_free_annual` annualise as in
/// `ratios`.
///
/// Errors: `InvalidWindow` for a window below two sessions; `NotFinite` for a strategy
/// return that is NaN or infinite, or an infinite benchmark return, at its session;
/// `LengthMismatch` for a benchmark of another length; those of `ratios::return_stats`
/// and `benchmark::benchmark_stats`.
pub fn rolling_stats(
    returns: &[f64],
    benchmark: Option<&[f64]>,
    window: usize,
    periods_per_year: f64,
    risk_free_annual: f64,
) -> Result<RollingStats, MetricsError> {
    if window < 2 {
        return Err(MetricsError::InvalidWindow { sessions: window });
    }
    check_finite(returns, "session return")?;
    if let Some(bench) = benchmark {
        check_length(bench.len(), returns.len(), "benchmark returns")?;
        // Checked over the whole series here, not window by window, so the error names
        // the session's row and not its place in the window.
        check_not_infinite(bench, "benchmark return")?;
    }
    let mut out = RollingStats {
        volatility: Vec::with_capacity(returns.len()),
        sharpe: Vec::with_capacity(returns.len()),
        sortino: Vec::with_capacity(returns.len()),
        win_rate: Vec::with_capacity(returns.len()),
        beta: Vec::new(),
    };
    for end in 1..=returns.len() {
        // A window that is not yet full is read as empty: every statistic of no
        // session is NaN.
        let range = end.saturating_sub(window)..end;
        let full = end >= window;
        let slice = returns.get(range.clone()).filter(|_| full).unwrap_or(&[]);
        let stats = return_stats(slice, periods_per_year, risk_free_annual)?;
        out.volatility.push(stats.volatility);
        out.sharpe.push(stats.sharpe);
        out.sortino.push(stats.sortino);
        out.win_rate.push(session_stats(slice)?.win_rate);
        if let Some(bench) = benchmark {
            let paired = bench.get(range).filter(|_| full).unwrap_or(&[]);
            let beta = benchmark_stats(slice, paired, periods_per_year, risk_free_annual)?.beta;
            out.beta.push(beta);
        }
    }
    Ok(out)
}
