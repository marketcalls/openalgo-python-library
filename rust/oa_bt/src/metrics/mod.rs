//! Performance metrics of a run: returns, risk, drawdowns, trade statistics, exposure,
//! benchmark-relative figures, the distribution of session returns, rolling statistics,
//! Monte Carlo resampling and the combined equity of several strategies.
//!
//! The kernel works on plain columns and knows nothing about Python or calendars:
//! the caller passes the equity series, the cash flows, the closed trades and the
//! annualisation factor (sessions per year, which the Python layer takes from the
//! session calendar).
//!
//! Conventions shared by every function:
//!
//! - Timestamps are UTC nanoseconds (`model::TsNanos`), strictly increasing within a
//!   series. The session date of a point is the IST calendar date of its timestamp
//!   (`model::ist_day`); no session crosses IST midnight.
//! - Money is `f64` rupees. Returns, rates and drawdowns are fractions (0.01 is one
//!   percent); a drawdown is zero or negative.
//! - The opening wealth is the capital before any P&L of the run. A caller that knows
//!   it passes it (`curve::wealth`), and every point of an equity series is then the
//!   marked equity at that instant: a backtest's first point already holds the P&L of
//!   its first slice. Without it, the first point is taken as the starting equity and
//!   every later point is a mark. The opening time, the instant the run opened with
//!   that capital, is passed the same way where elapsed time depends on it
//!   (`drawdown`); without it the opening is timed at the first point.
//! - Returns are time weighted: a cash flow changes the equity but is not a return
//!   (`curve`).
//! - A figure that the data cannot define is NaN (for example a win rate with no
//!   trades), and a ratio whose denominator is zero while its numerator is positive is
//!   infinite (a profit factor with no losing trade). Each function says which.
//! - Nothing panics on input: unusable input is a `MetricsError` naming the row.
//! - Every definition cites its source (a paper or a textbook) in its module docs.
//!
//! Modules: `curve` (the wealth index and session returns), `drawdown`, `ratios`
//! (volatility, Sharpe, Sortino, CAGR, Calmar), `adjusted` (smart, adjusted and
//! probabilistic ratios), `moments` (skewness, kurtosis, autocorrelation, quantiles),
//! `normal` (the normal distribution function), `sessions` (counting statistics of
//! session returns), `tails` (value at risk, expected shortfall, tail, outlier, Omega
//! and gain-to-pain ratios), `ulcer` (ulcer and serenity indexes, path statistics),
//! `trades`, `trade_ratios` (Kelly, risk of ruin, CPC index), `exposure`, `benchmark`,
//! `rolling`, `montecarlo` with its generator `rng`, `combine` and `error`.

mod adjusted;
mod benchmark;
mod combine;
mod curve;
mod drawdown;
mod error;
mod exposure;
mod moments;
mod montecarlo;
mod normal;
mod ratios;
mod rng;
mod rolling;
mod sessions;
mod tails;
mod trade_ratios;
mod trades;
mod ulcer;

#[cfg(test)]
mod tests;

pub use adjusted::{adjusted_ratios, autocorrelation_penalty, probabilistic_ratio, AdjustedRatios};
pub use benchmark::{benchmark_stats, BenchmarkStats};
pub use combine::combine_equity;
pub use curve::{session_returns, wealth, CashFlows, EquityCurve, SessionReturns, Wealth};
pub use drawdown::{drawdowns, DrawdownEpisode, Drawdowns};
pub use error::MetricsError;
pub use exposure::exposure;
pub use moments::{autocorrelation, excess_kurtosis, percentiles, skewness};
pub use montecarlo::{
    bootstrap_sessions, bootstrap_trades, MonteCarloRuns, MonteCarloSpec, BAND_PROBABILITIES,
    MAX_BAND_POINTS,
};
pub use normal::normal_cdf;
pub use ratios::{cagr, calmar, per_period_rate, return_stats, ReturnStats};
pub use rng::SplitMix64;
pub use rolling::{rolling_stats, RollingStats};
pub use sessions::{geometric_mean, loss_streaks, session_stats, SessionStats};
pub use tails::{
    gain_to_pain, omega, tail_stats, TailStats, OUTLIER_PROBABILITY, TAIL_PROBABILITY,
};
pub use trade_ratios::{risk_of_ruin, trade_ratios, TradeRatios};
pub use trades::{trade_stats, TradeColumns, TradeStats};
pub use ulcer::{
    close_drawdowns, conditional_drawdown, path_stats, serenity_index, ulcer_index,
    ulcer_performance_index, PathStats,
};

/// Days in an average calendar year (the Julian year). CAGR measures elapsed time in
/// these years, and the session calendar scales its session count to the same length
/// when it reports sessions per year, so the two annualisations agree.
pub const DAYS_PER_YEAR: f64 = 365.25;

/// Shortest elapsed time, in calendar days, over which a return is annualised. Below
/// four weeks the compounding exponent dominates the figure (one percent earned in a
/// single session annualises to several hundred percent), so CAGR and Calmar are NaN.
/// Four weeks is also the shortest window the session calendar annualises.
pub const MIN_CAGR_DAYS: f64 = 28.0;

/// Largest disagreement, in rupees, accepted between a trade's gross P&L minus its
/// costs and its net P&L: one paisa, because a table rounded to the paisa column by
/// column can differ from its own arithmetic by that much.
pub const PNL_TOLERANCE_RUPEES: f64 = 0.01;
