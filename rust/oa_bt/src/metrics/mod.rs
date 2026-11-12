//! Performance metrics of a run: returns, risk, drawdowns, trade statistics, exposure,
//! benchmark-relative figures and the combined equity of several strategies.
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
//! - The first point of an equity series is the starting equity: the capital before
//!   any P&L of the run. Every later point is the marked equity at that instant.
//! - Returns are time weighted: a cash flow changes the equity but is not a return
//!   (`curve`).
//! - A figure that the data cannot define is NaN (for example a win rate with no
//!   trades), and a ratio whose denominator is zero while its numerator is positive is
//!   infinite (a profit factor with no losing trade). Each function says which.
//! - Nothing panics on input: unusable input is a `MetricsError` naming the row.
//!
//! Modules: `curve` (the wealth index and session returns), `drawdown`, `ratios`
//! (volatility, Sharpe, Sortino, CAGR, Calmar), `trades`, `exposure`, `benchmark`,
//! `combine` and `error`.

mod benchmark;
mod combine;
mod curve;
mod drawdown;
mod error;
mod exposure;
mod ratios;
mod trades;

#[cfg(test)]
mod tests;

pub use benchmark::{benchmark_stats, BenchmarkStats};
pub use combine::combine_equity;
pub use curve::{session_returns, wealth, CashFlows, EquityCurve, SessionReturns, Wealth};
pub use drawdown::{drawdowns, DrawdownEpisode, Drawdowns};
pub use error::MetricsError;
pub use exposure::exposure;
pub use ratios::{cagr, calmar, per_period_rate, return_stats, ReturnStats};
pub use trades::{trade_stats, TradeColumns, TradeStats};

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
