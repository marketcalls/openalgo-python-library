//! Session-aligned resampling of base bars (1 minute or daily) to coarser intervals,
//! equal to the OpenAlgo server's Historify aggregation.
//!
//! The rule is the server's (`database/historify_db.py`, `_get_aggregated_ohlcv` and
//! `_get_daily_aggregated_ohlcv`):
//!
//! - Intraday (`resample_intraday`): a bucket of interval N starts at IST midnight plus
//!   the exchange's regular open plus `k * N`, on every date, special sessions
//!   included. Buckets are decided by the clock, never by counting rows, so a missing
//!   1-minute bar leaves the other bars in their buckets and the last bucket of a
//!   session may be short.
//! - Weekly, monthly, quarterly, yearly (`resample_period`): weeks anchored on Monday,
//!   and calendar months, quarters and years of the IST session date.
//! - Open is the first, high the maximum, low the minimum, close the last, volume the
//!   sum and open interest the last value of the bucket.
//!
//! `ts_open` of an output bar is its lattice point and `ts_close` the lattice end, cut
//! at the session close. For daily and longer bars the bounds are the open of the
//! first session and the close of the last session of the period, from the calendar.
//!
//! Every remaining difference from the server:
//!
//! - Rows outside the session are dropped, never merged into a session bucket: a base
//!   bar that ends at or before the session open (pre-open) or opens at or after the
//!   session close (post-close). The server keeps both in its REST read and drops
//!   only rows before the regular open in its Parquet export. The dropped rows are
//!   counted and returned.
//! - Rows on a date with no session are rejected (or dropped and counted, by option).
//!   The server aggregates whatever it stored.
//! - Unknown values (NaN) are skipped in every column: open is the first known open,
//!   close the last known close and open interest the last known value of the bucket.
//!   The server's `FIRST(open ORDER BY timestamp)`, `LAST(close ...)` and `LAST(oi
//!   ...)` take the first or last row's value as it is, and it stores 0 where the
//!   broker gave no open interest. With no unknown values the two agree.
//! - A bucket with no known price emits no bar; it is counted.
//! - Period labels: the server stamps a W, M, Q or Y bar with 00:00 UTC of the first
//!   calendar date of the period. Here `ts_open` is the open of the period's first
//!   session and `ts_close` the close of its last.
//! - The server does not build daily bars from 1-minute bars; `resample_daily` does.
//!   Its close is the last traded price of the session's bars, which is not the
//!   exchange's official closing price.
//!
//! Modules: `bars` (the column types and the bucket aggregate), `runs` (admitting bars
//! and aggregating runs of rows), `intraday`, `daily`, `period` (the three resamplers)
//! and `error`.

mod bars;
mod daily;
mod error;
mod intraday;
mod period;
mod runs;

#[cfg(test)]
mod tests;

pub use bars::{BarsIn, BarsOut};
pub use daily::resample_daily;
pub use error::ResampleError;
pub use intraday::resample_intraday;
pub use period::{resample_period, CalendarUnit};

/// What to do with bars on a date the calendar has no session for.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ClosedDayPolicy {
    /// Stop with an error that names the date.
    #[default]
    Error,
    /// Leave those bars out of the result and count them.
    Drop,
}

/// Input rows that did not reach an output bar, by reason.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Dropped {
    /// Bars that end at or before the session open.
    pub pre_open_bars: u64,
    /// Bars that open at or after the session close.
    pub post_close_bars: u64,
    /// Bars on a date with no session (only with `ClosedDayPolicy::Drop`).
    pub closed_day_bars: u64,
    /// Buckets whose bars had no known price, so no bar was emitted.
    pub empty_buckets: u64,
}

/// The result of a resampling: the bars, and what was left out.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Resampled {
    pub bars: BarsOut,
    pub dropped: Dropped,
}
