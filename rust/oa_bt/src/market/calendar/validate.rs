//! Validation of an intraday series against the session calendar (FR-DATA-06).
//!
//! The rule is that session bars sit on the bar lattice (`lattice`): the regular open
//! plus whole intervals. What is an error and what is only counted:
//!
//! - an error: a 5 hour 30 minute offset (see `offset` for when it is called), bars
//!   labelled by their close time (see `label`), bars that overlap the session off the
//!   lattice, and bars on a date with no session unless the caller allows them;
//! - counted for the caller to drop or report: pre-open and post-close bars, missing
//!   bars and days, and a first bar later than the session's first bar;
//! - a warning the caller should pass on: an inconclusive offset check.
//!
//! Nothing is shifted or dropped here.

use crate::model::{ist_day, Exchange, TsNanos};

use super::check::{check_intraday, IntradayCheck};
use super::offset::OffsetCheck;
use super::series_error::SeriesError;
use super::store::Calendar;

/// What `validate_intraday` does with bars on a date that has no session.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ValidateOptions {
    /// Count bars on dates with no session instead of rejecting the series. The
    /// counts and the per-bar placement let the caller drop them.
    pub allow_closed_days: bool,
}

/// The error for the first counted problem that means wrong timestamps, if any.
fn first_problem(
    calendar: &Calendar,
    exchange: Exchange,
    ts_open: &[TsNanos],
    interval_nanos: i64,
    options: ValidateOptions,
    check: &IntradayCheck,
) -> Option<SeriesError> {
    if let (OffsetCheck::Shifted(evidence), Some(&first_bar)) = (check.offset, ts_open.first()) {
        return Some(SeriesError::Offset {
            exchange,
            first_bar,
            bars: check.bars,
            evidence,
        });
    }
    if let (true, Some(first_at_close)) = (check.close_labelled, check.first_bar_at_close) {
        return Some(SeriesError::CloseLabelled {
            exchange,
            first_at_close,
            days_at_close: check.days_with_bar_at_close,
            days: check.session_days,
        });
    }
    if let Some(ts) = check.first_off_lattice {
        let lattice_anchor = calendar.lattice_anchor(exchange, ist_day(ts)).unwrap_or(ts);
        return Some(SeriesError::OffLattice {
            exchange,
            ts,
            lattice_anchor,
            interval_nanos,
            bars: check.off_lattice_bars,
        });
    }
    if let (Some(day), false) = (check.first_closed_day, options.allow_closed_days) {
        return Some(SeriesError::ClosedDay {
            exchange,
            day,
            days: check.closed_days,
            bars: check.closed_day_bars,
            coverage: calendar.coverage(exchange, day),
        });
    }
    None
}

/// Check an intraday series and fail on what means wrong timestamps.
///
/// `ts_open` holds bar opens in UTC nanoseconds, sorted ascending; `interval_nanos` is
/// the bar interval. On success the report carries every count, the placement of each
/// bar and the offset outcome, which may be inconclusive.
///
/// Errors, in this order: `InvalidInterval`, `UnsupportedTimestamp`, `NotSorted`,
/// `Offset` (the timestamps are 5 hours 30 minutes off), `CloseLabelled` (bars stamped
/// at their close), `OffLattice` (session bars off the lattice) and `ClosedDay` (bars
/// on a date with no session, unless allowed).
pub fn validate_intraday(
    calendar: &Calendar,
    exchange: Exchange,
    ts_open: &[TsNanos],
    interval_nanos: i64,
    options: ValidateOptions,
) -> Result<IntradayCheck, SeriesError> {
    let check = check_intraday(calendar, exchange, ts_open, interval_nanos)?;
    match first_problem(calendar, exchange, ts_open, interval_nanos, options, &check) {
        Some(error) => Err(error),
        None => Ok(check),
    }
}
