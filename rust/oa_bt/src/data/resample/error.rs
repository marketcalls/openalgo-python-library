//! Why bars could not be resampled, with messages that say what to do about it.

use std::fmt;

use crate::market::calendar::{format_day, format_ist, CalendarError, Coverage};
use crate::model::{Day, Exchange, TsNanos, NANOS_PER_SEC};

/// Bars that cannot be resampled. Indices are zero-based row numbers; instants are
/// UTC nanoseconds.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ResampleError {
    /// A value column whose length differs from `ts_open`.
    LengthMismatch {
        column: &'static str,
        len: usize,
        expected: usize,
    },
    /// A missing timestamp (the int64 minimum) or one outside the supported dates.
    UnsupportedTimestamp { index: usize, ts: TsNanos },
    /// Row `index` is earlier than the row before it.
    NotSorted { index: usize },
    /// Row `index` repeats the timestamp of the row before it.
    Duplicate { index: usize, ts: TsNanos },
    /// Two daily rows on one IST date.
    DuplicateDay { index: usize, day: Day },
    /// Intervals, in nanoseconds, that cannot be resampled from one to the other.
    InvalidInterval { base_nanos: i64, target_nanos: i64 },
    /// A bar on a date the calendar has no session for.
    ClosedDay {
        exchange: Exchange,
        day: Day,
        ts: TsNanos,
        coverage: Coverage,
    },
    /// The calendar could not answer (for example a malformed embedded table).
    Calendar(CalendarError),
}

fn closed_day_message(exchange: Exchange, day: Day, ts: TsNanos, coverage: Coverage) -> String {
    let template_note = match coverage {
        Coverage::Template => {
            " (no holiday data is loaded for that year, so only the weekday rule applies)"
        }
        Coverage::Unverified | Coverage::Sourced => "",
    };
    format!(
        "a bar at {} falls on {}, a date with no {} session in the calendar{template_note}. If \
         the exchange held a special session, add it to the calendar with add_session; \
         otherwise drop those rows or resample with the closed-day policy set to drop",
        format_ist(ts),
        format_day(day),
        exchange.code()
    )
}

impl fmt::Display for ResampleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ResampleError::LengthMismatch {
                column,
                len,
                expected,
            } => write!(
                f,
                "column {column} has {len} rows but ts_open has {expected}: every bar column \
                 must have the same length"
            ),
            ResampleError::UnsupportedTimestamp { index, ts } => write!(
                f,
                "row {index} has no usable ts_open ({ts}): a missing value (NaT) or an \
                 instant outside the supported dates; drop or repair the row before resampling"
            ),
            ResampleError::NotSorted { index } => write!(
                f,
                "bars are not sorted by ts_open: row {index} is earlier than the row before \
                 it; sort the bars before resampling"
            ),
            ResampleError::Duplicate { index, ts } => write!(
                f,
                "duplicate ts_open at row {index} ({}): drop duplicate bars before resampling",
                format_ist(*ts)
            ),
            ResampleError::DuplicateDay { index, day } => write!(
                f,
                "two daily bars fall on the IST date {} (row {index}): a daily series must \
                 have one bar per session date",
                format_day(*day)
            ),
            ResampleError::InvalidInterval {
                base_nanos,
                target_nanos,
            } => write!(
                f,
                "cannot resample bars of {} seconds to {} seconds: the target must be a whole \
                 multiple of the base interval, at least as long, and shorter than one day \
                 (use the daily resampler for D)",
                base_nanos / NANOS_PER_SEC,
                target_nanos / NANOS_PER_SEC
            ),
            ResampleError::ClosedDay {
                exchange,
                day,
                ts,
                coverage,
            } => f.write_str(&closed_day_message(*exchange, *day, *ts, *coverage)),
            ResampleError::Calendar(error) => error.fmt(f),
        }
    }
}

impl std::error::Error for ResampleError {}
