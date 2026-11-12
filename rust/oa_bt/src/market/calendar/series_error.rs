//! Why an intraday series was rejected, with messages that say what to do about it.

use std::fmt;

use crate::model::{Day, Exchange, TsNanos, NANOS_PER_SEC};

use super::error::{format_day, format_ist};
use super::offset::{ShiftDirection, ShiftEvidence};
use super::session::Coverage;

/// An intraday series that does not fit the session calendar. Indices are zero-based
/// row numbers; instants are UTC nanoseconds.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SeriesError {
    /// A bar interval, in nanoseconds, that is not positive and shorter than a day.
    InvalidInterval { nanos: i64 },
    /// A missing timestamp (the int64 minimum) or one outside the supported dates.
    UnsupportedTimestamp { index: usize, ts: TsNanos },
    /// Row `index` is earlier than the row before it.
    NotSorted { index: usize },
    /// The timestamps are 5 hours 30 minutes off.
    Offset {
        exchange: Exchange,
        first_bar: TsNanos,
        bars: u64,
        evidence: ShiftEvidence,
    },
    /// The bars look labelled by their close time although they were read as labelled
    /// by their open.
    CloseLabelled {
        exchange: Exchange,
        first_at_close: TsNanos,
        days_at_close: u32,
        days: u32,
    },
    /// Bars that overlap the session but do not open on a lattice point.
    OffLattice {
        exchange: Exchange,
        ts: TsNanos,
        lattice_anchor: TsNanos,
        interval_nanos: i64,
        bars: u64,
    },
    /// Bars on dates the calendar has no session for.
    ClosedDay {
        exchange: Exchange,
        day: Day,
        days: u32,
        bars: u64,
        coverage: Coverage,
    },
}

fn offset_message(
    exchange: Exchange,
    first_bar: TsNanos,
    bars: u64,
    evidence: ShiftEvidence,
) -> String {
    let (way, cause) = match evidence.direction {
        ShiftDirection::Late5h30 => (
            "late",
            "IST clock times were stored as if they were UTC (an XTS-family broker feed, or \
             naive IST times in a file read as UTC)",
        ),
        ShiftDirection::Early5h30 => (
            "early",
            "UTC instants were treated as IST clock times and converted to UTC a second time",
        ),
    };
    format!(
        "timestamps are 5 hours 30 minutes {way} against the {code} session calendar: \
         {misplaced} of {bars} bars are outside the session or off its lattice as given, \
         {undone} after undoing the offset, which then covers {days} days (first bar: \
         {first}). Likely cause: {cause}. Declare the time zone of the source, or set the \
         per-broker shift flag; timestamps are never shifted silently",
        code = exchange.code(),
        misplaced = evidence.misplaced,
        undone = evidence.misplaced_if_undone,
        days = evidence.covered_days_if_undone,
        first = format_ist(first_bar)
    )
}

fn close_labelled_message(
    exchange: Exchange,
    first_at_close: TsNanos,
    days_at_close: u32,
    days: u32,
) -> String {
    format!(
        "the bars look labelled by their close time, not their open time: on \
         {days_at_close} of {days} {} session days a bar opens at the session close \
         (first at {}) and no day has a bar at the first bar of its session. Served as \
         they are, every bar would close one interval after it really did. Declare that \
         the source stamps bars at their close (for your own data, label=\"close\"); \
         timestamps are never shifted silently",
        exchange.code(),
        format_ist(first_at_close)
    )
}

fn off_lattice_message(
    exchange: Exchange,
    ts: TsNanos,
    lattice_anchor: TsNanos,
    interval_nanos: i64,
    bars: u64,
) -> String {
    format!(
        "{bars} bars are not on the {} bar lattice: the bar at {} does not open at the \
         regular open ({}) plus a whole number of {}-second intervals. Bars are anchored \
         to the exchange's regular open (09:15 for NSE) on every date, not to midnight or \
         to the hour; check the interval and the resampling of the source",
        exchange.code(),
        format_ist(ts),
        format_ist(lattice_anchor),
        interval_nanos / NANOS_PER_SEC
    )
}

fn closed_day_message(
    exchange: Exchange,
    day: Day,
    days: u32,
    bars: u64,
    coverage: Coverage,
) -> String {
    let template_note = match coverage {
        Coverage::Template => {
            " (no holiday data is loaded for that year, so only the weekday rule applies)"
        }
        Coverage::Unverified | Coverage::Sourced => "",
    };
    format!(
        "{bars} bars fall on {days} date(s) with no {} session in the calendar, first on \
         {}{template_note}. If the exchange held a special session, add it to the calendar \
         with add_session; otherwise drop those rows, or allow closed days explicitly",
        exchange.code(),
        format_day(day)
    )
}

impl fmt::Display for SeriesError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match *self {
            SeriesError::InvalidInterval { nanos } => format!(
                "invalid bar interval of {nanos} ns: it must be positive and shorter than one \
                 day for an intraday series"
            ),
            SeriesError::UnsupportedTimestamp { index, ts } => format!(
                "row {index} has no usable timestamp ({ts}): a missing value (NaT) or an \
                 instant outside the supported dates; drop or repair the row before validating"
            ),
            SeriesError::NotSorted { index } => format!(
                "timestamps are not sorted: row {index} is earlier than the row before it; \
                 sort the bars by ts_open before validating"
            ),
            SeriesError::Offset {
                exchange,
                first_bar,
                bars,
                evidence,
            } => offset_message(exchange, first_bar, bars, evidence),
            SeriesError::CloseLabelled {
                exchange,
                first_at_close,
                days_at_close,
                days,
            } => close_labelled_message(exchange, first_at_close, days_at_close, days),
            SeriesError::OffLattice {
                exchange,
                ts,
                lattice_anchor,
                interval_nanos,
                bars,
            } => off_lattice_message(exchange, ts, lattice_anchor, interval_nanos, bars),
            SeriesError::ClosedDay {
                exchange,
                day,
                days,
                bars,
                coverage,
            } => closed_day_message(exchange, day, days, bars, coverage),
        };
        f.write_str(&message)
    }
}

impl std::error::Error for SeriesError {}
