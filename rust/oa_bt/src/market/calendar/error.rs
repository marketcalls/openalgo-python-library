//! Errors raised by the calendar, and the date formatting its messages share.

use std::fmt;

use crate::model::{civil_from_days, ist_day, ist_nanos_of_day, Day, Exchange, TsNanos};
use crate::model::{MAX_DAY, MIN_DAY, NANOS_PER_SEC};

use super::csv::TableError;

/// "YYYY-MM-DD" for a day number.
pub fn format_day(day: Day) -> String {
    let (year, month, day_of_month) = civil_from_days(day);
    format!("{year:04}-{month:02}-{day_of_month:02}")
}

/// "YYYY-MM-DD HH:MM:SS IST" for a UTC instant.
pub fn format_ist(ts: TsNanos) -> String {
    let sec_of_day = (ist_nanos_of_day(ts) / NANOS_PER_SEC) as u32;
    format!(
        "{} {} IST",
        format_day(ist_day(ts)),
        format_sec_of_day(sec_of_day)
    )
}

/// "HH:MM:SS" for seconds since IST midnight.
fn format_sec_of_day(sec_of_day: u32) -> String {
    format!(
        "{:02}:{:02}:{:02}",
        sec_of_day / 3_600,
        (sec_of_day / 60) % 60,
        sec_of_day % 60
    )
}

/// A calendar request that cannot be answered. Every message says what to change.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CalendarError {
    /// An exchange string that is not an OpenAlgo exchange code.
    UnknownExchange(String),
    /// A date outside the range whose instants fit in UTC nanoseconds.
    UnsupportedDay(Day),
    /// Session times (seconds since IST midnight) that are not `open < close <= 24:00`.
    InvalidSession {
        exchange: Exchange,
        day: Option<Day>,
        open_sec: u32,
        close_sec: u32,
    },
    /// A date range whose start is after its end.
    InvalidRange { from: Day, to: Day },
    /// A base or bar interval, in nanoseconds, that is not usable.
    InvalidInterval { nanos: i64 },
    /// An empty exchange list where at least one exchange is needed.
    NoExchanges,
    /// A grid that would hold more slices than the limit.
    GridTooLarge { points: u64, limit: u64 },
    /// A window of calendar days too short to annualise.
    WindowTooShort { days: i64, minimum: i64 },
    /// One of the embedded tables is malformed.
    Table(TableError),
}

impl From<TableError> for CalendarError {
    fn from(error: TableError) -> Self {
        CalendarError::Table(error)
    }
}

fn unknown_exchange_message(code: &str) -> String {
    let known: Vec<&str> = Exchange::ALL.iter().map(|e| e.code()).collect();
    format!(
        "unknown exchange {code:?}: use one of {} (upper case)",
        known.join(", ")
    )
}

fn invalid_session_message(
    exchange: Exchange,
    day: Option<Day>,
    open_sec: u32,
    close_sec: u32,
) -> String {
    let on_day = day
        .map(|day| format!(" on {}", format_day(day)))
        .unwrap_or_default();
    format!(
        "invalid {} session{on_day}: open {} and close {} IST must satisfy open < close <= \
         24:00 on one IST date (a session that crosses IST midnight is not supported; check \
         the source of these times)",
        exchange.code(),
        format_sec_of_day(open_sec),
        format_sec_of_day(close_sec)
    )
}

impl fmt::Display for CalendarError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CalendarError::UnknownExchange(code) => f.write_str(&unknown_exchange_message(code)),
            CalendarError::UnsupportedDay(_) => write!(
                f,
                "the date is outside the supported range: use a date from {} to {} \
                 (instants are stored as UTC nanoseconds)",
                format_day(MIN_DAY),
                format_day(MAX_DAY)
            ),
            CalendarError::InvalidSession {
                exchange,
                day,
                open_sec,
                close_sec,
            } => f.write_str(&invalid_session_message(
                *exchange, *day, *open_sec, *close_sec,
            )),
            CalendarError::InvalidRange { from, to } => write!(
                f,
                "invalid date range: start {} is after end {}",
                format_day(*from),
                format_day(*to)
            ),
            CalendarError::InvalidInterval { nanos } => write!(
                f,
                "invalid interval of {nanos} ns: an intraday interval must be positive, a \
                 whole number of seconds and shorter than one day (use the daily base for D)"
            ),
            CalendarError::NoExchanges => {
                write!(f, "no exchanges given: pass at least one exchange code")
            }
            CalendarError::GridTooLarge { points, limit } => write!(
                f,
                "the requested grid has {points} slices, above the limit of {limit}: use a \
                 shorter date range or a coarser base interval"
            ),
            CalendarError::WindowTooShort { days, minimum } => write!(
                f,
                "a window of {days} calendar days is too short to annualise: sessions per \
                 year needs at least {minimum} days, so that weekends and weekdays are \
                 counted in proportion"
            ),
            CalendarError::Table(error) => error.fmt(f),
        }
    }
}

impl std::error::Error for CalendarError {}
