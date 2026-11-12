//! Session calendars per exchange: regular hours, holidays and special sessions
//! supplied as data, the bar lattice, and the union-calendar grid the kernel iterates.
//!
//! How a session is resolved for an exchange and an IST date, first match wins:
//!
//! 1. A dated rule for that exchange and date (`Calendar::set_closed`,
//!    `Calendar::set_session`): a trading holiday, a special session such as Muhurat
//!    or a weekend session, or a per-date override of the session times.
//! 2. Saturday and Sunday are closed.
//! 3. The exchange's regular hours in force on that date (`regular_hours.csv`).
//!
//! The crate ships regular hours and default intraday times only, as dated and sourced
//! tables (`tables`). Holiday lists and special sessions are data that the Python
//! package loads from its built-in files or from an OpenAlgo `holidays(year)` response
//! and pushes in through the setters. A year for which nothing was loaded is reported
//! as `Coverage::Template` (weekday rule only), never as sourced.
//!
//! Conventions:
//! - A session is the half-open interval `[open, close)` in UTC nanoseconds. No
//!   session crosses IST midnight.
//! - Session times are stored as seconds since IST midnight (09:15 is 33_300).
//! - `NSE_INDEX` follows the NSE calendar and `BSE_INDEX` the BSE calendar.
//! - Bars of interval N open at the exchange's regular open plus `k * N` on every
//!   date (`lattice`), and a session bar closes at the lattice end or the session
//!   close, whichever comes first.
//! - Dates run from `model::MIN_DAY` to `model::MAX_DAY`. Outside that range a query
//!   answers `None` and anything else a typed error; nothing panics.
//!
//! Modules: `csv` and `tables` (the embedded tables), `session` (value types), `store`
//! (the calendar and its setters), `query` (sessions out of a calendar), `lattice`
//! (the bar lattice), `grid` (the union grid), `intraday` (default entry and
//! square-off times), `check`, `offset`, `label`, `validate` and `series_error`
//! (checking an intraday series against the calendar).

mod check;
pub(crate) mod csv;
mod error;
mod grid;
mod intraday;
mod label;
mod lattice;
mod offset;
mod query;
mod series_error;
mod session;
mod store;
mod tables;
mod validate;

#[cfg(test)]
pub(crate) mod testing;
#[cfg(test)]
mod tests;

pub use check::{check_intraday, IntradayCheck};
pub use csv::TableError;
pub use error::{format_day, format_ist, CalendarError};
pub use grid::{union_grid, BaseInterval, UnionGrid, MAX_GRID_POINTS, NO_BAR};
pub use intraday::{default_intraday_times, IntradayTimes};
pub use lattice::Placement;
pub(crate) use lattice::{is_supported_instant, LatticeCursor};
pub use offset::{OffsetCheck, ShiftDirection, ShiftEvidence};
pub use query::{SessionsPerYear, MIN_ANNUALISATION_DAYS};
pub use series_error::SeriesError;
pub use session::{calendar_exchange, Coverage, DayRule, RuleClose, Session, SessionKind};
pub use store::Calendar;
pub use tables::RegularHours;
pub use validate::{validate_intraday, ValidateOptions};

/// Seconds in one day; a session close may equal this value (IST midnight).
pub const SECS_PER_DAY: u32 = 86_400;

/// Seconds since IST midnight for an hour and minute, for example `hm(9, 15)`.
pub const fn hm(hour: u32, minute: u32) -> u32 {
    hour * 3_600 + minute * 60
}
