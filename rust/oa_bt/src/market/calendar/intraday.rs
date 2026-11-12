//! Default intraday times per segment: the last instant an intraday (MIS) entry may
//! fill, and the instant intraday positions are squared off.
//!
//! The values are data: `intraday_defaults.csv`, dated and sourced row by row (see
//! `tables`). A row gives clock times for the segment, or none (MCX), and two margins
//! in minutes before the session close.
//!
//! How a row is applied to a session:
//!
//! - The clock times apply when they fall after the session open and are not later
//!   than the margins before that day's close. That is every regular session.
//! - Otherwise both times are the margins before that day's close: MCX on every day,
//!   and any special session the clock times do not fit (an evening Muhurat session,
//!   a session that closes early). The result says which rule applied.
//!
//! Limits. The times are per exchange, not per instrument. From 2026-08-03 to
//! 2026-12-04 continuous trading in cash stocks that have derivatives ended at 15:15,
//! when the closing auction started, and brokers squared such positions off from
//! 15:00. The NSE and BSE rows from 2026-08-03 (14:50 and 15:00) are no later than any
//! broker time that was found, and are marked unverified until the maintainer confirms
//! them. The 15:15 end itself is not modelled: the cash session runs to 15:30 for
//! every stock. Nor is the 15:30 stop of index derivatives on their expiry day (from
//! 2026-12-07).

use crate::model::{Day, Exchange, TsNanos, NANOS_PER_MIN, NANOS_PER_SEC};

use super::session::{calendar_exchange, Session};
use super::store::{ts_at, Calendar};
use super::tables::{row_on, IntradayRow};

/// Default intraday times for one exchange and session, as UTC nanoseconds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IntradayTimes {
    /// An intraday entry is rejected when its fill bar would open at or after this.
    pub entry_until: TsNanos,
    /// Intraday positions are squared off at this instant.
    pub square_off: TsNanos,
    /// Open of the session these times belong to.
    pub session_open: TsNanos,
    /// Close of the session these times belong to.
    pub session_close: TsNanos,
    /// True when the times were taken relative to the session close rather than from
    /// the clock times of the segment.
    pub relative_to_close: bool,
    /// True when the times rest on something unverified: an unverified row of the
    /// defaults table, or a session close that is itself unverified when the times
    /// are relative to it.
    pub unverified: bool,
}

fn times_for_session(row: &IntradayRow, session: &Session) -> IntradayTimes {
    let before_close = |minutes: u32| session.close - i64::from(minutes) * NANOS_PER_MIN;
    let relative_entry = before_close(row.entry_before_close_min);
    let relative_square_off = before_close(row.square_off_before_close_min);
    let midnight = ts_at(session.day, 0);
    let at_clock = |sec: u32| midnight + i64::from(sec) * NANOS_PER_SEC;
    let clock = row
        .clock_sec
        .map(|(entry_sec, square_off_sec)| (at_clock(entry_sec), at_clock(square_off_sec)));
    let (entry, square_off, relative_to_close) = match clock {
        Some((entry, square_off))
            if entry > session.open
                && entry <= relative_entry
                && square_off <= relative_square_off =>
        {
            (entry, square_off, false)
        }
        _ => (relative_entry, relative_square_off, true),
    };
    IntradayTimes {
        // A session shorter than the margins leaves no entry window.
        entry_until: entry.max(session.open),
        square_off: square_off.max(session.open),
        session_open: session.open,
        session_close: session.close,
        relative_to_close,
        unverified: !row.verified || (relative_to_close && session.hours_unverified),
    }
}

/// Default intraday times of an exchange on an IST date, or `None` when the exchange
/// has no session that day (or the date is outside the supported range).
pub fn default_intraday_times(
    calendar: &Calendar,
    exchange: Exchange,
    day: Day,
) -> Option<IntradayTimes> {
    let session = calendar.session(exchange, day)?;
    let rows = &calendar.intraday[calendar_exchange(exchange) as usize];
    row_on(rows, day).map(|row| times_for_session(row, &session))
}
