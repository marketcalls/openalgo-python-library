//! The bar lattice: the one rule the resampler, the validator and the grid share.
//!
//! A bar of interval N opens at IST midnight plus the exchange's regular open plus
//! `k * N`, on every date, special sessions included. This is the rule of the OpenAlgo
//! server's aggregation (`database/historify_db.py`, `_get_aggregated_ohlcv`), so bars
//! the server builds sit on the lattice by construction.
//!
//! A bar covers `[open, open + N)`. Where it sits relative to the session of its IST
//! date:
//!
//! - in session when its interval overlaps the session. The first bar of a session
//!   whose open is not a lattice point starts before the session open (the 14:15 bar
//!   of a session that opens at 14:20, at 15 minutes): it belongs to the session.
//! - pre-open when it ends at or before the session open.
//! - post-close when it opens at or after the session close.
//!
//! The close of a session bar is the lattice end, cut at the session close.

use crate::model::{ist_day, Day, Exchange, TsNanos};

use super::session::Session;
use super::store::Calendar;

/// Where a bar sits relative to the session of its IST date.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Placement {
    /// On the lattice and overlapping the session.
    InSession = 0,
    /// Ends at or before the session open.
    PreOpen = 1,
    /// Opens at or after the session close.
    PostClose = 2,
    /// On a date with no session.
    ClosedDay = 3,
    /// Overlaps the session but does not open on a lattice point.
    OffLattice = 4,
}

/// The session and lattice of one exchange on one IST date.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct DayLattice {
    /// The session of the date, or `None` when the exchange is closed.
    pub(crate) session: Option<Session>,
    /// IST midnight plus the regular open of the date, UTC nanoseconds.
    anchor: TsNanos,
}

impl DayLattice {
    /// The lattice point at or before `ts` for bars of `interval_nanos`.
    pub(crate) fn bucket_start(&self, ts: TsNanos, interval_nanos: i64) -> TsNanos {
        self.anchor + (ts - self.anchor).div_euclid(interval_nanos) * interval_nanos
    }

    /// Where the bar that opens at `ts` and lasts `interval_nanos` sits.
    pub(crate) fn place(&self, ts: TsNanos, interval_nanos: i64) -> Placement {
        let Some(session) = self.session else {
            return Placement::ClosedDay;
        };
        if ts + interval_nanos <= session.open {
            Placement::PreOpen
        } else if ts >= session.close {
            Placement::PostClose
        } else if (ts - self.anchor).rem_euclid(interval_nanos) != 0 {
            Placement::OffLattice
        } else {
            Placement::InSession
        }
    }

    /// The (open, close) of every bar of `interval_nanos` in the session, in order.
    /// The first may open before the session open; the last is cut at the close.
    pub(crate) fn session_bars(
        &self,
        interval_nanos: i64,
    ) -> impl Iterator<Item = (TsNanos, TsNanos)> {
        let session = self.session;
        let first = session.map(|s| self.bucket_start(s.open, interval_nanos));
        let mut next = first;
        std::iter::from_fn(move || {
            let (session, open) = (session?, next?);
            if open >= session.close {
                return None;
            }
            next = Some(open + interval_nanos);
            Some((open, (open + interval_nanos).min(session.close)))
        })
    }

    /// Number of bars of `interval_nanos` in the session.
    pub(crate) fn session_bar_count(&self, interval_nanos: i64) -> u64 {
        match self.session {
            Some(session) => {
                let first = self.bucket_start(session.open, interval_nanos);
                ((session.close - first + interval_nanos - 1) / interval_nanos) as u64
            }
            None => 0,
        }
    }
}

impl Calendar {
    /// The session and lattice of an exchange on an IST date. `None` for a date
    /// outside the supported range.
    pub(crate) fn day_lattice(&self, exchange: Exchange, day: Day) -> Option<DayLattice> {
        let anchor = self.lattice_anchor(exchange, day)?;
        Some(DayLattice {
            session: self.session(exchange, day),
            anchor,
        })
    }
}

/// Walks sorted instants and hands out the lattice of each one's IST date, looking a
/// date up once.
pub(crate) struct LatticeCursor<'a> {
    calendar: &'a Calendar,
    exchange: Exchange,
    cached: Option<(Day, Option<DayLattice>)>,
}

impl<'a> LatticeCursor<'a> {
    pub(crate) fn new(calendar: &'a Calendar, exchange: Exchange) -> LatticeCursor<'a> {
        LatticeCursor {
            calendar,
            exchange,
            cached: None,
        }
    }

    /// The lattice of the IST date of `ts`, or `None` for an unsupported instant.
    pub(crate) fn at(&mut self, ts: TsNanos) -> Option<DayLattice> {
        let day = ist_day(ts);
        match self.cached {
            Some((cached_day, lattice)) if cached_day == day => lattice,
            _ => {
                let lattice = self.calendar.day_lattice(self.exchange, day);
                self.cached = Some((day, lattice));
                lattice
            }
        }
    }
}

/// Whether an instant lies on a supported IST date (and so is not a missing value:
/// the int64 minimum that marks a missing timestamp falls far outside).
pub(crate) fn is_supported_instant(ts: TsNanos) -> bool {
    crate::model::is_supported_day(ist_day(ts))
}
