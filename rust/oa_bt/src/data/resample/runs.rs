//! The two steps every resampler shares: deciding whether a bar takes part, and
//! aggregating runs of consecutive rows that fall in the same bucket.

use crate::market::calendar::{Calendar, LatticeCursor, Placement, Session};
use crate::model::{ist_day, Exchange, TsNanos};

use super::bars::{Accumulator, BarsIn};
use super::error::ResampleError;
use super::{ClosedDayPolicy, Dropped};

/// A base bar that takes part: its session and the start of its target bucket.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Admitted {
    pub(super) session: Session,
    /// The lattice point of the target interval at or before the bar's open.
    pub(super) bucket_start: TsNanos,
}

/// The error for a bar on a date the calendar has no session for.
pub(super) fn closed_day_error(
    calendar: &Calendar,
    exchange: Exchange,
    ts: TsNanos,
) -> ResampleError {
    let day = ist_day(ts);
    ResampleError::ClosedDay {
        exchange,
        day,
        ts,
        coverage: calendar.coverage(exchange, day),
    }
}

/// Decides which intraday base bars take part, and counts the ones that do not.
pub(super) struct Admission<'a> {
    calendar: &'a Calendar,
    exchange: Exchange,
    cursor: LatticeCursor<'a>,
    base_nanos: i64,
    target_nanos: i64,
    closed_day: ClosedDayPolicy,
    pub(super) dropped: Dropped,
}

impl<'a> Admission<'a> {
    /// `base_nanos` is the interval of the input bars and `target_nanos` the interval
    /// whose lattice the buckets sit on.
    pub(super) fn new(
        calendar: &'a Calendar,
        exchange: Exchange,
        base_nanos: i64,
        target_nanos: i64,
        closed_day: ClosedDayPolicy,
    ) -> Admission<'a> {
        let cursor = LatticeCursor::new(calendar, exchange);
        let dropped = Dropped::default();
        Admission {
            calendar,
            exchange,
            cursor,
            base_nanos,
            target_nanos,
            closed_day,
            dropped,
        }
    }

    /// Place the base bar that opens at `ts`. A bar outside the session is counted
    /// and left out (`None`): it is never merged into a session bucket.
    ///
    /// Errors: `ClosedDay` for a bar on a date with no session, unless the policy
    /// drops such bars.
    pub(super) fn admit(&mut self, ts: TsNanos) -> Result<Option<Admitted>, ResampleError> {
        let lattice = self.cursor.at(ts);
        let session = lattice.and_then(|lattice| lattice.session);
        let (Some(lattice), Some(session)) = (lattice, session) else {
            if self.closed_day == ClosedDayPolicy::Error {
                return Err(closed_day_error(self.calendar, self.exchange, ts));
            }
            self.dropped.closed_day_bars += 1;
            return Ok(None);
        };
        match lattice.place(ts, self.base_nanos) {
            Placement::PreOpen => self.dropped.pre_open_bars += 1,
            Placement::PostClose => self.dropped.post_close_bars += 1,
            // A bar off the base lattice still overlaps the session: it is bucketed
            // by its open. Whether such bars are acceptable is the validator's call.
            Placement::InSession | Placement::OffLattice | Placement::ClosedDay => {
                let bucket_start = lattice.bucket_start(ts, self.target_nanos);
                return Ok(Some(Admitted {
                    session,
                    bucket_start,
                }));
            }
        }
        Ok(None)
    }
}

/// Aggregate runs of consecutive rows that share a bucket key.
///
/// `key_of` gives the bucket of a row from its index and open instant, or `None` to
/// leave the row out. `emit` receives the key and the aggregate of each finished run
/// that holds at least one known price; runs with none are counted and returned.
/// `bars` must have been validated.
pub(super) fn aggregate_runs<K: PartialEq>(
    bars: &BarsIn<'_>,
    mut key_of: impl FnMut(usize, TsNanos) -> Result<Option<K>, ResampleError>,
    mut emit: impl FnMut(&K, &Accumulator),
) -> Result<u64, ResampleError> {
    let mut empty_buckets = 0u64;
    let mut finish = |key: &K, bucket: &Accumulator| {
        if bucket.has_price() {
            emit(key, bucket);
        } else {
            empty_buckets += 1;
        }
    };
    let mut run: Option<(K, Accumulator)> = None;
    for (row, &ts) in bars.ts_open.iter().enumerate() {
        let Some(key) = key_of(row, ts)? else {
            continue;
        };
        match run.as_mut() {
            Some((current, bucket)) if *current == key => bucket.push(bars, row),
            _ => {
                if let Some((done, bucket)) = run.take() {
                    finish(&done, &bucket);
                }
                let mut bucket = Accumulator::new();
                bucket.push(bars, row);
                run = Some((key, bucket));
            }
        }
    }
    if let Some((done, bucket)) = run {
        finish(&done, &bucket);
    }
    Ok(empty_buckets)
}
