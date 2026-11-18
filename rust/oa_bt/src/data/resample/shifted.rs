//! Bars whose boundaries end earlier (design 17, section 2.7): every bar of the target
//! interval ends `shift_nanos` before the bar the ordinary lattice makes, so a robustness
//! check can ask whether a strategy depends on where its bars start.
//!
//! - Intraday (`resample_intraday_ending_earlier`): the lattice of the target interval
//!   moves `shift_nanos` earlier, inside each session: a bucket starts at the day's
//!   lattice anchor less the shift plus `k * N`, so the first bucket of a session holds
//!   `N - shift` of its bars (it starts before the session open, as any first bucket off
//!   the lattice does), the others `N`, and the last is cut at the session close. Bars
//!   never cross a session.
//! - Daily (`resample_daily_ending_earlier`): each session's bar ends `shift_nanos`
//!   before its close, and the bars after that instant open the next session's bar: the
//!   bar of a session runs from the previous session's close less the shift to its own
//!   close less the shift (both from the calendar; the session's open when the calendar
//!   has no session in the `SEARCH_DAYS` before it). Its decision is made before the
//!   close, with the day's bars so far.
//!
//! The bucket rules (`shifted_start`, `day_bucket`) are the ones the engine's clock
//! uses for the decision instants of a run with shifted bars (`engine::clock`), so the
//! two agree by construction. A shift of zero is the ordinary resampler.

use crate::market::calendar::{Calendar, DayLattice, Session};
use crate::model::{Day, Exchange, TsNanos, NANOS_PER_DAY};

use super::bars::{BarsIn, BarsOut};
use super::error::ResampleError;
use super::runs::{aggregate_runs, Admission};
use super::{ClosedDayPolicy, Resampled};

/// Days searched for a neighbouring session: longer than any run of Indian market
/// holidays (as `engine::submit` searches for the next session).
pub(crate) const SEARCH_DAYS: i32 = 31;

/// The session of the nearest date before (`step` -1) or after (`step` 1) `day` that
/// has one, within `SEARCH_DAYS`.
pub(crate) fn neighbour(
    calendar: &Calendar,
    exchange: Exchange,
    day: Day,
    step: i32,
) -> Option<Session> {
    (1..=SEARCH_DAYS).find_map(|ahead| calendar.session(exchange, day + step * ahead))
}

/// The start of the intraday bucket of `target_nanos` holding the bar that opens at
/// `ts`, on `lattice` moved `shift_nanos` earlier (module docs); a shift of zero is the
/// lattice point at or before `ts`.
pub(crate) fn shifted_start(
    lattice: &DayLattice,
    ts: TsNanos,
    target_nanos: i64,
    shift_nanos: i64,
) -> TsNanos {
    lattice.bucket_start(ts + shift_nanos, target_nanos) - shift_nanos
}

/// The (open, close) of the daily bucket of a bar that opens at `ts` in `session`, each
/// day ending `shift_nanos` early (module docs); `None` for a bar after the end of its
/// session's bar when the calendar has no next session.
pub(crate) fn day_bucket(
    calendar: &Calendar,
    exchange: Exchange,
    session: &Session,
    ts: TsNanos,
    shift_nanos: i64,
) -> Option<(TsNanos, TsNanos)> {
    let end = session.close - shift_nanos;
    if ts >= end {
        let next = neighbour(calendar, exchange, session.day, 1)?;
        return Some((end, next.close - shift_nanos));
    }
    let open = neighbour(calendar, exchange, session.day, -1)
        .map_or(session.open, |previous| previous.close - shift_nanos);
    Some((open, end))
}

/// The daily bucket last found: the session day, whether the bar was after the day's
/// end, and the bucket (`day_bucket`), so a day's bars look the calendar up once.
type CachedDay = (Day, bool, Option<(TsNanos, TsNanos)>);

fn invalid_shift(base_nanos: i64, target_nanos: i64) -> ResampleError {
    ResampleError::InvalidInterval {
        base_nanos,
        target_nanos,
    }
}

/// Resample intraday bars of `base_nanos` to `target_nanos` with every bucket ending
/// `shift_nanos` earlier (module docs).
///
/// Errors: those of `resample_intraday`, and `InvalidInterval` unless `shift_nanos` is
/// a whole multiple of `base_nanos`, zero or more and shorter than `target_nanos`.
pub fn resample_intraday_ending_earlier(
    calendar: &Calendar,
    exchange: Exchange,
    bars: &BarsIn<'_>,
    (base_nanos, target_nanos): (i64, i64),
    shift_nanos: i64,
    closed_day: ClosedDayPolicy,
) -> Result<Resampled, ResampleError> {
    let usable = base_nanos > 0
        && target_nanos >= base_nanos
        && target_nanos % base_nanos == 0
        && target_nanos < NANOS_PER_DAY
        && shift_nanos >= 0
        && shift_nanos % base_nanos == 0
        && shift_nanos < target_nanos;
    if !usable {
        return Err(invalid_shift(base_nanos, target_nanos));
    }
    bars.validate()?;
    let mut admission = Admission::new(calendar, exchange, base_nanos, target_nanos, closed_day)
        .ending_earlier(shift_nanos);
    let mut out = BarsOut::default();
    let empty_buckets = aggregate_runs(
        bars,
        |_, ts| {
            Ok(admission.admit(ts)?.map(|bar| {
                let close = (bar.bucket_start + target_nanos).min(bar.session.close);
                (bar.bucket_start, close)
            }))
        },
        |&(open, close), aggregate| out.push(open, close, aggregate),
    )?;
    let mut dropped = admission.dropped;
    dropped.empty_buckets = empty_buckets;
    Ok(Resampled { bars: out, dropped })
}

/// Resample intraday bars of `base_nanos` to one bar per session, each ending
/// `shift_nanos` before its session's close (module docs).
///
/// Errors: those of `resample_daily`, and `InvalidInterval` unless `shift_nanos` is a
/// whole multiple of `base_nanos`, above zero and shorter than one day.
pub fn resample_daily_ending_earlier(
    calendar: &Calendar,
    exchange: Exchange,
    bars: &BarsIn<'_>,
    base_nanos: i64,
    shift_nanos: i64,
    closed_day: ClosedDayPolicy,
) -> Result<Resampled, ResampleError> {
    let usable = base_nanos > 0
        && base_nanos < NANOS_PER_DAY
        && shift_nanos > 0
        && shift_nanos % base_nanos == 0
        && shift_nanos < NANOS_PER_DAY;
    if !usable {
        return Err(invalid_shift(base_nanos, NANOS_PER_DAY));
    }
    bars.validate()?;
    let mut admission = Admission::new(calendar, exchange, base_nanos, base_nanos, closed_day);
    let mut out = BarsOut::default();
    let mut cached: Option<CachedDay> = None;
    let empty_buckets = aggregate_runs(
        bars,
        |_, ts| {
            let Some(bar) = admission.admit(ts)? else {
                return Ok(None);
            };
            let after = ts >= bar.session.close - shift_nanos;
            if let Some((day, side, bucket)) = cached {
                if day == bar.session.day && side == after {
                    return Ok(bucket);
                }
            }
            let bucket = day_bucket(calendar, exchange, &bar.session, ts, shift_nanos);
            cached = Some((bar.session.day, after, bucket));
            Ok(bucket)
        },
        |&(open, close), aggregate| out.push(open, close, aggregate),
    )?;
    let mut dropped = admission.dropped;
    dropped.empty_buckets = empty_buckets;
    Ok(Resampled { bars: out, dropped })
}
