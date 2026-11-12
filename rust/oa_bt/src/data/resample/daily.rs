//! Intraday bars to one bar per IST session date.

use crate::market::calendar::Calendar;
use crate::model::{Exchange, NANOS_PER_DAY};

use super::bars::{BarsIn, BarsOut};
use super::error::ResampleError;
use super::runs::{aggregate_runs, Admission};
use super::{ClosedDayPolicy, Resampled};

/// Resample intraday bars of `base_nanos` (nanoseconds) to one bar per IST session
/// date.
///
/// `ts_open` is the session open and `ts_close` the session close from the calendar,
/// whether or not the first and last base bars exist. Bars outside the session are
/// dropped and counted in the result, so a day with no session bar has no daily bar.
///
/// Errors: `InvalidInterval` unless `base_nanos` is positive and shorter than a day;
/// the errors of `BarsIn::validate`; `ClosedDay` for a bar on a date with no session
/// unless the policy drops it.
pub fn resample_daily(
    calendar: &Calendar,
    exchange: Exchange,
    bars: &BarsIn<'_>,
    base_nanos: i64,
    closed_day: ClosedDayPolicy,
) -> Result<Resampled, ResampleError> {
    if base_nanos <= 0 || base_nanos >= NANOS_PER_DAY {
        return Err(ResampleError::InvalidInterval {
            base_nanos,
            target_nanos: NANOS_PER_DAY,
        });
    }
    bars.validate()?;

    // The lattice interval is irrelevant here: bars are grouped by session.
    let mut admission = Admission::new(calendar, exchange, base_nanos, base_nanos, closed_day);
    let mut out = BarsOut::default();
    let empty_buckets = aggregate_runs(
        bars,
        |_, ts| Ok(admission.admit(ts)?.map(|bar| bar.session)),
        |session, aggregate| out.push(session.open, session.close, aggregate),
    )?;
    let mut dropped = admission.dropped;
    dropped.empty_buckets = empty_buckets;
    Ok(Resampled { bars: out, dropped })
}
