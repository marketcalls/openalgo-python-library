//! Intraday bars to coarser intraday bars.

use crate::market::calendar::Calendar;
use crate::model::{Exchange, TsNanos, NANOS_PER_DAY};

use super::bars::{BarsIn, BarsOut};
use super::error::ResampleError;
use super::runs::{aggregate_runs, Admission};
use super::{ClosedDayPolicy, Resampled};

/// One output bar: its lattice point and the instant it is complete.
#[derive(Clone, Copy, PartialEq, Eq)]
struct Bucket {
    open: TsNanos,
    close: TsNanos,
}

/// Resample intraday bars of `base_nanos` to `target_nanos` (both in nanoseconds).
///
/// A bucket starts at IST midnight plus the exchange's regular open plus a whole
/// number of target intervals, on every date; its `ts_close` is the lattice end, cut
/// at the session close. The first bucket of a session whose open is not a lattice
/// point starts before the session open and holds only bars of the session. Bars
/// outside the session are dropped and counted in the result.
///
/// Errors: `InvalidInterval` unless `target_nanos` is a whole multiple of `base_nanos`
/// (equal is allowed: the bars are then only filtered and given a `ts_close`) and
/// shorter than one day; the errors of `BarsIn::validate`; `ClosedDay` for a bar on a
/// date with no session unless the policy drops it.
pub fn resample_intraday(
    calendar: &Calendar,
    exchange: Exchange,
    bars: &BarsIn<'_>,
    base_nanos: i64,
    target_nanos: i64,
    closed_day: ClosedDayPolicy,
) -> Result<Resampled, ResampleError> {
    let usable = base_nanos > 0
        && target_nanos >= base_nanos
        && target_nanos % base_nanos == 0
        && target_nanos < NANOS_PER_DAY;
    if !usable {
        return Err(ResampleError::InvalidInterval {
            base_nanos,
            target_nanos,
        });
    }
    bars.validate()?;

    let mut admission = Admission::new(calendar, exchange, base_nanos, target_nanos, closed_day);
    let mut out = BarsOut::default();
    let empty_buckets = aggregate_runs(
        bars,
        |_, ts| {
            Ok(admission.admit(ts)?.map(|bar| Bucket {
                open: bar.bucket_start,
                close: (bar.bucket_start + target_nanos).min(bar.session.close),
            }))
        },
        |bucket, aggregate| out.push(bucket.open, bucket.close, aggregate),
    )?;
    let mut dropped = admission.dropped;
    dropped.empty_buckets = empty_buckets;
    Ok(Resampled { bars: out, dropped })
}
