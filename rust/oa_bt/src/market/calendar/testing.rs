//! Builders shared by the calendar and resampling tests.
//!
//! `ist` is plain epoch arithmetic, written out here so that inputs built with it do
//! not depend on the helpers under test. Tests that pin behaviour write their expected
//! instants as literal UTC nanoseconds next to the IST time they stand for.

use crate::model::{days_from_civil, Day, TsNanos};

use super::Calendar;

pub(crate) const MINUTE: i64 = 60_000_000_000;
const SECONDS_PER_DAY: i64 = 86_400;
const IST_OFFSET_SECONDS: i64 = 19_800;

/// A calendar with the built-in tables.
pub(crate) fn calendar() -> Calendar {
    Calendar::new().expect("the embedded tables parse")
}

/// The day number of a calendar date.
pub(crate) fn d(year: i32, month: u32, day: u32) -> Day {
    days_from_civil(year, month, day)
}

/// The UTC instant of an IST date and clock time.
pub(crate) fn ist(day: Day, hour: i64, minute: i64) -> TsNanos {
    let seconds =
        i64::from(day) * SECONDS_PER_DAY + hour * 3_600 + minute * 60 - IST_OFFSET_SECONDS;
    seconds * 1_000_000_000
}

/// Bar opens every `interval_nanos` from `from` up to, not including, `to` (IST clock
/// times on `day`).
pub(crate) fn opens(
    day: Day,
    from: (i64, i64),
    to: (i64, i64),
    interval_nanos: i64,
) -> Vec<TsNanos> {
    let (start, end) = (ist(day, from.0, from.1), ist(day, to.0, to.1));
    (0..)
        .map(|k| start + k * interval_nanos)
        .take_while(|&ts| ts < end)
        .collect()
}
