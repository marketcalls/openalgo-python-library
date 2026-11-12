//! Counting how an intraday series sits against the session calendar.
//!
//! `check_intraday` places every bar on the lattice (`lattice`), counts what it finds,
//! runs the offset check (`offset`) and the close-label check (`label`). It never
//! judges data. `validate_intraday` (in `validate`) turns the findings that mean
//! "wrong timestamps" into errors.

use crate::model::{ist_day, Day, Exchange, TsNanos, NANOS_PER_DAY};

use super::label::looks_close_labelled;
use super::lattice::{is_supported_instant, DayLattice, LatticeCursor, Placement};
use super::offset::{check_offset, OffsetCheck};
use super::series_error::SeriesError;
use super::session::Coverage;
use super::store::Calendar;

/// Counts from checking one intraday series against the calendar. Bars are identified
/// by their open instant, in UTC nanoseconds.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct IntradayCheck {
    /// Bars in the series.
    pub bars: u64,
    /// IST dates that have at least one bar.
    pub days: u32,
    /// Of those, dates on which the exchange has a session.
    pub session_days: u32,
    /// Dates that have bars although the calendar has no session.
    pub closed_days: u32,
    /// The first such date.
    pub first_closed_day: Option<Day>,
    /// Bars on those dates.
    pub closed_day_bars: u64,
    /// Session days whose first session bar is the first bar of the session.
    pub days_starting_at_open: u32,
    /// Session days whose first session bar is a later one. A count, never an error.
    pub days_starting_late: u32,
    /// The first such date.
    pub first_late_day: Option<Day>,
    /// Session days with bars, none of them a session bar.
    pub days_without_session_bars: u32,
    /// Session days with a bar that opens exactly at the session close: the stamp a
    /// bar labelled by its close gets for the last bar of the session.
    pub days_with_bar_at_close: u32,
    /// The first such bar.
    pub first_bar_at_close: Option<TsNanos>,
    /// Bars that end at or before the session open, on a session day.
    pub pre_open_bars: u64,
    /// Bars that open at or after the session close, on a session day.
    pub post_close_bars: u64,
    /// Bars that overlap the session but do not open on a lattice point.
    pub off_lattice_bars: u64,
    /// The open instant of the first such bar.
    pub first_off_lattice: Option<TsNanos>,
    /// Session bars on the lattice.
    pub bars_in_session: u64,
    /// Lattice bars absent on the session days that have data.
    pub missing_bars: u64,
    /// Sessions between the first and the last date that have no bar at all.
    pub missing_days: u32,
    /// Bars whose timestamp repeats the previous one.
    pub duplicate_bars: u64,
    /// Session days with data that rest on the weekday rule alone.
    pub template_only_days: u32,
    /// Session days with data whose session times are not verified.
    pub unverified_hours_days: u32,
    /// The outcome of the 5 hour 30 minute offset check.
    pub offset: OffsetCheck,
    /// True when the bars look labelled by their close time although they were read
    /// as labelled by their open (see `label` for the rule).
    pub close_labelled: bool,
    /// Where each bar sits, in input order: what a caller needs to drop the bars
    /// that are not session bars.
    pub placement: Vec<Placement>,
}

impl IntradayCheck {
    /// Count the bars of one IST date.
    fn add_day(&mut self, lattice: &DayLattice, day: Day, rows: &[TsNanos], interval_nanos: i64) {
        self.days += 1;
        let Some(session) = lattice.session else {
            self.closed_days += 1;
            self.closed_day_bars += rows.len() as u64;
            self.first_closed_day.get_or_insert(day);
            self.placement
                .extend(rows.iter().map(|_| Placement::ClosedDay));
            return;
        };
        self.session_days += 1;
        self.template_only_days += u32::from(session.coverage == Coverage::Template);
        self.unverified_hours_days += u32::from(session.hours_unverified);

        let mut distinct_session_bars = 0u64;
        let mut first_session_bar: Option<TsNanos> = None;
        let mut previous: Option<TsNanos> = None;
        let mut bar_at_close = false;
        for &ts in rows {
            bar_at_close |= ts == session.close;
            let placement = lattice.place(ts, interval_nanos);
            self.placement.push(placement);
            match placement {
                Placement::PreOpen => self.pre_open_bars += 1,
                Placement::PostClose => self.post_close_bars += 1,
                Placement::OffLattice => {
                    self.off_lattice_bars += 1;
                    self.first_off_lattice.get_or_insert(ts);
                }
                Placement::InSession => {
                    self.bars_in_session += 1;
                    first_session_bar.get_or_insert(ts);
                    distinct_session_bars += u64::from(previous != Some(ts));
                }
                Placement::ClosedDay => {}
            }
            previous = Some(ts);
        }
        if bar_at_close {
            self.days_with_bar_at_close += 1;
            self.first_bar_at_close.get_or_insert(session.close);
        }
        let first_of_session = lattice.bucket_start(session.open, interval_nanos);
        match first_session_bar {
            Some(ts) if ts == first_of_session => self.days_starting_at_open += 1,
            Some(_) => {
                self.days_starting_late += 1;
                self.first_late_day.get_or_insert(day);
            }
            None => self.days_without_session_bars += 1,
        }
        let expected = lattice.session_bar_count(interval_nanos);
        self.missing_bars += expected.saturating_sub(distinct_session_bars);
    }
}

/// Sessions of the exchange from the first to the last date of a sorted series.
fn sessions_spanned(calendar: &Calendar, exchange: Exchange, ts_open: &[TsNanos]) -> u32 {
    match (ts_open.first(), ts_open.last()) {
        (Some(&first), Some(&last)) => (ist_day(first)..=ist_day(last))
            .filter(|&day| calendar.is_open(exchange, day))
            .count() as u32,
        _ => 0,
    }
}

/// The structural problems that make a series unreadable: a bad interval, a missing or
/// out-of-range timestamp, or timestamps out of order.
fn check_readable(ts_open: &[TsNanos], interval_nanos: i64) -> Result<(), SeriesError> {
    if interval_nanos <= 0 || interval_nanos >= NANOS_PER_DAY {
        return Err(SeriesError::InvalidInterval {
            nanos: interval_nanos,
        });
    }
    if let Some(index) = ts_open.iter().position(|&ts| !is_supported_instant(ts)) {
        return Err(SeriesError::UnsupportedTimestamp {
            index,
            ts: ts_open[index],
        });
    }
    if let Some(index) = ts_open.windows(2).position(|pair| pair[1] < pair[0]) {
        return Err(SeriesError::NotSorted { index: index + 1 });
    }
    Ok(())
}

/// Count how an intraday series sits against the calendar.
///
/// `ts_open` holds bar opens in UTC nanoseconds, sorted ascending. `interval_nanos` is
/// the bar interval.
///
/// Errors: `InvalidInterval` unless the interval is positive and shorter than a day;
/// `UnsupportedTimestamp` for a missing value (the int64 minimum) or an instant
/// outside the supported dates; `NotSorted` when a timestamp is earlier than the one
/// before it. Data problems are counted, never raised.
pub fn check_intraday(
    calendar: &Calendar,
    exchange: Exchange,
    ts_open: &[TsNanos],
    interval_nanos: i64,
) -> Result<IntradayCheck, SeriesError> {
    check_readable(ts_open, interval_nanos)?;
    let mut check = IntradayCheck {
        bars: ts_open.len() as u64,
        placement: Vec::with_capacity(ts_open.len()),
        ..IntradayCheck::default()
    };
    let mut cursor = LatticeCursor::new(calendar, exchange);
    let mut remaining = ts_open;
    while let Some(&first) = remaining.first() {
        let day = ist_day(first);
        // Sorted input, so the rows of one IST date are a prefix.
        let rows_in_day = remaining.partition_point(|&ts| ist_day(ts) == day);
        let (rows, later) = remaining.split_at(rows_in_day);
        if let Some(lattice) = cursor.at(first) {
            check.add_day(&lattice, day, rows, interval_nanos);
        }
        remaining = later;
    }
    check.duplicate_bars = ts_open.windows(2).filter(|pair| pair[0] == pair[1]).count() as u64;
    check.missing_days =
        sessions_spanned(calendar, exchange, ts_open).saturating_sub(check.session_days);
    check.offset = check_offset(calendar, exchange, ts_open, interval_nanos);
    check.close_labelled = looks_close_labelled(&check);
    Ok(check)
}
