//! Daily bars to weekly, monthly, quarterly and yearly bars.
//!
//! Grouping equals the server's `_get_daily_aggregated_ohlcv`: weeks start on Monday
//! (its anchor is Monday 1970-01-05) and months, quarters and years are calendar
//! periods of the IST session date. The server labels a period bar with 00:00 UTC of
//! the period's first calendar date; here `ts_open` is the open of the first session
//! of the period and `ts_close` the close of its last session, both from the calendar,
//! so a bar is complete only when the period's last session has closed.

use crate::market::calendar::Calendar;
use crate::model::{civil_from_days, days_from_civil, ist_day, weekday, Day, Exchange, TsNanos};

use super::bars::{BarsIn, BarsOut};
use super::error::ResampleError;
use super::runs::{aggregate_runs, closed_day_error};
use super::{ClosedDayPolicy, Dropped, Resampled};

const DAYS_PER_WEEK: Day = 7;
const MONTHS_PER_YEAR: u32 = 12;

/// A calendar period longer than a day.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CalendarUnit {
    /// Monday to Sunday.
    Week,
    /// A calendar month.
    Month,
    /// January to March, April to June, July to September, October to December.
    Quarter,
    /// A calendar year.
    Year,
}

impl CalendarUnit {
    /// The unit for an OpenAlgo interval string: "W", "M", "Q" or "Y".
    pub fn from_code(code: &str) -> Option<CalendarUnit> {
        match code {
            "W" => Some(CalendarUnit::Week),
            "M" => Some(CalendarUnit::Month),
            "Q" => Some(CalendarUnit::Quarter),
            "Y" => Some(CalendarUnit::Year),
            _ => None,
        }
    }

    /// First and last calendar date (inclusive) of the period that contains `day`.
    pub(super) fn range(self, day: Day) -> (Day, Day) {
        let months = match self {
            CalendarUnit::Week => {
                let monday = day - weekday(day) as Day;
                return (monday, monday + DAYS_PER_WEEK - 1);
            }
            CalendarUnit::Month => 1,
            CalendarUnit::Quarter => 3,
            CalendarUnit::Year => MONTHS_PER_YEAR,
        };
        let (year, month, _) = civil_from_days(day);
        let first_month = (month - 1) / months * months + 1;
        let next_month = first_month + months;
        let next_start = if next_month > MONTHS_PER_YEAR {
            days_from_civil(year + 1, next_month - MONTHS_PER_YEAR, 1)
        } else {
            days_from_civil(year, next_month, 1)
        };
        (days_from_civil(year, first_month, 1), next_start - 1)
    }
}

/// One period bar: the calendar dates it covers and its session bounds.
#[derive(Clone, Copy, PartialEq, Eq)]
struct Period {
    first_day: Day,
    last_day: Day,
    /// Open of the first session of the period, UTC nanoseconds.
    open: TsNanos,
    /// Close of the last session of the period, UTC nanoseconds.
    close: TsNanos,
}

/// Maps each daily row to its period, checking the row on the way.
struct PeriodFinder<'a> {
    calendar: &'a Calendar,
    exchange: Exchange,
    unit: CalendarUnit,
    closed_day: ClosedDayPolicy,
    previous_day: Option<Day>,
    current: Option<Period>,
    closed_day_bars: u64,
}

impl PeriodFinder<'_> {
    /// The period of the daily row at `row`, or `None` when the closed-day policy
    /// drops the row.
    fn period_of(&mut self, row: usize, ts: TsNanos) -> Result<Option<Period>, ResampleError> {
        let day = ist_day(ts);
        if self.previous_day == Some(day) {
            return Err(ResampleError::DuplicateDay { index: row, day });
        }
        self.previous_day = Some(day);
        if !self.calendar.is_open(self.exchange, day) {
            if self.closed_day == ClosedDayPolicy::Error {
                return Err(closed_day_error(self.calendar, self.exchange, ts));
            }
            self.closed_day_bars += 1;
            return Ok(None);
        }
        let in_current = |period: &Period| (period.first_day..=period.last_day).contains(&day);
        if let Some(period) = self.current.filter(in_current) {
            return Ok(Some(period));
        }
        let (first_day, last_day) = self.unit.range(day);
        // `day` is a session inside the range, so the period always has bounds.
        let (open, close) = self
            .calendar
            .period_bounds(self.exchange, first_day, last_day)
            .ok_or_else(|| closed_day_error(self.calendar, self.exchange, ts))?;
        let period = Period {
            first_day,
            last_day,
            open,
            close,
        };
        self.current = Some(period);
        Ok(Some(period))
    }
}

/// Resample daily bars to one bar per week, month, quarter or year.
///
/// Each input row is one session: its IST date is taken from `ts_open`, which may be
/// the session open (the canonical layout) or 00:00 UTC of the session date (the raw
/// Historify and broker convention).
///
/// Errors: the errors of `BarsIn::validate`; `DuplicateDay` for two rows on one IST
/// date; `ClosedDay` for a row on a date with no session unless the policy drops it.
pub fn resample_period(
    calendar: &Calendar,
    exchange: Exchange,
    bars: &BarsIn<'_>,
    unit: CalendarUnit,
    closed_day: ClosedDayPolicy,
) -> Result<Resampled, ResampleError> {
    bars.validate()?;
    let mut finder = PeriodFinder {
        calendar,
        exchange,
        unit,
        closed_day,
        previous_day: None,
        current: None,
        closed_day_bars: 0,
    };
    let mut out = BarsOut::default();
    let empty_buckets = aggregate_runs(
        bars,
        |row, ts| finder.period_of(row, ts),
        |period, aggregate| out.push(period.open, period.close, aggregate),
    )?;
    let dropped = Dropped {
        closed_day_bars: finder.closed_day_bars,
        empty_buckets,
        ..Dropped::default()
    };
    Ok(Resampled { bars: out, dropped })
}
