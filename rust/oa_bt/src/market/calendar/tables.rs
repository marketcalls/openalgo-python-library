//! The two dated tables embedded in the calendar: regular trading hours and default
//! intraday times.
//!
//! Both are market data with a source and a verification status per row, kept in CSV
//! files next to this module so that a change of hours is a one-line diff:
//!
//! - `regular_hours.csv`: the session open and close of each exchange by date.
//! - `intraday_defaults.csv`: the default `entry_until` and `square_off` by date.
//!
//! The files are parsed once, on first use. A malformed file is a `TableError`: it
//! fails the unit tests and reaches a caller as a typed error, never as a panic.
//!
//! The rows of an exchange cover every date: the first has an empty `valid_from`, the
//! last an empty `valid_to`, and each `valid_from` is the day after the previous
//! `valid_to`. Times are IST, "HH:MM".

use std::sync::OnceLock;

use crate::model::{civil_from_days, days_from_civil, weekday, Day, Exchange};

use super::csv::{read, Record, TableError};
use super::session::calendar_exchange;
use super::SECS_PER_DAY;

pub(super) const EXCHANGE_COUNT: usize = Exchange::ALL.len();
/// Rows per exchange, indexed by the exchange's discriminant. The index exchanges
/// have none: they follow NSE and BSE.
pub(super) type PerExchange<T> = [Vec<T>; EXCHANGE_COUNT];

const HOURS_FILE: &str = "regular_hours.csv";
const HOURS_TEXT: &str = include_str!("regular_hours.csv");
const HOURS_HEADER: [&str; 8] = [
    "exchange",
    "valid_from",
    "valid_to",
    "open",
    "close",
    "close_us_dst",
    "source",
    "status",
];

const INTRADAY_FILE: &str = "intraday_defaults.csv";
const INTRADAY_TEXT: &str = include_str!("intraday_defaults.csv");
const INTRADAY_HEADER: [&str; 9] = [
    "exchange",
    "valid_from",
    "valid_to",
    "entry_until",
    "square_off",
    "entry_before_close_min",
    "square_off_before_close_min",
    "source",
    "status",
];

/// United States daylight saving runs from the second Sunday of March up to, not
/// including, the first Sunday of November (the rule in force since 2007). Earlier
/// years used other dates and are not modelled: rows that depend on it are unverified.
const US_DST_START: (u32, i32) = (3, 2);
const US_DST_END: (u32, i32) = (11, 1);

/// The regular hours of an exchange resolved for one date, in seconds since IST
/// midnight.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RegularHours {
    pub open_sec: u32,
    pub close_sec: u32,
    /// True when the row these hours come from is not verified.
    pub unverified: bool,
}

/// Regular hours of one exchange from `valid_from` to `valid_to`, both inclusive.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct HoursRow {
    pub(super) valid_from: Day,
    pub(super) valid_to: Day,
    pub(super) open_sec: u32,
    pub(super) close_sec: u32,
    /// The close while the United States is on daylight saving time, when it differs.
    pub(super) close_us_dst_sec: Option<u32>,
    pub(super) verified: bool,
}

impl HoursRow {
    pub(super) fn resolve(&self, day: Day) -> RegularHours {
        let close_sec = match self.close_us_dst_sec {
            Some(dst_close) if us_daylight_saving(day) => dst_close,
            _ => self.close_sec,
        };
        RegularHours {
            open_sec: self.open_sec,
            close_sec,
            unverified: !self.verified,
        }
    }
}

/// Default intraday times of one exchange from `valid_from` to `valid_to`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct IntradayRow {
    pub(super) valid_from: Day,
    pub(super) valid_to: Day,
    /// Fixed (entry_until, square_off) in seconds since IST midnight, when the
    /// segment has clock times.
    pub(super) clock_sec: Option<(u32, u32)>,
    /// Minutes before the session close at which entries stop.
    pub(super) entry_before_close_min: u32,
    /// Minutes before the session close at which positions are squared off.
    pub(super) square_off_before_close_min: u32,
    pub(super) verified: bool,
}

/// Rows that are valid over a date range.
pub(super) trait Dated {
    fn valid(&self) -> (Day, Day);
}

impl Dated for HoursRow {
    fn valid(&self) -> (Day, Day) {
        (self.valid_from, self.valid_to)
    }
}

impl Dated for IntradayRow {
    fn valid(&self) -> (Day, Day) {
        (self.valid_from, self.valid_to)
    }
}

/// The row in force on a date. `rows` is sorted and covers every date.
pub(super) fn row_on<T: Dated>(rows: &[T], day: Day) -> Option<&T> {
    let index = rows.partition_point(|row| row.valid().1 < day);
    rows.get(index).filter(|row| row.valid().0 <= day)
}

/// Both tables, parsed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Tables {
    pub(super) hours: PerExchange<HoursRow>,
    pub(super) intraday: PerExchange<IntradayRow>,
}

/// The embedded tables, parsed on first use.
pub(super) fn builtin() -> Result<&'static Tables, TableError> {
    static TABLES: OnceLock<Result<Tables, TableError>> = OnceLock::new();
    let parsed = TABLES.get_or_init(|| {
        Ok(Tables {
            hours: parse_regular_hours(HOURS_FILE, HOURS_TEXT)?,
            intraday: parse_intraday_defaults(INTRADAY_FILE, INTRADAY_TEXT)?,
        })
    });
    parsed.as_ref().map_err(Clone::clone)
}

// ---------------------------------------------------------------------------
// Field parsers
// ---------------------------------------------------------------------------

/// "HH:MM" as seconds since IST midnight; 24:00 is allowed.
fn parse_time(text: &str) -> Result<u32, String> {
    let bad = || format!("{text:?} is not a time: write HH:MM in IST, for example 09:15");
    let (hours, minutes) = text.split_once(':').ok_or_else(bad)?;
    let all_digits = |part: &str| part.len() == 2 && part.bytes().all(|b| b.is_ascii_digit());
    if !all_digits(hours) || !all_digits(minutes) {
        return Err(bad());
    }
    let hours: u32 = hours.parse().map_err(|_| bad())?;
    let minutes: u32 = minutes.parse().map_err(|_| bad())?;
    let seconds = hours * 3_600 + minutes * 60;
    if minutes > 59 || seconds > SECS_PER_DAY {
        return Err(bad());
    }
    Ok(seconds)
}

/// "YYYY-MM-DD" as a day number.
fn parse_date(text: &str) -> Result<Day, String> {
    let bad = || format!("{text:?} is not a date: write YYYY-MM-DD");
    let mut parts = text.split('-');
    let mut next_number = || parts.next().and_then(|part| part.parse::<u32>().ok());
    let (year, month, day) = (next_number(), next_number(), next_number());
    let (Some(year), Some(month), Some(day)) = (year, month, day) else {
        return Err(bad());
    };
    if text.len() != 10 || parts.next().is_some() {
        return Err(bad());
    }
    let number = days_from_civil(year as i32, month, day);
    // A date that does not exist (2026-02-30) does not survive the round trip.
    if civil_from_days(number) != (year as i32, month, day) {
        return Err(bad());
    }
    Ok(number)
}

fn parse_status(text: &str) -> Result<bool, String> {
    match text {
        "verified" => Ok(true),
        "unverified" => Ok(false),
        other => Err(format!("status {other:?} must be verified or unverified")),
    }
}

/// The fields every row starts and ends with: exchange, validity, source, status.
struct Common {
    exchange: Exchange,
    valid_from: Day,
    valid_to: Day,
    verified: bool,
}

fn parse_common(fields: &[String]) -> Result<Common, String> {
    let [code, valid_from, valid_to, .., source, status] = fields else {
        return Err("too few fields".to_string());
    };
    let exchange = Exchange::from_code(code)
        .ok_or_else(|| format!("{code:?} is not an OpenAlgo exchange code"))?;
    if calendar_exchange(exchange) != exchange {
        let owner = calendar_exchange(exchange).code();
        return Err(format!(
            "{code} follows the {owner} calendar: give its rows under {owner}"
        ));
    }
    let valid_from = if valid_from.is_empty() {
        Day::MIN
    } else {
        parse_date(valid_from)?
    };
    let valid_to = if valid_to.is_empty() {
        Day::MAX
    } else {
        parse_date(valid_to)?
    };
    if valid_from > valid_to {
        return Err("valid_from is after valid_to".to_string());
    }
    if source.trim().is_empty() {
        return Err("the source is empty: every row names where its values come from".to_string());
    }
    Ok(Common {
        exchange,
        valid_from,
        valid_to,
        verified: parse_status(status)?,
    })
}

/// Parse every record with `parse_row` and group the rows by exchange, checking that
/// each exchange with its own calendar is covered on every date without overlap.
fn parse_table<T: Dated>(
    table: &'static str,
    records: Vec<Record>,
    parse_row: impl Fn(&[String]) -> Result<(Exchange, T), String>,
) -> Result<PerExchange<T>, TableError> {
    let fail = |line: usize, problem: String| TableError {
        table,
        line,
        problem,
    };
    let mut rows: PerExchange<T> = std::array::from_fn(|_| Vec::new());
    for record in records {
        let (exchange, row) =
            parse_row(&record.fields).map_err(|problem| fail(record.line, problem))?;
        let of_exchange = &mut rows[exchange as usize];
        let expected_from = match of_exchange.last() {
            None => Day::MIN,
            Some(previous) if previous.valid().1 == Day::MAX => {
                let problem = format!("{} already has an open-ended row", exchange.code());
                return Err(fail(record.line, problem));
            }
            Some(previous) => previous.valid().1 + 1,
        };
        if row.valid().0 != expected_from {
            let problem = format!(
                "the rows of {} must be contiguous: valid_from must be empty on the first \
                 row and the day after the previous valid_to on the others",
                exchange.code()
            );
            return Err(fail(record.line, problem));
        }
        of_exchange.push(row);
    }
    for exchange in Exchange::ALL {
        let covered = rows[exchange as usize]
            .last()
            .is_some_and(|row| row.valid().1 == Day::MAX);
        if calendar_exchange(exchange) == exchange && !covered {
            let problem = format!("{} has no open-ended last row", exchange.code());
            return Err(fail(0, problem));
        }
    }
    Ok(rows)
}

// ---------------------------------------------------------------------------
// The two tables
// ---------------------------------------------------------------------------

fn hours_row(fields: &[String]) -> Result<(Exchange, HoursRow), String> {
    let common = parse_common(fields)?;
    let open_sec = parse_time(&fields[3])?;
    let close_sec = parse_time(&fields[4])?;
    let close_us_dst_sec = match fields[5].as_str() {
        "" => None,
        text => Some(parse_time(text)?),
    };
    let closes_after_open = |close: u32| open_sec < close;
    if !closes_after_open(close_sec) || !close_us_dst_sec.map_or(true, closes_after_open) {
        return Err("the open must be before the close".to_string());
    }
    let row = HoursRow {
        valid_from: common.valid_from,
        valid_to: common.valid_to,
        open_sec,
        close_sec,
        close_us_dst_sec,
        verified: common.verified,
    };
    Ok((common.exchange, row))
}

/// Parse a regular-hours table in the layout of `regular_hours.csv`.
pub(super) fn parse_regular_hours(
    table: &'static str,
    text: &str,
) -> Result<PerExchange<HoursRow>, TableError> {
    parse_table(table, read(table, text, &HOURS_HEADER)?, hours_row)
}

fn parse_minutes(text: &str) -> Result<u32, String> {
    text.parse::<u32>()
        .map_err(|_| format!("{text:?} is not a whole number of minutes"))
}

fn intraday_row(fields: &[String]) -> Result<(Exchange, IntradayRow), String> {
    let common = parse_common(fields)?;
    let clock_sec = match (fields[3].as_str(), fields[4].as_str()) {
        ("", "") => None,
        ("", _) | (_, "") => {
            return Err("give both entry_until and square_off, or neither".to_string())
        }
        (entry, square_off) => Some((parse_time(entry)?, parse_time(square_off)?)),
    };
    let entry_before_close_min = parse_minutes(&fields[5])?;
    let square_off_before_close_min = parse_minutes(&fields[6])?;
    let clock_ordered = clock_sec.map_or(true, |(entry, square_off)| entry <= square_off);
    if !clock_ordered || entry_before_close_min < square_off_before_close_min {
        return Err("entries must stop at or before the square-off".to_string());
    }
    let row = IntradayRow {
        valid_from: common.valid_from,
        valid_to: common.valid_to,
        clock_sec,
        entry_before_close_min,
        square_off_before_close_min,
        verified: common.verified,
    };
    Ok((common.exchange, row))
}

/// Parse a default-intraday-times table in the layout of `intraday_defaults.csv`.
pub(super) fn parse_intraday_defaults(
    table: &'static str,
    text: &str,
) -> Result<PerExchange<IntradayRow>, TableError> {
    parse_table(table, read(table, text, &INTRADAY_HEADER)?, intraday_row)
}

// ---------------------------------------------------------------------------
// United States daylight saving
// ---------------------------------------------------------------------------

/// The n-th Sunday (n from 1) of a month.
fn nth_sunday(year: i32, month: u32, n: i32) -> Day {
    const SUNDAY: i32 = 6;
    let first = days_from_civil(year, month, 1);
    let days_to_sunday = (SUNDAY - weekday(first) as i32).rem_euclid(7);
    first.saturating_add(days_to_sunday + 7 * (n - 1))
}

/// Whether the United States observes daylight saving time on this date. The switch
/// happens on a Sunday, so Indian trading days change from the next Monday.
pub(super) fn us_daylight_saving(day: Day) -> bool {
    let (year, _, _) = civil_from_days(day);
    let start = nth_sunday(year, US_DST_START.0, US_DST_START.1);
    let end = nth_sunday(year, US_DST_END.0, US_DST_END.1);
    day >= start && day < end
}
