//! The built-in schedule versions: `schedules.csv`, embedded and parsed once.
//!
//! The data file is the only place a built-in rate is written, so a rate change is a
//! one-line diff. Its columns:
//!
//! | Column | Meaning |
//! |---|---|
//! | `schedule_id` | The version a row belongs to |
//! | `valid_from`, `valid_to` | First and last trade date (`*` row); an empty `valid_to` is open |
//! | `segment` | A segment code, or `*` for the version's own row, which comes first |
//! | `verified_through` | Last trade date the row was checked for; empty when never verified |
//! | `stt_buy_rate` to `stt_exercise_rate` | The seven rates of a segment, as fractions |
//! | `gst_rate` | GST as a fraction; `*` row only |
//! | `calculator_tax_base` | `sides` or `turnover`: what the calculator taxes |
//! | `calculator_tax_unit` | `rupee` or `paise`: what the calculator rounds the tax to |
//! | `source` | Where the version's figures came from; `*` row only |
//! | `notes` | Caveats, for the version or for one row |
//!
//! Adding a version: add its `*` row and one row per segment, end the previous
//! version (`valid_to`) the day before, and fill `verified_through` only for rows
//! checked against the source for that period. A malformed file is a typed error
//! (`CostError::InvalidScheduleData`) and fails the unit tests; it never panics.

use std::sync::OnceLock;

use crate::model::{civil_from_days, days_from_civil, Day};

use super::book::ordered;
use super::error::CostError;
use super::schedule::{CostSchedule, SegmentRates, SegmentRow, TaxBase, TaxUnit};
use super::segment::Segment;

const DATA: &str = include_str!("schedules.csv");

const HEADER: &str = "schedule_id,valid_from,valid_to,segment,verified_through,stt_buy_rate,\
                      stt_sell_rate,exchange_txn_rate,ipft_rate,sebi_rate,stamp_buy_rate,\
                      stt_exercise_rate,gst_rate,calculator_tax_base,calculator_tax_unit,source,\
                      notes";
const COLUMNS: usize = 17;
/// The `segment` value of a version's own row.
const VERSION_ROW: &str = "*";

// Column positions.
const ID: usize = 0;
const VALID_FROM: usize = 1;
const VALID_TO: usize = 2;
const SEGMENT: usize = 3;
const VERIFIED_THROUGH: usize = 4;
const FIRST_RATE: usize = 5;
const GST_RATE: usize = 12;
const TAX_BASE: usize = 13;
const TAX_UNIT: usize = 14;
const SOURCE: usize = 15;
const NOTES: usize = 16;

type Reason = String;

/// Split one CSV line into fields. A field may be wrapped in double quotes, inside
/// which a comma is text and a doubled quote is one quote.
fn split_fields(line: &str) -> Result<Vec<String>, Reason> {
    let mut fields = Vec::with_capacity(COLUMNS);
    let mut field = String::new();
    let mut quoted = false;
    let mut chars = line.chars().peekable();
    while let Some(ch) = chars.next() {
        match (ch, quoted) {
            ('"', true) if chars.peek() == Some(&'"') => {
                chars.next();
                field.push('"');
            }
            ('"', true) => quoted = false,
            ('"', false) if field.is_empty() => quoted = true,
            ('"', false) => return Err("a quote inside an unquoted field".to_string()),
            (',', false) => fields.push(std::mem::take(&mut field)),
            (other, _) => field.push(other),
        }
    }
    if quoted {
        return Err("a quoted field is not closed".to_string());
    }
    fields.push(field);
    Ok(fields)
}

/// An ISO date (YYYY-MM-DD) as a day number.
fn parse_date(text: &str) -> Result<Day, Reason> {
    let bad = || format!("{text:?} is not a date of the form 2026-04-01");
    let bytes = text.as_bytes();
    let shaped = bytes.len() == 10
        && bytes.iter().enumerate().all(|(at, byte)| match at {
            4 | 7 => *byte == b'-',
            _ => byte.is_ascii_digit(),
        });
    if !shaped {
        return Err(bad());
    }
    let year: i32 = text[..4].parse().map_err(|_| bad())?;
    let month: u32 = text[5..7].parse().map_err(|_| bad())?;
    let dom: u32 = text[8..].parse().map_err(|_| bad())?;
    if !(1..=12).contains(&month) || !(1..=31).contains(&dom) {
        return Err(bad());
    }
    let day = days_from_civil(year, month, dom);
    // 31 February and the like do not survive the round trip.
    if civil_from_days(day) != (year, month, dom) {
        return Err(bad());
    }
    Ok(day)
}

fn optional_date(text: &str) -> Result<Option<Day>, Reason> {
    if text.is_empty() {
        Ok(None)
    } else {
        parse_date(text).map(Some)
    }
}

fn parse_rate(name: &str, text: &str) -> Result<f64, Reason> {
    match text.parse::<f64>() {
        Ok(rate) if rate.is_finite() => Ok(rate),
        _ => Err(format!(
            "{name} {text:?} is not a number; write a fraction such as 0.001"
        )),
    }
}

/// Require the columns that belong to the other kind of row to be empty.
fn require_empty(fields: &[String], columns: &[usize], row_kind: &str) -> Result<(), Reason> {
    match columns.iter().find(|&&column| !fields[column].is_empty()) {
        Some(&column) => Err(format!(
            "column {} must be empty on a {row_kind} row",
            column + 1
        )),
        None => Ok(()),
    }
}

/// The version described by a `*` row, with no segment rows yet.
fn version_row(fields: &[String]) -> Result<CostSchedule, Reason> {
    let rates_and_flags: Vec<usize> = (FIRST_RATE..GST_RATE).chain([TAX_BASE, TAX_UNIT]).collect();
    require_empty(fields, &rates_and_flags, "version (*)")?;
    Ok(CostSchedule {
        id: fields[ID].clone(),
        valid_from: parse_date(&fields[VALID_FROM])?,
        valid_to: optional_date(&fields[VALID_TO])?,
        gst_rate: parse_rate("gst_rate", &fields[GST_RATE])?,
        gst_verified_through: optional_date(&fields[VERIFIED_THROUGH])?,
        source: fields[SOURCE].clone(),
        notes: fields[NOTES].clone(),
        rows: std::array::from_fn(|_| None),
    })
}

/// The segment and rates described by a segment row.
fn segment_row(fields: &[String]) -> Result<(Segment, SegmentRow), Reason> {
    require_empty(fields, &[VALID_FROM, VALID_TO, GST_RATE, SOURCE], "segment")?;
    let segment = Segment::from_code(&fields[SEGMENT]).map_err(|error| error.to_string())?;
    let rate = |offset: usize, name: &str| parse_rate(name, &fields[FIRST_RATE + offset]);
    let base = &fields[TAX_BASE];
    let unit = &fields[TAX_UNIT];
    let rates = SegmentRates {
        stt_buy_rate: rate(0, "stt_buy_rate")?,
        stt_sell_rate: rate(1, "stt_sell_rate")?,
        exchange_txn_rate: rate(2, "exchange_txn_rate")?,
        ipft_rate: rate(3, "ipft_rate")?,
        sebi_rate: rate(4, "sebi_rate")?,
        stamp_buy_rate: rate(5, "stamp_buy_rate")?,
        stt_exercise_rate: rate(6, "stt_exercise_rate")?,
        calculator_tax_base: TaxBase::from_code(base)
            .ok_or_else(|| format!("calculator_tax_base {base:?} must be sides or turnover"))?,
        calculator_tax_unit: TaxUnit::from_code(unit)
            .ok_or_else(|| format!("calculator_tax_unit {unit:?} must be rupee or paise"))?,
    };
    let verified_through = optional_date(&fields[VERIFIED_THROUGH])?;
    Ok((
        segment,
        SegmentRow {
            rates,
            verified_through,
            notes: fields[NOTES].clone(),
        },
    ))
}

/// Apply one data line to the versions read so far.
fn read_line(line: &str, versions: &mut Vec<CostSchedule>) -> Result<(), Reason> {
    let fields = split_fields(line)?;
    if fields.len() != COLUMNS {
        return Err(format!(
            "expected {COLUMNS} columns, found {}",
            fields.len()
        ));
    }
    let id = &fields[ID];
    if fields[SEGMENT] == VERSION_ROW {
        if versions.iter().any(|version| &version.id == id) {
            return Err(format!("version {id:?} has two * rows"));
        }
        versions.push(version_row(&fields)?);
        return Ok(());
    }
    let (segment, row) = segment_row(&fields)?;
    match versions.last_mut() {
        Some(version) if &version.id == id => {
            let slot = &mut version.rows[segment.index()];
            if slot.is_some() {
                return Err(format!(
                    "segment {} appears twice in {id:?}",
                    segment.code()
                ));
            }
            *slot = Some(row);
            Ok(())
        }
        _ => Err(format!(
            "the * row of version {id:?} must come before its segment rows"
        )),
    }
}

/// Parse schedule data in the format of `schedules.csv` into validated versions,
/// oldest first.
///
/// Errors: `InvalidScheduleData` naming the line for a malformed file, and the
/// errors of `ScheduleBook::from_versions` for versions that are invalid or overlap.
pub(super) fn parse_schedules(data: &str) -> Result<Vec<CostSchedule>, CostError> {
    let mut lines = data
        .lines()
        .enumerate()
        .map(|(index, line)| (index + 1, line));
    let invalid = |line: usize, reason: Reason| CostError::InvalidScheduleData { line, reason };
    match lines.next() {
        Some((_, header)) if header == HEADER => {}
        _ => {
            return Err(invalid(
                1,
                format!("the first line must be the header {HEADER:?}"),
            ))
        }
    }
    let mut versions = Vec::new();
    for (number, line) in lines {
        if line.trim().is_empty() {
            continue;
        }
        read_line(line, &mut versions).map_err(|reason| invalid(number, reason))?;
    }
    ordered(versions)
}

/// The built-in schedule versions, oldest first. Parsed and validated once, on first
/// use.
///
/// Errors: as `parse_schedules`, if the embedded file is malformed.
pub fn builtin_schedules() -> Result<&'static [CostSchedule], CostError> {
    static VERSIONS: OnceLock<Result<Vec<CostSchedule>, CostError>> = OnceLock::new();
    match VERSIONS.get_or_init(|| parse_schedules(DATA)) {
        Ok(versions) => Ok(versions),
        Err(error) => Err(error.clone()),
    }
}
