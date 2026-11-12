//! What every row of the kernel's dated market tables carries (the default slippage of
//! `slippage_defaults.csv` and the default margin of `margin_defaults.csv`): an id, the
//! trade dates it applies to, where its figure comes from and whether the figure was
//! verified against that source.
//!
//! Status, as in the calendar's tables: `verified` means the row states what its source
//! states for the dates it covers; `unverified` means it does not (an observation of one
//! day applied to a range, or a figure still to be confirmed). A run that uses an
//! unverified row says so in its warnings. `default` marks a modelling default the
//! maintainer set in place of figures that differ by symbol and date (the futures margin
//! of decision D23): the user can override it, the Assumptions block states it, and a
//! run that uses it does not warn.
//!
//! The files are read with the calendar's CSV reader (`market::calendar::csv`): `#`
//! comments, one header line, optional double quotes.

use crate::market::calendar::csv::{read, Record};
use crate::model::{civil_from_days, days_from_civil, Day};

/// How far a row's figure can be trusted.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RowStatus {
    /// The row states what its source states for the dates it covers.
    Verified,
    /// It does not, or not yet; a run that uses it warns.
    Unverified,
    /// A modelling default set by the maintainer, which the user can override; a run
    /// that uses it states it in the Assumptions block and does not warn.
    Default,
}

impl RowStatus {
    /// "verified", "unverified" or "default".
    pub fn code(self) -> &'static str {
        match self {
            RowStatus::Verified => "verified",
            RowStatus::Unverified => "unverified",
            RowStatus::Default => "default",
        }
    }
}

/// The identity, dates and provenance of one table row.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RowMeta {
    /// Unique id within its table, for results and warnings.
    pub id: String,
    /// First trade date (IST) the row applies to; `None` for every earlier date.
    pub valid_from: Option<Day>,
    /// Last trade date (IST) the row applies to; `None` while it is in force.
    pub valid_to: Option<Day>,
    /// Verification status.
    pub status: RowStatus,
    /// Where the figure comes from.
    pub source: String,
    /// Caveats.
    pub notes: String,
}

impl RowMeta {
    /// True when the row applies to the IST trade date `day`.
    pub fn covers(&self, day: Day) -> bool {
        self.valid_from.map_or(true, |from| day >= from)
            && self.valid_to.map_or(true, |to| day <= to)
    }

    /// True when the two rows apply to at least one common date.
    pub fn overlaps(&self, other: &RowMeta) -> bool {
        let starts_before_other_ends = match (self.valid_from, other.valid_to) {
            (Some(from), Some(to)) => from <= to,
            _ => true,
        };
        let ends_after_other_starts = match (self.valid_to, other.valid_from) {
            (Some(to), Some(from)) => to >= from,
            _ => true,
        };
        starts_before_other_ends && ends_after_other_starts
    }
}

/// An empty field as `None`, "YYYY-MM-DD" as a day number.
pub(crate) fn parse_optional_day(text: &str) -> Result<Option<Day>, String> {
    if text.is_empty() {
        return Ok(None);
    }
    let bad = || format!("{text:?} is not a date: write YYYY-MM-DD, or leave it empty");
    let parts: Vec<&str> = text.split('-').collect();
    let [year, month, day] = parts.as_slice() else {
        return Err(bad());
    };
    let shaped = text.len() == 10 && text.bytes().filter(u8::is_ascii_digit).count() == 8;
    let (Ok(year), Ok(month), Ok(dom)) = (year.parse::<i32>(), month.parse(), day.parse()) else {
        return Err(bad());
    };
    let number = days_from_civil(year, month, dom);
    // A date that does not exist (2026-02-30) does not survive the round trip.
    if !shaped || civil_from_days(number) != (year, month, dom) {
        return Err(bad());
    }
    Ok(Some(number))
}

fn parse_status(text: &str) -> Result<RowStatus, String> {
    match text {
        "verified" => Ok(RowStatus::Verified),
        "unverified" => Ok(RowStatus::Unverified),
        "default" => Ok(RowStatus::Default),
        other => Err(format!(
            "status {other:?} must be verified, unverified or default"
        )),
    }
}

/// The meta of a record whose fields are `id, <key and figure fields>, valid_from,
/// valid_to, source, status, notes`, with the dates at `dates` and the last three fields
/// at the end.
pub(crate) fn parse_meta(fields: &[String], dates: usize) -> Result<RowMeta, String> {
    let (Some(id), Some(from), Some(to), [.., source, status, notes]) = (
        fields.first(),
        fields.get(dates),
        fields.get(dates + 1),
        fields,
    ) else {
        return Err("too few fields".to_string());
    };
    let valid_from = parse_optional_day(from)?;
    let valid_to = parse_optional_day(to)?;
    if let (Some(from), Some(to)) = (valid_from, valid_to) {
        if to < from {
            return Err("valid_to is before valid_from".to_string());
        }
    }
    if id.trim().is_empty() || source.trim().is_empty() {
        return Err("every row needs an id and a source".to_string());
    }
    Ok(RowMeta {
        id: id.clone(),
        valid_from,
        valid_to,
        status: parse_status(status)?,
        source: source.clone(),
        notes: notes.clone(),
    })
}

/// Read the records of an embedded table, as "<table> line <n>: <problem>" on error.
pub(crate) fn records(
    table: &'static str,
    text: &str,
    header: &[&str],
) -> Result<Vec<Record>, String> {
    read(table, text, header)
        .map_err(|error| format!("{} line {}: {}", error.table, error.line, error.problem))
}

/// "<table> line <n>: <problem>" for a problem found in one record.
pub(crate) fn at_line(table: &str, record: &Record, problem: &str) -> String {
    format!("{table} line {}: {problem}", record.line)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn meta(from: Option<Day>, to: Option<Day>) -> RowMeta {
        RowMeta {
            id: "x".to_string(),
            valid_from: from,
            valid_to: to,
            status: RowStatus::Verified,
            source: "s".to_string(),
            notes: String::new(),
        }
    }

    #[test]
    fn an_empty_start_covers_every_earlier_date() {
        let row = meta(None, Some(100));
        assert!(row.covers(-5_000) && row.covers(100) && !row.covers(101));
    }

    #[test]
    fn rows_overlap_when_they_share_a_date() {
        assert!(meta(None, Some(10)).overlaps(&meta(Some(10), None)));
        assert!(!meta(None, Some(9)).overlaps(&meta(Some(10), None)));
    }

    #[test]
    fn a_status_is_verified_unverified_or_default() {
        assert_eq!(parse_status("verified"), Ok(RowStatus::Verified));
        assert_eq!(parse_status("unverified"), Ok(RowStatus::Unverified));
        assert_eq!(parse_status("default"), Ok(RowStatus::Default));
        assert!(parse_status("assumed").is_err());
    }

    #[test]
    fn dates_are_iso_and_must_exist() {
        assert_eq!(parse_optional_day(""), Ok(None));
        assert_eq!(parse_optional_day("1970-01-02"), Ok(Some(1)));
        for bad in ["2026-02-30", "2026-2-03", "26-02-03", "2026/02/03"] {
            assert!(parse_optional_day(bad).is_err(), "{bad}");
        }
    }
}
