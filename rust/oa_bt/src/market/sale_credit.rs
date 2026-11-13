//! The credit for selling delivery holdings (04-execution 5): how much of the proceeds
//! of a sale of CNC equity bought on an earlier day a broker lets the account trade with
//! on the day of the sale, by trade date. The rest is credited on the next trading day.
//!
//! The dated rows are in `sale_credit_defaults.csv`, embedded and parsed once: the full
//! proceeds before 2020-12-01 and from 2024-10-07, 80 percent between (the peak margin
//! period, when 20 percent of the sale value stayed blocked until delivery). A malformed
//! embedded file is a typed error (`SaleCreditError::InvalidTable`) and fails the unit
//! tests; it never panics.

use std::fmt;
use std::sync::OnceLock;

use crate::model::Day;

use super::table_rows::{at_line, parse_meta, records, RowMeta};

const DATA: &str = include_str!("sale_credit_defaults.csv");
const TABLE: &str = "sale_credit_defaults.csv";
const HEADER: [&str; 7] = [
    "row_id",
    "valid_from",
    "valid_to",
    "same_day_percent",
    "source",
    "status",
    "notes",
];
/// Position of `valid_from` in a record.
const DATES: usize = 1;
/// The whole sale value, percent.
const WHOLE_PERCENT: f64 = 100.0;

/// What went wrong with the sale-credit table.
#[derive(Clone, Debug, PartialEq)]
pub enum SaleCreditError {
    /// The embedded table is malformed (a defect of the release).
    InvalidTable { reason: String },
}

impl fmt::Display for SaleCreditError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SaleCreditError::InvalidTable { reason } => {
                write!(f, "the built-in sale-credit table is malformed: {reason}")
            }
        }
    }
}

impl std::error::Error for SaleCreditError {}

/// One row of the sale-credit table.
#[derive(Clone, Debug, PartialEq)]
pub struct SaleCreditRow {
    /// Id, dates, source and status.
    pub meta: RowMeta,
    /// Percent of the sale value credited on the day of the sale, above 0, at most 100.
    pub same_day_percent: f64,
}

impl SaleCreditRow {
    /// The part of a sale worth `sale_rupees` that is credited only on the next trading
    /// day, rupees.
    pub fn next_day_rupees(&self, sale_rupees: f64) -> f64 {
        sale_rupees * (WHOLE_PERCENT - self.same_day_percent) / WHOLE_PERCENT
    }
}

fn parse_row(fields: &[String]) -> Result<SaleCreditRow, String> {
    let Some(percent) = fields.get(DATES + 2) else {
        return Err("too few fields".to_string());
    };
    let same_day_percent: f64 = percent
        .parse()
        .map_err(|_| format!("same_day_percent {percent:?} is not a number"))?;
    if !(same_day_percent.is_finite() && same_day_percent > 0.0 && same_day_percent <= 100.0) {
        return Err(format!(
            "same_day_percent {same_day_percent} must be above 0 and at most 100"
        ));
    }
    Ok(SaleCreditRow {
        meta: parse_meta(fields, DATES)?,
        same_day_percent,
    })
}

/// Parse a sale-credit table: every row valid, ids unique, no two rows on one date, and
/// every date covered (the first row open at the start, each next row starting the day
/// after the previous one ends, the last open at the end).
fn parse_table(text: &str) -> Result<Vec<SaleCreditRow>, String> {
    let mut rows: Vec<SaleCreditRow> = Vec::new();
    for record in records(TABLE, text, &HEADER)? {
        let row = parse_row(&record.fields).map_err(|problem| at_line(TABLE, &record, &problem))?;
        let follows = match rows.last() {
            None => row.meta.valid_from.is_none(),
            Some(previous) => match (previous.meta.valid_to, row.meta.valid_from) {
                (Some(to), Some(from)) => from == to + 1,
                _ => false,
            },
        };
        let repeated = rows.iter().any(|earlier| earlier.meta.id == row.meta.id);
        if repeated || !follows {
            let problem = format!(
                "row {} repeats an id, or does not start the day after the previous row ends",
                row.meta.id
            );
            return Err(at_line(TABLE, &record, &problem));
        }
        rows.push(row);
    }
    match rows.last() {
        Some(last) if last.meta.valid_to.is_none() => Ok(rows),
        _ => Err(format!(
            "{TABLE}: the last row must stay in force (empty valid_to)"
        )),
    }
}

/// `parse_table` with its problem as the table's typed error.
fn checked_table(text: &str) -> Result<Vec<SaleCreditRow>, SaleCreditError> {
    parse_table(text).map_err(|reason| SaleCreditError::InvalidTable { reason })
}

/// The built-in sale-credit rows, in date order.
///
/// Errors: `InvalidTable` if the embedded file is malformed (a unit test guards against
/// shipping one).
pub fn sale_credit_defaults() -> Result<&'static [SaleCreditRow], SaleCreditError> {
    static ROWS: OnceLock<Result<Vec<SaleCreditRow>, SaleCreditError>> = OnceLock::new();
    match ROWS.get_or_init(|| checked_table(DATA)) {
        Ok(rows) => Ok(rows),
        Err(error) => Err(error.clone()),
    }
}

/// The row in force on the IST trade date `day`.
///
/// Errors: `InvalidTable` as `sale_credit_defaults`, or when no row covers the day
/// (which the parse rules out).
pub fn sale_credit_row(day: Day) -> Result<&'static SaleCreditRow, SaleCreditError> {
    sale_credit_defaults()?
        .iter()
        .find(|row| row.meta.covers(day))
        .ok_or_else(|| SaleCreditError::InvalidTable {
            reason: format!("{TABLE}: no row covers day {day}"),
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::market::table_rows::RowStatus;
    use crate::model::days_from_civil;

    fn header() -> String {
        HEADER.join(",")
    }

    fn percent_on(year: i32, month: u32, day: u32) -> Option<f64> {
        sale_credit_row(days_from_civil(year, month, day))
            .ok()
            .map(|row| row.same_day_percent)
    }

    #[test]
    fn the_built_in_table_parses() {
        let rows = sale_credit_defaults().map_err(|e| e.to_string());
        assert_eq!(rows.map(|r| r.len()), Ok(3));
    }

    /// Zerodha Z-Connect of 2020-11-30 and Zerodha support (read 2026-11-12).
    #[test]
    fn eighty_percent_is_credited_on_the_day_from_december_2020_to_october_2024() {
        let rows = [
            ((2020, 11, 30), 100.0),
            ((2020, 12, 1), 80.0),
            ((2024, 10, 6), 80.0),
            ((2024, 10, 7), 100.0),
        ];
        for ((year, month, day), percent) in rows {
            assert_eq!(
                percent_on(year, month, day),
                Some(percent),
                "{year}-{month}-{day}"
            );
        }
    }

    #[test]
    fn the_rest_of_a_sale_is_what_the_row_withholds() {
        // 80 percent on the day: of 9,00,000, 1,80,000 wait for the next day.
        let row = sale_credit_row(days_from_civil(2023, 3, 1));
        assert_eq!(row.map(|r| r.next_day_rupees(900_000.0)), Ok(180_000.0));
    }

    #[test]
    fn only_the_row_before_the_peak_margin_rules_is_unverified() {
        let rows = sale_credit_defaults().unwrap_or_default();
        let statuses: Vec<RowStatus> = rows.iter().map(|row| row.meta.status).collect();
        assert_eq!(
            statuses,
            [
                RowStatus::Unverified,
                RowStatus::Verified,
                RowStatus::Verified
            ]
        );
    }

    #[test]
    fn a_malformed_table_is_an_error() {
        let zero = format!("{}\nS/1,,,0,src,verified,\n", header());
        assert!(parse_table(&zero).is_err_and(|e| e.contains("line 2")));
        let gap = format!(
            "{}\nS/1,,2020-11-29,100,src,verified,\nS/2,2020-12-01,,80,src,verified,\n",
            header()
        );
        assert!(parse_table(&gap).is_err_and(|e| e.contains("S/2")));
        let closed = format!("{}\nS/1,,2020-11-30,100,src,verified,\n", header());
        assert!(parse_table(&closed).is_err_and(|e| e.contains("in force")));
        let late_start = format!("{}\nS/1,2020-01-01,,100,src,verified,\n", header());
        assert!(parse_table(&late_start).is_err());
    }

    #[test]
    fn a_malformed_table_is_reported_as_the_sale_credit_table() {
        let zero = format!("{}\nS/1,,,0,src,verified,\n", header());
        let message = checked_table(&zero).map_err(|error| error.to_string());
        assert!(message.is_err_and(|text| text.starts_with(
            "the built-in sale-credit table is malformed: sale_credit_defaults.csv line 2: "
        )));
    }
}
