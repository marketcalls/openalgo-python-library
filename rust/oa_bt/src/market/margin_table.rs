//! The dated default margin table (04-execution 5): `margin_defaults.csv`, embedded and
//! parsed once, and its lookup by instrument kind, product, underlying and trade date.
//!
//! A malformed embedded file is a typed error (`MarginError::InvalidTable`) and fails
//! the unit tests; it never panics.

use std::sync::OnceLock;

use crate::model::{Day, InstrumentKind, Product};

use super::margin::MarginError;
use super::table_rows::{at_line, parse_meta, records, RowMeta};

const DATA: &str = include_str!("margin_defaults.csv");
const TABLE: &str = "margin_defaults.csv";
const HEADER: [&str; 11] = [
    "row_id",
    "kind",
    "product",
    "underlying",
    "valid_from",
    "valid_to",
    "model",
    "percent",
    "source",
    "status",
    "notes",
];
/// Position of `valid_from` in a record.
const DATES: usize = 4;
/// The `product` value of a row that applies to every product.
const ANY_PRODUCT: &str = "*";

/// How a row turns a position's value into margin.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MarginBasis {
    /// Value divided by the leverage asked for (1 when none); `least_percent` is the
    /// smallest margin allowed, so the most leverage is 100 / `least_percent`.
    Leverage { least_percent: f64 },
    /// `percent` of the value; no leverage can be asked for.
    Percent { percent: f64 },
}

/// One row of the default margin table.
#[derive(Clone, Debug, PartialEq)]
pub struct MarginRow {
    /// Id, dates, source and status.
    pub meta: RowMeta,
    /// Equity, future or option.
    pub kind: InstrumentKind,
    /// The product, or `None` for every product.
    pub product: Option<Product>,
    /// The underlying, or `None` for every underlying.
    pub underlying: Option<String>,
    /// How the margin is computed.
    pub basis: MarginBasis,
}

impl MarginRow {
    /// True when the row prices this kind, product and underlying on `day`.
    fn matches(
        &self,
        kind: InstrumentKind,
        product: Product,
        underlying: Option<&str>,
        day: Day,
    ) -> bool {
        let underlying_matches = match (&self.underlying, underlying) {
            (None, _) => true,
            (Some(row), Some(asked)) => row.eq_ignore_ascii_case(asked.trim()),
            (Some(_), None) => false,
        };
        self.kind == kind
            && self.product.map_or(true, |p| p == product)
            && underlying_matches
            && self.meta.covers(day)
    }
}

fn parse_basis(model: &str, percent: &str) -> Result<MarginBasis, String> {
    let value: f64 = percent
        .parse()
        .map_err(|_| format!("percent {percent:?} is not a number"))?;
    if !(value.is_finite() && value > 0.0 && value <= 100.0) {
        return Err(format!("percent {value} must be above 0 and at most 100"));
    }
    match model {
        "leverage" => Ok(MarginBasis::Leverage {
            least_percent: value,
        }),
        "percent" => Ok(MarginBasis::Percent { percent: value }),
        other => Err(format!("model {other:?} must be leverage or percent")),
    }
}

fn parse_row(fields: &[String]) -> Result<MarginRow, String> {
    let [_, kind, product, underlying, _, _, model, percent, ..] = fields else {
        return Err("too few fields".to_string());
    };
    let kind = match InstrumentKind::from_code(kind) {
        Some(kind @ (InstrumentKind::Equity | InstrumentKind::Future | InstrumentKind::Option)) => {
            kind
        }
        _ => return Err(format!("kind {kind:?} must be EQ, FUT or OPT")),
    };
    let product = match product.as_str() {
        ANY_PRODUCT => None,
        code => Some(
            Product::from_code(code)
                .ok_or_else(|| format!("product {code:?} must be CNC, MIS, NRML or *"))?,
        ),
    };
    Ok(MarginRow {
        meta: parse_meta(fields, DATES)?,
        kind,
        product,
        underlying: (!underlying.is_empty()).then(|| underlying.clone()),
        basis: parse_basis(model, percent)?,
    })
}

/// Parse a margin table: every row valid, ids unique, no two rows of one kind, product
/// and underlying on one date.
fn parse_table(text: &str) -> Result<Vec<MarginRow>, String> {
    let mut rows: Vec<MarginRow> = Vec::new();
    for record in records(TABLE, text, &HEADER)? {
        let row = parse_row(&record.fields).map_err(|problem| at_line(TABLE, &record, &problem))?;
        let clash = rows.iter().find(|earlier| {
            let same_key = earlier.kind == row.kind
                && earlier.product == row.product
                && earlier.underlying == row.underlying;
            earlier.meta.id == row.meta.id || (same_key && earlier.meta.overlaps(&row.meta))
        });
        if let Some(earlier) = clash {
            let problem = format!(
                "row {} repeats the id or dates of {}",
                row.meta.id, earlier.meta.id
            );
            return Err(at_line(TABLE, &record, &problem));
        }
        rows.push(row);
    }
    Ok(rows)
}

/// The built-in default margin rows, in file order.
///
/// Errors: `InvalidTable` if the embedded file is malformed (a unit test guards
/// against shipping one).
pub fn margin_defaults() -> Result<&'static [MarginRow], MarginError> {
    static ROWS: OnceLock<Result<Vec<MarginRow>, String>> = OnceLock::new();
    match ROWS.get_or_init(|| parse_table(DATA)) {
        Ok(rows) => Ok(rows),
        Err(reason) => Err(MarginError::InvalidTable {
            reason: reason.clone(),
        }),
    }
}

/// The row that prices a position of `kind` in `product` on `underlying` on the IST
/// trade date `day`, or `None` when the table has none.
///
/// Errors: `InvalidTable` as `margin_defaults`.
pub fn margin_row(
    kind: InstrumentKind,
    product: Product,
    underlying: Option<&str>,
    day: Day,
) -> Result<Option<&'static MarginRow>, MarginError> {
    Ok(margin_defaults()?
        .iter()
        .find(|row| row.matches(kind, product, underlying, day)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::market::table_rows::RowStatus;
    use crate::model::days_from_civil;

    fn header() -> String {
        HEADER.join(",")
    }

    #[test]
    fn the_built_in_table_parses() {
        let rows = margin_defaults().map_err(|e| e.to_string());
        // 4 equity and option rows, 5 NSE index futures, NIFTYFPI, 218 NSE stock
        // futures, 30 MCX futures and BANKEX.
        assert_eq!(rows.map(|r| r.len()), Ok(259));
    }

    #[test]
    fn futures_rows_are_one_day_observations_and_unverified() {
        for row in margin_defaults().unwrap_or_default() {
            if row.kind == InstrumentKind::Future {
                assert_eq!(row.meta.status, RowStatus::Unverified, "{}", row.meta.id);
            }
        }
    }

    #[test]
    fn a_row_for_one_underlying_does_not_price_another() {
        let day = days_from_civil(2025, 7, 1);
        let nifty = margin_row(InstrumentKind::Future, Product::Nrml, Some("nifty"), day);
        let sensex = margin_row(InstrumentKind::Future, Product::Nrml, Some("SENSEX"), day);
        assert_eq!(
            nifty.map(|r| r.map(|r| r.meta.id.as_str())),
            Ok(Some("FUT_NIFTY/1"))
        );
        assert_eq!(sensex.map(|r| r.is_none()), Ok(true));
    }

    #[test]
    fn the_intraday_row_changes_on_the_peak_margin_date() {
        let id = |y, m, d| {
            margin_row(
                InstrumentKind::Equity,
                Product::Mis,
                None,
                days_from_civil(y, m, d),
            )
            .ok()
            .flatten()
            .map(|row| row.meta.id.clone())
        };
        assert_eq!(id(2021, 8, 31).as_deref(), Some("MIS_EQ/1"));
        assert_eq!(id(2021, 9, 1).as_deref(), Some("MIS_EQ/2"));
    }

    #[test]
    fn a_malformed_table_is_an_error_naming_the_line() {
        let bad_kind = format!("{}\nX/1,INDEX,*,,,,percent,10,src,verified,\n", header());
        assert!(parse_table(&bad_kind).is_err_and(|e| e.contains("line 2")));
        let bad_percent = format!("{}\nX/1,FUT,*,,,,percent,0,src,verified,\n", header());
        assert!(parse_table(&bad_percent).is_err());
        let overlap = format!(
            "{}\nX/1,FUT,*,A,,,percent,10,src,verified,\nX/2,FUT,*,A,2025-01-01,,percent,12,src,verified,\n",
            header()
        );
        assert!(parse_table(&overlap).is_err_and(|e| e.contains("X/2")));
    }
}
