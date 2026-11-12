//! The default slippage table (04-execution 4.3): `slippage_defaults.csv`, embedded and
//! parsed once, and its lookup by slippage class and trade date.
//!
//! An instrument's class follows from its exchange and kind, except two choices the
//! caller makes: whether the underlying of an NFO or BFO derivative is an index (from
//! the reference tables) and whether a cash-equity symbol is low volume (named by the
//! run). A malformed embedded file is a typed error (`SlippageError::InvalidTable`) and
//! fails the unit tests; it never panics.

use std::sync::OnceLock;

use crate::model::{Day, Exchange, InstrumentKind};

use super::slippage::{SlippageError, SlippageSpec};
use super::table_rows::{at_line, parse_meta, records, RowMeta};

const DATA: &str = include_str!("slippage_defaults.csv");
const TABLE: &str = "slippage_defaults.csv";
const HEADER: [&str; 10] = [
    "row_id",
    "class",
    "valid_from",
    "valid_to",
    "model",
    "amount",
    "min_ticks",
    "source",
    "status",
    "notes",
];
/// Position of `valid_from` in a record.
const DATES: usize = 2;

/// Which row of the default slippage table applies to an instrument.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SlippageClass {
    /// NSE and BSE equity, large and mid cap (the default for cash equity).
    EquityLargeMid,
    /// NSE and BSE equity that the run names as low volume.
    EquityLowVolume,
    /// Futures on an index (NFO, BFO).
    IndexFutures,
    /// Futures on a stock (NFO, BFO).
    StockFutures,
    /// Options on an index (NFO, BFO).
    IndexOptions,
    /// Options on a stock (NFO, BFO).
    StockOptions,
    /// MCX futures.
    McxFutures,
    /// MCX options.
    McxOptions,
    /// CDS futures and options.
    CurrencyDerivatives,
}

impl SlippageClass {
    /// Every class, in table order.
    pub const ALL: [SlippageClass; 9] = [
        SlippageClass::EquityLargeMid,
        SlippageClass::EquityLowVolume,
        SlippageClass::IndexFutures,
        SlippageClass::StockFutures,
        SlippageClass::IndexOptions,
        SlippageClass::StockOptions,
        SlippageClass::McxFutures,
        SlippageClass::McxOptions,
        SlippageClass::CurrencyDerivatives,
    ];

    /// The code used in the table and in results.
    pub fn code(self) -> &'static str {
        match self {
            SlippageClass::EquityLargeMid => "EQ_LARGE_MID",
            SlippageClass::EquityLowVolume => "EQ_LOW_VOLUME",
            SlippageClass::IndexFutures => "INDEX_FUT",
            SlippageClass::StockFutures => "STOCK_FUT",
            SlippageClass::IndexOptions => "INDEX_OPT",
            SlippageClass::StockOptions => "STOCK_OPT",
            SlippageClass::McxFutures => "MCX_FUT",
            SlippageClass::McxOptions => "MCX_OPT",
            SlippageClass::CurrencyDerivatives => "CDS",
        }
    }

    /// The class with this code.
    pub fn from_code(code: &str) -> Option<SlippageClass> {
        SlippageClass::ALL
            .into_iter()
            .find(|class| class.code() == code)
    }

    /// The class of an instrument when its exchange and kind decide it: cash equity is
    /// large and mid cap (the caller chooses the low-volume class), MCX and CDS by kind.
    /// `None` for NFO and BFO derivatives, whose class depends on whether the
    /// underlying is an index, and for anything that cannot be traded.
    pub fn of(exchange: Exchange, kind: InstrumentKind) -> Option<SlippageClass> {
        match (exchange, kind) {
            (Exchange::Nse | Exchange::Bse, InstrumentKind::Equity) => {
                Some(SlippageClass::EquityLargeMid)
            }
            (Exchange::Mcx, InstrumentKind::Future) => Some(SlippageClass::McxFutures),
            (Exchange::Mcx, InstrumentKind::Option) => Some(SlippageClass::McxOptions),
            (Exchange::Cds, InstrumentKind::Future | InstrumentKind::Option) => {
                Some(SlippageClass::CurrencyDerivatives)
            }
            _ => None,
        }
    }
}

/// One row of the default slippage table.
#[derive(Clone, Debug, PartialEq)]
pub struct SlippageRow {
    /// Id, dates, source and status.
    pub meta: RowMeta,
    /// The class it prices.
    pub class: SlippageClass,
    /// The slippage per side.
    pub spec: SlippageSpec,
}

fn parse_spec(model: &str, amount: &str, min_ticks: &str) -> Result<SlippageSpec, String> {
    let amount: f64 = amount
        .parse()
        .map_err(|_| format!("amount {amount:?} is not a number"))?;
    let min_ticks: u32 = min_ticks
        .parse()
        .map_err(|_| format!("min_ticks {min_ticks:?} is not a whole number"))?;
    if !(amount.is_finite() && amount > 0.0) {
        return Err(format!("amount {amount} must be a positive number"));
    }
    match model {
        "ticks" if amount.fract() == 0.0 && amount <= f64::from(u32::MAX) => {
            Ok(SlippageSpec::Ticks(amount as u32))
        }
        "bps" => Ok(SlippageSpec::Bps {
            bps: amount,
            min_ticks,
        }),
        "percent" => Ok(SlippageSpec::PercentOfPrice {
            percent: amount,
            min_ticks,
        }),
        other => Err(format!(
            "model {other:?} with amount {amount}: use ticks (a whole number), bps or percent"
        )),
    }
}

fn parse_row(fields: &[String]) -> Result<SlippageRow, String> {
    let [_, class, _, _, model, amount, min_ticks, ..] = fields else {
        return Err("too few fields".to_string());
    };
    Ok(SlippageRow {
        meta: parse_meta(fields, DATES)?,
        class: SlippageClass::from_code(class).ok_or_else(|| format!("unknown class {class:?}"))?,
        spec: parse_spec(model, amount, min_ticks)?,
    })
}

/// Parse a slippage table: every row valid, ids unique, no two rows of one class on
/// one date.
fn parse_table(text: &str) -> Result<Vec<SlippageRow>, String> {
    let mut rows: Vec<SlippageRow> = Vec::new();
    for record in records(TABLE, text, &HEADER)? {
        let row = parse_row(&record.fields).map_err(|problem| at_line(TABLE, &record, &problem))?;
        let clash = rows.iter().find(|earlier| {
            earlier.meta.id == row.meta.id
                || (earlier.class == row.class && earlier.meta.overlaps(&row.meta))
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

/// The built-in default slippage rows, in file order.
///
/// Errors: `InvalidTable` if the embedded file is malformed (a unit test guards
/// against shipping one).
pub fn slippage_defaults() -> Result<&'static [SlippageRow], SlippageError> {
    static ROWS: OnceLock<Result<Vec<SlippageRow>, String>> = OnceLock::new();
    match ROWS.get_or_init(|| parse_table(DATA)) {
        Ok(rows) => Ok(rows),
        Err(reason) => Err(SlippageError::InvalidTable {
            reason: reason.clone(),
        }),
    }
}

/// The default slippage row of `class` on the IST trade date `day`.
///
/// Errors: `InvalidTable` as `slippage_defaults`; `NoDefault` when no row of the class
/// covers `day`.
pub fn default_slippage(
    class: SlippageClass,
    day: Day,
) -> Result<&'static SlippageRow, SlippageError> {
    slippage_defaults()?
        .iter()
        .find(|row| row.class == class && row.meta.covers(day))
        .ok_or(SlippageError::NoDefault {
            class: class.code(),
            day,
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::market::table_rows::RowStatus;
    use crate::model::days_from_civil;

    fn spec_on(class: SlippageClass) -> SlippageSpec {
        let day = days_from_civil(2025, 7, 1);
        match default_slippage(class, day) {
            Ok(row) => row.spec,
            Err(error) => panic!("{error}"),
        }
    }

    /// Every row of 04-execution-and-market-model 4.3, as the document states it.
    #[test]
    fn each_class_has_the_figure_of_the_prd_table() {
        let bps = |bps: f64| SlippageSpec::Bps { bps, min_ticks: 1 };
        let percent = |percent: f64| SlippageSpec::PercentOfPrice {
            percent,
            min_ticks: 1,
        };
        let expected = [
            (SlippageClass::EquityLargeMid, bps(2.0)),
            (SlippageClass::EquityLowVolume, bps(5.0)),
            (SlippageClass::IndexFutures, SlippageSpec::Ticks(1)),
            (SlippageClass::StockFutures, SlippageSpec::Ticks(2)),
            (SlippageClass::IndexOptions, percent(0.5)),
            (SlippageClass::StockOptions, percent(1.0)),
            (SlippageClass::McxFutures, SlippageSpec::Ticks(1)),
            (SlippageClass::McxOptions, percent(1.0)),
            (SlippageClass::CurrencyDerivatives, SlippageSpec::Ticks(1)),
        ];
        for (class, spec) in expected {
            assert_eq!(spec_on(class), spec, "{}", class.code());
        }
    }

    #[test]
    fn the_built_in_table_is_one_open_row_per_class_and_all_verified() {
        let rows = slippage_defaults().map_err(|e| e.to_string());
        assert_eq!(rows.as_ref().map(|r| r.len()), Ok(SlippageClass::ALL.len()));
        for row in rows.unwrap_or_default() {
            assert_eq!(row.meta.status, RowStatus::Verified, "{}", row.meta.id);
            assert_eq!((row.meta.valid_from, row.meta.valid_to), (None, None));
        }
    }

    #[test]
    fn classes_follow_from_exchange_and_kind_except_nfo_and_bfo_derivatives() {
        use InstrumentKind::{Equity, Future, Option};
        assert_eq!(
            SlippageClass::of(Exchange::Nse, Equity),
            Some(SlippageClass::EquityLargeMid)
        );
        assert_eq!(
            SlippageClass::of(Exchange::Mcx, Option),
            Some(SlippageClass::McxOptions)
        );
        assert_eq!(SlippageClass::of(Exchange::Nfo, Future), None);
        assert_eq!(
            SlippageClass::of(Exchange::NseIndex, InstrumentKind::Index),
            None
        );
    }

    #[test]
    fn a_malformed_table_is_an_error_naming_the_line() {
        let header = HEADER.join(",");
        let unknown = format!("{header}\nX/1,NOPE,,,ticks,1,0,src,verified,\n");
        assert!(parse_table(&unknown).is_err_and(|e| e.contains("line 2")));
        let fractional_ticks = format!("{header}\nX/1,CDS,,,ticks,1.5,0,src,verified,\n");
        assert!(parse_table(&fractional_ticks).is_err());
        let overlap = format!(
            "{header}\nX/1,CDS,,2025-01-31,ticks,1,0,src,verified,\nX/2,CDS,2025-01-31,,ticks,2,0,src,verified,\n"
        );
        assert!(parse_table(&overlap).is_err_and(|e| e.contains("X/2")));
    }

    #[test]
    fn a_date_no_row_covers_has_no_default() {
        let header = HEADER.join(",");
        let dated = format!("{header}\nX/1,CDS,2025-01-01,,ticks,1,0,src,verified,\n");
        let rows = parse_table(&dated).unwrap_or_default();
        assert!(!rows.is_empty() && !rows[0].meta.covers(days_from_civil(2024, 12, 31)));
    }
}
