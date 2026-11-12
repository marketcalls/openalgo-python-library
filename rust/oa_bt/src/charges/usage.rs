//! Which rows of the dated market tables a run used, so that a result can say which
//! figures it rests on and warn about each unverified one.

use crate::market::table_rows::RowMeta;
use crate::model::Day;

/// A dated market table of the kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MarketTable {
    /// `slippage_defaults.csv`.
    Slippage,
    /// `margin_defaults.csv`.
    Margin,
}

impl MarketTable {
    /// "slippage" or "margin".
    pub fn code(self) -> &'static str {
        match self {
            MarketTable::Slippage => "slippage",
            MarketTable::Margin => "margin",
        }
    }
}

/// One row a run used.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TableUse {
    /// The table.
    pub table: MarketTable,
    /// The row: id, dates, source and status.
    pub row: &'static RowMeta,
    /// The first IST trade date it was used for.
    pub first_day: Day,
}
