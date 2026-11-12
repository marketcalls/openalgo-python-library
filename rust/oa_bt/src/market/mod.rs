//! Indian market model: costs, session calendars, margin, slippage and the per-instrument
//! facts the market rules read.

#[cfg(feature = "time")]
pub mod calendar;
#[cfg(feature = "costs")]
pub mod costs;
#[cfg(all(feature = "costs", feature = "time"))]
pub mod facts;
#[cfg(feature = "time")]
pub mod margin;
#[cfg(feature = "time")]
mod margin_table;
#[cfg(all(feature = "costs", feature = "time"))]
pub mod slippage;
#[cfg(all(feature = "costs", feature = "time"))]
mod slippage_table;
#[cfg(feature = "time")]
pub mod table_rows;
