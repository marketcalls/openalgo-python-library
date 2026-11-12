//! Tests of the calendar, one file per topic.
//!
//! Expected instants are literal UTC nanoseconds with the IST time in a comment, or
//! counts worked out by hand in the test; none is produced by the code under test.
//! Market facts used as expectations (hours, closes) are the rows of the embedded
//! tables, whose sources are cited in the CSV files.

mod annualise;
mod grid;
mod intraday_times;
mod label;
mod lattice;
mod offset;
mod sessions;
mod tables;
mod validate;
