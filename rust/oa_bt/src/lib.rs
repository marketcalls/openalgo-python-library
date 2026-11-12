//! oa_bt: the OpenAlgo backtest and execution kernel.
//!
//! One crate, organised as modules (model, market, data, and later exec, rms,
//! portfolio, engine, expr, metrics). It has no dependencies and knows nothing about
//! Python; `oa_py` binds it into `openalgo._oaindicators.backtest`.
//!
//! Conventions shared by every module:
//! - Timestamps are UTC nanoseconds since the Unix epoch (`model::TsNanos`).
//! - Session dates are IST calendar dates as days since 1970-01-01 (`model::Day`).
//! - Prices and money are `f64` rupees; quantities are `f64` units (never lots).

pub mod data;
pub mod market;
#[cfg(feature = "metrics")]
pub mod metrics;
pub mod model;

/// Version of the kernel, reported in run manifests.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
