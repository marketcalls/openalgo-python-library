//! oa_bt: the OpenAlgo backtest and execution kernel.
//!
//! One crate, organised as modules: model, market, data, metrics, and the kernel
//! (engine, exec, portfolio, rms, sizing; see `engine` for the architecture). It has no
//! dependencies and knows nothing about Python; `oa_py` binds it into
//! `openalgo._oaindicators.backtest`.
//!
//! Conventions shared by every module:
//! - Timestamps are UTC nanoseconds since the Unix epoch (`model::TsNanos`).
//! - Session dates are IST calendar dates as days since 1970-01-01 (`model::Day`).
//! - Prices and money are `f64` rupees; quantities are `f64` units (never lots).

pub mod data;
#[cfg(all(feature = "costs", feature = "time"))]
pub mod engine;
#[cfg(all(feature = "costs", feature = "time"))]
pub mod exec;
pub mod market;
#[cfg(feature = "metrics")]
pub mod metrics;
pub mod model;
#[cfg(all(feature = "costs", feature = "time"))]
pub mod portfolio;
#[cfg(all(feature = "costs", feature = "time"))]
pub mod rms;
pub mod sizing;

/// Version of the kernel, reported in run manifests.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
