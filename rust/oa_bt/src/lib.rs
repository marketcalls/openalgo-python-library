//! oa_bt: the OpenAlgo backtest and execution kernel.
//!
//! One crate, organised as modules: model, market, data, metrics, and the kernel
//! (engine, exec, portfolio, rms, sizing; see `engine` for the architecture), with the
//! fund that advances several engines over one account (`fund`). It has no
//! dependencies and knows nothing about Python; `oa_py` binds it into
//! `openalgo._oaindicators.backtest`.
//!
//! Conventions shared by every module:
//! - Timestamps are UTC nanoseconds since the Unix epoch (`model::TsNanos`).
//! - Session dates are IST calendar dates as days since 1970-01-01 (`model::Day`).
//! - Prices and money are `f64` rupees; quantities are `f64` units (never lots).

#[cfg(all(feature = "costs", feature = "time"))]
pub mod charges;
pub mod data;
#[cfg(all(feature = "costs", feature = "time"))]
pub mod engine;
#[cfg(all(feature = "costs", feature = "time"))]
pub mod exec;
#[cfg(all(feature = "costs", feature = "time"))]
pub mod fund;
pub mod market;
#[cfg(feature = "metrics")]
pub mod metrics;
pub mod model;
#[cfg(all(feature = "costs", feature = "time"))]
pub mod portfolio;
#[cfg(all(feature = "costs", feature = "time"))]
pub mod rms;
pub mod sizing;
// Read by the tests of `rms` and `fund`, which need both features.
#[cfg(all(test, feature = "costs", feature = "time"))]
pub(crate) mod test_json;

/// Version of the kernel, reported in run manifests.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
