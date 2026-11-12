//! Bindings for the `oa_bt` kernel, exposed as `openalgo._oaindicators.backtest`.
//!
//! Each area has its own file with a `register` function, so areas can be worked on
//! independently. Arrays cross the boundary as contiguous NumPy arrays; timestamps are
//! int64 UTC nanoseconds.

use pyo3::prelude::*;

mod calendar;
mod costs;
mod metrics;
mod resample;

/// Version of the Rust kernel (recorded in run manifests).
#[pyfunction]
fn engine_version() -> &'static str {
    oa_bt::VERSION
}

pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new_bound(parent.py(), "backtest")?;
    m.add_function(wrap_pyfunction!(engine_version, &m)?)?;
    costs::register(&m)?;
    calendar::register(&m)?;
    resample::register(&m)?;
    metrics::register(&m)?;
    parent.add_submodule(&m)?;
    Ok(())
}
