//! The errors the engine binding raises (see the parent module's docs): Python's
//! `NotImplementedError` for a part not built yet, `NativeEngineInputError` for bad
//! configuration, data or requests, `NativeEngineError` for anything else, and the
//! package's `VenueRefusalError` for a result of a live session's venue the kernel
//! refused (`venue`).

use oa_bt::engine::{KernelError, UNKNOWN_ORDER_KIND};
use pyo3::create_exception;
use pyo3::exceptions::{PyException, PyNotImplementedError};
use pyo3::prelude::*;

create_exception!(
    backtest,
    NativeEngineError,
    PyException,
    "The engine cannot go on: a call at the wrong time, or a kernel defect."
);
create_exception!(
    backtest,
    NativeEngineInputError,
    NativeEngineError,
    "The configuration, the data or an order request cannot be used."
);

/// Add the native exception types to the module.
pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();
    m.add(
        "NativeEngineError",
        py.get_type_bound::<NativeEngineError>(),
    )?;
    m.add(
        "NativeEngineInputError",
        py.get_type_bound::<NativeEngineInputError>(),
    )
}

/// A kernel error as the Python exception that says what kind it is.
pub(super) fn engine_error(error: KernelError) -> PyErr {
    let message = error.to_string();
    error_of_kind(&error, message)
}

/// The Python exception of `error`'s kind (`engine_error`) with `message`: a portfolio
/// run's fund names the strategy whose engine refused (`super::super::fund`).
pub(in crate::backtest) fn error_of_kind(error: &KernelError, message: String) -> PyErr {
    if error.not_built_feature().is_some() {
        PyNotImplementedError::new_err(message)
    } else if error.is_input_error() {
        NativeEngineInputError::new_err(message)
    } else {
        NativeEngineError::new_err(message)
    }
}

/// The module that defines the package's exceptions, the one source of them.
const ERRORS_MODULE: &str = "openalgo.backtest.errors";

/// The error of a venue result: `VenueRefusalError` with its kind for a refusal (a venue
/// error, an unknown order), `engine_error` for anything else.
pub(super) fn venue_result_error(py: Python<'_>, error: KernelError) -> PyErr {
    let kind = match &error {
        KernelError::Venue(refusal) => refusal.kind(),
        KernelError::UnknownOrder(_) => UNKNOWN_ORDER_KIND,
        _ => return engine_error(error),
    };
    let refusal = py
        .import_bound(ERRORS_MODULE)
        .and_then(|module| module.getattr("VenueRefusalError"))
        .and_then(|class| class.call1((error.to_string(), kind)));
    match refusal {
        Ok(value) => PyErr::from_value_bound(value),
        Err(import_failed) => import_failed,
    }
}
