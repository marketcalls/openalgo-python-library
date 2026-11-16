//! A live session's saved state and the broker's own fills across the boundary
//! (`oa_bt::engine::venue_state`, `venue_external`; 16-session-design R14 and R15).
//!
//! - `live_state()`: the session's state as plain data, a dict of `None`, `bool`, `int`,
//!   `float`, `str`, `list` and `dict` with `str` keys only, so `json.dumps` writes it and
//!   `json.loads` gives the same dict back (no tuple, no number that is not finite). Its
//!   keys: "layout" (the layout version), "day", "slice" and "stage" (where the session
//!   stands), "clock_ns", "instruments", "account" (the cash), "positions" (with their
//!   tranches), "legs", "orders" (each working order with its "order" id, "role",
//!   "reason", "request", "status", units, filled units and what its fills were charged),
//!   "desk" (cancellations and modifications waiting for the broker's answer, the exits
//!   of the positions, the legs' stops, the targets fired, the entry locks, the sizing
//!   cap), "commands" (not yet taken), "risk" (trading state, daily loss totals, drawdown
//!   peak, the cooldown and time exit counters), "open_trades" and "session_buys". An
//!   order is named by its id while it works and by "ended" once it has ended.
//! - `restore_live_state(state)`: give a saved state to a session built for a restart,
//!   once, after its catch-up and before any order. Returns `{saved id: new id}` for
//!   every restored working order (the saved orders are numbered from o1 again), so the
//!   live layer re-keys its venue map. A state of another day, other instruments or
//!   another capital, of another layout, or one that cannot be read (the message names
//!   the field) raises `NativeEngineInputError`; a session that has orders or positions
//!   of its own, was restored already, has begun its close or stands before the saved
//!   point, and a backtest, raise `NativeEngineError`. Nothing of a refused state is
//!   restored.
//! - `apply_external_fill(instrument, product, side, units, price, at, reason)`: the
//!   broker closed or reduced the position by itself (its square-off, a position closed
//!   by hand): `units` (above 0) on `side` ("BUY", "SELL") at `price` rupees at `at` (UTC
//!   nanoseconds), recorded with the exit reason `reason` (an `ExitReason` code such as
//!   "manual" or "scheduler") through the one fill path, with no order and the flag
//!   "external"; returns the stamp as `apply_fill` does. One that would not only close or
//!   reduce the position (on its side, of a flat position, through zero) raises
//!   `VenueRefusalError` of kind "beyond_position" and records nothing; the other
//!   refusals are those of `apply_fill`.

use oa_bt::engine::{Engine, StateValue};
use oa_bt::exec::InstrumentId;
use oa_bt::model::Side;
use oa_bt::rms::ExitReason;
use pyo3::exceptions::PyOverflowError;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyFloat, PyList, PyLong, PyString, PyTuple};

use super::errors::{engine_error, NativeEngineInputError};
use super::setup::{parse_product, unknown};
use super::venue::{recorded, StampTuple};

/// A saved state's value as the Python value it is (module docs).
fn to_python(py: Python<'_>, value: &StateValue) -> PyResult<PyObject> {
    Ok(match value {
        StateValue::Null => py.None(),
        StateValue::Bool(flag) => flag.into_py(py),
        StateValue::Int(whole) => whole.into_py(py),
        StateValue::Float(number) => number.into_py(py),
        StateValue::Text(text) => text.into_py(py),
        StateValue::List(items) => {
            let items = items
                .iter()
                .map(|item| to_python(py, item))
                .collect::<PyResult<Vec<_>>>()?;
            PyList::new_bound(py, items).into_py(py)
        }
        StateValue::Map(entries) => {
            let dict = PyDict::new_bound(py);
            for (key, item) in entries {
                dict.set_item(key, to_python(py, item)?)?;
            }
            dict.into_py(py)
        }
    })
}

/// The error for a value at `path` that is not plain data.
fn not_plain(path: &str, value: &Bound<'_, PyAny>) -> PyErr {
    let kind = value
        .get_type()
        .name()
        .map_or_else(|_| "object".to_string(), |name| name.to_string());
    NativeEngineInputError::new_err(format!(
        "the saved live state cannot be restored: {path} is a {kind}, not plain data; pass \
         the state live_state() returned, unchanged or read back from JSON"
    ))
}

/// A Python value of a saved state as the kernel's plain data; `path` names it in errors.
fn from_python(value: &Bound<'_, PyAny>, path: &str) -> PyResult<StateValue> {
    if value.is_none() {
        return Ok(StateValue::Null);
    }
    if let Ok(flag) = value.downcast::<PyBool>() {
        return Ok(StateValue::Bool(flag.is_true()));
    }
    if value.is_instance_of::<PyLong>() {
        return value.extract::<i64>().map(StateValue::Int).map_err(|_| {
            PyOverflowError::new_err(format!("{path}: a whole number beyond 64 bits"))
        });
    }
    if let Ok(number) = value.downcast::<PyFloat>() {
        return Ok(StateValue::Float(number.value()));
    }
    if let Ok(text) = value.downcast::<PyString>() {
        return Ok(StateValue::Text(text.to_cow()?.into_owned()));
    }
    if value.is_instance_of::<PyList>() || value.is_instance_of::<PyTuple>() {
        let items = value
            .iter()?
            .enumerate()
            .map(|(index, item)| from_python(&item?, &format!("{path}[{index}]")))
            .collect::<PyResult<Vec<_>>>()?;
        return Ok(StateValue::List(items));
    }
    if let Ok(dict) = value.downcast::<PyDict>() {
        let mut entries = Vec::with_capacity(dict.len());
        for (key, item) in dict.iter() {
            let Ok(key) = key.downcast::<PyString>() else {
                return Err(not_plain(&format!("a key of {path}"), &key));
            };
            let key = key.to_cow()?.into_owned();
            let item = from_python(&item, &format!("{path}.{key}"))?;
            entries.push((key, item));
        }
        return Ok(StateValue::Map(entries));
    }
    Err(not_plain(path, value))
}

/// The session's state as plain Python data (module docs).
pub(super) fn live_state(py: Python<'_>, engine: &Engine) -> PyResult<PyObject> {
    to_python(py, &engine.live_state())
}

/// Restore `state` into `engine`; `{saved id: new id}` of the restored working orders.
///
/// Errors: `NativeEngineInputError` for a value that is not plain data and for a state
/// refused for what it is; `NativeEngineError` for a session that cannot take one now.
pub(super) fn restore_live_state<'py>(
    py: Python<'py>,
    engine: &mut Engine,
    state: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let state = from_python(state, "live_state")?;
    let pairs = engine.restore_live_state(&state).map_err(engine_error)?;
    let ids = PyDict::new_bound(py);
    for (saved, new) in pairs {
        ids.set_item(saved.0, new.0)?;
    }
    Ok(ids)
}

/// The facts of a broker's fill with no order, as `apply_external_fill` takes them.
pub(super) struct ExternalFill<'a> {
    pub(super) instrument: u32,
    pub(super) product: &'a str,
    pub(super) side: &'a str,
    pub(super) units: f64,
    pub(super) price: f64,
    pub(super) at: i64,
    pub(super) reason: &'a str,
}

/// Record the broker's fill with no order (module docs); its stamp.
///
/// Errors: `NativeEngineInputError` for an unknown product, side or reason;
/// `VenueRefusalError` for a refusal of the kernel.
pub(super) fn apply_external_fill(
    py: Python<'_>,
    engine: &mut Engine,
    fill: ExternalFill<'_>,
) -> PyResult<StampTuple> {
    let product = parse_product(fill.product)?;
    let side =
        Side::from_code(fill.side).ok_or_else(|| unknown("side", fill.side, "BUY or SELL"))?;
    let reason = ExitReason::from_code(fill.reason).ok_or_else(|| {
        let codes: Vec<&str> = ExitReason::ALL.iter().map(|reason| reason.code()).collect();
        unknown("exit reason", fill.reason, &codes.join(", "))
    })?;
    let recorded_fill = engine.apply_external_fill(
        InstrumentId(fill.instrument),
        product,
        side,
        fill.units,
        fill.price,
        fill.at,
        reason,
    );
    recorded(py, recorded_fill)
}
