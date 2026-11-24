//! A live session's saved state and the broker's own fills across the boundary
//! (`oa_bt::engine::venue_state`, `venue_external`; 16-session-design R14 and R15).
//!
//! - `live_state()`: the session's state as plain data, a dict of `None`, `bool`, `int`,
//!   `float`, `str`, `list` and `dict` with `str` keys only, so `json.dumps` writes it and
//!   `json.loads` gives the same dict back (no tuple, no number that is not finite). Its
//!   keys: "layout" (the layout version), "day", "slice" and "stage" (where the session
//!   stands), "clock_ns", "instruments", "account" (the cash), "positions" (with their
//!   tranches), "legs", "orders" (each working order with its "order" id, "role",
//!   "reason", "sent", "request", "status", units, filled units and what its fills were
//!   charged), "desk" (cancellations and modifications waiting for the broker's answer,
//!   the exits of the positions, the legs' stops, the targets fired, the entry locks, the
//!   sizing cap), "commands" (not yet taken), "risk" (trading state, daily loss totals,
//!   drawdown peak, the cooldown and time exit counters), "open_trades" and
//!   "session_buys". An order is named by its id while it works and by "ended" once it
//!   has ended. An order's "sent" is False only for a strategy exit that waits for the
//!   cancellation of the orders on its position's closing side: nothing of it is at the
//!   broker, so the restart's order check (16-session-design 3.6) skips it.
//! - `restore_live_state(state)`: give a saved state to a session built for a restart,
//!   once, after its catch-up and before any order. Returns `{saved id: new id}` for
//!   every restored working order (the saved orders are numbered from o1 again). The live
//!   layer rebuilds its venue map from these pairs alone: later orders take the next ids,
//!   which the earlier start may have given to orders that have ended since. The saved
//!   "commands" come back for `take_commands`, which drops those of an order the restart
//!   ends first (a rejection, a cancellation or a whole fill applied before they are
//!   taken); whether one still to send reached the broker before the crash is for the
//!   journal to say. The saved entry locks are not restored (the session keeps its own;
//!   the live layer sets again those that last the day, read from "desk"), and the saved
//!   sizing cap only into a session that has none. The events the saved session had
//!   recorded and not yet delivered are lost with it. A state of another day, other
//!   instruments or another capital, of another layout, one that cannot be read, nests
//!   deeper than any saved state, or whose parts do not hold together (the message names
//!   the field) raises `NativeEngineInputError`; a session that has orders or positions
//!   of its own, was restored already, has begun its close or stands before the saved
//!   point, a state saved while the session closed, and a backtest, raise
//!   `NativeEngineError`. Nothing of a refused state is restored.
//! - Since layout 6 a state also holds "gtts" (every GTT of the session with its
//!   "gtt" id, "status", "role" ("stop" or "order"), "order" and "terms") and, in "desk",
//!   what the session follows of them and an engine exit's fill "flags"; a same-day
//!   restore keeps every GTT's id.
//! - `carried_state()`: once the day's session has closed (after its close, or
//!   `stop_session()`), the state the next trading day's session takes, as plain data like
//!   `live_state()`'s: "kind" "carried", the "day" it closed, "instruments", "account",
//!   the open "positions" and their "legs", the strategy's GTT "orders" resting and the
//!   "gtts" at the venue (each with "to_cancel": True for a stop GTT that holds no open
//!   leg's stop, which the next session cancels at once), "stop_gtt_rejections" (each open
//!   leg's stop GTTs the broker rejected after placement), "risk" (the trading state, the
//!   daily loss measure's totals, the drawdown peak and the slices each position has been
//!   held) and the "open_trades". Before the close it raises `NativeEngineError`.
//! - `restore_carried_state(state)`: give it to the session of a later trading day, once,
//!   after its catch-up and before any order. Returns `{"gtts": {saved id: new id},
//!   "orders": {saved id: new id}}`. The daily loss measure and the trading state start the
//!   session as a backtest's do at a session boundary; positions are marked at the
//!   catch-up's last close; a stop GTT to cancel is cancelled at once ("cancel_gtt"). A
//!   state of the same or a later day, other instruments, another capital, an MIS
//!   position, or parts that do not hold together raises `NativeEngineInputError` (the
//!   message names the part of `carried_state`), and nothing of it is restored.
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

/// The call whose state `path` reads: its first part, "live_state" or "carried_state".
fn call_of(path: &str) -> &str {
    path.split(['.', '[']).next().unwrap_or(path)
}

/// The error for a value at `path` that is not plain data.
fn not_plain(path: &str, value: &Bound<'_, PyAny>) -> PyErr {
    let kind = value
        .get_type()
        .name()
        .map_or_else(|_| "object".to_string(), |name| name.to_string());
    let call = call_of(path);
    NativeEngineInputError::new_err(format!(
        "the saved state cannot be restored: {path} is a {kind}, not plain data; pass the \
         state {call}() returned, unchanged or read back from JSON"
    ))
}

/// How deep a saved state's values may nest. The deepest value `live_state()` writes is
/// about 10 levels down (the state, its desk, a modification waiting, its request, the
/// leg rules, a target, its distance, the distance's value); a value much deeper is not a
/// saved state, and reading it level by level would exhaust the stack.
const MAX_STATE_DEPTH: usize = 32;

/// A Python value of a saved state as the kernel's plain data; `path` names it in errors,
/// and `depth` is how far below the state it sits (the state itself is 0).
///
/// Errors: `NativeEngineInputError` for a value that is not plain data or nests deeper
/// than `MAX_STATE_DEPTH`; `OverflowError` for a whole number beyond 64 bits.
fn from_python(value: &Bound<'_, PyAny>, path: &str, depth: usize) -> PyResult<StateValue> {
    if depth > MAX_STATE_DEPTH {
        let call = call_of(path);
        return Err(NativeEngineInputError::new_err(format!(
            "the saved state cannot be restored: {path} is nested deeper than \
             {MAX_STATE_DEPTH} levels, which no saved state is; pass the state {call}() \
             returned, unchanged or read back from JSON"
        )));
    }
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
            .map(|(index, item)| from_python(&item?, &format!("{path}[{index}]"), depth + 1))
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
            let item = from_python(&item, &format!("{path}.{key}"), depth + 1)?;
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
    let state = from_python(state, "live_state", 0)?;
    let pairs = engine.restore_live_state(&state).map_err(engine_error)?;
    let ids = PyDict::new_bound(py);
    for (saved, new) in pairs {
        ids.set_item(saved.0, new.0)?;
    }
    Ok(ids)
}

/// The state carried to the next trading day as plain Python data (module docs).
///
/// Errors: `NativeEngineError` before the session has closed, and in a backtest.
pub(super) fn carried_state(py: Python<'_>, engine: &Engine) -> PyResult<PyObject> {
    let state = engine.carried_state().map_err(engine_error)?;
    to_python(py, &state)
}

/// Restore a carried `state` into `engine`; the new ids of its GTTs and GTT orders.
///
/// Errors: those of `restore_live_state`.
pub(super) fn restore_carried_state<'py>(
    py: Python<'py>,
    engine: &mut Engine,
    state: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let state = from_python(state, "carried_state", 0)?;
    let carried = engine.restore_carried_state(&state).map_err(engine_error)?;
    let gtts = PyDict::new_bound(py);
    for (saved, new) in carried.gtts {
        gtts.set_item(saved.0, new.0)?;
    }
    let orders = PyDict::new_bound(py);
    for (saved, new) in carried.orders {
        orders.set_item(saved.0, new.0)?;
    }
    let ids = PyDict::new_bound(py);
    ids.set_item("gtts", gtts)?;
    ids.set_item("orders", orders)?;
    Ok(ids)
}

/// A broker's fill with no order, as `apply_external_fill` takes it: (instrument index,
/// product, side, units, price in rupees, at in UTC nanoseconds, exit reason).
pub(super) type ExternalFill<'a> = (u32, &'a str, &'a str, f64, f64, i64, &'a str);

/// Record the broker's fill with no order (module docs); its stamp.
///
/// Errors: `NativeEngineInputError` for an unknown product, side or reason;
/// `VenueRefusalError` for a refusal of the kernel.
pub(super) fn apply_external_fill(
    py: Python<'_>,
    engine: &mut Engine,
    (instrument, product, side, units, price, at, reason): ExternalFill<'_>,
) -> PyResult<StampTuple> {
    let product = parse_product(product)?;
    let side = Side::from_code(side).ok_or_else(|| unknown("side", side, "BUY or SELL"))?;
    let reason = ExitReason::from_code(reason).ok_or_else(|| {
        let codes: Vec<&str> = ExitReason::ALL.iter().map(|reason| reason.code()).collect();
        unknown("exit reason", reason, &codes.join(", "))
    })?;
    let instrument = InstrumentId(instrument);
    let recorded_fill =
        engine.apply_external_fill(instrument, product, side, units, price, at, reason);
    recorded(py, recorded_fill)
}
