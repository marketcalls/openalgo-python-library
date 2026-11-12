//! Condition strategies across the boundary: the strategy bars a condition is evaluated
//! on, out to Python, and the condition arrays with their order templates, in from
//! `openalgo/backtest/_conditions.py`, parsed into `oa_bt::engine::SignalArrays`.
//!
//! The spec dict has the keys `instrument` (registration index), `product` (CNC, MIS
//! or NRML), `bar_opens` (int64 UTC nanoseconds), `entry`, `exit`, `short`, `cover`
//! (bool arrays, or None when not declared), `entry_order` and `short_order` (request
//! dicts as `request.rs` reads them, or None) and `pyramiding`.

use numpy::{IntoPyArray, PyReadonlyArray1};
use oa_bt::data::resample::BarsOut;
use oa_bt::engine::{EntrySignal, SignalArrays};
use oa_bt::exec::InstrumentId;
use oa_bt::model::Product;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use super::request::order_request;
use super::setup::{field, optional, unknown};
use super::NativeEngineInputError;

/// A bool array of the spec, or None when the condition was not declared.
fn flags(dict: &Bound<'_, PyDict>, key: &str) -> PyResult<Option<Vec<bool>>> {
    let array: Option<PyReadonlyArray1<'_, bool>> = optional(dict, key)?;
    Ok(array.map(|values| values.as_array().to_vec()))
}

/// An entry condition with its order; both are given or neither.
fn entry_signal(
    dict: &Bound<'_, PyDict>,
    key: &str,
    order_key: &str,
) -> PyResult<Option<EntrySignal>> {
    let order: Option<Bound<'_, PyDict>> = optional(dict, order_key)?;
    match (flags(dict, key)?, order) {
        (Some(when), Some(order)) => Ok(Some(EntrySignal {
            when,
            order: order_request(&order)?,
        })),
        (None, None) => Ok(None),
        _ => Err(NativeEngineInputError::new_err(format!(
            "{key} and {order_key} are given together or not at all"
        ))),
    }
}

/// The condition arrays and order templates of a run from their spec dict.
pub(super) fn signal_arrays(dict: &Bound<'_, PyDict>) -> PyResult<SignalArrays> {
    let product: String = field(dict, "product")?;
    let bar_opens: PyReadonlyArray1<'_, i64> = field(dict, "bar_opens")?;
    Ok(SignalArrays {
        instrument: InstrumentId(field(dict, "instrument")?),
        product: Product::from_code(&product)
            .ok_or_else(|| unknown("product", &product, "CNC, MIS or NRML"))?,
        bar_opens: bar_opens.as_array().to_vec(),
        entry: entry_signal(dict, "entry", "entry_order")?,
        exit: flags(dict, "exit")?,
        short: entry_signal(dict, "short", "short_order")?,
        cover: flags(dict, "cover")?,
        pyramiding: field(dict, "pyramiding")?,
    })
}

/// Strategy bars as a dict of NumPy arrays: `ts_open` and `ts_close` (int64 UTC
/// nanoseconds), `open`, `high`, `low`, `close` (rupees) and `volume` (units).
pub(super) fn strategy_bars<'py>(py: Python<'py>, bars: &BarsOut) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new_bound(py);
    dict.set_item("ts_open", bars.ts_open.clone().into_pyarray_bound(py))?;
    dict.set_item("ts_close", bars.ts_close.clone().into_pyarray_bound(py))?;
    let prices = [
        ("open", &bars.open),
        ("high", &bars.high),
        ("low", &bars.low),
        ("close", &bars.close),
        ("volume", &bars.volume),
    ];
    for (name, column) in prices {
        dict.set_item(name, column.clone().into_pyarray_bound(py))?;
    }
    Ok(dict)
}
