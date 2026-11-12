//! Bindings for `oa_bt::data::resample`.
//!
//! Bars cross the boundary as one tuple of seven contiguous arrays, (ts_open, open,
//! high, low, close, volume, oi): ts_open int64 UTC nanoseconds and the rest float64.
//! They come back as a pair: a tuple of eight arrays (ts_open, ts_close and the six
//! value columns) and a dict of what was left out (`pre_open_bars`, `post_close_bars`,
//! `closed_day_bars`, `empty_buckets`). `closed_day` is "error" or "drop".
//!
//! Errors are raised as `NativeResampleError`, its subclass `NativeClosedDayError` for
//! bars on a date with no session, and `NativeSeriesError` for a missing or
//! out-of-range timestamp; the Python package converts them to its public error types.

// Two lints fire inside PyO3 0.22 macro expansions on newer toolchains, not in this
// code: `create_exception!` checks `cfg(feature = "gil-refs")`, which this crate does
// not declare, and `#[pyfunction]` converts the error of every `PyResult` return with
// `.into()`.
#![allow(unexpected_cfgs, clippy::useless_conversion)]

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use oa_bt::data::resample as rs;
use pyo3::create_exception;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use super::calendar::{calendar_error, parse_exchange, NativeCalendar, NativeSeriesError};

create_exception!(
    backtest,
    NativeResampleError,
    PyException,
    "Bars cannot be resampled (bad interval, unsorted or duplicate rows)."
);
create_exception!(
    backtest,
    NativeClosedDayError,
    NativeResampleError,
    "Bars fall on a date the calendar has no session for."
);

fn resample_error(error: rs::ResampleError) -> PyErr {
    match error {
        rs::ResampleError::ClosedDay { .. } => NativeClosedDayError::new_err(error.to_string()),
        rs::ResampleError::UnsupportedTimestamp { .. } => {
            NativeSeriesError::new_err(error.to_string())
        }
        rs::ResampleError::Calendar(inner) => calendar_error(inner),
        other => NativeResampleError::new_err(other.to_string()),
    }
}

/// The input bars: (ts_open, open, high, low, close, volume, oi).
#[derive(FromPyObject)]
struct BarArrays<'py>(
    PyReadonlyArray1<'py, i64>,
    PyReadonlyArray1<'py, f64>,
    PyReadonlyArray1<'py, f64>,
    PyReadonlyArray1<'py, f64>,
    PyReadonlyArray1<'py, f64>,
    PyReadonlyArray1<'py, f64>,
    PyReadonlyArray1<'py, f64>,
);

impl BarArrays<'_> {
    /// Borrow the arrays as resampler input. Fails when an array is not contiguous.
    fn as_bars(&self) -> PyResult<rs::BarsIn<'_>> {
        Ok(rs::BarsIn {
            ts_open: self.0.as_slice()?,
            open: self.1.as_slice()?,
            high: self.2.as_slice()?,
            low: self.3.as_slice()?,
            close: self.4.as_slice()?,
            volume: self.5.as_slice()?,
            oi: self.6.as_slice()?,
        })
    }
}

fn parse_closed_day(policy: &str) -> PyResult<rs::ClosedDayPolicy> {
    match policy {
        "error" => Ok(rs::ClosedDayPolicy::Error),
        "drop" => Ok(rs::ClosedDayPolicy::Drop),
        other => Err(NativeResampleError::new_err(format!(
            "unknown closed-day policy {other:?}: use \"error\" or \"drop\""
        ))),
    }
}

/// The output bars: (ts_open, ts_close, open, high, low, close, volume, oi).
type BarColumns = (
    Py<PyArray1<i64>>,
    Py<PyArray1<i64>>,
    Py<PyArray1<f64>>,
    Py<PyArray1<f64>>,
    Py<PyArray1<f64>>,
    Py<PyArray1<f64>>,
    Py<PyArray1<f64>>,
    Py<PyArray1<f64>>,
);

fn output(py: Python<'_>, result: rs::Resampled) -> PyResult<(BarColumns, Bound<'_, PyDict>)> {
    let out = result.bars;
    let columns = (
        out.ts_open.into_pyarray_bound(py).unbind(),
        out.ts_close.into_pyarray_bound(py).unbind(),
        out.open.into_pyarray_bound(py).unbind(),
        out.high.into_pyarray_bound(py).unbind(),
        out.low.into_pyarray_bound(py).unbind(),
        out.close.into_pyarray_bound(py).unbind(),
        out.volume.into_pyarray_bound(py).unbind(),
        out.oi.into_pyarray_bound(py).unbind(),
    );
    let dropped = PyDict::new_bound(py);
    dropped.set_item("pre_open_bars", result.dropped.pre_open_bars)?;
    dropped.set_item("post_close_bars", result.dropped.post_close_bars)?;
    dropped.set_item("closed_day_bars", result.dropped.closed_day_bars)?;
    dropped.set_item("empty_buckets", result.dropped.empty_buckets)?;
    Ok((columns, dropped))
}

/// Resample intraday bars of `base_nanos` to `target_nanos` on the bar lattice.
#[pyfunction]
fn resample_intraday<'py>(
    py: Python<'py>,
    calendar: PyRef<'py, NativeCalendar>,
    exchange: &str,
    bars: BarArrays<'py>,
    base_nanos: i64,
    target_nanos: i64,
    closed_day: &str,
) -> PyResult<(BarColumns, Bound<'py, PyDict>)> {
    let result = rs::resample_intraday(
        &calendar.inner,
        parse_exchange(exchange)?,
        &bars.as_bars()?,
        base_nanos,
        target_nanos,
        parse_closed_day(closed_day)?,
    )
    .map_err(resample_error)?;
    output(py, result)
}

/// Resample intraday bars of `base_nanos` to one bar per IST session date.
#[pyfunction]
fn resample_daily<'py>(
    py: Python<'py>,
    calendar: PyRef<'py, NativeCalendar>,
    exchange: &str,
    bars: BarArrays<'py>,
    base_nanos: i64,
    closed_day: &str,
) -> PyResult<(BarColumns, Bound<'py, PyDict>)> {
    let result = rs::resample_daily(
        &calendar.inner,
        parse_exchange(exchange)?,
        &bars.as_bars()?,
        base_nanos,
        parse_closed_day(closed_day)?,
    )
    .map_err(resample_error)?;
    output(py, result)
}

/// Resample daily bars to weekly ("W"), monthly ("M"), quarterly ("Q") or yearly ("Y")
/// bars.
#[pyfunction]
fn resample_period<'py>(
    py: Python<'py>,
    calendar: PyRef<'py, NativeCalendar>,
    exchange: &str,
    bars: BarArrays<'py>,
    unit: &str,
    closed_day: &str,
) -> PyResult<(BarColumns, Bound<'py, PyDict>)> {
    let unit = rs::CalendarUnit::from_code(unit).ok_or_else(|| {
        NativeResampleError::new_err(format!(
            "unknown period {unit:?}: use \"W\", \"M\", \"Q\" or \"Y\""
        ))
    })?;
    let result = rs::resample_period(
        &calendar.inner,
        parse_exchange(exchange)?,
        &bars.as_bars()?,
        unit,
        parse_closed_day(closed_day)?,
    )
    .map_err(resample_error)?;
    output(py, result)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();
    m.add(
        "NativeResampleError",
        py.get_type_bound::<NativeResampleError>(),
    )?;
    m.add(
        "NativeClosedDayError",
        py.get_type_bound::<NativeClosedDayError>(),
    )?;
    m.add_function(wrap_pyfunction!(resample_intraday, m)?)?;
    m.add_function(wrap_pyfunction!(resample_daily, m)?)?;
    m.add_function(wrap_pyfunction!(resample_period, m)?)?;
    Ok(())
}
