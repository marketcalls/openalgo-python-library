//! Futures series across the boundary (`oa_bt::engine::futures`): the series dict
//! `openalgo/backtest/_futures.py` builds, the switches the kernel plans, the settlement
//! postings of a finished run, and the session arithmetic the Python package uses to
//! place a roll (one source of truth for the roll session).
//!
//! A series dict is `{"rule": "roll" | "settle_at_expiry", "contracts": [(instrument
//! index, hand-over day or None), ...]}`, days as days since 1970-01-01 (IST dates).

use oa_bt::engine::{
    roll_session_day, session_after, FuturesOutput, FuturesSeries, RollRule, SeriesContract, Switch,
};
use oa_bt::exec::InstrumentId;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use super::super::calendar::{parse_exchange, NativeCalendar, PyDay};
use super::engine_error;
use super::setup::{field, unknown};

/// A futures series from its dict (module docs).
pub(super) fn series(dict: &Bound<'_, PyDict>) -> PyResult<FuturesSeries> {
    let rule: String = field(dict, "rule")?;
    let rule = match rule.as_str() {
        "roll" => RollRule::Roll,
        "settle_at_expiry" => RollRule::SettleAtExpiry,
        _ => return Err(unknown("futures rule", &rule, "roll or settle_at_expiry")),
    };
    let contracts: Vec<(u32, Option<PyDay>)> = field(dict, "contracts")?;
    Ok(FuturesSeries {
        contracts: contracts
            .into_iter()
            .map(|(instrument, hand_over)| SeriesContract {
                instrument: InstrumentId(instrument),
                hand_over: hand_over.map(|day| day.0),
            })
            .collect(),
        rule,
    })
}

/// One switch as a dict: series, from, to (instrument indexes), slice, slice_ts (UTC
/// nanoseconds), day, spread_rupees, rolls_positions.
pub(super) fn switch<'py>(py: Python<'py>, s: &Switch) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new_bound(py);
    dict.set_item("series", s.series)?;
    dict.set_item("from", s.from.0)?;
    dict.set_item("to", s.to.0)?;
    dict.set_item("slice", s.slice)?;
    dict.set_item("slice_ts", s.slice_ts)?;
    dict.set_item("day", s.day)?;
    dict.set_item("spread_rupees", s.spread_rupees)?;
    dict.set_item("rolls_positions", s.rolls_positions)?;
    Ok(dict)
}

/// The switches as a list of dicts.
pub(super) fn switches<'py>(py: Python<'py>, all: &[Switch]) -> PyResult<Bound<'py, PyList>> {
    let items = all
        .iter()
        .map(|s| switch(py, s))
        .collect::<PyResult<Vec<_>>>()?;
    Ok(PyList::new_bound(py, items))
}

/// The futures part of a result: `switches` (list of dicts) and `settlements` (one
/// tuple per posting: day, slice_ts, instrument, product, quantity_units,
/// price_rupees, last_close_used, amount_rupees, kind).
pub(super) fn output<'py>(
    py: Python<'py>,
    futures: &FuturesOutput,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new_bound(py);
    dict.set_item("switches", switches(py, &futures.switches)?)?;
    let settlements: Vec<_> = futures
        .settlements
        .iter()
        .map(|r| {
            (
                r.day,
                r.slice_ts,
                r.instrument.0,
                r.product.code(),
                r.quantity_units,
                r.price_rupees,
                r.last_close_used,
                r.amount_rupees,
                r.kind.code(),
            )
        })
        .collect();
    dict.set_item("settlements", settlements)?;
    Ok(dict)
}

/// The IST day (days since 1970-01-01) of the roll session: `sessions_before` sessions
/// before `last_trading_day` on `exchange` (see `oa_bt::engine::roll_session_day`).
#[pyfunction]
fn futures_roll_session_day(
    calendar: PyRef<'_, NativeCalendar>,
    exchange: &str,
    last_trading_day: PyDay,
    sessions_before: u32,
) -> PyResult<i32> {
    let exchange = parse_exchange(exchange)?;
    roll_session_day(
        &calendar.inner,
        exchange,
        last_trading_day.0,
        sessions_before,
    )
    .map_err(engine_error)
}

/// The IST day of the first session of `exchange` after `day`.
#[pyfunction]
fn futures_session_after(
    calendar: PyRef<'_, NativeCalendar>,
    exchange: &str,
    day: PyDay,
) -> PyResult<i32> {
    let exchange = parse_exchange(exchange)?;
    session_after(&calendar.inner, exchange, day.0).map_err(engine_error)
}

/// Register the session functions.
pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(futures_roll_session_day, m)?)?;
    m.add_function(wrap_pyfunction!(futures_session_after, m)?)?;
    Ok(())
}
