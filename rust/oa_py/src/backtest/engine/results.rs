//! The result tables of a finished run, as Python reads them: fills, levels, trades and
//! orders as dicts of lists with string codes, the equity series as NumPy arrays, plus
//! the events, stale marks, final positions and gross P&L.

use numpy::IntoPyArray;
use oa_bt::engine::{RunEndCause, RunOutput};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use super::charges::charges;
use super::columns::{table, FILL_COLUMNS, LEVEL_COLUMNS, ORDER_COLUMNS, TRADE_COLUMNS};
use super::tables::events;

fn equity<'py>(py: Python<'py>, output: &RunOutput) -> PyResult<Bound<'py, PyDict>> {
    let e = &output.equity;
    let dict = PyDict::new_bound(py);
    dict.set_item("ts", e.ts.clone().into_pyarray_bound(py))?;
    dict.set_item("day", e.day.clone().into_pyarray_bound(py))?;
    let values = [
        ("equity_rupees", &e.equity_rupees),
        ("realized_rupees", &e.realized_rupees),
        ("unrealized_rupees", &e.unrealized_rupees),
        ("costs_rupees", &e.costs_rupees),
        ("margin_rupees", &e.margin_rupees),
        ("exposure_rupees", &e.exposure_rupees),
    ];
    for (key, column) in values {
        dict.set_item(key, column.clone().into_pyarray_bound(py))?;
    }
    Ok(dict)
}

fn finals<'py>(py: Python<'py>, output: &RunOutput) -> PyResult<Bound<'py, PyList>> {
    let mut items = Vec::new();
    for p in &output.final_positions {
        let dict = PyDict::new_bound(py);
        dict.set_item("instrument", p.instrument.0)?;
        dict.set_item("product", p.product.code())?;
        dict.set_item("quantity_units", p.quantity_units)?;
        dict.set_item("average_price_rupees", p.average_price_rupees)?;
        dict.set_item("mark_rupees", p.mark_rupees)?;
        dict.set_item("unrealized_rupees", p.unrealized_rupees)?;
        dict.set_item("stale", p.stale)?;
        items.push(dict);
    }
    Ok(PyList::new_bound(py, items))
}

/// The whole result as a dict of tables.
pub(super) fn output<'py>(py: Python<'py>, output: &RunOutput) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new_bound(py);
    dict.set_item("capital", output.capital_rupees)?;
    dict.set_item("fills", table(py, &output.fills, &FILL_COLUMNS)?)?;
    dict.set_item("levels", table(py, &output.levels, &LEVEL_COLUMNS)?)?;
    dict.set_item("trades", table(py, &output.trades, &TRADE_COLUMNS)?)?;
    dict.set_item("orders", table(py, &output.orders, &ORDER_COLUMNS)?)?;
    dict.set_item("equity", equity(py, output)?)?;
    dict.set_item("events", events(py, &output.events)?)?;
    let stale: Vec<(u32, i64, u32, f64)> = output
        .stale_marks
        .iter()
        .map(|m| (m.slice, m.slice_ts, m.instrument.0, m.mark_rupees))
        .collect();
    dict.set_item("stale_marks", stale)?;
    dict.set_item("charges", charges(py, &output.charges)?)?;
    dict.set_item("futures", super::futures::output(py, &output.futures)?)?;
    dict.set_item("final_positions", finals(py, output)?)?;
    let g = output.gross_pnl;
    dict.set_item(
        "gross_pnl",
        (g.realized_rupees, g.unrealized_rupees, g.total_rupees),
    )?;
    let end = output.run_end;
    let cause = end.cause.map(RunEndCause::code);
    dict.set_item("run_end", (end.close, end.range_close, cause))?;
    Ok(dict)
}
