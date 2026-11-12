//! The charges of a finished run as Python reads them (`openalgo/backtest/_result_costs.py`):
//!
//! - `rounding`: "contract_note", "calculator", or None for a cost-free run;
//! - `fills`: columns, one row per charged fill: `fill` (row of the fills table), `day`,
//!   `segment`, the six components, `cost`, `dp_charge`, `total`, `schedule` (position of
//!   the schedule version in the run's cost model), `trust` and `trade`;
//! - `notes`: columns, one row per contract note: `day`, `group`, `fills`, the six
//!   charged lines, `charged_total`, `accrued_total`, `posted`, `adjustment`, `settled`;
//! - `tables`: one (table, row id, status, first day, source, notes) per dated
//!   market-table row the run used.
//!
//! Money is rupees; days are IST days since 1970-01-01.

use oa_bt::charges::{ChargesOutput, FillCharge, SettledNote};
use oa_bt::market::costs::COMPONENT_NAMES;
use pyo3::prelude::*;
use pyo3::types::PyDict;

type TableTuple = (&'static str, String, &'static str, i32, String, String);

fn put<'py, T: ToPyObject>(dict: &Bound<'py, PyDict>, key: &str, column: Vec<T>) -> PyResult<()> {
    dict.set_item(key, column)
}

fn components<'py, T>(
    dict: &Bound<'py, PyDict>,
    rows: &[T],
    pick: fn(&T) -> [f64; 6],
) -> PyResult<()> {
    for (index, name) in COMPONENT_NAMES.iter().enumerate() {
        put(
            dict,
            name,
            rows.iter().map(|row| pick(row)[index]).collect(),
        )?;
    }
    Ok(())
}

fn fills<'py>(py: Python<'py>, rows: &[FillCharge]) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new_bound(py);
    put(&dict, "fill", rows.iter().map(|r| r.fill).collect())?;
    put(&dict, "day", rows.iter().map(|r| r.day).collect())?;
    put(
        &dict,
        "segment",
        rows.iter().map(|r| r.segment.code()).collect(),
    )?;
    components(&dict, rows, |r| r.cost.components())?;
    put(&dict, "cost", rows.iter().map(|r| r.cost_rupees).collect())?;
    let dp = rows
        .iter()
        .map(|r| r.dp_charge.map_or(0.0, |dp| dp.total()));
    put(&dict, "dp_charge", dp.collect())?;
    put(
        &dict,
        "total",
        rows.iter().map(FillCharge::total_rupees).collect(),
    )?;
    put(
        &dict,
        "schedule",
        rows.iter().map(|r| r.status.schedule).collect(),
    )?;
    put(
        &dict,
        "trust",
        rows.iter().map(|r| r.status.trust.code()).collect(),
    )?;
    put(&dict, "trade", rows.iter().map(|r| r.trade).collect())?;
    Ok(dict)
}

fn notes<'py>(py: Python<'py>, rows: &[SettledNote]) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new_bound(py);
    put(&dict, "day", rows.iter().map(|r| r.note.day).collect())?;
    put(
        &dict,
        "group",
        rows.iter().map(|r| r.note.group.code()).collect(),
    )?;
    put(&dict, "fills", rows.iter().map(|r| r.note.fills).collect())?;
    components(&dict, rows, |r| r.note.charged.components())?;
    put(
        &dict,
        "charged_total",
        rows.iter().map(|r| r.note.charged_total).collect(),
    )?;
    let accrued = rows.iter().map(|r| r.note.accrued.total());
    put(&dict, "accrued_total", accrued.collect())?;
    put(
        &dict,
        "posted",
        rows.iter().map(|r| r.posted_rupees).collect(),
    )?;
    put(
        &dict,
        "adjustment",
        rows.iter().map(|r| r.adjustment_rupees).collect(),
    )?;
    put(&dict, "settled", rows.iter().map(|r| r.settled).collect())?;
    Ok(dict)
}

/// The run's charges as a dict (see the module documentation).
pub(super) fn charges<'py>(
    py: Python<'py>,
    output: &ChargesOutput,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new_bound(py);
    dict.set_item("rounding", output.rounding.map(|r| r.code()))?;
    dict.set_item("fills", fills(py, &output.fills)?)?;
    dict.set_item("notes", notes(py, &output.notes)?)?;
    let tables: Vec<TableTuple> = output
        .tables
        .iter()
        .map(|used| {
            let row = used.row;
            let status = row.status.code();
            let (source, notes) = (row.source.clone(), row.notes.clone());
            (
                used.table.code(),
                row.id.clone(),
                status,
                used.first_day,
                source,
                notes,
            )
        })
        .collect();
    dict.set_item("tables", tables)?;
    Ok(dict)
}
