//! The result tables of a finished run, as Python reads them: fills, levels, trades and
//! orders as dicts of lists with string codes, the equity series as NumPy arrays, plus
//! the events, stale marks, final positions and gross P&L.

use numpy::IntoPyArray;
use oa_bt::engine::{FillRecord, LevelRow, RunOutput, TradeRecord};
use oa_bt::exec::FillFlags;
use oa_bt::model::Side;
use oa_bt::rms::leg::TargetState;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use super::charges::charges;
use super::tables::{events, order, stamp_into};

/// Rows as a dict of columns: every row gives the same keys in the same order.
fn columns<'py>(
    py: Python<'py>,
    rows: Vec<Bound<'py, PyDict>>,
    keys: &[&str],
) -> PyResult<Bound<'py, PyDict>> {
    let table = PyDict::new_bound(py);
    for key in keys {
        let mut column = Vec::with_capacity(rows.len());
        for row in &rows {
            column.push(row.get_item(key)?.map_or_else(|| py.None(), |v| v.unbind()));
        }
        table.set_item(*key, column)?;
    }
    Ok(table)
}

const FILL_KEYS: [&str; 15] = [
    "slice",
    "slice_ts",
    "step",
    "instrument",
    "product",
    "side",
    "quantity_units",
    "price_rupees",
    "fill_at",
    "reason",
    "order_id",
    "target_index",
    "flags",
    "costs_rupees",
    "filled_at_last_close",
];

fn fill<'py>(py: Python<'py>, record: &FillRecord) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new_bound(py);
    stamp_into(&dict, record.stamp)?;
    let f = record.fill;
    dict.set_item("instrument", f.instrument.0)?;
    dict.set_item("product", record.product.code())?;
    dict.set_item("side", f.side.code())?;
    dict.set_item("quantity_units", f.quantity_units)?;
    dict.set_item("price_rupees", f.price_rupees)?;
    dict.set_item("fill_at", f.at.code())?;
    dict.set_item("reason", f.reason.code())?;
    dict.set_item("order_id", f.order.map(|id| id.0))?;
    dict.set_item("target_index", f.target_index)?;
    dict.set_item("flags", f.flags.names())?;
    dict.set_item("costs_rupees", f.costs_rupees)?;
    let last_close = f.flags.contains(FillFlags::FILLED_AT_LAST_CLOSE);
    dict.set_item("filled_at_last_close", last_close)?;
    Ok(dict)
}

const LEVEL_KEYS: [&str; 16] = [
    "slice",
    "slice_ts",
    "instrument",
    "product",
    "quantity_units",
    "average_price_rupees",
    "stop_rupees",
    "stop_units",
    "targets",
    "remainder_units",
    "trail_armed",
    "trail_extreme_rupees",
    "breakeven_armed",
    "breakeven_trigger_rupees",
    "breakeven_level_rupees",
    "realized_rupees",
];

fn level<'py>(py: Python<'py>, row: &LevelRow) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new_bound(py);
    dict.set_item("slice", row.slice)?;
    dict.set_item("slice_ts", row.slice_ts)?;
    dict.set_item("instrument", row.instrument.0)?;
    dict.set_item("product", row.product.code())?;
    dict.set_item("quantity_units", row.quantity_units)?;
    dict.set_item("average_price_rupees", row.average_price_rupees)?;
    dict.set_item("stop_rupees", row.stop.map(|s| s.level_rupees))?;
    dict.set_item("stop_units", row.stop.map(|s| s.quantity_units))?;
    let targets: Vec<(u16, f64, f64, bool)> = row
        .targets
        .iter()
        .map(|t| {
            (
                t.index,
                t.level_rupees,
                t.quantity_units,
                t.state == TargetState::Filled,
            )
        })
        .collect();
    dict.set_item("targets", targets)?;
    dict.set_item("remainder_units", row.remainder_units)?;
    dict.set_item("trail_armed", row.trail.map(|t| t.armed))?;
    dict.set_item(
        "trail_extreme_rupees",
        row.trail.map(|t| t.favourable_extreme_rupees),
    )?;
    dict.set_item("breakeven_armed", row.breakeven.map(|b| b.armed))?;
    dict.set_item(
        "breakeven_trigger_rupees",
        row.breakeven.map(|b| b.trigger_price_rupees),
    )?;
    dict.set_item(
        "breakeven_level_rupees",
        row.breakeven.map(|b| b.level_rupees),
    )?;
    dict.set_item("realized_rupees", row.realized_rupees)?;
    Ok(dict)
}

const TRADE_KEYS: [&str; 17] = [
    "trade_id",
    "instrument",
    "product",
    "direction",
    "entry_slice",
    "entry_ts",
    "exit_slice",
    "exit_ts",
    "entry_units",
    "max_units",
    "entry_price_rupees",
    "exit_price_rupees",
    "gross_pnl_rupees",
    "costs_rupees",
    "net_pnl_rupees",
    "exit_reason",
    "fills",
];

fn trade<'py>(py: Python<'py>, t: &TradeRecord) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new_bound(py);
    dict.set_item("trade_id", t.number)?;
    dict.set_item("instrument", t.instrument.0)?;
    dict.set_item("product", t.product.code())?;
    let direction = if t.direction == Side::Buy {
        "long"
    } else {
        "short"
    };
    dict.set_item("direction", direction)?;
    dict.set_item("entry_slice", t.entry_slice)?;
    dict.set_item("entry_ts", t.entry_ts)?;
    dict.set_item("exit_slice", t.exit_slice)?;
    dict.set_item("exit_ts", t.exit_ts)?;
    dict.set_item("entry_units", t.entry_units)?;
    dict.set_item("max_units", t.max_units)?;
    dict.set_item("entry_price_rupees", t.entry_price_rupees)?;
    dict.set_item("exit_price_rupees", t.exit_price_rupees)?;
    dict.set_item("gross_pnl_rupees", t.gross_pnl_rupees)?;
    dict.set_item("costs_rupees", t.costs_rupees)?;
    dict.set_item("net_pnl_rupees", t.net_pnl_rupees)?;
    dict.set_item("exit_reason", t.exit_reason.code())?;
    dict.set_item("fills", t.fills)?;
    Ok(dict)
}

const ORDER_KEYS: [&str; 14] = [
    "order_id",
    "instrument",
    "side",
    "order_type",
    "product",
    "status",
    "decided_at",
    "good_until",
    "quantity_units",
    "filled_units",
    "average_fill_rupees",
    "reject_code",
    "reduce_only",
    "tag",
];

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
    let fills = output.fills.iter().map(|r| fill(py, r));
    let levels = output.levels.iter().map(|r| level(py, r));
    let trades = output.trades.iter().map(|r| trade(py, r));
    let orders = output.orders.iter().map(|r| order(py, r));
    let fills = fills.collect::<PyResult<Vec<_>>>()?;
    let levels = levels.collect::<PyResult<Vec<_>>>()?;
    let trades = trades.collect::<PyResult<Vec<_>>>()?;
    let orders = orders.collect::<PyResult<Vec<_>>>()?;
    dict.set_item("capital", output.capital_rupees)?;
    dict.set_item("fills", columns(py, fills, &FILL_KEYS)?)?;
    dict.set_item("levels", columns(py, levels, &LEVEL_KEYS)?)?;
    dict.set_item("trades", columns(py, trades, &TRADE_KEYS)?)?;
    dict.set_item("orders", columns(py, orders, &ORDER_KEYS)?)?;
    dict.set_item("equity", equity(py, output)?)?;
    dict.set_item("events", events(py, &output.events)?)?;
    let stale: Vec<(u32, i64, u32, f64)> = output
        .stale_marks
        .iter()
        .map(|m| (m.slice, m.slice_ts, m.instrument.0, m.mark_rupees))
        .collect();
    dict.set_item("stale_marks", stale)?;
    dict.set_item("charges", charges(py, &output.charges)?)?;
    dict.set_item("final_positions", finals(py, output)?)?;
    let g = output.gross_pnl;
    dict.set_item(
        "gross_pnl",
        (g.realized_rupees, g.unrealized_rupees, g.total_rupees),
    )?;
    Ok(dict)
}
