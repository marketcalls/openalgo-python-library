//! The row tables of a run (fills, levels, trades, orders) defined once, as lists of
//! columns: each column is a key and the value a row gives for it. `table` builds a
//! whole table column by column, one Python list per key, and `row` builds one row as a
//! dict (`Engine.order`) from the same list, so the keys, their order and their values
//! have one definition. Building columns directly, rather than a dict per row that is
//! then read back key by key, keeps the cost of a result linear and small in its rows.

use oa_bt::engine::{FillRecord, LevelRow, TradeRecord};
use oa_bt::exec::{FillFlags, Order, QuantitySpec, Validity};
use oa_bt::model::Side;
use oa_bt::rms::leg::TargetState;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

/// One column of a table: its key and the value of a row.
pub(super) type Column<R> = (&'static str, fn(Python<'_>, &R) -> PyObject);

/// Rows as a dict of columns: one list per key, in the order of `columns`.
pub(super) fn table<'py, R>(
    py: Python<'py>,
    rows: &[R],
    columns: &[Column<R>],
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new_bound(py);
    for (key, value) in columns {
        let column = PyList::new_bound(py, rows.iter().map(|row| value(py, row)));
        dict.set_item(*key, column)?;
    }
    Ok(dict)
}

/// One row as a dict with the keys of `columns`.
pub(super) fn row<'py, R>(
    py: Python<'py>,
    row: &R,
    columns: &[Column<R>],
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new_bound(py);
    for (key, value) in columns {
        dict.set_item(*key, value(py, row))?;
    }
    Ok(dict)
}

/// The fills: the stamp of the fill, what filled, where and why, its order or GTT, its
/// flags and its charges (rupees).
pub(super) const FILL_COLUMNS: [Column<FillRecord>; 16] = [
    ("slice", |py, r| r.stamp.slice.into_py(py)),
    ("slice_ts", |py, r| r.stamp.slice_ts.into_py(py)),
    ("step", |py, r| r.stamp.step.number().into_py(py)),
    ("instrument", |py, r| r.fill.instrument.0.into_py(py)),
    ("product", |py, r| r.product.code().into_py(py)),
    ("side", |py, r| r.fill.side.code().into_py(py)),
    ("quantity_units", |py, r| r.fill.quantity_units.into_py(py)),
    ("price_rupees", |py, r| r.fill.price_rupees.into_py(py)),
    ("fill_at", |py, r| r.fill.at.code().into_py(py)),
    ("reason", |py, r| r.fill.reason.code().into_py(py)),
    ("order_id", |py, r| r.fill.order.map(|id| id.0).into_py(py)),
    ("gtt_id", |py, r| r.fill.gtt.map(|id| id.0).into_py(py)),
    ("target_index", |py, r| r.fill.target_index.into_py(py)),
    ("flags", |py, r| r.fill.flags.names().into_py(py)),
    ("costs_rupees", |py, r| r.fill.costs_rupees.into_py(py)),
    ("filled_at_last_close", |py, r| {
        let flags = r.fill.flags;
        flags.contains(FillFlags::FILLED_AT_LAST_CLOSE).into_py(py)
    }),
];

/// The targets of a level row as (index, level rupees, units, filled).
fn targets(py: Python<'_>, row: &LevelRow) -> PyObject {
    let targets: Vec<(u16, f64, f64, bool)> = row
        .targets
        .iter()
        .map(|t| {
            let filled = t.state == TargetState::Filled;
            (t.index, t.level_rupees, t.quantity_units, filled)
        })
        .collect();
    targets.into_py(py)
}

/// The position history: the position and its levels at the end of a slice where they
/// changed (prices rupees, quantities units).
pub(super) const LEVEL_COLUMNS: [Column<LevelRow>; 16] = [
    ("slice", |py, r| r.slice.into_py(py)),
    ("slice_ts", |py, r| r.slice_ts.into_py(py)),
    ("instrument", |py, r| r.instrument.0.into_py(py)),
    ("product", |py, r| r.product.code().into_py(py)),
    ("quantity_units", |py, r| r.quantity_units.into_py(py)),
    ("average_price_rupees", |py, r| {
        r.average_price_rupees.into_py(py)
    }),
    ("stop_rupees", |py, r| {
        r.stop.map(|s| s.level_rupees).into_py(py)
    }),
    ("stop_units", |py, r| {
        r.stop.map(|s| s.quantity_units).into_py(py)
    }),
    ("targets", targets),
    ("remainder_units", |py, r| r.remainder_units.into_py(py)),
    ("trail_armed", |py, r| r.trail.map(|t| t.armed).into_py(py)),
    ("trail_extreme_rupees", |py, r| {
        r.trail.map(|t| t.favourable_extreme_rupees).into_py(py)
    }),
    ("breakeven_armed", |py, r| {
        r.breakeven.map(|b| b.armed).into_py(py)
    }),
    ("breakeven_trigger_rupees", |py, r| {
        r.breakeven.map(|b| b.trigger_price_rupees).into_py(py)
    }),
    ("breakeven_level_rupees", |py, r| {
        r.breakeven.map(|b| b.level_rupees).into_py(py)
    }),
    ("realized_rupees", |py, r| r.realized_rupees.into_py(py)),
];

/// The closed round trips: their number, direction, entry and exit, P&L and charges
/// (rupees), and the initial risk in rupees (None with no stop at the open).
pub(super) const TRADE_COLUMNS: [Column<TradeRecord>; 18] = [
    ("trade_id", |py, t| t.number.into_py(py)),
    ("instrument", |py, t| t.instrument.0.into_py(py)),
    ("product", |py, t| t.product.code().into_py(py)),
    ("direction", |py, t| {
        let long = t.direction == Side::Buy;
        if long { "long" } else { "short" }.into_py(py)
    }),
    ("entry_slice", |py, t| t.entry_slice.into_py(py)),
    ("entry_ts", |py, t| t.entry_ts.into_py(py)),
    ("exit_slice", |py, t| t.exit_slice.into_py(py)),
    ("exit_ts", |py, t| t.exit_ts.into_py(py)),
    ("entry_units", |py, t| t.entry_units.into_py(py)),
    ("max_units", |py, t| t.max_units.into_py(py)),
    ("entry_price_rupees", |py, t| {
        t.entry_price_rupees.into_py(py)
    }),
    ("exit_price_rupees", |py, t| t.exit_price_rupees.into_py(py)),
    ("gross_pnl_rupees", |py, t| t.gross_pnl_rupees.into_py(py)),
    ("costs_rupees", |py, t| t.costs_rupees.into_py(py)),
    ("net_pnl_rupees", |py, t| t.net_pnl_rupees.into_py(py)),
    ("exit_reason", |py, t| t.exit_reason.code().into_py(py)),
    ("fills", |py, t| t.fills.into_py(py)),
    ("initial_risk_rupees", |py, t| {
        t.initial_risk_rupees.into_py(py)
    }),
];

/// The orders: the request (type, prices in rupees, validity, product), the state and
/// what filled; `sizer` names the sizer that set the quantity (None for a quantity
/// given in units, lots or a fraction of the position).
pub(super) const ORDER_COLUMNS: [Column<Order>; 18] = [
    ("order_id", |py, o| o.id.0.into_py(py)),
    ("instrument", |py, o| o.request.instrument.0.into_py(py)),
    ("side", |py, o| o.request.side.code().into_py(py)),
    ("order_type", |py, o| {
        o.request.order_type.code().into_py(py)
    }),
    ("limit_rupees", |py, o| {
        o.request.order_type.limit_rupees().into_py(py)
    }),
    ("trigger_rupees", |py, o| {
        o.request.order_type.trigger_rupees().into_py(py)
    }),
    ("validity", |py, o| {
        match o.request.validity {
            Validity::Day => "DAY",
            Validity::Gtt => "GTT",
        }
        .into_py(py)
    }),
    ("product", |py, o| o.request.product.code().into_py(py)),
    ("status", |py, o| o.status.code().into_py(py)),
    ("decided_at", |py, o| o.decided_at.into_py(py)),
    ("good_until", |py, o| o.good_until.into_py(py)),
    ("quantity_units", |py, o| o.quantity_units.into_py(py)),
    ("filled_units", |py, o| o.filled_units.into_py(py)),
    ("average_fill_rupees", |py, o| {
        o.average_fill_rupees.into_py(py)
    }),
    ("reject_code", |py, o| {
        o.reject_code.map(|c| c.code()).into_py(py)
    }),
    ("reduce_only", |py, o| o.request.reduce_only.into_py(py)),
    ("tag", |py, o| o.request.tag.clone().into_py(py)),
    ("sizer", |py, o| {
        match o.request.quantity {
            QuantitySpec::Sizer(sizer) => Some(sizer.code()),
            _ => None,
        }
        .into_py(py)
    }),
];
