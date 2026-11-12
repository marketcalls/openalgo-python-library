//! What crosses back to Python while the run advances: pauses, events, positions and
//! orders, as dicts with string codes. Order ids cross as integers (Python shows
//! `o<n>`), instruments as their registration index, instants as int64 UTC nanoseconds.

use oa_bt::engine::{Event, EventStamp, OrderEventKind, Pause, StrategyBar};
use oa_bt::exec::Order;
use oa_bt::portfolio::Position;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

pub(super) fn stamp_into(dict: &Bound<'_, PyDict>, stamp: EventStamp) -> PyResult<()> {
    dict.set_item("slice", stamp.slice)?;
    dict.set_item("slice_ts", stamp.slice_ts)?;
    dict.set_item("step", stamp.step.number())
}

/// One event as a dict with the fixture's keys plus `slice`, `slice_ts` and `step`.
pub(super) fn event<'py>(py: Python<'py>, event: &Event) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new_bound(py);
    stamp_into(&dict, event.stamp())?;
    match *event {
        Event::Order { order, kind, .. } => {
            let (quantity, price) = match kind {
                OrderEventKind::Filled {
                    quantity_units,
                    price_rupees,
                } => (Some(quantity_units), Some(price_rupees)),
                _ => (None, None),
            };
            let code = match kind {
                OrderEventKind::Rejected(code) => Some(code.code()),
                _ => None,
            };
            dict.set_item("type", "order")?;
            dict.set_item("name", kind.code())?;
            dict.set_item("order_id", order.0)?;
            dict.set_item("quantity_units", quantity)?;
            dict.set_item("price_rupees", price)?;
            dict.set_item("reject_code", code)?;
        }
        Event::Position {
            instrument,
            product,
            change,
            quantity_units,
            ..
        } => {
            dict.set_item("type", "position")?;
            dict.set_item("name", change.code())?;
            dict.set_item("instrument", instrument.0)?;
            dict.set_item("product", product.code())?;
            dict.set_item("position_quantity_units", quantity_units)?;
        }
        Event::Risk {
            instrument, event, ..
        } => {
            dict.set_item("type", "risk")?;
            dict.set_item("name", event.name.code())?;
            dict.set_item("scope", event.scope.code())?;
            dict.set_item("instrument", instrument.map(|i| i.0))?;
            dict.set_item("level_rupees", event.level_rupees)?;
            dict.set_item("target_index", event.target_index)?;
            dict.set_item("observed_rupees", event.observed_rupees)?;
        }
    }
    Ok(dict)
}

pub(super) fn events<'py>(py: Python<'py>, list: &[Event]) -> PyResult<Bound<'py, PyList>> {
    let items = list
        .iter()
        .map(|e| event(py, e))
        .collect::<PyResult<Vec<_>>>()?;
    Ok(PyList::new_bound(py, items))
}

fn bar_tuple(bar: &StrategyBar) -> (u32, i64, i64, f64, f64, f64, f64, f64) {
    (
        bar.instrument.0,
        bar.ts_open,
        bar.ts_close,
        bar.open,
        bar.high,
        bar.low,
        bar.close,
        bar.volume,
    )
}

/// A pause as a dict: `kind` ("bar_close", "timers" or "finished"), its stamp, the
/// decision instant, the strategy bars and the events.
pub(super) fn pause<'py>(py: Python<'py>, pause: &Pause) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new_bound(py);
    dict.set_item("events", events(py, pause.events())?)?;
    match pause {
        Pause::BarClose {
            stamp,
            decided_at,
            strategy_bar_closed,
            bars,
            ..
        } => {
            dict.set_item("kind", "bar_close")?;
            stamp_into(&dict, *stamp)?;
            dict.set_item("decided_at", decided_at)?;
            dict.set_item("strategy_bar_closed", strategy_bar_closed)?;
            let tuples: Vec<_> = bars.iter().map(bar_tuple).collect();
            dict.set_item("bars", tuples)?;
        }
        Pause::Timers {
            stamp, decided_at, ..
        } => {
            dict.set_item("kind", "timers")?;
            stamp_into(&dict, *stamp)?;
            dict.set_item("decided_at", decided_at)?;
        }
        Pause::Finished { .. } => dict.set_item("kind", "finished")?,
    }
    Ok(dict)
}

/// A position as a dict.
pub(super) fn position<'py>(py: Python<'py>, p: &Position) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new_bound(py);
    dict.set_item("quantity_units", p.quantity_units)?;
    let average = (!p.is_flat()).then_some(p.average_price_rupees);
    dict.set_item("average_price_rupees", average)?;
    dict.set_item("adds", p.adds)?;
    dict.set_item("realized_rupees", p.realized_rupees)?;
    dict.set_item("unrealized_rupees", p.unrealized_rupees())?;
    dict.set_item("mark_rupees", p.mark_rupees)?;
    let tranches: Vec<(u32, f64, f64, i64)> = p
        .tranches
        .iter()
        .map(|t| (t.number, t.open_units, t.price_rupees, t.opened_at))
        .collect();
    dict.set_item("tranches", tranches)?;
    Ok(dict)
}

/// An order as a dict.
pub(super) fn order<'py>(py: Python<'py>, o: &Order) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new_bound(py);
    dict.set_item("order_id", o.id.0)?;
    dict.set_item("instrument", o.request.instrument.0)?;
    dict.set_item("side", o.request.side.code())?;
    dict.set_item("order_type", o.request.order_type.code())?;
    dict.set_item("product", o.request.product.code())?;
    dict.set_item("status", o.status.code())?;
    dict.set_item("decided_at", o.decided_at)?;
    dict.set_item("good_until", o.good_until)?;
    dict.set_item("quantity_units", o.quantity_units)?;
    dict.set_item("filled_units", o.filled_units)?;
    dict.set_item("average_fill_rupees", o.average_fill_rupees)?;
    dict.set_item("reject_code", o.reject_code.map(|c| c.code()))?;
    dict.set_item("reduce_only", o.request.reduce_only)?;
    dict.set_item("tag", o.request.tag.clone())?;
    Ok(dict)
}
