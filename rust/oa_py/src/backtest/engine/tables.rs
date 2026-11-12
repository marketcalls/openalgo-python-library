//! What crosses back to Python while the run advances: pauses, events, positions and
//! orders, as dicts with string codes. Order ids cross as integers (Python shows
//! `o<n>`), instruments as their registration index, instants as int64 UTC nanoseconds.

use numpy::IntoPyArray;
use oa_bt::engine::{Event, EventStamp, OrderEventKind, Pause, StrategyBar};
use oa_bt::exec::{GttEvent, Order};
use oa_bt::portfolio::Position;
use oa_bt::rms::{RiskEventName, RiskScope, TradingState, TradingStateCause};
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyString};

use super::columns::{row, ORDER_COLUMNS};

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
        Event::TradingState {
            from, to, cause, ..
        } => trading_state_into(&dict, from, to, cause)?,
        Event::Gtt { event, .. } => gtt_into(&dict, &event)?,
    }
    Ok(dict)
}

/// A GTT event: its name, the GTT's id (an integer, `g<n>` in Python), the instrument,
/// the child's quantity, the triggers, the fired leg, the child's limit and fill price
/// (rupees; `None` where the event does not carry the field).
fn gtt_into(dict: &Bound<'_, PyDict>, event: &GttEvent) -> PyResult<()> {
    dict.set_item("type", "gtt")?;
    dict.set_item("name", event.name.code())?;
    dict.set_item("gtt_id", event.gtt.0)?;
    dict.set_item("instrument", event.instrument.0)?;
    dict.set_item("product", event.product.code())?;
    dict.set_item("quantity_units", event.quantity_units)?;
    dict.set_item("stop_trigger_rupees", event.stop_trigger_rupees)?;
    dict.set_item("target_trigger_rupees", event.target_trigger_rupees)?;
    dict.set_item("order_trigger_rupees", event.order_trigger_rupees)?;
    dict.set_item("fired_leg", event.fired_leg.map(|leg| leg.code()))?;
    dict.set_item("child_limit_rupees", event.child_limit_rupees)?;
    dict.set_item("price_rupees", event.price_rupees)
}

/// A trading-state change as a risk event named `trading_state_changed` (scope
/// strategy), with the state before and after it and its cause.
fn trading_state_into(
    dict: &Bound<'_, PyDict>,
    from: TradingState,
    to: TradingState,
    cause: TradingStateCause,
) -> PyResult<()> {
    dict.set_item("type", "risk")?;
    dict.set_item("name", RiskEventName::TradingStateChanged.code())?;
    dict.set_item("scope", RiskScope::Strategy.code())?;
    dict.set_item("instrument", None::<u32>)?;
    dict.set_item("level_rupees", None::<f64>)?;
    dict.set_item("target_index", None::<u16>)?;
    dict.set_item("observed_rupees", None::<f64>)?;
    dict.set_item("previous_state", from.code())?;
    dict.set_item("trading_state", to.code())?;
    dict.set_item("cause", cause.code())
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

/// What `pause` returns: (kind, decision instant, strategy bars, events, timer ids).
pub(super) type PauseTuple<'py> = (
    Bound<'py, PyString>,
    Option<i64>,
    Bound<'py, PyList>,
    Bound<'py, PyList>,
    Bound<'py, PyList>,
);

/// A pause as a tuple (kind, decided_at, bars, events, timers): `kind` "bar_close",
/// "timers" or "finished" (interned, so the driver compares it by identity first), the
/// decision instant (UTC nanoseconds, None when finished), the strategy bars as tuples
/// (instrument, ts_open, ts_close, open, high, low, close, volume), the events as dicts
/// and the timer ids (timer pauses). A tuple, not a dict: the event driver unpacks one
/// per strategy-bar close, and its cost is part of every callback (NFR-PERF-04).
pub(super) fn pause<'py>(py: Python<'py>, pause: &Pause) -> PyResult<PauseTuple<'py>> {
    let events = events(py, pause.events())?;
    let none = || PyList::empty_bound(py);
    Ok(match pause {
        Pause::BarClose {
            decided_at, bars, ..
        } => {
            let tuples = bars
                .iter()
                .map(|bar| -> PyObject { bar_tuple(bar).into_py(py) });
            let kind = intern!(py, "bar_close").clone();
            (
                kind,
                Some(*decided_at),
                PyList::new_bound(py, tuples),
                events,
                none(),
            )
        }
        Pause::Timers {
            decided_at, timers, ..
        } => {
            let ids = timers.iter().map(|timer| timer.0);
            let kind = intern!(py, "timers").clone();
            (
                kind,
                Some(*decided_at),
                none(),
                events,
                PyList::new_bound(py, ids),
            )
        }
        Pause::Finished { .. } => (
            intern!(py, "finished").clone(),
            None,
            none(),
            events,
            none(),
        ),
    })
}

/// Strategy bars as a dict of arrays, one per column.
pub(super) fn strategy_bars<'py>(
    py: Python<'py>,
    bars: &[StrategyBar],
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new_bound(py);
    let times = |get: fn(&StrategyBar) -> i64| bars.iter().map(get).collect::<Vec<i64>>();
    let values = |get: fn(&StrategyBar) -> f64| bars.iter().map(get).collect::<Vec<f64>>();
    dict.set_item("ts_open", times(|b| b.ts_open).into_pyarray_bound(py))?;
    dict.set_item("ts_close", times(|b| b.ts_close).into_pyarray_bound(py))?;
    dict.set_item("open", values(|b| b.open).into_pyarray_bound(py))?;
    dict.set_item("high", values(|b| b.high).into_pyarray_bound(py))?;
    dict.set_item("low", values(|b| b.low).into_pyarray_bound(py))?;
    dict.set_item("close", values(|b| b.close).into_pyarray_bound(py))?;
    dict.set_item("volume", values(|b| b.volume).into_pyarray_bound(py))?;
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

/// An order as a dict (the keys of the orders table, `columns::ORDER_COLUMNS`).
pub(super) fn order<'py>(py: Python<'py>, o: &Order) -> PyResult<Bound<'py, PyDict>> {
    row(py, o, &ORDER_COLUMNS)
}
