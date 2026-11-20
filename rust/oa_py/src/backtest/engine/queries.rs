//! The queries of a run across the boundary (the parent module's `position`, `order`,
//! `working_orders`, `entry_locks` and `strategy_bars`): what the kernel answers, as the
//! dicts and lists of `tables`.

use oa_bt::engine::Engine;
use oa_bt::exec::{InstrumentId, OrderId};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use super::setup::parse_product;
use super::tables;

/// `position`: the position of an instrument and product, None if never opened.
pub(super) fn position<'py>(
    py: Python<'py>,
    engine: &Engine,
    instrument: u32,
    product: &str,
) -> PyResult<Option<Bound<'py, PyDict>>> {
    let product = parse_product(product)?;
    let position = engine.position(InstrumentId(instrument), product);
    position.map(|p| tables::position(py, p)).transpose()
}

/// `order`: an order as a dict, None for an unknown id.
pub(super) fn order<'py>(
    py: Python<'py>,
    engine: &Engine,
    order_id: u64,
) -> PyResult<Option<Bound<'py, PyDict>>> {
    let order = engine.order(OrderId(order_id));
    order.map(|o| tables::order(py, o)).transpose()
}

/// `working_orders`: the ids of the orders that can still fill.
pub(super) fn working_orders(engine: &Engine) -> Vec<u64> {
    engine.working_orders().iter().map(|id| id.0).collect()
}

/// `entry_locks`: the reasons a live session's entries are locked for, sorted.
pub(super) fn entry_locks(engine: &Engine) -> Vec<String> {
    let locks = engine.entry_locks();
    locks.iter().map(|reason| reason.to_string()).collect()
}

/// `strategy_bars`: every strategy bar of an instrument over the run, as arrays.
pub(super) fn strategy_bars<'py>(
    py: Python<'py>,
    engine: &Engine,
    instrument: u32,
) -> PyResult<Bound<'py, PyDict>> {
    tables::strategy_bars(py, &engine.strategy_bars(InstrumentId(instrument)))
}
