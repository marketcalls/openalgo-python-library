//! The order calls of a decision point across the boundary: `submit`, `target`,
//! `rebalance_target`, `modify`, `cancel` and `sizing_inputs` (the parent module's
//! methods of the same names). Each reads its request dict (`request`, `sizing`), asks the
//! kernel and returns what crosses back: a submission tuple (`request::submission_tuple`),
//! an order's state code or a sizer's inputs.

use oa_bt::engine::Engine;
use oa_bt::exec::OrderId;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use super::errors::engine_error;
use super::request::{order_request, submission_tuple, SubmissionTuple};
use super::sizing;

/// `submit`: place an order at the current decision point.
pub(super) fn submit(
    engine: &mut Engine,
    request: &Bound<'_, PyDict>,
) -> PyResult<SubmissionTuple> {
    let request = order_request(request)?;
    let outcome = engine.submit(request).map_err(engine_error)?;
    Ok(submission_tuple(outcome))
}

/// `target`: the order for the difference to a target position, None when there is none.
pub(super) fn target(
    engine: &mut Engine,
    request: &Bound<'_, PyDict>,
) -> PyResult<Option<SubmissionTuple>> {
    let request = sizing::target_request(request)?;
    let outcome = engine.submit_target(request).map_err(engine_error)?;
    Ok(outcome.map(submission_tuple))
}

/// `rebalance_target`: the same order as `target`, as part of a rebalance.
pub(super) fn rebalance_target(
    engine: &mut Engine,
    request: &Bound<'_, PyDict>,
) -> PyResult<Option<SubmissionTuple>> {
    let request = sizing::target_request(request)?;
    let outcome = engine.submit_rebalance(request).map_err(engine_error)?;
    Ok(outcome.map(submission_tuple))
}

/// `modify`: a full replacement of a working order's request.
pub(super) fn modify(
    engine: &mut Engine,
    order_id: u64,
    request: &Bound<'_, PyDict>,
) -> PyResult<SubmissionTuple> {
    let request = order_request(request)?;
    let outcome = engine.modify(OrderId(order_id), request);
    Ok(submission_tuple(outcome.map_err(engine_error)?))
}

/// `cancel`: the order's state code after the cancellation.
pub(super) fn cancel(engine: &mut Engine, order_id: u64) -> PyResult<&'static str> {
    let status = engine.cancel(OrderId(order_id));
    Ok(status.map_err(engine_error)?.code())
}

/// `sizing_inputs`: what a sizer sees for an order request at the decision.
pub(super) fn sizing_inputs<'py>(
    py: Python<'py>,
    engine: &Engine,
    request: &Bound<'_, PyDict>,
) -> PyResult<Bound<'py, PyDict>> {
    let request = order_request(request)?;
    let ctx = engine.sizing_inputs(&request);
    sizing::inputs(py, &ctx.map_err(engine_error)?, request.side)
}
