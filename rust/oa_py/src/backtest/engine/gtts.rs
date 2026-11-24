//! A live session's GTTs across the boundary (`oa_bt::engine::venue_gtts` and the modules
//! around it; design 20-positional-live, unit N1): the GTT commands, the venue's answers
//! about them, and the queries.
//!
//! - **Commands.** `take_commands()` hands a GTT's command as a dict with `command`
//!   ("place_gtt", "modify_gtt" or "cancel_gtt"), `gtt_id` (the kernel's GTT, an integer;
//!   Python shows `g<n>`) and its terms: `instrument` (the registration index), `product`
//!   ("CNC" or "NRML"), `side` (of the child order the GTT places: "BUY" or "SELL"),
//!   `quantity_units` (units, never lots), `trigger_price` and `price` (the trigger and the
//!   child's limit, rupees on the tick) and `trigger_below` (True when the trigger lies
//!   below the price it was placed from: a long's stop; the live layer sends it in the
//!   field of its direction). "place_gtt" adds `order_id`: the strategy's GTT order it
//!   rests as, or None for a leg's stop. A "modify_gtt" carries the full replacement; a
//!   "cancel_gtt" the terms as they rest.
//! - **Answers**, taken while the session waits, as `apply_fill` is, each returning the
//!   stamp it was recorded with (slice index, slice open in UTC nanoseconds, step number)
//!   or None: `apply_gtt_placed(gtt_id)`, `apply_gtt_refused(gtt_id)` (a leg's stop:
//!   exited at market, "risk_exit" "stop_rejected", and entries locked for the session
//!   with `GTT_REFUSED_LOCK`; a strategy's GTT order ends rejected),
//!   `apply_gtt_modified(gtt_id)`, `apply_gtt_modify_refused(gtt_id)` (None),
//!   `apply_gtt_cancelled(gtt_id)` (one the session did not ask for ended outside it: the
//!   leg gets a new stop GTT), `apply_gtt_cancel_refused(gtt_id)` (None),
//!   `apply_gtt_ended(gtt_id, how)` (`how` one of `GTT_ENDS`: "expired", "rejected"; a
//!   leg's stop rejected a second time exits the leg) and `apply_gtt_triggered(gtt_id, at)`
//!   (the venue fired it at `at`, UTC nanoseconds: returns the kernel order id the child
//!   is followed as, whose fills, cancellation and rejection then come with `apply_fill`,
//!   `apply_cancel`, `apply_reject`). An unknown GTT, or one whose state takes no such
//!   answer (a second answer to one request), raises `VenueRefusalError` of kind
//!   "unknown_gtt" or "gtt_state" and changes nothing; a fill of a strategy GTT order the
//!   venue has not fired raises kind "gtt_resting".
//! - **Queries.** `gtt(gtt_id)` (None for an unknown id) and `gtts(instrument, product)`
//!   (in id order) give dicts with `gtt_id`, the terms above, `purpose` ("stop" or
//!   "order"), `order_id` (the strategy's order, None for a stop), `state` (one of
//!   `GTT_STATES`: "Requested" until the venue confirms the place, "Active",
//!   "ModifyPending" and "CancelPending" while an answer waits, "Triggered",
//!   "ChildFilled", "ChildCanceled", "Canceled", "Rejected", "VenueEnded", "Expired"),
//!   `pending` (the terms of a modification waiting for its answer, as a dict, or None)
//!   and `child_order_id` (the fired GTT's child, or None).

use oa_bt::engine::{Engine, GttEnd, GttTerms, GttView, GTT_REFUSED_LOCK};
use oa_bt::exec::{GttId, GttPurpose, GttStatus, InstrumentId, OrderId};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

use super::setup::{parse_product, unknown};
use super::venue_result_error;

/// The terms of a GTT into `dict` (module docs).
pub(super) fn terms_into(dict: &Bound<'_, PyDict>, terms: &GttTerms) -> PyResult<()> {
    dict.set_item("instrument", terms.instrument.0)?;
    dict.set_item("product", terms.product.code())?;
    dict.set_item("side", terms.side.code())?;
    dict.set_item("quantity_units", terms.units)?;
    dict.set_item("trigger_price", terms.trigger_rupees)?;
    dict.set_item("price", terms.limit_rupees)?;
    dict.set_item("trigger_below", terms.trigger_below)
}

/// A GTT's command dict (module docs): `command`, `gtt_id`, its terms, and for a place the
/// strategy order it rests as.
pub(super) fn command_into(
    dict: &Bound<'_, PyDict>,
    (command, gtt): (&str, GttId),
    order: Option<Option<OrderId>>,
    terms: &GttTerms,
) -> PyResult<()> {
    dict.set_item("command", command)?;
    dict.set_item("gtt_id", gtt.0)?;
    if let Some(order) = order {
        dict.set_item("order_id", order.map(|id| id.0))?;
    }
    terms_into(dict, terms)
}

/// The way `how` names a GTT's end (module docs).
///
/// Errors: `NativeEngineInputError` for a way that names none.
pub(super) fn gtt_end(how: &str) -> PyResult<GttEnd> {
    GttEnd::from_code(how).ok_or_else(|| {
        let codes: Vec<&str> = GttEnd::ALL.iter().map(|end| end.code()).collect();
        unknown("GTT end", how, &format!("one of {}", codes.join(", ")))
    })
}

/// `apply_gtt_triggered`: the kernel order id of the child, or the refusal.
pub(super) fn triggered(py: Python<'_>, engine: &mut Engine, gtt: u64, at: i64) -> PyResult<u64> {
    let child = engine.apply_gtt_triggered(GttId(gtt), at);
    child
        .map(|order| order.0)
        .map_err(|error| venue_result_error(py, error))
}

fn view_dict<'py>(py: Python<'py>, view: &GttView) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new_bound(py);
    dict.set_item("gtt_id", view.gtt.id.0)?;
    terms_into(&dict, &view.terms)?;
    let (purpose, order) = match view.gtt.purpose {
        GttPurpose::Protect => ("stop", None),
        GttPurpose::Order(order) => ("order", Some(order.0)),
    };
    dict.set_item("purpose", purpose)?;
    dict.set_item("order_id", order)?;
    dict.set_item("state", view.gtt.status.code())?;
    let pending = match &view.pending {
        Some(terms) => {
            let pending = PyDict::new_bound(py);
            terms_into(&pending, terms)?;
            pending.into_py(py)
        }
        None => py.None(),
    };
    dict.set_item("pending", pending)?;
    dict.set_item("child_order_id", view.child.map(|id| id.0))?;
    Ok(dict)
}

/// `gtt(gtt_id)`: the GTT as a dict, None for an unknown id (module docs).
pub(super) fn gtt<'py>(
    py: Python<'py>,
    engine: &Engine,
    gtt: u64,
) -> PyResult<Option<Bound<'py, PyDict>>> {
    engine
        .gtt_view(GttId(gtt))
        .map(|view| view_dict(py, &view))
        .transpose()
}

/// `gtts(instrument, product)`: the GTTs of a position, in id order (module docs).
///
/// Errors: `NativeEngineInputError` for an unknown product.
pub(super) fn gtts<'py>(
    py: Python<'py>,
    engine: &Engine,
    (instrument, product): (u32, &str),
) -> PyResult<Bound<'py, PyList>> {
    let product = parse_product(product)?;
    let views = engine.gtt_views(InstrumentId(instrument), product);
    let dicts = views
        .iter()
        .map(|view| view_dict(py, view))
        .collect::<PyResult<Vec<_>>>()?;
    Ok(PyList::new_bound(py, dicts))
}

/// Add `GTT_ENDS`, `GTT_STATES` and `GTT_REFUSED_LOCK` to the module, from the kernel's
/// own lists.
pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();
    let ends = GttEnd::ALL.map(GttEnd::code);
    m.add("GTT_ENDS", PyTuple::new_bound(py, ends))?;
    let states = GttStatus::ALL.map(GttStatus::code);
    m.add("GTT_STATES", PyTuple::new_bound(py, states))?;
    m.add("GTT_REFUSED_LOCK", GTT_REFUSED_LOCK)
}
