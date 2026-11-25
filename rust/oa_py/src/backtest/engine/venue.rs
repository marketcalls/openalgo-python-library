//! A live session across the boundary (`oa_bt::engine::live_clock`, `venue`): its bars in,
//! the commands it hands the adapter, and the venue's results coming back.
//!
//! A live session (`Engine.session`, the inputs of a run with the bars of the slices
//! already closed) is built before its bars. The adapter hands it each closed slice's
//! bars with `add_bar(instrument, bar)` (a bar is (ts_open UTC nanoseconds, open, high,
//! low, close, volume)) and releases the slice with `release(slices)`; `advance` returns
//! the pause kind "waiting" before a slice not released. Its venue is external (`venue()`
//! says "external"; a backtest's is "simulated"): every order is handed over as a command
//! (`take_commands()`), and the broker's results come back with `apply_fill(order_id,
//! units, price, at, flags=())` (the units newly filled, the venue's price in rupees, the
//! fill time in UTC nanoseconds, and the names of the flags the venue reports with the
//! fill: "price_unverified" for a price not known, recorded at a fired GTT's child limit,
//! "missed_day" for a fill made on a trading day no session ran, the names of
//! `VENUE_FILL_FLAGS`; a name that is no flag's raises `NativeEngineInputError`, one of
//! the engine's own flags `VenueRefusalError` of kind "invalid_fill", and nothing is
//! recorded; every flag's name is in `FILL_FLAGS`; the fill and its trade carry them in
//! the "flags" columns of the result's fills and trades), `apply_reject(order_id, reason)`
//! (a reject code: "venue", or a kernel code such as "margin") and
//! `apply_cancel(order_id)`, and the broker's answers to
//! what it was sent with `apply_cancel_refused(order_id)`, `apply_modify(order_id)` and
//! `apply_modify_refused(order_id)`. A closing order (a leg's stop, an exit, the engine's
//! exit) whose rejection says the broker may not hold its units goes to
//! `apply_margin_reject(order_id)` instead of `apply_reject`: it ends rejected with the
//! reject code "venue", and its position is held: the call itself, and every call after
//! it, sends nothing that closes the position or covers more of it (no exit at market, no
//! stop placed again or given more units, an exit that waits stays unsent, none at later
//! opens) until the session calls `request_exit` (which also sends at once an engine exit
//! the hold kept waiting) or `protect` (which moves the waiting exits on and puts the stop
//! back over the position), or a new leg opens there. So the session calls those two only
//! once a book read agrees with what the strategies hold (18-multi-strategy 4.14). The
//! hold is not saved in `live_state`. An entry or add raises `NativeEngineInputError`.
//! Every fill of an order is passed before its terminal result (`apply_cancel`,
//! `apply_reject`, `apply_margin_reject`): after it the order takes no more. A result the
//! kernel refuses raises `openalgo.backtest.errors.VenueRefusalError`, whose `kind` says
//! why (one of the module's `VENUE_REFUSAL_KINDS`), and changes nothing, except a fill
//! beyond the position of an order that may not reverse it (kind "would_reverse"): the
//! units that closed the position are recorded, the rest are not.
//!
//! A command crosses as a dict with `command` ("place", "modify" or "cancel") and
//! `order_id` (the kernel order, an integer; Python shows `o<n>`); "place" adds
//! `instrument` (the registration index), `product` ("CNC", "MIS", "NRML"), `side`
//! ("BUY", "SELL"), `role` (what the order is for, one of `ORDER_ROLES`: "entry", "add",
//! "leg_stop", "exit", "risk_exit", "square_off"), `reason` (the risk reason of a
//! "risk_exit", one of `RISK_EXIT_REASONS`; None for any other role) and `flags` (the
//! names of the flags its fills carry: "late_decision" for an order of a pre-open decision
//! made after the open, "gtt_child_unfilled" or "stop_crossed" for the engine's exits that
//! carry them; empty for any other order); "place" and "modify" add `quantity_units`
//! (units, never lots),
//! `order_type` ("MARKET", "LIMIT", "SL", "SL-M"), `price` (the limit in rupees, None for
//! MARKET and SL-M) and `trigger_price` (rupees, None for MARKET and LIMIT). A result
//! returns the stamp it was recorded with as (slice index, slice open in UTC nanoseconds,
//! step number). The session's own controls (an exit it asks for, the entry locks, the
//! sizing cap, the stop) are in `session`; a GTT's commands and answers in `gtts`.

use oa_bt::engine::{BaseBar, Engine, EventStamp, KernelError, VenueCommand, VENUE_REFUSAL_KINDS};
use oa_bt::exec::{FillFlags, InstrumentId, OrderId, OrderType, RejectCode};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

use super::errors::{engine_error, venue_result_error};
use super::gtts::command_into;
use super::session::role_codes;
use super::setup::unknown;

/// A base bar as it crosses: (ts_open UTC nanoseconds, open, high, low, close, volume).
pub(super) type BarTuple = (i64, f64, f64, f64, f64, f64);

/// A recorded result's stamp: (slice index, slice open in UTC nanoseconds, step number).
pub(super) type StampTuple = (u32, i64, u8);

fn base_bar((ts_open, open, high, low, close, volume): BarTuple) -> BaseBar {
    BaseBar {
        ts_open,
        open,
        high,
        low,
        close,
        volume,
    }
}

/// `add_bar`: hand a live session an instrument's base bar once its slice has closed.
pub(super) fn add_bar(engine: &mut Engine, instrument: u32, bar: BarTuple) -> PyResult<()> {
    let added = engine.add_bar(InstrumentId(instrument), base_bar(bar));
    added.map_err(engine_error)
}

/// `apply_fill`: the venue's fill of `units` of `order_id` at `price` rupees at `at` (UTC
/// nanoseconds), with the flags `flags` names (module docs); the stamp.
///
/// Errors: `NativeEngineInputError` for a name that is no fill flag's, before anything
/// is recorded; `VenueRefusalError` for a refusal of the kernel.
pub(super) fn fill(
    py: Python<'_>,
    engine: &mut Engine,
    (order_id, units, price, at): (u64, f64, f64, i64),
    flags: &[String],
) -> PyResult<StampTuple> {
    let mut parsed = FillFlags::NONE;
    for name in flags {
        let flag = FillFlags::from_name(name).ok_or_else(|| {
            let venue = FillFlags::FROM_VENUE.names().join(" or ");
            unknown("fill flag", name, &venue)
        })?;
        parsed = parsed.with(flag);
    }
    let filled = engine.apply_fill_flagged(OrderId(order_id), units, price, at, parsed);
    recorded(py, filled)
}

/// `apply_reject`: the venue rejected an order with the reject code `reason`; the stamp.
pub(super) fn reject(
    py: Python<'_>,
    engine: &mut Engine,
    order_id: u64,
    reason: &str,
) -> PyResult<StampTuple> {
    let reason = reject_code(reason)?;
    recorded(py, engine.apply_reject(OrderId(order_id), reason))
}

/// A venue result's stamp, or the error it was refused with (`VenueRefusalError` for a
/// refusal).
pub(super) fn recorded(
    py: Python<'_>,
    result: Result<EventStamp, KernelError>,
) -> PyResult<StampTuple> {
    let stamp = result.map_err(|error| venue_result_error(py, error))?;
    Ok((stamp.slice, stamp.slice_ts, stamp.step.number()))
}

/// A venue answer that records no stamp, or the error it was refused with.
pub(super) fn answered(py: Python<'_>, result: Result<(), KernelError>) -> PyResult<()> {
    result.map_err(|error| venue_result_error(py, error))
}

/// Add every kind of `VenueRefusalError` to the module as `VENUE_REFUSAL_KINDS`, every
/// fill flag's name as `FILL_FLAGS` and those a venue reports with a fill as
/// `VENUE_FILL_FLAGS`, from the kernel's own lists.
pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();
    m.add(
        "VENUE_REFUSAL_KINDS",
        PyTuple::new_bound(py, VENUE_REFUSAL_KINDS),
    )?;
    let names = FillFlags::NAMED.map(|(_, name)| name);
    m.add("FILL_FLAGS", PyTuple::new_bound(py, names))?;
    let venue = FillFlags::FROM_VENUE.names();
    m.add("VENUE_FILL_FLAGS", PyTuple::new_bound(py, venue))
}

/// The commands as a list of dicts (module docs); `flags` gives an order's fill flags.
pub(super) fn commands<'py>(
    py: Python<'py>,
    commands: &[VenueCommand],
    flags: impl Fn(OrderId) -> FillFlags,
) -> PyResult<Bound<'py, PyList>> {
    let dicts = commands
        .iter()
        .map(|item| command(py, item, &flags))
        .collect::<PyResult<Vec<_>>>()?;
    Ok(PyList::new_bound(py, dicts))
}

/// The reject code of a venue's rejection: "venue" for a reason of the venue's own, or
/// a kernel code such as "margin" when the venue's reason is one.
fn reject_code(code: &str) -> PyResult<RejectCode> {
    RejectCode::from_code(code).ok_or_else(|| {
        let codes: Vec<&str> = RejectCode::ALL.iter().map(|code| code.code()).collect();
        unknown("reject code", code, &format!("one of {}", codes.join(", ")))
    })
}

fn order_type_into(dict: &Bound<'_, PyDict>, units: f64, order_type: OrderType) -> PyResult<()> {
    dict.set_item("quantity_units", units)?;
    dict.set_item("order_type", order_type.code())?;
    dict.set_item("price", order_type.limit_rupees())?;
    dict.set_item("trigger_price", order_type.trigger_rupees())
}

/// One command as a dict (module docs).
fn command<'py>(
    py: Python<'py>,
    command: &VenueCommand,
    flags: &impl Fn(OrderId) -> FillFlags,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new_bound(py);
    match *command {
        VenueCommand::Place {
            order,
            instrument,
            product,
            side,
            units,
            order_type,
            role,
        } => {
            dict.set_item("command", "place")?;
            dict.set_item("order_id", order.0)?;
            dict.set_item("instrument", instrument.0)?;
            dict.set_item("product", product.code())?;
            dict.set_item("side", side.code())?;
            let (role, reason) = role_codes(role);
            dict.set_item("role", role)?;
            dict.set_item("reason", reason)?;
            dict.set_item("flags", flags(order).names())?;
            order_type_into(&dict, units, order_type)?;
        }
        VenueCommand::Modify {
            order,
            units,
            order_type,
        } => {
            dict.set_item("command", "modify")?;
            dict.set_item("order_id", order.0)?;
            order_type_into(&dict, units, order_type)?;
        }
        VenueCommand::Cancel { order } => {
            dict.set_item("command", "cancel")?;
            dict.set_item("order_id", order.0)?;
        }
        VenueCommand::PlaceGtt { gtt, order, terms } => {
            command_into(&dict, ("place_gtt", gtt), Some(order), &terms)?;
        }
        VenueCommand::ModifyGtt { gtt, terms } => {
            command_into(&dict, ("modify_gtt", gtt), None, &terms)?;
        }
        VenueCommand::CancelGtt { gtt, terms } => {
            command_into(&dict, ("cancel_gtt", gtt), None, &terms)?;
        }
    }
    Ok(dict)
}
