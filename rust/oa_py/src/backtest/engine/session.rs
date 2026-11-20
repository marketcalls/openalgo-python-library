//! A live session's controls across the boundary (`oa_bt::engine::venue_session`,
//! `venue_roles`, `venue_gates`): what the live layer asks of an `Engine.session` besides
//! its venue's results.
//!
//! - `request_exit(instrument, product, reason)`: the engine's exit of the whole position
//!   (`instrument` the registration index, `product` "CNC", "MIS" or "NRML"), through the
//!   position's exit owner: the orders on its closing side are cancelled, then it exits.
//!   `reason` is "square_off" or a risk reason of `RISK_EXIT_REASONS` ("unprotected",
//!   "stop_rejected", "stop_unfilled", "kill_switch", "account_loss", "account_profit",
//!   "operator_stop", and the kernel's own "daily_loss", "daily_profit", "drawdown",
//!   "time_exit"); its "place" command carries
//!   the role "square_off", or "risk_exit" with the reason. Taken while the session waits
//!   or is paused at a decision point. Returns what it did, one of `EXIT_REQUESTS`:
//!   "sent", "waits_for_cancels" (sent once the orders on the closing side have ended, if
//!   the market is open then), and nothing at all for "already_exiting" (an engine exit
//!   holds the position), "flat" and "market_closed". Asking again says where it stands.
//! - `cancel(order_id)` is taken while the session waits too, and sends the cancellation
//!   again for an order whose cancellation the venue refused (`apply_cancel_refused`).
//! - `lock_entries(reason)` and `unlock_entries(reason)` (True when the set of reasons
//!   changed) and `entry_locks()` (the reasons, sorted): while any reason is set, an entry
//!   or add is rejected at submission with the reject code "live_lock"; exits are not. A
//!   reason is a name with no spaces around it.
//! - `cap_sizing_equity(rupees)`: the engine default sizes from the lesser of the
//!   strategy's equity and the cap (the account's available cash).
//! - `daily_loss_measure()`: the strategy's daily loss measure, rupees on its risk basis,
//!   negative for a loss.
//! - `trades_today()`: the entries the strategy has counted on its session day, for the
//!   trade caps (18-multi-strategy 2.5a). `set_account_trades(limit, others)`: the
//!   account's trade cap (None for none) and the entries the session's other strategies
//!   have counted today; while set, an entry when the two counts together reach it is
//!   rejected at submission with "account_trade_limit".
//! - `set_group_side(instrument, product, side)`: on an instrument other strategies of the
//!   session trade too, the side they hold or are entering, one of `GROUP_SIDES` ("long",
//!   "short"), or None to clear it: while set, an entry, add or reversal on the other side
//!   is rejected at submission with the reject code "group_other_side"; exits are not.
//! - `set_shared(instrument, product, shared)`: while shared, a strategy order that rests
//!   (LIMIT, SL, SL-M), entry or exit, is rejected at submission with "shared_resting";
//!   the leg's stop, the engine's exits and the price rule's LIMIT orders are not.
//! - `order_exit_reason(order_id)`: the exit reason an order's fills record, or None: the
//!   live layer tells the exit a leg's target makes on a tick ("leg_target") from the
//!   strategy's own exits, which share its role "exit".
//! - `stop_session()`: the session ends where it stands. Nothing is sent, working orders
//!   and open positions are reported as they are by `finish()` (a strategy exit that waited
//!   for its cancels was never placed and ends "canceled"), the slice in progress is closed
//!   for the records as a backtest closes a session (the day's contract notes settled, then
//!   its position-history and equity rows, so the equity holds every fill and the notes'
//!   charges), and the pause tuple "finished" is returned with the events not delivered
//!   yet. Commands not taken stay for
//!   `take_commands()`; they were never sent.
//!
//! A session also rejects at submission an entry or add while another order of its
//! instrument and product works ("live_in_flight") and one that would take the position
//! above the freeze quantity ("freeze_qty"); an order against the position beyond what the
//! orders working to close it leave is an entry, and a modification that would change an
//! order's role raises `NativeEngineInputError`. Every "place" command carries the order's
//! `role`, one of `ORDER_ROLES`, and `reason`, the risk reason of a "risk_exit" (None
//! otherwise). In a backtest every control but the queries `entry_locks`,
//! `daily_loss_measure` and `trades_today`, and `set_account_trades`, raises
//! `NativeEngineError` (and `cancel` is taken at a decision point only); a blank lock
//! reason or one with spaces around it, an unknown exit reason, an unknown group side, a
//! cap that is not rupees of 0 or more and an account trade cap of 0 raise
//! `NativeEngineInputError`.

use oa_bt::engine::{
    Engine, EngineExit, ExitRequest, GroupSide, KernelError, OrderRole, Protect, RiskExitReason,
};
use oa_bt::exec::{InstrumentId, OrderId};
use oa_bt::rms::{AccountTrades, ExitReason};
use pyo3::prelude::*;
use pyo3::types::PyTuple;

use super::engine_error;
use super::setup::{parse_product, unknown};

/// `request_exit`: the session's exit of a position for `reason`, at market or as LIMIT
/// orders at `limit_rupees`; what it did, one of `EXIT_REQUESTS` (module docs).
pub(super) fn request_exit(
    engine: &mut Engine,
    (instrument, product): (u32, &str),
    reason: &str,
    limit_rupees: Option<f64>,
) -> PyResult<&'static str> {
    let (product, exit) = (parse_product(product)?, engine_exit(reason)?);
    let instrument = InstrumentId(instrument);
    let asked = match limit_rupees {
        None => engine.request_exit(instrument, product, exit),
        Some(limit) => engine.request_limit_exit(instrument, product, exit, limit),
    };
    asked.map(ExitRequest::code).map_err(engine_error)
}

/// `protect`: the leg's stop placed again; what it did, one of `PROTECT_OUTCOMES`.
pub(super) fn protect(
    engine: &mut Engine,
    instrument: u32,
    product: &str,
) -> PyResult<&'static str> {
    let done = engine.protect(InstrumentId(instrument), parse_product(product)?);
    done.map(Protect::code).map_err(engine_error)
}

/// `set_group_side`: the side the other strategies hold or enter on an instrument, or
/// None (module docs).
pub(super) fn set_group_side(
    engine: &mut Engine,
    (instrument, product): (u32, &str),
    side: Option<&str>,
) -> PyResult<()> {
    let (product, side) = (parse_product(product)?, group_side(side)?);
    let set = engine.set_group_side(InstrumentId(instrument), product, side);
    set.map_err(engine_error)
}

/// `set_shared`: mark an instrument as traded by several strategies, or not (module
/// docs).
pub(super) fn set_shared(
    engine: &mut Engine,
    (instrument, product): (u32, &str),
    shared: bool,
) -> PyResult<()> {
    let set = engine.set_shared(InstrumentId(instrument), parse_product(product)?, shared);
    set.map_err(engine_error)
}

/// `set_account_trades`: the account's trade cap and the other strategies' entries of the
/// day (module docs).
pub(super) fn set_account_trades(
    engine: &mut Engine,
    limit: Option<u32>,
    others: u32,
) -> PyResult<()> {
    let trades = limit.map(|limit| AccountTrades { limit, others });
    engine.set_account_trades(trades).map_err(engine_error)
}

/// The engine exit a session's `reason` names (module docs).
///
/// Errors: `NativeEngineInputError` for a reason that names none.
pub(super) fn engine_exit(reason: &str) -> PyResult<EngineExit> {
    EngineExit::from_code(reason).ok_or_else(|| {
        let codes: Vec<&str> = RiskExitReason::ALL.iter().map(|r| r.code()).collect();
        let allowed = format!("\"square_off\" or one of {}", codes.join(", "));
        unknown("exit reason", reason, &allowed)
    })
}

/// The group side a session's `side` names, or None (module docs).
///
/// Errors: `NativeEngineInputError` for a side that names none.
pub(super) fn group_side(side: Option<&str>) -> PyResult<Option<GroupSide>> {
    side.map(|code| {
        GroupSide::from_code(code).ok_or_else(|| {
            let allowed = format!("one of {} or None", GroupSide::CODES.join(", "));
            unknown("group side", code, &allowed)
        })
    })
    .transpose()
}

/// The `role` and `reason` keys of a "place" command (module docs).
pub(super) fn role_codes(role: OrderRole) -> (&'static str, Option<&'static str>) {
    (role.code(), role.reason().map(RiskExitReason::code))
}

/// The exit reason the fills of order `id` record (module docs).
///
/// Errors: `NativeEngineError` for an order the engine does not know.
pub(super) fn order_exit_reason(engine: &Engine, id: OrderId) -> PyResult<Option<&'static str>> {
    let order = engine
        .order(id)
        .ok_or_else(|| engine_error(KernelError::UnknownOrder(id)))?;
    Ok(order.request.exit_reason.map(ExitReason::code))
}

/// Add the vocabularies of the controls to the module, from the kernel's own lists:
/// `ORDER_ROLES`, `RISK_EXIT_REASONS`, `EXIT_REQUESTS` and `GROUP_SIDES`.
pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();
    m.add("ORDER_ROLES", PyTuple::new_bound(py, OrderRole::CODES))?;
    let reasons = RiskExitReason::ALL.map(RiskExitReason::code);
    m.add("RISK_EXIT_REASONS", PyTuple::new_bound(py, reasons))?;
    m.add("EXIT_REQUESTS", PyTuple::new_bound(py, ExitRequest::CODES))?;
    m.add("GROUP_SIDES", PyTuple::new_bound(py, GroupSide::CODES))
}
