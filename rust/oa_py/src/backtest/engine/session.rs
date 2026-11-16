//! A live session's controls across the boundary (`oa_bt::engine::venue_session`,
//! `venue_roles`, `venue_gates`): what the live layer asks of an `Engine.session` besides
//! its venue's results.
//!
//! - `request_exit(instrument, product, reason)`: the engine's exit of the whole position
//!   (`instrument` the registration index, `product` "CNC", "MIS" or "NRML"), through the
//!   position's exit owner: the orders on its closing side are cancelled, then it exits.
//!   `reason` is "square_off" or a risk reason of `RISK_EXIT_REASONS` ("unprotected",
//!   "stop_rejected", "stop_unfilled", "kill_switch", "account_loss", "operator_stop", and
//!   the kernel's own "daily_loss", "drawdown", "time_exit"); its "place" command carries
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
//! - `stop_session()`: the session ends where it stands. Nothing is sent, working orders
//!   and open positions are reported as they are by `finish()` (a strategy exit that waited
//!   for its cancels was never placed and ends "canceled"), and the pause tuple "finished"
//!   is returned with the events not delivered yet. Commands not taken stay for
//!   `take_commands()`; they were never sent.
//!
//! A session also rejects at submission an entry or add while another order of its
//! instrument and product works ("live_in_flight") and one that would take the position
//! above the freeze quantity ("freeze_qty"); an order against the position beyond what the
//! orders working to close it leave is an entry, and a modification that would change an
//! order's role raises `NativeEngineInputError`. Every "place" command carries the order's
//! `role`, one of `ORDER_ROLES`, and `reason`, the risk reason of a "risk_exit" (None
//! otherwise). In a backtest every control but the queries `entry_locks` and
//! `daily_loss_measure` raises `NativeEngineError` (and `cancel` is taken at a decision
//! point only); a blank lock reason or one with spaces around it, an unknown exit reason
//! and a cap that is not rupees of 0 or more raise `NativeEngineInputError`.

use oa_bt::engine::{EngineExit, ExitRequest, OrderRole, RiskExitReason};
use pyo3::prelude::*;
use pyo3::types::PyTuple;

use super::setup::unknown;

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

/// The `role` and `reason` keys of a "place" command (module docs).
pub(super) fn role_codes(role: OrderRole) -> (&'static str, Option<&'static str>) {
    (role.code(), role.reason().map(RiskExitReason::code))
}

/// Add the vocabularies of the controls to the module, from the kernel's own lists:
/// `ORDER_ROLES`, `RISK_EXIT_REASONS` and `EXIT_REQUESTS`.
pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();
    m.add("ORDER_ROLES", PyTuple::new_bound(py, OrderRole::CODES))?;
    let reasons = RiskExitReason::ALL.map(RiskExitReason::code);
    m.add("RISK_EXIT_REASONS", PyTuple::new_bound(py, reasons))?;
    m.add("EXIT_REQUESTS", PyTuple::new_bound(py, ExitRequest::CODES))
}
