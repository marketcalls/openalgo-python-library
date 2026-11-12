//! Modifying a working order while the engine is paused: a full replacement.
//!
//! A modification replaces the order's request in full, never a patch: quantity, order
//! type, limit and trigger, leg rules and tag all come from the new request. It keeps
//! the order's id, instrument, side, product, validity and whether it only reduces;
//! anything else is a different order (cancel it and place a new one). The replacement
//! passes the submission checks of a new order at the current decision instant, and is
//! eligible from the first base bar at or after it (SPEC-35); a triggered stop order
//! starts over and must trigger again. A replacement that fails a check is refused and
//! the order works on unchanged. A GTT order's GTT is replaced with it.

use crate::exec::{ExecError, OrderId, OrderRequest};

use super::error::KernelError;
use super::event::{Event, OrderEventKind, Step};
use super::kernel::{Engine, Phase};
use super::submit::{Resolved, Submission};

/// Why a request cannot replace an order, if it cannot.
fn replacement_error(current: &OrderRequest, replacement: &OrderRequest) -> Option<ExecError> {
    let same = current.instrument == replacement.instrument
        && current.side == replacement.side
        && current.product == replacement.product
        && current.validity == replacement.validity
        && current.reduce_only == replacement.reduce_only;
    (!same).then(|| ExecError::InvalidRequest {
        reason: "a modification keeps the order's symbol, side, product, validity and \
                 exit-only kind; cancel the order and place a new one instead"
            .to_string(),
    })
}

impl Engine {
    /// Replace a working order's request in full (see the module documentation).
    ///
    /// Returns `Submitted(id)` when the order was replaced (event `Modified`, step 7),
    /// `Rejected(id, code)` when the replacement failed a submission check (the order
    /// is unchanged and no event is written), `Skipped` when a sizer gave no quantity.
    ///
    /// Errors: `NotAtDecisionPoint` unless paused; `UnknownOrder`; `Exec(InvalidRequest)`
    /// for an order that no longer works or a replacement that changes what identifies
    /// the order; the errors of `submit` for the replacement itself.
    pub fn modify(
        &mut self,
        id: OrderId,
        request: OrderRequest,
    ) -> Result<Submission, KernelError> {
        let Phase::Paused(slice) = self.phase else {
            return Err(KernelError::NotAtDecisionPoint);
        };
        let order = self.book.get(id).ok_or(KernelError::UnknownOrder(id))?;
        if !order.status.is_working() {
            return Err(KernelError::Exec(ExecError::InvalidRequest {
                reason: format!(
                    "order {id} is {} and can no longer be modified",
                    order.status.code()
                ),
            }));
        }
        if let Some(error) = replacement_error(&order.request, &request) {
            return Err(error.into());
        }
        self.check_buildable(&request)?;
        let request = self.venue_request(request)?;
        let units = match self.resolve_units(&request, &slice)? {
            Resolved::Units(units) => units,
            Resolved::AtFill => 0.0,
            Resolved::Skipped(reason) => return Ok(Submission::Skipped(reason)),
        };
        let good_until = match self.submission_checks(&request, units, &slice)? {
            Ok(good_until) => good_until,
            Err(code) => return Ok(Submission::Rejected(id, code)),
        };
        self.book
            .replace(id, request, slice.close, good_until, units)?;
        self.emit(Event::Order {
            stamp: Self::stamp(&slice, Step::Callbacks),
            order: id,
            kind: OrderEventKind::Modified,
        });
        self.modify_order_gtt(&slice, id)?;
        Ok(Submission::Submitted(id))
    }
}
