//! Modifying a working order while the engine is paused: a full replacement.
//!
//! A modification replaces the order's request in full, never a patch: quantity, order
//! type, limit and trigger, leg rules and tag all come from the new request. It keeps
//! the order's id, instrument, side, product, validity and whether it only reduces;
//! anything else is a different order (cancel it and place a new one). The replacement
//! passes the submission checks of a new order at the current decision instant, and is
//! eligible from the first base bar at or after it (SPEC-35); a triggered stop order
//! starts over and must trigger again. A replacement that fails a check is refused and
//! the order works on unchanged; so is one above the order value cap (`daily_caps`),
//! which is not a new trade for the trade caps. A GTT order's GTT is replaced with it.
//! Under a delay the replacement takes effect when its delay ends, the order working on
//! its old terms until then (`held_changes`).

use crate::exec::{ExecError, OrderId, OrderRequest, RejectCode};

use super::clock::SliceInfo;
use super::daily_caps::Capped;
use super::error::KernelError;
use super::kernel::Engine;
use super::submit::{Resolved, Submission};
use super::venue::Replacement;

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
    /// Returns `Submitted(id)` when the order was replaced (event `Modified`, in the step
    /// of the decision, or when a held replacement takes effect: `held_changes`),
    /// `Rejected(id, code)` when the replacement failed a submission
    /// check (the order is unchanged and no event is written), `Skipped` when a sizer gave
    /// no quantity.
    ///
    /// In a live session the replacement is sent to the venue and the order keeps its
    /// old terms until the venue confirms it (`Engine::apply_modify`, which writes
    /// `Modified`) or refuses it (`venue`).
    ///
    /// Errors: `NotAtDecisionPoint` unless paused; `UnknownOrder`; `Exec(InvalidRequest)`
    /// for an order that no longer works or a replacement that changes what identifies
    /// the order, and in a live session for an order whose last modification the venue
    /// has not answered, an exit not sent yet, or a replacement that passes the backtest's
    /// checks but would change what the order is for (its role, `venue_roles`); `NotBuilt`
    /// for a partly filled order in a live session; the errors of `submit` for the
    /// replacement itself.
    pub fn modify(
        &mut self,
        id: OrderId,
        request: OrderRequest,
    ) -> Result<Submission, KernelError> {
        let Some((slice, decided_at, step)) = self.decision_point() else {
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
        self.check_venue_replacement(order)?;
        self.check_buildable(&request)?;
        self.check_venue_request(&request)?;
        let request = self.venue_request(request)?;
        let (units, refused) = match self.resolve_units(&request, &slice)? {
            Resolved::Units(units) => (units, None),
            Resolved::AtFill => (0.0, None),
            Resolved::Refused(code) => (0.0, Some(code)),
            Resolved::Skipped(reason) => return Ok(Submission::Skipped(reason)),
        };
        let units = self.venue_units(&request, units);
        self.check_replacement_units(units)?;
        // As in `submit`: the checks come first, then the account's refusal.
        let checks = self.submission_checks(&request, units, &slice, decided_at, Some(id))?;
        let checked = match (checks, refused) {
            (Err(code), _) | (Ok(_), Some(code)) => Err(code),
            (Ok(good_until), None) => {
                self.check_replacement_role(id, &request, units)?;
                match self.live_checks(&request, units, &slice, Some(id))? {
                    Ok(()) => self.replacement_caps(&request, units, &slice)?,
                    Err(code) => Err(code),
                }
                .map(|()| good_until)
            }
        };
        let good_until = match checked {
            Ok(good_until) => good_until,
            Err(code) => return Ok(Submission::Rejected(id, code)),
        };
        let replacement = Replacement {
            request,
            decided_at,
            good_until,
            units,
        };
        if self.is_external() {
            self.send_modify(id, replacement);
        } else {
            self.modify_or_hold(Self::stamp(&slice, step), id, replacement)?;
        }
        Ok(Submission::Submitted(id))
    }

    /// The daily caps of a replacement of `units` (module docs): only the order value cap
    /// can refuse it, since a modification is not a new trade; `Err(code)` refuses it.
    ///
    /// Errors: `Exec` for an instrument the run does not have.
    fn replacement_caps(
        &self,
        request: &OrderRequest,
        units: f64,
        slice: &SliceInfo,
    ) -> Result<Result<(), RejectCode>, KernelError> {
        Ok(match self.daily_caps(request, units, slice, true)? {
            Capped::Pass(_) => Ok(()),
            Capped::Refuse(code) | Capped::CutReversal(code) => Err(code),
        })
    }
}
