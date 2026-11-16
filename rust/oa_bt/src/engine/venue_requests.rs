//! What an external session refuses of a request (`venue`): the parts its venue cannot
//! carry yet (`KernelError::NotBuilt`), and the modifications it cannot send. A backtest
//! refuses none of them here.
//!
//! | Refused | Why |
//! |---|---|
//! | Leg rules on CNC or NRML | those legs are held as GTTs, which a session does not carry; MIS legs are the broker's and the kernel's (`venue_legs`, `venue_ticks`) |
//! | A GTT order, `rebalance()` | not built for a live session |
//! | Modifying a partly filled order | not built: cancel it and place a new one for the rest |
//! | Modifying an order whose last modification waits, or an exit not sent yet | one modification at a time; an exit waiting for its cancels has nothing at the venue |
//! | A replacement left with no units | the venue takes no order of none: cancel it |

use crate::exec::{ExecError, Order, OrderRequest, Validity};

use super::error::KernelError;
use super::gtt::held_as_gtt;
use super::kernel::Engine;

fn not_built(feature: &'static str, instead: &'static str) -> KernelError {
    KernelError::NotBuilt { feature, instead }
}

fn invalid(reason: &str) -> KernelError {
    KernelError::Exec(ExecError::InvalidRequest {
        reason: reason.to_string(),
    })
}

impl Engine {
    /// Refuse, in an external session, what the venue cannot carry yet (module docs).
    ///
    /// Errors: `NotBuilt` for leg rules on CNC or NRML, a GTT order and a rebalance order.
    pub(super) fn check_venue_request(&self, request: &OrderRequest) -> Result<(), KernelError> {
        if !self.is_external() {
            return Ok(());
        }
        if !request.leg_rules.is_empty() && held_as_gtt(request.product) {
            return Err(not_built(
                "stop=, target=, trailing and breakeven on CNC or NRML in a live session",
                "live trading is MIS only: trade MIS, where the broker holds the stop",
            ));
        }
        if request.validity == Validity::Gtt {
            return Err(not_built(
                "a GTT order in a live session",
                "live trading is MIS only: place a DAY order",
            ));
        }
        if request.rebalance {
            return Err(not_built(
                "rebalance() in a live session",
                "trade each symbol with self.target(), the sales before the buys",
            ));
        }
        Ok(())
    }

    /// Refuse, in an external session, a modification the venue cannot carry: an order
    /// that has partly filled, one whose last modification is not answered yet, and an
    /// exit not sent yet.
    ///
    /// Errors: `NotBuilt` for a partly filled order; `Exec(InvalidRequest)` for the others.
    pub(super) fn check_venue_replacement(&self, order: &Order) -> Result<(), KernelError> {
        if !self.is_external() {
            return Ok(());
        }
        if order.filled_units > 0.0 {
            return Err(not_built(
                "modifying a partly filled order in a live session",
                "cancel the order and place a new one for the rest",
            ));
        }
        if self.venue.modifying.iter().any(|(id, _)| *id == order.id) {
            return Err(invalid(
                "the order's last modification waits for the venue's answer: modify it again \
                 once the venue has confirmed or refused it",
            ));
        }
        if self.is_unsent(order.id) {
            return Err(invalid(
                "the exit waits for the orders it cancels and has not been sent: cancel it and \
                 exit again instead",
            ));
        }
        Ok(())
    }

    /// Refuse, in an external session, a replacement left with no units (an exit with
    /// nothing left to reduce): the venue takes no order of none.
    ///
    /// Errors: `Exec(InvalidRequest)` saying to cancel the order instead.
    pub(super) fn check_replacement_units(&self, units: f64) -> Result<(), KernelError> {
        if self.is_external() && units <= 0.0 {
            return Err(invalid(
                "the replacement has nothing left to reduce: cancel the order instead",
            ));
        }
        Ok(())
    }
}
