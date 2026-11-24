//! What an external session refuses of a request (`venue`): the parts its venue cannot
//! carry yet (`KernelError::NotBuilt`), and the modifications it cannot send. A backtest
//! refuses none of them here.
//!
//! Leg rules on CNC and NRML are taken: their stop is a GTT at the broker (`venue_gtts`),
//! their targets the kernel's on ticks (`venue_ticks`). So is a strategy's GTT order that
//! opens or adds (`validity="gtt"`), which rests as a GTT.
//!
//! | Refused | Why |
//! |---|---|
//! | A GTT order that only reduces, or any GTT order against the position held | not built for a live session: a resting GTT exit beside the leg's stop GTT would be a second exit at the broker, and both could fire (a backtest cancels the stop GTT before such an exit fills; the broker does not); use `stop=` and `target=` |
//! | `rebalance()` | not built for a live session |
//! | Modifying a partly filled order | not built: cancel it and place a new one |
//! | Modifying a GTT order | not built: cancel it and place a new one. The venue holds only its GTT, whose answers carry the GTT's terms and not the order's request (its leg rules, tag, units), and a GTT is modified at most once a minute in live (04-execution 4.4 item 5): the replacement would wait up to a minute, with no answer path to apply it to the order |
//! | Modifying an order whose last modification waits, or an exit not sent yet | one modification at a time; an exit waiting for its cancels has nothing at the venue |
//! | A replacement left with no units | the venue takes no order of none: cancel it |

use crate::exec::{ExecError, Order, OrderRequest, Validity};
use crate::portfolio::PositionKey;

use super::error::KernelError;
use super::kernel::Engine;
use super::venue_roles::goes_against;

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
    /// Errors: `NotBuilt` for a GTT order that only reduces or goes against the position
    /// held, and a rebalance order.
    pub(super) fn check_venue_request(&self, request: &OrderRequest) -> Result<(), KernelError> {
        if !self.is_external() {
            return Ok(());
        }
        let instead = "hold the position's exits with stop= and target=, or place a DAY order";
        if request.validity == Validity::Gtt && request.reduce_only {
            return Err(not_built(
                "a GTT order that only reduces, in a live session",
                instead,
            ));
        }
        let key = PositionKey {
            instrument: request.instrument,
            product: request.product,
        };
        if request.validity == Validity::Gtt && goes_against(self.held_units(key), request.side) {
            return Err(not_built(
                "a GTT order against the position held, in a live session",
                instead,
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
    /// Errors: `NotBuilt` for a partly filled order and a GTT order; `Exec(InvalidRequest)`
    /// for the others.
    pub(super) fn check_venue_replacement(&self, order: &Order) -> Result<(), KernelError> {
        if !self.is_external() {
            return Ok(());
        }
        if order.request.validity == Validity::Gtt {
            return Err(not_built(
                "modifying a GTT order in a live session",
                "cancel the order and place a new one",
            ));
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
