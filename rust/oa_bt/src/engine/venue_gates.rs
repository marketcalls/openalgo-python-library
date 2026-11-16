//! The entry gates an external session adds (`venue`; 16-session-design 6.6, requirements
//! R9, R16 and R17; 08-live-trading section 6.1). They judge an order that opens or adds
//! (`OrderRole::opens`) at submission and at modification, after the backtest's checks, in
//! this order. That includes an order against the position beyond what the orders working
//! to close it leave (`venue_roles`): it could open the other side.
//!
//! | Gate | Rejects the entry or add | Code |
//! |---|---|---|
//! | Entry locks | while any reason locks entries (`Engine::lock_entries`) | `live_lock` |
//! | Nothing in flight | while any other order of its instrument and product works, but the open leg's resting stop, which keeps it out too while a modification or cancellation of the stop waits for the venue and once the stop is repriced to exit (`venue_protect`); a stop a closed leg left working (its cancellation refused) keeps it out, since a new leg's stop beside it would make two | `live_in_flight` |
//! | Freeze cap | when it would take the position above the instrument's freeze quantity | `freeze_qty` |
//!
//! Why: a backtest fills a MARKET entry at the next open, before the next decision, but a
//! broker may report its fill later (a polling broker), so an entry on top of one in flight
//! could double the position; and one stop order and one exit order cover a position, while
//! the broker refuses any order above the freeze quantity, so a position may never exceed
//! it. Exits and stops pass every gate. A backtest has none of them: its freeze rule judges
//! each order alone (`exec::checks::freeze_rule`).

use crate::exec::{Order, OrderId, OrderRequest, RejectCode};
use crate::portfolio::PositionKey;

use super::clock::SliceInfo;
use super::error::KernelError;
use super::kernel::Engine;

fn key_of(request: &OrderRequest) -> PositionKey {
    PositionKey {
        instrument: request.instrument,
        product: request.product,
    }
}

/// True when a working order keeps an entry or add of its instrument and product out
/// (module docs): any order but the open leg's resting stop (`legs_stop`), and that stop
/// too while it is busy (the venue has not answered a modification or cancellation of
/// it, or it was repriced to exit).
fn holds_entries(legs_stop: bool, stop_busy: bool) -> bool {
    !legs_stop || stop_busy
}

impl Engine {
    /// The checks of an external session after the backtest's, for `request` of `units`
    /// decided in `slice`: its entry gates (module docs), then what the fill path could not
    /// record (`venue_checks`). `replacing` is the order a modification replaces. A
    /// backtest checks nothing here; `Ok(Err(code))` rejects the order.
    ///
    /// Errors: those of `venue_checks`.
    pub(super) fn live_checks(
        &self,
        request: &OrderRequest,
        units: f64,
        slice: &SliceInfo,
        replacing: Option<OrderId>,
    ) -> Result<Result<(), RejectCode>, KernelError> {
        if let Err(code) = self.session_entry_gates(request, units, replacing) {
            return Ok(Err(code));
        }
        self.venue_checks(request, units, slice, replacing)
    }

    /// The gates of the module docs, in order.
    ///
    /// Errors: the reject code of the first gate that refuses the order.
    fn session_entry_gates(
        &self,
        request: &OrderRequest,
        units: f64,
        replacing: Option<OrderId>,
    ) -> Result<(), RejectCode> {
        if !self.is_external() || !self.submission_role(request, units, replacing).opens() {
            return Ok(());
        }
        if !self.venue.controls.entry_locks.is_empty() {
            return Err(RejectCode::LiveLock);
        }
        let key = key_of(request);
        let legs_stop = self
            .resting_leg_stop(key)
            .filter(|_| self.held_units(key) != 0.0);
        let in_flight = |order: &Order| {
            key_of(&order.request) == key
                && Some(order.id) != replacing
                && holds_entries(legs_stop == Some(order.id), self.stop_busy(key, order.id))
        };
        if self.book.working().any(in_flight) {
            return Err(RejectCode::LiveInFlight);
        }
        let freeze = self
            .instruments
            .get(request.instrument.index())
            .and_then(|spec| spec.freeze_quantity_units);
        let position_after = (self.held_units(key) + units * request.side.sign()).abs();
        if freeze.is_some_and(|limit| position_after > f64::from(limit)) {
            return Err(RejectCode::FreezeQty);
        }
        Ok(())
    }

    /// True while a cancellation or modification of order `id` waits for the venue's
    /// answer, or once it is `key`'s leg stop repriced to exit (`venue_protect`).
    fn stop_busy(&self, key: PositionKey, id: OrderId) -> bool {
        self.venue.cancelling.contains(&id)
            || self.venue.modifying.iter().any(|(m, _)| *m == id)
            || self.repriced_leg_stop(key) == Some(id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_working_order_but_the_open_legs_resting_stop_keeps_entries_out() {
        // (the open leg's stop, busy at the venue, keeps entries out)
        let table = [
            (true, false, false),
            (true, true, true),
            (false, false, true),
            (false, true, true),
        ];
        for (legs_stop, stop_busy, keeps_out) in table {
            assert_eq!(
                holds_entries(legs_stop, stop_busy),
                keeps_out,
                "the leg's stop {legs_stop}, busy {stop_busy}"
            );
        }
    }
}
