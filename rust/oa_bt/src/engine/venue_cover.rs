//! Orders sent to close never reverse a position in an external session (`venue`).
//!
//! A backtest fills a reduce-only order for at most the position it finds at the fill bar
//! (`fill_bar`); a broker fills the units it holds. So a session keeps the orders working
//! on a position's closing side within the position:
//!
//! - **At submission and modification.** The closing side of a position has room for its
//!   units less what every working order on that side may still fill (what each has left,
//!   or what a modification waiting for the venue's answer would give it, the larger),
//!   reduce-only or not, and for nothing while an exit of the position waits for its
//!   cancels (that exit closes the whole position, and cancels what is placed meanwhile):
//!   `closing_room`. A reduce-only order beyond the room is rejected `exit_cover`. One
//!   stop for the whole position rests; a stop and a target for it both do not, and
//!   neither do a stop and a plain sale of the long. An exit (`exit()`, `reduce()`) is
//!   sized to fit when it is sent (`venue_owner`) and is checked here only when it is
//!   modified. An order against the position that is not reduce-only counts as an exit
//!   only within the room (`venue_roles`); a MARKET one also within the room the leg's
//!   resting stop makes for it, since the exit owner cuts the stop down or cancels it for
//!   it first (`venue_owner`).
//! - **When the position shrinks** (after every venue result, and at each open the session
//!   reaches). The orders sent only to close it (`Engine::sent_to_close`: reduce-only, or a
//!   plain order whose role is an exit) are fitted to it. The session's own fills keep them
//!   within it, by the room above; a position restored, adopted or reduced at the broker
//!   may not. Those of a flat position, and those on a side that no longer closes it, are
//!   cancelled. On the closing side, the leg's resting stop first and the others in the
//!   order they were placed, each keeps what still fits in the position: one that fits
//!   only in part is cut down to that with a modification (one at a time, as `modify`: a
//!   waiting one is answered first), and one that has partly filled, or has nothing left
//!   to fit, is cancelled. An order whose
//!   cancellation waits counts at what it may still fill.
//! - **A fill beyond the position**, from a race at the broker, is recorded up to the
//!   position and the rest refused (`venue_fills`, `VenueError::WouldReverse`).
//! - **A CNC or NRML leg's stop GTT** (`venue_gtts`) is not an order, but the broker
//!   fills its child as one: its units take their room on the closing side as the leg's
//!   resting stop does, first, and the GTT itself is cut to the position by
//!   `venue_gtt_sync`.

use crate::exec::{Order, OrderId, OrderRequest, QuantitySpec, RejectCode};
use crate::portfolio::PositionKey;

use super::error::KernelError;
use super::kernel::Engine;
use super::venue::{is_exit, Replacement, SendAt};
use super::venue_owner::closing_side;

fn key_of(request: &OrderRequest) -> PositionKey {
    PositionKey {
        instrument: request.instrument,
        product: request.product,
    }
}

impl Engine {
    /// The units a working order may still fill: what it has left, or what a modification
    /// waiting for the venue's answer would leave it, the larger; 0 once it has ended.
    pub(super) fn units_to_fill(&self, order: &Order) -> f64 {
        if !order.status.is_working() {
            return 0.0;
        }
        let pending = self
            .venue
            .modifying
            .iter()
            .find(|(id, _)| *id == order.id)
            .map_or(0.0, |(_, replacement)| replacement.units);
        order.quantity_units.max(pending) - order.filled_units
    }

    /// The units the closing side of `key`'s position has room for (module docs): the
    /// position less what every working order on that side but `except` may still fill;
    /// 0 for a flat position and while an exit of it waits for its cancels. `except` is
    /// the order being judged, once it is in the book (a modification's own order).
    pub(super) fn closing_room(&self, key: PositionKey, except: Option<OrderId>) -> f64 {
        let held = self.held_units(key);
        if held == 0.0 || self.has_waiting_exit(key) {
            return 0.0;
        }
        let side = closing_side(held);
        let working: f64 = self
            .book
            .working()
            .filter(|order| {
                key_of(&order.request) == key
                    && order.request.side == side
                    && Some(order.id) != except
            })
            .map(|order| self.units_to_fill(order))
            .sum();
        (held.abs() - working - self.stop_gtt_units(key)).max(0.0)
    }

    /// In an external session, the cover rule of a reduce-only order of `units` (above 0)
    /// at submission, or as the replacement of `replacing` (module docs); a backtest and
    /// an exit at submission pass.
    ///
    /// Errors: `RejectCode::ExitCover` when the order does not fit.
    pub(super) fn check_exit_cover(
        &self,
        request: &OrderRequest,
        units: f64,
        replacing: Option<OrderId>,
    ) -> Result<(), RejectCode> {
        let sized_when_sent = is_exit(request) && replacing.is_none();
        if !self.is_external() || !request.reduce_only || sized_when_sent {
            return Ok(());
        }
        if units > self.closing_room(key_of(request), replacing) {
            return Err(RejectCode::ExitCover);
        }
        Ok(())
    }

    /// Fit the orders sent only to close every position to what is open (module docs),
    /// decided and stamped as `at` says. A backtest has nothing to fit.
    ///
    /// Errors: those of `withdraw_order`.
    pub(super) fn fit_exit_cover(&mut self, at: SendAt) -> Result<(), KernelError> {
        if !self.is_external() {
            return Ok(());
        }
        let mut keys: Vec<PositionKey> = Vec::new();
        for order in self.book.working().filter(|o| self.sent_to_close(o)) {
            let key = key_of(&order.request);
            if !keys.contains(&key) {
                keys.push(key);
            }
        }
        for key in keys {
            self.fit_position_cover(at, key)?;
        }
        Ok(())
    }

    fn fit_position_cover(&mut self, at: SendAt, key: PositionKey) -> Result<(), KernelError> {
        let held = self.held_units(key);
        let closing = (held != 0.0).then(|| closing_side(held));
        let mut orders: Vec<Order> = self
            .book
            .working()
            .filter(|order| {
                key_of(&order.request) == key
                    && self.sent_to_close(order)
                    && !self.is_unsent(order.id)
            })
            .cloned()
            .collect();
        // The leg's resting stop keeps its room before the others, as it takes it from
        // them when it is placed or grows (`venue_stop_terms`): the two never undo each
        // other. The sort is stable: the others stay in the order they were placed.
        let stop = self.resting_leg_stop(key);
        orders.sort_by_key(|order| Some(order.id) != stop);
        let mut room = (held.abs() - self.stop_gtt_units(key)).max(0.0);
        for order in orders {
            if Some(order.request.side) != closing {
                self.withdraw_order(at.stamp(), order.id)?;
                continue;
            }
            let units = self.units_to_fill(&order);
            let fits = units.min(room);
            room -= fits;
            if fits < units {
                self.cut_down(at, &order, fits)?;
            }
        }
        Ok(())
    }

    /// Cut a working order sent only to close down to `fits` units still to fill, or cancel
    /// it (module docs).
    fn cut_down(&mut self, at: SendAt, order: &Order, fits: f64) -> Result<(), KernelError> {
        let cancel =
            fits <= 0.0 || order.filled_units > 0.0 || self.venue.cancelling.contains(&order.id);
        if cancel {
            self.withdraw_order(at.stamp(), order.id)?;
            return Ok(());
        }
        if self.venue.modifying.iter().any(|(id, _)| *id == order.id) {
            return Ok(());
        }
        let request = OrderRequest {
            quantity: QuantitySpec::Units(fits),
            ..order.request.clone()
        };
        let replacement = Replacement {
            request,
            decided_at: order.decided_at,
            good_until: order.good_until,
            units: fits,
        };
        self.send_modify(order.id, replacement);
        Ok(())
    }
}

#[cfg(test)]
mod shrink_tests;
