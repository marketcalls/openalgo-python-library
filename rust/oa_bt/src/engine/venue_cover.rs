//! Reduce-only orders never reverse a position in an external session (`venue`).
//!
//! A backtest fills a reduce-only order for at most the position it finds at the fill bar
//! (`fill_bar`); a broker fills the units it holds. So a session keeps the reduce-only
//! orders resting on a position's closing side within the position:
//!
//! - **At submission and modification.** A reduce-only order that would take the units of
//!   the working reduce-only orders on the closing side (what each still has to fill, or
//!   what a modification waiting for the venue's answer would give it, the larger) above
//!   the open position is rejected `exit_cover`, and so is one placed while an exit of the
//!   position waits for its cancels (that exit would cancel it). One stop for the whole
//!   position rests; a stop and a target for it both do not. An exit (`exit()`,
//!   `reduce()`) is sized to fit when it is sent (`venue_owner`) and is checked here only
//!   when it is modified.
//! - **When the position shrinks** (after every venue result, and at each open the session
//!   reaches). The reduce-only orders of a flat position, and those on a side that no
//!   longer closes it, are cancelled. On the closing side, in the order they were placed,
//!   each keeps what still fits in the position: one that fits only in part is cut down to
//!   that with a modification (one at a time, as `modify`: a waiting one is answered
//!   first), and one that has partly filled, or has nothing left to fit, is cancelled. An
//!   order whose cancellation waits counts at what it may still fill.
//! - **A fill beyond the position**, from a race at the broker, is recorded up to the
//!   position and the rest refused (`venue_fills`, `VenueError::WouldReverse`).

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
        let key = key_of(request);
        if self.has_waiting_exit(key) {
            return Err(RejectCode::ExitCover);
        }
        let resting: f64 = self
            .book
            .working()
            .filter(|order| {
                key_of(&order.request) == key
                    && order.request.reduce_only
                    && order.request.side == request.side
                    && Some(order.id) != replacing
            })
            .map(|order| self.units_to_fill(order))
            .sum();
        if resting + units > self.held_units(key).abs() {
            return Err(RejectCode::ExitCover);
        }
        Ok(())
    }

    /// Fit the reduce-only orders of every position to what is open (module docs),
    /// decided and stamped as `at` says. A backtest has nothing to fit.
    ///
    /// Errors: those of `withdraw_order`.
    pub(super) fn fit_exit_cover(&mut self, at: SendAt) -> Result<(), KernelError> {
        if !self.is_external() {
            return Ok(());
        }
        let mut keys: Vec<PositionKey> = Vec::new();
        for order in self.book.working().filter(|o| o.request.reduce_only) {
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
        let orders: Vec<Order> = self
            .book
            .working()
            .filter(|order| {
                key_of(&order.request) == key
                    && order.request.reduce_only
                    && !self.is_unsent(order.id)
            })
            .cloned()
            .collect();
        let mut room = held.abs();
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

    /// Cut a working reduce-only order down to `fits` units still to fill, or cancel it
    /// (module docs).
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
