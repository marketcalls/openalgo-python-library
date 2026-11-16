//! The venue's answers to the cancellations and modifications an external session sent
//! it (`venue`), recorded in the slice in progress as its fills are (`venue_fills`).
//!
//! A refused cancellation leaves the order working; the session sends it again in a later
//! slice, not in the one it was refused in (`withdraw_order`). A confirmed modification
//! gives the order the replacement's terms in full; one cut to the units the order has
//! filled (the venue filled them while the modification waited) leaves nothing to fill,
//! so the order completes. A modification below the units filled is refused: the venue
//! refuses it too, and says so with `apply_modify_refused`.

use crate::exec::{OrderAction, OrderId};

use super::error::KernelError;
use super::event::{Event, EventStamp, OrderEventKind};
use super::kernel::Engine;
use super::venue_error::{Pending, VenueError};
use super::venue_fills::result_step;

impl Engine {
    /// Record that the venue refused to cancel `order`: it works on, and the session sends
    /// the cancellation again in a later slice when it still withdraws the order (module
    /// docs).
    ///
    /// Errors: those of `apply_fill` for the session, the engine's state and the order;
    /// `Venue(NotPending)` when no cancellation of it was sent.
    pub fn apply_cancel_refused(&mut self, order: OrderId) -> Result<(), KernelError> {
        self.result_slice()?;
        self.live_order(order)?;
        if !self.venue.cancelling.contains(&order) {
            let what = Pending::Cancel;
            return Err(VenueError::NotPending { order, what }.into());
        }
        self.venue.cancelling.retain(|id| *id != order);
        Ok(())
    }

    /// Record that the venue confirmed the modification of `order` it was sent: the
    /// order takes the replacement's terms in full (`Modified`), and completes when the
    /// replacement's units are those it has filled (module docs). Returns the stamp.
    ///
    /// Errors: those of `apply_fill` for the session, the engine's state and the order;
    /// `Venue(NotPending)` when no modification of it was sent; `Venue(ModifyBelowFilled)`
    /// when the replacement has fewer units than the order has filled.
    pub fn apply_modify(&mut self, order: OrderId) -> Result<EventStamp, KernelError> {
        let slice = self.result_slice()?;
        let filled_units = self.live_order(order)?.filled_units;
        let Some(slot) = self.venue.modifying.iter().position(|(id, _)| *id == order) else {
            let what = Pending::Modify;
            return Err(VenueError::NotPending { order, what }.into());
        };
        let units = self.venue.modifying[slot].1.units;
        if units < filled_units {
            return Err(VenueError::ModifyBelowFilled {
                order,
                units,
                filled_units,
            }
            .into());
        }
        let (_, replacement) = self.venue.modifying.remove(slot);
        self.book.replace(
            order,
            replacement.request,
            replacement.decided_at,
            replacement.good_until,
            replacement.units,
        )?;
        let stamp = Self::stamp(&slice, self.live_order(order).map(result_step)?);
        self.emit(Event::Order {
            stamp,
            order,
            kind: OrderEventKind::Modified,
        });
        if units <= filled_units {
            // Nothing is left to fill: the order has filled all its units.
            self.book.apply(order, OrderAction::Fill)?;
        }
        self.after_result(&slice)?;
        Ok(stamp)
    }

    /// Record that the venue refused the modification of `order` it was sent: the order
    /// keeps its old terms.
    ///
    /// Errors: those of `apply_modify`, but for the units.
    pub fn apply_modify_refused(&mut self, order: OrderId) -> Result<(), KernelError> {
        self.result_slice()?;
        self.live_order(order)?;
        if !self.venue.modifying.iter().any(|(id, _)| *id == order) {
            let what = Pending::Modify;
            return Err(VenueError::NotPending { order, what }.into());
        }
        self.venue.modifying.retain(|(id, _)| *id != order);
        Ok(())
    }
}
