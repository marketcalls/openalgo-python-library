//! The venue's rejections and cancellations of an external session's orders (`venue`),
//! and where every venue result is recorded: the slice in progress (`venue_fills`).
//!
//! A rejection or a cancellation ends the order with its event; a partly filled order
//! keeps its fills. When the order is a leg's broker-held stop, the leg is protected again
//! (`venue_legs`): whether the kernel had asked for a cancellation is read before the
//! order ends.

use crate::exec::{Order, OrderAction, OrderId, RejectCode};

use super::clock::SliceInfo;
use super::error::KernelError;
use super::event::{EventStamp, OrderEventKind, Step};
use super::kernel::{Engine, Phase};
use super::venue::SendAt;
use super::venue_error::VenueError;
use super::venue_fills::result_step;
use super::venue_legs::StopEnd;

impl Engine {
    /// Record that the venue of an external session rejected `order` (`Rejected` with
    /// `reason`: `RejectCode::Venue`, or a kernel code when the venue's reason is one, such
    /// as `Margin`). A partly filled order keeps its fills. A leg's stop rejected while its
    /// leg is open exits the leg at market (`venue_legs`). Returns the stamp.
    ///
    /// Errors: those of `apply_fill` for the session, the engine's state and the order.
    pub fn apply_reject(
        &mut self,
        order: OrderId,
        reason: RejectCode,
    ) -> Result<EventStamp, KernelError> {
        let kind = OrderEventKind::Rejected(reason);
        self.end_at_venue(order, OrderAction::Reject, kind, StopEnd::Rejected)
    }

    /// Record that the venue of an external session cancelled `order`, on the session's
    /// request or of its own accord (`Canceled`). A partly filled order keeps its fills. A
    /// leg's stop the venue ended without a cancellation the kernel sent (cancelled by the
    /// broker, or not placed) is re-placed once (`venue_legs`). Returns the stamp.
    ///
    /// Errors: those of `apply_fill` for the session, the engine's state and the order.
    pub fn apply_cancel(&mut self, order: OrderId) -> Result<EventStamp, KernelError> {
        let requested = self.venue.cancelling.contains(&order);
        let how = StopEnd::Canceled { requested };
        self.end_at_venue(order, OrderAction::Cancel, OrderEventKind::Canceled, how)
    }

    /// End `order` at the venue with `action` and its event `kind`, then protect the leg
    /// as `how` the stop ended says (`venue_legs`) and move the exits on (`after_result`).
    ///
    /// Errors: those of `apply_reject`.
    pub(super) fn end_at_venue(
        &mut self,
        order: OrderId,
        action: OrderAction,
        kind: OrderEventKind,
        how: StopEnd,
    ) -> Result<EventStamp, KernelError> {
        let slice = self.result_slice()?;
        let stamp = Self::stamp(&slice, result_step(self.live_order(order)?));
        self.end_order(stamp, order, action, kind)?;
        self.after_stop_end(self.waiting_at(&slice), order, how)?;
        self.after_result(&slice)?;
        Ok(stamp)
    }

    /// After a venue result, decided now in the slice in progress (at its close when it is
    /// the session's closing slice): the reduce-only orders are fitted to their positions,
    /// then the waiting exits move on and the legs' stops follow their legs.
    ///
    /// Errors: those of `fit_exit_cover` and of `progress_exits`.
    pub(super) fn after_result(&mut self, slice: &SliceInfo) -> Result<(), KernelError> {
        let at = self.waiting_at(slice);
        self.fit_exit_cover(at)?;
        self.progress_exits(at)
    }

    /// Where the engine decides while an external session waits: in `slice`, the slice in
    /// progress (`result_slice`), at its open, or at its close when it is the session's
    /// closing slice waiting for its orders to end (`venue_clock`); stamped in step 2.
    pub(super) fn waiting_at(&self, slice: &SliceInfo) -> SendAt {
        let decided_at = match self.phase {
            Phase::Closing(_) => slice.close,
            _ => slice.open,
        };
        SendAt {
            slice: *slice,
            step: Step::OpenFills,
            decided_at,
        }
    }

    /// The slice a venue's result is recorded in (`venue_fills` module docs).
    ///
    /// Errors: `Venue(NotExternal)` in a backtest, `Venue(AtBarClose)` while paused at a
    /// bar close, `Venue(SessionOver)` after the session.
    pub(super) fn result_slice(&self) -> Result<SliceInfo, KernelError> {
        if !self.is_external() {
            return Err(VenueError::NotExternal.into());
        }
        let index = match self.phase {
            Phase::Ready(index) => index,
            Phase::Opened(slice)
            | Phase::Timers(slice)
            | Phase::Ahead(slice)
            | Phase::Closing(slice) => return Ok(slice),
            Phase::Paused(_) => return Err(VenueError::AtBarClose.into()),
            Phase::Finished => return Err(VenueError::SessionOver.into()),
        };
        self.timeline
            .slice(index)
            .ok_or_else(|| VenueError::SessionOver.into())
    }

    /// The order a result is for: it works and was sent.
    ///
    /// Errors: `UnknownOrder`, `Venue(OrderEnded)`, `Venue(NotSent)`.
    pub(super) fn live_order(&self, id: OrderId) -> Result<&Order, KernelError> {
        let order = self.book.get(id).ok_or(KernelError::UnknownOrder(id))?;
        if !order.status.is_working() {
            let status = order.status;
            return Err(VenueError::OrderEnded { order: id, status }.into());
        }
        if self.is_unsent(id) {
            return Err(VenueError::NotSent { order: id }.into());
        }
        Ok(order)
    }
}
