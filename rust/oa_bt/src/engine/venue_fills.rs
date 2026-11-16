//! Results from the venue of an external session (`venue`): its fills, rejections and
//! cancellations, recorded as a backtest records its own, and its answers to the
//! cancellations and modifications it was sent.
//!
//! A fill takes the one fill path, `fills::execute_fill` (position, ledger, the cost
//! model's charges, the trade book, `Filled` and the position event), at the venue's price
//! exactly: no slippage model, and the fill is never re-priced. The units are those newly
//! filled (fills are increments), so partial fills accumulate, are costed once on their
//! total (`charges::parts`), and the order completes when its units are filled. A fill
//! that takes a position through zero is recorded as a backtest records a reversal: the
//! close, then the open on the other side. The first fill of a stop order (SL, SL-M)
//! writes `Triggered` before it, since the venue filled it only once its trigger was
//! reached. A fill is checked whole before anything of it is recorded (the margin rule
//! and the charges of each of its parts): it is recorded all or nothing.
//!
//! Every result is recorded in the slice in progress: the first slice that has not run,
//! the slice of a timer pause, or the session-closing slice that waits for its orders to
//! end. That is the slice that contains the venue's fill time when the venue reports in
//! time; a result reported after its slice has run is recorded in the slice in progress,
//! since a slice that has run is never rewritten, and so is one whose time lies ahead of
//! it (a venue clock ahead of the session's). The step is the one in which a backtest
//! fills that kind of order: 2 for a MARKET order, 3 for LIMIT, SL and SL-M. A fill at the
//! slice's open instant is at the `Open`, any other at a `Level`. A result is refused
//! while the engine is paused at a bar close (step 7): the paused slice completes with
//! `advance` first. After a result the exit owners move on (`venue_owner`): an exit
//! waiting for this order is sent once nothing blocks it.
//!
//! The contract for the live layer: every fill of an order is delivered before its
//! terminal result (`apply_cancel`, `apply_reject`); after it the order takes no more.

use crate::exec::{
    FillAt, FillFlags, FillPricing, Order, OrderAction, OrderId, OrderStatus, OrderType, RejectCode,
};
use crate::model::{Side, TsNanos};
use crate::portfolio::PositionKey;
use crate::rms::{ExitReason, LegRules};

use super::clock::SliceInfo;
use super::error::KernelError;
use super::event::{Event, EventStamp, OrderEventKind, Step};
use super::fills::FillRequest;
use super::futures::FillOrigin;
use super::kernel::{Engine, Phase};
use super::venue::SendAt;
use super::venue_error::{Pending, VenueError};

/// The step in which a backtest fills an order of this kind.
fn result_step(order: &Order) -> Step {
    match order.request.order_type {
        OrderType::Market => Step::OpenFills,
        _ => Step::Levels,
    }
}

/// Refuse units and a price that cannot be a fill.
fn check_fill(units: f64, price_rupees: f64) -> Result<(), VenueError> {
    if !(units.is_finite() && units > 0.0) {
        return Err(VenueError::InvalidFill {
            reason: format!("{units} units: pass the units newly filled, above 0"),
        });
    }
    if !(price_rupees.is_finite() && price_rupees > 0.0) {
        return Err(VenueError::InvalidFill {
            reason: format!("price {price_rupees}: pass the venue's fill price in rupees, above 0"),
        });
    }
    Ok(())
}

impl Engine {
    /// Record a fill of `units` (the units newly filled, above 0) of `order` at
    /// `price_rupees` (above 0), filled at `at` (UTC nanoseconds) by the venue of an
    /// external session (module docs). Returns the stamp it was recorded with.
    ///
    /// Errors: `Venue(NotExternal)` in a backtest; `Venue(AtBarClose)` while paused at a
    /// bar close and `Venue(SessionOver)` after the session finished; `Venue(InvalidFill)`
    /// for units or a price that is not finite or not above 0; `UnknownOrder`;
    /// `Venue(OrderEnded)` for an order that has ended and `Venue(NotSent)` for an exit not
    /// sent; `Venue(TooManyUnits)` for more units than the order has left; and those of
    /// the margin rule and the cost model, before anything is recorded.
    pub fn apply_fill(
        &mut self,
        order: OrderId,
        units: f64,
        price_rupees: f64,
        at: TsNanos,
    ) -> Result<EventStamp, KernelError> {
        let slice = self.result_slice()?;
        check_fill(units, price_rupees)?;
        let placed = self.live_order(order)?.clone();
        let left_units = placed.quantity_units - placed.filled_units;
        if units > left_units {
            return Err(VenueError::TooManyUnits {
                order,
                units,
                left_units,
            }
            .into());
        }
        let parts = self.fill_parts(&placed, units);
        self.check_recordable(&placed, parts, price_rupees, &slice)?;
        let stamp = Self::stamp(&slice, result_step(&placed));
        let fill_at = if at == slice.open {
            FillAt::Open
        } else {
            FillAt::Level
        };
        self.trigger_before_fill(stamp, &placed)?;
        let fill = VenueFill {
            slice,
            stamp,
            parts,
            price_rupees,
            at: fill_at,
        };
        self.record_venue_fill(&placed, fill)?;
        self.after_result(&slice)?;
        Ok(stamp)
    }

    /// Record that the venue of an external session rejected `order` (`Rejected` with
    /// `reason`: `RejectCode::Venue`, or a kernel code when the venue's reason is one, such
    /// as `Margin`). A partly filled order keeps its fills. Returns the stamp.
    ///
    /// Errors: those of `apply_fill` for the session, the engine's state and the order.
    pub fn apply_reject(
        &mut self,
        order: OrderId,
        reason: RejectCode,
    ) -> Result<EventStamp, KernelError> {
        let kind = OrderEventKind::Rejected(reason);
        self.end_at_venue(order, OrderAction::Reject, kind)
    }

    /// Record that the venue of an external session cancelled `order`, on the session's
    /// request or of its own accord (`Canceled`). A partly filled order keeps its fills.
    /// Returns the stamp.
    ///
    /// Errors: those of `apply_fill` for the session, the engine's state and the order.
    pub fn apply_cancel(&mut self, order: OrderId) -> Result<EventStamp, KernelError> {
        self.end_at_venue(order, OrderAction::Cancel, OrderEventKind::Canceled)
    }

    /// Record that the venue refused to cancel `order`: it works on, and may be cancelled
    /// again (an exit waiting for it sends the cancellation again at the next open).
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
    /// order takes the replacement's terms in full (`Modified`). Returns the stamp.
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

    fn end_at_venue(
        &mut self,
        order: OrderId,
        action: OrderAction,
        kind: OrderEventKind,
    ) -> Result<EventStamp, KernelError> {
        let slice = self.result_slice()?;
        let stamp = Self::stamp(&slice, result_step(self.live_order(order)?));
        self.end_order(stamp, order, action, kind)?;
        self.after_result(&slice)?;
        Ok(stamp)
    }

    /// After a venue result: the exit owners move on, decided now in the slice in
    /// progress (at its close when it is the session's closing slice).
    fn after_result(&mut self, slice: &SliceInfo) -> Result<(), KernelError> {
        let decided_at = match self.phase {
            Phase::Closing(_) => slice.close,
            _ => slice.open,
        };
        let at = SendAt {
            slice: *slice,
            step: Step::OpenFills,
            decided_at,
        };
        self.progress_exits(at)
    }

    /// The slice a venue's result is recorded in (module docs).
    fn result_slice(&self) -> Result<SliceInfo, KernelError> {
        if !self.is_external() {
            return Err(VenueError::NotExternal.into());
        }
        let index = match self.phase {
            Phase::Ready(index) => index,
            Phase::Timers(slice) | Phase::Ahead(slice) | Phase::Closing(slice) => return Ok(slice),
            Phase::Paused(_) => return Err(VenueError::AtBarClose.into()),
            Phase::Finished => return Err(VenueError::SessionOver.into()),
        };
        self.timeline
            .slice(index)
            .ok_or_else(|| VenueError::SessionOver.into())
    }

    /// The order a result is for: it works and was sent.
    fn live_order(&self, id: OrderId) -> Result<&Order, KernelError> {
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

    /// The units of a fill of `order` that reduce its position, and those that open or
    /// add (a reversal has both).
    fn fill_parts(&self, order: &Order, units: f64) -> (f64, f64) {
        let key = PositionKey {
            instrument: order.request.instrument,
            product: order.request.product,
        };
        let held = self.held_units(key);
        let reduces = held != 0.0 && (held > 0.0) != (order.request.side == Side::Buy);
        let closing = if reduces { units.min(held.abs()) } else { 0.0 };
        (closing, units - closing)
    }

    /// Check what the fill path could fail on, recording nothing: the charges of each
    /// part, the margin rule of the opening part.
    fn check_recordable(
        &self,
        order: &Order,
        (closing, opening): (f64, f64),
        price: f64,
        slice: &SliceInfo,
    ) -> Result<(), KernelError> {
        let request = &order.request;
        let key = PositionKey {
            instrument: request.instrument,
            product: request.product,
        };
        for units in [closing, opening].into_iter().filter(|units| *units > 0.0) {
            self.check_chargeable(key, request.side, units, price, slice.day)?;
        }
        if opening > 0.0 {
            let (day, side, leverage) = (slice.day, request.side, request.leverage);
            self.margin_quote(day, key, side, opening, price, leverage)?;
        }
        Ok(())
    }

    /// The first fill of a stop order: its trigger was reached (`Triggered`).
    fn trigger_before_fill(&mut self, stamp: EventStamp, order: &Order) -> Result<(), KernelError> {
        let waiting = matches!(order.status, OrderStatus::Submitted | OrderStatus::Accepted);
        if order.request.order_type.trigger_rupees().is_none() || !waiting {
            return Ok(());
        }
        self.book.apply(order.id, OrderAction::Trigger)?;
        self.emit(Event::Order {
            stamp,
            order: order.id,
            kind: OrderEventKind::Triggered,
        });
        Ok(())
    }

    /// Execute a venue's fill: the part that reduces the position, then the part that
    /// opens or adds, with the margin it blocks (module docs).
    fn record_venue_fill(&mut self, order: &Order, fill: VenueFill) -> Result<(), KernelError> {
        let request = &order.request;
        let key = PositionKey {
            instrument: request.instrument,
            product: request.product,
        };
        let (closing, opening) = fill.parts;
        let base = FillRequest {
            key,
            side: request.side,
            units: closing,
            reference_rupees: fill.price_rupees,
            at: fill.at,
            order: Some(order.id),
            exit_reason: request.exit_reason.unwrap_or(ExitReason::Signal),
            flags: FillFlags::NONE,
            margin_rupees: 0.0,
            leg_rules: LegRules::default(),
            target_index: None,
            pricing: FillPricing::Resting,
            gtt: None,
            origin: FillOrigin::Trade,
        };
        if closing > 0.0 {
            self.execute_fill(fill.stamp, base.clone())?;
        }
        if opening > 0.0 {
            let (day, side, price) = (fill.slice.day, request.side, fill.price_rupees);
            let margin = self.opening_margin(day, key, side, opening, price, request.leverage)?;
            let open = FillRequest {
                units: opening,
                margin_rupees: margin,
                leg_rules: request.leg_rules.clone(),
                ..base
            };
            self.execute_fill(fill.stamp, open)?;
        }
        Ok(())
    }
}

/// A fill from the venue, placed in the slice it is recorded in.
#[derive(Clone, Copy, Debug)]
struct VenueFill {
    slice: SliceInfo,
    stamp: EventStamp,
    /// Units that reduce the position, and units that open or add.
    parts: (f64, f64),
    price_rupees: f64,
    at: FillAt,
}
