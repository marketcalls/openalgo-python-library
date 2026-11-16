//! Results from the venue of an external session (`venue`): its fills, rejections and
//! cancellations, recorded as a backtest records its own.
//!
//! A fill takes the one fill path, `fills::execute_fill` (position, ledger, the cost
//! model's charges, the trade book, `Filled` and the position event), at the venue's price
//! exactly: no slippage model, and the fill is never re-priced. The units are those newly
//! filled (fills are increments), so partial fills accumulate and the order completes
//! when its units are filled. A fill that takes a position through zero is recorded as a
//! backtest records a reversal: the close, then the open on the other side. The first
//! fill of a stop order (SL, SL-M) writes `Triggered` before it, since the venue filled
//! it only once its trigger was reached.
//!
//! Every result is recorded in the slice in progress: the first slice that has not run,
//! or the slice of a timer pause. That is the slice that contains the venue's fill time
//! when the venue reports in time; a result reported after its slice has run is recorded
//! in the slice in progress, since a slice that has run is never rewritten, and so is one
//! whose time lies ahead of it (a venue clock ahead of the session's). The step is the
//! one in which a backtest fills that kind of order: 2 for a MARKET order, 3 for LIMIT,
//! SL and SL-M. A fill at the slice's open instant is at the `Open`, any other at a
//! `Level`. A result is refused while the engine is paused at a bar close (step 7): the
//! paused slice completes with `advance` first.

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
use super::venue_error::VenueError;

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
    /// bar close, `RunFinished` after the session finished and `Venue(SessionOver)` once
    /// every slice has run; `Venue(InvalidFill)` for units or a price that is not finite
    /// or not above 0; `UnknownOrder`; `Venue(OrderEnded)` for an order that has ended;
    /// `Venue(TooManyUnits)` for more units than the order has left; and those of
    /// `fills::execute_fill` and the margin model.
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
            units,
            price_rupees,
            at: fill_at,
        };
        self.record_venue_fill(&placed, fill)?;
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

    fn end_at_venue(
        &mut self,
        order: OrderId,
        action: OrderAction,
        kind: OrderEventKind,
    ) -> Result<EventStamp, KernelError> {
        let slice = self.result_slice()?;
        let stamp = Self::stamp(&slice, result_step(self.live_order(order)?));
        self.end_order(stamp, order, action, kind)?;
        Ok(stamp)
    }

    /// The slice a venue's result is recorded in (module docs).
    fn result_slice(&self) -> Result<SliceInfo, KernelError> {
        if !self.is_external() {
            return Err(VenueError::NotExternal.into());
        }
        let index = match self.phase {
            Phase::Ready(index) => index,
            Phase::Timers(slice) => return Ok(slice),
            Phase::Paused(_) => return Err(VenueError::AtBarClose.into()),
            Phase::Finished => return Err(KernelError::RunFinished),
        };
        self.timeline
            .slice(index)
            .ok_or_else(|| VenueError::SessionOver.into())
    }

    /// The order a result is for, while it works.
    fn live_order(&self, id: OrderId) -> Result<&Order, KernelError> {
        let order = self.book.get(id).ok_or(KernelError::UnknownOrder(id))?;
        if !order.status.is_working() {
            let status = order.status;
            return Err(VenueError::OrderEnded { order: id, status }.into());
        }
        Ok(order)
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
        let held = self.held_units(key);
        let reduces = held != 0.0 && (held > 0.0) != (request.side == Side::Buy);
        let closing = if reduces {
            fill.units.min(held.abs())
        } else {
            0.0
        };
        let opening = fill.units - closing;
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
    units: f64,
    price_rupees: f64,
    at: FillAt,
}
