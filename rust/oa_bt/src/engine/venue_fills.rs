//! Fills from the venue of an external session (`venue`), recorded as a backtest records
//! its own. Its rejections and cancellations are in `venue_ends`, its answers to the
//! cancellations and modifications it was sent in `venue_answers`.
//!
//! A fill takes the one fill path, `fills::execute_fill` (position, ledger, the cost
//! model's charges, the trade book, `Filled` and the position event), at the venue's price
//! exactly: no slippage model, and the fill is never re-priced. The units are those newly
//! filled (fills are increments), so partial fills accumulate, are costed once on their
//! total (`charges::parts`), and the order completes when its units are filled. A fill
//! that takes a position through zero is recorded as a backtest records a reversal (the
//! close, then the open on the other side) for an order with `reverse=True`; for any other
//! order that reduces the position, and for every reduce-only order, the units beyond the
//! position are not recorded, as a backtest fills or allows none of them at the fill bar
//! (`fill_bar`): the order's units shrink by them and the fill is refused for them
//! (`VenueError::WouldReverse`), after the units that closed the position are recorded.
//! So is every unit beyond the position of an order whose role only closes (`venue_roles`:
//! an exit, a leg's broker-held stop, the engine's exits), even when the position is flat
//! already: such an order was sent to close, and never opens one. (16-session-design R19
//! asks for such a fill to be refused whole; the kernel records its closing part instead,
//! so the session's books keep every unit the broker really closed, and refuses only the
//! excess, which the live layer treats as a reconciliation mismatch.)
//! The first fill of a stop order (SL, SL-M) writes `Triggered` before it, since the venue
//! filled it only once its trigger was reached, and the first fill of a leg's stop writes
//! `leg_sl_hit` before that fill (`venue_legs`). A fill is checked whole before anything of
//! it is recorded (the margin rule and the charges of each of its parts): what is recorded
//! of it is recorded all or nothing.
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
//! `advance` first. After a result the reduce-only orders are fitted to their positions
//! (`venue_cover`), the waiting exits move on (`venue_owner`): an exit waiting for this
//! order is sent once nothing blocks it, and the legs' stops follow their legs
//! (`venue_legs`): a fill that opens a leg with a stop level places its stop, one that adds
//! to it modifies the stop's units, and the fill of a target exit marks its target. A CNC
//! or NRML leg's stop is a GTT instead (`venue_gtt_sync`). A fill of a fired GTT's child
//! carries the GTT's id, and its first resolves the GTT (`GttChildFilled`); the fills of
//! an engine exit flagged `gtt_child_unfilled` or `stop_crossed` carry the flag
//! (`venue_gtt_child`, `venue_gtt_protect`).
//!
//! The contract for the live layer: every fill of an order is delivered before its
//! terminal result (`apply_cancel`, `apply_reject`); after it the order takes no more.

use crate::exec::{FillAt, FillPricing, Order, OrderAction, OrderId, OrderStatus, OrderType};
use crate::model::{Side, TsNanos};
use crate::portfolio::{FillKind, PositionKey};
use crate::rms::{ExitReason, LegRules};

use super::clock::SliceInfo;
use super::error::KernelError;
use super::event::{Event, EventStamp, OrderEventKind, Step};
use super::fills::FillRequest;
use super::futures::FillOrigin;
use super::kernel::Engine;
use super::venue_error::VenueError;
use super::venue_legs::StopEnd;

/// The step in which a backtest fills an order of this kind.
pub(super) fn result_step(order: &Order) -> Step {
    match order.request.order_type {
        OrderType::Market => Step::OpenFills,
        _ => Step::Levels,
    }
}

/// Refuse units and a price that cannot be a fill.
pub(super) fn check_fill(units: f64, price_rupees: f64) -> Result<(), VenueError> {
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

/// The units of a fill with `(closing, opening)` parts that may not be recorded: the
/// opening part of an order sent only to close (`Engine::sent_to_close`), and of any other
/// order that reduces the position without `reverse` (module docs).
fn beyond_position(reverse: bool, sent_to_close: bool, (closing, opening): (f64, f64)) -> f64 {
    let may_reverse = reverse && !sent_to_close;
    let goes_through_zero = sent_to_close || closing > 0.0;
    if goes_through_zero && !may_reverse {
        opening
    } else {
        0.0
    }
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
    /// the margin rule and the cost model, before anything is recorded. The one refusal
    /// that records something is `Venue(WouldReverse)`, for a fill beyond the position of
    /// an order that may not reverse it: the units up to the position are recorded (none
    /// when it was flat already), the order's units shrink by the rest, and an order left
    /// with nothing to fill and no fill recorded ends `Canceled`. The broker's position
    /// then differs from the session's by those units.
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
        let (closing, opening) = self.fill_parts(&placed, units);
        let closes = self.sent_to_close(&placed);
        let excess_units = beyond_position(placed.request.reverse, closes, (closing, opening));
        let parts = (closing, opening - excess_units);
        self.check_recordable(&placed, parts, price_rupees, &slice)?;
        self.venue.clock_ns = self.venue.clock_ns.max(at);
        let stamp = Self::stamp(&slice, result_step(&placed));
        if excess_units > 0.0 {
            self.shrink_order(stamp, order, excess_units, parts)?;
        }
        if parts.0 + parts.1 > 0.0 {
            let fill_at = if at == slice.open {
                FillAt::Open
            } else {
                FillAt::Level
            };
            self.trigger_before_fill(stamp, &placed)?;
            self.before_stop_fill(stamp, order);
            let fill = VenueFill {
                slice,
                stamp,
                parts,
                price_rupees,
                at: fill_at,
            };
            self.record_venue_fill(&placed, fill)?;
        }
        self.after_venue_fill(&slice, stamp, &placed)?;
        if excess_units > 0.0 {
            return Err(VenueError::WouldReverse {
                order,
                excess_units,
            }
            .into());
        }
        Ok(stamp)
    }

    /// After a fill of `placed` was recorded: a target exit marks its target, a leg's stop
    /// that filled all its units protects the rest of its leg (`venue_legs`), then the
    /// results of `after_result`.
    ///
    /// Errors: those of the leg rules, of `after_stop_end` and of `after_result`.
    fn after_venue_fill(
        &mut self,
        slice: &SliceInfo,
        stamp: EventStamp,
        placed: &Order,
    ) -> Result<(), KernelError> {
        self.after_target_exit_fill(stamp, placed)?;
        let filled = self
            .book
            .get(placed.id)
            .is_some_and(|order| order.status == OrderStatus::Filled);
        if filled {
            self.after_stop_end(self.waiting_at(slice), placed.id, StopEnd::Filled)?;
            self.after_gtt_child_end(self.waiting_at(slice), placed.id, StopEnd::Filled)?;
        }
        self.after_result(slice)
    }

    /// Take `excess_units` of a fill beyond the position off `order`'s units, before the
    /// `parts` of the fill that are recorded; an order left with nothing to fill and
    /// nothing to record ends `Canceled` (`apply_fill`).
    fn shrink_order(
        &mut self,
        stamp: EventStamp,
        order: OrderId,
        excess_units: f64,
        (closing, opening): (f64, f64),
    ) -> Result<(), KernelError> {
        let Some(shrunk) = self.book.get_mut(order) else {
            return Err(KernelError::UnknownOrder(order));
        };
        shrunk.quantity_units -= excess_units;
        let nothing_left = shrunk.quantity_units <= shrunk.filled_units;
        if nothing_left && closing + opening <= 0.0 {
            let kind = OrderEventKind::Canceled;
            self.end_order(stamp, order, OrderAction::Cancel, kind)?;
        }
        Ok(())
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
    /// opens or adds, with the margin it blocks (module docs). A part that opens a leg
    /// gives the leg its stop record (`venue_legs`).
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
            flags: self.exit_flags_of(order.id),
            margin_rupees: 0.0,
            leg_rules: LegRules::default(),
            target_index: self.target_of(order.id),
            pricing: FillPricing::Resting,
            gtt: self.child_gtt_of(order.id),
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
            if self.execute_fill(fill.stamp, open)?.kind == FillKind::Entry {
                self.open_leg_stop(key);
            }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_fill_is_recorded_beyond_the_position_only_for_an_order_that_may_open() {
        // (what, reverse, sent only to close, (closing, opening) units, units refused)
        let table = [
            ("an entry from flat", false, false, (0.0, 100.0), 0.0),
            ("a reversal", true, false, (40.0, 60.0), 0.0),
            (
                "a plain order through zero",
                false,
                false,
                (40.0, 60.0),
                60.0,
            ),
            (
                "an exit within the position",
                false,
                true,
                (100.0, 0.0),
                0.0,
            ),
            ("an exit through zero", false, true, (40.0, 60.0), 60.0),
            ("an exit from flat", false, true, (0.0, 100.0), 100.0),
            (
                "an exit with reverse from flat",
                true,
                true,
                (0.0, 100.0),
                100.0,
            ),
        ];
        for (what, reverse, closes, parts, refused) in table {
            assert_eq!(beyond_position(reverse, closes, parts), refused, "{what}");
        }
    }
}
