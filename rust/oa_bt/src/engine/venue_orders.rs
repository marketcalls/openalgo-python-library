//! Commands out of an external session (`venue`): the units an order is sent with, the
//! order sent at once or made its position's exit, a modification sent and held until the
//! venue answers, and a cancellation, which the engine's own withdrawals send at most once
//! in a slice (`cancel` sends it again at once after a refusal: `venue_session`). A
//! strategy's GTT order rests as its GTT (`venue_gtts`): nothing of the order is placed,
//! and withdrawing it asks for its GTT's cancellation until the venue fires it.

use crate::exec::{OrderAction, OrderId, OrderRequest, OrderStatus, QuantitySpec, Validity};
use crate::model::Side;
use crate::portfolio::PositionKey;
use crate::sizing::round_down_to_venue;

use super::error::KernelError;
use super::event::{EventStamp, OrderEventKind};
use super::kernel::Engine;
use super::venue::{Replacement, SendAt, VenueCommand};
use super::venue_roles::OrderRole;

impl Engine {
    /// The units an order is sent with: in a backtest `units` as resolved; in an external
    /// session an order that only reduces gets at most the position held now, and a
    /// fraction of it is taken of that position, rounded down to the venue quantity. An
    /// order that would reduce nothing gets 0.
    pub(super) fn venue_units(&self, request: &OrderRequest, units: f64) -> f64 {
        if !self.is_external() || !request.reduce_only {
            return units;
        }
        let key = PositionKey {
            instrument: request.instrument,
            product: request.product,
        };
        let held = self.held_units(key);
        let reduces = held != 0.0 && (held > 0.0) != (request.side == Side::Buy);
        if !reduces {
            return 0.0;
        }
        let wanted = match request.quantity {
            QuantitySpec::PositionFraction(fraction) => {
                let lot = self
                    .instruments
                    .get(request.instrument.index())
                    .map_or(1, |spec| spec.lot_size_units);
                round_down_to_venue(held.abs() * fraction, lot, false)
            }
            _ => units,
        };
        wanted.min(held.abs())
    }

    /// In an external session, after an order has passed the checks (`at` is its
    /// decision): its role is recorded (`venue_roles`), an order the exit owner sends (an
    /// exit, a plain MARKET order against a position whose leg's stop rests:
    /// `Engine::owner_sends`) becomes the latest of its position's exits (`venue_owner`),
    /// any other order is sent at once. An order with no units ends `Canceled` and sends
    /// nothing. Then every waiting exit moves on, so it also cancels an order just placed
    /// on its closing side. A backtest sends nothing.
    ///
    /// Errors: those of `fills::end_order` and of the exits.
    pub(super) fn send_order(&mut self, at: SendAt, id: OrderId) -> Result<(), KernelError> {
        if !self.is_external() {
            return Ok(());
        }
        let order = self.book.get(id).ok_or(KernelError::UnknownOrder(id))?;
        let exit = self.owner_sends(&order.request);
        if order.quantity_units <= 0.0 {
            let kind = OrderEventKind::Canceled;
            return self.end_order(at.stamp(), id, OrderAction::Cancel, kind);
        }
        let role = self.submission_role(&order.request, order.quantity_units, Some(id));
        if order.request.validity == Validity::Gtt {
            // It rests as its GTT, whose place `place_order_gtt` has sent.
            self.record_role(id, role);
            return self.progress_exits(at);
        }
        if exit {
            self.record_role(id, role);
            self.own_strategy_exit(at, id)?;
        } else {
            self.place(id, role);
        }
        self.progress_exits(at)
    }

    /// Send an order to the venue with its units and type as they stand, for `role`
    /// (recorded, `venue_roles`).
    pub(super) fn place(&mut self, id: OrderId, role: OrderRole) {
        self.record_role(id, role);
        if let Some(order) = self.book.get(id) {
            let command = VenueCommand::Place {
                order: id,
                instrument: order.request.instrument,
                product: order.request.product,
                side: order.request.side,
                units: order.quantity_units,
                order_type: order.request.order_type,
                role,
            };
            self.venue.commands.push(command);
        }
    }

    /// In an external session, a modification that passed the checks: sent to the venue,
    /// and held until the venue confirms it (`apply_modify`) or refuses it; the order
    /// keeps its old terms until then.
    pub(super) fn send_modify(&mut self, id: OrderId, replacement: Replacement) {
        let command = VenueCommand::Modify {
            order: id,
            units: replacement.units,
            order_type: replacement.request.order_type,
        };
        self.venue.commands.push(command);
        self.venue.modifying.push((id, replacement));
    }

    /// Withdraw a working order. A backtest ends it `Canceled` now. An external session
    /// sends `Cancel` and leaves the order working until the venue's result: once while
    /// it waits for that result, and at most once in a slice, so one the venue refused
    /// (`apply_cancel_refused`) goes again in a later slice (`stamp.slice` is the slice
    /// in progress), or at once on a `cancel` (`cancel_on_request` forgets the slice it
    /// was sent in). An exit not sent yet has nothing at the venue and ends `Canceled`
    /// now. Returns the order's state afterwards.
    ///
    /// Errors: `UnknownOrder`; those of `fills::end_order`.
    pub(super) fn withdraw_order(
        &mut self,
        stamp: EventStamp,
        id: OrderId,
    ) -> Result<OrderStatus, KernelError> {
        if !self.is_external() {
            self.end_order(stamp, id, OrderAction::Cancel, OrderEventKind::Canceled)?;
            return Ok(OrderStatus::Canceled);
        }
        let status = self
            .book
            .get(id)
            .ok_or(KernelError::UnknownOrder(id))?
            .status;
        if !status.is_working() {
            return Ok(status);
        }
        if self.is_unsent(id) {
            self.end_order(stamp, id, OrderAction::Cancel, OrderEventKind::Canceled)?;
            self.tidy_venue();
            return Ok(OrderStatus::Canceled);
        }
        if let Some(gtt) = self.resting_gtt_of(id) {
            self.ask_cancel_gtt(stamp.slice, gtt)?;
            return Ok(status);
        }
        let sent_this_slice = self.venue.cancels_sent.contains(&(id, stamp.slice));
        if self.venue.cancelling.contains(&id) || sent_this_slice {
            return Ok(status);
        }
        self.venue.cancelling.push(id);
        self.venue.cancels_sent.retain(|(known, _)| *known != id);
        self.venue.cancels_sent.push((id, stamp.slice));
        self.venue.commands.push(VenueCommand::Cancel { order: id });
        Ok(status)
    }
}
