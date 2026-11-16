//! The exit owner of a position in an external session (`venue`): cancel, then exit.
//!
//! An exit is the strategy's `exit()` or `reduce()` (a reduce-only MARKET order) or one
//! of the engine's exits (`venue_exits`). No working order counts as covering the
//! position, so an exit first sends `Cancel` for every working order of its position on
//! the closing side, and is sent once they have all ended (cancelled, or filled), for the
//! position then open: nothing when they filled it. While it waits it is the position's
//! one exit owner, and it stays so until the orders sent for it have ended: a strategy
//! exit of a position that has an owner ends `Canceled` at once, and the engine makes no
//! second exit of it. An order placed on the closing side while an exit waits is
//! cancelled too.
//!
//! When it is sent, the strategy's exit takes the units it asks for (a fraction of the
//! position then open, rounded down to the venue quantity, or its units) up to the
//! position and the freeze quantity, on the order the strategy holds. The engine's exit
//! closes the whole position in orders of at most the freeze quantity each, separate
//! orders sent one after another (never one split call). An exit due when the market is
//! closed is not sent: the strategy's ends `Rejected` with `market_closed`.

use crate::exec::state::next_status;
use crate::exec::{
    Order, OrderAction, OrderId, OrderRequest, OrderStatus, QuantitySpec, RejectCode,
};
use crate::model::Side;
use crate::portfolio::PositionKey;
use crate::rms::ExitReason;
use crate::sizing::{cap_at_freeze, round_down_to_venue};

use super::error::KernelError;
use super::event::{Event, OrderEventKind};
use super::kernel::Engine;
use super::venue::{ExitBy, ExitOwner, SendAt};

/// The side that closes a position of `held` signed units.
fn closing_side(held: f64) -> Side {
    if held > 0.0 {
        Side::Sell
    } else {
        Side::Buy
    }
}

impl Engine {
    /// The engine's exit of `key` for `reason`, decided as `at` says: the position's exit
    /// owner, unless it is flat or has one already (module docs).
    ///
    /// Errors: those of `progress_exits`.
    pub(super) fn request_engine_exit(
        &mut self,
        at: SendAt,
        key: PositionKey,
        reason: ExitReason,
    ) -> Result<(), KernelError> {
        if self.held_units(key) == 0.0 || self.exit_owner(key).is_some() {
            return Ok(());
        }
        self.venue.exits.push(ExitOwner {
            key,
            by: ExitBy::Engine(reason),
            sent: Vec::new(),
        });
        self.progress_exits(at)
    }

    /// Move every waiting exit owner on (module docs): cancel what blocks it, or send it
    /// once nothing does; then forget the owners whose orders have all ended.
    ///
    /// Errors: those of the order book and of `fills::end_order`.
    pub(super) fn progress_exits(&mut self, at: SendAt) -> Result<(), KernelError> {
        self.tidy_venue();
        let waiting: Vec<ExitOwner> = self
            .venue
            .exits
            .iter()
            .filter(|owner| owner.sent.is_empty())
            .cloned()
            .collect();
        for owner in waiting {
            self.progress_exit(at, &owner)?;
        }
        self.tidy_venue();
        Ok(())
    }

    fn progress_exit(&mut self, at: SendAt, owner: &ExitOwner) -> Result<(), KernelError> {
        let held = self.held_units(owner.key);
        if held == 0.0 {
            return self.drop_exit(at, owner, OrderEventKind::Canceled);
        }
        let own = match owner.by {
            ExitBy::Strategy(id) => Some(id),
            ExitBy::Engine(_) => None,
        };
        let closing = closing_side(held);
        let blocking: Vec<OrderId> = self
            .book
            .working()
            .filter(|order| {
                Some(order.id) != own
                    && order.request.instrument == owner.key.instrument
                    && order.request.product == owner.key.product
                    && order.request.side == closing
            })
            .map(|order| order.id)
            .collect();
        if !blocking.is_empty() {
            for id in blocking {
                self.withdraw_order(at.stamp(), id)?;
            }
            return Ok(());
        }
        match owner.by {
            ExitBy::Strategy(id) => self.send_strategy_exit(at, owner, id, held),
            ExitBy::Engine(reason) => self.send_engine_exit(at, owner, reason, held),
        }
    }

    /// Give up an exit: the strategy's order ends as `kind` says; the owner is forgotten.
    fn drop_exit(
        &mut self,
        at: SendAt,
        owner: &ExitOwner,
        kind: OrderEventKind,
    ) -> Result<(), KernelError> {
        self.venue.exits.retain(|known| known.key != owner.key);
        let ExitBy::Strategy(id) = owner.by else {
            return Ok(());
        };
        let action = match kind {
            OrderEventKind::Rejected(_) => OrderAction::Reject,
            _ => OrderAction::Cancel,
        };
        self.end_order(at.stamp(), id, action, kind)
    }

    fn set_sent(&mut self, key: PositionKey, sent: Vec<OrderId>) {
        if let Some(owner) = self.venue.exits.iter_mut().find(|o| o.key == key) {
            owner.sent = sent;
        }
    }

    /// The largest piece of `units` an order may carry: the freeze quantity in whole
    /// lots, or all of it when the instrument has none.
    fn piece_of(&self, key: PositionKey, units: f64) -> Result<f64, KernelError> {
        let spec = self.spec(key.instrument)?;
        let piece = cap_at_freeze(units, spec.freeze_quantity_units, spec.lot_size_units);
        Ok(if piece > 0.0 { piece } else { units })
    }

    fn send_strategy_exit(
        &mut self,
        at: SendAt,
        owner: &ExitOwner,
        id: OrderId,
        held: f64,
    ) -> Result<(), KernelError> {
        let Some(order) = self.book.get(id).cloned() else {
            return Err(KernelError::UnknownOrder(id));
        };
        if order.request.side != closing_side(held) {
            return self.drop_exit(at, owner, OrderEventKind::Canceled);
        }
        let wanted = match order.request.quantity {
            QuantitySpec::PositionFraction(fraction) => {
                let lot = self.spec(owner.key.instrument)?.lot_size_units;
                round_down_to_venue(held.abs() * fraction, lot, false)
            }
            _ => order.quantity_units,
        };
        let units = self.piece_of(owner.key, wanted.min(held.abs()))?;
        if units <= 0.0 {
            return self.drop_exit(at, owner, OrderEventKind::Canceled);
        }
        if self.day_validity(&at.slice, at.decided_at).is_err() {
            let kind = OrderEventKind::Rejected(RejectCode::MarketClosed);
            return self.drop_exit(at, owner, kind);
        }
        if let Some(sending) = self.book.get_mut(id) {
            sending.quantity_units = units;
        }
        self.place(id);
        self.set_sent(owner.key, vec![id]);
        Ok(())
    }

    fn send_engine_exit(
        &mut self,
        at: SendAt,
        owner: &ExitOwner,
        reason: ExitReason,
        held: f64,
    ) -> Result<(), KernelError> {
        // A DAY order needs a session left to fill in; at the session's last close there
        // is none, and the position stays open (`venue_exits`).
        let good_until = match self.day_validity(&at.slice, at.decided_at) {
            Ok(good_until) => good_until,
            Err(_market_closed) => return self.drop_exit(at, owner, OrderEventKind::Canceled),
        };
        let mut left = held.abs();
        let mut sent = Vec::new();
        while left > 0.0 {
            let units = self.piece_of(owner.key, left)?;
            let request = OrderRequest {
                product: owner.key.product,
                reduce_only: true,
                exit_reason: Some(reason),
                ..OrderRequest::market(
                    owner.key.instrument,
                    closing_side(held),
                    QuantitySpec::Units(units),
                )
            };
            let id = self.book.next_id();
            self.book.insert(Order {
                id,
                request,
                status: next_status(OrderStatus::Initialized, OrderAction::PassChecks)?,
                decided_at: at.decided_at,
                good_until,
                quantity_units: units,
                filled_units: 0.0,
                average_fill_rupees: 0.0,
                reject_code: None,
            })?;
            self.emit(Event::Order {
                stamp: at.stamp(),
                order: id,
                kind: OrderEventKind::Submitted,
            });
            self.place(id);
            sent.push(id);
            left -= units;
        }
        self.set_sent(owner.key, sent);
        Ok(())
    }
}
