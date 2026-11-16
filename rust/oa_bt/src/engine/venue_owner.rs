//! The exit owner of a position in an external session (`venue`): cancel, then exit.
//!
//! An exit is the strategy's `exit()` or `reduce()` (a reduce-only MARKET order) or one
//! of the engine's exits (`venue_exits`). No working order counts as covering the
//! position, so an exit first sends `Cancel` for every working order of its position on
//! the closing side, and is sent once they have all ended (cancelled, or filled), for the
//! position then open: nothing when they filled it. While it waits it is the position's
//! one exit owner, and it stays so until the orders sent for it have ended; an order
//! placed on the closing side while an exit waits is cancelled too.
//!
//! A later strategy exit of the position takes the owner's place (`reduce()` then
//! `exit()` in one bar, say): an earlier exit still waiting ends `Canceled`, as nothing
//! of it was sent; one already sent is a working order on the closing side, so it is
//! cancelled (or fills) first. The later exit then closes the larger of what the two ask
//! for, of the position then open, so the position is closed as a backtest closes it. A
//! strategy exit of a position the engine's exit owns ends `Canceled` (that exit closes
//! the whole position), and the engine makes no second exit of a position that has an
//! owner.
//!
//! When it is sent, the strategy's exit takes the units it asks for (a fraction of the
//! position then open, rounded down to the venue quantity, or its units) up to the
//! position; the engine's exit takes the whole position. Either goes out in orders of at
//! most the freeze quantity each, separate orders sent one after another (never one split
//! call): the strategy's own order carries the first, the engine makes the others with
//! its exit reason. An exit due when the market is closed is not sent: the strategy's
//! ends `Rejected` with `market_closed`.

use crate::exec::state::next_status;
use crate::exec::{
    Order, OrderAction, OrderId, OrderRequest, OrderStatus, QuantitySpec, RejectCode,
};
use crate::model::{Side, TsNanos};
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
    /// The strategy's exit `id`, decided as `at` says: the position's exit owner, in the
    /// place of an earlier strategy exit (module docs); `Canceled` when the engine's exit
    /// owns the position.
    ///
    /// Errors: those of `fills::end_order`.
    pub(super) fn own_strategy_exit(&mut self, at: SendAt, id: OrderId) -> Result<(), KernelError> {
        let order = self.book.get(id).ok_or(KernelError::UnknownOrder(id))?;
        let key = PositionKey {
            instrument: order.request.instrument,
            product: order.request.product,
        };
        let mut superseded = Vec::new();
        if let Some(owner) = self.exit_owner(key).cloned() {
            let ExitBy::Strategy(earlier) = owner.by else {
                let kind = OrderEventKind::Canceled;
                return self.end_order(at.stamp(), id, OrderAction::Cancel, kind);
            };
            superseded = owner.superseded;
            superseded.push(earlier);
            self.venue.exits.retain(|known| known.key != key);
            if owner.sent.is_empty() {
                let kind = OrderEventKind::Canceled;
                self.end_order(at.stamp(), earlier, OrderAction::Cancel, kind)?;
            }
        }
        self.venue.exits.push(ExitOwner {
            key,
            by: ExitBy::Strategy(id),
            sent: Vec::new(),
            superseded,
        });
        Ok(())
    }

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
            superseded: Vec::new(),
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

    /// `units` in pieces of at most the freeze quantity in whole lots (all of it in one
    /// when the instrument has none), in order.
    fn pieces_of(&self, key: PositionKey, units: f64) -> Result<Vec<f64>, KernelError> {
        let spec = self.spec(key.instrument)?;
        let mut pieces = Vec::new();
        let mut left = units;
        while left > 0.0 {
            let piece = cap_at_freeze(left, spec.freeze_quantity_units, spec.lot_size_units);
            let piece = if piece > 0.0 { piece } else { left };
            pieces.push(piece);
            left -= piece;
        }
        Ok(pieces)
    }

    /// What a strategy exit asks for of a position of `held` signed units: a fraction of
    /// it rounded down to the venue quantity, or its units.
    fn exit_wanted(&self, order: &Order, held: f64) -> Result<f64, KernelError> {
        Ok(match order.request.quantity {
            QuantitySpec::PositionFraction(fraction) => {
                let lot = self.spec(order.request.instrument)?.lot_size_units;
                round_down_to_venue(held.abs() * fraction, lot, false)
            }
            _ => order.quantity_units,
        })
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
        let mut wanted = self.exit_wanted(&order, held)?;
        for earlier in &owner.superseded {
            if let Some(earlier) = self.book.get(*earlier) {
                wanted = wanted.max(self.exit_wanted(earlier, held)?);
            }
        }
        let units = wanted.min(held.abs());
        if units <= 0.0 {
            return self.drop_exit(at, owner, OrderEventKind::Canceled);
        }
        if self.day_validity(&at.slice, at.decided_at).is_err() {
            let kind = OrderEventKind::Rejected(RejectCode::MarketClosed);
            return self.drop_exit(at, owner, kind);
        }
        let pieces = self.pieces_of(owner.key, units)?;
        let (first, rest) = pieces
            .split_first()
            .map_or((units, &[][..]), |(f, r)| (*f, r));
        if let Some(sending) = self.book.get_mut(id) {
            sending.quantity_units = first;
        }
        self.place(id);
        let mut sent = vec![id];
        let terms = (
            closing_side(held),
            order.request.exit_reason,
            order.good_until,
        );
        for units in rest {
            sent.push(self.make_exit_order(at, owner.key, terms, *units)?);
        }
        self.set_sent(owner.key, sent);
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
        let terms = (closing_side(held), Some(reason), good_until);
        let mut sent = Vec::new();
        for units in self.pieces_of(owner.key, held.abs())? {
            sent.push(self.make_exit_order(at, owner.key, terms, units)?);
        }
        self.set_sent(owner.key, sent);
        Ok(())
    }

    /// Make and send one of the engine's exit orders of `key`: a reduce-only MARKET DAY
    /// order of `units` with `(side, exit reason, good until)`, decided and stamped as
    /// `at` says.
    ///
    /// Errors: those of the order book.
    fn make_exit_order(
        &mut self,
        at: SendAt,
        key: PositionKey,
        (side, exit_reason, good_until): (Side, Option<ExitReason>, TsNanos),
        units: f64,
    ) -> Result<OrderId, KernelError> {
        let request = OrderRequest {
            product: key.product,
            reduce_only: true,
            exit_reason,
            ..OrderRequest::market(key.instrument, side, QuantitySpec::Units(units))
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
        Ok(id)
    }
}
