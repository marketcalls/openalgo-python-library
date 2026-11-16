//! The exits of a position in an external session (`venue`): cancel, then exit.
//!
//! An exit is the strategy's `exit()` or `reduce()` (a reduce-only MARKET order) or one
//! of the engine's exits (`venue_exits`). No working order counts as covering the
//! position, so an exit first sends `Cancel` for every working order of its position on
//! the closing side but the position's exits and the orders sent for them, and is sent
//! once they have all ended (cancelled, or filled), for the position then open: nothing
//! when they filled it. An order placed on the closing side while an exit waits is
//! cancelled too.
//!
//! The strategy's exits of a position add up, as a backtest fills them one after another
//! at the next open: each is sized when it is sent, on the position less what the earlier
//! exits still have to fill (what they filled has already left it). So two `reduce(0.5)`
//! close three quarters of it, `reduce()` of 50 units then of 10 close 60, and `reduce()`
//! then `exit()` close it all. A later exit is sent after an earlier one that waits, and
//! one with nothing left to close ends `Canceled`.
//!
//! The engine's exit takes over from the strategy's: their exits not sent yet end
//! `Canceled`, those sent are cancelled (or fill) first like any order on the closing
//! side, and the engine's exit then closes the whole position. A strategy exit of a
//! position the engine's exit holds ends `Canceled`, and the engine makes no second exit
//! of a position it holds.
//!
//! When it is sent, a strategy exit takes the units it asks for (a fraction of the
//! position it is sized on, rounded down to the venue quantity, or its units), at most
//! that position; the engine's exit takes the whole position. Either goes out in orders of
//! at most the freeze quantity each, separate orders sent one after another (never one
//! split call): the strategy's own order carries the first, so its units are what the
//! venue holds for it, and the engine makes the others with its exit reason. What a
//! strategy exit asks for is read from its request, never from its order's units. An exit
//! due when the market is closed is not sent: the strategy's ends `Rejected` with
//! `market_closed`.

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
use super::venue::{ExitBy, PositionExit, SendAt};

/// The side that closes a position of `held` signed units.
pub(super) fn closing_side(held: f64) -> Side {
    if held > 0.0 {
        Side::Sell
    } else {
        Side::Buy
    }
}

impl Engine {
    /// The strategy's exit `id`, decided as `at` says: the latest of its position's exits
    /// (module docs); `Canceled` when the engine's exit holds the position.
    ///
    /// Errors: those of `fills::end_order`.
    pub(super) fn own_strategy_exit(&mut self, at: SendAt, id: OrderId) -> Result<(), KernelError> {
        let order = self.book.get(id).ok_or(KernelError::UnknownOrder(id))?;
        let key = PositionKey {
            instrument: order.request.instrument,
            product: order.request.product,
        };
        if self.has_engine_exit(key) {
            let kind = OrderEventKind::Canceled;
            return self.end_order(at.stamp(), id, OrderAction::Cancel, kind);
        }
        self.venue.exits.push(PositionExit {
            key,
            by: ExitBy::Strategy(id),
            sent: Vec::new(),
        });
        Ok(())
    }

    /// The engine's exit of `key` for `reason`, decided as `at` says: it takes over from
    /// the strategy's exits of the position (module docs), unless the position is flat or
    /// the engine's exit holds it already.
    ///
    /// Errors: those of `fills::end_order` and of `progress_exits`.
    pub(super) fn request_engine_exit(
        &mut self,
        at: SendAt,
        key: PositionKey,
        reason: ExitReason,
    ) -> Result<(), KernelError> {
        if self.held_units(key) == 0.0 || self.has_engine_exit(key) {
            return Ok(());
        }
        let unsent: Vec<OrderId> = self
            .venue
            .exits
            .iter()
            .filter(|exit| exit.key == key && exit.sent.is_empty())
            .filter_map(|exit| match exit.by {
                ExitBy::Strategy(id) => Some(id),
                ExitBy::Engine(_) => None,
            })
            .collect();
        self.venue.exits.retain(|exit| exit.key != key);
        for id in unsent {
            let kind = OrderEventKind::Canceled;
            self.end_order(at.stamp(), id, OrderAction::Cancel, kind)?;
        }
        self.venue.exits.push(PositionExit {
            key,
            by: ExitBy::Engine(reason),
            sent: Vec::new(),
        });
        self.progress_exits(at)
    }

    /// Move every waiting exit on, in the order they were made (module docs): cancel what
    /// blocks it, or send it once nothing does; then forget the exits whose orders have
    /// all ended.
    ///
    /// Errors: those of the order book and of `fills::end_order`.
    pub(super) fn progress_exits(&mut self, at: SendAt) -> Result<(), KernelError> {
        self.tidy_venue();
        let waiting: Vec<PositionExit> = self
            .venue
            .exits
            .iter()
            .filter(|exit| exit.sent.is_empty())
            .cloned()
            .collect();
        for exit in waiting {
            self.progress_exit(at, &exit)?;
        }
        self.tidy_venue();
        Ok(())
    }

    fn progress_exit(&mut self, at: SendAt, exit: &PositionExit) -> Result<(), KernelError> {
        let held = self.held_units(exit.key);
        if held == 0.0 {
            return self.drop_exit(at, exit, OrderEventKind::Canceled);
        }
        let blocking = self.blocking(exit.key, closing_side(held));
        if !blocking.is_empty() {
            for id in blocking {
                self.withdraw_order(at.stamp(), id)?;
            }
            return Ok(());
        }
        match exit.by {
            ExitBy::Strategy(id) => self.send_strategy_exit(at, exit, id, held),
            ExitBy::Engine(reason) => self.send_engine_exit(at, exit, reason, held),
        }
    }

    /// The working orders of `key` on `side` that an exit of it cancels first: all but
    /// the position's exits and the orders sent for them.
    fn blocking(&self, key: PositionKey, side: Side) -> Vec<OrderId> {
        let exits: Vec<OrderId> = self
            .venue
            .exits
            .iter()
            .filter(|exit| exit.key == key)
            .flat_map(|exit| {
                let own = match exit.by {
                    ExitBy::Strategy(id) => Some(id),
                    ExitBy::Engine(_) => None,
                };
                own.into_iter().chain(exit.sent.iter().copied())
            })
            .collect();
        self.book
            .working()
            .filter(|order| {
                order.request.instrument == key.instrument
                    && order.request.product == key.product
                    && order.request.side == side
                    && !exits.contains(&order.id)
            })
            .map(|order| order.id)
            .collect()
    }

    /// Give up an exit: the strategy's order ends as `kind` says; the exit is forgotten.
    fn drop_exit(
        &mut self,
        at: SendAt,
        exit: &PositionExit,
        kind: OrderEventKind,
    ) -> Result<(), KernelError> {
        self.venue
            .exits
            .retain(|known| !(known.key == exit.key && known.by == exit.by));
        let ExitBy::Strategy(id) = exit.by else {
            return Ok(());
        };
        let action = match kind {
            OrderEventKind::Rejected(_) => OrderAction::Reject,
            _ => OrderAction::Cancel,
        };
        self.end_order(at.stamp(), id, action, kind)
    }

    fn set_sent(&mut self, exit: &PositionExit, sent: Vec<OrderId>) {
        let known = self
            .venue
            .exits
            .iter_mut()
            .find(|known| known.key == exit.key && known.by == exit.by);
        if let Some(known) = known {
            known.sent = sent;
        }
    }

    /// The units the orders sent for the exits of `key` still have to fill.
    fn left_to_exit(&self, key: PositionKey) -> f64 {
        self.venue
            .exits
            .iter()
            .filter(|exit| exit.key == key)
            .flat_map(|exit| exit.sent.iter())
            .filter_map(|id| self.book.get(*id))
            .map(|order| self.units_to_fill(order))
            .sum()
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

    /// What a strategy exit asks for of a position of `open` units (above 0): a fraction
    /// of it rounded down to the venue quantity, or the units of its request.
    fn exit_wanted(&self, order: &Order, open: f64) -> Result<f64, KernelError> {
        let lot = self.spec(order.request.instrument)?.lot_size_units;
        Ok(match order.request.quantity {
            QuantitySpec::PositionFraction(fraction) => {
                round_down_to_venue(open * fraction, lot, false)
            }
            QuantitySpec::Units(units) => units,
            QuantitySpec::Lots(lots) => f64::from(lots) * f64::from(lot),
            // A sizer's units were resolved at submission, and the order not sent since.
            QuantitySpec::Sizer(_) => order.quantity_units,
        })
    }

    fn send_strategy_exit(
        &mut self,
        at: SendAt,
        exit: &PositionExit,
        id: OrderId,
        held: f64,
    ) -> Result<(), KernelError> {
        let Some(order) = self.book.get(id).cloned() else {
            return Err(KernelError::UnknownOrder(id));
        };
        if order.request.side != closing_side(held) {
            return self.drop_exit(at, exit, OrderEventKind::Canceled);
        }
        let open = held.abs() - self.left_to_exit(exit.key);
        let units = if open > 0.0 {
            self.exit_wanted(&order, open)?.min(open)
        } else {
            0.0
        };
        if units <= 0.0 {
            return self.drop_exit(at, exit, OrderEventKind::Canceled);
        }
        if self.day_validity(&at.slice, at.decided_at).is_err() {
            let kind = OrderEventKind::Rejected(RejectCode::MarketClosed);
            return self.drop_exit(at, exit, kind);
        }
        let pieces = self.pieces_of(exit.key, units)?;
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
            sent.push(self.make_exit_order(at, exit.key, terms, *units)?);
        }
        self.set_sent(exit, sent);
        Ok(())
    }

    fn send_engine_exit(
        &mut self,
        at: SendAt,
        exit: &PositionExit,
        reason: ExitReason,
        held: f64,
    ) -> Result<(), KernelError> {
        // A DAY order needs a session left to fill in; at the session's last close there
        // is none, and the position stays open (`venue_exits`).
        let good_until = match self.day_validity(&at.slice, at.decided_at) {
            Ok(good_until) => good_until,
            Err(_market_closed) => return self.drop_exit(at, exit, OrderEventKind::Canceled),
        };
        let terms = (closing_side(held), Some(reason), good_until);
        let mut sent = Vec::new();
        for units in self.pieces_of(exit.key, held.abs())? {
            sent.push(self.make_exit_order(at, exit.key, terms, units)?);
        }
        self.set_sent(exit, sent);
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
