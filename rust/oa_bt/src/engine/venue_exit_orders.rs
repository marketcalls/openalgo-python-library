//! The orders an exit of a position goes out in (`venue_owner`): what a strategy exit asks
//! for, sized when it is sent on the position less what the earlier exits still have to
//! fill; the engine's exit for the whole position, at market or at the session's limit;
//! both in orders of at most the freeze quantity, separate orders sent one after another
//! (never one split call), the strategy's own order carrying the first piece so its units
//! are what the venue holds for it.

use crate::exec::state::next_status;
use crate::exec::{
    FillFlags, Order, OrderAction, OrderId, OrderRequest, OrderStatus, OrderType, QuantitySpec,
    RejectCode,
};
use crate::model::{Side, TsNanos};
use crate::portfolio::PositionKey;
use crate::rms::ExitReason;
use crate::sizing::{cap_at_freeze, round_down_to_venue};

use super::error::KernelError;
use super::event::{Event, OrderEventKind};
use super::kernel::Engine;
use super::venue::{ExitBy, PositionExit, SendAt};
use super::venue_owner::closing_side;
use super::venue_roles::{EngineExit, OrderRole};

/// `units` in pieces of at most `freeze_units` in whole lots of `lot_units`, in order:
/// all of it in one when there is no freeze quantity, or when it is below one lot.
fn freeze_pieces(units: f64, freeze_units: Option<u32>, lot_units: u32) -> Vec<f64> {
    let mut pieces = Vec::new();
    let mut left = units;
    while left > 0.0 {
        let piece = cap_at_freeze(left, freeze_units, lot_units);
        let piece = if piece > 0.0 { piece } else { left };
        pieces.push(piece);
        left -= piece;
    }
    pieces
}

/// The terms of the engine's exit orders of one exit: closing side, exit reason, validity
/// and order type (MARKET, or the session's LIMIT).
#[derive(Clone, Copy, Debug, PartialEq)]
struct ExitTerms {
    side: Side,
    exit_reason: Option<ExitReason>,
    good_until: TsNanos,
    order_type: OrderType,
}

impl Engine {
    /// The units a waiting strategy exit asks for when that is part of the `open` units
    /// left to exit (above 0 and below them); `None` for an engine exit, a full exit and
    /// an exit with nothing to close.
    ///
    /// Errors: `UnknownOrder`; those of the instrument's spec.
    pub(super) fn partial_exit_units(
        &self,
        exit: &PositionExit,
        open: f64,
    ) -> Result<Option<f64>, KernelError> {
        let ExitBy::Strategy(id) = exit.by else {
            return Ok(None);
        };
        let order = self.book.get(id).ok_or(KernelError::UnknownOrder(id))?;
        if open <= 0.0 {
            return Ok(None);
        }
        let units = self.exit_wanted(order, open)?.min(open);
        Ok((units > 0.0 && units < open).then_some(units))
    }

    /// True when the exits of `key` that wait, one after another as a backtest fills them,
    /// ask for all of the `open` units left to exit: an engine exit, or strategy exits
    /// whose units add up to them (`venue_owner`).
    ///
    /// Errors: `UnknownOrder`; those of the instrument's spec.
    pub(super) fn waiting_exits_close_all(
        &self,
        key: PositionKey,
        open: f64,
    ) -> Result<bool, KernelError> {
        let mut left = open;
        let waiting = self
            .venue
            .exits
            .iter()
            .filter(|exit| exit.key == key && exit.sent.is_empty());
        for exit in waiting {
            if left <= 0.0 {
                break;
            }
            let ExitBy::Strategy(id) = exit.by else {
                return Ok(true);
            };
            let order = self.book.get(id).ok_or(KernelError::UnknownOrder(id))?;
            left -= self.exit_wanted(order, left)?.min(left);
        }
        Ok(left <= 0.0)
    }

    /// `units` of `key`'s instrument in pieces of at most its freeze quantity
    /// (`freeze_pieces`). The entry gates keep a position the session builds within one
    /// piece (`venue_gates`).
    fn pieces_of(&self, key: PositionKey, units: f64) -> Result<Vec<f64>, KernelError> {
        let spec = self.spec(key.instrument)?;
        let (freeze, lot) = (spec.freeze_quantity_units, spec.lot_size_units);
        Ok(freeze_pieces(units, freeze, lot))
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

    /// Send the strategy's exit `id` of `exit`'s position of `held` signed units: sized on
    /// what the earlier exits leave, dropped when nothing is left or it no longer closes,
    /// `market_closed` when no session is left. A reversal the owner held while the leg's
    /// stop was cancelled for it (`venue_roles`) goes whole, as an entry.
    ///
    /// Errors: `UnknownOrder`; those of the order book and of `fills::end_order`.
    pub(super) fn send_strategy_exit(
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
        if let Some(role) = self.order_role(id).filter(|role| role.opens()) {
            if self.day_validity(&at.slice, at.decided_at).is_err() {
                let kind = OrderEventKind::Rejected(RejectCode::MarketClosed);
                return self.drop_exit(at, exit, kind);
            }
            self.place(id, role);
            self.set_sent(exit, vec![id]);
            return Ok(());
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
        self.place(id, OrderRole::Exit);
        let mut sent = vec![id];
        let terms = ExitTerms {
            side: closing_side(held),
            exit_reason: order.request.exit_reason,
            good_until: order.good_until,
            order_type: OrderType::Market,
        };
        for units in rest {
            sent.push(self.make_exit_order(at, exit.key, terms, *units, OrderRole::Exit)?);
        }
        self.set_sent(exit, sent);
        Ok(())
    }

    /// Send the engine's exit `by` of `exit`'s position of `held` signed units, the whole
    /// of it, at market or at the exit's limit, its orders' fills flagged as the exit is;
    /// dropped when no session is left.
    ///
    /// Errors: those of the order book and of `fills::end_order`.
    pub(super) fn send_engine_exit(
        &mut self,
        at: SendAt,
        exit: &PositionExit,
        by: EngineExit,
        held: f64,
    ) -> Result<(), KernelError> {
        // A DAY order needs a session left to fill in; at the session's last close there
        // is none, and the position stays open (`venue_exits`).
        let good_until = match self.day_validity(&at.slice, at.decided_at) {
            Ok(good_until) => good_until,
            Err(_market_closed) => return self.drop_exit(at, exit, OrderEventKind::Canceled),
        };
        let order_type =
            exit.limit_rupees
                .map_or(OrderType::Market, |limit_rupees| OrderType::Limit {
                    limit_rupees,
                });
        let terms = ExitTerms {
            side: closing_side(held),
            exit_reason: Some(by.exit_reason()),
            good_until,
            order_type,
        };
        let mut sent = Vec::new();
        for units in self.pieces_of(exit.key, held.abs())? {
            sent.push(self.make_exit_order(at, exit.key, terms, units, by.into())?);
        }
        if exit.flags != FillFlags::NONE {
            let flagged = sent.iter().map(|id| (*id, exit.flags));
            self.venue.exit_flags.extend(flagged);
        }
        self.set_sent(exit, sent);
        Ok(())
    }

    /// What the fills of order `id` are flagged: an engine exit's flags
    /// (`PositionExit::flags`), `NONE` for any other order.
    pub(super) fn exit_flags_of(&self, id: OrderId) -> FillFlags {
        let flagged = self.venue.exit_flags.iter().find(|(order, _)| *order == id);
        flagged.map_or(FillFlags::NONE, |(_, flags)| *flags)
    }

    /// Make and send one of the engine's exit orders of `key`: a reduce-only DAY order of
    /// `units` on the terms given, for `role`, decided and stamped as `at` says.
    ///
    /// Errors: those of the order book.
    fn make_exit_order(
        &mut self,
        at: SendAt,
        key: PositionKey,
        terms: ExitTerms,
        units: f64,
        role: OrderRole,
    ) -> Result<OrderId, KernelError> {
        let request = OrderRequest {
            product: key.product,
            reduce_only: true,
            exit_reason: terms.exit_reason,
            order_type: terms.order_type,
            ..OrderRequest::market(key.instrument, terms.side, QuantitySpec::Units(units))
        };
        let id = self.insert_engine_order(at, request, units, terms.good_until)?;
        self.place(id, role);
        Ok(id)
    }

    /// Put an order the engine makes in the book, passed and `Submitted`, decided and
    /// stamped as `at` says; nothing is sent yet.
    ///
    /// Errors: those of the order book.
    pub(super) fn insert_engine_order(
        &mut self,
        at: SendAt,
        request: OrderRequest,
        units: f64,
        good_until: TsNanos,
    ) -> Result<OrderId, KernelError> {
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
        Ok(id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_exit_goes_out_in_pieces_of_at_most_the_freeze_quantity_in_whole_lots() {
        let table: [(f64, Option<u32>, u32, &[f64]); 5] = [
            (100.0, Some(40), 1, &[40.0, 40.0, 20.0]),
            (40.0, Some(40), 1, &[40.0]),
            (100.0, None, 1, &[100.0]),
            // 1,000 units of 75-unit lots: 13 lots (975) at most in one order.
            (1_050.0, Some(1_000), 75, &[975.0, 75.0]),
            // A freeze quantity below one lot cannot split it.
            (150.0, Some(50), 75, &[150.0]),
        ];
        for (units, freeze, lot, pieces) in table {
            assert_eq!(
                freeze_pieces(units, freeze, lot),
                pieces,
                "{units} of {freeze:?}"
            );
        }
    }
}
