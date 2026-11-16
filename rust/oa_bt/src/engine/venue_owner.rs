//! The exits of a position in an external session (`venue`): cancel, then exit.
//!
//! An exit is the strategy's `exit()` or `reduce()` (a reduce-only MARKET order), a plain
//! MARKET order against a position whose leg's stop rests (a sale, a reversal:
//! `venue_roles`), a leg's target (`venue_ticks`, made for the strategy), or one of the
//! engine's exits
//! (`venue_exits`, and the session's own, `venue_session`). No working order counts as
//! covering the position, so an exit first sends `Cancel` for every working order of its
//! position on the closing side but the position's exits and the orders sent for them,
//! and is sent once they have all ended (cancelled, or filled), for the position then
//! open: nothing when they filled it. An order placed on the closing side while an exit
//! waits is cancelled too.
//!
//! A partial exit beside a leg's broker-held stop is the exception (16-session-design
//! 4.5, requirement R4): a strategy exit that asks for part of the position keeps the
//! stop, which is modified down to what stays open first (`venue_legs`), and is sent once
//! the venue has confirmed that; meanwhile it waits. A full exit cancels the stop like
//! any order on the closing side, and so do a partial exit and the later exits waiting
//! behind it when together they close the whole position (a `reduce()` then an `exit()`):
//! nothing is left for the stop to protect, and a later exit never waits forever behind a
//! cut of the stop the venue keeps refusing. A stop repriced to a LIMIT (`venue_protect`)
//! is exiting the whole position: it counts among the position's exits for the
//! strategy's, which are sized on what it leaves (nothing), and an engine exit cancels it
//! first.
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
//! How an exit goes out when nothing blocks it (its units, the freeze quantity, a closed
//! market) is `venue_exit_orders`.

use crate::exec::{OrderAction, OrderId};
use crate::model::Side;
use crate::portfolio::PositionKey;

use super::error::KernelError;
use super::event::OrderEventKind;
use super::kernel::Engine;
use super::venue::{ExitBy, PositionExit, SendAt};
use super::venue_roles::EngineExit;

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
            limit_rupees: None,
        });
        Ok(())
    }

    /// The engine's exit of `key` (`exit`: the square-off or a risk exit), decided as `at`
    /// says, at market or at `limit_rupees`: it takes over from the strategy's exits of the
    /// position (module docs), unless the position is flat or the engine's exit holds it
    /// already.
    ///
    /// Errors: those of `fills::end_order` and of `progress_exits`.
    pub(super) fn request_engine_exit(
        &mut self,
        at: SendAt,
        key: PositionKey,
        exit: EngineExit,
        limit_rupees: Option<f64>,
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
            by: ExitBy::Engine(exit),
            sent: Vec::new(),
            limit_rupees,
        });
        self.progress_exits(at)
    }

    /// Move every waiting exit on, in the order they were made (module docs): cancel what
    /// blocks it, or send it once nothing does; an exit of a position waits behind an
    /// earlier one of it that still waits. Then forget the exits whose orders have all
    /// ended, and keep the legs' stops on their legs (`venue_legs`).
    ///
    /// Errors: those of the order book, of `fills::end_order` and of the leg stops.
    pub(super) fn progress_exits(&mut self, at: SendAt) -> Result<(), KernelError> {
        self.tidy_venue();
        let waiting: Vec<PositionExit> = self
            .venue
            .exits
            .iter()
            .filter(|exit| exit.sent.is_empty())
            .cloned()
            .collect();
        let mut held_back: Vec<PositionKey> = Vec::new();
        for exit in waiting {
            if !held_back.contains(&exit.key) && self.progress_exit(at, &exit)? {
                held_back.push(exit.key);
            }
        }
        self.tidy_venue();
        self.sync_leg_stops(at)
    }

    /// Move one waiting exit on; true when it still waits.
    fn progress_exit(&mut self, at: SendAt, exit: &PositionExit) -> Result<bool, KernelError> {
        let held = self.held_units(exit.key);
        if held == 0.0 {
            self.drop_exit(at, exit, OrderEventKind::Canceled)?;
            return Ok(false);
        }
        let open = held.abs() - self.left_to_exit(exit.key);
        let partial = self.partial_exit_units(exit, open)?;
        let keeps_stop = partial.is_some() && !self.waiting_exits_close_all(exit.key, open)?;
        let kept_stop = if keeps_stop {
            self.resting_leg_stop(exit.key)
        } else {
            None
        };
        let blocking: Vec<OrderId> = self
            .blocking(exit, closing_side(held))
            .into_iter()
            .filter(|id| Some(*id) != kept_stop)
            .collect();
        if !blocking.is_empty() {
            for id in blocking {
                self.withdraw_order(at.stamp(), id)?;
            }
            return Ok(true);
        }
        if let (Some(units), Some(stop)) = (partial, kept_stop) {
            let covers = self.book.get(stop).map_or(0.0, |o| self.units_to_fill(o));
            if covers > open - units {
                // The stop is cut down to what stays open first (`venue_legs`).
                return Ok(true);
            }
        }
        match exit.by {
            ExitBy::Strategy(id) => self.send_strategy_exit(at, exit, id, held)?,
            ExitBy::Engine(by) => self.send_engine_exit(at, exit, by, held)?,
        }
        Ok(false)
    }

    /// The working orders of `exit`'s position on `side` that it cancels first: all but
    /// the position's exits and the orders sent for them, and for a strategy exit a leg
    /// stop repriced to exit the position (module docs).
    fn blocking(&self, exit: &PositionExit, side: Side) -> Vec<OrderId> {
        let key = exit.key;
        let mut exits = self.exit_orders_of(key);
        if matches!(exit.by, ExitBy::Strategy(_)) {
            exits.extend(self.repriced_leg_stop(key));
        }
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

    /// The orders of `key`'s exits: each strategy exit's own order and the orders sent for
    /// every exit.
    pub(super) fn exit_orders_of(&self, key: PositionKey) -> Vec<OrderId> {
        self.venue
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
            .collect()
    }

    /// Give up an exit: the strategy's order ends as `kind` says; the exit is forgotten.
    ///
    /// Errors: those of `fills::end_order`.
    pub(super) fn drop_exit(
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

    /// Record the orders sent for `exit`.
    pub(super) fn set_sent(&mut self, exit: &PositionExit, sent: Vec<OrderId>) {
        let known = self
            .venue
            .exits
            .iter_mut()
            .find(|known| known.key == exit.key && known.by == exit.by);
        if let Some(known) = known {
            known.sent = sent;
        }
    }

    /// The units the orders sent for the exits of `key` still have to fill on its closing
    /// side, with a leg stop repriced to exit the position (module docs). A reversal's
    /// units left once it has taken the position through zero add to the new one.
    pub(super) fn left_to_exit(&self, key: PositionKey) -> f64 {
        let repriced = self.repriced_leg_stop(key);
        let side = closing_side(self.held_units(key));
        self.venue
            .exits
            .iter()
            .filter(|exit| exit.key == key)
            .flat_map(|exit| exit.sent.iter())
            .chain(repriced.iter())
            .filter_map(|id| self.book.get(*id))
            .filter(|order| order.request.side == side)
            .map(|order| self.units_to_fill(order))
            .sum()
    }
}

#[cfg(test)]
mod freeze_split_tests;
