//! A leg's broker-held stop kept on its leg (`venue_legs`; 16-session-design 4.1 and 4.3,
//! requirements R2, R4 and R6): its units and its trigger follow the leg's, the trigger
//! under a throttle.
//!
//! The stop covers the position less what the position's exits sent still have to fill,
//! and less the units of a partial exit that waits for it (`venue_owner`). It never works
//! beyond the room the other orders sent only to close the position leave on its closing
//! side (`venue_cover`), and it takes that room from them: when it wants more than they
//! leave (a trail or breakeven arms beside a resting sale, `protect`, a re-placed stop),
//! they are cancelled, the stop goes out (or stays) at what they leave, and it covers the
//! rest once they have ended. So the two rules never undo each other: the cover fit keeps
//! the stop first and cuts the others (`venue_cover`). When its units change (a further
//! fill of the entry, an add, a partial exit, an exit that ended unfilled, a resting sale
//! cancelled) the kernel sends `Modify` of its units at once. When the leg's level moves (a
//! trail, a breakeven) it sends `Modify` of its trigger once the move is at least
//! `TRAIL_MODIFY_MIN_BASIS_POINTS` of the trigger the venue last confirmed and
//! `MODIFY_EVERY_NS` have passed since its last modification was sent, by the session's
//! clock (`VenueDesk::clock_ns`: a tick's or a fill's time, or the open of a slice the
//! session reaches). A modification carries the newest terms, units and trigger together.
//! One waits for the venue's answer at a time; meanwhile the newest terms are kept (they
//! are the leg's) and go out with the answer, or once the throttle allows. One the venue
//! refused is sent again `MODIFY_EVERY_NS` after it was sent. The stop is cancelled when it
//! has nothing left to cover; one repriced to a LIMIT (`venue_protect`) no longer follows.
//! While a margin rejection holds the position (`venue_group`) no stop of it is placed and
//! none is given more units than it covers: a cut, a cancellation and a trigger move go.
//!
//! The same throttle would hold a backtest on recorded ticks; the kernel has no tick
//! backtest yet, so a backtest moves its engine-held stop at every ratchet.

use crate::exec::price::whole_units;
use crate::exec::{OrderId, OrderRequest, QuantitySpec};
use crate::model::TsNanos;
use crate::portfolio::PositionKey;

use super::error::KernelError;
use super::kernel::Engine;
use super::venue::{Replacement, SendAt};
use super::venue_owner::closing_side;

/// The least move of a stop's trigger worth a modification, in basis points (0.25
/// percent) of the trigger the venue last confirmed (08-live-trading 6.2, requirement R6).
pub(super) const TRAIL_MODIFY_MIN_BASIS_POINTS: i128 = 25;
/// Basis points in a whole: 100 percent.
const BASIS_POINTS_PER_WHOLE: i128 = 10_000;
/// The least time between two modifications of one stop, UTC nanoseconds: 5 s
/// (08-live-trading 6.2, requirement R6).
pub(super) const MODIFY_EVERY_NS: TsNanos = 5_000_000_000;

/// True when a trigger moved from `confirmed_rupees` to `wanted_rupees` by enough for a
/// modification: at least `TRAIL_MODIFY_MIN_BASIS_POINTS` of the confirmed trigger,
/// compared on whole price units.
fn moved_enough(confirmed_rupees: f64, wanted_rupees: f64) -> bool {
    let confirmed = i128::from(whole_units(confirmed_rupees));
    let moved = (i128::from(whole_units(wanted_rupees)) - confirmed).abs();
    moved > 0 && moved * BASIS_POINTS_PER_WHOLE >= confirmed * TRAIL_MODIFY_MIN_BASIS_POINTS
}

/// True when `MODIFY_EVERY_NS` have passed at `now` since a modification sent at
/// `last_sent` (always, when none was).
fn throttle_passed(last_sent: Option<TsNanos>, now: TsNanos) -> bool {
    last_sent.map_or(true, |sent| now.saturating_sub(sent) >= MODIFY_EVERY_NS)
}

impl Engine {
    /// Keep every leg's stop on its leg (module docs), decided and stamped as `at` says:
    /// place the stop of a leg that has a level and has had none (`venue_legs`), modify one
    /// whose units or trigger are due, cancel one with nothing left to cover. A backtest
    /// has none.
    ///
    /// Errors: those of the order book, of `withdraw_order` and of the stop's shape.
    pub(super) fn sync_leg_stops(&mut self, at: SendAt) -> Result<(), KernelError> {
        if !self.is_external() {
            return Ok(());
        }
        let positions = &self.portfolio;
        let open = |key: PositionKey| {
            positions
                .position(key)
                .is_some_and(|position| !position.is_flat())
        };
        self.venue.leg_stops.retain(|stop| open(stop.key));
        self.venue.fired_targets.retain(|(key, _)| open(*key));
        let keys: Vec<PositionKey> = self.venue.leg_stops.iter().map(|stop| stop.key).collect();
        for key in keys {
            self.sync_leg_stop(at, key)?;
        }
        Ok(())
    }

    /// Keep `key`'s leg stop on its leg (`sync_leg_stops`): a stop not placed yet (the leg
    /// has had none, or `renew_leg_stop` asked for a new one) is placed once the leg has a
    /// level, no exit closing the whole position waits and no margin rejection holds it.
    ///
    /// Errors: those of `sync_leg_stops`.
    pub(super) fn sync_leg_stop(
        &mut self,
        at: SendAt,
        key: PositionKey,
    ) -> Result<(), KernelError> {
        let Some(stop) = self.leg_stop_of(key) else {
            return Ok(());
        };
        let Some(trigger) = self.stop_trigger(key)? else {
            return Ok(());
        };
        let wanted = self.stop_cover_units(key)?;
        let Some(id) = stop.order else {
            if wanted > 0.0 && !self.venue.group.holds(key) && !self.exit_closes_leg(key)? {
                let units = self.take_room_for_stop(at, key, None, wanted)?;
                self.place_leg_stop(at, key, units, stop.replaced)?;
            }
            return Ok(());
        };
        let Some(order) = self.book.get(id).filter(|o| o.status.is_working()).cloned() else {
            return Ok(());
        };
        let answer_pending = self.venue.cancelling.contains(&id)
            || self.venue.modifying.iter().any(|(known, _)| *known == id);
        let Some(confirmed) = order.request.order_type.trigger_rupees() else {
            return Ok(());
        };
        if stop.repriced || answer_pending {
            return Ok(());
        }
        if wanted <= 0.0 {
            self.withdraw_order(at.stamp(), id)?;
            return Ok(());
        }
        let wanted = self.stop_units_allowed(key, &order, wanted);
        let cover = self.take_room_for_stop(at, key, Some(id), wanted)?;
        if cover <= 0.0 {
            // The orders in its way are being cancelled; it is never cancelled for them.
            return Ok(());
        }
        let units = order.filled_units + cover;
        let now = self.venue.clock_ns;
        let throttled = !throttle_passed(stop.modified_at, now);
        let units_due = units != order.quantity_units && !(stop.refused && throttled);
        let trigger_due = moved_enough(confirmed, trigger) && !throttled;
        if !(units_due || trigger_due) {
            return Ok(());
        }
        let held = self.held_units(key);
        let request = OrderRequest {
            quantity: QuantitySpec::Units(units),
            order_type: self.stop_type(key, held, trigger)?,
            ..order.request.clone()
        };
        let replacement = Replacement {
            request,
            decided_at: order.decided_at,
            good_until: order.good_until,
            units,
        };
        self.send_modify(id, replacement);
        if let Some(stop) = self.leg_stop_mut(key) {
            stop.modified_at = Some(now);
            stop.refused = false;
        }
        Ok(())
    }

    /// The units `key`'s leg stop (`stop`, once it is in the book) may cover now, at most
    /// `wanted` (module docs): the position less what its exits still have to fill and
    /// less what the other orders sent only to close it may fill on its closing side.
    /// When that leaves less than `wanted`, those orders are cancelled, decided and
    /// stamped as `at` says, so the stop covers the rest once they have ended.
    ///
    /// Errors: those of `withdraw_order`.
    fn take_room_for_stop(
        &mut self,
        at: SendAt,
        key: PositionKey,
        stop: Option<OrderId>,
        wanted: f64,
    ) -> Result<f64, KernelError> {
        let others = self.closing_orders_beside_stop(key, stop);
        let taken: f64 = others.iter().map(|(_, units)| units).sum();
        let room = (self.held_units(key).abs() - self.left_to_exit(key) - taken).max(0.0);
        if wanted > room {
            for (id, _) in others {
                self.withdraw_order(at.stamp(), id)?;
            }
        }
        Ok(wanted.min(room))
    }

    /// The working orders of `key`'s position on its closing side sent only to close it
    /// (`Engine::sent_to_close`) that are neither its leg stop `stop` nor one of its exits
    /// (`venue_owner`), with the units each may still fill.
    fn closing_orders_beside_stop(
        &self,
        key: PositionKey,
        stop: Option<OrderId>,
    ) -> Vec<(OrderId, f64)> {
        let held = self.held_units(key);
        if held == 0.0 {
            return Vec::new();
        }
        let side = closing_side(held);
        let mut exits = self.exit_orders_of(key);
        exits.extend(self.repriced_leg_stop(key));
        self.book
            .working()
            .filter(|order| {
                order.request.instrument == key.instrument
                    && order.request.product == key.product
                    && order.request.side == side
                    && Some(order.id) != stop
                    && !exits.contains(&order.id)
                    && self.sent_to_close(order)
            })
            .map(|order| (order.id, self.units_to_fill(order)))
            .collect()
    }

    /// The venue answered the modification of order `id`: confirmed, or refused (module
    /// docs). Nothing for an order that is no leg's stop.
    pub(super) fn leg_stop_answered(&mut self, id: OrderId, confirmed: bool) {
        let stop = self
            .venue
            .leg_stops
            .iter_mut()
            .find(|stop| stop.order == Some(id));
        if let Some(stop) = stop {
            stop.refused = !confirmed;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_trigger_is_modified_only_for_a_move_of_a_quarter_percent_of_the_confirmed_one() {
        // (confirmed trigger, wanted trigger, modified)
        let table = [
            (500.0, 501.25, true),
            (500.0, 501.20, false),
            (500.0, 498.75, true),
            (500.0, 500.0, false),
            (100.0, 100.25, true),
            (100.0, 100.20, false),
            (40.0, 40.10, true),
        ];
        for (confirmed, wanted, modified) in table {
            assert_eq!(
                moved_enough(confirmed, wanted),
                modified,
                "{confirmed} to {wanted}"
            );
        }
    }

    #[test]
    fn a_stop_is_modified_at_most_once_every_five_seconds() {
        let second = 1_000_000_000;
        // (last modification sent, now, may send)
        let table = [
            (None, 0, true),
            (Some(0), 4 * second, false),
            (Some(0), 5 * second, true),
            (Some(10 * second), 14 * second + second / 2, false),
            (Some(10 * second), 16 * second, true),
        ];
        for (sent, now, may) in table {
            assert_eq!(throttle_passed(sent, now), may, "{sent:?} at {now}");
        }
    }
}
