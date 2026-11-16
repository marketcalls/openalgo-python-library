//! A leg's broker-held stop kept on its leg (`venue_legs`; 16-session-design 4.1 and 4.3,
//! requirements R2, R4 and R6): its units and its trigger follow the leg's, the trigger
//! under a throttle.
//!
//! The stop covers the position less what the position's exits sent still have to fill,
//! and less the units of a partial exit that waits for it (`venue_owner`). When those units
//! change (a further fill of the entry, an add, a partial exit, an exit that ended
//! unfilled) the kernel sends `Modify` of its units at once. When the leg's level moves (a
//! trail, a breakeven) it sends `Modify` of its trigger once the move is at least
//! `TRAIL_MODIFY_MIN_BASIS_POINTS` of the trigger the venue last confirmed and
//! `MODIFY_EVERY_NS` have passed since its last modification was sent, by the session's
//! clock (`VenueDesk::clock_ns`: a tick's or a fill's time, or the open of a slice the
//! session reaches). A modification carries the newest terms, units and trigger together.
//! One waits for the venue's answer at a time; meanwhile the newest terms are kept (they
//! are the leg's) and go out with the answer, or once the throttle allows. One the venue
//! refused is sent again `MODIFY_EVERY_NS` after it was sent. The stop is cancelled when it
//! has nothing left to cover; one repriced to a LIMIT (`venue_protect`) no longer follows.
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
        self.venue.leg_stops.retain(|stop| {
            positions
                .position(stop.key)
                .is_some_and(|position| !position.is_flat())
        });
        let keys: Vec<PositionKey> = self.venue.leg_stops.iter().map(|stop| stop.key).collect();
        for key in keys {
            self.sync_leg_stop(at, key)?;
        }
        Ok(())
    }

    fn sync_leg_stop(&mut self, at: SendAt, key: PositionKey) -> Result<(), KernelError> {
        let Some(stop) = self.leg_stop_of(key) else {
            return Ok(());
        };
        let Some(trigger) = self.stop_trigger(key)? else {
            return Ok(());
        };
        let cover = self.stop_cover_units(key)?;
        let Some(id) = stop.order else {
            if !self.has_waiting_exit(key) {
                self.place_leg_stop(at, key, cover, false)?;
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
        if cover <= 0.0 {
            self.withdraw_order(at.stamp(), id)?;
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
