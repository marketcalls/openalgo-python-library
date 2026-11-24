//! A CNC or NRML leg's stop GTT kept on its leg in a live session (`venue_gtts`; design
//! 20-positional-live section 2, 04-execution 4.4 items 2 and 5, unit N1 item N1-1).
//!
//! The leg state stays the one source of the leg's levels (`gtt_plan`). Its stop is held
//! at the broker as one SINGLE GTT, as a backtest holds it (`gtt`):
//!
//! - **Placed** once the leg has a stop level: with the fill that opens it (for the units
//!   filled), or when a trail or a breakeven first sets one; also while the market is
//!   closed, since brokers take GTTs outside market hours. Its trigger is the leg's level
//!   on the tick grid, below the price for a long and above it for a short, and its child a
//!   LIMIT beyond the trigger by the segment buffer (GTT-1).
//! - **Kept on the leg**: its units follow what the stop covers (`stop_cover_units`: the
//!   position less what its exits still have to fill, and less a partial exit waiting for
//!   the cut), at once; its trigger follows a trail or a breakeven when it moves by at
//!   least one tick (GTT-3) and at most once every `GTT_MODIFY_EVERY_NS` (04-execution 4.4
//!   item 5: in live at most once a minute). A modification the venue refused is sent
//!   again once that minute has passed. One waits for the venue's answer at a time; the
//!   newest terms are the leg's and go out with the answer, or once the minute allows. A
//!   unit change is not held for the minute, since a partial exit waits for the cut: a cut
//!   made within the minute keeps the resting trigger, and the trigger follows a minute
//!   after the cut, so the trigger moves at most once a minute.
//! - **Cancelled** when it has nothing left to cover (the position closed at the broker,
//!   a stop level removed), and before an exit of the whole position (`venue_owner`).
//! - **Placed only beside no other**: while a stop GTT an earlier leg left rests on its
//!   side (`Engine::loose_stop_gtts`), a new leg's stop GTT waits for its cancellation.
//! - **Renewed** at a session open once older than the policy allows (GTT-2): the old one
//!   is cancelled, and the new one goes once the venue has taken that, by the rule above.
//! - **Not placed again by itself** once it has fired, been cancelled for an exit or
//!   refused: the rules of `venue_gtt_answers` and `venue_gtt_protect` decide that.
//!
//! As for an MIS leg's stop, it takes the room the strategy's resting orders sent only to
//! close the position leave (`venue_stop_terms`), and while a margin rejection holds the
//! position (`venue_group`) it is neither placed nor given more units.

use crate::exec::price::{tick_units, whole_units};
use crate::exec::{Gtt, GttId, GttLegRole, GttPurpose, GttStatus, TriggerDirection};
use crate::model::TsNanos;
use crate::portfolio::PositionKey;

use super::clock::SliceInfo;
use super::error::KernelError;
use super::event::Step;
use super::gtt::held_as_gtt;
use super::kernel::Engine;
use super::venue::SendAt;
use super::venue_gtts::{GttLegStop, GttTerms};
use super::venue_owner::closing_side;

/// The least time between two trigger modifications of one GTT in a live session, UTC
/// nanoseconds: one minute (04-execution 4.4 item 5).
pub(super) const GTT_MODIFY_EVERY_NS: TsNanos = 60_000_000_000;

/// True when `GTT_MODIFY_EVERY_NS` have passed at `now` since a modification sent at
/// `last_sent` (always, when none was).
fn minute_passed(last_sent: Option<TsNanos>, now: TsNanos) -> bool {
    last_sent.map_or(true, |sent| now.saturating_sub(sent) >= GTT_MODIFY_EVERY_NS)
}

impl Engine {
    /// A CNC or NRML leg opened at `key`: its stop GTT is placed once it has a level
    /// (module docs). A GTT of an earlier leg is no longer the leg's: one still at the
    /// venue is cancelled (`loose_stop_gtts`).
    pub(super) fn open_leg_gtt(&mut self, key: PositionKey) {
        self.venue.gtts.legs.retain(|leg| leg.key != key);
        self.venue.gtts.legs.push(GttLegStop {
            key,
            gtt: None,
            rejections: 0,
        });
    }

    /// Keep every CNC and NRML leg's stop GTT on its leg (module docs), decided as `at`
    /// says; cancel the stop GTTs no leg holds (`loose_stop_gtts`); then cancel the
    /// children of fired GTTs that did not fill through a base bar (`venue_gtt_child`). A
    /// backtest has none.
    ///
    /// Errors: those of the order book, of the GTT book and of the stop's terms.
    pub(super) fn sync_stop_gtts(&mut self, at: SendAt) -> Result<(), KernelError> {
        if !self.is_external() {
            return Ok(());
        }
        let keys: Vec<PositionKey> = self.venue.gtts.legs.iter().map(|leg| leg.key).collect();
        for key in keys {
            self.sync_stop_gtt(at, key)?;
        }
        let book = &self.gtts.book;
        let positions = &self.portfolio;
        self.venue.gtts.legs.retain(|leg| {
            let open = positions.position(leg.key).is_some_and(|p| !p.is_flat());
            let at_venue = leg
                .gtt
                .and_then(|id| book.get(id))
                .is_some_and(|gtt| gtt.status.at_venue());
            open || at_venue
        });
        for id in self.loose_stop_gtts() {
            self.ask_cancel_gtt(at.stamp().slice, id)?;
        }
        self.cancel_unfilled_children(at)
    }

    /// Keep `key`'s stop GTT on its leg (`sync_stop_gtts`): place it, keep it on the
    /// leg's units and trigger, or cancel it once it has nothing to cover.
    ///
    /// Errors: those of `sync_stop_gtts`.
    pub(super) fn sync_stop_gtt(
        &mut self,
        at: SendAt,
        key: PositionKey,
    ) -> Result<(), KernelError> {
        let Some(record) = self.leg_gtt_of(key) else {
            return Ok(());
        };
        let trigger = if self.held_units(key) == 0.0 {
            None
        } else {
            self.stop_trigger(key)?
        };
        let wanted = match trigger {
            Some(_) => self.stop_cover_units(key)?,
            None => 0.0,
        };
        match (self.stop_gtt_at_venue(key).cloned(), trigger) {
            (None, Some(trigger)) if record.gtt.is_none() && wanted > 0.0 => {
                self.place_wanted_stop_gtt(at, key, trigger, wanted)
            }
            (Some(gtt), _) if wanted <= 0.0 => self.ask_cancel_gtt(at.stamp().slice, gtt.id),
            (Some(gtt), Some(trigger)) => self.keep_stop_gtt(at, key, &gtt, (trigger, wanted)),
            _ => Ok(()),
        }
    }

    /// Place `key`'s stop GTT at `trigger_rupees` for the `wanted` units it covers, unless
    /// a margin rejection holds the position, an exit closes it, or a stop GTT an earlier
    /// leg left rests on its side (module docs, `loose_stop_gtts`): the broker never holds
    /// two.
    ///
    /// Errors: those of `sync_stop_gtts`.
    fn place_wanted_stop_gtt(
        &mut self,
        at: SendAt,
        key: PositionKey,
        trigger_rupees: f64,
        wanted: f64,
    ) -> Result<(), KernelError> {
        let side = closing_side(self.held_units(key));
        let beside_loose = self.loose_stop_gtts().into_iter().any(|id| {
            self.gtts.book.get(id).is_some_and(|gtt| {
                gtt.instrument == key.instrument && gtt.product == key.product && gtt.side == side
            })
        });
        if beside_loose || self.venue.group.holds(key) || self.exit_closes_leg(key)? {
            return Ok(());
        }
        let units = self.take_room_for_stop(at, key, None, wanted)?;
        if units > 0.0 {
            self.place_stop_gtt(at, key, trigger_rupees, units)?;
        }
        Ok(())
    }

    /// Keep `key`'s resting stop GTT `gtt` on the leg's `trigger_rupees` and the `wanted`
    /// units it covers (module docs): the units at once, a cut on the resting trigger while
    /// the minute since the last modification has not passed; the trigger once it moves by
    /// a tick and the minute allows.
    ///
    /// Errors: those of `sync_stop_gtts`.
    fn keep_stop_gtt(
        &mut self,
        at: SendAt,
        key: PositionKey,
        gtt: &Gtt,
        (trigger_rupees, wanted): (f64, f64),
    ) -> Result<(), KernelError> {
        let follow = self.gtt_follow(gtt.id);
        if gtt.status != GttStatus::Active || follow.is_some_and(|f| f.cancel_wanted) {
            // Its answer comes first; a wanted cancellation goes with it.
            return Ok(());
        }
        let wanted = match self.venue.group.holds(key) {
            true => wanted.min(gtt.quantity_units),
            false => wanted,
        };
        let units = self.take_room_for_stop(at, key, None, wanted)?;
        if units <= 0.0 {
            // The orders in its way are being cancelled; it is never cancelled for them.
            return Ok(());
        }
        let terms = self.stop_gtt_terms(key, trigger_rupees, units)?;
        let throttled = !minute_passed(follow.and_then(|f| f.modified_at), self.venue.clock_ns);
        let refused = follow.is_some_and(|f| f.refused);
        let units_due =
            whole_units(units) != whole_units(gtt.quantity_units) && !(refused && throttled);
        let trigger_due = self.moved_a_tick(gtt, &terms)? && !throttled;
        match (trigger_due, units_due) {
            (true, _) => self.send_modify_gtt(gtt.id, terms),
            (false, true) => {
                let cut = GttTerms {
                    units,
                    ..GttTerms::of(gtt)
                };
                self.send_modify_gtt(gtt.id, cut)
            }
            (false, false) => Ok(()),
        }
    }

    /// At the session open of `slice`, a leg's stop GTT older than the policy allows
    /// (GTT-2, design 20-positional-live 4.2 step 6) is no longer the leg's: it is
    /// cancelled (`loose_stop_gtts`), and the leg's new stop GTT, at its stop for the units
    /// it covers, goes once the venue has taken that cancellation (module docs). Until
    /// then the old one protects the leg.
    ///
    /// Errors: those of `sync_stop_gtts`.
    pub(super) fn renew_old_stop_gtts(&mut self, slice: &SliceInfo) -> Result<(), KernelError> {
        let policy = self.gtts.policy;
        let book = &self.gtts.book;
        let mut renewed = false;
        for leg in &mut self.venue.gtts.legs {
            let old = leg.gtt.and_then(|id| book.get(id)).is_some_and(|gtt| {
                gtt.status.at_venue() && policy.is_due_for_renewal(gtt.created_on, slice.day)
            });
            if old {
                leg.gtt = None;
                renewed = true;
            }
        }
        if !renewed {
            return Ok(());
        }
        let at = SendAt {
            slice: *slice,
            step: Step::SessionOpen,
            decided_at: slice.open,
        };
        self.sync_stop_gtts(at)
    }

    /// True when `terms` move `gtt`'s trigger by at least the policy's ticks (GTT-3).
    ///
    /// Errors: `Exec` for an unusable tick.
    fn moved_a_tick(&self, gtt: &Gtt, terms: &GttTerms) -> Result<bool, KernelError> {
        let tick = self.spec(gtt.instrument)?.tick_size_rupees;
        let least = i64::from(self.gtts.policy.min_modify_ticks) * tick_units(tick)?;
        let moved = whole_units(terms.trigger_rupees) - whole_units(gtt.leg.trigger_rupees);
        Ok(moved != 0 && moved.abs() >= least)
    }

    /// The terms of `key`'s stop GTT at `trigger_rupees` for `units`: the child on the
    /// closing side, beyond the trigger by the segment buffer, both on the tick grid
    /// (`gtt_plan::ChildTerms`).
    ///
    /// Errors: `Exec` for a level or tick that is not a usable price.
    pub(super) fn stop_gtt_terms(
        &self,
        key: PositionKey,
        trigger_rupees: f64,
        units: f64,
    ) -> Result<GttTerms, KernelError> {
        let child = self.child_terms(key)?;
        let leg = child.stop(trigger_rupees)?;
        Ok(GttTerms {
            instrument: key.instrument,
            product: key.product,
            side: child.child_side(),
            units,
            trigger_rupees: leg.trigger_rupees,
            limit_rupees: leg.child_limit_rupees,
            trigger_below: leg.direction == TriggerDirection::Falling,
        })
    }

    /// Place a stop GTT of `units` for `key`'s leg at `trigger_rupees`, decided as `at`
    /// says, and make it the leg's (module docs).
    ///
    /// Errors: those of `stop_gtt_terms` and of the GTT book.
    pub(super) fn place_stop_gtt(
        &mut self,
        at: SendAt,
        key: PositionKey,
        trigger_rupees: f64,
        units: f64,
    ) -> Result<GttId, KernelError> {
        let terms = self.stop_gtt_terms(key, trigger_rupees, units)?;
        let gtt = Gtt {
            id: self.gtts.book.next_id(),
            instrument: key.instrument,
            product: key.product,
            side: terms.side,
            quantity_units: units,
            leg: terms.leg(GttLegRole::Stop),
            purpose: GttPurpose::Protect,
            status: GttStatus::Requested,
            created_on: self.trade_day(at.stamp()),
            eligible_from: at.decided_at,
            last_modified_slice: None,
        };
        let id = self.send_place_gtt(gtt, None)?;
        if self.leg_gtt_of(key).is_none() {
            self.open_leg_gtt(key);
        }
        if let Some(leg) = self.leg_gtt_mut(key) {
            leg.gtt = Some(id);
        }
        Ok(id)
    }

    /// The units `key`'s stop GTT may still sell or buy at the venue: what it rests for,
    /// or what a modification waiting for the venue's answer would give it, the larger; 0
    /// with no stop GTT at the venue. A GTT's place that waits counts at its units.
    pub(super) fn stop_gtt_units(&self, key: PositionKey) -> f64 {
        let Some(gtt) = self.stop_gtt_at_venue(key) else {
            return 0.0;
        };
        let pending = self
            .gtt_follow(gtt.id)
            .and_then(|follow| follow.pending)
            .map_or(0.0, |terms| terms.units);
        gtt.quantity_units.max(pending)
    }

    /// True when `key` is a position whose stop a live session holds as a GTT.
    pub(super) fn holds_stop_gtt(&self, key: PositionKey) -> bool {
        self.is_external() && held_as_gtt(key.product)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_gtt_trigger_is_modified_at_most_once_a_minute() {
        let second = 1_000_000_000;
        // (last modification sent, now, may send)
        let table = [
            (None, 0, true),
            (Some(0), 59 * second, false),
            (Some(0), 60 * second, true),
            (Some(10 * second), 69 * second, false),
            (Some(10 * second), 70 * second, true),
        ];
        for (sent, now, may) in table {
            assert_eq!(minute_passed(sent, now), may, "{sent:?} at {now}");
        }
    }
}
