//! Leg rules in the slice sequence: levels tested in step 3, ratchets in step 4, and
//! the rule that an add keeps the position's rules (SPEC-14).
//!
//! Legs are processed instrument by instrument in registration order, and within an
//! instrument in the order their positions opened, so the events of one step follow
//! 03-architecture 4.5 and no leg's result depends on another's.
//!
//! A level hit is an engine-held exit: it emits its risk event (`leg_sl_hit` or
//! `leg_target_hit` with the level in force) and then fills through
//! `fills::execute_fill` with no order, reason `leg_sl` or `leg_target` and the target
//! index, at the open or at the level plus slippage. Ratchets emit `leg_trail_armed`
//! and `leg_breakeven_armed` once each, and a moved stop or a new armed state writes a
//! row of the position history.

use crate::exec::OrderRequest;
use crate::model::Side;
use crate::portfolio::PositionKey;
use crate::rms::{
    ExitReason, LegState, LevelHit, LevelKind, RiskEvent, RiskEventName, RiskScope, RmsError,
};

use super::clock::SliceInfo;
use super::error::KernelError;
use super::event::{Event, EventStamp, Step};
use super::fills::FillRequest;
use super::gtt::held_as_gtt;
use super::kernel::Engine;

impl Engine {
    /// Step 3: every engine-held leg's levels against its instrument's bar in this
    /// slice. Legs opened or added to in step 2 are tested with the levels step 2 set
    /// (SPEC-02, SPEC-11). A CNC or NRML leg's levels are GTTs, tested by `gtt_fire`.
    ///
    /// Errors: those of `execute_fill` and of the leg rules.
    pub(super) fn test_leg_levels(&mut self, slice: &SliceInfo) -> Result<(), KernelError> {
        for key in self.leg_order() {
            if held_as_gtt(key.product) {
                continue;
            }
            let Some(bar) = self.market.bar(key.instrument, slice.index) else {
                continue;
            };
            let hits = match self.leg_mut(key) {
                Some(leg) => leg.test_bar(&bar),
                None => continue,
            };
            for hit in hits {
                if self.held_units(key) == 0.0 {
                    break;
                }
                self.exit_at_level(Self::stamp(slice, Step::Levels), key, hit)?;
            }
        }
        Ok(())
    }

    /// Step 4: trail and breakeven of every leg with a bar, from the bar's extreme; the
    /// levels they move are in force from the next slice.
    ///
    /// Errors: those of the leg rules.
    pub(super) fn ratchet_legs(&mut self, slice: &SliceInfo) -> Result<(), KernelError> {
        let stamp = Self::stamp(slice, Step::Ratchets);
        for key in self.leg_order() {
            let Some(bar) = self.market.bar(key.instrument, slice.index) else {
                continue;
            };
            let outcome = match self.leg_mut(key) {
                Some(leg) => leg.ratchet(&bar)?,
                None => continue,
            };
            if outcome.levels_moved {
                self.record.touch(key);
            }
            for event in outcome.events {
                self.emit_leg_event(stamp, key, event);
            }
        }
        Ok(())
    }

    /// At submission: leg rules attach to the position an order opens (SPEC-14). An add
    /// may repeat the position's rules or give none, but may not change them; an order
    /// that can only reduce (no `reverse=True`) may not carry any.
    ///
    /// Errors: `Rms(InvalidRule)` for an add whose rules differ from the position's, or
    /// for rules on an order that only reduces.
    pub(super) fn check_leg_rules(&self, request: &OrderRequest) -> Result<(), KernelError> {
        let key = PositionKey {
            instrument: request.instrument,
            product: request.product,
        };
        let held = self.held_units(key);
        let same_side = (held > 0.0) == (request.side == Side::Buy);
        let only_reduces = held != 0.0 && !same_side && !request.reverse;
        if !request.leg_rules.is_empty() && (request.reduce_only || only_reduces) {
            return Err(RmsError::InvalidRule {
                rule: "leg rules",
                reason: "stop, target, trail and breakeven attach to the position an order \
                         opens; this order only reduces the position (pass reverse=True to \
                         open the rest on the other side)"
                    .to_string(),
            }
            .into());
        }
        match self.legs.iter().find(|(k, _)| *k == key) {
            Some((_, leg)) if held != 0.0 && same_side && !request.reduce_only => {
                Ok(leg.accepts_add_rules(&request.leg_rules)?)
            }
            _ => Ok(()),
        }
    }

    /// The keys of the open legs: instruments in registration order, then legs in the
    /// order their positions opened.
    fn leg_order(&self) -> Vec<PositionKey> {
        let mut keys: Vec<PositionKey> = self.legs.iter().map(|(key, _)| *key).collect();
        keys.sort_by_key(|key| key.instrument);
        keys
    }

    fn leg_mut(&mut self, key: PositionKey) -> Option<&mut LegState> {
        self.legs
            .iter_mut()
            .find(|(k, _)| *k == key)
            .map(|(_, leg)| leg)
    }

    fn emit_leg_event(&mut self, stamp: EventStamp, key: PositionKey, event: RiskEvent) {
        self.emit(Event::Risk {
            stamp,
            instrument: Some(key.instrument),
            event,
        });
    }

    /// Exit at a level hit: the risk event, the fill, and for a target the
    /// `after_first_target` move.
    pub(super) fn exit_at_level(
        &mut self,
        stamp: EventStamp,
        key: PositionKey,
        hit: LevelHit,
    ) -> Result<(), KernelError> {
        let (name, reason, target_index) = match hit.kind {
            LevelKind::Stop => (RiskEventName::LegSlHit, ExitReason::LegSl, None),
            LevelKind::Target(index) => (
                RiskEventName::LegTargetHit,
                ExitReason::LegTarget,
                Some(index),
            ),
        };
        let event = RiskEvent {
            name,
            scope: RiskScope::Leg,
            level_rupees: Some(hit.level_rupees),
            target_index,
            observed_rupees: None,
        };
        self.emit_leg_event(stamp, key, event);
        let held = self.held_units(key);
        let exit = FillRequest::engine_exit(key, held, hit.price_rupees, reason).at(hit.at);
        let request = FillRequest {
            units: hit.quantity_units.min(held.abs()),
            target_index,
            ..exit
        };
        self.execute_fill(stamp, request)?;
        if target_index.is_some() {
            let moved = match self.leg_mut(key) {
                Some(leg) => leg.after_target_fill()?,
                None => None,
            };
            if let Some(event) = moved {
                self.emit_leg_event(stamp, key, event);
            }
        }
        Ok(())
    }
}
