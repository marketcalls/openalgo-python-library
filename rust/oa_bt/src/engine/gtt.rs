//! GTT orders for positional trades in the slice sequence (04-execution 4.4): the GTT
//! that holds a CNC or NRML position's stop, its life between fills, and the
//! session-open check. The position's targets are the engine's, as an MIS leg's are
//! (design 20-positional-live P6); the leg test of step 3 decides whether the stop or a
//! target comes first in a bar (`gtt_levels`).
//!
//! | When | What | Step |
//! |---|---|---|
//! | A fill opens a CNC or NRML position with a stop | its stop GTT is placed for all the open units (`GttPlaced`), in force from step 3 of a step-2 fill (SPEC-02), from the next slice otherwise | the fill's |
//! | The leg's stop or units change (trail, breakeven, an add, a partial exit, a target that leaves units) | the stop GTT follows by `gtt_plan`: modified (`GttModified`, GTT-3), placed or cancelled; in force from the next slice | 4 |
//! | An exit that leaves units (the strategy's partial exit, a target's) | the stop GTT stays in force and is cut to the units left in step 4; if it fires before, its child sells at most the units left | the exit's |
//! | Any other exit that is not the GTT's own and closes the position (a signal or a target for all of it, the daily loss limit, the drawdown halt, the time exit, the close of a reversal) | the stop GTT is cancelled first (`GttCanceled`), then the exit fills (cancel-then-exit, GTT-9) | the exit's |
//! | A strategy GTT order ends (cancelled, rejected) | its GTT is cancelled | the order's |
//! | A session opens | every GTT older than 360 days is cancelled and placed again (GTT-2); a backtest assumes the engine runs at every session open (GTT-4) | 1 |
//!
//! Firing in step 3 and the engine's exit after an unfilled child are in `gtt_fire`;
//! strategy GTT orders (`validity="gtt"`) in `gtt_orders`. A live session sends a command
//! at each of these moments instead, and its venue holds and fires the GTT
//! (`venue_gtts`): none of the rules of this table acts in a session. Its renewal at a
//! session open cancels a leg's old stop GTT and places the new one once the venue has
//! taken the cancellation (`venue_gtt_sync`); a strategy GTT order is not renewed in a
//! session (not built): the broker's expiry ends it `Expired` (`venue_gtt_answers`), which
//! the strategy sees, where a backtest renews it, a difference only for an order resting
//! more than 360 days.

use crate::exec::{
    Gtt, GttAction, GttBook, GttEvent, GttEventName, GttId, GttPolicy, GttPurpose, GttStatus,
    OrderId,
};
use crate::model::{Product, Side, TsNanos};
use crate::portfolio::{FillEffect, FillKind, PositionKey};

use super::clock::SliceInfo;
use super::error::KernelError;
use super::event::{Event, EventStamp, Step};
use super::fills::FillRequest;
use super::gtt_plan::{needs_modification, protective_plan, ChildTerms, PlannedStop};
use super::kernel::Engine;

/// The engine's own exit after a stop GTT's child did not fill through its bar, waiting
/// for the instrument's next base-bar open (04-execution 4.4 item 3, LEAD-10), reason
/// `leg_sl`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct FollowUpExit {
    pub(super) key: PositionKey,
    pub(super) gtt: GttId,
    pub(super) units: f64,
}

/// The GTTs of a run and what the slice sequence keeps about them.
#[derive(Clone, Debug, Default, PartialEq)]
pub(super) struct GttDesk {
    pub(super) book: GttBook,
    pub(super) policy: GttPolicy,
    /// Exits waiting for a next base-bar open, in the order the children failed.
    pub(super) follow_ups: Vec<FollowUpExit>,
}

/// True when a product holds its stop as a GTT: CNC and NRML, which carry overnight.
/// By decision GTT-10 reading A, 04-execution 4.4 then governs their stop (a trigger and
/// a child LIMIT) instead of the engine-held rule that a level crossed at the open fills
/// at the open; their targets keep that rule, as an MIS leg's (design
/// 20-positional-live P6).
pub(super) fn held_as_gtt(product: Product) -> bool {
    matches!(product, Product::Cnc | Product::Nrml)
}

/// From when a GTT placed at `stamp` is tested: from step 3 of the same slice when it
/// is placed at the session open or with a fill at the open (SPEC-02), otherwise from
/// the next slice (a modification or placement takes effect from the next base bar).
fn eligible_from(stamp: EventStamp, slice: &SliceInfo) -> TsNanos {
    match stamp.step {
        Step::SessionOpen | Step::OpenFills => slice.open,
        _ => slice.close,
    }
}

impl Engine {
    /// Every GTT of the run, in creation order.
    pub fn gtts(&self) -> &[Gtt] {
        self.gtts.book.gtts()
    }

    /// Write the event `name` for the GTT `id` as it now stands.
    pub(super) fn emit_gtt(&mut self, stamp: EventStamp, name: GttEventName, id: GttId) {
        if let Some(gtt) = self.gtts.book.get(id) {
            let event = GttEvent::of(name, gtt);
            self.emit(Event::Gtt { stamp, event });
        }
    }

    /// The slice a stamp belongs to.
    fn slice_of(&self, stamp: EventStamp) -> Result<SliceInfo, KernelError> {
        self.timeline
            .slice(stamp.slice as usize)
            .ok_or(KernelError::InvalidConfig {
                field: "slice",
                reason: format!("no slice {} in the run's grid", stamp.slice),
            })
    }

    /// How the child of `key`'s stop GTT is priced.
    pub(super) fn child_terms(&self, key: PositionKey) -> Result<ChildTerms, KernelError> {
        let spec = self.spec(key.instrument)?;
        let held = self.held_units(key);
        Ok(ChildTerms {
            position_side: if held < 0.0 { Side::Sell } else { Side::Buy },
            tick_rupees: spec.tick_size_rupees,
            buffer_percent: self.gtts.policy.child_buffer_percent(spec.kind),
        })
    }

    /// After every fill: a fill that opens a CNC or NRML position with a stop places the
    /// GTT that holds it, right after the fill's events.
    ///
    /// Errors: `Exec` for a level that is not a usable price.
    pub(super) fn place_gtts_after_entry(
        &mut self,
        stamp: EventStamp,
        key: PositionKey,
        effect: &FillEffect,
    ) -> Result<(), KernelError> {
        if effect.kind != FillKind::Entry || !held_as_gtt(key.product) || self.is_external() {
            return Ok(());
        }
        let Some(planned) = self
            .leg(key.instrument, key.product)
            .and_then(protective_plan)
        else {
            return Ok(());
        };
        let eligible = eligible_from(stamp, &self.slice_of(stamp)?);
        self.place_protective(stamp, key, &planned, eligible)?;
        Ok(())
    }

    /// Place the planned stop GTT on `key` and write `GttPlaced`.
    fn place_protective(
        &mut self,
        stamp: EventStamp,
        key: PositionKey,
        planned: &PlannedStop,
        eligible_from: TsNanos,
    ) -> Result<GttId, KernelError> {
        let terms = self.child_terms(key)?;
        let id = self.gtts.book.next_id();
        let gtt = Gtt {
            id,
            instrument: key.instrument,
            product: key.product,
            side: terms.child_side(),
            quantity_units: planned.quantity_units,
            leg: terms.stop(planned.stop_rupees)?,
            purpose: GttPurpose::Protect,
            status: GttStatus::Active,
            created_on: self.trade_day(stamp),
            eligible_from,
            last_modified_slice: None,
        };
        self.gtts.book.insert(gtt)?;
        self.emit_gtt(stamp, GttEventName::Placed, id);
        Ok(id)
    }

    /// Cancel an active GTT and write `GttCanceled`.
    pub(super) fn cancel_gtt(&mut self, stamp: EventStamp, id: GttId) -> Result<(), KernelError> {
        self.gtts.book.apply(id, GttAction::Cancel)?;
        self.emit_gtt(stamp, GttEventName::Canceled, id);
        Ok(())
    }

    /// Cancel-then-exit, decision GTT-9 reading A: before any fill that closes a CNC or
    /// NRML position and is not its stop GTT's own exit (a signal or a target for all of
    /// it, a strategy GTT, the daily loss limit, the drawdown halt, the time exit, the
    /// close of a reversal), the GTT holding the position's stop is cancelled in the same
    /// step, so the position never has two exit owners (08-live-trading 6.2). An exit that
    /// leaves units (the strategy's partial exit, a target's) keeps the GTT in force
    /// instead, which step 4 cuts to the units left, in backtest and live alike (design
    /// 20-positional-live section 2; the lead's ruling on unit N0's verification, option
    /// B). A live session cancels its GTTs at the venue before it sends the exit
    /// (`venue_owner`), so nothing is cancelled here.
    pub(super) fn cancel_gtts_before_exit(
        &mut self,
        stamp: EventStamp,
        request: &FillRequest,
    ) -> Result<(), KernelError> {
        let key = request.key;
        let own_exit = request.gtt.is_some_and(|id| {
            let purpose = self.gtts.book.get(id).map(|gtt| gtt.purpose);
            purpose == Some(GttPurpose::Protect)
        });
        if own_exit || !held_as_gtt(key.product) || self.is_external() {
            return Ok(());
        }
        let held = self.held_units(key);
        let reduces = held != 0.0 && (held > 0.0) != (request.side == Side::Buy);
        let leaves_units = request.units < held.abs();
        if !reduces || leaves_units {
            return Ok(());
        }
        for id in self.gtts.book.protecting(key.instrument, key.product) {
            self.cancel_gtt(stamp, id)?;
        }
        Ok(())
    }

    /// When a strategy order ends (cancelled, rejected, expired), the GTT it rests as
    /// is cancelled first. A live session's GTT order ends with its GTT's answer instead
    /// (`venue_gtt_answers`).
    pub(super) fn cancel_order_gtt(
        &mut self,
        stamp: EventStamp,
        order: OrderId,
    ) -> Result<(), KernelError> {
        if self.is_external() {
            return Ok(());
        }
        match self.gtts.book.of_order(order) {
            Some(id) => self.cancel_gtt(stamp, id),
            None => Ok(()),
        }
    }

    /// Step 4, after the ratchets: every CNC or NRML leg's stop GTT follows its stop and
    /// units. A GTT in force is modified when GTT-3 allows, a stop with none is placed, a
    /// GTT whose leg has no stop is cancelled. A position whose GTT fired and awaits the
    /// engine's exit is left alone.
    ///
    /// Errors: `Exec` for a level that is not a usable price.
    pub(super) fn sync_gtts(&mut self, slice: &SliceInfo) -> Result<(), KernelError> {
        let stamp = Self::stamp(slice, Step::Ratchets);
        let keys: Vec<PositionKey> = self
            .legs
            .iter()
            .map(|(key, _)| *key)
            .filter(|key| held_as_gtt(key.product))
            .collect();
        for key in keys {
            let waiting = self.gtts.follow_ups.iter().any(|exit| exit.key == key);
            let plan = match self.leg(key.instrument, key.product) {
                Some(leg) if !waiting => protective_plan(leg),
                _ => continue,
            };
            self.sync_position(stamp, slice, key, plan)?;
        }
        Ok(())
    }

    /// The position's stop GTT made to match `plan`: the first one in force modified (or
    /// one placed when none is), any other cancelled; all cancelled when `plan` is none.
    fn sync_position(
        &mut self,
        stamp: EventStamp,
        slice: &SliceInfo,
        key: PositionKey,
        plan: Option<PlannedStop>,
    ) -> Result<(), KernelError> {
        let active = self.gtts.book.protecting(key.instrument, key.product);
        let kept = plan.and(active.first().copied());
        for id in active.iter().filter(|id| Some(**id) != kept) {
            self.cancel_gtt(stamp, *id)?;
        }
        match (plan, kept) {
            (Some(planned), Some(id)) => self.modify_protective(stamp, slice, key, id, &planned),
            (Some(planned), None) => self
                .place_protective(stamp, key, &planned, slice.close)
                .map(|_| ()),
            (None, _) => Ok(()),
        }
    }

    /// Modify the stop GTT to its planned level and quantity when GTT-3 allows.
    fn modify_protective(
        &mut self,
        stamp: EventStamp,
        slice: &SliceInfo,
        key: PositionKey,
        id: GttId,
        planned: &PlannedStop,
    ) -> Result<(), KernelError> {
        let terms = self.child_terms(key)?;
        let leg = terms.stop(planned.stop_rupees)?;
        let slice_number = stamp.slice;
        let Some(gtt) = self.gtts.book.get(id) else {
            return Ok(());
        };
        let policy = self.gtts.policy;
        let quantity = planned.quantity_units;
        if !needs_modification(
            gtt,
            &leg,
            quantity,
            slice_number,
            &policy,
            terms.tick_rupees,
        )? {
            return Ok(());
        }
        self.gtts.book.apply(id, GttAction::Modify)?;
        if let Some(gtt) = self.gtts.book.get_mut(id) {
            gtt.leg = leg;
            gtt.quantity_units = quantity;
            gtt.last_modified_slice = Some(slice_number);
            gtt.eligible_from = slice.close;
        }
        self.emit_gtt(stamp, GttEventName::Modified, id);
        Ok(())
    }

    /// Step 1: at a session open every active GTT older than the policy allows is
    /// cancelled and placed again with the same trigger and quantity (decision GTT-2), so a
    /// position held past a broker's GTT expiry stays protected. The re-created GTT is
    /// in force from step 3 of this slice. A live session renews its legs' stop GTTs
    /// through the venue (`venue_gtt_sync::renew_old_stop_gtts`).
    ///
    /// Errors: `Exec` for a GTT the book cannot move (a kernel defect).
    pub(super) fn renew_old_gtts(&mut self, slice: &SliceInfo) -> Result<(), KernelError> {
        if !slice.opens_session {
            return Ok(());
        }
        if self.is_external() {
            return self.renew_old_stop_gtts(slice);
        }
        let stamp = Self::stamp(slice, Step::SessionOpen);
        let policy = self.gtts.policy;
        let due: Vec<GttId> = self
            .gtts
            .book
            .active()
            .filter(|gtt| policy.is_due_for_renewal(gtt.created_on, slice.day))
            .map(|gtt| gtt.id)
            .collect();
        for id in due {
            self.renew(stamp, slice, id)?;
        }
        Ok(())
    }

    fn renew(
        &mut self,
        stamp: EventStamp,
        slice: &SliceInfo,
        id: GttId,
    ) -> Result<(), KernelError> {
        let Some(old) = self.gtts.book.get(id).cloned() else {
            return Ok(());
        };
        self.gtts.book.apply(id, GttAction::Expire)?;
        self.emit_gtt(stamp, GttEventName::Canceled, id);
        let renewed = self.gtts.book.next_id();
        self.gtts.book.insert(Gtt {
            id: renewed,
            created_on: slice.day,
            eligible_from: slice.open,
            last_modified_slice: None,
            ..old
        })?;
        self.emit_gtt(stamp, GttEventName::Placed, renewed);
        Ok(())
    }
}
