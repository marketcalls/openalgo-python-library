//! GTT orders for positional trades in the slice sequence (04-execution 4.4): the GTTs
//! that hold a CNC or NRML position's stop and target, their life between fills, and
//! the session-open check.
//!
//! | When | What | Step |
//! |---|---|---|
//! | A fill opens a CNC or NRML position with levels | its GTTs are placed (`GttPlaced`), in force from step 3 of a step-2 fill (SPEC-02), from the next slice otherwise | the fill's |
//! | The leg's levels or quantity change (trail, breakeven, an add, a partial exit, a target) | the GTTs follow by `gtt_plan`: modified (`GttModified`, GTT-3), placed or cancelled; in force from the next slice | 4 |
//! | Any exit that is not a GTT's own (a signal, the daily loss limit, the drawdown halt, the time exit) | every GTT on the position is cancelled first (`GttCanceled`), then the exit fills (cancel-then-exit, GTT-9) | the exit's |
//! | A strategy GTT order ends (cancelled, rejected) | its GTT is cancelled | the order's |
//! | A session opens | every GTT older than 360 days is cancelled and placed again (GTT-2); a backtest assumes the engine runs at every session open (GTT-4) | 1 |
//!
//! Firing in step 3 and the engine's exit after an unfilled child are in `gtt_fire`;
//! strategy GTT orders (`validity="gtt"`) in `gtt_orders`.

use crate::exec::{
    Gtt, GttAction, GttBook, GttEvent, GttEventName, GttId, GttPolicy, GttPurpose, GttStatus,
    OrderId,
};
use crate::model::{Product, Side, TsNanos};
use crate::portfolio::{FillEffect, FillKind, PositionKey};
use crate::rms::ExitReason;

use super::clock::SliceInfo;
use super::error::KernelError;
use super::event::{Event, EventStamp, Step};
use super::fills::FillRequest;
use super::gtt_plan::{kind_of, needs_modification, protective_plan, ChildTerms, PlannedGtt};
use super::kernel::Engine;

/// The engine's own exit after a GTT child did not fill through its bar, waiting for
/// the instrument's next base-bar open (04-execution 4.4 item 3, LEAD-10).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct FollowUpExit {
    pub(super) key: PositionKey,
    pub(super) gtt: GttId,
    pub(super) units: f64,
    pub(super) reason: ExitReason,
}

/// The GTTs of a run and what the slice sequence keeps about them.
#[derive(Clone, Debug, Default, PartialEq)]
pub(super) struct GttDesk {
    pub(super) book: GttBook,
    pub(super) policy: GttPolicy,
    /// Exits waiting for a next base-bar open, in the order the children failed.
    pub(super) follow_ups: Vec<FollowUpExit>,
}

/// True when a product holds its stops and targets as GTTs: CNC and NRML, which carry
/// overnight. By decision GTT-10 reading A, 04-execution 4.4 then governs their stops
/// and targets (a trigger and a child LIMIT) instead of the engine-held rule that a
/// level crossed at the open fills at the open, which stays for MIS.
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

    /// How the children of `key`'s GTTs are priced.
    fn child_terms(&self, key: PositionKey) -> Result<ChildTerms, KernelError> {
        let spec = self.spec(key.instrument)?;
        let held = self.held_units(key);
        Ok(ChildTerms {
            position_side: if held < 0.0 { Side::Sell } else { Side::Buy },
            tick_rupees: spec.tick_size_rupees,
            buffer_percent: self.gtts.policy.child_buffer_percent(spec.kind),
        })
    }

    /// After every fill: a fill that opens a CNC or NRML position with levels places
    /// the GTTs that hold them, right after the fill's events.
    ///
    /// Errors: `Exec` for a level that is not a usable price.
    pub(super) fn place_gtts_after_entry(
        &mut self,
        stamp: EventStamp,
        key: PositionKey,
        effect: &FillEffect,
    ) -> Result<(), KernelError> {
        if effect.kind != FillKind::Entry || !held_as_gtt(key.product) {
            return Ok(());
        }
        let Some(leg) = self.leg(key.instrument, key.product) else {
            return Ok(());
        };
        let plan = protective_plan(leg);
        let eligible = eligible_from(stamp, &self.slice_of(stamp)?);
        for planned in plan {
            self.place_protective(stamp, key, &planned, eligible)?;
        }
        Ok(())
    }

    /// Place one planned GTT on `key` and write `GttPlaced`.
    fn place_protective(
        &mut self,
        stamp: EventStamp,
        key: PositionKey,
        planned: &PlannedGtt,
        eligible_from: TsNanos,
    ) -> Result<GttId, KernelError> {
        let terms = self.child_terms(key)?;
        let legs = terms.legs(planned)?;
        let id = self.gtts.book.next_id();
        let gtt = Gtt {
            id,
            instrument: key.instrument,
            product: key.product,
            side: terms.child_side(),
            quantity_units: planned.quantity_units,
            kind: kind_of(&legs),
            legs,
            purpose: GttPurpose::Protect(planned.slot),
            status: GttStatus::Active,
            created_on: self.trade_day(stamp),
            eligible_from,
            last_modified_slice: None,
            fired: None,
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

    /// Cancel-then-exit, decision GTT-9 reading A: before any fill that reduces a CNC or
    /// NRML position and is not a protective GTT's own exit (a signal, a strategy GTT,
    /// the daily loss limit, the drawdown halt, the time exit, the close of a reversal),
    /// every GTT holding the
    /// position's levels is cancelled in the same step, so the position never has two
    /// exit owners (08-live-trading 6.2). Levels that remain after a partial exit get
    /// new GTTs in step 4.
    pub(super) fn cancel_gtts_before_exit(
        &mut self,
        stamp: EventStamp,
        request: &FillRequest,
    ) -> Result<(), KernelError> {
        let key = request.key;
        let own_exit = request.gtt.is_some_and(|id| {
            let purpose = self.gtts.book.get(id).map(|gtt| gtt.purpose);
            matches!(purpose, Some(GttPurpose::Protect(_)))
        });
        if own_exit || !held_as_gtt(key.product) {
            return Ok(());
        }
        let held = self.held_units(key);
        let reduces = held != 0.0 && (held > 0.0) != (request.side == Side::Buy);
        if !reduces {
            return Ok(());
        }
        for id in self.gtts.book.protecting(key.instrument, key.product) {
            self.cancel_gtt(stamp, id)?;
        }
        Ok(())
    }

    /// When a strategy order ends (cancelled, rejected, expired), the GTT it rests as
    /// is cancelled first.
    pub(super) fn cancel_order_gtt(
        &mut self,
        stamp: EventStamp,
        order: OrderId,
    ) -> Result<(), KernelError> {
        match self.gtts.book.of_order(order) {
            Some(id) => self.cancel_gtt(stamp, id),
            None => Ok(()),
        }
    }

    /// Step 4, after the ratchets: every CNC or NRML leg's GTTs follow its levels and
    /// quantity. A slot whose GTT is in force is modified when GTT-3 allows, a slot
    /// with none is placed, a GTT whose slot is gone is cancelled. A position whose GTT
    /// fired and awaits the engine's exit is left alone.
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
            self.sync_position(stamp, slice, key, &plan)?;
        }
        Ok(())
    }

    fn sync_position(
        &mut self,
        stamp: EventStamp,
        slice: &SliceInfo,
        key: PositionKey,
        plan: &[PlannedGtt],
    ) -> Result<(), KernelError> {
        let active = self.gtts.book.protecting(key.instrument, key.product);
        for id in &active {
            let slot = self.gtts.book.get(*id).map(|gtt| gtt.purpose);
            let kept = plan
                .iter()
                .any(|p| slot == Some(GttPurpose::Protect(p.slot)));
            if !kept {
                self.cancel_gtt(stamp, *id)?;
            }
        }
        for planned in plan {
            let held = active.iter().copied().find(|id| {
                self.gtts.book.get(*id).map(|gtt| gtt.purpose)
                    == Some(GttPurpose::Protect(planned.slot))
            });
            match held {
                Some(id) => self.modify_protective(stamp, slice, key, id, planned)?,
                None => {
                    self.place_protective(stamp, key, planned, slice.close)?;
                }
            }
        }
        Ok(())
    }

    /// Modify a protective GTT to its planned levels and quantity when GTT-3 allows.
    fn modify_protective(
        &mut self,
        stamp: EventStamp,
        slice: &SliceInfo,
        key: PositionKey,
        id: GttId,
        planned: &PlannedGtt,
    ) -> Result<(), KernelError> {
        let terms = self.child_terms(key)?;
        let legs = terms.legs(planned)?;
        let slice_number = stamp.slice;
        let Some(gtt) = self.gtts.book.get(id) else {
            return Ok(());
        };
        let policy = self.gtts.policy;
        let quantity = planned.quantity_units;
        if !needs_modification(
            gtt,
            &legs,
            quantity,
            slice_number,
            &policy,
            terms.tick_rupees,
        )? {
            return Ok(());
        }
        self.gtts.book.apply(id, GttAction::Modify)?;
        if let Some(gtt) = self.gtts.book.get_mut(id) {
            gtt.kind = kind_of(&legs);
            gtt.legs = legs;
            gtt.quantity_units = quantity;
            gtt.last_modified_slice = Some(slice_number);
            gtt.eligible_from = slice.close;
        }
        self.emit_gtt(stamp, GttEventName::Modified, id);
        Ok(())
    }

    /// Step 1: at a session open every active GTT older than the policy allows is
    /// cancelled and placed again with the same legs and quantity (decision GTT-2), so a
    /// position held past a broker's GTT expiry stays protected. The re-created GTT is
    /// in force from step 3 of this slice.
    ///
    /// Errors: `Exec` for a GTT the book cannot move (a kernel defect).
    pub(super) fn renew_old_gtts(&mut self, slice: &SliceInfo) -> Result<(), KernelError> {
        if !slice.opens_session {
            return Ok(());
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
