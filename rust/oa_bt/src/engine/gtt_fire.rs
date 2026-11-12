//! GTTs against the bar (step 3) and the engine's exit after an unfilled child (step 2),
//! 04-execution 4.4 items 3 and 4.
//!
//! In step 3 every active GTT eligible in the slice is tested against its instrument's
//! bar, instruments in registration order, GTTs in id order:
//!
//! 1. A trigger crossed at the open fires first; with both of an OCO's triggers inside
//!    the range the stop fires (item 4). The other leg of an OCO is cancelled with it.
//!    `GttTriggered`, then for a protective GTT the leg's risk event (`leg_sl_hit`,
//!    `leg_target_hit`, decision GTT-8).
//! 2. The child LIMIT, beyond the trigger by the segment buffer, fills by
//!    `exec::matching::limit_after_trigger` (GTT-5, GTT-7): an engine-held exit with
//!    reason `leg_sl` or `leg_target`, the GTT's id and no order (`GttChildFilled`
//!    before the position event); for a strategy GTT, the strategy's order fills.
//! 3. A child that does not fill through its bar is cancelled: `gtt_child_unfilled`
//!    (level: the limit; observed: the best price after the trigger), then
//!    `GttChildCanceled`, and only after that cancel the engine's own exit waits for
//!    the instrument's next base-bar open, where it fills plus slippage, flagged
//!    `gtt_child_unfilled` (LEAD-10). A backtest assumes the engine runs at every
//!    session open, so that open is the next bar's, whatever the session (GTT-4).

use crate::exec::matching::{limit_after_trigger, trigger_point};
use crate::exec::{
    BarPoint, FillAt, FillFlags, FillPricing, Gtt, GttAction, GttEvent, GttEventName, GttId,
    GttLeg, GttLegRole, GttPurpose, GttSlot, GttStatus, InstrumentId, OrderAction, OrderId,
};
use crate::portfolio::PositionKey;
use crate::rms::{BarPrices, ExitReason, RiskEvent, RiskEventName, RiskScope};

use super::clock::SliceInfo;
use super::error::KernelError;
use super::event::{Event, EventStamp, OrderEventKind, Step};
use super::fills::FillRequest;
use super::gtt::FollowUpExit;
use super::kernel::Engine;
use super::order_fill::OrderFill;

/// The leg that fires in a bar, and where: a trigger crossed at the open first, else
/// the stop when both triggers lie inside the range (04-execution 4.4 item 4).
fn fired_leg(gtt: &Gtt, bar: &BarPrices) -> Option<(GttLeg, BarPoint)> {
    let reached: Vec<(GttLeg, BarPoint)> = gtt
        .legs
        .iter()
        .filter_map(|leg| trigger_point(leg.direction, leg.trigger_rupees, bar).map(|p| (*leg, p)))
        .collect();
    let at_open = reached.iter().find(|(_, point)| point.at == FillAt::Open);
    let stop = reached.iter().find(|(leg, _)| leg.role == GttLegRole::Stop);
    at_open.or(stop).or(reached.first()).copied()
}

/// The exit reason of a protective GTT's child, and of the engine's exit after an
/// unfilled child (decision GTT-8 reading A): `leg_sl` for the stop, `leg_target` for
/// the target.
fn exit_reason_of(role: GttLegRole) -> ExitReason {
    match role {
        GttLegRole::Target => ExitReason::LegTarget,
        GttLegRole::Stop | GttLegRole::Order => ExitReason::LegSl,
    }
}

/// The target index a fill of a GTT's leg reports.
fn target_index_of(gtt: &Gtt, role: GttLegRole) -> Option<u16> {
    match (gtt.purpose, role) {
        (GttPurpose::Protect(GttSlot::Target(index)), GttLegRole::Target) => Some(index),
        _ => None,
    }
}

impl Engine {
    /// Step 3: every active GTT eligible in this slice against its instrument's bar.
    ///
    /// Errors: those of `execute_fill` and of the leg rules.
    pub(super) fn test_gtts(&mut self, slice: &SliceInfo) -> Result<(), KernelError> {
        for index in 0..self.instruments.len() {
            let instrument = InstrumentId(index as u32);
            let Some(bar) = self.market.bar(instrument, slice.index) else {
                continue;
            };
            for id in self.gtts.book.active_for(instrument) {
                self.test_gtt(slice, id, &bar, Step::Levels)?;
            }
        }
        Ok(())
    }

    /// Test one GTT against a bar, its events stamped with `step` (3; a futures roll
    /// fires a trigger crossed at the near contract's open in step 2, GTT-9).
    pub(super) fn test_gtt(
        &mut self,
        slice: &SliceInfo,
        id: GttId,
        bar: &BarPrices,
        step: Step,
    ) -> Result<(), KernelError> {
        let Some(gtt) = self.gtts.book.get(id).cloned() else {
            return Ok(());
        };
        if !gtt.is_active() || slice.open < gtt.eligible_from {
            return Ok(());
        }
        let Some((leg, fired)) = fired_leg(&gtt, bar) else {
            return Ok(());
        };
        let stamp = Self::stamp(slice, step);
        let key = PositionKey {
            instrument: gtt.instrument,
            product: gtt.product,
        };
        let protects = matches!(gtt.purpose, GttPurpose::Protect(_));
        if protects && self.held_units(key) == 0.0 {
            // A flat position holds no GTT.
            return self.cancel_gtt(stamp, id);
        }
        self.gtts.book.apply(id, GttAction::Fire)?;
        if let Some(fired_gtt) = self.gtts.book.get_mut(id) {
            fired_gtt.fired = Some(leg.role);
        }
        self.emit_gtt(stamp, GttEventName::Triggered, id);
        let outcome = limit_after_trigger(gtt.side, leg.child_limit_rupees, fired, bar);
        let filled = outcome.fill(leg.child_limit_rupees);
        match gtt.purpose {
            GttPurpose::Protect(_) => {
                self.emit_level_hit(stamp, &gtt, leg);
                match filled {
                    Some((point, pricing)) => {
                        self.fill_protective(stamp, &gtt, leg, point, pricing)
                    }
                    None => self.child_unfilled(stamp, &gtt, leg, outcome.best_after_trigger()),
                }
            }
            GttPurpose::Order(order) => match filled {
                Some((point, pricing)) => {
                    self.fill_order_gtt(slice, stamp, &gtt, order, point, pricing)
                }
                None => {
                    self.cancel_child(stamp, id)?;
                    let kind = OrderEventKind::Canceled;
                    self.end_order(stamp, order, OrderAction::Cancel, kind)
                }
            },
        }
    }

    /// The leg's risk event when a protective GTT fires: `leg_sl_hit` or
    /// `leg_target_hit` with the trigger as the level.
    fn emit_level_hit(&mut self, stamp: EventStamp, gtt: &Gtt, leg: GttLeg) {
        let name = match leg.role {
            GttLegRole::Target => RiskEventName::LegTargetHit,
            GttLegRole::Stop | GttLegRole::Order => RiskEventName::LegSlHit,
        };
        let event = RiskEvent {
            name,
            scope: RiskScope::Leg,
            level_rupees: Some(leg.trigger_rupees),
            target_index: target_index_of(gtt, leg.role),
            observed_rupees: None,
        };
        self.emit(Event::Risk {
            stamp,
            instrument: Some(gtt.instrument),
            event,
        });
    }

    /// A protective GTT's child filled: the engine-held exit of its quantity.
    fn fill_protective(
        &mut self,
        stamp: EventStamp,
        gtt: &Gtt,
        leg: GttLeg,
        point: BarPoint,
        pricing: FillPricing,
    ) -> Result<(), KernelError> {
        let key = PositionKey {
            instrument: gtt.instrument,
            product: gtt.product,
        };
        let held = self.held_units(key);
        let reason = exit_reason_of(leg.role);
        let exit = FillRequest::engine_exit(key, held, point.price_rupees, reason).at(point.at);
        let target_index = target_index_of(gtt, leg.role);
        let request = FillRequest {
            units: gtt.quantity_units.min(held.abs()),
            target_index,
            pricing,
            gtt: Some(gtt.id),
            ..exit
        };
        self.execute_fill(stamp, request)?;
        if target_index.is_some() {
            let moved = match self.legs.iter_mut().find(|(k, _)| *k == key) {
                Some((_, state)) => state.after_target_fill()?,
                None => None,
            };
            if let Some(event) = moved {
                let instrument = Some(gtt.instrument);
                self.emit(Event::Risk {
                    stamp,
                    instrument,
                    event,
                });
            }
        }
        Ok(())
    }

    /// A protective GTT's child did not fill through its bar: the risk event, the
    /// child's cancel, and the engine's exit queued for the next base-bar open.
    fn child_unfilled(
        &mut self,
        stamp: EventStamp,
        gtt: &Gtt,
        leg: GttLeg,
        observed_rupees: Option<f64>,
    ) -> Result<(), KernelError> {
        let event = RiskEvent {
            name: RiskEventName::GttChildUnfilled,
            scope: RiskScope::Leg,
            level_rupees: Some(leg.child_limit_rupees),
            target_index: target_index_of(gtt, leg.role),
            observed_rupees,
        };
        self.emit(Event::Risk {
            stamp,
            instrument: Some(gtt.instrument),
            event,
        });
        self.cancel_child(stamp, gtt.id)?;
        self.gtts.follow_ups.push(FollowUpExit {
            key: PositionKey {
                instrument: gtt.instrument,
                product: gtt.product,
            },
            gtt: gtt.id,
            units: gtt.quantity_units,
            reason: exit_reason_of(leg.role),
        });
        Ok(())
    }

    /// The child of a fired GTT did not fill: `GttChildCanceled`.
    fn cancel_child(&mut self, stamp: EventStamp, id: GttId) -> Result<(), KernelError> {
        self.gtts.book.apply(id, GttAction::CancelChild)?;
        self.emit_gtt(stamp, GttEventName::ChildCanceled, id);
        Ok(())
    }

    /// A strategy GTT's child filled: the strategy's order fills through the checks of
    /// the fill bar; when one refuses it, the child is cancelled.
    fn fill_order_gtt(
        &mut self,
        slice: &SliceInfo,
        stamp: EventStamp,
        gtt: &Gtt,
        order: OrderId,
        point: BarPoint,
        pricing: FillPricing,
    ) -> Result<(), KernelError> {
        let Some(placed) = self.book.get(order).cloned() else {
            return Err(KernelError::UnknownOrder(order));
        };
        let how = OrderFill {
            gtt: Some(gtt.id),
            ..OrderFill::in_bar(point, pricing)
        };
        self.fill_order(slice, &placed, how)?;
        let unresolved = self
            .gtts
            .book
            .get(gtt.id)
            .is_some_and(|g| g.status == GttStatus::Triggered);
        if unresolved {
            self.cancel_child(stamp, gtt.id)?;
        }
        Ok(())
    }

    /// The fill hook: a fill that is a fired GTT's child resolves it with
    /// `GttChildFilled` (written before the position event). The engine's exit after an
    /// unfilled child also carries the GTT's id; its GTT is already spent.
    pub(super) fn gtt_child_filled(
        &mut self,
        stamp: EventStamp,
        id: GttId,
        price_rupees: f64,
    ) -> Result<(), KernelError> {
        let fired = self
            .gtts
            .book
            .get(id)
            .is_some_and(|gtt| gtt.status == GttStatus::Triggered);
        if !fired {
            return Ok(());
        }
        self.gtts.book.apply(id, GttAction::FillChild)?;
        if let Some(gtt) = self.gtts.book.get(id) {
            let event = GttEvent::child_filled(gtt, price_rupees);
            self.emit(Event::Gtt { stamp, event });
        }
        Ok(())
    }

    /// Step 2, before the strategy's orders: the engine's exits after unfilled children
    /// of this instrument fill at this open plus slippage, flagged `gtt_child_unfilled`.
    ///
    /// Errors: those of `execute_fill`.
    pub(super) fn gtt_exits_at_open(
        &mut self,
        slice: &SliceInfo,
        instrument: InstrumentId,
        open: f64,
    ) -> Result<(), KernelError> {
        let (due, waiting): (Vec<FollowUpExit>, Vec<FollowUpExit>) = self
            .gtts
            .follow_ups
            .iter()
            .partition(|exit| exit.key.instrument == instrument);
        self.gtts.follow_ups = waiting;
        for exit in due {
            let held = self.held_units(exit.key);
            if held == 0.0 {
                continue;
            }
            let request = FillRequest {
                units: exit.units.min(held.abs()),
                gtt: Some(exit.gtt),
                ..FillRequest::engine_exit(exit.key, held, open, exit.reason)
                    .flagged(FillFlags::GTT_CHILD_UNFILLED)
            };
            self.execute_fill(Self::stamp(slice, Step::OpenFills), request)?;
        }
        Ok(())
    }
}
