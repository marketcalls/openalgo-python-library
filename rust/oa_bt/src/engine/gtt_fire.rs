//! GTTs against the bar (step 3) and the engine's exit after an unfilled child (step 2),
//! 04-execution 4.4 item 3.
//!
//! A position's stop GTT is fired through its leg's test (`gtt_levels`), which puts the
//! stop and the engine's targets in the order of 03-architecture 4.2 step 3. Strategy GTT
//! orders (`validity="gtt"`) are tested after every leg (`test_gtts`), instruments in
//! registration order, GTTs in id order. A GTT that fires:
//!
//! 1. Fires at the open when the bar opens at or beyond its trigger, else at the trigger
//!    inside the range: `GttTriggered`, then for a stop GTT the leg's risk event
//!    (`leg_sl_hit`, decision GTT-8).
//! 2. The child LIMIT, beyond the trigger by the segment buffer, fills by
//!    `exec::matching::limit_after_trigger` (GTT-5, GTT-7): for a stop GTT an engine-held
//!    exit with reason `leg_sl`, the GTT's id and no order (`GttChildFilled` before the
//!    position event); for a strategy GTT, the strategy's order fills.
//! 3. A child that does not fill through its bar is cancelled: `gtt_child_unfilled`
//!    (level: the limit; observed: the best price after the trigger), then
//!    `GttChildCanceled`, and only after that cancel the engine's own exit waits for
//!    the instrument's next base-bar open, where it fills plus slippage, flagged
//!    `gtt_child_unfilled` (LEAD-10). A backtest assumes the engine runs at every
//!    session open, so that open is the next bar's, whatever the session (GTT-4).

use crate::exec::matching::{limit_after_trigger, trigger_point};
use crate::exec::{
    BarPoint, FillFlags, FillPricing, Gtt, GttAction, GttEvent, GttEventName, GttId, GttPurpose,
    GttStatus, InstrumentId, OrderAction, OrderId,
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

/// The exit reason of a stop GTT's child, and of the engine's exit after an unfilled
/// child (decision GTT-8 reading A).
const STOP_EXIT_REASON: ExitReason = ExitReason::LegSl;

impl Engine {
    /// Step 3, after every leg: the strategy GTT orders eligible in this slice against
    /// their instrument's bar. A position's stop GTT fires through its leg (`gtt_levels`).
    ///
    /// Errors: those of `execute_fill` and of the fill-bar checks.
    pub(super) fn test_gtts(&mut self, slice: &SliceInfo) -> Result<(), KernelError> {
        for index in 0..self.instruments.len() {
            let instrument = InstrumentId(index as u32);
            let Some(bar) = self.market.bar(instrument, slice.index) else {
                continue;
            };
            for id in self.gtts.book.active_for(instrument) {
                let protects = self
                    .gtts
                    .book
                    .get(id)
                    .is_some_and(|gtt| gtt.purpose == GttPurpose::Protect);
                if !protects {
                    self.test_gtt(slice, id, &bar, Step::Levels)?;
                }
            }
        }
        Ok(())
    }

    /// Test one GTT against a bar, its events stamped with `step`: in step 3 a strategy
    /// GTT order (`test_gtts`) or a position's stop GTT that its leg's test found reached
    /// (`gtt_levels`); in step 2 a stop crossed at a futures roll's near-contract open
    /// (GTT-9).
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
        let leg = gtt.leg;
        let Some(fired) = trigger_point(leg.direction, leg.trigger_rupees, bar) else {
            return Ok(());
        };
        let stamp = Self::stamp(slice, step);
        let key = PositionKey {
            instrument: gtt.instrument,
            product: gtt.product,
        };
        let protects = gtt.purpose == GttPurpose::Protect;
        if protects && self.held_units(key) == 0.0 {
            // A flat position holds no GTT.
            return self.cancel_gtt(stamp, id);
        }
        self.gtts.book.apply(id, GttAction::Fire)?;
        self.emit_gtt(stamp, GttEventName::Triggered, id);
        let outcome = limit_after_trigger(gtt.side, leg.child_limit_rupees, fired, bar);
        let filled = outcome.fill(leg.child_limit_rupees);
        match gtt.purpose {
            GttPurpose::Protect => {
                self.emit_stop_hit(stamp, &gtt);
                match filled {
                    Some((point, pricing)) => self.fill_protective(stamp, &gtt, point, pricing),
                    None => self.child_unfilled(stamp, &gtt, outcome.best_after_trigger()),
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

    /// The leg's risk event when its stop GTT fires: `leg_sl_hit` with the trigger as
    /// the level.
    fn emit_stop_hit(&mut self, stamp: EventStamp, gtt: &Gtt) {
        let event = RiskEvent {
            name: RiskEventName::LegSlHit,
            scope: RiskScope::Leg,
            level_rupees: Some(gtt.leg.trigger_rupees),
            target_index: None,
            observed_rupees: None,
        };
        self.emit(Event::Risk {
            stamp,
            instrument: Some(gtt.instrument),
            event,
        });
    }

    /// A stop GTT's child filled: the engine-held exit of its quantity.
    fn fill_protective(
        &mut self,
        stamp: EventStamp,
        gtt: &Gtt,
        point: BarPoint,
        pricing: FillPricing,
    ) -> Result<(), KernelError> {
        let key = PositionKey {
            instrument: gtt.instrument,
            product: gtt.product,
        };
        let held = self.held_units(key);
        let exit =
            FillRequest::engine_exit(key, held, point.price_rupees, STOP_EXIT_REASON).at(point.at);
        let request = FillRequest {
            units: gtt.quantity_units.min(held.abs()),
            pricing,
            gtt: Some(gtt.id),
            ..exit
        };
        self.execute_fill(stamp, request)?;
        Ok(())
    }

    /// A stop GTT's child did not fill through its bar: the risk event, the child's
    /// cancel, and the engine's exit queued for the next base-bar open.
    fn child_unfilled(
        &mut self,
        stamp: EventStamp,
        gtt: &Gtt,
        observed_rupees: Option<f64>,
    ) -> Result<(), KernelError> {
        let event = RiskEvent {
            name: RiskEventName::GttChildUnfilled,
            scope: RiskScope::Leg,
            level_rupees: Some(gtt.leg.child_limit_rupees),
            target_index: None,
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
                ..FillRequest::engine_exit(exit.key, held, open, STOP_EXIT_REASON)
                    .flagged(FillFlags::GTT_CHILD_UNFILLED)
            };
            self.execute_fill(Self::stamp(slice, Step::OpenFills), request)?;
        }
        Ok(())
    }
}
