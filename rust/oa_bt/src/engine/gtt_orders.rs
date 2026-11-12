//! A strategy's own GTT orders (`validity="gtt"`, CNC and NRML, 14-strategy-api 2.4): a
//! resting entry, add or exit that waits for days.
//!
//! The order stays in the order book (the strategy holds its handle; it never expires
//! at a session close) and rests as a SINGLE GTT placed with it in step 7: its trigger
//! is the order's trigger price (`trigger_price=`), or its price when it has none, and
//! its child LIMIT is the order's price when both are given, else the trigger moved by
//! the segment buffer (GTT-1). The trigger waits on the side of the decision price it
//! was placed from: one above the last close fires on a rise, one below on a fall. The
//! GTT is tested from the first base bar at or after the decision (SPEC-35); when it
//! fires, the order fills through the checks of the fill bar at the child's price
//! (`gtt_fire`). A modification replaces the GTT's trigger and child in full
//! (`GttModified`), from the next base bar; cancelling the order cancels the GTT.

use crate::exec::gtt::child_limit;
use crate::exec::segment::venue_order_type;
use crate::exec::{
    ExecError, Gtt, GttAction, GttEventName, GttKind, GttLeg, GttLegRole, GttPurpose, GttStatus,
    OrderId, OrderRequest, OrderType, TriggerDirection, Validity,
};

use super::clock::SliceInfo;
use super::error::KernelError;
use super::event::Step;
use super::kernel::Engine;

impl Engine {
    /// The request as the instrument's segment accepts it (04-execution 4.1): an
    /// options SL-M becomes an SL with its limit beyond the trigger.
    ///
    /// Errors: `Exec` for an unusable trigger.
    pub(super) fn venue_request(&self, request: OrderRequest) -> Result<OrderRequest, KernelError> {
        let Some(spec) = self.instruments.get(request.instrument.index()) else {
            return Ok(request);
        };
        let tick = spec.tick_size_rupees;
        let order_type = venue_order_type(spec.kind, request.order_type, request.side, tick)?;
        Ok(OrderRequest {
            order_type,
            ..request
        })
    }

    /// The single leg of a GTT order: its trigger, the side the price reaches it from
    /// (the decision price, the last close), and its child limit.
    ///
    /// Errors: `Exec` for a price that is unusable or an order with no last close to
    /// place its trigger against.
    fn order_gtt_leg(&self, order: OrderId) -> Result<GttLeg, KernelError> {
        let placed = self
            .book
            .get(order)
            .ok_or(KernelError::UnknownOrder(order))?;
        let request = &placed.request;
        let spec = self.spec(request.instrument)?;
        let (trigger, limit) = match request.order_type {
            OrderType::Limit { limit_rupees } => (limit_rupees, None),
            OrderType::StopLimit {
                trigger_rupees,
                limit_rupees,
            } => (trigger_rupees, Some(limit_rupees)),
            OrderType::StopMarket { trigger_rupees } => (trigger_rupees, None),
            OrderType::Market => return Err(not_resting()),
        };
        let decision_price = self
            .last_close(request.instrument)
            .ok_or_else(|| KernelError::Exec(no_decision_price()))?;
        let buffer = self.gtts.policy.child_buffer_percent(spec.kind);
        let child = match limit {
            Some(limit) => limit,
            None => child_limit(trigger, request.side, buffer, spec.tick_size_rupees)?,
        };
        Ok(GttLeg {
            role: GttLegRole::Order,
            trigger_rupees: trigger,
            direction: TriggerDirection::toward(trigger, decision_price),
            child_limit_rupees: child,
        })
    }

    /// Step 7: a GTT order that passed the submission checks rests as its GTT
    /// (`GttPlaced` after `Submitted`). Other orders place nothing.
    ///
    /// Errors: as `order_gtt_leg`.
    pub(super) fn place_order_gtt(
        &mut self,
        slice: &SliceInfo,
        order: OrderId,
    ) -> Result<(), KernelError> {
        let Some(placed) = self.book.get(order).cloned() else {
            return Err(KernelError::UnknownOrder(order));
        };
        if placed.request.validity != Validity::Gtt {
            return Ok(());
        }
        let leg = self.order_gtt_leg(order)?;
        let id = self.gtts.book.next_id();
        self.gtts.book.insert(Gtt {
            id,
            instrument: placed.request.instrument,
            product: placed.request.product,
            side: placed.request.side,
            quantity_units: placed.quantity_units,
            kind: GttKind::Single,
            legs: vec![leg],
            purpose: GttPurpose::Order(order),
            status: GttStatus::Active,
            created_on: slice.day,
            eligible_from: placed.decided_at,
            last_modified_slice: None,
            fired: None,
        })?;
        self.emit_gtt(
            Self::stamp(slice, Step::Callbacks),
            GttEventName::Placed,
            id,
        );
        Ok(())
    }

    /// Step 7, after a modification of a GTT order: its GTT takes the new trigger,
    /// child and quantity in full (`GttModified`), tested from the next base bar.
    ///
    /// Errors: as `order_gtt_leg`.
    pub(super) fn modify_order_gtt(
        &mut self,
        slice: &SliceInfo,
        order: OrderId,
    ) -> Result<(), KernelError> {
        let Some(id) = self.gtts.book.of_order(order) else {
            return Ok(());
        };
        let leg = self.order_gtt_leg(order)?;
        let quantity = self.book.get(order).map_or(0.0, |o| o.quantity_units);
        self.gtts.book.apply(id, GttAction::Modify)?;
        if let Some(gtt) = self.gtts.book.get_mut(id) {
            gtt.legs = vec![leg];
            gtt.quantity_units = quantity;
            gtt.eligible_from = slice.close;
            gtt.last_modified_slice = u32::try_from(slice.index).ok();
        }
        let stamp = Self::stamp(slice, Step::Callbacks);
        self.emit_gtt(stamp, GttEventName::Modified, id);
        Ok(())
    }
}

fn not_resting() -> KernelError {
    KernelError::Exec(ExecError::InvalidRequest {
        reason: "a GTT order rests until a trigger: give price= (and optionally trigger_price=)"
            .to_string(),
    })
}

fn no_decision_price() -> ExecError {
    ExecError::InvalidRequest {
        reason: "a GTT order is placed against the last close, and the instrument has none \
                 yet: place it after its first bar"
            .to_string(),
    }
}
