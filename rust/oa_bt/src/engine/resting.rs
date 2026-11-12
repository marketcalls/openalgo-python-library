//! Resting DAY orders in step 3: LIMIT, SL (stop-limit) and SL-M (stop-market),
//! matched against the base bar (03-architecture 4.2 step 3, 04-execution 4.1 and 4.2).
//!
//! A resting order is tested in every slice from its first eligible bar (the first base
//! bar of its instrument that opens at or after its decision, SPEC-35) until it fills,
//! is cancelled or expires at its session close (step 8). Instruments are taken in
//! registration order, orders in id order. Where in the bar a price is reached is
//! `exec::matching`; the fill goes through `fills::fill_order`, so the checks of the
//! fill bar (the position, the entry window, margin) apply as for any order.
//!
//! | Order | Fills | Pricing |
//! |---|---|---|
//! | LIMIT | at the open when the open is at the limit or better, else at the limit when the range reaches it | no slippage, never worse than the limit |
//! | SL-M | triggered at the open on a gap, else at the trigger inside the range, and fills there | a market order: plus slippage |
//! | SL | triggered as SL-M; its limit then fills at the price that fired it when marketable there, or at the limit when the bar trades back to it | plus slippage held at the limit (GTT-5); at the limit, none |
//!
//! A triggered SL that has not filled within its bar rests as a LIMIT from the next
//! bar, except an exit: an exit triggered but not filled within one base bar is
//! converted to a marketable order and fills at the next base-bar open plus slippage
//! (04-execution 4.1, the live rule of 08-live-trading 6.2). Both stay DAY orders: the
//! session close still ends them.
//!
//! Positions an order opens in step 3 are first tested against their levels in the
//! next slice: the bar's prices say nothing of the order of what happened after the
//! fill, so the same bar is not used twice.

use crate::exec::matching::{limit_after_trigger, limit_point, trigger_point};
use crate::exec::{
    FillPricing, InstrumentId, Order, OrderAction, OrderId, OrderStatus, OrderType,
    TriggerDirection,
};
use crate::model::Side;
use crate::portfolio::PositionKey;
use crate::rms::BarPrices;

use super::clock::SliceInfo;
use super::error::KernelError;
use super::event::{Event, OrderEventKind, Step};
use super::kernel::Engine;
use super::order_fill::OrderFill;

impl Engine {
    /// Step 3: every resting DAY order against its instrument's bar.
    ///
    /// Errors: those of `fill_order`.
    pub(super) fn test_resting_orders(&mut self, slice: &SliceInfo) -> Result<(), KernelError> {
        for index in 0..self.instruments.len() {
            let instrument = InstrumentId(index as u32);
            let Some(bar) = self.market.bar(instrument, slice.index) else {
                continue;
            };
            for id in self.book.resting_for(instrument) {
                self.test_resting(slice, id, &bar)?;
            }
        }
        Ok(())
    }

    fn test_resting(
        &mut self,
        slice: &SliceInfo,
        id: OrderId,
        bar: &BarPrices,
    ) -> Result<(), KernelError> {
        let Some(order) = self.book.get(id).cloned() else {
            return Err(KernelError::UnknownOrder(id));
        };
        if slice.open < order.decided_at {
            return Ok(());
        }
        let side = order.request.side;
        match order.request.order_type {
            OrderType::Limit { limit_rupees } => {
                self.rest_as_limit(slice, &order, limit_rupees, bar)
            }
            OrderType::StopLimit { limit_rupees, .. } if order.status == OrderStatus::Triggered => {
                self.rest_as_limit(slice, &order, limit_rupees, bar)
            }
            OrderType::StopMarket { trigger_rupees } => {
                let direction = TriggerDirection::of_stop(side);
                let Some(point) = trigger_point(direction, trigger_rupees, bar) else {
                    return Ok(());
                };
                self.trigger_order(slice, id)?;
                let how = OrderFill::in_bar(point, FillPricing::Market);
                self.fill_order(slice, &order, how)
            }
            OrderType::StopLimit {
                trigger_rupees,
                limit_rupees,
            } => {
                let direction = TriggerDirection::of_stop(side);
                let Some(fired) = trigger_point(direction, trigger_rupees, bar) else {
                    return Ok(());
                };
                self.trigger_order(slice, id)?;
                match limit_after_trigger(side, limit_rupees, fired, bar).fill(limit_rupees) {
                    Some((point, pricing)) => {
                        self.fill_order(slice, &order, OrderFill::in_bar(point, pricing))
                    }
                    None if self.is_exit(&order) => Ok(self.book.queue(id)?),
                    None => Ok(()),
                }
            }
            OrderType::Market => Ok(()),
        }
    }

    /// A LIMIT through the whole bar: at the open or at its limit, no slippage.
    fn rest_as_limit(
        &mut self,
        slice: &SliceInfo,
        order: &Order,
        limit_rupees: f64,
        bar: &BarPrices,
    ) -> Result<(), KernelError> {
        match limit_point(order.request.side, limit_rupees, bar) {
            Some(point) => {
                self.fill_order(slice, order, OrderFill::in_bar(point, FillPricing::Resting))
            }
            None => Ok(()),
        }
    }

    /// A stop order's trigger was reached: `Triggered`.
    fn trigger_order(&mut self, slice: &SliceInfo, id: OrderId) -> Result<(), KernelError> {
        self.book.apply(id, OrderAction::Trigger)?;
        self.emit(Event::Order {
            stamp: Self::stamp(slice, Step::Levels),
            order: id,
            kind: OrderEventKind::Triggered,
        });
        Ok(())
    }

    /// True when the order would reduce its position now: an exit.
    fn is_exit(&self, order: &Order) -> bool {
        let request = &order.request;
        let held = self.held_units(PositionKey {
            instrument: request.instrument,
            product: request.product,
        });
        let reduces = held != 0.0 && (held > 0.0) != (request.side == Side::Buy);
        request.reduce_only || reduces
    }
}
