//! The order book of a run: every order by id, and the queue of orders waiting for an
//! open.
//!
//! Orders are stored in submission order and never removed, so a result can list
//! them all. The queue holds the orders that fill at the next open, in id order: the
//! working MARKET orders, and a stop-limit exit converted to a marketable order after it
//! triggered without filling (04-execution 4.1). Step 2 of every slice walks it per
//! instrument; step 3 walks the resting DAY orders (LIMIT, SL, SL-M) the same way, which
//! keeps fills deterministic.

use crate::model::TsNanos;

use super::error::ExecError;
use super::instrument::InstrumentId;
use super::order::{Order, OrderId, OrderRequest, OrderType, Validity};
use super::state::{next_status, OrderAction, OrderStatus};

/// Every order of a run, and the open queue.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct OrderBook {
    orders: Vec<Order>,
    queued: Vec<OrderId>,
}

impl OrderBook {
    /// The id the next order will get: `o1` for the first.
    pub fn next_id(&self) -> OrderId {
        OrderId(self.orders.len() as u64 + 1)
    }

    /// Add an order. A working MARKET order joins the open queue.
    ///
    /// Errors: `InvalidRequest` when the order's id is not `next_id` (ids are dense and
    /// in submission order).
    pub fn insert(&mut self, order: Order) -> Result<OrderId, ExecError> {
        if order.id != self.next_id() {
            return Err(ExecError::InvalidRequest {
                reason: format!("order id {} is out of sequence", order.id),
            });
        }
        let id = order.id;
        if order.status.is_working() && order.request.order_type == OrderType::Market {
            self.queued.push(id);
        }
        self.orders.push(order);
        Ok(id)
    }

    fn slot(id: OrderId) -> Option<usize> {
        usize::try_from(id.0).ok()?.checked_sub(1)
    }

    /// The order with this id.
    pub fn get(&self, id: OrderId) -> Option<&Order> {
        self.orders.get(Self::slot(id)?)
    }

    /// The order with this id, for an update of its fill fields.
    pub fn get_mut(&mut self, id: OrderId) -> Option<&mut Order> {
        let slot = Self::slot(id)?;
        self.orders.get_mut(slot)
    }

    /// Move an order by the state table; an order that stops working leaves the queue.
    ///
    /// Errors: `InvalidRequest` for an unknown id, `IllegalOrderTransition` when the
    /// table forbids the move.
    pub fn apply(&mut self, id: OrderId, action: OrderAction) -> Result<OrderStatus, ExecError> {
        let order = self.get_mut(id).ok_or_else(|| ExecError::InvalidRequest {
            reason: format!("no order {id}"),
        })?;
        let status = next_status(order.status, action)?;
        order.status = status;
        if !status.is_working() {
            self.queued.retain(|queued| *queued != id);
        }
        Ok(status)
    }

    /// Queue a working order for the next open of its instrument (a converted stop-limit
    /// exit). Errors: `InvalidRequest` for an order that is unknown or no longer works.
    pub fn queue(&mut self, id: OrderId) -> Result<(), ExecError> {
        if !self.get(id).is_some_and(|order| order.status.is_working()) {
            return Err(ExecError::InvalidRequest {
                reason: format!("order {id} does not work and cannot be queued"),
            });
        }
        if !self.queued.contains(&id) {
            self.queued.push(id);
            self.queued.sort_unstable();
        }
        Ok(())
    }

    /// Replace a working order's request in full (a modification): the order keeps its
    /// id, takes the new decision instant, validity and quantity, and joins or leaves
    /// the open queue by its new type.
    ///
    /// Errors: `InvalidRequest` for an unknown id, `IllegalOrderTransition` for an order
    /// the table does not let be modified.
    pub fn replace(
        &mut self,
        id: OrderId,
        request: OrderRequest,
        decided_at: TsNanos,
        good_until: TsNanos,
        quantity_units: f64,
    ) -> Result<OrderStatus, ExecError> {
        let status = self.apply(id, OrderAction::Modify)?;
        let market = request.order_type == OrderType::Market;
        if let Some(order) = self.get_mut(id) {
            order.request = request;
            order.decided_at = decided_at;
            order.good_until = good_until;
            order.quantity_units = quantity_units;
        }
        self.queued.retain(|queued| *queued != id);
        if market {
            self.queued.push(id);
            self.queued.sort_unstable();
        }
        Ok(status)
    }

    /// The resting DAY orders of an instrument (LIMIT, SL and SL-M not waiting for an
    /// open), in id order. A GTT order rests in the GTT book instead.
    pub fn resting_for(&self, instrument: InstrumentId) -> Vec<OrderId> {
        self.orders
            .iter()
            .filter(|order| {
                order.status.is_working()
                    && order.request.instrument == instrument
                    && order.request.order_type != OrderType::Market
                    && order.request.validity == Validity::Day
                    && !self.queued.contains(&order.id)
            })
            .map(|order| order.id)
            .collect()
    }

    /// The orders of an instrument that fill at its next open, in id order.
    pub fn queued_for(&self, instrument: InstrumentId) -> Vec<OrderId> {
        self.queued
            .iter()
            .copied()
            .filter(|id| {
                self.get(*id)
                    .is_some_and(|o| o.request.instrument == instrument)
            })
            .collect()
    }

    /// Every order, in submission order.
    pub fn orders(&self) -> &[Order] {
        &self.orders
    }

    /// The orders that can still fill, in id order.
    pub fn working(&self) -> impl Iterator<Item = &Order> {
        self.orders.iter().filter(|order| order.status.is_working())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exec::order::QuantitySpec;
    use crate::model::Side;

    fn submitted(id: u64, order_type: OrderType) -> Order {
        let request = OrderRequest {
            order_type,
            ..OrderRequest::market(InstrumentId(0), Side::Buy, QuantitySpec::Units(1.0))
        };
        Order {
            id: OrderId(id),
            request,
            status: OrderStatus::Submitted,
            decided_at: 0,
            good_until: 100,
            quantity_units: 1.0,
            filled_units: 0.0,
            average_fill_rupees: 0.0,
            reject_code: None,
        }
    }

    const LIMIT: OrderType = OrderType::Limit {
        limit_rupees: 100.0,
    };

    #[test]
    fn a_market_order_waits_for_the_open_and_a_limit_rests() {
        let mut book = OrderBook::default();
        assert!(book.insert(submitted(1, OrderType::Market)).is_ok());
        assert!(book.insert(submitted(2, LIMIT)).is_ok());
        assert_eq!(book.queued_for(InstrumentId(0)), [OrderId(1)]);
        assert_eq!(book.resting_for(InstrumentId(0)), [OrderId(2)]);
    }

    #[test]
    fn a_queued_resting_order_fills_at_the_next_open_instead_of_resting() {
        let mut book = OrderBook::default();
        assert!(book.insert(submitted(1, LIMIT)).is_ok());
        assert_eq!(book.queue(OrderId(1)), Ok(()));
        assert_eq!(book.queued_for(InstrumentId(0)), [OrderId(1)]);
        assert!(book.resting_for(InstrumentId(0)).is_empty());
    }

    #[test]
    fn a_replacement_moves_the_order_between_the_queue_and_the_resting_orders() {
        let mut book = OrderBook::default();
        assert!(book.insert(submitted(1, OrderType::Market)).is_ok());
        let replacement = submitted(1, LIMIT).request;
        let status = book.replace(OrderId(1), replacement, 50, 200, 2.0);
        assert_eq!(status, Ok(OrderStatus::Submitted));
        assert!(book.queued_for(InstrumentId(0)).is_empty());
        assert_eq!(book.resting_for(InstrumentId(0)), [OrderId(1)]);
        let order = book
            .get(OrderId(1))
            .map(|o| (o.decided_at, o.good_until, o.quantity_units));
        assert_eq!(order, Some((50, 200, 2.0)));
    }

    #[test]
    fn a_filled_order_can_be_neither_queued_nor_replaced() {
        let mut book = OrderBook::default();
        assert!(book.insert(submitted(1, LIMIT)).is_ok());
        assert_eq!(
            book.apply(OrderId(1), OrderAction::Fill),
            Ok(OrderStatus::Filled)
        );
        assert!(book.queue(OrderId(1)).is_err());
        let replacement = submitted(1, LIMIT).request;
        assert!(book.replace(OrderId(1), replacement, 0, 0, 1.0).is_err());
    }
}
