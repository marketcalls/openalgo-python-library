//! The order book of a run: every order by id, and the queue of orders waiting for an
//! open.
//!
//! Orders are stored in submission order and never removed, so a result can list
//! them all. The queue holds the working MARKET orders in id order; step 2 of every
//! slice walks it per instrument, which keeps fills deterministic.

use super::error::ExecError;
use super::instrument::InstrumentId;
use super::order::{Order, OrderId, OrderType};
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

    /// The working MARKET orders of an instrument, in id order.
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
