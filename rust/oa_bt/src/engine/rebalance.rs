//! The kernel's part of `rebalance()` (11-portfolio-rebalancing 2.1, the subset built):
//! each symbol's trade to its target weight is a `target()` order marked as part of a
//! rebalance (`OrderRequest::rebalance`, tag `rebalance`), and the rebalance's sales
//! fill before its buys.
//!
//! In step 2 every instrument's queued orders fill in registration order, which would
//! let a buy of an instrument registered early fill before the sale of one registered
//! later and find no cash. So a rebalance order that opens or adds is skipped there
//! (`fills::fill_queued`) and filled by `fill_rebalance_buys` once every instrument's
//! queued orders have filled, in id order: by then the rebalance's sales (reduce-only
//! orders, filled in the ordinary pass) have credited their proceeds as the sale-credit
//! rule allows (`sale_credit`). A buy the free cash does not cover at its fill is
//! rejected `margin`, like any order. In a slice with a timer pause the rebalance buys
//! due at its open fill before the timers fire (`timers::fill_due_orders`): a timer that
//! rebalances again sizes its orders from the position they made, not from orders still
//! queued, whose reduce-only sales would find nothing to reduce. The buys of a timer's
//! own rebalance fill after the timers' orders.
//!
//! A rebalance buy is also an intended add: the averaging-down refusal of the scale-in
//! limits does not apply to it (`scale_in`).
//!
//! Not built here (11-portfolio-rebalancing 2.1): scaling every buy by the cash left at
//! the fill (rule 4), the no-trade rules, the rebalance log and its reason codes.

use crate::exec::{Order, OrderRequest, OrderType, Validity};

use super::clock::SliceInfo;
use super::error::KernelError;
use super::kernel::Engine;
use super::order_fill::OrderFill;
use super::submit::Submission;
use super::target::TargetRequest;

/// The tag of every order of a rebalance, shown in the orders table.
const REBALANCE_TAG: &str = "rebalance";

/// True for a rebalance order that opens or adds: it fills after the ordinary queue.
pub(super) fn waits_for_sales(order: &Order) -> bool {
    order.request.rebalance && !order.request.reduce_only
}

impl Engine {
    /// Trade to `target` as part of a rebalance: the order of `submit_target`, marked as
    /// a rebalance order and tagged `rebalance`. Returns `None` when the position
    /// already is the target, else the outcome of its submission.
    ///
    /// Errors: those of `submit_target`.
    pub fn submit_rebalance(
        &mut self,
        target: TargetRequest,
    ) -> Result<Option<Submission>, KernelError> {
        let Some(request) = self.target_order(target)? else {
            return Ok(None);
        };
        let request = OrderRequest {
            rebalance: true,
            tag: Some(REBALANCE_TAG.to_string()),
            ..request
        };
        self.submit(request).map(Some)
    }

    /// Step 2, after every instrument's queued orders: fill the rebalance orders due at
    /// this open that open or add (`waits_for_sales`; a rebalance never targets a short,
    /// so these are its buys), in id order (see the module documentation). One whose
    /// instrument has no bar in the slice stays queued (SPEC-06).
    ///
    /// Errors: those of `fill_order`.
    pub(super) fn fill_rebalance_buys(&mut self, slice: &SliceInfo) -> Result<(), KernelError> {
        let due: Vec<Order> = self
            .book
            .working()
            .filter(|order| {
                waits_for_sales(order)
                    && order.request.order_type == OrderType::Market
                    && order.request.validity == Validity::Day
                    && slice.open >= order.decided_at
            })
            .cloned()
            .collect();
        for order in due {
            let Some(bar) = self.market.bar(order.request.instrument, slice.index) else {
                continue;
            };
            self.fill_order(slice, &order, OrderFill::at_open(bar.open))?;
        }
        Ok(())
    }
}
