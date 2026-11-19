//! Cancellations and modifications held for a delay (design 17, section 2.14;
//! `delay_entries`, `delay_exits`; `delay` has the rules of the delay itself).
//!
//! Under a delay every decision about an order reaches the market late, not only its
//! placement: a change the strategy decides (`Engine::cancel`, `Engine::modify`) takes
//! effect at the instant an order placed by that decision could first fill
//! (`Engine::held_until`, judged on the order's request, or the new request of a
//! modification, against the position at the decision), and never before the order
//! itself may fill. Until then the order works on its old terms and may fill on them.
//!
//! A change whose instant has come takes effect at the open of the first slice at or
//! after it, in step 2 before any fill of that slice (`apply_held_changes`): a
//! cancellation cancels the order, a modification replaces its request with the new one
//! (which may fill from that instant), each with its event stamped there. Changes due at
//! the same open take effect in the order of their instants, then of their decisions. A
//! change that reaches an order no longer working (filled, expired, cancelled) does
//! nothing, as a broker refuses it; a change held past the run's last strategy bar never
//! takes effect. A second cancellation of an order already being cancelled adds nothing.
//!
//! A change that is not held (no delay, or a live session, whose venue holds its own
//! cancellations and modifications) acts at the decision, as it always did.

use crate::exec::{Order, OrderId, OrderRequest};
use crate::model::TsNanos;
use crate::portfolio::PositionKey;

use super::clock::SliceInfo;
use super::error::KernelError;
use super::event::{Event, EventStamp, OrderEventKind, Step};
use super::kernel::Engine;
use super::venue::Replacement;

/// What a held change does to its order (a replacement boxed: it is much larger than a
/// cancellation).
#[derive(Clone, Debug, PartialEq)]
pub(super) enum Change {
    Cancel,
    Modify(Box<Replacement>),
}

/// A change of `order` decided by the strategy and held until `effective_at`.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct HeldChange {
    pub(super) order: OrderId,
    pub(super) effective_at: TsNanos,
    pub(super) change: Change,
}

impl Engine {
    /// The instant a change of `order` to `request`, decided at `decided_at`, takes
    /// effect (module docs): `decided_at` when it acts at once.
    fn change_effective_at(
        &self,
        order: &Order,
        request: &OrderRequest,
        decided_at: TsNanos,
    ) -> TsNanos {
        if self.is_external() {
            return decided_at;
        }
        let key = PositionKey {
            instrument: request.instrument,
            product: request.product,
        };
        let until = self.held_until(request, self.held_units(key), decided_at);
        until.max(self.fills_from(order))
    }

    /// Hold the strategy's cancellation of the working order `id`, decided at
    /// `decided_at`, when it does not act at once (module docs). True when it is held,
    /// and the order works on.
    ///
    /// Errors: `UnknownOrder`.
    pub(super) fn hold_cancel(
        &mut self,
        id: OrderId,
        decided_at: TsNanos,
    ) -> Result<bool, KernelError> {
        let order = self.book.get(id).ok_or(KernelError::UnknownOrder(id))?;
        let effective_at = self.change_effective_at(order, &order.request, decided_at);
        if effective_at == decided_at {
            return Ok(false);
        }
        let cancelling = self
            .held_changes
            .iter()
            .any(|held| held.order == id && held.change == Change::Cancel);
        if !cancelling {
            self.held_changes.push(HeldChange {
                order: id,
                effective_at,
                change: Change::Cancel,
            });
        }
        Ok(true)
    }

    /// Hold the strategy's modification of the working order `id` to `replacement`
    /// when it does not act at once (module docs); else replace the order now, with its
    /// event stamped `stamp`.
    ///
    /// Errors: `UnknownOrder`, and those of `replace_order`.
    pub(super) fn modify_or_hold(
        &mut self,
        stamp: EventStamp,
        id: OrderId,
        replacement: Replacement,
    ) -> Result<(), KernelError> {
        let order = self.book.get(id).ok_or(KernelError::UnknownOrder(id))?;
        let decided_at = replacement.decided_at;
        let effective_at = self.change_effective_at(order, &replacement.request, decided_at);
        if effective_at == decided_at {
            return self.replace_order(stamp, id, replacement, decided_at);
        }
        self.held_changes.push(HeldChange {
            order: id,
            effective_at,
            change: Change::Modify(Box::new(replacement)),
        });
        Ok(())
    }

    /// Replace the order `id`'s request with `replacement`, eligible to fill from
    /// `fills_from`, with its `Modified` event and its GTT replaced.
    ///
    /// Errors: those of the order book and of the GTT book.
    fn replace_order(
        &mut self,
        stamp: EventStamp,
        id: OrderId,
        replacement: Replacement,
        fills_from: TsNanos,
    ) -> Result<(), KernelError> {
        let Replacement {
            request,
            decided_at,
            good_until,
            units,
        } = replacement;
        self.book
            .replace(id, request, decided_at, good_until, units)?;
        self.hold_order(id, fills_from, decided_at);
        self.emit(Event::Order {
            stamp,
            order: id,
            kind: OrderEventKind::Modified,
        });
        self.modify_order_gtt(stamp, id)
    }

    /// Step 2, before any fill of `slice`: every held change whose instant has come
    /// takes effect (module docs).
    ///
    /// Errors: those of `withdraw_order` and `replace_order`.
    pub(super) fn apply_held_changes(&mut self, slice: &SliceInfo) -> Result<(), KernelError> {
        if self.held_changes.is_empty() {
            return Ok(());
        }
        let (mut due, waiting): (Vec<HeldChange>, Vec<HeldChange>) =
            std::mem::take(&mut self.held_changes)
                .into_iter()
                .partition(|held| held.effective_at <= slice.open);
        self.held_changes = waiting;
        // A stable sort: changes due at one instant keep the order they were decided in.
        due.sort_by_key(|held| held.effective_at);
        let stamp = Self::stamp(slice, Step::OpenFills);
        for held in due {
            let working = self
                .book
                .get(held.order)
                .is_some_and(|order| order.status.is_working());
            if !working {
                continue;
            }
            match held.change {
                Change::Cancel => {
                    self.withdraw_order(stamp, held.order)?;
                }
                Change::Modify(replacement) => {
                    self.replace_order(stamp, held.order, *replacement, held.effective_at)?
                }
            }
        }
        Ok(())
    }
}
