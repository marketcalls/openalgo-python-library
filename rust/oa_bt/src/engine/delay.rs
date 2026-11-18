//! Decided orders held for a delay (design 17, section 2.14; `delay_entries=`,
//! `delay_exits=`): the delayed-orders check asks whether a strategy survives entering or
//! exiting a bar late.
//!
//! An order the strategy decides is held for `delay` strategy bars (`Perturbation`):
//! decided at the close of strategy bar i, or at any instant inside it (a timer, an
//! event callback), it may fill from the close of strategy bar i + delay on, by the
//! clock, as if it were decided there, so a MARKET order fills at the next base-bar open
//! after that close. A decision inside bar i therefore waits for the rest of bar i and
//! `delay` whole bars more: never less than `delay` bars.
//!
//! Its DAY validity is the one an order decided at that instant would have
//! (`held_day_validity`): an intraday order must fill in its decision's session, a
//! daily one in the first session that opens at or after the end of its delay. An order
//! whose delay ends at or after its session's close (intraday), or after the run's last
//! strategy bar (any), is rejected `delay_past_close`. The submission checks otherwise
//! run at the decision, the checks of the fill bar at the fill.
//!
//! Which delay: an order that closes or reduces the position it meets at the decision
//! (`exit`, `reduce`, a sell of a long, a reversal, whose opening part follows its
//! close) is an exit and takes `delay_exits`; any other takes `delay_entries`. The
//! engine's own exits (the legs' stops, targets and trails, the risk exits, the time
//! exit, the square-off, the exit after an unfilled GTT child) are never held: they
//! rest with the broker or are the engine's rules.
//!
//! Every decision about an order is held, not only its placement: a cancellation or a
//! modification reaches the order as late as the order would if it were placed by that
//! decision (its delay judged the same way), and never before the order itself may fill
//! (`held_changes`). Until then the order works on its old terms, so a strategy that
//! cancels and places its entry again at every bar keeps an order working, one decided
//! `delay` bars earlier. A condition strategy counts a held entry as an entry pending
//! (SPEC-29). An event strategy sees a held order as a working order, not as a position:
//! one that decides its entries by its position alone enters again while its entry is
//! held, and holds more than it would on time (the delayed-orders check finds such a run
//! from its fills and does not judge it).

use crate::exec::{Order, OrderId, OrderRequest, RejectCode};
use crate::model::{ist_day, Side, TsNanos};

use super::clock::SliceInfo;
use super::kernel::Engine;

/// Whether `request` closes or reduces a position of `held_units` (module docs).
fn is_exit(request: &OrderRequest, held_units: f64) -> bool {
    let opposite = held_units != 0.0 && (held_units > 0.0) != (request.side == Side::Buy);
    request.reduce_only || opposite
}

impl Engine {
    /// The instant from which `request`, decided at `decided_at` against a position of
    /// `held_units`, may fill: `decided_at` when it is not held, else the close of the
    /// strategy bar its delay ends at, or `TsNanos::MAX` when that is after the run's
    /// strategy bars (module docs).
    pub(super) fn held_until(
        &self,
        request: &OrderRequest,
        held_units: f64,
        decided_at: TsNanos,
    ) -> TsNanos {
        let change = &self.config.perturbation;
        let delay = if is_exit(request, held_units) {
            change.delay_exits
        } else {
            change.delay_entries
        };
        if delay == 0 {
            return decided_at;
        }
        // A decision inside a bar first waits for that bar's close.
        let inside = !self.timeline.closes_strategy_bar(decided_at);
        let closes = delay.saturating_add(u32::from(inside));
        self.timeline
            .strategy_close_after(decided_at, closes)
            .unwrap_or(TsNanos::MAX)
    }

    /// The DAY validity of `request`, decided at `decided_at` in `slice` against a
    /// position of `held_units` (module docs): `day_validity` when it is not held, else
    /// the close of the session it may fill in from the end of its delay.
    ///
    /// Errors: `MarketClosed` as `day_validity` for an order not held; `DelayPastClose`
    /// for a held order with no session to fill in.
    pub(super) fn held_day_validity(
        &self,
        slice: &SliceInfo,
        request: &OrderRequest,
        held_units: f64,
        decided_at: TsNanos,
    ) -> Result<TsNanos, RejectCode> {
        let until = self.held_until(request, held_units, decided_at);
        if until == decided_at {
            return self.day_validity(slice, decided_at);
        }
        let past = RejectCode::DelayPastClose;
        if until == TsNanos::MAX {
            return Err(past);
        }
        if self.config.strategy_interval.is_intraday() {
            return self.day_validity(slice, until).map_err(|_| past);
        }
        self.session_close_from(ist_day(until), until).ok_or(past)
    }

    /// Record that order `id`, decided at `decided_at`, may fill from `until` (none kept
    /// when that is its decision instant).
    pub(super) fn hold_order(&mut self, id: OrderId, until: TsNanos, decided_at: TsNanos) {
        if until > decided_at {
            self.held.insert(id, until);
        } else {
            self.held.remove(&id);
        }
    }

    /// The instant from which `order` may fill: the end of its delay, else its decision
    /// instant (SPEC-35).
    pub(super) fn fills_from(&self, order: &Order) -> TsNanos {
        self.held
            .get(&order.id)
            .copied()
            .unwrap_or(order.decided_at)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exec::{InstrumentId, QuantitySpec};

    fn order(side: Side, reduce_only: bool) -> OrderRequest {
        OrderRequest {
            reduce_only,
            ..OrderRequest::market(InstrumentId(0), side, QuantitySpec::Units(1.0))
        }
    }

    #[test]
    fn an_order_that_closes_or_reduces_the_position_is_an_exit() {
        // (side, reduce_only, held units, exit)
        let rows = [
            (Side::Buy, false, 0.0, false),
            (Side::Sell, false, 0.0, false),
            (Side::Buy, false, 10.0, false),
            (Side::Sell, false, 10.0, true),
            (Side::Buy, false, -10.0, true),
            (Side::Sell, true, 10.0, true),
            (Side::Buy, true, 0.0, true),
        ];
        for (side, reduce_only, held, exit) in rows {
            assert_eq!(
                is_exit(&order(side, reduce_only), held),
                exit,
                "{side:?} {reduce_only} {held}"
            );
        }
    }
}
