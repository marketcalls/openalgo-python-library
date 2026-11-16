//! The engine's own orders in an external session (`venue`): the exits a backtest fills
//! with no order (SPEC-21), sent as orders where a backtest queues or fills them, and the
//! entries a backtest would refuse at their fill bar, withdrawn.
//!
//! Each exit is a reduce-only MARKET DAY order with the backtest's exit reason, for the
//! units of the position that no working order on its other side already covers (so the
//! working exits never exceed the position), and nothing when none are left.
//!
//! | Exit | Sent | A backtest fills it |
//! |---|---|---|
//! | Daily loss limit, drawdown halt | in step 6, when the rule fires | at the next open (step 2) |
//! | Square-off | when the session reaches the first open at or after `square_off` | at that open |
//! | Time exit (`max_hold`) | when the session reaches the open at which it is due | at that open |
//!
//! "When the session reaches an open" is the close of the slice before it (step 8 of that
//! slice, after its decisions): a live session runs a slice once it has closed, so this
//! is the moment the open arrives, and the venue then fills the exit in the slice a
//! backtest would. It does not wait for a bar: a square-off is sent even when no bar came
//! in the slice before it, nor will in the slice it is due. The order is stamped in step
//! 2 of the slice it is due in, the stamp of the backtest's fill, and re-sent at each
//! later open while the position is still held and uncovered (a rejected square-off).
//!
//! Also at each open: an MIS order that would open or add, still working once the open is
//! at or after `entry_until`, is withdrawn (`Cancel`), since a backtest rejects it at its
//! fill bar (the entry window, SPEC-04).
//!
//! In the slice that closes the session the market is closed: nothing is sent, and a
//! position still open is reported open.

use crate::exec::state::next_status;
use crate::exec::{
    InstrumentId, Order, OrderAction, OrderId, OrderRequest, OrderStatus, QuantitySpec,
};
use crate::model::{Product, Side, TsNanos};
use crate::portfolio::PositionKey;
use crate::rms::ExitReason;

use super::clock::SliceInfo;
use super::error::KernelError;
use super::event::{Event, OrderEventKind, Step};
use super::kernel::Engine;

impl Engine {
    /// Send the engine's exit of the position `key` for `reason`, decided at
    /// `decided_at` and stamped in `step` of `slice` (module docs). Nothing is sent when
    /// working orders already cover the position, or when the market is closed by then.
    ///
    /// Errors: those of the order book.
    pub(super) fn send_engine_exit(
        &mut self,
        slice: &SliceInfo,
        step: Step,
        decided_at: TsNanos,
        key: PositionKey,
        reason: ExitReason,
    ) -> Result<(), KernelError> {
        let held = self.held_units(key);
        let units = held.abs() - self.units_closing(key, held, None).min(held.abs());
        if units <= 0.0 {
            return Ok(());
        }
        // A DAY order needs a session left to fill in; at the session's last close there
        // is none, and the position stays open (module docs).
        let good_until = match self.day_validity(slice, decided_at) {
            Ok(good_until) => good_until,
            Err(_market_closed) => return Ok(()),
        };
        let side = if held > 0.0 { Side::Sell } else { Side::Buy };
        let request = OrderRequest {
            product: key.product,
            reduce_only: true,
            exit_reason: Some(reason),
            ..OrderRequest::market(key.instrument, side, QuantitySpec::Units(units))
        };
        let id = self.book.next_id();
        self.book.insert(Order {
            id,
            request,
            status: next_status(OrderStatus::Initialized, OrderAction::PassChecks)?,
            decided_at,
            good_until,
            quantity_units: units,
            filled_units: 0.0,
            average_fill_rupees: 0.0,
            reject_code: None,
        })?;
        let stamp = Self::stamp(slice, step);
        self.emit(Event::Order {
            stamp,
            order: id,
            kind: OrderEventKind::Submitted,
        });
        self.send_order(stamp, id)
    }

    /// Step 8 of a slice of an external session that does not close it: what a backtest
    /// does at the next open, sent now (module docs). Instruments in registration order,
    /// the time exit before the square-off, as in step 2.
    ///
    /// Errors: `InvalidConfig` when the session has no intraday times; those of
    /// `send_engine_exit` and `withdraw_order`.
    pub(super) fn venue_at_next_open(&mut self, slice: &SliceInfo) -> Result<(), KernelError> {
        if slice.closes_session {
            return Ok(());
        }
        let Some(next) = self.timeline.slice(slice.index + 1) else {
            return Ok(());
        };
        for index in 0..self.instruments.len() {
            let instrument = InstrumentId(index as u32);
            self.send_time_exits(&next, instrument)?;
            self.send_square_off(&next, instrument)?;
        }
        self.withdraw_late_entries(&next)
    }

    /// The time exits due at the open of `next` (`entry_rules::time_exit_at_open`).
    fn send_time_exits(
        &mut self,
        next: &SliceInfo,
        instrument: InstrumentId,
    ) -> Result<(), KernelError> {
        let Some(max_hold) = self.config.risk.max_hold_bars else {
            return Ok(());
        };
        for (key, entered) in self.risk.entries_on(instrument) {
            let due = next.index.saturating_sub(entered) >= max_hold as usize;
            if due && self.held_units(key) != 0.0 {
                let reason = ExitReason::TimeExit;
                self.send_engine_exit(next, Step::OpenFills, next.open, key, reason)?;
            }
        }
        Ok(())
    }

    /// The square-off due at the open of `next` (`session::square_off_at_open`).
    fn send_square_off(
        &mut self,
        next: &SliceInfo,
        instrument: InstrumentId,
    ) -> Result<(), KernelError> {
        let key = PositionKey {
            instrument,
            product: Product::Mis,
        };
        if self.held_units(key) == 0.0 || next.open < self.session_times(next.day)?.square_off {
            return Ok(());
        }
        let reason = ExitReason::Scheduler;
        self.send_engine_exit(next, Step::OpenFills, next.open, key, reason)
    }

    /// Withdraw the MIS orders that would open or add and still work at an open at or
    /// after `entry_until` (module docs).
    fn withdraw_late_entries(&mut self, next: &SliceInfo) -> Result<(), KernelError> {
        let entries: Vec<OrderId> = self
            .book
            .working()
            .filter(|order| order.request.product == Product::Mis && self.opens_or_adds(order))
            .map(|order| order.id)
            .collect();
        if entries.is_empty() || next.open < self.session_times(next.day)?.entry.until {
            return Ok(());
        }
        let stamp = Self::stamp(next, Step::OpenFills);
        for id in entries {
            self.withdraw_order(stamp, id)?;
        }
        Ok(())
    }

    /// True when the order would open or add to its position if it filled now.
    fn opens_or_adds(&self, order: &Order) -> bool {
        let key = PositionKey {
            instrument: order.request.instrument,
            product: order.request.product,
        };
        let held = self.held_units(key);
        let adds = held == 0.0 || (held > 0.0) == (order.request.side == Side::Buy);
        !order.request.reduce_only && adds
    }
}
