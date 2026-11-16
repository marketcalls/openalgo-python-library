//! The engine's own exits in an external session (`venue`, `venue_owner`): the exits a
//! backtest fills with no order (SPEC-21), made where a backtest queues or fills them,
//! sent again while they still apply, and the entries a backtest would refuse at their
//! fill bar, withdrawn.
//!
//! | Exit | Made | A backtest fills it |
//! |---|---|---|
//! | Daily loss limit, drawdown halt | in step 6, when the rule fires | at the next open (step 2) |
//! | Square-off | when the session reaches the first open at or after `square_off` | at that open |
//! | Time exit (`max_hold`) | when the session reaches the open at which it is due | at that open |
//!
//! Each is a reduce-only MARKET DAY order with the backtest's exit reason for the whole
//! position, made as the position's exit (`venue_owner`): it takes over from the
//! strategy's exits of the position and cancels the working orders on its closing side
//! first, theirs among them. "When the session reaches an open" is the close of the slice
//! before it, after its decisions and timers (`venue_clock`): a live session runs a slice
//! once it has closed, so this is the moment the open arrives, and the venue then fills
//! the exit in the slice a backtest would. It does not wait for a bar. The order is
//! stamped in step 2 of the slice it is due in, the stamp of the backtest's fill.
//!
//! Sent again: at every open the session reaches, a position that is still open and that
//! no exit of the engine holds gets the exit that applies, in this order: the risk lock
//! (REDUCING after a daily loss breach, HALTED after the drawdown halt), the time exit,
//! the square-off. So an exit the venue rejected, or a position a cancel race reopened, is
//! exited again at the next open; how often to retry and when to stop are the live
//! layer's to decide.
//!
//! Also at each open: an MIS order that would open or add, still working once the open is
//! at or after `entry_until`, is withdrawn (`Cancel`), since a backtest rejects it at its
//! fill bar (the entry window, SPEC-04).
//!
//! In the slice that closes the session the market is closed: nothing is sent, and a
//! position still open is reported open.

use crate::exec::{InstrumentId, Order, OrderId};
use crate::model::{Product, Side};
use crate::portfolio::PositionKey;
use crate::rms::{ExitReason, TradingState, TradingStateCause};

use super::clock::SliceInfo;
use super::error::KernelError;
use super::event::Step;
use super::kernel::Engine;
use super::venue::SendAt;

impl Engine {
    /// What a backtest does at the open of `next`, done as the external session reaches
    /// it (module docs): the reduce-only orders are fitted to their positions
    /// (`venue_cover`), the waiting exits move on, every open position that no exit of the
    /// engine holds gets the exit that applies, instruments in registration order, and the
    /// late MIS entries are withdrawn.
    ///
    /// Errors: `InvalidConfig` when the session has no intraday times; those of the exits
    /// and of `withdraw_order`.
    pub(super) fn venue_at_next_open(&mut self, next: &SliceInfo) -> Result<(), KernelError> {
        let at = SendAt {
            slice: *next,
            step: Step::OpenFills,
            decided_at: next.open,
        };
        self.fit_exit_cover(at)?;
        self.progress_exits(at)?;
        for index in 0..self.instruments.len() {
            let instrument = InstrumentId(index as u32);
            let open: Vec<PositionKey> = self
                .portfolio
                .positions()
                .iter()
                .filter(|position| position.key.instrument == instrument && !position.is_flat())
                .map(|position| position.key)
                .collect();
            for key in open {
                if self.has_engine_exit(key) {
                    continue;
                }
                if let Some(reason) = self.exit_due(next, key)? {
                    self.request_engine_exit(at, key, reason)?;
                }
            }
        }
        self.withdraw_late_entries(next)
    }

    /// The engine's exit that applies to the open position `key` at the open of `next`.
    fn exit_due(
        &mut self,
        next: &SliceInfo,
        key: PositionKey,
    ) -> Result<Option<ExitReason>, KernelError> {
        if let Some(reason) = self.risk_lock_exit() {
            return Ok(Some(reason));
        }
        if let Some(max_hold) = self.config.risk.max_hold_bars {
            let entered = self
                .risk
                .entries_on(key.instrument)
                .into_iter()
                .find(|(held, _)| *held == key)
                .map(|(_, slice)| slice);
            let due = |slice: usize| next.index.saturating_sub(slice) >= max_hold as usize;
            if entered.is_some_and(due) {
                return Ok(Some(ExitReason::TimeExit));
            }
        }
        if key.product == Product::Mis && next.open >= self.session_times(next.day)?.square_off {
            return Ok(Some(ExitReason::Scheduler));
        }
        Ok(None)
    }

    /// The exit of a risk lock in force: the daily loss limit while REDUCING after its
    /// breach, the drawdown halt while HALTED by it.
    fn risk_lock_exit(&self) -> Option<ExitReason> {
        match (self.risk.lock.state(), self.risk.lock.cause()) {
            (TradingState::Reducing, Some(TradingStateCause::DailyLossBreach)) => {
                Some(ExitReason::DailyLossLimit)
            }
            (TradingState::Halted, Some(TradingStateCause::DrawdownHalt)) => {
                Some(ExitReason::DrawdownHalt)
            }
            _ => None,
        }
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
