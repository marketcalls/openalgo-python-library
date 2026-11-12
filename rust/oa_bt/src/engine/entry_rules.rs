//! The strategy rules that act on single orders and positions rather than on the
//! closes: the entry gates at submission, the time exit at the open, and the fill hook
//! that keeps their bookkeeping (`strategy_risk`) current.
//!
//! Entry gates (03-architecture 4.4, after the closed-market check), for an order that
//! would open or add:
//!
//! - Trading state (item 6): REDUCING after a daily loss breach rejects with
//!   `daily_loss_gate` until the next session (SPEC-18; the gate is always on and there
//!   is no check against the limit before a breach, lead rulings); HALTED rejects with
//!   `halted`.
//! - Cooldown and maximum positions (item 7), for an entry from flat only: `cooldown`
//!   while fewer than `cooldown` base bars have closed after the bar of the
//!   instrument's last exit, `max_positions` when the positions open or with an entry
//!   working have reached the cap.
//!
//! Time exit (`max_hold`, 05-risk-management 3): a position held `max_hold` base bars
//! exits at the open of the next bar of its instrument, reason `time_exit`, in step 2
//! after the strategy's orders (a timer, like the square-off). It writes the position
//! event only, as the square-off does: the vocabulary has no risk event for it. A roll
//! does not restart the count.

use crate::exec::{Fill, InstrumentId, RejectCode};
use crate::portfolio::{FillEffect, Position, PositionKey};
use crate::rms::{cooldown_gate, max_positions_gate, ExitReason, TradingState};

use super::clock::SliceInfo;
use super::error::KernelError;
use super::event::{EventStamp, Step};
use super::fills::FillRequest;
use super::kernel::Engine;
use super::strategy_risk::FillFacts;

impl Engine {
    /// The strategy's trading state: ACTIVE, REDUCING or HALTED.
    pub fn trading_state(&self) -> TradingState {
        self.risk.lock.state()
    }

    /// The entry gates for an order that would open or add to `key`, whose position
    /// holds `held_units` (signed), decided at the close of `slice`.
    ///
    /// Errors: the reject code of the first gate that refuses it.
    pub(super) fn entry_gates(
        &self,
        key: PositionKey,
        held_units: f64,
        slice: &SliceInfo,
    ) -> Result<(), RejectCode> {
        self.risk.lock.entry_gate()?;
        if held_units != 0.0 {
            return Ok(());
        }
        let risk = &self.config.risk;
        let last_exit = self.risk.last_exit_slice(key.instrument);
        cooldown_gate(risk.cooldown_bars, last_exit, slice.index)?;
        let in_use = self.positions_in_use();
        if in_use.contains(&key) {
            return Ok(());
        }
        max_positions_gate(risk.max_positions, in_use.len())
    }

    /// Positions that are open or have a working order that would open them.
    fn positions_in_use(&self) -> Vec<PositionKey> {
        let mut keys: Vec<PositionKey> = self
            .portfolio
            .positions()
            .iter()
            .filter(|position| !position.is_flat())
            .map(|position| position.key)
            .collect();
        let entries = self
            .book
            .working()
            .filter(|order| !order.request.reduce_only);
        for order in entries {
            let key = PositionKey {
                instrument: order.request.instrument,
                product: order.request.product,
            };
            if !keys.contains(&key) {
                keys.push(key);
            }
        }
        keys
    }

    /// Step 2, after the strategy's orders: exit the instrument's positions held
    /// `max_hold` base bars or more, at this open.
    pub(super) fn time_exit_at_open(
        &mut self,
        slice: &SliceInfo,
        instrument: InstrumentId,
        open: f64,
    ) -> Result<(), KernelError> {
        let Some(max_hold) = self.config.risk.max_hold_bars else {
            return Ok(());
        };
        for (key, entered) in self.risk.entries_on(instrument) {
            let held = self.held_units(key);
            if held == 0.0 || slice.index.saturating_sub(entered) < max_hold as usize {
                continue;
            }
            let request = FillRequest::engine_exit(key, held, open, ExitReason::TimeExit);
            self.execute_fill(Self::stamp(slice, Step::OpenFills), request)?;
        }
        Ok(())
    }

    /// The fill hook: called by `execute_fill` after every fill is recorded.
    pub(super) fn risk_after_fill(
        &mut self,
        stamp: EventStamp,
        key: PositionKey,
        fill: &Fill,
        effect: &FillEffect,
    ) {
        let facts = FillFacts {
            slice: stamp.slice as usize,
            key,
            reason: fill.reason,
            effect: *effect,
            totals: self.account_totals(),
            strategy_flat: self.portfolio.positions().iter().all(Position::is_flat),
        };
        self.risk.after_fill(facts);
    }
}
