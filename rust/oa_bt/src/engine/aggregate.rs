//! Step 6, the aggregate rules on the synchronised closes (03-architecture 4.2), with
//! the session boundary of step 1 and the exits the rules queue for steps 2 and 8.
//!
//! Phase 1 runs single positions per strategy, so the aggregate rules are the daily
//! loss limit and the drawdown halt (the combined rules of a leg group are Phase 3).
//! Both are evaluated after step 5 has marked every position, in this order:
//!
//! 1. Daily loss (05-risk-management 5, SPEC-16, SPEC-18): while ACTIVE, a measure at
//!    or beyond `-max_daily_loss` on the run's `risk_basis` writes
//!    `daily_loss_limit_hit` (scope session, the limit as a negative level, the measure
//!    as the observed value), moves the strategy to REDUCING until the next session
//!    whatever `after_risk_exit` says, and exits every position, reason
//!    `daily_loss_limit`.
//! 2. Drawdown halt: unless HALTED, equity at or below the peak less the limit writes
//!    `risk_halt` (scope strategy, the floor as the level, the equity as observed),
//!    latches HALTED for the rest of the run and exits every position, reason
//!    `drawdown_halt`. When both fire on one close, the daily loss reason is kept for
//!    the exits (a daily loss breach overrides the other aggregate rules) and the state
//!    ends HALTED.
//!
//! Each change of trading state writes its own event after the rule's event. Working
//! orders that would open or add are cancelled when entries lock, because the gate at
//! submission cannot see an order placed before the breach that still waits for a bar
//! (SPEC-06). Exits are queued for each position's next base-bar open (step 2, before
//! the strategy's orders, which were decided later). In the slice that closes the
//! session there is no later bar, so the exit fills at once at the last close, flagged
//! (on a daily base every bar is such a slice, SPEC-17); an exit still waiting when the
//! session closes fills in step 8 the same way. The exits are engine-held: no order
//! event, the risk event already written, then the position event of the fill.
//!
//! Step 1: when a slice opens a later session, the daily loss measure starts its new
//! session and REDUCING returns to ACTIVE, with its event; HALTED stays. The session
//! boundary of the live module is 03:00 IST, which lies between any two sessions of the
//! Indian calendar, so the kernel crosses it where the slice's IST session date
//! changes.

use crate::exec::{FillAt, FillFlags, InstrumentId, OrderAction, OrderId};
use crate::model::Side;
use crate::portfolio::PositionKey;
use crate::rms::{
    is_breached, AccountTotals, ExitReason, RiskBasis, RiskEvent, RiskEventName, RiskScope,
    TradingState, TradingStateCause,
};

use super::clock::SliceInfo;
use super::error::KernelError;
use super::event::{Event, EventStamp, OrderEventKind, Step};
use super::fills::FillRequest;
use super::kernel::Engine;

impl Engine {
    /// Step 1: a later session ends REDUCING and restarts the daily loss measure.
    pub(super) fn risk_session_open(&mut self, slice: &SliceInfo) {
        if !self.risk.enter_session(slice.day) {
            return;
        }
        let stamp = Self::stamp(slice, Step::SessionOpen);
        self.change_trading_state(stamp, TradingStateCause::SessionBoundary);
    }

    /// Step 6: the daily loss limit, then the drawdown halt (see the module
    /// documentation).
    pub(super) fn step_aggregate_rules(&mut self, slice: &SliceInfo) -> Result<(), KernelError> {
        let stamp = Self::stamp(slice, Step::AggregateRules);
        let mut exit_reason = None;
        if let Some(event) = self.daily_loss_breach() {
            self.emit(Event::Risk {
                stamp,
                instrument: None,
                event,
            });
            self.change_trading_state(stamp, TradingStateCause::DailyLossBreach);
            exit_reason = Some(ExitReason::DailyLossLimit);
        }
        if let Some(event) = self.drawdown_breach() {
            self.emit(Event::Risk {
                stamp,
                instrument: None,
                event,
            });
            self.change_trading_state(stamp, TradingStateCause::DrawdownHalt);
            exit_reason.get_or_insert(ExitReason::DrawdownHalt);
        }
        match exit_reason {
            Some(reason) => {
                self.cancel_working_entries(stamp)?;
                self.exit_every_position(slice, stamp, reason)
            }
            None => Ok(()),
        }
    }

    /// Step 2: the exits waiting for this instrument fill at its open, before the
    /// strategy's orders.
    pub(super) fn risk_exits_at_open(
        &mut self,
        slice: &SliceInfo,
        instrument: InstrumentId,
        open: f64,
    ) -> Result<(), KernelError> {
        for pending in self.risk.take_exits(Some(instrument)) {
            let held = self.held_units(pending.key);
            if held != 0.0 {
                let request = FillRequest::engine_exit(pending.key, held, open, pending.reason);
                self.execute_fill(Self::stamp(slice, Step::OpenFills), request)?;
            }
        }
        Ok(())
    }

    /// Step 8: exits still waiting (their instrument had no later bar in the session)
    /// fill at the last close, flagged.
    pub(super) fn risk_exits_at_session_close(
        &mut self,
        slice: &SliceInfo,
    ) -> Result<(), KernelError> {
        let stamp = Self::stamp(slice, Step::SessionClose);
        for pending in self.risk.take_exits(None) {
            self.exit_at_last_close(stamp, pending.key, pending.reason)?;
        }
        Ok(())
    }

    /// The daily loss event when the limit is breached while ACTIVE.
    fn daily_loss_breach(&self) -> Option<RiskEvent> {
        let limit = self.config.risk.max_daily_loss_rupees?;
        if self.risk.lock.state() != TradingState::Active {
            return None;
        }
        let measure = self.risk.daily_loss.measure_rupees(
            self.config.risk.risk_basis,
            self.account_totals(),
            self.portfolio.unrealized_rupees(),
        );
        is_breached(measure, limit).then_some(RiskEvent {
            name: RiskEventName::DailyLossLimitHit,
            scope: RiskScope::Session,
            level_rupees: Some(-limit),
            target_index: None,
            observed_rupees: Some(measure),
        })
    }

    /// The drawdown halt's event when equity reaches the floor and the strategy is not
    /// HALTED yet. The peak is updated at every close.
    fn drawdown_breach(&mut self) -> Option<RiskEvent> {
        let totals = self.account_totals();
        let charges = match self.config.risk.risk_basis {
            RiskBasis::Gross => 0.0,
            RiskBasis::Net => totals.costs_rupees,
        };
        let equity = self.portfolio.account.capital_rupees + totals.realized_rupees - charges
            + self.portfolio.unrealized_rupees();
        let halted = self.risk.lock.state() == TradingState::Halted;
        let breach = self.risk.drawdown.as_mut()?.observe(equity)?;
        (!halted).then_some(RiskEvent {
            name: RiskEventName::RiskHalt,
            scope: RiskScope::Strategy,
            level_rupees: Some(breach.floor_rupees),
            target_index: None,
            observed_rupees: Some(breach.equity_rupees),
        })
    }

    /// The account's realized P&L and charges so far.
    pub(super) fn account_totals(&self) -> AccountTotals {
        AccountTotals {
            realized_rupees: self.portfolio.account.realized_rupees,
            costs_rupees: self.portfolio.account.costs_rupees,
        }
    }

    /// Move the trading state by a cause and write the event when it changed.
    fn change_trading_state(&mut self, stamp: EventStamp, cause: TradingStateCause) {
        if let Some(change) = self.risk.lock.apply(cause) {
            self.emit(Event::TradingState {
                stamp,
                from: change.from,
                to: change.to,
                cause,
            });
        }
    }

    /// Cancel every working order that would open or add, in id order.
    fn cancel_working_entries(&mut self, stamp: EventStamp) -> Result<(), KernelError> {
        let entries: Vec<OrderId> = self
            .book
            .working()
            .filter(|order| {
                let key = PositionKey {
                    instrument: order.request.instrument,
                    product: order.request.product,
                };
                let held = self.held_units(key);
                let adds = held == 0.0 || (held > 0.0) == (order.request.side == Side::Buy);
                !order.request.reduce_only && adds
            })
            .map(|order| order.id)
            .collect();
        for id in entries {
            self.end_order(stamp, id, OrderAction::Cancel, OrderEventKind::Canceled)?;
        }
        Ok(())
    }

    /// Exit every open position, in instrument registration order: at once at the last
    /// close when the slice closes the session, otherwise at the next open.
    fn exit_every_position(
        &mut self,
        slice: &SliceInfo,
        stamp: EventStamp,
        reason: ExitReason,
    ) -> Result<(), KernelError> {
        let mut open: Vec<PositionKey> = self
            .portfolio
            .positions()
            .iter()
            .filter(|position| !position.is_flat() && !self.risk.exit_is_pending(position.key))
            .map(|position| position.key)
            .collect();
        open.sort_by_key(|key| key.instrument);
        for key in open {
            if slice.closes_session {
                self.exit_at_last_close(stamp, key, reason)?;
            } else {
                self.risk.queue_exit(key, reason);
            }
        }
        Ok(())
    }

    /// Close a position at its instrument's last close, flagged.
    fn exit_at_last_close(
        &mut self,
        stamp: EventStamp,
        key: PositionKey,
        reason: ExitReason,
    ) -> Result<(), KernelError> {
        let held = self.held_units(key);
        let Some(last) = self.last_close(key.instrument) else {
            return Ok(());
        };
        if held == 0.0 {
            return Ok(());
        }
        let request = FillRequest::engine_exit(key, held, last, reason)
            .at(FillAt::Close)
            .flagged(FillFlags::FILLED_AT_LAST_CLOSE);
        self.execute_fill(stamp, request)?;
        Ok(())
    }
}
