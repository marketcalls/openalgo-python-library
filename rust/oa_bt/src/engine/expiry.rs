//! A derivatives position exited before the exchange settles its contract by delivery
//! (design 20-positional-live section 9; decision 11.2, extended by the lead to options
//! and MCX delivery contracts), in a backtest and in a live session alike.
//!
//! Each derivative instrument carries its positional exit day
//! (`InstrumentSpec::positional_exit_day`), worked out by the Python package from its
//! reference tables: the last trading day of a stock future, of a stock option and of
//! every option; the trading day before an MCX delivery contract's tender period begins;
//! none for a contract the exchange settles in cash.
//!
//! The positional exit is the square-off time of that day. From then on, and on every
//! later day, the contract is not entered (`past_positional_exit`):
//!
//! | Contract | Backtest | Live session |
//! |---|---|---|
//! | With a positional exit day | on that day every position still open on it exits at the first base-bar open at or after the exchange's square-off time, as MIS positions are squared off, reason `expiry_exit`, at the open plus slippage (`expiry_exit_at_open`, step 2, after the square-off); its stop GTT is cancelled first (`gtt`); one still open at that session's close exits at its last close, flagged `filled_at_last_close` (`close_expiring_positions`, step 8, as `eod` closes an MIS position), so its settlement never happens. A position still open on a later day (an exit day the run has no session for) exits at that day's first open, or at its close when it has no bar | the engine's exit at that open, `EngineExit::Risk(ExpiryExit)` (`venue_exits`), its stop GTT or stop order cancelled first by the exit owner (`venue_owner`); a position carried past its exit day (a session not started that day, an exit not filled), up to the contract's last trading day, is exited as the next session reaches its first open (`venue_exits::expiry_exits_at_session_open`); one carried past its last trading day is refused at the restore (`venue_state::carry_expired`) |
//! | Settled in cash (index futures, MCX cash contracts, an MCX delivery contract the tables give no tender period for) | settled at expiry as before (`settlement`) | at the close of its last trading day its stop GTT is cancelled (`cancel_expiring_stop_gtts`), and none is placed again; the position is left to the exchange. The next day's session records its exit at the carried last close, reason `expiry`, flagged `filled_at_last_close` (`venue_state::carry_restore`) |
//!
//! An order that would open or add on a contract past its positional exit is refused,
//! `contract_expiring`, at submission (`submit`) and at its fill (`fill_bar`); in a live
//! session one still working when the session reaches an open past the exit is withdrawn
//! (`venue_exits`). Exits are never refused by it.

use crate::exec::{FillAt, FillFlags, InstrumentId};
use crate::model::{Day, TsNanos};
use crate::portfolio::PositionKey;
use crate::rms::ExitReason;

use super::clock::SliceInfo;
use super::error::KernelError;
use super::event::Step;
use super::fills::FillRequest;
use super::kernel::{Engine, Phase};

impl Engine {
    /// The positional exit day of `instrument`, if it has one.
    fn positional_exit_day(&self, instrument: InstrumentId) -> Option<Day> {
        self.instruments
            .get(instrument.index())
            .and_then(|spec| spec.positional_exit_day)
    }

    /// True when `at` (UTC nanoseconds) on the session `day` is at or after `instrument`'s
    /// positional exit: the square-off time of its positional exit day, or any later day
    /// (module docs). False for a contract with no positional exit day.
    ///
    /// Errors: `InvalidConfig` when the exit day's session has no intraday times.
    pub(super) fn past_positional_exit(
        &mut self,
        instrument: InstrumentId,
        day: Day,
        at: TsNanos,
    ) -> Result<bool, KernelError> {
        match self.positional_exit_day(instrument) {
            Some(exit_day) if day == exit_day => Ok(at >= self.session_times(day)?.square_off),
            Some(exit_day) => Ok(day > exit_day),
            None => Ok(false),
        }
    }

    /// True when `slice` opens past `instrument`'s positional exit while the contract
    /// still trades: at or after the square-off time of its exit day, or on any later day
    /// up to its last trading day, so a position the exit missed (an exit day with no
    /// session, a live session not started that day) is exited at the next open (module
    /// docs).
    ///
    /// Errors: those of `past_positional_exit`.
    pub(super) fn expiry_exit_due(
        &mut self,
        slice: &SliceInfo,
        instrument: InstrumentId,
    ) -> Result<bool, KernelError> {
        let last_trading_day = self
            .instruments
            .get(instrument.index())
            .and_then(|spec| spec.last_trading_day);
        if last_trading_day.is_some_and(|last| slice.day > last) {
            return Ok(false);
        }
        self.past_positional_exit(instrument, slice.day, slice.open)
    }

    /// The open positions of `instrument`, every product, in key order.
    fn open_positions_of(&self, instrument: InstrumentId) -> Vec<PositionKey> {
        self.portfolio
            .positions()
            .iter()
            .filter(|position| position.key.instrument == instrument && !position.is_flat())
            .map(|position| position.key)
            .collect()
    }

    /// Step 2 of a backtest, after the square-off: on `instrument`'s positional exit day,
    /// at the first base-bar open at or after the square-off time, every position still
    /// open on it exits at `open_rupees` plus slippage, reason `expiry_exit` (module docs).
    ///
    /// Errors: those of `expiry_exit_due` and of `execute_fill`.
    pub(super) fn expiry_exit_at_open(
        &mut self,
        slice: &SliceInfo,
        instrument: InstrumentId,
        open_rupees: f64,
    ) -> Result<(), KernelError> {
        if !self.expiry_exit_due(slice, instrument)? {
            return Ok(());
        }
        let stamp = Self::stamp(slice, Step::OpenFills);
        for key in self.open_positions_of(instrument) {
            let held = self.held_units(key);
            let exit = FillRequest::engine_exit(key, held, open_rupees, ExitReason::ExpiryExit);
            self.execute_fill(stamp, exit)?;
        }
        Ok(())
    }

    /// Step 8 of a backtest's session close: a position still open on an instrument whose
    /// positional exit day this session is, or was before it (no base bar at or after the
    /// square-off time, a daily base, or no bar of it since), exits at its last close,
    /// flagged `filled_at_last_close`, reason `expiry_exit`, before the futures settlement
    /// (module docs).
    ///
    /// Errors: those of `execute_fill`.
    pub(super) fn close_expiring_positions(
        &mut self,
        slice: &SliceInfo,
    ) -> Result<(), KernelError> {
        let stamp = Self::stamp(slice, Step::SessionClose);
        for index in 0..self.instruments.len() {
            let instrument = InstrumentId(index as u32);
            if !self
                .positional_exit_day(instrument)
                .is_some_and(|exit_day| slice.day >= exit_day)
            {
                continue;
            }
            let Some(last) = self.last_close(instrument) else {
                continue;
            };
            for key in self.open_positions_of(instrument) {
                let held = self.held_units(key);
                let exit = FillRequest::engine_exit(key, held, last, ExitReason::ExpiryExit)
                    .at(FillAt::Close)
                    .flagged(FillFlags::FILLED_AT_LAST_CLOSE);
                self.execute_fill(stamp, exit)?;
            }
        }
        Ok(())
    }

    /// True once a live session has reached the close of the last trading day of
    /// `instrument`'s contract, or stands on a later day: no stop of it is placed or kept
    /// at the venue any more (module docs).
    pub(super) fn contract_has_ended(&self, instrument: InstrumentId) -> bool {
        let Some(last) = self
            .instruments
            .get(instrument.index())
            .and_then(|spec| spec.last_trading_day)
        else {
            return false;
        };
        let day = match self.phase {
            Phase::Closing(slice) => return slice.closes_session && slice.day >= last,
            Phase::Finished => return true,
            Phase::Ready(index) => self.timeline.slice(index).map(|slice| slice.day),
            Phase::Opened(slice)
            | Phase::Paused(slice)
            | Phase::Timers(slice)
            | Phase::Ahead(slice)
            | Phase::PreOpen(slice, _) => Some(slice.day),
        };
        day.is_some_and(|day| day > last)
    }

    /// True when `instrument`'s contract had its last trading day before `day`: a live
    /// session holds it only to record its exit at expiry (`venue_state::carry_expired`).
    pub(super) fn contract_expired_before(&self, instrument: InstrumentId, day: Day) -> bool {
        self.instruments
            .get(instrument.index())
            .and_then(|spec| spec.last_trading_day)
            .is_some_and(|last| last < day)
    }

    /// At a live session's close: the stop GTT of every position on a contract whose last
    /// trading day this session is, is cancelled, and no other is placed (module docs):
    /// the exchange settles what is left.
    ///
    /// Errors: those of `ask_cancel_gtt`.
    pub(super) fn cancel_expiring_stop_gtts(
        &mut self,
        slice: &SliceInfo,
    ) -> Result<(), KernelError> {
        let ending: Vec<PositionKey> = self
            .venue
            .gtts
            .legs
            .iter()
            .map(|leg| leg.key)
            .filter(|key| {
                let spec = self.instruments.get(key.instrument.index());
                spec.is_some_and(|spec| spec.last_trading_day == Some(slice.day))
            })
            .collect();
        let stamp = Self::stamp(slice, Step::SessionClose);
        for key in ending {
            if let Some(id) = self.stop_gtt_at_venue(key).map(|gtt| gtt.id) {
                self.ask_cancel_gtt(stamp.slice, id)?;
            }
        }
        Ok(())
    }
}
