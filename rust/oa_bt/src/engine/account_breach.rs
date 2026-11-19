//! An account breach in one engine (18-multi-strategy 2.5): the account's daily loss
//! limit of a portfolio run, found at a round of the fund, as each strategy of the run
//! takes it (`Engine::account_breach`).
//!
//! - **The lock.** From the breach on, an entry or add decided in a slice of the round's
//!   IST session day is rejected `account_loss_gate` (`entry_rules`); the next session
//!   boundary lifts it, as the boundary ends REDUCING (`aggregate`). The lock is named
//!   `account_loss`, as a live session's entry lock for the account's limit. It is not a
//!   trading state: the strategy's own rules keep theirs.
//! - **The record.** The risk event `account_loss_hit` (scope account, minus the limit as
//!   its level, the account's measure as observed), then every working order that would
//!   open or add is cancelled (as a live kill switch's first step), then every open
//!   position with no exit waiting is exited by step 6's rule, as the strategy's own daily
//!   loss limit exits it: at its instrument's next base-bar open, or at once at its last
//!   close, flagged, when the slice closes the session; one whose instrument has no later
//!   bar in the session fills at the session close (`aggregate`). The trades record the
//!   exit reason `account_loss_limit`. A position whose exit already waits keeps it, with
//!   its own reason.
//! - **Where it is recorded** (2.5, the breach's stamps), by where the engine stands when
//!   the fund finds the breach at the round closing at `at`:
//!
//! | The engine | Recorded at |
//! |---|---|
//! | Its slice closes at `at`: a participant of the round, its levels phase run | That slice, step 6, at once |
//! | Its slice is open and closes after `at` (a coarser base) | That slice's step 6, when its levels phase runs (`steps::levels_phase`) |
//! | Between slices | Its next slice, step 1, so its exits fill at that slice's open (`steps::open_phase`) |
//! | Its run has ended | Nothing: its positions stay as its run left them |
//!
//! A backtest's only: a live session's risk watch fires the kill switch for the account's
//! limit (16-session-design 6.3).

use crate::model::{Day, TsNanos};
use crate::rms::{ExitReason, RiskEvent, RiskEventName, RiskScope};

use super::clock::SliceInfo;
use super::error::KernelError;
use super::event::{Event, Step};
use super::kernel::{Engine, Phase};

/// The account's daily loss limit breached at a round of a portfolio run's fund, as the
/// fund hands it to each engine (module docs).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AccountBreach {
    /// The close instant of the round the fund found the breach at, UTC nanoseconds.
    pub at: TsNanos,
    /// The round's IST session day: entries are locked for it.
    pub day: Day,
    /// The account's daily loss limit, rupees, positive (`account_max_daily_loss`).
    pub limit_rupees: f64,
    /// The account's daily loss measure at the breach, rupees, negative for a loss.
    pub measure_rupees: f64,
}

impl Engine {
    /// Take the account's breach (module docs): lock entries for its session day and
    /// record it where the engine stands, now or at its next step 6 or step 1. An engine
    /// whose run has ended takes nothing.
    ///
    /// Errors: `InvalidConfig` (field `account_max_daily_loss`) for a limit that is not a
    /// positive amount of rupees or a measure that is not a finite one; `NotBuilt` in a live
    /// session, whose risk watch holds the account's limit; those of the record.
    pub fn account_breach(&mut self, breach: AccountBreach) -> Result<(), KernelError> {
        let usable = breach.limit_rupees.is_finite()
            && breach.limit_rupees > 0.0
            && breach.measure_rupees.is_finite();
        if !usable {
            return Err(KernelError::InvalidConfig {
                field: "account_max_daily_loss",
                reason: format!(
                    "a breach needs a positive limit and a finite measure, got a limit of {} \
                     and a measure of {} rupees",
                    breach.limit_rupees, breach.measure_rupees
                ),
            });
        }
        if self.is_external() {
            return Err(KernelError::NotBuilt {
                feature: "the account's daily loss breach in a live session's kernel",
                instead: "a live session's risk watch fires the kill switch for the \
                          account's daily loss limit",
            });
        }
        match self.phase {
            Phase::Finished => Ok(()),
            Phase::Opened(slice) if slice.close <= breach.at => {
                self.account.locked_day = Some(breach.day);
                self.record_account_breach(&slice, Step::AggregateRules, breach)
            }
            _ => {
                self.account.locked_day = Some(breach.day);
                self.account.waiting = Some(breach);
                Ok(())
            }
        }
    }

    /// Step 1 of `slice`: the breach found while the engine stood between slices (module
    /// docs); its exits fill at this slice's open.
    ///
    /// Errors: those of the record.
    pub(super) fn account_breach_at_open(&mut self, slice: &SliceInfo) -> Result<(), KernelError> {
        match self.account.waiting.take() {
            Some(breach) => self.record_account_breach(slice, Step::SessionOpen, breach),
            None => Ok(()),
        }
    }

    /// Step 6 of `slice`, after the strategy's own aggregate rules: the breach found while
    /// the slice was open (module docs).
    ///
    /// Errors: those of the record.
    pub(super) fn account_breach_at_close(&mut self, slice: &SliceInfo) -> Result<(), KernelError> {
        match self.account.waiting.take() {
            Some(breach) => self.record_account_breach(slice, Step::AggregateRules, breach),
            None => Ok(()),
        }
    }

    /// The record of module docs, stamped at `step` of `slice`: the event, the cancels of
    /// the working entries and adds, the exits.
    fn record_account_breach(
        &mut self,
        slice: &SliceInfo,
        step: Step,
        breach: AccountBreach,
    ) -> Result<(), KernelError> {
        let stamp = Self::stamp(slice, step);
        let event = RiskEvent {
            name: RiskEventName::AccountLossHit,
            scope: RiskScope::Account,
            level_rupees: Some(-breach.limit_rupees),
            target_index: None,
            observed_rupees: Some(breach.measure_rupees),
        };
        self.emit(Event::Risk {
            stamp,
            instrument: None,
            event,
        });
        self.cancel_working_entries(stamp)?;
        let reason = ExitReason::AccountLossLimit;
        for key in self.positions_to_exit() {
            // Step 1 queues the exits for this slice's own open; step 6 follows its rule.
            match step {
                Step::AggregateRules => self.exit_at_step_six(slice, stamp, key, reason)?,
                _ => self.risk.queue_exit(key, reason),
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
