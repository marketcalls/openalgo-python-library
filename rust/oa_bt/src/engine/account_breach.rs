//! An account breach in one engine (18-multi-strategy 2.5 and 2.5a rule 6): the account's
//! daily loss limit, or its daily profit target, of a portfolio run, found at a round of
//! the fund, as each strategy of the run takes it (`Engine::account_breach`). The two
//! rules (`AccountRule`) act alike under their own names:
//!
//! | Rule | Lock | Reject code | Risk event | Trades' exit reason |
//! |---|---|---|---|---|
//! | `Loss` (`account_max_daily_loss`) | `account_loss` | `account_loss_gate` | `account_loss_hit` | `account_loss_limit` |
//! | `Profit` (`account_daily_profit_target`) | `account_profit` | `account_profit_gate` | `account_profit_hit` | `account_profit_target` |
//!
//! - **The lock.** From the breach on, an entry or add decided in a slice of the round's
//!   IST session day is rejected with the rule's code (`entry_rules`); the next session
//!   boundary lifts it, as the boundary ends REDUCING (`aggregate`). The lock is named as
//!   a live session's entry lock for the rule is. It is not a trading state: the
//!   strategy's own rules keep theirs.
//! - **The record.** The rule's risk event (scope account; its level minus the loss limit,
//!   or the profit target; the account's measure as observed), then every working order
//!   that would open or add is cancelled (as a live kill switch's first step), then every
//!   open position with no exit waiting is exited by step 6's rule, as the strategy's own
//!   daily loss limit exits it: at its instrument's next base-bar open, or at once at its
//!   last close, flagged, when the slice closes the session; one whose instrument has no
//!   later bar in the session fills at the session close (`aggregate`). The trades record
//!   the rule's exit reason. A position whose exit already waits keeps it, with its own
//!   reason.
//! - **Where it is recorded** (2.5, the breach's stamps), by where the engine stands when
//!   the fund finds the breach at the round closing at `at`:
//!
//! | The engine | Recorded at |
//! |---|---|
//! | Its slice closes at `at`: a participant of the round, its levels phase run | That slice, step 6, at once |
//! | Its slice is open and closes after `at` (a coarser base) | That slice's step 6, when its levels phase runs; its working entries and adds are cancelled at that slice's step 3, before its levels, so none fills inside the bar (`steps::levels_phase`) |
//! | Between slices | Its next slice, step 1, so its exits fill at that slice's open (`steps::open_phase`) |
//! | Its run has not started, and the breach's session day is before its first slice's | Nothing: it holds no position or order, and the lock's day is over before it starts |
//! | Its run has ended | Nothing: its positions stay as its run left them |
//!
//! Every breach the engine takes is recorded, in the order the fund found them: an engine
//! that stands between slices across two session days of another exchange's rounds (an
//! NSE engine over an NSE holiday on which MCX trades) records both at its next slice. A
//! position is exited once, since one whose exit already waits keeps it.
//!
//! A backtest's only: a live session's risk watch fires the kill switch for the account's
//! limit and target (16-session-design 6.3).

use crate::exec::RejectCode;
use crate::model::{Day, TsNanos};
use crate::rms::{ExitReason, RiskEvent, RiskEventName, RiskScope};

use super::clock::SliceInfo;
use super::error::KernelError;
use super::event::{Event, EventStamp, Step};
use super::kernel::{Engine, Phase};

/// Which of the account's daily rules a breach is (module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AccountRule {
    /// The account's daily loss limit (`account_max_daily_loss`).
    Loss,
    /// The account's daily profit target (`account_daily_profit_target`).
    Profit,
}

impl AccountRule {
    /// Every rule, in the order of the module docs.
    pub const ALL: [AccountRule; 2] = [AccountRule::Loss, AccountRule::Profit];

    /// The name of the rule's entry lock: `account_loss` or `account_profit`.
    pub fn code(self) -> &'static str {
        match self {
            AccountRule::Loss => "account_loss",
            AccountRule::Profit => "account_profit",
        }
    }

    /// The setting the rule is: `account_max_daily_loss` or `account_daily_profit_target`.
    pub fn setting(self) -> &'static str {
        match self {
            AccountRule::Loss => "account_max_daily_loss",
            AccountRule::Profit => "account_daily_profit_target",
        }
    }

    /// The reject code of an entry or add while the rule locks entries.
    pub fn gate(self) -> RejectCode {
        match self {
            AccountRule::Loss => RejectCode::AccountLossGate,
            AccountRule::Profit => RejectCode::AccountProfitGate,
        }
    }

    /// The risk event of a breach of the rule.
    fn event(self) -> RiskEventName {
        match self {
            AccountRule::Loss => RiskEventName::AccountLossHit,
            AccountRule::Profit => RiskEventName::AccountProfitHit,
        }
    }

    /// The exit reason of the trades the rule's exits close.
    fn exit_reason(self) -> ExitReason {
        match self {
            AccountRule::Loss => ExitReason::AccountLossLimit,
            AccountRule::Profit => ExitReason::AccountProfitTarget,
        }
    }

    /// The level the risk event records, rupees: minus the loss limit, or the profit
    /// target.
    fn level_rupees(self, limit_rupees: f64) -> f64 {
        match self {
            AccountRule::Loss => -limit_rupees,
            AccountRule::Profit => limit_rupees,
        }
    }
}

/// The account's daily loss limit breached, or its daily profit target reached, at a round
/// of a portfolio run's fund, as the fund hands it to each engine (module docs).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AccountBreach {
    /// The close instant of the round the fund found the breach at, UTC nanoseconds.
    pub at: TsNanos,
    /// The round's IST session day: entries are locked for it.
    pub day: Day,
    /// Which rule: the daily loss limit or the daily profit target.
    pub rule: AccountRule,
    /// The rule's amount, rupees, positive: the loss limit (`account_max_daily_loss`) or
    /// the profit target (`account_daily_profit_target`).
    pub limit_rupees: f64,
    /// The account's daily measure at the breach, rupees, negative for a loss.
    pub measure_rupees: f64,
}

impl Engine {
    /// Take the account's breach (module docs): lock entries for its session day and
    /// record it where the engine stands, now or at its next step 6 or step 1. An engine
    /// whose run has ended takes nothing.
    ///
    /// Errors: `InvalidConfig` (field the rule's setting, `AccountRule::setting`) for a
    /// limit that is not a positive amount of rupees or a measure that is not a finite
    /// one; `NotBuilt` in a live session, whose risk watch holds the account's rules; those
    /// of the record.
    pub fn account_breach(&mut self, breach: AccountBreach) -> Result<(), KernelError> {
        let usable = breach.limit_rupees.is_finite()
            && breach.limit_rupees > 0.0
            && breach.measure_rupees.is_finite();
        if !usable {
            return Err(KernelError::InvalidConfig {
                field: breach.rule.setting(),
                reason: format!(
                    "a breach needs a positive limit and a finite measure, got a limit of {} \
                     and a measure of {} rupees",
                    breach.limit_rupees, breach.measure_rupees
                ),
            });
        }
        if self.is_external() {
            return Err(KernelError::NotBuilt {
                feature: "the account's daily loss breach or profit target in a live \
                          session's kernel",
                instead: "a live session's risk watch fires the kill switch for the \
                          account's daily loss limit and daily profit target",
            });
        }
        match self.phase {
            Phase::Finished => Ok(()),
            Phase::Ready(0) if self.starts_after(breach.day) => Ok(()),
            Phase::Opened(slice) if slice.close <= breach.at => {
                self.account.locked = Some((breach.day, breach.rule));
                self.record_account_breach(&slice, Step::AggregateRules, breach)
            }
            _ => {
                self.account.locked = Some((breach.day, breach.rule));
                self.account.waiting.push(breach);
                Ok(())
            }
        }
    }

    /// True when the run's first slice is on a session day after `day`, or the run has no
    /// slice.
    fn starts_after(&self, day: Day) -> bool {
        self.timeline.slice(0).map_or(true, |first| first.day > day)
    }

    /// Step 1 of `slice`: the breaches found while the engine stood between slices (module
    /// docs); their exits fill at this slice's open.
    ///
    /// Errors: those of the record.
    pub(super) fn account_breach_at_open(&mut self, slice: &SliceInfo) -> Result<(), KernelError> {
        self.record_waiting_breaches(slice, Step::SessionOpen)
    }

    /// Step 3 of `slice`, before its levels: a breach found while the slice was open
    /// cancels the working entries and adds, so none fills inside the bar from the breach
    /// on (module docs). The breach itself is recorded at step 6.
    ///
    /// Errors: those of the cancels.
    pub(super) fn account_breach_inside_bar(
        &mut self,
        slice: &SliceInfo,
    ) -> Result<(), KernelError> {
        let Some(first) = self.account.waiting.first() else {
            return Ok(());
        };
        let rule = first.rule;
        self.cancel_entries_for_account(Self::stamp(slice, Step::Levels), rule)
    }

    /// Cancel every working order that would open or add, in id order, as the strategy's
    /// own daily loss limit cancels them (`aggregate`), and keep each, with the `rule` that
    /// ended it, for the fund's record of the orders the account ended (`fund_view`).
    ///
    /// Errors: those of the cancels.
    fn cancel_entries_for_account(
        &mut self,
        stamp: EventStamp,
        rule: AccountRule,
    ) -> Result<(), KernelError> {
        for id in self.working_entries() {
            self.withdraw_order(stamp, id)?;
            self.account.cancels.push((stamp, id, rule));
        }
        Ok(())
    }

    /// Step 6 of `slice`, after the strategy's own aggregate rules: the breaches found
    /// while the slice was open (module docs).
    ///
    /// Errors: those of the record.
    pub(super) fn account_breach_at_close(&mut self, slice: &SliceInfo) -> Result<(), KernelError> {
        self.record_waiting_breaches(slice, Step::AggregateRules)
    }

    /// Record every waiting breach at `step` of `slice`, in the order the fund found them.
    ///
    /// Errors: those of the record.
    fn record_waiting_breaches(
        &mut self,
        slice: &SliceInfo,
        step: Step,
    ) -> Result<(), KernelError> {
        for breach in std::mem::take(&mut self.account.waiting) {
            self.record_account_breach(slice, step, breach)?;
        }
        Ok(())
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
        let rule = breach.rule;
        let event = RiskEvent {
            name: rule.event(),
            scope: RiskScope::Account,
            level_rupees: Some(rule.level_rupees(breach.limit_rupees)),
            target_index: None,
            observed_rupees: Some(breach.measure_rupees),
        };
        self.emit(Event::Risk {
            stamp,
            instrument: None,
            event,
        });
        self.cancel_entries_for_account(stamp, rule)?;
        let reason = rule.exit_reason();
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
