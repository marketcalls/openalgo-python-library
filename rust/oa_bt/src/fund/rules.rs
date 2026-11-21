//! `AccountRules`: the account's daily loss limit and daily profit target of a portfolio
//! run (18-multi-strategy 2.5 and 2.5a rule 6), evaluated at phase A of every round.
//!
//! - **The account's measure** (H3): the sum, in call order, of the engines' daily loss
//!   measures (each on the run's risk basis, R10), where an engine's measure counts only
//!   when the session day of its figures is the round's IST session day, and counts 0
//!   otherwise. So an NSE engine's loss of yesterday never adds to an MCX evening round of
//!   today, and an engine on a daily base that has not opened today's slice adds nothing.
//! - **The loss limit.** The measure at or past minus the limit (`rms::is_breached`, in
//!   whole paise), once per session day: the fund hands every engine the `AccountBreach`
//!   of the rule `Loss`.
//! - **The profit target.** The measure at or past the target (`rms::is_target_reached`,
//!   in whole paise), once per session day: the `AccountBreach` of the rule `Profit`. A
//!   measure cannot be at both, so a round breaches one rule at most.
//!
//! The same rules, as a table of cases, are read by this module's tests and by the live
//! session's risk watch's (`test/backtest/data/account_rule_cases.json`, design rule 4).

use crate::engine::{AccountBreach, AccountRule};
use crate::model::{Day, TsNanos};
use crate::portfolio::round_to_paise;
use crate::rms::{is_breached, is_target_reached};

/// The account's measure at a round of session day `day` (module docs), rupees in paise,
/// negative for a loss, from each engine's (measure, session day) in call order.
pub(super) fn account_measure(day: Day, engines: impl Iterator<Item = (f64, Option<Day>)>) -> f64 {
    let counted = engines.map(|(measure, of)| if of == Some(day) { measure } else { 0.0 });
    round_to_paise(counted.sum())
}

/// The account's daily loss limit and daily profit target, and the session day each was
/// last breached on.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct AccountRules {
    /// Rupees, above 0; `None` for no limit.
    limit_rupees: Option<f64>,
    /// The session day of the last breach of the limit: it fires once per session day.
    breached_on: Option<Day>,
    /// Rupees, above 0; `None` for no profit target.
    target_rupees: Option<f64>,
    /// The session day the target was last reached on: it fires once per session day.
    reached_on: Option<Day>,
}

impl AccountRules {
    /// The rules of an account with the daily loss limit `limit_rupees` and the daily
    /// profit target `target_rupees` (each checked by the fund: above 0), or none.
    pub(super) fn new(limit_rupees: Option<f64>, target_rupees: Option<f64>) -> AccountRules {
        AccountRules {
            limit_rupees,
            breached_on: None,
            target_rupees,
            reached_on: None,
        }
    }

    /// The breach of the round closing at `at` on session day `day` with the account's
    /// `measure_rupees`: the loss limit reached, or the profit target, when it was not on
    /// that day already.
    pub(super) fn breach(
        &mut self,
        at: TsNanos,
        day: Day,
        measure_rupees: f64,
    ) -> Option<AccountBreach> {
        let breach = |rule, limit_rupees| AccountBreach {
            at,
            day,
            rule,
            limit_rupees,
            measure_rupees,
        };
        if let Some(limit) = self.limit_rupees {
            if self.breached_on != Some(day) && is_breached(measure_rupees, limit) {
                self.breached_on = Some(day);
                return Some(breach(AccountRule::Loss, limit));
            }
        }
        let target = self.target_rupees?;
        if self.reached_on == Some(day) || !is_target_reached(measure_rupees, target) {
            return None;
        }
        self.reached_on = Some(day);
        Some(breach(AccountRule::Profit, target))
    }
}

#[cfg(test)]
mod table_tests;
