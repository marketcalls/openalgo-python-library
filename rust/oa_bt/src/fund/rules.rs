//! `AccountRules`: the account's daily loss limit of a portfolio run (18-multi-strategy
//! 2.5), evaluated at phase A of every round.
//!
//! - **The account's measure** (H3): the sum, in call order, of the engines' daily loss
//!   measures (each on the run's risk basis, R10), where an engine's measure counts only
//!   when the session day of its figures is the round's IST session day, and counts 0
//!   otherwise. So an NSE engine's loss of yesterday never adds to an MCX evening round of
//!   today, and an engine on a daily base that has not opened today's slice adds nothing.
//! - **The breach.** The measure at or past the limit (`rms::is_breached`, in whole
//!   paise), once per session day: the fund hands every engine the `AccountBreach`.
//!
//! The same rule, as a table of cases, is read by this module's tests and by the live
//! session's risk watch's (`test/backtest/data/account_rule_cases.json`, design rule 4).

use crate::engine::AccountBreach;
use crate::model::{Day, TsNanos};
use crate::portfolio::round_to_paise;
use crate::rms::is_breached;

/// The account's measure at a round of session day `day` (module docs), rupees in paise,
/// negative for a loss, from each engine's (measure, session day) in call order.
pub(super) fn account_measure(day: Day, engines: impl Iterator<Item = (f64, Option<Day>)>) -> f64 {
    let counted = engines.map(|(measure, of)| if of == Some(day) { measure } else { 0.0 });
    round_to_paise(counted.sum())
}

/// The account's daily loss limit and the session day it was last breached on.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct AccountRules {
    /// Rupees, above 0; `None` for no limit.
    limit_rupees: Option<f64>,
    /// The session day of the last breach: the limit fires once per session day.
    breached_on: Option<Day>,
}

impl AccountRules {
    /// The rules of an account with the daily loss limit `limit_rupees` (checked by the
    /// fund: above 0), or none.
    pub(super) fn new(limit_rupees: Option<f64>) -> AccountRules {
        AccountRules {
            limit_rupees,
            breached_on: None,
        }
    }

    /// The breach of the round closing at `at` on session day `day` with the account's
    /// `measure_rupees`, when the limit is reached and was not on that day already.
    pub(super) fn breach(
        &mut self,
        at: TsNanos,
        day: Day,
        measure_rupees: f64,
    ) -> Option<AccountBreach> {
        let limit_rupees = self.limit_rupees?;
        if self.breached_on == Some(day) || !is_breached(measure_rupees, limit_rupees) {
            return None;
        }
        self.breached_on = Some(day);
        Some(AccountBreach {
            at,
            day,
            limit_rupees,
            measure_rupees,
        })
    }
}

#[cfg(test)]
mod table_tests;
